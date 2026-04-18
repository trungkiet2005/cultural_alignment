"""Calibration-only baselines: per-country temperature and margin scaling.

Motivation (Round 2 reviewer W1b):
----------------------------------
The reviewer asks for a head-to-head between SWA-DPBR and pure calibration
scaling on the same 20-country grid, to isolate what the PT-IS + reliability
gate add *on top of* a simple directional shift.

Two scalar calibration baselines are exposed:
    1. TempScaleBaseline   -- per-country temperature T_c on the A/B logit gap.
    2. MarginShiftBaseline -- per-country additive margin m_c on the logit gap.

Both are calibrated directly from a small held-out split per country using the
target human AMCE on that split. They have access to the *same* country-level
supervision signal that the reviewer considers "light" (a handful of per-country
scalars), so the comparison is fair and favorable to the baseline.

These baselines do NOT use personas, WVS profiles, or the IS update -- they just
shift / scale the vanilla A/B logits. If SWA-DPBR merely replicates their effect,
the gap vs. vanilla should collapse. If it doesn't, the within-country
disagreement signal (the core claim) is doing something these simpler knobs
cannot.

Entry point:
    :func:`run_baseline_calibration_scaling` fits parameters on a calibration
    slice (first ``cal_frac`` of the scenarios by default) and applies them to
    the remainder. Returns the same-shaped dict as
    :func:`src.baseline_runner.run_baseline_vanilla`.

Fitting objective: we pick the scalar (T_c or m_c) that minimises the L2
distance between per-category mean p_spare and the country's human AMCE on the
calibration split. Because each scaling knob is a 1-D scalar per country, we
use a 40-point grid search --- exact, derivative-free, and cheap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.amce import (
    compute_alignment_metrics,
    compute_amce_from_preferences,
    load_human_amce,
)
from src.baseline_runner import resolve_decision_tokens_for_lang
from src.constants import COUNTRY_LANG, LABEL_TO_CRITERION
from src.i18n import BASE_ASSISTANT_I18N, PROMPT_FRAME_I18N
from src.model import ChatTemplateHelper, gather_last_logits


# ----------------------------------------------------------------------------- #
# Step 1: extract raw A/B logit gaps for every scenario (single forward pass).   #
# ----------------------------------------------------------------------------- #


def _extract_raw_gaps(model, tokenizer, scenario_df: pd.DataFrame,
                      country: str, batch_size: int = 8) -> pd.DataFrame:
    """Run a single vanilla forward pass per scenario and return the *raw*
    A/B logit gap ``delta = logit_B - logit_A`` so downstream scaling knobs
    can be applied without re-running the model.
    """
    device = next(model.parameters()).device
    lang = COUNTRY_LANG.get(country, "en")
    chat_helper = ChatTemplateHelper(tokenizer)
    base_text = BASE_ASSISTANT_I18N.get(lang, BASE_ASSISTANT_I18N["en"])
    base_ids = chat_helper.build_prefix_ids(base_text, device)

    a_id, b_id = resolve_decision_tokens_for_lang(tokenizer, chat_helper, lang)
    if hasattr(model, "set_decision_tokens"):
        model.set_decision_tokens(int(a_id), int(b_id))
    setattr(tokenizer, "_moral_vllm_ab", (int(a_id), int(b_id)))

    frame = PROMPT_FRAME_I18N.get(lang, PROMPT_FRAME_I18N["en"])
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    pending: List[Tuple[dict, torch.Tensor]] = []
    for _, row in scenario_df.iterrows():
        prompt = row.get("Prompt", row.get("prompt", ""))
        if not prompt:
            continue
        user_content = frame.format(scenario=prompt)
        query_ids = chat_helper.encode_query_suffix(user_content, device)
        full_ids = torch.cat([base_ids, query_ids], dim=1)[0]
        pending.append((row.to_dict(), full_ids))

    def _pad(seqs):
        lens = torch.tensor([s.size(0) for s in seqs], dtype=torch.long, device=device)
        m = int(lens.max().item())
        out_ids = torch.full((len(seqs), m), pad_id, dtype=seqs[0].dtype, device=device)
        for i, s in enumerate(seqs):
            out_ids[i, : s.size(0)] = s
        return out_ids, lens

    rows_out = []
    for start in tqdm(range(0, len(pending), batch_size),
                      desc=f"RawGaps [{country}]"):
        chunk = pending[start:start + batch_size]
        ids, lens = _pad([c[1] for c in chunk])
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=False)
            batch_idx = torch.arange(ids.size(0), device=device)
            last = gather_last_logits(out, batch_idx, lens)
            delta = (last[:, b_id] - last[:, a_id]).float().cpu().numpy()
        for (row, _), d in zip(chunk, delta):
            rec = dict(row)
            rec["raw_logit_gap"] = float(d)
            rows_out.append(rec)
    return pd.DataFrame(rows_out)


# ----------------------------------------------------------------------------- #
# Step 2: fit per-country calibration scalar (T or m) on a calibration split.   #
# ----------------------------------------------------------------------------- #


@dataclass
class CalibParams:
    method: Literal["temperature", "margin"]
    value: float
    cal_loss: float
    n_cal: int


def _p_from_gap(delta: np.ndarray, temp: float, margin: float,
                preferred_right: np.ndarray) -> np.ndarray:
    """σ((δ + margin) / T), then orient to preferred side."""
    t = max(float(temp), 1e-10)
    shifted = (delta + margin) / t
    p_right = 1.0 / (1.0 + np.exp(-shifted))
    return np.where(preferred_right == 1, p_right, 1.0 - p_right)


def _amce_from_p(df: pd.DataFrame, p_spare: np.ndarray) -> Dict[str, float]:
    """Per-category MPR from the given p_spare vector (same semantics as
    compute_amce_from_preferences; minus the df copy / column indirection)."""
    out: Dict[str, float] = {}
    cats = ["Species", "Gender", "Age", "Fitness", "SocialValue", "Utilitarianism"]
    pref_suffix = {
        "Species": "Humans", "Gender": "Female", "Age": "Young",
        "Fitness": "Fit", "SocialValue": "High", "Utilitarianism": "More",
    }
    mask = df["phenomenon_category"].values
    for c in cats:
        sel = mask == c
        if sel.sum() < 3:
            continue
        out[f"{c}_{pref_suffix[c]}"] = float(p_spare[sel].mean()) * 100.0
    return out


def _mis_from_amce(model_amce: Dict[str, float], human_amce: Dict[str, float]) -> float:
    common = sorted(set(model_amce.keys()) & set(human_amce.keys()))
    if len(common) < 2:
        return float("nan")
    m = np.array([model_amce[k] for k in common], dtype=np.float64) / 100.0
    h = np.array([human_amce[k] for k in common], dtype=np.float64) / 100.0
    return float(np.linalg.norm(m - h))


def _fit_calib_param(
    cal_df: pd.DataFrame,
    human_amce: Dict[str, float],
    method: Literal["temperature", "margin"],
    grid: np.ndarray,
) -> CalibParams:
    """Grid-search the scalar (T or m) that minimises MIS on the calibration
    split versus the country's human AMCE."""
    delta = cal_df["raw_logit_gap"].values.astype(np.float64)
    pref_r = cal_df["preferred_on_right"].values.astype(np.int64)

    best_val = float("inf")
    best_param = grid[0]
    for v in grid:
        if method == "temperature":
            p = _p_from_gap(delta, temp=float(v), margin=0.0,
                            preferred_right=pref_r)
        else:  # margin
            p = _p_from_gap(delta, temp=1.0, margin=float(v),
                            preferred_right=pref_r)
        amce = _amce_from_p(cal_df, p)
        mis = _mis_from_amce(amce, human_amce)
        if np.isfinite(mis) and mis < best_val:
            best_val = mis
            best_param = float(v)
    return CalibParams(method=method, value=best_param,
                       cal_loss=best_val, n_cal=len(cal_df))


# ----------------------------------------------------------------------------- #
# Step 3: full runner (apply fitted scalars to the held-out split).             #
# ----------------------------------------------------------------------------- #


def run_baseline_calibration_scaling(
    model, tokenizer, scenario_df: pd.DataFrame, country: str, cfg,
    *,
    method: Literal["temperature", "margin"] = "temperature",
    cal_frac: float = 0.25,
    seed: int = 42,
) -> dict:
    """Fit per-country T or m on a calibration split, then score the rest.

    Returns the usual {model_amce, human_amce, alignment, results_df,
    calib_params} dict. ``results_df`` covers the full scenario set:
    calibration rows are included with a ``split='cal'`` marker so downstream
    consumers can filter, and held-out rows are scored with the fitted scalar.
    """
    if method not in ("temperature", "margin"):
        raise ValueError(f"method must be 'temperature' or 'margin'; got {method!r}")

    # --- Step 1: raw logit gaps (single forward pass) ---
    all_df = _extract_raw_gaps(model, tokenizer, scenario_df, country)

    # --- Step 2: cal/test split ---
    rng = np.random.RandomState(seed)
    n = len(all_df)
    idx = rng.permutation(n)
    n_cal = max(20, int(round(cal_frac * n)))
    cal_idx = set(idx[:n_cal].tolist())
    all_df = all_df.copy()
    all_df["split"] = ["cal" if i in cal_idx else "test" for i in range(n)]

    cal_df = all_df[all_df["split"] == "cal"].copy()
    test_df = all_df[all_df["split"] == "test"].copy()

    human_amce = load_human_amce(cfg.human_amce_path, country)
    if not human_amce:
        print(f"[CALIB {country}] WARN: human AMCE empty; falling back to T=1.0 / m=0.0")
        fitted = CalibParams(method=method, value=1.0 if method == "temperature" else 0.0,
                             cal_loss=float("nan"), n_cal=len(cal_df))
    else:
        if method == "temperature":
            grid = np.concatenate([np.linspace(0.25, 1.00, 16),
                                   np.linspace(1.05, 4.00, 24)])
        else:
            grid = np.linspace(-4.0, 4.0, 41)
        fitted = _fit_calib_param(cal_df, human_amce, method, grid)
        print(f"[CALIB {country}] {method}={fitted.value:.4f}  "
              f"cal_MIS={fitted.cal_loss:.4f}  n_cal={fitted.n_cal}")

    # --- Step 3: apply fitted scalar to test split ---
    def _apply(df: pd.DataFrame) -> pd.DataFrame:
        delta = df["raw_logit_gap"].values.astype(np.float64)
        pref_r = df["preferred_on_right"].values.astype(np.int64)
        if method == "temperature":
            p = _p_from_gap(delta, temp=fitted.value, margin=0.0, preferred_right=pref_r)
            p_raw_right = 1.0 / (1.0 + np.exp(-delta / max(fitted.value, 1e-10)))
        else:
            p = _p_from_gap(delta, temp=1.0, margin=fitted.value, preferred_right=pref_r)
            p_raw_right = 1.0 / (1.0 + np.exp(-(delta + fitted.value)))
        p_left = np.where(pref_r == 1, 1.0 - p, p)
        p_right = np.where(pref_r == 1, p, 1.0 - p)
        df = df.copy()
        df["p_spare_preferred"] = p
        df["p_left"] = p_left
        df["p_right"] = p_right
        df["calib_method"] = method
        df["calib_value"] = fitted.value
        return df

    scored_df = pd.concat([_apply(cal_df), _apply(test_df)], ignore_index=True)

    # The AMCE is computed on the TEST split only, so it is honest held-out
    # evaluation (not fitted on the same data).
    scored_test = scored_df[scored_df["split"] == "test"].copy()
    scored_test["country"] = country
    model_amce = compute_amce_from_preferences(scored_test)
    alignment = compute_alignment_metrics(model_amce, human_amce)

    return {
        "model_amce": model_amce,
        "human_amce": human_amce,
        "alignment":  alignment,
        "results_df": scored_df,
        "calib_params": {
            "method":  fitted.method,
            "value":   fitted.value,
            "cal_loss": fitted.cal_loss,
            "n_cal":    fitted.n_cal,
            "n_test":   len(scored_test),
        },
    }
