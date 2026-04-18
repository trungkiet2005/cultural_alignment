"""DIFFPO-adapted black-box baseline for binary country-conditioned moral decisions.

Motivation (Round 2 reviewer W1c):
----------------------------------
DIFFPO (Chen et al., 2503.04240) is a black-box inference-time alignment method
that trains a small diffusion-style sentence refiner to nudge a frozen LLM's
outputs toward a preference target. DIFFPO was designed for open-ended
generation, so it does not transfer verbatim to MultiTP's binary A/B format.
We implement the *spirit* of DIFFPO adapted to the binary decision setting:

    p_aligned = (1 - alpha) * p_model + alpha * p_target

where ``p_target`` is a **black-box, country-conditioned reference** distribution
obtained from the country's *public* human AMCE (loaded from the same file that
the MIS metric uses for evaluation -- so this baseline has strictly more
information than SWA-DPBR, which uses *only* the WVS persona panel and never
looks at the human AMCE at inference time). The mixing weight ``alpha`` is the
analogue of DIFFPO's guidance-strength hyperparameter, fit on a small
calibration split per country via grid search (same protocol as
:mod:`src.calibration_baselines`).

Why this is a fair (actually favourable) baseline:
  * It operates entirely on *probabilities* of the frozen model -- no weight
    updates, matching SWA-DPBR's black-box-with-logits regime.
  * It is explicitly country-conditioned: the ``p_target`` carries the human
    AMCE for the target country, so it cannot be dismissed as "generic
    calibration" (that's what :mod:`src.calibration_baselines` covers).
  * It *leaks* the evaluation target. SWA-DPBR does not. If DIFFPO-binary
    matches SWA-DPBR, scale it. If it doesn't, SWA-DPBR's within-country
    disagreement signal is doing something a direct oracle cannot.

Entry point: :func:`run_baseline_diffpo_binary` -- same signature as
:func:`src.baseline_runner.run_baseline_vanilla`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.amce import (
    compute_alignment_metrics,
    compute_amce_from_preferences,
    load_human_amce,
)
from src.calibration_baselines import _extract_raw_gaps, _p_from_gap


# Map from phenomenon_category -> canonical AMCE key (matches LABEL_TO_CRITERION
# target values in src/constants.py) and which side is the "preferred" group.
_CAT_TO_KEY: Dict[str, str] = {
    "Species":        "Species_Humans",
    "Gender":         "Gender_Female",
    "Age":            "Age_Young",
    "Fitness":        "Fitness_Fit",
    "SocialValue":    "SocialValue_High",
    "Utilitarianism": "Utilitarianism_More",
}


@dataclass
class DiffpoParams:
    alpha: float
    cal_loss: float
    n_cal: int


def _vanilla_p(df: pd.DataFrame) -> np.ndarray:
    """p(spare_preferred) under the vanilla model -- sigmoid of the raw gap,
    then oriented toward the preferred side."""
    delta = df["raw_logit_gap"].values.astype(np.float64)
    pref_r = df["preferred_on_right"].values.astype(np.int64)
    return _p_from_gap(delta, temp=1.0, margin=0.0, preferred_right=pref_r)


def _country_target_p(df: pd.DataFrame, human_amce: Dict[str, float]) -> np.ndarray:
    """Construct the country-conditioned ``p_target`` per scenario.

    We use the country's human AMCE as a per-category constant target:
    for every scenario in category ``c``, ``p_target[i] = human_amce[c] / 100``.
    This is the "sentence-level alignment module" equivalent of DIFFPO adapted
    to a single binary decision: the target is the population-level sparing
    rate for that category.
    """
    cats = df["phenomenon_category"].values
    out = np.full(len(df), 0.5, dtype=np.float64)
    for c, key in _CAT_TO_KEY.items():
        if key in human_amce:
            out[cats == c] = float(human_amce[key]) / 100.0
    return out


def _mis_from_mixed(
    df: pd.DataFrame,
    p_vanilla: np.ndarray,
    p_target: np.ndarray,
    alpha: float,
    human_amce: Dict[str, float],
) -> float:
    """Compute held-out MIS for a given ``alpha`` mixing weight."""
    p = (1.0 - alpha) * p_vanilla + alpha * p_target
    model_amce: Dict[str, float] = {}
    cats = df["phenomenon_category"].values
    for c, key in _CAT_TO_KEY.items():
        sel = cats == c
        if sel.sum() < 3:
            continue
        model_amce[key] = float(p[sel].mean()) * 100.0
    common = sorted(set(model_amce.keys()) & set(human_amce.keys()))
    if len(common) < 2:
        return float("nan")
    m = np.array([model_amce[k] for k in common]) / 100.0
    h = np.array([human_amce[k] for k in common]) / 100.0
    return float(np.linalg.norm(m - h))


def _fit_alpha(cal_df: pd.DataFrame, human_amce: Dict[str, float]) -> DiffpoParams:
    """Grid-search the mixing weight alpha in [0, 1] on the calibration split."""
    p_van = _vanilla_p(cal_df)
    p_tgt = _country_target_p(cal_df, human_amce)

    grid = np.linspace(0.0, 1.0, 41)  # 0.025 steps
    best_loss = float("inf")
    best_a = 0.0
    for a in grid:
        loss = _mis_from_mixed(cal_df, p_van, p_tgt, float(a), human_amce)
        if np.isfinite(loss) and loss < best_loss:
            best_loss = loss
            best_a = float(a)
    return DiffpoParams(alpha=best_a, cal_loss=best_loss, n_cal=len(cal_df))


def run_baseline_diffpo_binary(
    model, tokenizer, scenario_df: pd.DataFrame, country: str, cfg,
    *,
    cal_frac: float = 0.25,
    seed: int = 42,
) -> dict:
    """Fit per-country alpha, then evaluate on the held-out split.

    Returns the same-shaped dict as :func:`run_baseline_vanilla`.
    """
    all_df = _extract_raw_gaps(model, tokenizer, scenario_df, country)
    human_amce = load_human_amce(cfg.human_amce_path, country)

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(all_df))
    n_cal = max(20, int(round(cal_frac * len(all_df))))
    cal_idx = set(idx[:n_cal].tolist())
    all_df = all_df.copy()
    all_df["split"] = ["cal" if i in cal_idx else "test" for i in range(len(all_df))]

    cal_df = all_df[all_df["split"] == "cal"].copy()

    if not human_amce:
        print(f"[DIFFPO {country}] WARN: human AMCE empty; alpha=0 (== vanilla)")
        fitted = DiffpoParams(alpha=0.0, cal_loss=float("nan"), n_cal=len(cal_df))
    else:
        fitted = _fit_alpha(cal_df, human_amce)
        print(f"[DIFFPO {country}] alpha={fitted.alpha:.3f}  "
              f"cal_MIS={fitted.cal_loss:.4f}  n_cal={fitted.n_cal}")

    p_van = _vanilla_p(all_df)
    p_tgt = _country_target_p(all_df, human_amce)
    p_mix = (1.0 - fitted.alpha) * p_van + fitted.alpha * p_tgt

    scored = all_df.copy()
    pref_r = scored["preferred_on_right"].values.astype(np.int64)
    scored["p_spare_preferred"] = p_mix
    scored["p_right"] = np.where(pref_r == 1, p_mix, 1.0 - p_mix)
    scored["p_left"] = 1.0 - scored["p_right"]
    scored["diffpo_alpha"] = fitted.alpha

    scored_test = scored[scored["split"] == "test"].copy()
    scored_test["country"] = country
    model_amce = compute_amce_from_preferences(scored_test)
    alignment = compute_alignment_metrics(model_amce, human_amce)
    return {
        "model_amce": model_amce,
        "human_amce": human_amce,
        "alignment":  alignment,
        "results_df": scored,
        "diffpo_params": {
            "alpha":   fitted.alpha,
            "cal_loss": fitted.cal_loss,
            "n_cal":   fitted.n_cal,
            "n_test":  len(scored_test),
        },
    }
