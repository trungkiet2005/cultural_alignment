"""Stage 2 of the open-ended SWA-DPBR variant — judge + DPBR + AMCE.

Reads Stage-1 JSONL files, runs a 70B+ judge LLM on each actor generation,
extracts ``{choice, confidence}``, converts to a scalar pseudo-delta via
:func:`src.pseudo_delta.pseudo_delta_from_judge`, debiases the pseudo-delta
with the standard ``(pass1 - pass2) / 2`` transform, and feeds the result to
:class:`exp_paper.openended.dpbr_offline.Exp24DualPassControllerOffline` so the
PT-IS, dual-pass bootstrap reliability, ESS anchor blend, and hierarchical
prior all run unchanged. Per-country AMCE + alignment metrics are written
using :func:`src.amce.compute_amce_from_preferences` and
:func:`src.amce.compute_alignment_metrics`.

A "vanilla-from-base" analog is also emitted using only ``agent_role=="base"``
rows (no PT-IS, just the sigmoid of the debiased base delta), to mirror the
structure of ``exp_model/_base_dpbr.py`` which writes both a vanilla baseline
and a DPBR sweep.

Resume: judge calls are cached to ``{country}_judged.jsonl`` keyed by
sha1(scenario_en + actor_text). A second invocation will re-use any cached
verdicts and only run the judge on new rows.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from experiment_DM.exp24_dpbr_core import BootstrapPriorState, PRIOR_STATE
from src.amce import (
    compute_alignment_metrics,
    compute_amce_from_preferences,
    compute_per_dimension_alignment,
    load_human_amce,
)
from src.judge_prompts import (
    JUDGE_SYSTEM_PROMPT,
    build_judge_prompt,
    parse_judge_output,
)
from src.model import load_model_hf_native, setup_seeds
from src.pseudo_delta import (
    T_DECISION,
    pseudo_delta_from_judge,
    pseudo_p_right_from_delta,
)

from exp_paper.openended.dpbr_offline import Exp24DualPassControllerOffline
from exp_paper.openended.stage1_actor_phi4 import AGENT_ROLES, DEBIAS_VARIANTS


@dataclass
class Stage2Config:
    judge_model_name: str = "Qwen/Qwen2.5-72B-Instruct"
    stage1_jsonl_dir: str = "/kaggle/working/cultural_alignment/results/openended/stage1"
    results_base: str = "/kaggle/working/cultural_alignment/results/openended"
    human_amce_path: str = ""
    load_in_4bit: bool = True
    max_new_tokens: int = 64
    seed: int = 42
    max_parse_fail_pct: float = 5.0
    countries: List[str] = field(default_factory=list)
    lambda_coop: float = 0.70


# ----------------------------------------------------------------------------
# Judge I/O
# ----------------------------------------------------------------------------
def _record_hash(scenario_en: str, actor_text: str) -> str:
    h = hashlib.sha1()
    h.update(scenario_en.encode("utf-8"))
    h.update(b"\x00")
    h.update(actor_text.encode("utf-8"))
    return h.hexdigest()


def _load_stage1_rows(jsonl_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _load_judge_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = obj.get("hash")
            if k:
                cache[k] = obj
    return cache


@torch.no_grad()
def _judge_generate(
    judge_model, judge_tokenizer,
    scenario_en: str, actor_text: str,
    max_new_tokens: int, device: torch.device,
) -> str:
    """Generate one judge completion using the tokenizer's chat template."""
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": build_judge_prompt(scenario_en, actor_text)},
    ]
    if hasattr(judge_tokenizer, "apply_chat_template"):
        templated = judge_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        # Newer Transformers may return a BatchEncoding dict instead of a Tensor.
        if isinstance(templated, torch.Tensor):
            input_ids = templated.to(device)
        else:
            input_ids = templated["input_ids"].to(device)
    else:
        prompt = (
            f"{JUDGE_SYSTEM_PROMPT}\n\n"
            f"{build_judge_prompt(scenario_en, actor_text)}\n"
        )
        input_ids = judge_tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    out = judge_model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=(
            judge_tokenizer.pad_token_id
            if judge_tokenizer.pad_token_id is not None
            else judge_tokenizer.eos_token_id
        ),
    )
    new_ids = out[0, input_ids.shape[1]:]
    return judge_tokenizer.decode(new_ids, skip_special_tokens=True)


# ----------------------------------------------------------------------------
# Per-country pipeline
# ----------------------------------------------------------------------------
def _judge_rows_for_country(
    rows: List[Dict[str, Any]],
    judge_model, judge_tokenizer,
    cache_path: Path,
    max_new_tokens: int,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Attach judge {choice, confidence, pseudo_delta} to every row.

    Caches by sha1(scenario_en + actor_text). Skipped rows (swap_unchanged)
    get a synthetic UNCERTAIN verdict so downstream aggregation is simple.
    """
    cache = _load_judge_cache(cache_path)
    stats = {"judged": 0, "cached": 0, "parse_fail": 0, "skipped": 0}

    with cache_path.open("a", encoding="utf-8") as cache_fh:
        for r in rows:
            if r.get("skipped_reason") == "swap_unchanged":
                r["judge_choice"] = "UNCERTAIN"
                r["judge_confidence"] = 0.0
                r["judge_raw"] = ""
                r["judge_parse_ok"] = False
                r["pseudo_delta"] = 0.0
                stats["skipped"] += 1
                continue

            scenario_en = str(r.get("scenario_en", ""))
            actor_text = str(r.get("actor_text", ""))
            key = _record_hash(scenario_en, actor_text)

            if key in cache:
                hit = cache[key]
                raw = hit.get("raw", "")
                parsed = {
                    "choice": hit.get("choice", "UNCERTAIN"),
                    "confidence": float(hit.get("confidence", 0.0)),
                    "parse_ok": bool(hit.get("parse_ok", False)),
                }
                stats["cached"] += 1
            else:
                raw = _judge_generate(
                    judge_model, judge_tokenizer,
                    scenario_en, actor_text, max_new_tokens, device,
                )
                parsed = parse_judge_output(raw)
                cache_fh.write(json.dumps({
                    "hash": key,
                    "raw": raw,
                    "choice": parsed["choice"],
                    "confidence": parsed["confidence"],
                    "parse_ok": parsed["parse_ok"],
                }, ensure_ascii=False) + "\n")
                cache_fh.flush()
                stats["judged"] += 1

            if not parsed["parse_ok"]:
                stats["parse_fail"] += 1

            r["judge_choice"] = parsed["choice"]
            r["judge_confidence"] = float(parsed["confidence"])
            r["judge_raw"] = raw
            r["judge_parse_ok"] = bool(parsed["parse_ok"])
            r["pseudo_delta"] = pseudo_delta_from_judge(
                parsed["choice"], float(parsed["confidence"])
            )
    return rows, stats


def _assemble_per_scenario(
    rows: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Group rows by scenario_id and emit one input dict per scenario.

    Expected: 10 rows/scenario (5 agents x 2 debiasing variants). When
    swap_unchanged, pass2 rows are synthetic UNCERTAIN (delta=0) -> Stage 2
    falls back to d2=d1 so delta_*_deb = d1 and positional_bias = 0.
    """
    by_sid: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        sid = int(r.get("scenario_id", -1))
        by_sid.setdefault(sid, []).append(r)

    out: List[Dict[str, Any]] = []
    for sid in sorted(by_sid.keys()):
        group = by_sid[sid]
        by_key = {(r["agent_role"], r["debias_variant"]): r for r in group}
        missing = [
            (role, variant) for role in AGENT_ROLES for variant in DEBIAS_VARIANTS
            if (role, variant) not in by_key
        ]
        if missing:
            print(f"[WARN] scenario_id={sid} missing rows {missing} — skipping")
            continue

        sample = group[0]
        swap_changed = all(bool(r.get("swap_changed", False)) for r in group)

        def _delta(role: str, variant: str) -> float:
            return float(by_key[(role, variant)].get("pseudo_delta", 0.0))

        d1_base = _delta("base", "pass1")
        d2_base = _delta("base", "pass2") if swap_changed else d1_base
        d1_ag = np.array([_delta(f"persona_{i}", "pass1") for i in range(4)], dtype=np.float64)
        d2_ag = (
            np.array([_delta(f"persona_{i}", "pass2") for i in range(4)], dtype=np.float64)
            if swap_changed else d1_ag.copy()
        )

        if swap_changed:
            delta_base_deb = (d1_base - d2_base) / 2.0
            delta_agents_deb = (d1_ag - d2_ag) / 2.0
            positional_bias = (d1_base + d2_base) / 2.0
        else:
            delta_base_deb = d1_base
            delta_agents_deb = d1_ag
            positional_bias = 0.0

        out.append({
            "scenario_id": sid,
            "phenomenon_category": sample.get("phenomenon_category", "default"),
            "this_group_name": sample.get("this_group_name", ""),
            "preferred_on_right": int(sample.get("preferred_on_right", 1)),
            "n_left": int(sample.get("n_left", 0)),
            "n_right": int(sample.get("n_right", 0)),
            "lang": sample.get("lang", "en"),
            "country": sample.get("country", ""),
            "delta_base_deb": float(delta_base_deb),
            "delta_agents_deb": [float(x) for x in delta_agents_deb.tolist()],
            "positional_bias": float(positional_bias),
            "swap_changed": bool(swap_changed),
            "base_judge_choice": by_key[("base", "pass1")].get("judge_choice", "UNCERTAIN"),
            "base_judge_confidence": float(
                by_key[("base", "pass1")].get("judge_confidence", 0.0)
            ),
            "n_uncertain_agents": sum(
                1 for r in group if r.get("judge_choice") == "UNCERTAIN"
            ),
        })
    return out


def _run_dpbr_for_country(
    country: str,
    scenario_inputs: List[Dict[str, Any]],
    cfg: Stage2Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply Exp24DualPassControllerOffline to every scenario. Also build a
    vanilla-from-base DataFrame (no PT-IS)."""
    PRIOR_STATE.pop(country, None)
    PRIOR_STATE[country] = BootstrapPriorState()

    ctrl = Exp24DualPassControllerOffline(
        country_iso=country,
        lambda_coop=cfg.lambda_coop,
        decision_temperature=T_DECISION,
    )

    swa_rows: List[Dict[str, Any]] = []
    van_rows: List[Dict[str, Any]] = []
    for rec in scenario_inputs:
        pred = ctrl.predict_from_deltas(
            delta_base_deb=rec["delta_base_deb"],
            delta_agents_deb=rec["delta_agents_deb"],
            preferred_on_right=rec["preferred_on_right"],
            phenomenon_category=rec["phenomenon_category"],
            positional_bias=rec["positional_bias"],
            swap_changed=rec["swap_changed"],
        )
        swa_rows.append({
            "scenario_id": rec["scenario_id"],
            "phenomenon_category": rec["phenomenon_category"],
            "this_group_name": rec["this_group_name"],
            "preferred_on_right": rec["preferred_on_right"],
            "n_left": rec["n_left"],
            "n_right": rec["n_right"],
            "lang": rec["lang"],
            **pred,
            "n_uncertain_agents": rec["n_uncertain_agents"],
        })

        p_right_vanilla = pseudo_p_right_from_delta(
            rec["delta_base_deb"], t_decision=T_DECISION
        )
        p_pref_vanilla = (
            p_right_vanilla if rec["preferred_on_right"] else 1.0 - p_right_vanilla
        )
        van_rows.append({
            "scenario_id": rec["scenario_id"],
            "phenomenon_category": rec["phenomenon_category"],
            "this_group_name": rec["this_group_name"],
            "preferred_on_right": rec["preferred_on_right"],
            "n_left": rec["n_left"],
            "n_right": rec["n_right"],
            "lang": rec["lang"],
            "p_left": 1.0 - p_right_vanilla,
            "p_right": p_right_vanilla,
            "p_spare_preferred": p_pref_vanilla,
            "delta_base_deb": rec["delta_base_deb"],
            "judge_choice": rec["base_judge_choice"],
            "judge_confidence": rec["base_judge_confidence"],
            "positional_bias": rec["positional_bias"],
        })

    return pd.DataFrame(swa_rows), pd.DataFrame(van_rows)


def _summarize(
    results_df: pd.DataFrame, human_amce: Dict[str, float]
) -> Dict[str, Any]:
    model_amce = compute_amce_from_preferences(results_df)
    alignment = compute_alignment_metrics(model_amce, human_amce) if human_amce else {}
    per_dim = (
        compute_per_dimension_alignment(model_amce, human_amce) if human_amce else {}
    )
    return {
        "model_amce": model_amce,
        "alignment": alignment,
        "per_dimension_alignment": per_dim,
        "n_scenarios": int(len(results_df)),
    }


# ----------------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------------
def run_stage2(cfg: Stage2Config) -> None:
    if not cfg.countries:
        raise ValueError("Stage2Config.countries is empty")
    setup_seeds(cfg.seed)

    stage1_dir = Path(cfg.stage1_jsonl_dir)
    results_base = Path(cfg.results_base)
    swa_root = results_base / "swa"
    cmp_root = results_base / "compare"
    judge_cache_dir = results_base / "judge_cache"
    for d in (swa_root, cmp_root, judge_cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  OPEN-ENDED Stage 2 — judge=[{cfg.judge_model_name}]")
    print(f"{'='*70}")
    print(f"[CFG] stage1={stage1_dir}  out={results_base}  countries={cfg.countries}")

    judge_model, judge_tokenizer = load_model_hf_native(
        cfg.judge_model_name, max_seq_length=4096, load_in_4bit=cfg.load_in_4bit,
    )
    device = next(judge_model.parameters()).device

    compare_rows: List[Dict[str, Any]] = []
    try:
        for ci, country in enumerate(cfg.countries):
            jsonl_path = stage1_dir / f"{country}.jsonl"
            if not jsonl_path.exists():
                print(f"[SKIP] {country}: no Stage-1 JSONL at {jsonl_path}")
                continue
            print(f"\n[{ci+1}/{len(cfg.countries)}] judging {country}")

            rows = _load_stage1_rows(jsonl_path)
            if not rows:
                print(f"[SKIP] {country}: empty JSONL")
                continue

            cache_path = judge_cache_dir / f"{country}_judged.jsonl"
            t0 = time.time()
            rows, jstats = _judge_rows_for_country(
                rows, judge_model, judge_tokenizer,
                cache_path, cfg.max_new_tokens, device,
            )
            dt = time.time() - t0
            n_total = max(1, jstats["judged"] + jstats["cached"])
            fail_pct = 100.0 * jstats["parse_fail"] / n_total
            print(f"  judged={jstats['judged']}  cached={jstats['cached']}  "
                  f"skipped={jstats['skipped']}  parse_fail%={fail_pct:.1f}  t={dt:.1f}s")
            if fail_pct >= cfg.max_parse_fail_pct:
                print(f"[ERROR] {country} parse-fail rate {fail_pct:.1f}% "
                      f">= {cfg.max_parse_fail_pct:.1f}% — aborting this country")
                continue

            scen_inputs = _assemble_per_scenario(rows)
            if not scen_inputs:
                print(f"[SKIP] {country}: no complete scenarios after assembly")
                continue

            swa_df, van_df = _run_dpbr_for_country(country, scen_inputs, cfg)

            country_dir = swa_root / country
            country_dir.mkdir(parents=True, exist_ok=True)
            swa_df.to_csv(country_dir / "swa_results.csv", index=False)
            van_df.to_csv(country_dir / "vanilla_results.csv", index=False)

            human_amce = (
                load_human_amce(cfg.human_amce_path, country)
                if cfg.human_amce_path else {}
            )
            swa_summary = _summarize(swa_df, human_amce)
            van_summary = _summarize(van_df, human_amce)
            (country_dir / "summary.json").write_text(
                json.dumps({
                    "country": country,
                    "swa": swa_summary,
                    "vanilla_from_base": van_summary,
                    "judge_stats": jstats,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            ps = PRIOR_STATE.get(country, BootstrapPriorState()).stats
            compare_rows.append({
                "model": "phi4_openended",
                "judge": cfg.judge_model_name,
                "method": "baseline_vanilla_from_base",
                "country": country,
                **{f"align_{k}": v for k, v in van_summary.get("alignment", {}).items()},
                "n_scenarios": van_summary["n_scenarios"],
            })
            compare_rows.append({
                "model": "phi4_openended",
                "judge": cfg.judge_model_name,
                "method": "openended_dpbr",
                "country": country,
                **{f"align_{k}": v for k, v in swa_summary.get("alignment", {}).items()},
                "n_scenarios": swa_summary["n_scenarios"],
                "final_delta_country": ps["delta_country"],
                "final_alpha_h": ps["alpha_h"],
                "mean_reliability_r": float(swa_df["reliability_r"].mean())
                    if "reliability_r" in swa_df.columns else float("nan"),
                "mean_bootstrap_var": float(swa_df["bootstrap_var"].mean())
                    if "bootstrap_var" in swa_df.columns else float("nan"),
                "mean_ess_pass1": float(swa_df["ess_pass1"].mean())
                    if "ess_pass1" in swa_df.columns else float("nan"),
                "mean_ess_pass2": float(swa_df["ess_pass2"].mean())
                    if "ess_pass2" in swa_df.columns else float("nan"),
                "mean_ess_anchor_alpha": float(swa_df["ess_anchor_alpha"].mean())
                    if "ess_anchor_alpha" in swa_df.columns else float("nan"),
                "mean_positional_bias": float(swa_df["positional_bias"].mean())
                    if "positional_bias" in swa_df.columns else float("nan"),
                "mean_uncertain_per_scenario": float(swa_df["n_uncertain_agents"].mean())
                    if "n_uncertain_agents" in swa_df.columns else float("nan"),
                "judge_parse_fail_pct": fail_pct,
            })

            print(
                f"  [OK] {country}  SWA MIS={swa_summary.get('alignment', {}).get('mis', float('nan')):.4f}"
                f"  VAN MIS={van_summary.get('alignment', {}).get('mis', float('nan')):.4f}"
            )
            torch.cuda.empty_cache()
            gc.collect()

    finally:
        del judge_model, judge_tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cmp_df = pd.DataFrame(compare_rows)
    cmp_df.to_csv(cmp_root / "comparison.csv", index=False)
    print(f"\n[Stage 2] DONE — comparison at {cmp_root/'comparison.csv'}")
