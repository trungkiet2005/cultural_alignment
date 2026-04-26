"""Stage 2 of the open-ended DISCA variant — judge + DPBR + AMCE.

Reads Stage-1 JSONL files, runs a judge LLM on each actor generation, extracts
``{choice, confidence}``, converts to a scalar pseudo-delta via
:func:`src.pseudo_delta.pseudo_delta_from_judge`, and feeds it directly to
:class:`exp_paper.openended.dpbr_offline.Exp24DualPassControllerOffline` so PT-IS,
dual-pass bootstrap reliability, ESS anchor blend, and the hierarchical prior all
run unchanged. Per-country AMCE + alignment metrics are written using
:func:`src.amce.compute_amce_from_preferences` and
:func:`src.amce.compute_alignment_metrics`.

A "vanilla-from-base" analog is also emitted using only ``agent_role=="base"``
rows (no PT-IS, just the sigmoid of the base pseudo-delta).

Note (2026-04-26): The A↔B positional debiasing pass was removed (see
:mod:`exp_paper.openended.stage1_actor_phi4`). pseudo-deltas are used as-is.
Legacy JSONL rows with ``debias_variant == "pass2"`` are filtered out at load.

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
from exp_paper.openended.stage1_actor_phi4 import AGENT_ROLES


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
    model_label: str = "phi4_openended"


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
    """Load Stage-1 JSONL rows. Filters out legacy ``pass2`` rows — the A↔B
    positional debiasing pass was disabled (see :mod:`stage1_actor_phi4`)."""
    rows: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(rec.get("debias_variant", "pass1")) != "pass1":
                continue
            rows.append(rec)
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

    Single-pass: 5 rows/scenario (5 agents, pass1 only). pseudo-deltas are used
    directly — no positional debiasing (the A↔B swap pass was removed because in
    free-text greedy decoding it injects noise instead of cancelling bias).
    """
    by_sid: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        sid = int(r.get("scenario_id", -1))
        by_sid.setdefault(sid, []).append(r)

    out: List[Dict[str, Any]] = []
    for sid in sorted(by_sid.keys()):
        group = by_sid[sid]
        by_role = {r["agent_role"]: r for r in group}
        missing = [role for role in AGENT_ROLES if role not in by_role]
        if missing:
            print(f"[WARN] scenario_id={sid} missing roles {missing} — skipping")
            continue

        sample = group[0]

        def _delta(role: str) -> float:
            return float(by_role[role].get("pseudo_delta", 0.0))

        delta_base_deb = _delta("base")
        delta_agents_deb = np.array(
            [_delta(f"persona_{i}") for i in range(4)], dtype=np.float64
        )

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
            "positional_bias": 0.0,
            "swap_changed": False,
            "base_judge_choice": by_role["base"].get("judge_choice", "UNCERTAIN"),
            "base_judge_confidence": float(
                by_role["base"].get("judge_confidence", 0.0)
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
    vanilla-from-base DataFrame (no PT-IS).

    Hierarchical EMA prior is disabled in the offline controller (see
    :class:`Exp24DualPassControllerOffline`) — no per-country prior state
    setup is needed here.
    """
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
    print(f"  OPEN-ENDED Stage 2 (SWA-DPBR) — judge=[{cfg.judge_model_name}]")
    print(f"{'='*70}")
    print(f"[CFG] stage1={stage1_dir}  out={results_base}  countries={cfg.countries}")
    print(f"[CFG] max_new_tokens={cfg.max_new_tokens}  4bit={cfg.load_in_4bit}  "
          f"max_parse_fail_pct={cfg.max_parse_fail_pct}  model_label={cfg.model_label}")
    print(f"[CFG] EXP24_VAR_SCALE={os.environ.get('EXP24_VAR_SCALE', '0.04')}  "
          f"EXP24_K_HALF={os.environ.get('EXP24_K_HALF', '64')}  "
          f"EXP24_ESS_ANCHOR_REG={os.environ.get('EXP24_ESS_ANCHOR_REG', '1')}")

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

            compare_rows.append({
                "model": cfg.model_label,
                "judge": cfg.judge_model_name,
                "method": "baseline_vanilla_from_base",
                "country": country,
                **{f"align_{k}": v for k, v in van_summary.get("alignment", {}).items()},
                "n_scenarios": van_summary["n_scenarios"],
            })
            compare_rows.append({
                "model": cfg.model_label,
                "judge": cfg.judge_model_name,
                "method": "openended_dpbr",
                "country": country,
                **{f"align_{k}": v for k, v in swa_summary.get("alignment", {}).items()},
                "n_scenarios": swa_summary["n_scenarios"],
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

            swa_align = swa_summary.get("alignment", {})
            van_align = van_summary.get("alignment", {})
            swa_mis = swa_align.get("mis", float("nan"))
            van_mis = van_align.get("mis", float("nan"))
            mis_delta = van_mis - swa_mis if (
                not math.isnan(swa_mis) and not math.isnan(van_mis)
            ) else float("nan")
            print(
                f"  [OK] {country}  "
                f"SWA MIS={swa_mis:.4f}  r={swa_align.get('pearson_r', float('nan')):+.3f}  "
                f"JSD={swa_align.get('jsd', float('nan')):.4f}  |  "
                f"VAN MIS={van_mis:.4f}  r={van_align.get('pearson_r', float('nan')):+.3f}  "
                f"JSD={van_align.get('jsd', float('nan')):.4f}  |  "
                f"ΔMIS={mis_delta:+.4f}  n={swa_summary['n_scenarios']}  "
                f"parse_fail%={fail_pct:.1f}"
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

    if not cmp_df.empty:
        print(f"\n{'='*70}")
        print(f"  FINAL SUMMARY — OPEN-ENDED SWA-DPBR  ({len(cmp_df)} rows)")
        print(f"{'='*70}")
        cols = [c for c in (
            "country", "method", "align_mis", "align_pearson_r", "align_jsd",
            "n_scenarios", "mean_reliability_r", "mean_bootstrap_var",
            "mean_ess_pass1", "mean_ess_pass2", "mean_ess_anchor_alpha",
            "mean_positional_bias", "judge_parse_fail_pct",
        ) if c in cmp_df.columns]
        with pd.option_context("display.max_rows", None, "display.width", 200):
            print(cmp_df[cols].to_string(index=False))

        if "method" in cmp_df.columns and "align_mis" in cmp_df.columns:
            for method in cmp_df["method"].unique():
                sub = cmp_df[cmp_df["method"] == method]
                print(
                    f"\n[MEAN method={method}]  "
                    f"MIS={sub['align_mis'].mean():.4f}  "
                    f"r={sub['align_pearson_r'].mean():+.3f}  "
                    f"JSD={sub['align_jsd'].mean():.4f}  "
                    f"({len(sub)} countries)"
                )
            swa_sub = cmp_df[cmp_df["method"] == "openended_dpbr"]
            van_sub = cmp_df[cmp_df["method"] == "baseline_vanilla_from_base"]
            if not swa_sub.empty and not van_sub.empty:
                merged = swa_sub.merge(
                    van_sub[["country", "align_mis"]],
                    on="country", suffixes=("_swa", "_van"),
                )
                if not merged.empty:
                    delta = merged["align_mis_van"] - merged["align_mis_swa"]
                    print(
                        f"\n[ΔMIS van→swa] mean={delta.mean():+.4f}  "
                        f"median={delta.median():+.4f}  "
                        f"min={delta.min():+.4f}  max={delta.max():+.4f}"
                    )
