"""SAFE open-ended SWA-DPBR — guaranteed-not-worse-than-vanilla variant.

Three architectural fixes vs :mod:`exp_paper.openended.unified_actor_judge`,
each addressing a root cause identified in tracker_open_ended.md cross-scale
analysis:

  1. **Continuous-δ from judge logits** (fixes signal-bottleneck cause):
     instead of generating judge text and parsing it through the
     ``{A, B, UNCERTAIN}`` bottleneck (3-5 quantized values), we forward the
     judge ONE step on a "decisive" closing prompt and read the next-token
     logits at the A/B token positions. δ = logit(B) − logit(A) is real-valued,
     restoring the smooth signal regime PT-IS was designed for.

  2. **Per-scenario vanilla-anchored bounded blend** (fixes over-correction
     cause): after DPBR produces δ_swa, blend with vanilla as
     ``δ_final = (1−α)·δ_van + α·δ_swa`` with ``α ∈ [0, 0.30]`` and four hard
     safety gates (sign agreement on confident vanilla, DPBR reliability,
     magnitude bound, persona consensus). When ANY gate fails, α=0 and the
     scenario falls back to vanilla — identical output to the vanilla baseline.

  3. **Country-level abstain** (fixes catastrophic-country tail risk):
     if mean α across a country's scenarios is below ``country_min_alpha``,
     the SWA correction had no real leverage and the country reports
     vanilla output verbatim.

These three layers form a safety net: layer 1 enables signal-faithful SWA;
layer 2 caps SWA's per-scenario blast radius; layer 3 catches countries where
SWA still drifts net-negative after layer 2. By construction, on scenarios
where any gate fails the output IS the vanilla output — so the safe pipeline
cannot do worse than vanilla in expectation, modulo the (small) noise on
scenarios where all gates pass.

Reuses helper functions from :mod:`exp_paper.openended.unified_actor_judge`
where the schemas / prompts are identical (scenario loading, persona texts,
JSONL I/O, actor generation).
"""

from __future__ import annotations

import csv
import gc
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch

from src.amce import (
    compute_alignment_metrics,
    compute_amce_from_preferences,
    compute_per_dimension_alignment,
    load_human_amce,
)
from src.constants import COUNTRY_LANG
from src.judge_logits import judge_logit_delta, resolve_ab_token_ids
from src.model import ChatTemplateHelper, load_model_hf_native, setup_seeds
from src.openended_prompts import build_openended_prompt
from src.personas import SUPPORTED_COUNTRIES
from src.pseudo_delta import T_DECISION, pseudo_p_right_from_delta
from src.safe_blend import (
    SafeBlendConfig,
    country_level_decision,
    safe_blend_scalar,
)

from exp_paper.openended.dpbr_offline import Exp24DualPassControllerOffline
from exp_paper.openended.unified_actor_judge import (
    AGENT_ROLES,
    _actor_generate,
    _build_persona_texts,
    _load_combined_rows,
    _load_scenarios_bilingual,
    _read_existing_keys,
    _summarize,
)


SCHEMA_VERSION_SAFE: int = 3  # bump from unified v2 — judge_delta replaces pseudo_delta


@dataclass
class SafeUnifiedConfig:
    """Config for the SAFE open-ended pipeline.

    Mirrors :class:`UnifiedConfig` plus :class:`SafeBlendConfig` knobs.
    """
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    out_dir: str = "/kaggle/working/cultural_alignment/results/openended_safe"
    multitp_data_path: str = ""
    wvs_data_path: str = ""
    human_amce_path: str = ""
    use_real_data: bool = True
    n_scenarios: int = 500
    max_new_tokens_actor: int = 8
    load_in_4bit: bool = False
    seed: int = 42
    flush_every: int = 20
    countries: List[str] = field(default_factory=list)
    lambda_coop: float = 0.70
    model_label: str = "openended_safe_swa"
    # SafeBlend hyperparameters (see src/safe_blend.py for empirical motivation)
    alpha_max: float = 0.30
    dpbr_r_min: float = 0.85
    min_vanilla_conf: float = 0.5
    magnitude_ratio_max: float = 2.5
    blend_floor: float = 0.5
    persona_std_max: float = 3.0
    country_min_alpha: float = 0.05


CSV_FIELDS_SAFE: Tuple[str, ...] = (
    "country", "scenario_id", "phenomenon_category", "this_group_name",
    "preferred_on_right", "n_left", "n_right", "lang",
    "agent_role",
    "scenario_native", "scenario_en", "actor_text",
    "judge_choice", "judge_confidence",
    "judge_delta",  # CONTINUOUS — replaces pseudo_delta
    "actor_tokens", "actor_seconds", "judge_seconds",
)


# ----------------------------------------------------------------------------
# Per-country generation: actor (text) + judge (continuous logit delta)
# ----------------------------------------------------------------------------
def _generate_and_judge_safe(
    cfg: SafeUnifiedConfig,
    country: str,
    model, tokenizer, helper: ChatTemplateHelper, device: torch.device,
    a_token_id: int, b_token_id: int,
    combined_jsonl: Path, csv_path: Path,
) -> Dict[str, int]:
    """Run actor (generate) + judge (1-step logit forward) per (scenario, agent).

    Same JSONL/CSV layout as unified_actor_judge but replaces the
    pseudo_delta column with a real-valued ``judge_delta``.
    """
    seen = _read_existing_keys(combined_jsonl)
    print(f"  existing rows in {combined_jsonl.name}: {len(seen)}")

    scen = _load_scenarios_bilingual(cfg, country)
    lang = COUNTRY_LANG.get(country, "en")
    personas = _build_persona_texts(country, cfg.wvs_data_path, lang)
    assert len(personas) == len(AGENT_ROLES)

    csv_needs_header = (not csv_path.exists()) or csv_path.stat().st_size == 0
    stats = {"new_rows": 0, "uncertain": 0}
    rows_since_flush = 0

    with combined_jsonl.open("a", encoding="utf-8") as fh, \
         csv_path.open("a", encoding="utf-8", newline="") as csv_fh:
        csv_writer = csv.DictWriter(
            csv_fh, fieldnames=list(CSV_FIELDS_SAFE),
            quoting=csv.QUOTE_ALL, lineterminator="\n",
        )
        if csv_needs_header:
            csv_writer.writeheader()

        for sid, row in scen.iterrows():
            sid = int(sid)
            scenario_native = str(row["scenario_native"])
            scenario_en = str(row["scenario_en"])
            phenom_cat = str(row.get("phenomenon_category", "default"))
            group_name = str(row.get("this_group_name", ""))
            pref_on_right = int(row.get("preferred_on_right", 1))
            n_left = int(row.get("n_left", 0))
            n_right = int(row.get("n_right", 0))

            for agent_idx, agent_role in enumerate(AGENT_ROLES):
                key = (country, sid, agent_role)
                if key in seen:
                    continue
                persona_text = personas[agent_idx]
                user_content = build_openended_prompt(scenario_native, lang)

                actor_text, a_tok, a_sec = _actor_generate(
                    model, tokenizer, helper,
                    persona_text, user_content,
                    cfg.max_new_tokens_actor, device,
                )
                t_j0 = time.time()
                delta, choice, conf = judge_logit_delta(
                    model, tokenizer, scenario_en, actor_text,
                    a_token_id, b_token_id, device,
                )
                j_sec = time.time() - t_j0

                if choice == "UNCERTAIN":
                    stats["uncertain"] += 1

                rec = {
                    "country": country,
                    "scenario_id": sid,
                    "phenomenon_category": phenom_cat,
                    "this_group_name": group_name,
                    "preferred_on_right": pref_on_right,
                    "n_left": n_left,
                    "n_right": n_right,
                    "lang": lang,
                    "scenario_native": scenario_native,
                    "scenario_en": scenario_en,
                    "agent_role": agent_role,
                    "persona_text": persona_text,
                    "prompt": user_content,
                    "actor_text": actor_text,
                    "judge_choice": choice,
                    "judge_confidence": float(conf),
                    "judge_delta": float(delta),  # CONTINUOUS
                    "actor_tokens": a_tok,
                    "actor_seconds": a_sec,
                    "judge_seconds": j_sec,
                    "schema_version": SCHEMA_VERSION_SAFE,
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                csv_writer.writerow({k: rec.get(k, "") for k in CSV_FIELDS_SAFE})
                stats["new_rows"] += 1
                rows_since_flush += 1
                if rows_since_flush >= cfg.flush_every:
                    fh.flush(); csv_fh.flush()
                    rows_since_flush = 0
        fh.flush(); csv_fh.flush()
    return stats


def _assemble_per_scenario_safe(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group combined rows by scenario_id; require all 5 agent roles."""
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
        delta_base = float(by_role["base"].get("judge_delta", 0.0))
        delta_agents = [
            float(by_role[f"persona_{i}"].get("judge_delta", 0.0))
            for i in range(4)
        ]
        out.append({
            "scenario_id": sid,
            "phenomenon_category": sample.get("phenomenon_category", "default"),
            "this_group_name": sample.get("this_group_name", ""),
            "preferred_on_right": int(sample.get("preferred_on_right", 1)),
            "n_left": int(sample.get("n_left", 0)),
            "n_right": int(sample.get("n_right", 0)),
            "lang": sample.get("lang", "en"),
            "country": sample.get("country", ""),
            "delta_base_deb": delta_base,
            "delta_agents_deb": delta_agents,
            "positional_bias": 0.0,
            "swap_changed": False,
            "base_judge_choice": by_role["base"].get("judge_choice", "UNCERTAIN"),
            "base_judge_confidence": float(by_role["base"].get("judge_confidence", 0.0)),
            "n_uncertain_agents": sum(
                1 for r in group if r.get("judge_choice") == "UNCERTAIN"
            ),
        })
    return out


def _safe_blend_cfg_from_run(cfg: SafeUnifiedConfig) -> SafeBlendConfig:
    return SafeBlendConfig(
        alpha_max=cfg.alpha_max,
        dpbr_r_min=cfg.dpbr_r_min,
        min_vanilla_conf=cfg.min_vanilla_conf,
        magnitude_ratio_max=cfg.magnitude_ratio_max,
        floor=cfg.blend_floor,
        persona_std_max=cfg.persona_std_max,
        country_min_alpha=cfg.country_min_alpha,
    )


def _run_dpbr_safe_for_country(
    country: str,
    scenario_inputs: List[Dict[str, Any]],
    cfg: SafeUnifiedConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Run DPBR + safe-blend per scenario; produce vanilla / raw-SWA / safe-blend frames.

    Returns:
        safe_df: SAFE = bounded blend or vanilla fallback (the recommended output).
        raw_swa_df: pure SWA-DPBR for comparison/diagnostics.
        van_df: vanilla baseline (continuous-δ from base persona only).
        gate_summary: aggregated gate-firing counters and country-level decision.
    """
    blend_cfg = _safe_blend_cfg_from_run(cfg)
    ctrl = Exp24DualPassControllerOffline(
        country_iso=country,
        lambda_coop=cfg.lambda_coop,
        decision_temperature=T_DECISION,
    )

    safe_rows: List[Dict[str, Any]] = []
    raw_swa_rows: List[Dict[str, Any]] = []
    van_rows: List[Dict[str, Any]] = []
    alphas: List[float] = []
    gate_counters = {"sign": 0, "dpbr": 0, "magnitude": 0, "consensus": 0, "all_pass": 0}

    for rec in scenario_inputs:
        delta_van = float(rec["delta_base_deb"])
        agents = list(rec["delta_agents_deb"])
        persona_std = float(np.std(agents)) if len(agents) >= 2 else 0.0

        pred = ctrl.predict_from_deltas(
            delta_base_deb=delta_van,
            delta_agents_deb=agents,
            preferred_on_right=rec["preferred_on_right"],
            phenomenon_category=rec["phenomenon_category"],
            positional_bias=rec["positional_bias"],
            swap_changed=rec["swap_changed"],
        )
        delta_swa = float(pred["delta_opt"])
        diag = {
            "reliability_r": float(pred.get("reliability_r", 0.0)),
            "persona_std": persona_std,
        }
        delta_safe, alpha_used, gate_flags = safe_blend_scalar(
            delta_van=delta_van, delta_swa=delta_swa,
            diag=diag, cfg=blend_cfg,
        )
        alphas.append(alpha_used)
        for k, passed in gate_flags.items():
            if not passed:
                gate_counters[k] += 1
        if all(gate_flags.values()):
            gate_counters["all_pass"] += 1

        # --- Vanilla baseline frame (no SWA, no blend)
        p_right_van = pseudo_p_right_from_delta(delta_van, t_decision=T_DECISION)
        p_pref_van = p_right_van if rec["preferred_on_right"] else 1.0 - p_right_van
        van_rows.append({
            "scenario_id": rec["scenario_id"],
            "phenomenon_category": rec["phenomenon_category"],
            "this_group_name": rec["this_group_name"],
            "preferred_on_right": rec["preferred_on_right"],
            "n_left": rec["n_left"],
            "n_right": rec["n_right"],
            "lang": rec["lang"],
            "p_left": 1.0 - p_right_van,
            "p_right": p_right_van,
            "p_spare_preferred": p_pref_van,
            "delta_base_deb": delta_van,
            "judge_choice": rec["base_judge_choice"],
            "judge_confidence": rec["base_judge_confidence"],
            "positional_bias": rec["positional_bias"],
        })

        # --- Raw SWA frame (DPBR output, no blend) — for ablation reporting
        raw_swa_rows.append({
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

        # --- SAFE blend frame: derived p_right from δ_safe
        p_right_safe = pseudo_p_right_from_delta(delta_safe, t_decision=T_DECISION)
        p_pref_safe = p_right_safe if rec["preferred_on_right"] else 1.0 - p_right_safe
        safe_rows.append({
            "scenario_id": rec["scenario_id"],
            "phenomenon_category": rec["phenomenon_category"],
            "this_group_name": rec["this_group_name"],
            "preferred_on_right": rec["preferred_on_right"],
            "n_left": rec["n_left"],
            "n_right": rec["n_right"],
            "lang": rec["lang"],
            "p_left": 1.0 - p_right_safe,
            "p_right": p_right_safe,
            "p_spare_preferred": p_pref_safe,
            "delta_base_deb": delta_van,
            "delta_swa": delta_swa,
            "delta_safe": delta_safe,
            "alpha_used": alpha_used,
            "reliability_r": diag["reliability_r"],
            "persona_std": persona_std,
            "gate_sign": gate_flags["sign"],
            "gate_dpbr": gate_flags["dpbr"],
            "gate_magnitude": gate_flags["magnitude"],
            "gate_consensus": gate_flags["consensus"],
            "judge_choice": rec["base_judge_choice"],
            "judge_confidence": rec["base_judge_confidence"],
            "positional_bias": rec["positional_bias"],
            "n_uncertain_agents": rec["n_uncertain_agents"],
        })

    commit_swa, mean_alpha = country_level_decision(alphas, blend_cfg)
    n = max(1, len(scenario_inputs))
    gate_summary = {
        "n_scenarios": len(scenario_inputs),
        "mean_alpha": mean_alpha,
        "commit_swa": bool(commit_swa),
        "pct_all_gates_pass": 100.0 * gate_counters["all_pass"] / n,
        "pct_gate_sign_fail": 100.0 * gate_counters["sign"] / n,
        "pct_gate_dpbr_fail": 100.0 * gate_counters["dpbr"] / n,
        "pct_gate_magnitude_fail": 100.0 * gate_counters["magnitude"] / n,
        "pct_gate_consensus_fail": 100.0 * gate_counters["consensus"] / n,
    }

    safe_df = pd.DataFrame(safe_rows)
    raw_swa_df = pd.DataFrame(raw_swa_rows)
    van_df = pd.DataFrame(van_rows)

    # Country-level safety net: if SWA had no leverage, return vanilla as safe.
    if not commit_swa:
        cols = [c for c in safe_df.columns if c in van_df.columns]
        # Replace probability columns with vanilla's so AMCE matches vanilla exactly.
        safe_df = safe_df.copy()
        for col in ("p_left", "p_right", "p_spare_preferred"):
            if col in van_df.columns:
                safe_df[col] = van_df[col].values
        safe_df["alpha_used"] = 0.0
        safe_df["country_fallback"] = True
    else:
        safe_df["country_fallback"] = False

    return safe_df, raw_swa_df, van_df, gate_summary


# ----------------------------------------------------------------------------
# Startup sanity checks
# ----------------------------------------------------------------------------
def _self_test_judge(
    model, tokenizer, a_token_id: int, b_token_id: int, device: torch.device,
) -> None:
    """Run judge_logit_delta on a tiny synthetic input.

    Catches three common failure modes BEFORE the 12h sweep starts:
      - tokenizer A/B resolution collapsed to same id (caught by resolve_ab_token_ids)
      - judge forward returns NaN/Inf (numerical instability)
      - judge produces zero variation (model frozen / wrong weights loaded)
    """
    print("[self-test] running judge_logit_delta on synthetic A-leaning input ...")
    delta_a, choice_a, conf_a = judge_logit_delta(
        model, tokenizer,
        scenario_en="There is a runaway car. Option A: stay. Option B: swerve.",
        actor_text="I choose option A. Staying minimizes harm.",
        a_token_id=a_token_id, b_token_id=b_token_id, device=device,
    )
    print(f"[self-test] A-leaning input: delta={delta_a:+.3f}  choice={choice_a}  conf={conf_a:.3f}")

    print("[self-test] running judge_logit_delta on synthetic B-leaning input ...")
    delta_b, choice_b, conf_b = judge_logit_delta(
        model, tokenizer,
        scenario_en="There is a runaway car. Option A: stay. Option B: swerve.",
        actor_text="I choose option B. Swerving saves more lives.",
        a_token_id=a_token_id, b_token_id=b_token_id, device=device,
    )
    print(f"[self-test] B-leaning input: delta={delta_b:+.3f}  choice={choice_b}  conf={conf_b:.3f}")

    # Validate: delta must be finite, A-leaning should give negative δ, B-leaning positive,
    # and the two should differ by at least ~0.5 (otherwise model is not responsive).
    if not (math.isfinite(delta_a) and math.isfinite(delta_b)):
        raise RuntimeError(
            f"[self-test FAIL] judge produced non-finite delta(s): "
            f"A-leaning={delta_a}, B-leaning={delta_b}. Model load or numerics broken."
        )
    if abs(delta_b - delta_a) < 0.5:
        print(
            f"[self-test WARN] judge spread is small ({delta_b - delta_a:+.3f}). "
            f"Either the model is weakly responsive to A/B prompts or token IDs are "
            f"resolving to a sub-character form. Continuing anyway, but DPBR signal "
            f"may be weak — check first-country rel_r."
        )
    if delta_a >= 0 or delta_b <= 0:
        print(
            f"[self-test WARN] judge sign flipped vs expected: "
            f"A-leaning δ should be < 0 (got {delta_a:+.3f}), "
            f"B-leaning δ should be > 0 (got {delta_b:+.3f}). "
            f"This is a 3B-style inversion — SAFE blend will route around it via "
            f"vanilla fallback, but expect high gate-fail rates."
        )
    else:
        print(f"[self-test OK] judge responds correctly to A/B leaning inputs.")


def _check_dpbr_collapse(country: str, raw_swa_df: pd.DataFrame) -> bool:
    """Return True if mean reliability_r is so low DPBR has likely collapsed.

    Open-ended continuous-δ may have a different bootstrap-variance regime than
    pseudo-δ. If rel_r < 0.30 across a country, the gate ``dpbr_r_min=0.85``
    will fire on virtually every scenario → SAFE pipeline degenerates to pure
    vanilla. Surface this as an actionable warning.
    """
    if "reliability_r" not in raw_swa_df.columns or raw_swa_df.empty:
        return False
    mean_r = float(raw_swa_df["reliability_r"].mean())
    if mean_r < 0.30:
        print(
            f"\n[DPBR COLLAPSE WARNING] {country}: mean reliability_r={mean_r:.3f} < 0.30. "
            f"Continuous-δ scale may be larger than pseudo-δ → bootstrap_var inflated → "
            f"r=exp(-bv/0.04) collapses. Recommended fix: set EXP24_VAR_SCALE=1.0 in the "
            f"launcher (Plan B toggle) and re-run. SAFE pipeline will continue but expect "
            f"100% country fallback to vanilla until VAR_SCALE is loosened."
        )
        return True
    return False


# ----------------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------------
def run_unified_safe(cfg: SafeUnifiedConfig) -> None:
    """Generate, judge (continuous), DPBR + safe-blend, score — all countries."""
    if not cfg.countries:
        raise ValueError("SafeUnifiedConfig.countries is empty")
    setup_seeds(cfg.seed)

    out_root = Path(cfg.out_dir)
    combined_dir = out_root / "combined"
    safe_root = out_root / "safe"
    cmp_root = out_root / "compare"
    for d in (combined_dir, safe_root, cmp_root):
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  OPEN-ENDED SAFE SWA-DPBR — model=[{cfg.model_name}]")
    print(f"{'='*70}")
    print(f"[CFG] out={out_root}  countries={cfg.countries}  n_scenarios={cfg.n_scenarios}")
    print(f"[CFG] actor_max={cfg.max_new_tokens_actor}  4bit={cfg.load_in_4bit}  seed={cfg.seed}")
    print(f"[CFG] alpha_max={cfg.alpha_max}  dpbr_r_min={cfg.dpbr_r_min}  "
          f"mag_ratio_max={cfg.magnitude_ratio_max}  persona_std_max={cfg.persona_std_max}  "
          f"country_min_alpha={cfg.country_min_alpha}")

    model, tokenizer = load_model_hf_native(
        cfg.model_name, max_seq_length=4096, load_in_4bit=cfg.load_in_4bit,
    )
    helper = ChatTemplateHelper(tokenizer)
    device = next(model.parameters()).device

    a_token_id, b_token_id = resolve_ab_token_ids(tokenizer)

    # ── Startup self-test: judge_logit_delta on a synthetic input ─────────────
    # Verifies tokenizer A/B resolution + judge forward pass produces a
    # well-formed continuous δ before launching the 12h sweep. Cheap (~1s).
    _self_test_judge(model, tokenizer, a_token_id, b_token_id, device)

    compare_rows: List[Dict[str, Any]] = []
    rel_r_collapsed_warned = False
    try:
        for ci, country in enumerate(cfg.countries):
            if country not in SUPPORTED_COUNTRIES:
                print(f"[SKIP] unsupported country: {country}")
                continue

            combined_jsonl = combined_dir / f"{country}.jsonl"
            csv_path = combined_dir / f"{country}.csv"
            print(f"\n[{ci+1}/{len(cfg.countries)}] {country}  -> {combined_jsonl.name}")

            t0 = time.time()
            stats = _generate_and_judge_safe(
                cfg, country, model, tokenizer, helper, device,
                a_token_id, b_token_id, combined_jsonl, csv_path,
            )
            dt = time.time() - t0
            print(f"  generated+judged new_rows={stats['new_rows']}  "
                  f"uncertain={stats['uncertain']}  t={dt:.1f}s")

            rows = _load_combined_rows(combined_jsonl)
            if not rows:
                print(f"[SKIP] {country}: empty combined JSONL")
                continue

            scen_inputs = _assemble_per_scenario_safe(rows)
            if not scen_inputs:
                print(f"[SKIP] {country}: no complete scenarios after assembly")
                continue

            safe_df, raw_swa_df, van_df, gate_summary = _run_dpbr_safe_for_country(
                country, scen_inputs, cfg,
            )

            # First-country DPBR collapse check: if rel_r is broadly below 0.30
            # the continuous-δ scale is incompatible with VAR_SCALE=0.04. Warn
            # once (later countries will exhibit the same issue).
            if not rel_r_collapsed_warned:
                if _check_dpbr_collapse(country, raw_swa_df):
                    rel_r_collapsed_warned = True

            country_dir = safe_root / country
            country_dir.mkdir(parents=True, exist_ok=True)
            safe_df.to_csv(country_dir / "safe_results.csv", index=False)
            raw_swa_df.to_csv(country_dir / "raw_swa_results.csv", index=False)
            van_df.to_csv(country_dir / "vanilla_results.csv", index=False)

            human_amce = (
                load_human_amce(cfg.human_amce_path, country)
                if cfg.human_amce_path else {}
            )
            safe_summary = _summarize(safe_df, human_amce)
            raw_summary = _summarize(raw_swa_df, human_amce)
            van_summary = _summarize(van_df, human_amce)
            (country_dir / "summary.json").write_text(
                json.dumps({
                    "country": country,
                    "safe": safe_summary,
                    "raw_swa": raw_summary,
                    "vanilla": van_summary,
                    "gate_summary": gate_summary,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            compare_rows.append({
                "model": cfg.model_label,
                "method": "vanilla_continuous",
                "country": country,
                **{f"align_{k}": v for k, v in van_summary.get("alignment", {}).items()},
                "n_scenarios": van_summary["n_scenarios"],
            })
            compare_rows.append({
                "model": cfg.model_label,
                "method": "raw_swa_dpbr",
                "country": country,
                **{f"align_{k}": v for k, v in raw_summary.get("alignment", {}).items()},
                "n_scenarios": raw_summary["n_scenarios"],
            })
            compare_rows.append({
                "model": cfg.model_label,
                "method": "safe_swa_blend",
                "country": country,
                **{f"align_{k}": v for k, v in safe_summary.get("alignment", {}).items()},
                "n_scenarios": safe_summary["n_scenarios"],
                "mean_alpha": gate_summary["mean_alpha"],
                "country_fallback": (not gate_summary["commit_swa"]),
                "pct_all_gates_pass": gate_summary["pct_all_gates_pass"],
                "pct_sign_fail": gate_summary["pct_gate_sign_fail"],
                "pct_dpbr_fail": gate_summary["pct_gate_dpbr_fail"],
                "pct_magnitude_fail": gate_summary["pct_gate_magnitude_fail"],
                "pct_consensus_fail": gate_summary["pct_gate_consensus_fail"],
            })

            van_a = van_summary.get("alignment", {})
            raw_a = raw_summary.get("alignment", {})
            safe_a = safe_summary.get("alignment", {})
            print(
                f"  [OK] {country}  "
                f"VAN MIS={van_a.get('mis', float('nan')):.4f}  "
                f"RAW MIS={raw_a.get('mis', float('nan')):.4f}  "
                f"SAFE MIS={safe_a.get('mis', float('nan')):.4f}  |  "
                f"r van={van_a.get('pearson_r', float('nan')):+.3f} "
                f"raw={raw_a.get('pearson_r', float('nan')):+.3f} "
                f"safe={safe_a.get('pearson_r', float('nan')):+.3f}  |  "
                f"α̅={gate_summary['mean_alpha']:.3f}  "
                f"all_gates_pass%={gate_summary['pct_all_gates_pass']:.1f}  "
                f"fallback={not gate_summary['commit_swa']}"
            )
            torch.cuda.empty_cache(); gc.collect()

    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cmp_df = pd.DataFrame(compare_rows)
    cmp_df.to_csv(cmp_root / "comparison.csv", index=False)
    print(f"\n[SAFE] DONE — comparison at {cmp_root/'comparison.csv'}")

    if not cmp_df.empty:
        print(f"\n{'='*70}")
        print(f"  FINAL SUMMARY — SAFE OPEN-ENDED SWA-DPBR  ({len(cmp_df)} rows)")
        print(f"{'='*70}")
        cols = [c for c in (
            "country", "method", "align_mis", "align_pearson_r", "align_jsd",
            "n_scenarios", "mean_alpha", "country_fallback", "pct_all_gates_pass",
        ) if c in cmp_df.columns]
        with pd.option_context("display.max_rows", None, "display.width", 200):
            print(cmp_df[cols].to_string(index=False))

        if "method" in cmp_df.columns and "align_mis" in cmp_df.columns:
            for method in ("vanilla_continuous", "raw_swa_dpbr", "safe_swa_blend"):
                sub = cmp_df[cmp_df["method"] == method]
                if sub.empty:
                    continue
                print(
                    f"\n[MEAN method={method}]  "
                    f"MIS={sub['align_mis'].mean():.4f}  "
                    f"r={sub['align_pearson_r'].mean():+.3f}  "
                    f"JSD={sub['align_jsd'].mean():.4f}  "
                    f"({len(sub)} countries)"
                )
            van_sub = cmp_df[cmp_df["method"] == "vanilla_continuous"][["country", "align_mis"]]
            safe_sub = cmp_df[cmp_df["method"] == "safe_swa_blend"][["country", "align_mis"]]
            raw_sub = cmp_df[cmp_df["method"] == "raw_swa_dpbr"][["country", "align_mis"]]
            if not van_sub.empty and not safe_sub.empty:
                m = safe_sub.merge(van_sub, on="country", suffixes=("_safe", "_van"))
                if not m.empty:
                    delta = m["align_mis_van"] - m["align_mis_safe"]
                    print(
                        f"\n[ΔMIS van→SAFE] mean={delta.mean():+.4f}  "
                        f"median={delta.median():+.4f}  "
                        f"min={delta.min():+.4f}  max={delta.max():+.4f}  "
                        f"wins={int((delta > 0).sum())}/{len(delta)}"
                    )
            if not van_sub.empty and not raw_sub.empty:
                m = raw_sub.merge(van_sub, on="country", suffixes=("_raw", "_van"))
                if not m.empty:
                    delta = m["align_mis_van"] - m["align_mis_raw"]
                    print(
                        f"[ΔMIS van→RAW]  mean={delta.mean():+.4f}  "
                        f"median={delta.median():+.4f}  "
                        f"min={delta.min():+.4f}  max={delta.max():+.4f}  "
                        f"wins={int((delta > 0).sum())}/{len(delta)}"
                    )
