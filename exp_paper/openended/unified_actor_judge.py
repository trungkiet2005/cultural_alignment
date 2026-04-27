"""Unified open-ended SWA-DPBR — actor + judge merged into a single pass.

Equivalent to running ``stage1_actor_phi4.run_stage1`` followed by
``stage2_judge_qwen72b.run_stage2`` back-to-back, but the model is loaded ONCE
(actor and judge are the same Qwen2.5-7B BF16 weights in the DISCA variant).

Per (country, scenario, agent) the pipeline:
    1. Generate the actor's A/B response (greedy, ``max_new_tokens_actor`` tokens).
    2. Run the judge on (scenario_en, actor_text) -> {choice, confidence}.
    3. Convert to a scalar pseudo-delta via :func:`pseudo_delta_from_judge`.

Per country, after all scenarios are generated+judged:
    4. Assemble per-scenario inputs (5 agents: base + 4 personas).
    5. Run :class:`Exp24DualPassControllerOffline` (PT-IS + DPBR + ESS-anchor).
    6. Compute AMCE and alignment vs human (MIS / Pearson r / JSD).

Resume-safe: every (country, sid, agent_role) row is appended to
``{out_dir}/{country}.jsonl`` with both actor_text AND judge fields. Re-running
the script reads existing keys and skips them. Per-country DPBR/AMCE step is
re-run from cached rows on every invocation (cheap).

Intended entry point: :func:`run_unified` — replaces the two-stage
``run_stage1`` + ``run_stage2`` pair. See
``exp_paper/exp_paper_openended_with_DISCA.py`` for the Kaggle launcher.
"""

from __future__ import annotations

import csv
import gc
import hashlib
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
from src.data import load_multitp_dataset
from src.i18n import BASE_ASSISTANT_I18N
from src.judge_prompts import (
    JUDGE_SYSTEM_PROMPT,
    build_judge_prompt,
    parse_judge_output,
)
from src.model import ChatTemplateHelper, load_model_hf_native, setup_seeds
from src.openended_prompts import build_openended_prompt
from src.personas import SUPPORTED_COUNTRIES, build_country_personas
from src.pseudo_delta import (
    T_DECISION,
    pseudo_delta_from_judge,
    pseudo_p_right_from_delta,
)
from src.scenarios import generate_multitp_scenarios

from exp_paper.openended.dpbr_offline import Exp24DualPassControllerOffline


AGENT_ROLES: Tuple[str, ...] = ("base", "persona_0", "persona_1", "persona_2", "persona_3")
SCHEMA_VERSION: int = 2  # bumped from stage1's v1 — combined rows include judge fields

CSV_FIELDS: Tuple[str, ...] = (
    "country", "scenario_id", "phenomenon_category", "this_group_name",
    "preferred_on_right", "n_left", "n_right", "lang",
    "agent_role",
    "scenario_native", "scenario_en", "actor_text",
    "judge_choice", "judge_confidence", "judge_parse_ok",
    "pseudo_delta",
    "actor_tokens", "actor_seconds", "judge_tokens", "judge_seconds",
)


@dataclass
class UnifiedConfig:
    """Single-pass config — replaces Stage1Config + Stage2Config.

    The same model weights are used for the actor (constrained A/B emission)
    and the judge (free-form parse). Loading once cuts wall-time in half on
    Kaggle and removes the need for two separate sessions.
    """
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    out_dir: str = "/kaggle/working/cultural_alignment/results/openended"
    multitp_data_path: str = ""
    wvs_data_path: str = ""
    human_amce_path: str = ""
    use_real_data: bool = True
    n_scenarios: int = 500
    max_new_tokens_actor: int = 8
    max_new_tokens_judge: int = 64
    load_in_4bit: bool = False
    seed: int = 42
    flush_every: int = 20
    countries: List[str] = field(default_factory=list)
    lambda_coop: float = 0.70
    model_label: str = "qwen25_7b_openended_unified"
    max_parse_fail_pct: float = 5.0
    # If True, ALSO write a separate ``stage1/{country}.jsonl`` mirror for
    # back-compat with downstream tools. Off by default (combined JSONL is the
    # source of truth).
    write_stage1_mirror: bool = False


# ----------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------
def _load_scenarios_bilingual(
    cfg: UnifiedConfig, country: str
) -> pd.DataFrame:
    """Load native + English scenarios; same seed -> aligned ordering."""
    lang = COUNTRY_LANG.get(country, "en")
    if cfg.use_real_data and cfg.multitp_data_path and os.path.isdir(cfg.multitp_data_path):
        df_native = load_multitp_dataset(
            data_base_path=cfg.multitp_data_path, lang=lang,
            n_scenarios=cfg.n_scenarios, seed=cfg.seed,
        ).reset_index(drop=True)
        df_en = load_multitp_dataset(
            data_base_path=cfg.multitp_data_path, lang="en",
            n_scenarios=cfg.n_scenarios, seed=cfg.seed,
        ).reset_index(drop=True)
    else:
        df_native = generate_multitp_scenarios(
            cfg.n_scenarios, seed=cfg.seed, lang=lang
        ).reset_index(drop=True)
        df_en = generate_multitp_scenarios(
            cfg.n_scenarios, seed=cfg.seed, lang="en"
        ).reset_index(drop=True)

    n = min(len(df_native), len(df_en))
    df_native = df_native.iloc[:n].copy()
    df_en = df_en.iloc[:n].copy()
    df_native["scenario_en"] = df_en["Prompt"].values
    df_native["lang"] = lang
    df_native.rename(columns={"Prompt": "scenario_native"}, inplace=True)
    return df_native


def _build_persona_texts(country: str, wvs_path: str, lang: str) -> List[str]:
    base_text = BASE_ASSISTANT_I18N.get(lang, BASE_ASSISTANT_I18N["en"])
    cultural = build_country_personas(country, wvs_path=wvs_path)
    if len(cultural) < 4:
        cultural = list(cultural) + [cultural[-1]] * (4 - len(cultural))
    return [base_text] + list(cultural[:4])


def _read_existing_keys(jsonl_path: Path) -> Set[Tuple[str, int, str]]:
    keys: Set[Tuple[str, int, str]] = set()
    if not jsonl_path.exists():
        return keys
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys.add((
                str(rec.get("country", "")),
                int(rec.get("scenario_id", -1)),
                str(rec.get("agent_role", "")),
            ))
    return keys


def _load_combined_rows(jsonl_path: Path) -> List[Dict[str, Any]]:
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
            rows.append(rec)
    return rows


# ----------------------------------------------------------------------------
# Generation primitives — actor and judge share the same model handle
# ----------------------------------------------------------------------------
@torch.no_grad()
def _actor_generate(
    model, tokenizer, helper: ChatTemplateHelper,
    persona_text: str, user_content: str,
    max_new_tokens: int, device: torch.device,
) -> Tuple[str, int, float]:
    """Greedy decode the actor. Returns (text, n_new_tokens, seconds)."""
    prefix_ids = helper.build_prefix_ids(persona_text, device)
    query_ids = helper.encode_query_suffix(user_content, device)
    input_ids = torch.cat([prefix_ids, query_ids], dim=1)
    t0 = time.time()
    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=(
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        ),
    )
    elapsed = time.time() - t0
    new_ids = out[0, input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True), int(new_ids.shape[0]), elapsed


@torch.no_grad()
def _judge_generate(
    model, tokenizer,
    scenario_en: str, actor_text: str,
    max_new_tokens: int, device: torch.device,
) -> Tuple[str, int, float]:
    """Greedy decode the judge. Returns (raw_text, n_new_tokens, seconds)."""
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": build_judge_prompt(scenario_en, actor_text)},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        templated = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
        )
        if isinstance(templated, torch.Tensor):
            input_ids = templated.to(device)
        else:
            input_ids = templated["input_ids"].to(device)
    else:
        prompt = (
            f"{JUDGE_SYSTEM_PROMPT}\n\n"
            f"{build_judge_prompt(scenario_en, actor_text)}\n"
        )
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    t0 = time.time()
    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=(
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        ),
    )
    elapsed = time.time() - t0
    new_ids = out[0, input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True), int(new_ids.shape[0]), elapsed


# ----------------------------------------------------------------------------
# Per-country pipeline pieces
# ----------------------------------------------------------------------------
def _generate_and_judge_country(
    cfg: UnifiedConfig,
    country: str,
    model, tokenizer, helper: ChatTemplateHelper, device: torch.device,
    combined_jsonl: Path, csv_path: Path,
) -> Dict[str, int]:
    """Run actor+judge for every (scenario, agent) of one country.

    Appends one combined row per (sid, agent_role) to ``combined_jsonl`` and a
    column-subset mirror to ``csv_path``. Returns simple counters for logging.
    """
    seen = _read_existing_keys(combined_jsonl)
    print(f"  existing rows in {combined_jsonl.name}: {len(seen)}")

    scen = _load_scenarios_bilingual(cfg, country)
    lang = COUNTRY_LANG.get(country, "en")
    personas = _build_persona_texts(country, cfg.wvs_data_path, lang)
    assert len(personas) == len(AGENT_ROLES), (
        f"Expected {len(AGENT_ROLES)} personas, got {len(personas)}"
    )

    csv_needs_header = (not csv_path.exists()) or csv_path.stat().st_size == 0
    stats = {"new_rows": 0, "parse_fail": 0, "uncertain": 0}
    rows_since_flush = 0

    with combined_jsonl.open("a", encoding="utf-8") as fh, \
         csv_path.open("a", encoding="utf-8", newline="") as csv_fh:
        csv_writer = csv.DictWriter(
            csv_fh, fieldnames=list(CSV_FIELDS),
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
                judge_raw, j_tok, j_sec = _judge_generate(
                    model, tokenizer,
                    scenario_en, actor_text,
                    cfg.max_new_tokens_judge, device,
                )
                parsed = parse_judge_output(judge_raw)
                pdelta = pseudo_delta_from_judge(
                    parsed["choice"], float(parsed["confidence"])
                )

                if not parsed["parse_ok"]:
                    stats["parse_fail"] += 1
                if parsed["choice"] == "UNCERTAIN":
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
                    "judge_raw": judge_raw,
                    "judge_choice": parsed["choice"],
                    "judge_confidence": float(parsed["confidence"]),
                    "judge_parse_ok": bool(parsed["parse_ok"]),
                    "pseudo_delta": float(pdelta),
                    "actor_tokens": a_tok,
                    "actor_seconds": a_sec,
                    "judge_tokens": j_tok,
                    "judge_seconds": j_sec,
                    "schema_version": SCHEMA_VERSION,
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                csv_writer.writerow({k: rec.get(k, "") for k in CSV_FIELDS})
                stats["new_rows"] += 1
                rows_since_flush += 1
                if rows_since_flush >= cfg.flush_every:
                    fh.flush(); csv_fh.flush()
                    rows_since_flush = 0
        fh.flush(); csv_fh.flush()
    return stats


def _assemble_per_scenario(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group combined rows by scenario_id; drop scenarios missing any agent."""
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
        delta_base = float(by_role["base"].get("pseudo_delta", 0.0))
        delta_agents = [
            float(by_role[f"persona_{i}"].get("pseudo_delta", 0.0))
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


def _run_dpbr_for_country(
    country: str, scenario_inputs: List[Dict[str, Any]], cfg: UnifiedConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        p_right_van = pseudo_p_right_from_delta(
            rec["delta_base_deb"], t_decision=T_DECISION
        )
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
def run_unified(cfg: UnifiedConfig) -> None:
    """Generate, judge, score — one model, one pass, all countries."""
    if not cfg.countries:
        raise ValueError("UnifiedConfig.countries is empty")
    setup_seeds(cfg.seed)

    out_root = Path(cfg.out_dir)
    combined_dir = out_root / "combined"
    swa_root = out_root / "swa"
    cmp_root = out_root / "compare"
    for d in (combined_dir, swa_root, cmp_root):
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  OPEN-ENDED UNIFIED (SWA-DPBR) — model=[{cfg.model_name}]")
    print(f"{'='*70}")
    print(f"[CFG] out={out_root}  countries={cfg.countries}  n_scenarios={cfg.n_scenarios}")
    print(f"[CFG] actor_max={cfg.max_new_tokens_actor}  judge_max={cfg.max_new_tokens_judge}  "
          f"4bit={cfg.load_in_4bit}  seed={cfg.seed}")
    print(f"[CFG] EXP24_VAR_SCALE={os.environ.get('EXP24_VAR_SCALE', '0.04')}  "
          f"EXP24_K_HALF={os.environ.get('EXP24_K_HALF', '64')}  "
          f"EXP24_ESS_ANCHOR_REG={os.environ.get('EXP24_ESS_ANCHOR_REG', '1')}")

    model, tokenizer = load_model_hf_native(
        cfg.model_name, max_seq_length=4096, load_in_4bit=cfg.load_in_4bit,
    )
    helper = ChatTemplateHelper(tokenizer)
    device = next(model.parameters()).device

    compare_rows: List[Dict[str, Any]] = []
    try:
        for ci, country in enumerate(cfg.countries):
            if country not in SUPPORTED_COUNTRIES:
                print(f"[SKIP] unsupported country: {country}")
                continue

            combined_jsonl = combined_dir / f"{country}.jsonl"
            csv_path = combined_dir / f"{country}.csv"
            print(f"\n[{ci+1}/{len(cfg.countries)}] {country}  -> {combined_jsonl.name}")

            t0 = time.time()
            stats = _generate_and_judge_country(
                cfg, country, model, tokenizer, helper, device,
                combined_jsonl, csv_path,
            )
            dt = time.time() - t0
            n_total = max(1, stats["new_rows"])
            fail_pct = 100.0 * stats["parse_fail"] / n_total if stats["new_rows"] else 0.0
            print(f"  generated+judged new_rows={stats['new_rows']}  "
                  f"parse_fail={stats['parse_fail']} ({fail_pct:.1f}%)  "
                  f"uncertain={stats['uncertain']}  t={dt:.1f}s")

            rows = _load_combined_rows(combined_jsonl)
            if not rows:
                print(f"[SKIP] {country}: empty combined JSONL")
                continue

            # Recompute parse-fail % across ALL rows (including cached ones from
            # previous runs) so the abort check is stable across resumes.
            n_all = len(rows)
            all_fail = sum(1 for r in rows if not r.get("judge_parse_ok", False))
            all_fail_pct = 100.0 * all_fail / n_all
            if all_fail_pct >= cfg.max_parse_fail_pct:
                print(f"[ERROR] {country} parse-fail rate {all_fail_pct:.1f}% "
                      f">= {cfg.max_parse_fail_pct:.1f}% — skipping DPBR for this country")
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
                    "judge_stats": {
                        "new_rows": stats["new_rows"],
                        "parse_fail_new": stats["parse_fail"],
                        "uncertain_new": stats["uncertain"],
                        "parse_fail_pct_all": all_fail_pct,
                        "n_all_rows": n_all,
                    },
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            compare_rows.append({
                "model": cfg.model_label,
                "judge": cfg.model_name,
                "method": "baseline_vanilla_from_base",
                "country": country,
                **{f"align_{k}": v for k, v in van_summary.get("alignment", {}).items()},
                "n_scenarios": van_summary["n_scenarios"],
            })
            compare_rows.append({
                "model": cfg.model_label,
                "judge": cfg.model_name,
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
                "judge_parse_fail_pct": all_fail_pct,
            })

            swa_a = swa_summary.get("alignment", {})
            van_a = van_summary.get("alignment", {})
            swa_mis = swa_a.get("mis", float("nan"))
            van_mis = van_a.get("mis", float("nan"))
            mis_delta = van_mis - swa_mis if (
                not math.isnan(swa_mis) and not math.isnan(van_mis)
            ) else float("nan")
            print(
                f"  [OK] {country}  "
                f"SWA MIS={swa_mis:.4f}  r={swa_a.get('pearson_r', float('nan')):+.3f}  "
                f"JSD={swa_a.get('jsd', float('nan')):.4f}  |  "
                f"VAN MIS={van_mis:.4f}  r={van_a.get('pearson_r', float('nan')):+.3f}  "
                f"JSD={van_a.get('jsd', float('nan')):.4f}  |  "
                f"ΔMIS={mis_delta:+.4f}  n={swa_summary['n_scenarios']}  "
                f"parse_fail%={all_fail_pct:.1f}"
            )
            torch.cuda.empty_cache(); gc.collect()

    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cmp_df = pd.DataFrame(compare_rows)
    cmp_df.to_csv(cmp_root / "comparison.csv", index=False)
    print(f"\n[UNIFIED] DONE — comparison at {cmp_root/'comparison.csv'}")

    if not cmp_df.empty:
        print(f"\n{'='*70}")
        print(f"  FINAL SUMMARY — UNIFIED OPEN-ENDED SWA-DPBR  ({len(cmp_df)} rows)")
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
