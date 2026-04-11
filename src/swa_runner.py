"""SWA-PTIS experiment runner for per-country evaluation."""

import gc
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import pandas as pd
from tqdm.auto import tqdm

from src.config import SWAConfig
from src.constants import COUNTRY_LANG
from src.i18n import PROMPT_FRAME_I18N
from src.controller import ImplicitSWAController
from src.amce import (
    compute_amce_from_preferences,
    compute_per_dimension_alignment,
    compute_utilitarianism_slope,
    load_human_amce,
    compute_alignment_metrics,
)


def run_country_experiment(
    model,
    tokenizer,
    country_iso: str,
    personas: List[str],
    scenario_df: pd.DataFrame,
    cfg: SWAConfig,
) -> Tuple[pd.DataFrame, Dict]:
    lang = COUNTRY_LANG.get(country_iso, "en")
    print(f"\n{'='*60}")
    print(f"[EXPERIMENT] Country: {country_iso} | Lang: {lang} | Agents: {len(personas)}")
    print(f"{'='*60}")

    scenario_df = scenario_df.copy()
    if "lang" not in scenario_df.columns:
        scenario_df["lang"] = lang

    controller = ImplicitSWAController(
        model=model,
        tokenizer=tokenizer,
        personas=personas,
        lambda_coop=cfg.lambda_coop,
        alpha_ctl=cfg.alpha_ctl,
        K_samples=cfg.K_samples,
        noise_std=cfg.noise_std,
        temperature=cfg.temperature,
        logit_temperature=cfg.logit_temperature,
        category_logit_temperatures=cfg.category_logit_temperatures,
        pt_alpha=cfg.pt_alpha,
        pt_beta=cfg.pt_beta,
        pt_kappa=cfg.pt_kappa,
        decision_temperature=cfg.decision_temperature,
        assistant_lang=lang,
        country_iso=country_iso,
    )

    # Debug: print 3 sample prompts with model prediction (logit extraction)
    frame = PROMPT_FRAME_I18N.get(lang, PROMPT_FRAME_I18N["en"])
    sample_rows = scenario_df.head(3)
    print(f"\n[DEBUG] 3 sample prompts for {country_iso} (lang={lang}):")
    for si, (_, srow) in enumerate(sample_rows.iterrows()):
        sp = srow.get("Prompt", srow.get("prompt", ""))
        cat = srow.get("phenomenon_category", "?")
        pref_right = bool(srow.get("preferred_on_right", 1))
        pref_side = "B" if pref_right else "A"
        formatted_sample = frame.format(scenario=sp)
        # Run quick prediction to show what model outputs
        debug_pred = controller.predict(
            sp, preferred_on_right=pref_right,
            phenomenon_category=cat, lang=lang,
        )
        model_choice = "B" if debug_pred["p_right"] > 0.5 else "A"
        print(f"  ── Sample {si+1} [{cat}] (preferred={pref_side}) ──")
        print(f"  {formatted_sample[:500]}{'...' if len(formatted_sample) > 500 else ''}")
        print(f"  >>> Model: p(B)={debug_pred['p_right']:.3f}  p(A)={debug_pred['p_left']:.3f}"
              f"  -> {model_choice}  |  p(spare_preferred)={debug_pred['p_spare_preferred']:.3f}")
        print()

    results = []
    diagnostics = {
        "variances": [], "flip_count": 0, "total_count": 0,
        "delta_z_norms": [], "agent_reward_matrix": [],
        "latencies": [], "decision_gaps": [],
        "logit_temps_used": [],
    }

    for idx, row in tqdm(scenario_df.iterrows(), total=len(scenario_df),
                          desc=f"SWA-v3 [{country_iso}]"):
        prompt = row.get("Prompt", row.get("prompt", ""))
        if not prompt:
            continue

        phenomenon_cat = row.get("phenomenon_category", "default")
        # Raw scenario text; predict() applies native-language framing internally
        preferred_on_right = bool(row.get("preferred_on_right", 1))

        t0 = time.time()
        pred = controller.predict(
            prompt,                        # raw scenario text (already native-language for synth)
            preferred_on_right=preferred_on_right,
            phenomenon_category=phenomenon_cat,
            lang=lang,
        )
        latency = time.time() - t0

        diagnostics["variances"].append(pred["variance"])
        diagnostics["flip_count"] += int(pred["mppi_flipped"])
        diagnostics["total_count"] += 1
        diagnostics["delta_z_norms"].append(pred["delta_z_norm"])
        diagnostics["agent_reward_matrix"].append(pred["agent_rewards"])
        diagnostics["latencies"].append(latency)
        diagnostics["decision_gaps"].append(pred["delta_consensus"])
        diagnostics["logit_temps_used"].append(pred["logit_temp_used"])

        agent_rewards_arr = np.asarray(pred["agent_rewards"], dtype=float)
        row_out = {
            "country": country_iso,
            "scenario_idx": idx,
            "Prompt": prompt,
            "phenomenon_category": phenomenon_cat,
            "this_group_name": row.get("this_group_name", "Unknown"),
            "preferred_on_right": int(preferred_on_right),
            "n_left": int(row.get("n_left", 1)),
            "n_right": int(row.get("n_right", 1)),
            "p_left": float(pred.get("p_left", 1.0 - pred["p_right"])),
            "p_right": float(pred["p_right"]),
            "p_spare_preferred": float(pred["p_spare_preferred"]),
            "mppi_variance": float(pred["variance"]),
            "mppi_flipped": bool(pred["mppi_flipped"]),
            "delta_z_norm": float(pred["delta_z_norm"]),
            "delta_consensus": float(pred["delta_consensus"]),
            "logit_temp_used": float(pred["logit_temp_used"]),
            "agent_reward_mean": float(agent_rewards_arr.mean()) if agent_rewards_arr.size else 0.0,
            "agent_reward_min":  float(agent_rewards_arr.min())  if agent_rewards_arr.size else 0.0,
            "agent_reward_max":  float(agent_rewards_arr.max())  if agent_rewards_arr.size else 0.0,
            "agent_reward_std":  float(agent_rewards_arr.std())  if agent_rewards_arr.size else 0.0,
            "latency_ms": latency * 1000,
        }
        # Optional controller diagnostics (EXP-24 dual-pass, etc.) — forward to results_df means.
        for _k in (
            "reliability_r", "bootstrap_var", "ess_pass1", "ess_pass2",
            "delta_star_1", "delta_star_2", "delta_opt_micro",
            "delta_country", "alpha_h", "prior_step",
            "p_spare_preferred_is_pass1_micro", "p_spare_preferred_is_pass2_micro",
        ):
            if _k not in pred:
                continue
            _v = pred[_k]
            if hasattr(_v, "item"):
                _v = _v.item()
            try:
                row_out[_k] = float(_v)
            except (TypeError, ValueError):
                row_out[_k] = _v
        results.append(row_out)

    results_df = pd.DataFrame(results)

    # Primary metric: mean preference rate per category (MultiTP convention).
    model_amce = compute_amce_from_preferences(results_df)

    # Supplementary diagnostic: OLS slope on the Utilitarianism scenarios.
    # This is NOT used in the JSD/MIS metrics (they need MPRs to match the
    # human ground-truth format), but it isolates the slope from the MPR
    # intercept confound discussed in the paper.
    util_slope = compute_utilitarianism_slope(results_df)

    human_amce = load_human_amce(cfg.human_amce_path, country_iso)
    alignment = compute_alignment_metrics(model_amce, human_amce)

    # Per-dimension breakdown (paper: "largest analytical gap"). This is a
    # post-hoc decomposition of the aggregate JSD/MIS/MAE into per-dimension
    # errors so readers can see *where* the alignment gain comes from.
    per_dim = compute_per_dimension_alignment(model_amce, human_amce)

    summary = {
        "country": country_iso,
        "n_scenarios": diagnostics["total_count"],
        "flip_rate": diagnostics["flip_count"] / max(1, diagnostics["total_count"]),
        "flip_count": diagnostics["flip_count"],
        "mean_variance": np.mean(diagnostics["variances"]),
        "mean_delta_z_norm": np.mean(diagnostics["delta_z_norms"]),
        "mean_latency_ms": np.mean(diagnostics["latencies"]) * 1000,
        "median_latency_ms": np.median(diagnostics["latencies"]) * 1000,
        "mean_decision_gap": np.mean(diagnostics["decision_gaps"]),
        "model_amce": model_amce,
        "human_amce": human_amce,
        "alignment": alignment,
        "per_dimension_alignment": per_dim,
        "utilitarianism_slope": util_slope,
        "diagnostics": diagnostics,
    }

    print(f"\n[RESULT] {country_iso}:")
    print(f"  Flip rate:         {summary['flip_count']}/{diagnostics['total_count']} ({summary['flip_rate']:.1%} of all scenarios)")
    print(f"  Mean variance:     {summary['mean_variance']:.6f}")
    print(f"  Mean decision gap: {summary['mean_decision_gap']:.4f}")
    print(f"  Mean latency:      {summary['mean_latency_ms']:.1f} ms")
    if "mis" in alignment:
        _d = int(alignment.get("n_criteria", 6) or 6)
        _worst = float(np.sqrt(_d))
        print(f"  MIS (L2, paper):   {alignment['mis']:.4f}   "
              f"[0=perfect, worst≤√d≈{_worst:.2f} for d={_d} matched dims]")
    if "jsd" in alignment:
        print(f"  JSD vs Human:      {alignment['jsd']:.4f}")
        print(f"  Pearson r:         {alignment['pearson_r']:.4f} (p={alignment['pearson_p']:.4f})")
        print(f"  Cosine sim:        {alignment['cosine_sim']:.4f}")
        print(f"  MAE:               {alignment['mae']:.2f}")
    print(f"  Model MPR : { {k: f'{v:.1f}' for k, v in model_amce.items()} }")
    print(f"  Human MPR : { {k: f'{v:.1f}' for k, v in human_amce.items()} }")
    if per_dim:
        print("  Per-dim |err| (|model-human|, sorted):")
        for k, d in sorted(per_dim.items(), key=lambda kv: -kv[1]["abs_err"]):
            print(f"    {k:<28s}  human={d['human']:5.1f}  "
                  f"model={d['model']:5.1f}  |err|={d['abs_err']:4.1f}")
    if util_slope.get("n_obs", 0) >= 3:
        print(f"  Util slope: b_hat={util_slope['slope_hat']:+.4f} "
              f"(se={util_slope['slope_se']:.4f}, "
              f"intercept a_hat={util_slope['intercept_hat']:.3f}, "
              f"n={util_slope['n_obs']})  [diagnostic, NOT in JSD]")

    del controller
    torch.cuda.empty_cache()
    gc.collect()

    return results_df, summary
