#!/usr/bin/env python3
"""
Entry point: SWA-MPPI v3 for Cross-Cultural Moral Machine Experiment.

Runs Socially-Weighted Alignment via Model Predictive Path Integral (MPPI)
across 15 countries with WVS-based cultural personas.

Usage:
    python run_swa_mppi.py \
        --multitp-data-path ./data/multitp \
        --human-amce-path ./data/country_specific_ACME.csv \
        --wvs-data-path ./data/WVS_Wave7.csv \
        --output-dir results/swa_mppi

    python run_swa_mppi.py --use-synthetic-data --n-scenarios 50 --countries USA CHN JPN
"""

import os
import gc
import json
import textwrap
import argparse
import pickle
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import pandas as pd

from src.config import SWAConfig, add_common_args, add_swa_args, config_from_args, resolve_output_dir
from src.constants import COUNTRY_LANG
from src.model import setup_seeds, load_model
from src.data import load_multitp_dataset
from src.scenarios import generate_multitp_scenarios
from src.personas import build_country_personas, SUPPORTED_COUNTRIES
from src.swa_runner import run_country_experiment
from src.viz import (
    setup_matplotlib,
    plot_radar_single, plot_radar_grid,
    plot_amce_comparison_bar, plot_results_table,
    plot_cultural_clustering,
    plot_alignment_heatmap,
    plot_decision_gap_analysis,
    SWA_COLOR,
)


def print_final_statistics(all_summaries, config):
    """Print comprehensive SWA-MPPI statistics for paper."""
    n_countries = len(all_summaries)
    all_mis = [s["alignment"].get("mis", np.nan) for s in all_summaries]
    all_jsd = [s["alignment"].get("jsd", np.nan) for s in all_summaries]
    all_cosine = [s["alignment"].get("cosine_sim", np.nan) for s in all_summaries]
    all_pearson = [s["alignment"].get("pearson_r", np.nan) for s in all_summaries]
    all_spearman = [s["alignment"].get("spearman_rho", np.nan) for s in all_summaries]
    all_mae = [s["alignment"].get("mae", np.nan) for s in all_summaries]
    all_flip = [s["flip_rate"] for s in all_summaries]
    all_latency = [s["mean_latency_ms"] for s in all_summaries]

    print(f"\n{'='*70}")
    print(f"  SWA-MPPI v3 AGGREGATE RESULTS (N={n_countries} countries)")
    print(f"{'='*70}")
    print(f"  MIS (paper, L2):          {np.nanmean(all_mis):.4f} \u00b1 {np.nanstd(all_mis):.4f}   [0=perfect, \u221a6\u22482.45=worst]")
    print(f"  Jensen-Shannon Distance:  {np.nanmean(all_jsd):.4f} \u00b1 {np.nanstd(all_jsd):.4f}")
    print(f"  Cosine Similarity:        {np.nanmean(all_cosine):.4f} \u00b1 {np.nanstd(all_cosine):.4f}")
    print(f"  Pearson Correlation:      {np.nanmean(all_pearson):.4f} \u00b1 {np.nanstd(all_pearson):.4f}")
    print(f"  Spearman Correlation:     {np.nanmean(all_spearman):.4f} \u00b1 {np.nanstd(all_spearman):.4f}")
    print(f"  Mean Absolute Error:      {np.nanmean(all_mae):.2f} \u00b1 {np.nanstd(all_mae):.2f} pp")
    print(f"  MPPI Flip Rate:           {np.mean(all_flip):.1%} \u00b1 {np.std(all_flip):.1%} (of all scenarios)")
    print(f"  Mean Latency:             {np.mean(all_latency):.1f} \u00b1 {np.std(all_latency):.1f} ms")

    print(f"\n{'='*70}")
    print(f"  PER-COUNTRY RANKING (by JSD \u2193)")
    print(f"{'='*70}")
    ranked = sorted(zip([s["country"] for s in all_summaries], all_jsd), key=lambda x: x[1])
    for i, (country, jsd) in enumerate(ranked):
        marker = "\u2605" if i < 3 else " "
        print(f"  {marker} {i+1:2d}. {country:5s}  JSD={jsd:.4f}")

    print(f"\n{'='*70}")
    print(f"  CATEGORY-LEVEL BIAS SUMMARY (Model AMCE \u2212 Human AMCE)")
    print(f"{'='*70}")
    cats = ["Species_Humans", "Gender_Female", "Age_Young",
            "Fitness_Fit", "SocialValue_High", "Utilitarianism_More"]
    for cat in cats:
        m_vals = [s["model_amce"].get(cat, np.nan) for s in all_summaries]
        h_vals = [s["human_amce"].get(cat, np.nan) for s in all_summaries]
        diffs = [m - h for m, h in zip(m_vals, h_vals)
                 if not np.isnan(m) and not np.isnan(h)]
        if diffs:
            mean_d = np.mean(diffs)
            direction = "\u2191 OVER" if mean_d > 2 else ("\u2193 UNDER" if mean_d < -2 else "\u2248 OK")
            print(f"  {cat:25s}: {mean_d:+6.2f} pp  {direction}")

    total_scenarios = sum(s["n_scenarios"] for s in all_summaries)
    total_flips = sum(s["flip_count"] for s in all_summaries)
    print(f"\n{'='*70}")
    print(f"  MPPI IMPACT")
    print(f"{'='*70}")
    print(f"  Total scenarios:  {total_scenarios:,}")
    print(f"  MPPI flipped:     {total_flips:,} ({total_flips/max(1,total_scenarios):.1%} of all scenarios)")
    print(f"\n{'='*70}")
    print(f"  Experiment complete. All results in: {config.output_dir}/")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="SWA-MPPI v3: Cross-Cultural Value Negotiation via Implicit Pre-Logit Control",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser)
    add_swa_args(parser)
    args = parser.parse_args()
    config = config_from_args(args, SWAConfig)

    # Auto-separate outputs per model so multi-model runs don't overwrite
    config.output_dir = resolve_output_dir(config.output_dir, config.model_name)

    # Setup
    setup_matplotlib()
    setup_seeds(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)
    print(f"[OUTPUT] Results -> {config.output_dir}")

    # Dump config snapshot for reproducibility
    cfg_dict = {k: v for k, v in asdict(config).items()}
    cfg_dict["_run"] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "script": "run_swa_mppi.py",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    with open(os.path.join(config.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=2, default=str, ensure_ascii=False)
    print(f"[SAVE] Config snapshot -> config.json")

    # Collect personas across all countries (dumped after build below)
    all_personas: dict = {}

    print(f"[CONFIG] SWA-MPPI v3")
    print(f"  model_name:             {config.model_name}")
    print(f"  n_scenarios:            {config.n_scenarios}")
    print(f"  noise_std:              {config.noise_std}")
    print(f"  decision_temperature:   {config.decision_temperature}")
    print(f"  category_temperatures:  {config.category_logit_temperatures}")

    # Load human AMCE data
    amce_path = Path(config.human_amce_path)
    if not amce_path.exists():
        raise FileNotFoundError(f"Human AMCE file not found: {amce_path}")
    amce_df = pd.read_csv(amce_path)
    country_col = "Country" if "Country" in amce_df.columns else "ISO3"
    available_countries = amce_df[country_col].unique()
    _missing = [c for c in config.target_countries if c not in available_countries]
    if _missing:
        print(f"[WARN] Countries not in AMCE: {_missing}")
    print(f"[DATA] Loaded human AMCE from {amce_path} "
          f"({len(available_countries)} countries, {len(amce_df)} rows)")

    # Verify data source
    if config.use_real_data:
        print(f"\n[DATA] Will load REAL MultiTP dataset per-country (native language prompts)")
        if not os.path.isdir(config.multitp_data_path):
            raise FileNotFoundError(f"MultiTP data path not found: {config.multitp_data_path}")
    else:
        print(f"\n[DATA] Will generate synthetic scenarios per-country (native language)")

    # Load model
    model, tokenizer = load_model(
        config.model_name, config.max_seq_length, config.load_in_4bit,
    )

    # Per-country loop
    print("\n" + "=" * 70)
    print("RUNNING: SWA-MPPI v3 per country")
    print("=" * 70)

    all_results, all_summaries = [], []

    for ci, country in enumerate(config.target_countries):
        if country not in SUPPORTED_COUNTRIES:
            print(f"[SKIP] No personas for {country}")
            continue

        lang = COUNTRY_LANG.get(country, "en")
        print(f"\n{'='*70}")
        print(f"  [{ci+1}/{len(config.target_countries)}] {country} (lang={lang})")
        print(f"{'='*70}")

        # Load scenarios
        if config.use_real_data:
            country_base_df = load_multitp_dataset(
                data_base_path=config.multitp_data_path,
                lang=lang,
                translator=config.multitp_translator,
                suffix=config.multitp_suffix,
                n_scenarios=config.n_scenarios,
            )
        else:
            country_base_df = generate_multitp_scenarios(
                config.n_scenarios, lang=lang,
            )
        country_df = country_base_df.copy()
        country_df["lang"] = lang

        # Build personas
        personas = build_country_personas(country, wvs_path=config.wvs_data_path)
        print(f"\n  [PERSONAS] {country} ({len(personas)} personas):")
        for pi, ptxt in enumerate(personas):
            print(f"    P{pi+1}:")
            print(textwrap.indent(ptxt, prefix="      "))
            print()
        all_personas[country] = personas

        # Dump first 5 raw scenario prompts (reproducibility)
        sample_prompts_path = os.path.join(
            config.output_dir, f"prompts_sample_{country}.txt"
        )
        with open(sample_prompts_path, "w", encoding="utf-8") as f:
            f.write(f"# Sample prompts for {country} (lang={lang})\n\n")
            for i, (_, srow) in enumerate(country_df.head(5).iterrows()):
                f.write(f"--- Sample {i+1} [{srow.get('phenomenon_category','?')}] ---\n")
                f.write(str(srow.get("Prompt", srow.get("prompt", ""))) + "\n\n")

        # Run SWA-MPPI
        print(f"\n  SWA-MPPI v3 for {country}...")
        results_df, summary = run_country_experiment(
            model, tokenizer, country, personas, country_df, config,
        )
        all_results.append(results_df)
        all_summaries.append(summary)

        # Save per-country CSV
        results_df.to_csv(
            os.path.join(config.output_dir, f"swa_results_{country}.csv"),
            index=False,
        )

        # Save per-country MPPI diagnostics as flat CSV (Excel/R-friendly)
        diag = summary["diagnostics"]
        n_diag = len(diag["variances"])
        diag_df = pd.DataFrame({
            "scenario_idx":    list(range(n_diag)),
            "variance":        diag["variances"],
            "delta_z_norm":    diag["delta_z_norms"],
            "decision_gap":    diag["decision_gaps"],
            "logit_temp_used": diag["logit_temps_used"],
            "latency_s":       diag["latencies"],
        })
        # agent_reward_matrix: list of per-agent reward arrays -> spread into columns
        try:
            arm = np.asarray(diag["agent_reward_matrix"], dtype=float)
            if arm.ndim == 2:
                for ai in range(arm.shape[1]):
                    diag_df[f"agent{ai+1}_reward"] = arm[:, ai]
        except Exception:
            pass
        diag_df.to_csv(
            os.path.join(config.output_dir, f"swa_diagnostics_{country}.csv"),
            index=False,
        )

        # Plot per-country radar
        plot_radar_single(
            summary["model_amce"], summary["human_amce"],
            country, summary["alignment"],
            save_path=os.path.join(config.output_dir, f"radar_swa_{country}.png"),
        )

        swa_jsd = summary["alignment"].get("jsd", float("nan"))
        print(f"    JSD={swa_jsd:.4f}")

        torch.cuda.empty_cache()
        gc.collect()

    if not all_summaries or not all_results:
        print("\n[ERROR] No valid country results were produced. Exiting without saving/plotting.")
        return

    # Save combined results
    full_results = pd.concat(all_results, ignore_index=True)
    full_results.to_csv(
        os.path.join(config.output_dir, "swa_all_results.csv"), index=False,
    )
    print(f"[SAVE] SWA all results -> swa_all_results.csv ({len(full_results)} rows)")

    # AMCE summary
    amce_rows = []
    for s in all_summaries:
        row = {"country": s["country"]}
        for k, v in s["model_amce"].items():
            row[f"swa_{k}"] = v
        for k, v in s["human_amce"].items():
            row[f"human_{k}"] = v
        for k, v in s["alignment"].items():
            row[f"align_{k}"] = v
        amce_rows.append(row)
    amce_df_out = pd.DataFrame(amce_rows)
    amce_df_out.to_csv(
        os.path.join(config.output_dir, "swa_amce_summary.csv"), index=False,
    )
    print(f"[SAVE] AMCE summary -> swa_amce_summary.csv ({len(amce_df_out)} countries)")

    # Summary CSV
    summary_rows = []
    for s in all_summaries:
        row = {k: v for k, v in s.items()
               if k not in ("model_amce", "human_amce", "diagnostics")}
        row.update({f"alignment_{k}": v for k, v in s.get("alignment", {}).items()})
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(config.output_dir, "swa_summary.csv"), index=False,
    )

    # Pickle
    with open(os.path.join(config.output_dir, "all_summaries.pkl"), "wb") as f:
        pickle.dump(all_summaries, f)

    # Personas snapshot (for prompt-level reproducibility)
    with open(os.path.join(config.output_dir, "personas.json"), "w", encoding="utf-8") as f:
        json.dump(all_personas, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] Personas -> personas.json ({len(all_personas)} countries)")

    print(f"\n[ALL COUNTRIES COMPLETE] {len(all_summaries)} countries evaluated.")

    # Generate figures
    print("\n[PLOT] Fig 1: Radar grid -- SWA-MPPI vs Human...")
    plot_radar_grid(
        all_summaries, config.output_dir,
        amce_key="model_amce", alignment_key="alignment",
        title_suffix="", file_suffix="_swa",
        fig_title="SWA-MPPI v3 vs Human Preferences (15 Countries)",
    )

    print("\n[PLOT] Fig 2: Alignment heatmap...")
    plot_alignment_heatmap(all_summaries, config.output_dir)

    print("\n[PLOT] Fig 5: Decision gap analysis...")
    plot_decision_gap_analysis(all_summaries, config, config.output_dir)

    print("\n[PLOT] Fig 6: Results table...")
    plot_results_table(all_summaries, config.output_dir, mode="swa")

    print("\n[PLOT] Fig 7: Cultural clustering...")
    plot_cultural_clustering(all_summaries, config.output_dir)

    print("\n[PLOT] AMCE per-criterion bar chart...")
    plot_amce_comparison_bar(all_summaries, config.output_dir)

    # Print statistics
    print_final_statistics(all_summaries, config)

    print(f"\n{'='*70}")
    print(f"ALL FIGURES SAVED TO: {config.output_dir}/")
    print(f"{'='*70}")
    for f_path in sorted(Path(config.output_dir).glob("*")):
        size_kb = f_path.stat().st_size / 1024
        print(f"  {f_path.name:45s} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
