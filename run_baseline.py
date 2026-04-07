#!/usr/bin/env python3
"""
Entry point: Vanilla LLM Baseline for Cross-Cultural Moral Machine Experiment.

Runs token-logit extraction baseline across 15 countries and compares
model moral preferences against human AMCE data from MultiTP.

Usage:
    python run_baseline.py \
        --multitp-data-path ./data/multitp \
        --human-amce-path ./data/country_specific_ACME.csv \
        --output-dir results/baseline

    python run_baseline.py --use-synthetic-data --n-scenarios 50 --countries USA CHN JPN
"""

import os
import gc
import json
import argparse
import pickle
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import pandas as pd

from src.config import BaselineConfig, add_common_args, config_from_args, resolve_output_dir
from src.constants import COUNTRY_LANG
from src.model import setup_seeds, load_model
from src.data import load_multitp_dataset, balance_scenario_dataset
from src.scenarios import generate_multitp_scenarios
from src.baseline_runner import run_baseline_vanilla
from src.viz import (
    setup_matplotlib,
    plot_radar_single, plot_radar_grid,
    plot_amce_comparison_bar, plot_results_table,
    plot_cultural_clustering,
    BASELINE_COLOR,
)


def print_baseline_statistics(all_summaries, config):
    """Print comprehensive baseline statistics for paper."""
    n_countries = len(all_summaries)
    all_jsd = [s["alignment"].get("jsd", np.nan) for s in all_summaries]
    all_cosine = [s["alignment"].get("cosine_sim", np.nan) for s in all_summaries]
    all_pearson = [s["alignment"].get("pearson_r", np.nan) for s in all_summaries]
    all_spearman = [s["alignment"].get("spearman_rho", np.nan) for s in all_summaries]
    all_mae = [s["alignment"].get("mae", np.nan) for s in all_summaries]
    all_rmse = [s["alignment"].get("rmse", np.nan) for s in all_summaries]

    print(f"\n{'='*70}")
    print(f"  VANILLA LLM BASELINE AGGREGATE RESULTS (N={n_countries} countries)")
    print(f"{'='*70}")
    print(f"  Jensen-Shannon Distance:  {np.nanmean(all_jsd):.4f} \u00b1 {np.nanstd(all_jsd):.4f}")
    print(f"  Cosine Similarity:        {np.nanmean(all_cosine):.4f} \u00b1 {np.nanstd(all_cosine):.4f}")
    print(f"  Pearson Correlation:      {np.nanmean(all_pearson):.4f} \u00b1 {np.nanstd(all_pearson):.4f}")
    print(f"  Spearman Correlation:     {np.nanmean(all_spearman):.4f} \u00b1 {np.nanstd(all_spearman):.4f}")
    print(f"  Mean Absolute Error:      {np.nanmean(all_mae):.2f} \u00b1 {np.nanstd(all_mae):.2f} pp")
    print(f"  RMSE:                     {np.nanmean(all_rmse):.2f} \u00b1 {np.nanstd(all_rmse):.2f} pp")

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
    print(f"\n{'='*70}")
    print(f"  TOTAL SCENARIOS: {total_scenarios:,}")
    print(f"  Experiment complete. All results in: {config.output_dir}/")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Vanilla LLM Baseline for Cross-Cultural Moral Machine Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser)
    args = parser.parse_args()
    config = config_from_args(args, BaselineConfig)

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
        "script": "run_baseline.py",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    with open(os.path.join(config.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=2, default=str, ensure_ascii=False)
    print(f"[SAVE] Config snapshot -> config.json")

    print(f"[CONFIG] Vanilla LLM Baseline")
    print(f"  model_name:             {config.model_name}")
    print(f"  n_scenarios:            {config.n_scenarios}")
    print(f"  decision_temperature:   {config.decision_temperature}")

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
    print("RUNNING: Vanilla LLM Baseline per country")
    print("=" * 70)

    all_summaries = []
    all_vanilla_results = []

    for ci, country in enumerate(config.target_countries):
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
        country_df = balance_scenario_dataset(
            country_base_df, min_per_category=50, seed=config.seed, lang=lang,
        )
        country_df["lang"] = lang

        # Dump first 3 raw scenario prompts (reproducibility)
        sample_prompts_path = os.path.join(
            config.output_dir, f"prompts_sample_{country}.txt"
        )
        with open(sample_prompts_path, "w", encoding="utf-8") as f:
            f.write(f"# Sample prompts for {country} (lang={lang})\n\n")
            for i, (_, srow) in enumerate(country_df.head(5).iterrows()):
                f.write(f"--- Sample {i+1} [{srow.get('phenomenon_category','?')}] ---\n")
                f.write(str(srow.get("Prompt", srow.get("prompt", ""))) + "\n\n")

        # Run baseline
        print(f"\n  Vanilla LLM baseline for {country}...")
        bl = run_baseline_vanilla(model, tokenizer, country_df, country, config)

        # Save per-country CSV
        bl["results_df"].to_csv(
            os.path.join(config.output_dir, f"vanilla_results_{country}.csv"),
            index=False,
        )
        all_vanilla_results.append(bl["results_df"])

        # Plot per-country radar
        plot_radar_single(
            bl["model_amce"], bl["human_amce"],
            country, bl["alignment"],
            save_path=os.path.join(config.output_dir, f"radar_baseline_{country}.png"),
            model_label="Vanilla LLM", model_color=BASELINE_COLOR,
        )

        # Build summary dict
        summary = {
            "country": country,
            "n_scenarios": len(bl["results_df"]),
            "model_amce": bl["model_amce"],
            "human_amce": bl["human_amce"],
            "alignment": bl["alignment"],
        }
        all_summaries.append(summary)

        bl_jsd = bl["alignment"].get("jsd", float("nan"))
        print(f"    JSD={bl_jsd:.4f}")
        print(f"    Model AMCE: { {k: f'{v:.1f}' for k, v in bl['model_amce'].items()} }")
        print(f"    Human AMCE: { {k: f'{v:.1f}' for k, v in bl['human_amce'].items()} }")

        torch.cuda.empty_cache()
        gc.collect()

    if not all_summaries or not all_vanilla_results:
        print("\n[ERROR] No valid country results were produced. Exiting without saving/plotting.")
        return

    # Save combined results
    full_vanilla = pd.concat(all_vanilla_results, ignore_index=True)
    full_vanilla.to_csv(
        os.path.join(config.output_dir, "vanilla_all_results.csv"), index=False,
    )
    print(f"[SAVE] Vanilla all results -> vanilla_all_results.csv ({len(full_vanilla)} rows)")

    # AMCE summary
    amce_rows = []
    for s in all_summaries:
        row = {"country": s["country"]}
        for k, v in s["model_amce"].items():
            row[f"vanilla_{k}"] = v
        for k, v in s["human_amce"].items():
            row[f"human_{k}"] = v
        for k, v in s["alignment"].items():
            row[f"align_{k}"] = v
        amce_rows.append(row)
    amce_df_out = pd.DataFrame(amce_rows)
    amce_df_out.to_csv(
        os.path.join(config.output_dir, "baseline_amce_summary.csv"), index=False,
    )
    print(f"[SAVE] AMCE summary -> baseline_amce_summary.csv ({len(amce_df_out)} countries)")

    # Save pickle
    with open(os.path.join(config.output_dir, "baseline_summaries.pkl"), "wb") as f:
        pickle.dump(all_summaries, f)

    print(f"\n[ALL COUNTRIES COMPLETE] {len(all_summaries)} countries evaluated.")

    # Generate figures
    print("\n[PLOT] Fig 1: Radar grid -- Vanilla LLM vs Human...")
    plot_radar_grid(
        all_summaries, config.output_dir,
        amce_key="model_amce", alignment_key="alignment",
        title_suffix="", file_suffix="_baseline",
        model_label="Vanilla LLM", model_color=BASELINE_COLOR,
        fig_title="Vanilla LLM vs Human Preferences (15 Countries)",
    )

    print("\n[PLOT] AMCE per-criterion bar chart...")
    plot_amce_comparison_bar(all_summaries, config.output_dir)

    print("\n[PLOT] Results table...")
    plot_results_table(all_summaries, config.output_dir, mode="baseline")

    print("\n[PLOT] Cultural clustering...")
    plot_cultural_clustering(all_summaries, config.output_dir)

    # Print statistics
    print_baseline_statistics(all_summaries, config)

    print(f"\n{'='*70}")
    print(f"ALL FIGURES SAVED TO: {config.output_dir}/")
    print(f"{'='*70}")
    for f_path in sorted(Path(config.output_dir).glob("*")):
        size_kb = f_path.stat().st_size / 1024
        print(f"  {f_path.name:45s} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
