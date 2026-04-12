#!/usr/bin/env python3
"""
Hyperparameter tuning + full evaluation for SWA-MPPI MC.

Target params:
  1. noise_std      [0.005 → 0.5]
  2. lambda_coop    [0.0   → 0.9]
  3. temperature    [0.01  → 1.0]

Pipeline:
  Phase 1 — Random search: 200 configs on 50 questions (UK)     ~2.3h
  Phase 2 — Fine grid:     27 configs on 100 questions (UK)     ~35 min
  Phase 3 — Full eval:     best config, 5 countries × 500q      ~33 min
  Total: ~3.5 hours (fits in 12h Kaggle)

Random search > grid search for 3+ dims (Bergstra & Bengio 2012):
  covers more unique values per axis with same compute budget.
"""

import sys
import os
import gc
import subprocess

_ON_KAGGLE = os.path.exists("/kaggle/working")
if _ON_KAGGLE:
    subprocess.run("pip install -q bitsandbytes scipy tqdm", shell=True,
                    capture_output=True)
    subprocess.run("pip install --upgrade --no-deps unsloth", shell=True,
                    capture_output=True)
    subprocess.run("pip install -q unsloth_zoo", shell=True, capture_output=True)
    subprocess.run("pip install --quiet --no-deps --force-reinstall pyarrow",
                    shell=True, capture_output=True)
    subprocess.run("pip install --quiet 'datasets>=3.4.1,<4.4.0'", shell=True,
                    capture_output=True)

try:
    import unsloth  # noqa: F401
except Exception:
    pass

import time
import itertools
import json
import csv
from pathlib import Path
from dataclasses import replace

import torch
import numpy as np
import pandas as pd

from swa_mppi_mc import (
    CulturalMCConfig,
    CulturalMCController,
    build_mc_personas,
    load_mc_questions,
    load_model,
    wilson_ci,
    write_csv_row,
    run_country_experiment,
    print_mc_summary,
    MC_DATA_DIR,
    MC_FILES,
    MC_COUNTRIES,
    MODEL_NAME,
    SAMPLE_SEED,
    WORK_DIR,
    RESULTS_DIR,
    EVAL_DIR,
)


# ============================================================================
# CONFIG
# ============================================================================
TUNE_COUNTRY = "UK"
TUNE_SEED = 42

TUNE_DIR = WORK_DIR / "tuning"
TUNE_DIR.mkdir(parents=True, exist_ok=True)
TUNE_LOG = TUNE_DIR / "tune_results.csv"

# ============================================================================
# SEARCH RANGES — full range, sampled randomly
# ============================================================================
PARAM_RANGES = {
    #                  (low,   high,  scale)
    "noise_std":       (0.005, 0.5,   "log"),   # log-uniform: more density at low end
    "lambda_coop":     (0.0,   0.9,   "linear"),
    "temperature":     (0.01,  1.0,   "log"),   # log-uniform: more density at low end
}

PHASE1_N_SAMPLES = 200    # random configs to try
PHASE1_N_QUESTIONS = 50

PHASE2_N_QUESTIONS = 100


def _sample_random_configs(n: int, seed: int = 42) -> list:
    """Sample n random configs from param ranges."""
    rng = np.random.RandomState(seed)
    configs = []
    for _ in range(n):
        params = {}
        for name, (lo, hi, scale) in PARAM_RANGES.items():
            if scale == "log":
                # Log-uniform: more samples at lower values
                val = np.exp(rng.uniform(np.log(lo), np.log(hi)))
            else:
                val = rng.uniform(lo, hi)
            # Round to 4 decimals for readability
            params[name] = round(float(val), 4)
        configs.append(params)
    return configs


def _fine_grid_around(best: dict) -> dict:
    """Phase 2: fine grid ±small step around Phase 1 winner."""
    ns = best["noise_std"]
    lc = best["lambda_coop"]
    t = best["temperature"]

    # Steps proportional to value (finer at small values)
    ns_step = max(0.002, ns * 0.3)
    t_step = max(0.005, t * 0.3)
    lc_step = 0.05

    return {
        "noise_std": sorted(set([
            round(max(0.001, ns - ns_step), 4),
            round(ns, 4),
            round(ns + ns_step, 4),
        ])),
        "lambda_coop": sorted(set([
            round(max(0.0, lc - lc_step), 4),
            round(lc, 4),
            round(min(0.95, lc + lc_step), 4),
        ])),
        "temperature": sorted(set([
            round(max(0.005, t - t_step), 4),
            round(t, 4),
            round(t + t_step, 4),
        ])),
    }


# ============================================================================
# EVALUATE
# ============================================================================
def evaluate_config(model, tokenizer, mc_df, config, personas, config_name=""):
    controller = CulturalMCController(
        model=model, tokenizer=tokenizer,
        personas=personas, config=config,
    )
    n_correct = 0
    n_total = 0
    t0 = time.time()

    for _, row in mc_df.iterrows():
        result = controller.predict_mc(row["prompt"], row["choices"])
        if result["predicted_answer"] == str(row["answer_idx"]).strip():
            n_correct += 1
        n_total += 1

    elapsed = time.time() - t0
    accuracy = n_correct / n_total * 100 if n_total > 0 else 0.0
    ci_lo, ci_hi = wilson_ci(n_correct, n_total)

    print(f"  [{config_name:>25s}] "
          f"acc={accuracy:5.1f}% [{ci_lo:.1f}-{ci_hi:.1f}] "
          f"t={elapsed:.0f}s ({elapsed/max(n_total,1):.2f}s/q)")

    return {
        "config_name": config_name,
        "accuracy": accuracy, "ci_lower": ci_lo, "ci_upper": ci_hi,
        "n_correct": n_correct, "n_total": n_total,
        "elapsed_sec": elapsed, "sec_per_q": elapsed / max(n_total, 1),
    }


def _log_result(result: dict):
    write_header = not TUNE_LOG.exists()
    with open(TUNE_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(result)


# ============================================================================
# PHASE 1: Random search
# ============================================================================
def run_random_search(model, tokenizer, mc_df, personas, base_config):
    """Phase 1: random search over full param ranges."""
    configs = _sample_random_configs(PHASE1_N_SAMPLES, seed=TUNE_SEED)

    print(f"\n{'='*60}")
    print(f"  Phase 1: Random search")
    print(f"  {PHASE1_N_SAMPLES} configs × {len(mc_df)} questions")
    est_min = PHASE1_N_SAMPLES * len(mc_df) * 0.8 / 60
    print(f"  Estimated: ~{est_min:.0f} min")
    print(f"  Ranges:")
    for name, (lo, hi, scale) in PARAM_RANGES.items():
        print(f"    {name}: [{lo}, {hi}] ({scale})")
    print(f"{'='*60}")

    results = []
    for i, params in enumerate(configs):
        label = (f"{i+1:>3d}/{PHASE1_N_SAMPLES} "
                 f"ns={params['noise_std']:.4f}|"
                 f"lc={params['lambda_coop']:.2f}|"
                 f"t={params['temperature']:.4f}")

        config = replace(base_config, **params)
        r = evaluate_config(model, tokenizer, mc_df, config, personas, label)
        r.update(params)
        r["phase"] = "phase1_random"
        results.append(r)
        _log_result(r)

        # Print running best every 20 configs
        if (i + 1) % 20 == 0:
            best_so_far = max(results, key=lambda x: x["accuracy"])
            print(f"  --- Best so far ({i+1}/{PHASE1_N_SAMPLES}): "
                  f"{best_so_far['accuracy']:.1f}% "
                  f"ns={best_so_far['noise_std']} "
                  f"lc={best_so_far['lambda_coop']} "
                  f"t={best_so_far['temperature']} ---")

    results.sort(key=lambda x: (-x["accuracy"], x["elapsed_sec"]))
    return results


# ============================================================================
# PHASE 2: Fine grid
# ============================================================================
def run_fine_grid(model, tokenizer, mc_df, personas, base_config, grid):
    param_names = sorted(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in param_names)))

    print(f"\n{'='*60}")
    print(f"  Phase 2: Fine grid")
    print(f"  {len(combos)} configs × {len(mc_df)} questions")
    est_min = len(combos) * len(mc_df) * 0.8 / 60
    print(f"  Estimated: ~{est_min:.0f} min")
    print(f"  Grid: {grid}")
    print(f"{'='*60}")

    results = []
    for i, combo in enumerate(combos):
        params = dict(zip(param_names, combo))
        label = (f"ns={params['noise_std']}|"
                 f"lc={params['lambda_coop']}|"
                 f"t={params['temperature']}")

        config = replace(base_config, **params)
        r = evaluate_config(model, tokenizer, mc_df, config, personas, label)
        r.update(params)
        r["phase"] = "phase2_fine"
        results.append(r)
        _log_result(r)

    results.sort(key=lambda x: (-x["accuracy"], x["elapsed_sec"]))
    return results


# ============================================================================
# PHASE 3: Full evaluation
# ============================================================================
def run_full_evaluation(model, tokenizer, best_params: dict):
    print(f"\n{'='*70}")
    print(f"  PHASE 3: Full Evaluation with Tuned Hyperparameters")
    print(f"{'='*70}")
    print(f"  noise_std:   {best_params['noise_std']}")
    print(f"  lambda_coop: {best_params['lambda_coop']}")
    print(f"  temperature: {best_params['temperature']}")
    print(f"  Countries:   {MC_COUNTRIES}")
    print(f"  Q/country:   500")
    print(f"{'='*70}")

    config = CulturalMCConfig(
        noise_std=best_params["noise_std"],
        lambda_coop=best_params["lambda_coop"],
        temperature=best_params["temperature"],
    )

    # Clean old per-country result CSVs to avoid resume mixing old+new configs
    for country in MC_COUNTRIES:
        old_csv = RESULTS_DIR / f"swa_mppi_mc_{country}_results.csv"
        if old_csv.exists():
            old_csv.unlink()
            print(f"  Removed old results: {old_csv.name}")

    print("\n  Loading full MC dataset...")
    available_files = [f for f in MC_FILES
                       if os.path.exists(os.path.join(MC_DATA_DIR, f))]
    mc_df = load_mc_questions(
        mc_dir=MC_DATA_DIR,
        mc_files=available_files,
        countries=MC_COUNTRIES,
        n_per_country=config.n_per_country,
        seed=SAMPLE_SEED,
    )

    eval_result_file = str(EVAL_DIR / "swa_mppi_mc_tuned_evaluation.csv")
    # Overwrite (not append) to avoid mixing with previous tuning runs
    with open(eval_result_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["model", "method", "country", "accuracy", "ci_lower", "ci_upper",
             "n_total", "n_correct", "mppi_trigger_rate", "mppi_flip_rate",
             "tau_calibrated",
             "noise_std", "lambda_coop", "temperature"]
        )

    results = []
    t0_full = time.time()

    for i, country in enumerate(MC_COUNTRIES):
        print(f"\n  [{i+1}/{len(MC_COUNTRIES)}] {country}...")
        result = run_country_experiment(
            model, tokenizer, country, mc_df, config)
        results.append(result)

        write_csv_row(
            [MODEL_NAME, "SWA-MPPI-tuned", result["country"],
             result["accuracy"], result["ci_lower"], result["ci_upper"],
             result["n_total"], result["n_correct"],
             result["mppi_trigger_rate"], result["mppi_flip_rate"],
             result.get("tau_calibrated", 0),
             best_params["noise_std"], best_params["lambda_coop"],
             best_params["temperature"]],
            eval_result_file,
        )

        gc.collect()
        torch.cuda.empty_cache()

    elapsed_full = time.time() - t0_full
    print_mc_summary(results, config)
    print(f"\n  Full evaluation time: {elapsed_full/60:.1f} min")
    print(f"  Results saved to: {eval_result_file}")
    return results


# ============================================================================
# MAIN
# ============================================================================
def main():
    base_config = CulturalMCConfig()
    t0_total = time.time()

    print("=" * 70)
    print("SWA-MPPI Hyperparameter Tuning + Full Evaluation")
    print("=" * 70)
    print(f"  Phase 1: Random search {PHASE1_N_SAMPLES} configs × {PHASE1_N_QUESTIONS}q")
    print(f"  Phase 2: Fine grid ~27 configs × {PHASE2_N_QUESTIONS}q")
    print(f"  Phase 3: Full eval 5 countries × 500q")
    est_total = (PHASE1_N_SAMPLES * PHASE1_N_QUESTIONS * 0.8
                 + 27 * PHASE2_N_QUESTIONS * 0.8
                 + 2500 * 0.8) / 3600
    print(f"  Estimated total: ~{est_total:.1f} hours")
    print("=" * 70)

    # ---- Load tune data ----
    print("\n[1/6] Loading tune data...")
    available_files = [f for f in MC_FILES
                       if os.path.exists(os.path.join(MC_DATA_DIR, f))]
    full_val_df = load_mc_questions(
        mc_dir=MC_DATA_DIR,
        mc_files=available_files,
        countries=[TUNE_COUNTRY],
        n_per_country=PHASE2_N_QUESTIONS,
        seed=TUNE_SEED,
    )
    phase1_df = full_val_df.head(PHASE1_N_QUESTIONS)
    phase2_df = full_val_df

    # ---- Load model ONCE ----
    print("\n[2/6] Loading model...")
    model, tokenizer = load_model(base_config.model_name)

    # ---- Build personas ONCE ----
    print("\n[3/6] Building personas...")
    personas = build_mc_personas(TUNE_COUNTRY, base_config)

    # ---- Phase 1: Random search ----
    print("\n[4/6] Phase 1: Random search...")
    p1_results = run_random_search(
        model, tokenizer, phase1_df, personas, base_config)

    print(f"\n  Top 10 Phase 1:")
    for i, r in enumerate(p1_results[:10]):
        print(f"    #{i+1} acc={r['accuracy']:.1f}%  "
              f"ns={r['noise_std']:.4f} lc={r['lambda_coop']:.2f} "
              f"t={r['temperature']:.4f}")

    best_p1 = {
        "noise_std": p1_results[0]["noise_std"],
        "lambda_coop": p1_results[0]["lambda_coop"],
        "temperature": p1_results[0]["temperature"],
    }
    print(f"\n  Phase 1 best: {best_p1}")

    # ---- Phase 2: Fine grid ----
    print("\n[5/6] Phase 2: Fine grid...")
    grid2 = _fine_grid_around(best_p1)
    p2_results = run_fine_grid(
        model, tokenizer, phase2_df, personas, base_config, grid2)

    print(f"\n  Top 5 Phase 2:")
    for i, r in enumerate(p2_results[:5]):
        print(f"    #{i+1} acc={r['accuracy']:.1f}%  "
              f"ns={r['noise_std']} lc={r['lambda_coop']} "
              f"t={r['temperature']}")

    best_final = {
        "noise_std": p2_results[0]["noise_std"],
        "lambda_coop": p2_results[0]["lambda_coop"],
        "temperature": p2_results[0]["temperature"],
    }

    config_out = TUNE_DIR / "best_config.json"
    with open(config_out, "w") as f:
        json.dump(best_final, f, indent=2)

    tune_elapsed = time.time() - t0_total
    print(f"\n  Tuning done in {tune_elapsed/60:.1f} min")
    print(f"  Best: {best_final}")

    # ---- Phase 3: Full eval ----
    print("\n[6/6] Phase 3: Full evaluation...")
    run_full_evaluation(model, tokenizer, best_final)

    # ---- Cleanup ----
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    total_elapsed = time.time() - t0_total
    print(f"\n{'='*70}")
    print(f"  ALL DONE — {total_elapsed/60:.1f} min total")
    print(f"{'='*70}")
    print(f"  Best params:  {best_final}")
    print(f"  Tune log:     {TUNE_LOG}")
    print(f"  Best config:  {config_out}")
    print(f"  Full results: {EVAL_DIR}")


if __name__ == "__main__":
    main()
