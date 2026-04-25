#!/usr/bin/env python3
"""Experiment 2 (playbook) — Country-Level Correlation (Figure 3).

Aggregates per-scenario data (from exp_r4_scenario_logging.py extended to all
20 paper countries) to the country level, then plots:
  x-axis: mean inter-persona variance per country
  y-axis: MIS improvement (vanilla_mis - disca_mis) per country

Can be run in two modes:
  1. Standalone post-hoc (recommended): load scenario_analysis_all_countries.csv
     and main_results.csv from previous runs.
  2. Full rerun: re-runs DISCA on all 20 countries to collect fresh per-scenario
     variance data (set R4_RERUN=1).

Inputs (auto-detected in RESULTS_BASE or R4_SCENARIO_CSV / R4_RESULTS_CSV):
  scenario_analysis_all_countries.csv   — per-scenario rows (from scenario_logging)
  main_results_phi4.csv                 — per-country vanilla_mis + disca_mis columns

Outputs (in RESULTS_BASE/):
  figure3_country_correlation.pdf / .png
  country_level_analysis.csv

Env overrides:
  R4_MODEL            HF id for rerun (default: microsoft/phi-4)
  R4_N_SCENARIOS      per-country for rerun (default: 500)
  R4_BACKEND          vllm (default) | hf_native
  R4_RERUN            1 = re-collect per-scenario data; 0 = post-hoc only (default: 0)
  R4_SCENARIO_CSV     path to existing scenario_analysis CSV (skips rerun)
  R4_RESULTS_CSV      path to existing main_results CSV (mis per country)

Kaggle:
    !python exp_paper/playbook/exp_r4_country_correlation.py
"""

from __future__ import annotations

import os as _os, subprocess as _sp, sys as _sys

_REPO_URL = "https://github.com/trungkiet2005/cultural_alignment.git"
_REPO_DIR_KAGGLE = "/kaggle/working/cultural_alignment"


def _bootstrap() -> str:
    here = _os.getcwd()
    if _os.path.isfile(_os.path.join(here, "src", "controller.py")):
        if here not in _sys.path:
            _sys.path.insert(0, here)
        return here
    if not _os.path.isdir("/kaggle/input"):
        raise RuntimeError("Not on Kaggle and not inside the repo root.")
    if not _os.path.isdir(_REPO_DIR_KAGGLE):
        _sp.run(["git", "clone", "--depth", "1", _REPO_URL, _REPO_DIR_KAGGLE], check=True)
    _os.chdir(_REPO_DIR_KAGGLE)
    _sys.path.insert(0, _REPO_DIR_KAGGLE)
    return _REPO_DIR_KAGGLE


_bootstrap()
_os.environ.setdefault("MORAL_MODEL_BACKEND", _os.environ.get("R4_BACKEND", "vllm"))

import time
from pathlib import Path
from typing import Dict, List, Optional

from exp_paper._r2_common import build_cfg, load_model_timed, load_scenarios, on_kaggle
from exp_paper.paper_runtime import configure_paper_env, install_paper_kaggle_deps

configure_paper_env()
from src.hf_env import apply_hf_credentials

apply_hf_credentials()
install_paper_kaggle_deps()

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_DM.exp24_dpbr_core import BootstrapPriorState, PRIOR_STATE, patch_swa_runner_controller
from src.model import setup_seeds
from src.personas import SUPPORTED_COUNTRIES, build_country_personas
from src.swa_runner import run_country_experiment
from src.baseline_runner import run_baseline_vanilla
from exp_paper.paper_countries import PAPER_20_COUNTRIES

MODEL_NAME = _os.environ.get("R4_MODEL", "microsoft/phi-4")
N_SCEN = int(_os.environ.get("R4_N_SCENARIOS", "500"))
DO_RERUN = _os.environ.get("R4_RERUN", "0").strip() == "1"

RESULTS_BASE = (
    "/kaggle/working/cultural_alignment/results/exp24_round4/country_correlation"
    if on_kaggle()
    else str(Path(__file__).parent.parent / "results" / "exp24_round4" / "country_correlation")
)
WVS_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"


def _run_all_countries(model, tokenizer, cfg, countries: List[str]) -> pd.DataFrame:
    """Run vanilla + DISCA for each country; return per-country summary rows."""
    rows: List[Dict] = []
    for ci, country in enumerate(countries):
        if country not in SUPPORTED_COUNTRIES:
            continue
        print(f"[{ci+1}/{len(countries)}] {country}")
        scen = load_scenarios(cfg, country)
        personas = build_country_personas(country, wvs_path=WVS_PATH)

        # Vanilla
        t0 = time.time()
        bl = run_baseline_vanilla(model, tokenizer, scen, country, cfg)
        vanilla_mis = float(bl["alignment"].get("mis", float("nan")))

        # DISCA
        PRIOR_STATE.clear()
        PRIOR_STATE[country] = BootstrapPriorState()
        patch_swa_runner_controller()
        results_df, summary = run_country_experiment(
            model, tokenizer, country, personas, scen, cfg)
        disca_mis = float(summary["alignment"].get("mis", float("nan")))

        mean_variance = float(results_df["mppi_variance"].mean())
        rows.append({
            "country":       country,
            "vanilla_mis":   vanilla_mis,
            "disca_mis":     disca_mis,
            "delta_mis":     vanilla_mis - disca_mis,
            "mean_variance": mean_variance,
            "elapsed_sec":   time.time() - t0,
        })
        print(f"  vanilla_mis={vanilla_mis:.4f}  disca_mis={disca_mis:.4f}  "
              f"delta={vanilla_mis - disca_mis:+.4f}  mean_var={mean_variance:.4f}")

    return pd.DataFrame(rows)


def _load_from_csvs(scenario_csv: str, results_csv: str) -> pd.DataFrame:
    """Build country_df from pre-existing CSVs."""
    scen_df = pd.read_csv(scenario_csv)
    res_df  = pd.read_csv(results_csv)

    # Scenario CSV: compute mean_variance per country
    mean_var = scen_df.groupby("country")["persona_variance"].mean().rename("mean_variance")

    # Results CSV: need vanilla_mis and disca_mis columns (or method column)
    if "vanilla_mis" in res_df.columns and "disca_mis" in res_df.columns:
        country_df = res_df.set_index("country")[["vanilla_mis", "disca_mis"]].copy()
        country_df["delta_mis"] = country_df["vanilla_mis"] - country_df["disca_mis"]
    elif "method" in res_df.columns and "mis" in res_df.columns:
        pivot = res_df.pivot_table(index="country", columns="method", values="mis")
        # Accept common method name variants
        van_col  = next((c for c in pivot.columns if "vanilla" in c.lower()), None)
        swa_col  = next((c for c in pivot.columns if any(k in c.lower() for k in ("swa", "disca", "dpbr"))), None)
        if van_col is None or swa_col is None:
            raise ValueError(f"Cannot find vanilla/DISCA columns in {results_csv}. "
                             f"Found columns: {list(pivot.columns)}")
        country_df = pd.DataFrame({
            "vanilla_mis": pivot[van_col],
            "disca_mis":   pivot[swa_col],
            "delta_mis":   pivot[van_col] - pivot[swa_col],
        })
    else:
        raise ValueError(f"Cannot parse {results_csv}. "
                         "Expected (country, vanilla_mis, disca_mis) or (country, method, mis).")

    country_df = country_df.join(mean_var, how="inner").reset_index()
    return country_df


def make_figure3(country_df: pd.DataFrame, out_dir: Path) -> None:
    from scipy.stats import pearsonr

    fig, ax = plt.subplots(figsize=(7, 5))

    colors = ["#1A8A66" if d > 0 else "#C04E28" for d in country_df["delta_mis"]]
    ax.scatter(country_df["mean_variance"], country_df["delta_mis"],
               s=80, c=colors, alpha=0.7, edgecolors="black", linewidths=0.5)

    for _, row in country_df.iterrows():
        ax.annotate(row["country"],
                    (row["mean_variance"], row["delta_mis"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=9)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    r, p = pearsonr(country_df["mean_variance"], country_df["delta_mis"])
    z = np.polyfit(country_df["mean_variance"], country_df["delta_mis"], 1)
    x_range = np.linspace(country_df["mean_variance"].min(),
                          country_df["mean_variance"].max(), 100)
    ax.plot(x_range, z[0] * x_range + z[1], "k-", linewidth=1, alpha=0.5)

    label = f"Pearson r = {r:.3f}\np = {p:.3f}" if p >= 0.001 else f"Pearson r = {r:.3f}\np < 0.001"
    ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))

    ax.set_xlabel("Mean inter-persona variance per country", fontsize=12)
    ax.set_ylabel("MIS improvement (vanilla − DISCA)", fontsize=12)
    plt.tight_layout()

    stem = out_dir / "figure3_country_correlation"
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(stem) + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {stem}.pdf / .png  |  Pearson r={r:.4f}  p={p:.4f}")
    print(f"Interpretation: {'STRONG' if r > 0.5 else 'MODERATE' if r > 0.3 else 'WEAK'} "
          f"country-level signal  (target r > 0.5)")


def main() -> None:
    setup_seeds(42)
    out_dir = Path(RESULTS_BASE)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Mode 1: load from existing CSVs ───────────────────────────────────────
    scenario_csv = _os.environ.get("R4_SCENARIO_CSV", "")
    results_csv  = _os.environ.get("R4_RESULTS_CSV", "")

    if not DO_RERUN and scenario_csv and results_csv:
        print(f"Post-hoc mode: loading from\n  {scenario_csv}\n  {results_csv}")
        country_df = _load_from_csvs(scenario_csv, results_csv)
    elif not DO_RERUN:
        # Try auto-detect from round4/scenario_logging output
        candidate_scen = (
            out_dir.parent / "scenario_logging" / "scenario_analysis.csv"
        )
        candidate_res  = (
            out_dir.parent / "scenario_logging" / "main_results_phi4.csv"
        )
        # Also accept scenario_analysis from a broader run
        for scen_try in [candidate_scen,
                         out_dir.parent / "scenario_logging" / "scenario_analysis_partial.csv"]:
            if scen_try.exists():
                candidate_scen = scen_try
                break
        if candidate_scen.exists() and candidate_res.exists():
            print(f"Auto-detected CSVs:\n  {candidate_scen}\n  {candidate_res}")
            country_df = _load_from_csvs(str(candidate_scen), str(candidate_res))
        else:
            print("No pre-existing CSVs found — switching to full rerun mode.")
            DO_RERUN_local = True
    else:
        DO_RERUN_local = True

    if DO_RERUN or (not DO_RERUN and not scenario_csv):
        countries = [c for c in PAPER_20_COUNTRIES if c in SUPPORTED_COUNTRIES]
        cfg = build_cfg(MODEL_NAME, RESULTS_BASE, countries,
                        n_scenarios=N_SCEN, load_in_4bit=False)
        backend = _os.environ.get("MORAL_MODEL_BACKEND", "vllm").strip().lower()
        model, tokenizer = load_model_timed(MODEL_NAME, backend=backend, load_in_4bit=False)
        country_df = _run_all_countries(model, tokenizer, cfg, countries)
        country_df.to_csv(out_dir / "country_rerun_results.csv", index=False)

    csv_path = out_dir / "country_level_analysis.csv"
    country_df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(country_df)}-country table → {csv_path}")
    print(country_df[["country", "mean_variance", "vanilla_mis", "disca_mis", "delta_mis"]]
          .sort_values("delta_mis", ascending=False).to_string(index=False))

    make_figure3(country_df, out_dir)


if __name__ == "__main__":
    main()
