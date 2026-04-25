#!/usr/bin/env python3
"""Experiment 10 (playbook) — Reliability Weight Distribution (Appendix figure).

Loads per-scenario reliability weights (r) from existing results CSVs produced
by exp_r4_scenario_logging.py or any DISCA run that emits `reliability_r`.
If no pre-existing CSVs are found, optionally re-runs DISCA on 5 countries with
Phi-4 to collect the weights fresh.

Outputs (in RESULTS_BASE/):
  reliability_weights.csv              — combined per-scenario reliability_r data
  figure_reliability_distribution.pdf / .png

Env overrides:
  R4_MODEL          HF id for fresh run (default: microsoft/phi-4)
  R4_COUNTRIES      comma ISO3 (default: USA,JPN,DEU,VNM,ETH)
  R4_N_SCENARIOS    per-country for fresh run (default: 500)
  R4_BACKEND        vllm (default) | hf_native
  R4_INPUT_DIR      directory to scan for existing results CSVs (auto-detects swa_results_*.csv)
  R4_RERUN          1 = always re-run; 0 = use existing if found (default: 0)

Kaggle:
    !python exp_paper/playbook/exp_r4_reliability_dist.py
"""

from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _kaggle_setup import bootstrap_offline, zip_outputs as _zip_outputs

bootstrap_offline()
_os.environ.setdefault("MORAL_MODEL_BACKEND", _os.environ.get("R4_BACKEND", "vllm"))

import time
from pathlib import Path
from typing import List, Optional

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

MODEL_NAME = _os.environ.get("R4_MODEL", "microsoft/phi-4")
N_SCEN = int(_os.environ.get("R4_N_SCENARIOS", "500"))
COUNTRIES = [c.strip() for c in _os.environ.get("R4_COUNTRIES", "USA,JPN,DEU,VNM,ETH").split(",") if c.strip()]
DO_RERUN = _os.environ.get("R4_RERUN", "0").strip() == "1"

RESULTS_BASE = (
    "/kaggle/working/cultural_alignment/results/exp24_round4/reliability_dist"
    if on_kaggle()
    else str(Path(__file__).parent.parent / "results" / "exp24_round4" / "reliability_dist")
)
WVS_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"


def _scan_existing_csvs(search_dir: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Scan for swa_results_*.csv files that contain a reliability_r column."""
    search_roots = []
    if search_dir:
        search_roots.append(Path(search_dir))
    # Auto-detect common results dirs
    repo_root = Path(__file__).parent.parent.parent
    for candidate in [
        repo_root / "results",
        Path("/kaggle/working/cultural_alignment/results"),
        Path(__file__).parent.parent / "results",
    ]:
        if candidate.exists():
            search_roots.append(candidate)

    frames: List[pd.DataFrame] = []
    for root in search_roots:
        for csv_path in root.rglob("swa_results_*.csv"):
            try:
                df = pd.read_csv(csv_path)
                if "reliability_r" in df.columns:
                    sub = df[["country", "reliability_r"]].copy()
                    sub["source"] = csv_path.stem
                    frames.append(sub)
            except Exception:
                pass
        # Also check scenario_analysis.csv from exp_r4_scenario_logging
        for csv_path in root.rglob("scenario_analysis*.csv"):
            try:
                df = pd.read_csv(csv_path)
                if "reliability_r" in df.columns:
                    sub = df[["country", "reliability_r"]].copy()
                    sub["source"] = csv_path.stem
                    frames.append(sub)
            except Exception:
                pass

    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["reliability_r"])
    return combined if len(combined) > 0 else None


def _collect_fresh(model, tokenizer, cfg) -> pd.DataFrame:
    """Run DISCA on configured countries and collect per-scenario reliability_r."""
    frames: List[pd.DataFrame] = []
    for country in COUNTRIES:
        if country not in SUPPORTED_COUNTRIES:
            continue
        print(f"  Collecting reliability weights for {country}…")
        scen = load_scenarios(cfg, country)
        personas = build_country_personas(country, wvs_path=WVS_PATH)
        PRIOR_STATE.clear()
        PRIOR_STATE[country] = BootstrapPriorState()
        patch_swa_runner_controller()

        results_df, _ = run_country_experiment(model, tokenizer, country, personas, scen, cfg)
        if "reliability_r" in results_df.columns:
            sub = results_df[["country", "reliability_r"]].copy()
            sub["source"] = "fresh_run"
            frames.append(sub)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def make_histogram(df: pd.DataFrame, out_dir: Path) -> None:
    weights = df["reliability_r"].dropna().values
    if len(weights) == 0:
        print("No reliability_r data found — skipping histogram.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ── Left: overall histogram ────────────────────────────────────────────────
    ax = axes[0]
    ax.hist(weights, bins=40, color="#534AB7", alpha=0.7,
            edgecolor="black", linewidth=0.3)
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1.2,
               label="r = 0.5 (threshold)")
    frac_low = (weights < 0.5).mean()
    frac_high = (weights >= 0.5).mean()
    ax.text(0.03, 0.92, f"r < 0.5:  {frac_low*100:.1f}%\nr ≥ 0.5:  {frac_high*100:.1f}%",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))
    ax.set_xlabel("Reliability weight $r$", fontsize=12)
    ax.set_ylabel("Number of scenarios", fontsize=12)
    ax.set_title("Overall reliability weight distribution", fontsize=11)
    ax.legend(fontsize=10)

    # ── Right: per-country CDF ─────────────────────────────────────────────────
    ax2 = axes[1]
    country_colors = {
        "USA": "#2D5F9A", "JPN": "#1A8A66", "DEU": "#534AB7",
        "VNM": "#C04E28", "ETH": "#EF9F27",
    }
    for country in df["country"].unique():
        sub_w = df[df["country"] == country]["reliability_r"].dropna().values
        if len(sub_w) == 0:
            continue
        sorted_w = np.sort(sub_w)
        cdf = np.arange(1, len(sorted_w) + 1) / len(sorted_w)
        color = country_colors.get(country, "#888888")
        ax2.plot(sorted_w, cdf, label=country, color=color, linewidth=1.5)

    ax2.axvline(0.5, color="red", linestyle="--", linewidth=1.0)
    ax2.set_xlabel("Reliability weight $r$", fontsize=12)
    ax2.set_ylabel("CDF", fontsize=12)
    ax2.set_title("Per-country CDF of reliability weight", fontsize=11)
    ax2.legend(fontsize=9, loc="lower right")

    plt.tight_layout()

    stem = out_dir / "figure_reliability_distribution"
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(stem) + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {stem}.pdf / .png")

    # Summary stats
    print(f"\nReliability weight statistics across {len(weights)} scenarios:")
    print(f"  mean  = {weights.mean():.4f}")
    print(f"  std   = {weights.std():.4f}")
    print(f"  median= {np.median(weights):.4f}")
    print(f"  < 0.5 = {frac_low*100:.1f}%  (gate mostly suppressed IS correction)")
    print(f"  ≥ 0.5 = {frac_high*100:.1f}%  (gate passes IS correction)")
    print(f"\n  Paper claim: the DPBR gate is selective, not always-on.")
    print(f"  If < 0.5 rate is > 20%, the gate provides meaningful filtering.")


def main() -> None:
    setup_seeds(42)
    out_dir = Path(RESULTS_BASE)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_dir = _os.environ.get("R4_INPUT_DIR", "")
    rel_df: Optional[pd.DataFrame] = None

    if not DO_RERUN:
        print("Scanning for existing results with reliability_r column…")
        rel_df = _scan_existing_csvs(input_dir if input_dir else None)
        if rel_df is not None:
            print(f"Found {len(rel_df)} scenarios with reliability_r data "
                  f"across {rel_df['country'].nunique()} countries.")

    if rel_df is None or DO_RERUN:
        print("Running DISCA to collect fresh reliability weights…")
        cfg = build_cfg(MODEL_NAME, RESULTS_BASE, COUNTRIES, n_scenarios=N_SCEN, load_in_4bit=False)
        backend = _os.environ.get("MORAL_MODEL_BACKEND", "vllm").strip().lower()
        model, tokenizer = load_model_timed(MODEL_NAME, backend=backend, load_in_4bit=False)
        rel_df = _collect_fresh(model, tokenizer, cfg)

    if rel_df is None or len(rel_df) == 0:
        print("ERROR: No reliability_r data available. "
              "Run a DISCA experiment first, or set R4_RERUN=1.")
        return

    csv_path = out_dir / "reliability_weights.csv"
    rel_df.to_csv(csv_path, index=False)
    print(f"Saved {len(rel_df)} rows → {csv_path}")

    make_histogram(rel_df, out_dir)


if __name__ == "__main__":
    main()
    try:
        _zip_outputs(RESULTS_BASE if 'RESULTS_BASE' in globals() else OUT_DIR if 'OUT_DIR' in globals() else '.')
    except Exception as _e:
        print(f'[ZIP] failed: {_e}')
