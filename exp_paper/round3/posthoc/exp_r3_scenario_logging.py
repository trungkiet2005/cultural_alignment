#!/usr/bin/env python3
"""Experiment 1 (playbook) — Disagreement-Correction Correlation (Figure 2).

For each scenario in 5 representative countries, logs two values:
  * persona_variance   = inter-persona variance of debiased logit gaps  (mppi_variance in results_df)
  * correction_magnitude = |delta_star| applied by DISCA               (delta_z_norm in results_df)

Then makes the scatter plot: log10(persona_variance) vs correction_magnitude with
a LOWESS trend line and Pearson r annotation.

Outputs (in RESULTS_BASE/):
  scenario_analysis.csv          — per-scenario rows
  figure2_scenario_correlation.pdf / .png

Env overrides:
  R3_MODEL        HF id (default: microsoft/phi-4)
  R3_COUNTRIES    comma ISO3 list (default: USA,JPN,DEU,VNM,ETH)
  R3_N_SCENARIOS  per-country (default: 500)
  R3_BACKEND      vllm (default) | hf_native

Kaggle:
    !python exp_paper/round3/posthoc/exp_r3_scenario_logging.py
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
_os.environ.setdefault("MORAL_MODEL_BACKEND", _os.environ.get("R3_BACKEND", "vllm"))

import time
from pathlib import Path
from typing import List

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

MODEL_NAME = _os.environ.get("R3_MODEL", "microsoft/phi-4")
N_SCEN = int(_os.environ.get("R3_N_SCENARIOS", "500"))
COUNTRIES = [
    c.strip()
    for c in _os.environ.get("R3_COUNTRIES", "USA,JPN,DEU,VNM,ETH").split(",")
    if c.strip()
]

RESULTS_BASE = (
    "/kaggle/working/cultural_alignment/results/exp24_round3/scenario_logging"
    if on_kaggle()
    else str(Path(__file__).parent.parent / "results" / "exp24_round3" / "scenario_logging")
)
WVS_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"

COUNTRY_COLORS = {
    "USA": "#2D5F9A",
    "JPN": "#1A8A66",
    "DEU": "#534AB7",
    "VNM": "#C04E28",
    "ETH": "#EF9F27",
}


def _run_country(model, tokenizer, cfg, country: str) -> pd.DataFrame:
    """Run DISCA on one country; return per-scenario DataFrame with variance + correction columns."""
    scen = load_scenarios(cfg, country)
    personas = build_country_personas(country, wvs_path=WVS_PATH)
    PRIOR_STATE.clear()
    PRIOR_STATE[country] = BootstrapPriorState()
    patch_swa_runner_controller()

    results_df, _ = run_country_experiment(model, tokenizer, country, personas, scen, cfg)

    # Build analysis slice; fall back gracefully if dual-pass columns absent
    out = pd.DataFrame({
        "scenario_id": results_df["scenario_idx"].astype(str),
        "country":     country,
        "dimension":   results_df.get("phenomenon_category", pd.Series(["default"] * len(results_df))),
        "persona_variance":     results_df["mppi_variance"].astype(float),
        "correction_magnitude": results_df["delta_z_norm"].astype(float),
        "reliability_r":        results_df.get("reliability_r", pd.Series([float("nan")] * len(results_df))).astype(float),
    })
    return out


def make_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    from scipy.stats import pearsonr
    from scipy.signal import savgol_filter

    df = df.dropna(subset=["persona_variance", "correction_magnitude"]).copy()
    df["log_variance"] = np.log10(df["persona_variance"].clip(lower=1e-6) + 1e-4)

    fig, ax = plt.subplots(figsize=(7, 5))

    for country in COUNTRIES:
        sub = df[df["country"] == country]
        if sub.empty:
            continue
        color = COUNTRY_COLORS.get(country, "#888888")
        ax.scatter(sub["log_variance"], sub["correction_magnitude"],
                   alpha=0.3, s=10, color=color, label=country)

    sorted_df = df.sort_values("log_variance")
    window = max(51, len(sorted_df) // 20 * 2 + 1)
    if window < len(sorted_df):
        try:
            smoothed = savgol_filter(sorted_df["correction_magnitude"].values, window, 3)
            ax.plot(sorted_df["log_variance"], smoothed, "k-",
                    linewidth=1.5, label="LOWESS trend")
        except Exception:
            pass

    r, p = pearsonr(df["log_variance"], df["correction_magnitude"])
    label = (f"Pearson r = {r:.3f}\np < 0.001" if p < 0.001
             else f"Pearson r = {r:.3f}\np = {p:.3f}")
    ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))

    ax.set_xlabel(r"Inter-persona variance, $\log_{10} S(x)$", fontsize=12)
    ax.set_ylabel(r"Correction magnitude, $|\delta^*(x)|$", fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    stem = out_dir / "figure2_scenario_correlation"
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(stem) + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {stem}.pdf / .png  |  Pearson r={r:.4f}  p={p:.2e}")
    print(f"  Interpretation: {'STRONG' if r > 0.4 else 'MODERATE' if r > 0.2 else 'WEAK'} signal")


def main() -> None:
    setup_seeds(42)
    out_dir = Path(RESULTS_BASE)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(MODEL_NAME, RESULTS_BASE, COUNTRIES,
                    n_scenarios=N_SCEN, load_in_4bit=False)
    backend = _os.environ.get("MORAL_MODEL_BACKEND", "vllm").strip().lower()
    model, tokenizer = load_model_timed(MODEL_NAME, backend=backend, load_in_4bit=False)

    all_frames: List[pd.DataFrame] = []
    for country in COUNTRIES:
        if country not in SUPPORTED_COUNTRIES:
            print(f"[SKIP] {country} not in SUPPORTED_COUNTRIES")
            continue
        print(f"\n[{country}] running DISCA logging…")
        t0 = time.time()
        df_c = _run_country(model, tokenizer, cfg, country)
        print(f"  {len(df_c)} scenarios in {time.time()-t0:.0f}s  |  "
              f"mean_var={df_c['persona_variance'].mean():.4f}  "
              f"mean_corr={df_c['correction_magnitude'].mean():.4f}")
        all_frames.append(df_c)
        # Partial save in case of crash
        pd.concat(all_frames).to_csv(out_dir / "scenario_analysis_partial.csv", index=False)

    if not all_frames:
        print("No data collected — exiting.")
        return

    full_df = pd.concat(all_frames, ignore_index=True)
    csv_path = out_dir / "scenario_analysis.csv"
    full_df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(full_df)} rows → {csv_path}")

    print("\nMaking Figure 2 scatter plot…")
    make_scatter(full_df, out_dir)

    # Summary stats per country
    summary = full_df.groupby("country").agg(
        n=("persona_variance", "count"),
        mean_variance=("persona_variance", "mean"),
        mean_correction=("correction_magnitude", "mean"),
    ).reset_index()
    print("\nPer-country summary:")
    print(summary.to_string(index=False))
    summary.to_csv(out_dir / "scenario_analysis_summary.csv", index=False)


if __name__ == "__main__":
    main()
