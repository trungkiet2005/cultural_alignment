#!/usr/bin/env python3
"""Experiment 7 (playbook) — Predictive Failure Model.

Per-cell post-hoc analysis: regress DISCA's improvement (Δ MIS = vanilla -
disca) on three vanilla-pass predictors, computed from a single vanilla
forward pass per country:

  * mean_decision_margin  = mean of |p_B - p_A|
  * mean_logit_entropy    = mean of -p_A log p_A - p_B log p_B
  * vanilla_mis           = the vanilla MIS itself

Then makes the diagnostic scatter (vanilla_mis vs Δ MIS, color = country).

This script is GPU-cheap: runs vanilla on N countries, reads existing
DISCA per-country MIS from a CSV.

Outputs (in RESULTS_BASE/):
  failure_features.csv             — per-country (country, n, mean_margin, mean_entropy,
                                       vanilla_mis, disca_mis, delta_mis)
  failure_regression.csv           — single row: R², coefficients, intercept
  failure_scatter.pdf / .png       — vanilla_mis vs Δ MIS scatter

Defends against:
  R3: "When does DISCA fail? Can we predict it?"

Env overrides:
  R4_MODEL        HF id (default: microsoft/phi-4)
  R4_COUNTRIES    comma ISO3 list (default: PAPER_20_COUNTRIES)
  R4_N_SCENARIOS  per-country (default: 500)
  R4_BACKEND      vllm (default) | hf_native
  R4_DISCA_CSV    [REQUIRED] CSV with columns: country, mis (or disca_mis).
                  These are the per-country DISCA MIS values from your main run.

Kaggle:
    R4_DISCA_CSV=/kaggle/working/disca_phi4_results.csv \
        python exp_paper/playbook/exp_r4_failure_prediction.py
"""

from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _kaggle_setup import bootstrap_offline, zip_outputs as _zip_outputs

bootstrap_offline()
_os.environ.setdefault("MORAL_MODEL_BACKEND", _os.environ.get("R4_BACKEND", "vllm"))

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from exp_paper._r2_common import build_cfg, load_model_timed, load_scenarios, on_kaggle
from exp_paper.paper_countries import PAPER_20_COUNTRIES
from exp_paper.paper_runtime import configure_paper_env, install_paper_kaggle_deps

configure_paper_env()
from src.hf_env import apply_hf_credentials

apply_hf_credentials()
install_paper_kaggle_deps()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.baseline_runner import run_baseline_vanilla
from src.model import setup_seeds
from src.personas import SUPPORTED_COUNTRIES

MODEL_NAME = _os.environ.get("R4_MODEL", "microsoft/phi-4")
N_SCEN = int(_os.environ.get("R4_N_SCENARIOS", "500"))
COUNTRIES = (
    [c.strip() for c in _os.environ.get("R4_COUNTRIES", "").split(",") if c.strip()]
    or list(PAPER_20_COUNTRIES)
)
DISCA_CSV = _os.environ.get("R4_DISCA_CSV", "").strip()

RESULTS_BASE = (
    "/kaggle/working/cultural_alignment/results/exp24_round4/failure_prediction"
    if on_kaggle()
    else str(Path(__file__).parent.parent / "results" / "exp24_round4" / "failure_prediction")
)


def _vanilla_features(results_df: pd.DataFrame) -> Dict[str, float]:
    """Extract per-country vanilla-pass diagnostics from a baseline_runner DataFrame.

    Margin and entropy are computed from p_left/p_right when present.
    """
    pl = results_df.get("p_left")
    pr = results_df.get("p_right")
    if pl is None or pr is None:
        return {"mean_margin": float("nan"), "mean_entropy": float("nan")}
    pl = pd.to_numeric(pl, errors="coerce").to_numpy()
    pr = pd.to_numeric(pr, errors="coerce").to_numpy()
    eps = 1e-12
    margin = np.abs(pr - pl)
    entropy = -(pl * np.log(np.clip(pl, eps, 1)) + pr * np.log(np.clip(pr, eps, 1)))
    margin = margin[np.isfinite(margin)]
    entropy = entropy[np.isfinite(entropy)]
    return {
        "mean_margin": float(margin.mean()) if margin.size else float("nan"),
        "mean_entropy": float(entropy.mean()) if entropy.size else float("nan"),
    }


def _load_disca() -> Dict[str, float]:
    if not DISCA_CSV:
        raise RuntimeError(
            "R4_DISCA_CSV not set. Provide a CSV with per-country DISCA MIS "
            "(columns: country, mis | disca_mis)."
        )
    p = Path(DISCA_CSV)
    if not p.is_file():
        raise FileNotFoundError(f"R4_DISCA_CSV is not a file: {p}")
    df = pd.read_csv(p)
    col = "disca_mis" if "disca_mis" in df.columns else "mis"
    if col not in df.columns or "country" not in df.columns:
        raise ValueError(f"DISCA CSV needs 'country' and 'mis' (or 'disca_mis'); got {list(df.columns)}")
    return {str(r["country"]): float(r[col]) for _, r in df.iterrows()}


def main() -> None:
    setup_seeds(42)
    disca_mis = _load_disca()
    out_dir = Path(RESULTS_BASE)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(MODEL_NAME, RESULTS_BASE, COUNTRIES,
                    n_scenarios=N_SCEN, load_in_4bit=False)
    backend = _os.environ.get("MORAL_MODEL_BACKEND", "vllm").strip().lower()
    model, tokenizer = load_model_timed(MODEL_NAME, backend=backend, load_in_4bit=False)

    rows: List[Dict] = []
    for country in COUNTRIES:
        if country not in SUPPORTED_COUNTRIES:
            print(f"[SKIP] {country} not in SUPPORTED_COUNTRIES")
            continue
        if country not in disca_mis:
            print(f"[SKIP] {country} missing in DISCA CSV")
            continue
        print(f"\n[{country}] vanilla forward pass…")
        scen = load_scenarios(cfg, country)
        out = run_baseline_vanilla(model, tokenizer, scen, country, cfg)
        van_mis = float(out.get("alignment", {}).get("mis", np.nan))
        feats = _vanilla_features(out.get("results_df", pd.DataFrame()))
        rows.append({
            "country": country,
            "n_scenarios": N_SCEN,
            "mean_margin": feats["mean_margin"],
            "mean_entropy": feats["mean_entropy"],
            "vanilla_mis": van_mis,
            "disca_mis": float(disca_mis[country]),
            "delta_mis": van_mis - float(disca_mis[country]),
        })
        pd.DataFrame(rows).to_csv(out_dir / "failure_features_partial.csv", index=False)

    feat_df = pd.DataFrame(rows)
    feat_df.to_csv(out_dir / "failure_features.csv", index=False)
    if feat_df.empty:
        print("No countries collected; aborting.")
        return

    # --- Regression ---
    fit_df = feat_df.dropna(subset=["mean_margin", "mean_entropy",
                                    "vanilla_mis", "delta_mis"])
    if len(fit_df) < 4:
        print(f"Only {len(fit_df)} valid rows; skipping regression.")
        return
    from sklearn.linear_model import LinearRegression

    X = fit_df[["mean_margin", "mean_entropy", "vanilla_mis"]].to_numpy()
    y = fit_df["delta_mis"].to_numpy()
    reg = LinearRegression().fit(X, y)
    r2 = float(reg.score(X, y))
    coefs = reg.coef_.tolist()
    intercept = float(reg.intercept_)

    pd.DataFrame([{
        "n_cells": len(fit_df),
        "r_squared": r2,
        "coef_margin": coefs[0],
        "coef_entropy": coefs[1],
        "coef_vanilla_mis": coefs[2],
        "intercept": intercept,
    }]).to_csv(out_dir / "failure_regression.csv", index=False)

    # --- Scatter plot: vanilla_mis vs Δ MIS ---
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#1A8A66" if d > 0 else "#C04E28" for d in fit_df["delta_mis"]]
    ax.scatter(fit_df["vanilla_mis"], fit_df["delta_mis"],
               s=70, c=colors, alpha=0.75, edgecolors="black", linewidths=0.5)
    for _, r in fit_df.iterrows():
        ax.annotate(r["country"], (r["vanilla_mis"], r["delta_mis"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=9)
    ax.axhline(0, color="gray", lw=0.7, ls="--")

    if len(fit_df) >= 3:
        z = np.polyfit(fit_df["vanilla_mis"], fit_df["delta_mis"], 1)
        xs = np.linspace(fit_df["vanilla_mis"].min(), fit_df["vanilla_mis"].max(), 100)
        ax.plot(xs, z[0] * xs + z[1], "k-", lw=1, alpha=0.5)

    ax.set_xlabel("Vanilla MIS", fontsize=12)
    ax.set_ylabel(r"$\Delta$MIS (vanilla $-$ DISCA)", fontsize=12)
    ax.text(0.05, 0.95, f"R² = {r2:.3f}\nn = {len(fit_df)}",
            transform=ax.transAxes, fontsize=11, va="top",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))
    plt.tight_layout()

    pdf = out_dir / "failure_scatter.pdf"
    png = out_dir / "failure_scatter.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\n" + "-" * 70)
    print("  Predictive Failure Model — RESULT")
    print("-" * 70)
    print(f"  n_cells              : {len(fit_df)}")
    print(f"  R²                   : {r2:.4f}")
    print(f"  coef margin          : {coefs[0]:+.4f}")
    print(f"  coef entropy         : {coefs[1]:+.4f}")
    print(f"  coef vanilla_mis     : {coefs[2]:+.4f}")
    print(f"  intercept            : {intercept:+.4f}")
    print(f"\n  saved -> {out_dir}")


if __name__ == "__main__":
    main()
    try:
        _zip_outputs(RESULTS_BASE if 'RESULTS_BASE' in globals() else OUT_DIR if 'OUT_DIR' in globals() else '.')
    except Exception as _e:
        print(f'[ZIP] failed: {_e}')
