#!/usr/bin/env python3
"""Experiment 3 (playbook) — Multi-Seed Confidence Intervals.

Runs full DISCA-DPBR pipeline across N countries × K seeds and reports
per-country MIS mean±std plus a macro CI across seeds.

Outputs (in RESULTS_BASE/):
  multiseed_raw.csv         — one row per (seed, country) with MIS/JSD/Pearson r
  multiseed_country.csv     — per-country aggregates: mis_mean, mis_std, r_mean, r_std
  multiseed_macro.csv       — single-row macro: mean MIS across seeds + std across seeds

Defends against:
  R1: "Single-seed results. Where are the error bars?"

Env overrides:
  R4_MODEL        HF id (default: microsoft/phi-4)
  R4_COUNTRIES    comma ISO3 list (default: USA,JPN,DEU,VNM,ETH)
  R4_N_SCENARIOS  per-country (default: 500)
  R4_SEEDS        comma list (default: 42,101,2026)
  R4_BACKEND      vllm (default) | hf_native

Kaggle:
    !python exp_paper/playbook/exp_r4_multiseed.py
"""

from __future__ import annotations

import os as _os
import subprocess as _sp
import sys as _sys

_REPO_URL = "https://github.com/trungkiet2005/cultural_alignment.git"
_REPO_DIR_KAGGLE = "/kaggle/working/cultural_alignment"


def _bootstrap() -> str:
    here = _os.getcwd()
    if _os.path.isfile(_os.path.join(here, "src", "controller.py")):
        if here not in _sys.path:
            _sys.path.insert(0, here)
        return here
    if not _os.path.isdir("/kaggle/input"):
        raise RuntimeError("Not on Kaggle and not inside repo root.")
    if not _os.path.isdir(_REPO_DIR_KAGGLE):
        _sp.run(["git", "clone", "--depth", "1", _REPO_URL, _REPO_DIR_KAGGLE], check=True)
    _os.chdir(_REPO_DIR_KAGGLE)
    _sys.path.insert(0, _REPO_DIR_KAGGLE)
    return _REPO_DIR_KAGGLE


_bootstrap()
_os.environ.setdefault("MORAL_MODEL_BACKEND", _os.environ.get("R4_BACKEND", "vllm"))

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from exp_paper._r2_common import build_cfg, load_model_timed, load_scenarios, on_kaggle
from exp_paper.paper_runtime import configure_paper_env, install_paper_kaggle_deps

configure_paper_env()
from src.hf_env import apply_hf_credentials

apply_hf_credentials()
install_paper_kaggle_deps()

from experiment_DM.exp24_dpbr_core import (
    BootstrapPriorState,
    PRIOR_STATE,
    patch_swa_runner_controller,
)
from src.model import setup_seeds
from src.personas import SUPPORTED_COUNTRIES, build_country_personas
from src.swa_runner import run_country_experiment

MODEL_NAME = _os.environ.get("R4_MODEL", "microsoft/phi-4")
N_SCEN = int(_os.environ.get("R4_N_SCENARIOS", "500"))
COUNTRIES = [
    c.strip()
    for c in _os.environ.get("R4_COUNTRIES", "USA,JPN,DEU,VNM,ETH").split(",")
    if c.strip()
]
SEEDS = [
    int(s)
    for s in _os.environ.get("R4_SEEDS", "42,101,2026").split(",")
    if s.strip()
]

WVS_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
RESULTS_BASE = (
    "/kaggle/working/cultural_alignment/results/exp24_round4/multiseed"
    if on_kaggle()
    else str(Path(__file__).parent.parent / "results" / "exp24_round4" / "multiseed")
)


def main() -> None:
    out_dir = Path(RESULTS_BASE)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(MODEL_NAME, RESULTS_BASE, COUNTRIES,
                    n_scenarios=N_SCEN, load_in_4bit=False)
    backend = _os.environ.get("MORAL_MODEL_BACKEND", "vllm").strip().lower()
    model, tokenizer = load_model_timed(MODEL_NAME, backend=backend, load_in_4bit=False)
    patch_swa_runner_controller()

    rows: List[dict] = []
    for seed in SEEDS:
        print(f"\n{'='*60}\n  SEED = {seed}\n{'='*60}")
        setup_seeds(seed)
        PRIOR_STATE.clear()  # do not let prior leak across seeds
        for country in COUNTRIES:
            if country not in SUPPORTED_COUNTRIES:
                print(f"[SKIP] {country} not in SUPPORTED_COUNTRIES")
                continue
            print(f"\n[seed={seed}] country={country}")
            PRIOR_STATE[country] = BootstrapPriorState()
            scen = load_scenarios(cfg, country)
            personas = build_country_personas(country, wvs_path=WVS_PATH)
            _df, summary = run_country_experiment(
                model, tokenizer, country, personas, scen, cfg,
            )
            a = summary.get("alignment", {})
            rows.append({
                "seed": seed,
                "country": country,
                "mis": float(a.get("mis", np.nan)),
                "jsd": float(a.get("jsd", np.nan)),
                "pearson_r": float(a.get("pearson_r", np.nan)),
                "mae": float(a.get("mae", np.nan)),
            })
            # incremental save
            pd.DataFrame(rows).to_csv(out_dir / "multiseed_raw_partial.csv", index=False)

    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(out_dir / "multiseed_raw.csv", index=False)
    print(f"\nSaved {len(raw_df)} rows -> {out_dir/'multiseed_raw.csv'}")

    # Per-country aggregates
    country_stats = (
        raw_df.groupby("country", as_index=False)
        .agg(
            n_seeds=("mis", "size"),
            mis_mean=("mis", "mean"),
            mis_std=("mis", "std"),
            r_mean=("pearson_r", "mean"),
            r_std=("pearson_r", "std"),
        )
        .round(4)
    )
    country_stats.to_csv(out_dir / "multiseed_country.csv", index=False)

    # Macro across seeds: per-seed macro MIS, then mean+std of those K macros.
    per_seed_macro = raw_df.groupby("seed", as_index=False)["mis"].mean().rename(
        columns={"mis": "macro_mis"}
    )
    macro_mean = float(per_seed_macro["macro_mis"].mean())
    macro_std = (
        float(per_seed_macro["macro_mis"].std(ddof=1))
        if len(per_seed_macro) > 1 else float("nan")
    )
    pd.DataFrame([{
        "model": MODEL_NAME,
        "n_seeds": len(SEEDS),
        "n_countries": int(raw_df["country"].nunique()),
        "macro_mean_mis": macro_mean,
        "macro_std_across_seeds": macro_std,
    }]).to_csv(out_dir / "multiseed_macro.csv", index=False)

    # Console verdict
    print("\n" + "-" * 70)
    print("  Multi-Seed CI — RESULT")
    print("-" * 70)
    print(f"  seeds                    : {SEEDS}")
    print(f"  macro_mean_mis           : {macro_mean:.4f}")
    print(f"  macro_std (across seeds) : {macro_std:.4f}")
    if np.isfinite(macro_std):
        verdict = ("STABLE" if macro_std < 0.01 else
                   "ACCEPTABLE" if macro_std < 0.05 else "UNSTABLE")
        print(f"  stability                : [{verdict}]  (paper target: <0.01)")
    print("\n  Per-country mean ± std (mis):")
    print(country_stats[["country", "n_seeds", "mis_mean", "mis_std"]].to_string(index=False))
    print(f"\n  saved -> {out_dir}")


if __name__ == "__main__":
    main()
