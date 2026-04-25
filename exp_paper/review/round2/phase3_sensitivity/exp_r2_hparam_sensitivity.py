#!/usr/bin/env python3
"""Round-2 Reviewer W3+W4 -- Hyperparameter sensitivity sweep.

Sweeps each of the four hand-tuned SWA-DPBR hyperparameters independently
and reports JSD / MIS / Pearson r on a small country panel so we can show
the robustness band around each default in the paper appendix.

Axes swept (one at a time; others held at default):
    s         VAR_SCALE      ∈ {0.010, 0.020, 0.040, 0.080, 0.160}   (reliability gate)
    lambda    lambda_coop    ∈ {0.30, 0.50, 0.70, 0.85, 0.95}        (consensus vs individual)
    sigma     noise_std      ∈ {0.10, 0.20, 0.30, 0.45, 0.60}        (proposal floor)
    T_cat     logit_temp     ∈ {1.0, 2.0, 3.0, 4.0, 5.0}             (uniform rescale)

A single model load is shared across all 5×4 = 20 runs. Each run reuses the
cached scenarios for the country so we pay the I/O only once per country.

Kaggle:
    !python exp_paper/review/round2/phase3_sensitivity/exp_r2_hparam_sensitivity.py

Env overrides:
    R2_MODEL          HF id (default: microsoft/phi-4)
    R2_COUNTRIES      comma ISO3 list (default: USA,VNM,DEU)
    R2_N_SCENARIOS    per-country (default: 250 -- half of the headline run
                      so the sweep fits in a single Kaggle session)
    R2_AXES           comma list from {s,lambda,sigma,tcat} (default: all)
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Self-bootstrap — works when copy-pasted into a fresh Kaggle notebook cell.
# Clones the repo on Kaggle if not already on sys.path, then adds it. Safe to
# run multiple times (idempotent: detects src/controller.py in cwd).
# ─────────────────────────────────────────────────────────────────────────────
import os as _os, subprocess as _sp, sys as _sys

_REPO_URL = "https://github.com/trungkiet2005/cultural_alignment.git"
_REPO_DIR_KAGGLE = "/kaggle/working/cultural_alignment"


def _r2_bootstrap() -> str:
    here = _os.getcwd()
    if _os.path.isfile(_os.path.join(here, "src", "controller.py")):
        if here not in _sys.path:
            _sys.path.insert(0, here)
        return here
    if not _os.path.isdir("/kaggle/input"):
        raise RuntimeError(
            "Not on Kaggle and not inside the repo root. "
            "Either cd into the cultural_alignment repo first, or run on Kaggle."
        )
    if not _os.path.isdir(_REPO_DIR_KAGGLE):
        _sp.run(["git", "clone", "--depth", "1", _REPO_URL, _REPO_DIR_KAGGLE], check=True)
    _os.chdir(_REPO_DIR_KAGGLE)
    _sys.path.insert(0, _REPO_DIR_KAGGLE)
    return _REPO_DIR_KAGGLE


_r2_bootstrap()


# Set backend BEFORE paper_runtime is imported so install_paper_kaggle_deps()
# picks the correct pip branch (vLLM vs Unsloth).
_os.environ.setdefault("MORAL_MODEL_BACKEND", _os.environ.get("R2_BACKEND", "vllm"))
import gc
import os
import time
from pathlib import Path
from typing import Dict, List

from exp_paper._r2_common import (
    build_cfg,
    load_model_timed,
    load_scenarios,
    on_kaggle,
    save_summary,
)
from exp_paper.paper_runtime import configure_paper_env, install_paper_kaggle_deps  # noqa: E402

configure_paper_env()
from src.hf_env import apply_hf_credentials  # noqa: E402

apply_hf_credentials()
install_paper_kaggle_deps()
import pandas as pd  # noqa: E402
import torch  # noqa: E402

# IMPORTANT: import AFTER env is configured -- VAR_SCALE / K_HALF are captured
# at import time from env. We'll reassign them on the fly below.
from experiment_DM import exp24_dpbr_core as _dpbr  # noqa: E402
from experiment_DM.exp24_dpbr_core import (  # noqa: E402
    BootstrapPriorState,
    PRIOR_STATE,
    patch_swa_runner_controller,
)
from src.model import setup_seeds  # noqa: E402
from src.personas import SUPPORTED_COUNTRIES, build_country_personas  # noqa: E402
from src.swa_runner import run_country_experiment  # noqa: E402

MODEL_NAME = os.environ.get("R2_MODEL", "microsoft/phi-4")
N_SCEN = int(os.environ.get("R2_N_SCENARIOS", "250"))
COUNTRIES = [
    c.strip()
    for c in os.environ.get("R2_COUNTRIES", "USA,VNM,DEU").split(",")
    if c.strip()
]
AXES = set(c.strip() for c in os.environ.get("R2_AXES", "s,lambda,sigma,tcat").split(","))

RESULTS_BASE = (
    "/kaggle/working/cultural_alignment/results/exp24_round2/hparam_sensitivity"
    if on_kaggle()
    else str(Path(__file__).parent.parent / "results" / "exp24_round2" / "hparam_sensitivity")
)
WVS_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"


# ----- Default (Full SWA-DPBR) hyperparameters ------------------------------ #
DEFAULT_S      = 0.04
DEFAULT_LAMBDA = 0.70
DEFAULT_SIGMA  = 0.30
DEFAULT_TCAT   = 3.0  # uniform scalar; per-category defaults remain on the cfg

# ----- Sweep grids ---------------------------------------------------------- #
GRID = {
    "s":      [0.010, 0.020, 0.040, 0.080, 0.160],
    "lambda": [0.30, 0.50, 0.70, 0.85, 0.95],
    "sigma":  [0.10, 0.20, 0.30, 0.45, 0.60],
    "tcat":   [1.0,  2.0,  3.0,  4.0,  5.0],
}


def _reset_prior(country: str) -> None:
    PRIOR_STATE.clear()
    PRIOR_STATE[country] = BootstrapPriorState()


def _run_one(model, tokenizer, cfg, country, scen, personas,
             axis: str, value: float) -> Dict:
    """Apply one (axis, value) override, run the country, return row."""
    # Snapshot default cfg so we can revert after this run.
    _snap = dict(
        lambda_coop=cfg.lambda_coop,
        noise_std=cfg.noise_std,
        logit_temperature=cfg.logit_temperature,
        category_logit_temperatures=dict(cfg.category_logit_temperatures),
    )
    _snap_var = _dpbr.VAR_SCALE

    if axis == "s":
        _dpbr.VAR_SCALE = float(value)
    elif axis == "lambda":
        cfg.lambda_coop = float(value)
    elif axis == "sigma":
        cfg.noise_std = float(value)
    elif axis == "tcat":
        # Uniform rescale: every category temperature is multiplied by
        # value / DEFAULT_TCAT. This probes the effect of globally sharper /
        # flatter decision logits without perturbing the relative balance
        # between categories.
        scale = float(value) / DEFAULT_TCAT
        cfg.logit_temperature = DEFAULT_TCAT * scale
        cfg.category_logit_temperatures = {
            k: float(v) * scale for k, v in _snap["category_logit_temperatures"].items()
        }
    else:
        raise ValueError(f"unknown axis {axis!r}")

    _reset_prior(country)
    patch_swa_runner_controller()

    t0 = time.time()
    try:
        results_df, summary = run_country_experiment(
            model, tokenizer, country, personas, scen, cfg,
        )
        dt = time.time() - t0
        a = summary["alignment"]
        row = {
            "axis":    axis,
            "value":   float(value),
            "country": country,
            "mis":       a.get("mis",         float("nan")),
            "jsd":       a.get("jsd",         float("nan")),
            "pearson_r": a.get("pearson_r",   float("nan")),
            "spearman_rho": a.get("spearman_rho", float("nan")),
            "mae":       a.get("mae",         float("nan")),
            "rmse":      a.get("rmse",        float("nan")),
            "flip_rate": summary.get("flip_rate", float("nan")),
            "mean_reliability_r":  float(results_df["reliability_r"].mean())
                if "reliability_r" in results_df.columns else float("nan"),
            "mean_bootstrap_var":  float(results_df["bootstrap_var"].mean())
                if "bootstrap_var" in results_df.columns else float("nan"),
            "mean_ess_pass1":      float(results_df["ess_pass1"].mean())
                if "ess_pass1" in results_df.columns else float("nan"),
            "elapsed_sec": dt,
            "n_scenarios": len(results_df),
        }
    finally:
        # Revert
        cfg.lambda_coop = _snap["lambda_coop"]
        cfg.noise_std = _snap["noise_std"]
        cfg.logit_temperature = _snap["logit_temperature"]
        cfg.category_logit_temperatures = _snap["category_logit_temperatures"]
        _dpbr.VAR_SCALE = _snap_var
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return row


def main() -> None:
    setup_seeds(42)
    out_dir = Path(RESULTS_BASE)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(
        MODEL_NAME, RESULTS_BASE, COUNTRIES,
        n_scenarios=N_SCEN, load_in_4bit=False,
    )
    backend = os.environ.get("MORAL_MODEL_BACKEND", "vllm").strip().lower()
    model, tokenizer = load_model_timed(
        MODEL_NAME, backend=backend, load_in_4bit=False,
    )

    # Cache scenarios + personas per country (MultiTP load is O(1M rows)).
    scen_cache: Dict[str, pd.DataFrame] = {}
    pers_cache: Dict[str, List[str]]    = {}
    for c in COUNTRIES:
        if c not in SUPPORTED_COUNTRIES:
            print(f"[SKIP] {c}: not in SUPPORTED_COUNTRIES")
            continue
        scen_cache[c] = load_scenarios(cfg, c)
        pers_cache[c] = build_country_personas(c, wvs_path=WVS_PATH)

    all_rows: List[Dict] = []
    for axis in ("s", "lambda", "sigma", "tcat"):
        if axis not in AXES:
            continue
        print(f"\n{'#' * 72}\n# Axis: {axis}  |  Grid: {GRID[axis]}\n{'#' * 72}")
        for val in GRID[axis]:
            for country in COUNTRIES:
                if country not in scen_cache:
                    continue
                print(f"\n[{axis}={val}] -> {country}")
                try:
                    row = _run_one(
                        model, tokenizer, cfg, country,
                        scen_cache[country], pers_cache[country],
                        axis=axis, value=val,
                    )
                    all_rows.append(row)
                    pd.DataFrame(all_rows).to_csv(
                        out_dir / "sensitivity_partial.csv", index=False,
                    )
                except Exception as exc:  # keep going; log the failure row
                    print(f"[ERROR] {axis}={val} {country}: {exc}")
                    all_rows.append({
                        "axis": axis, "value": float(val), "country": country,
                        "error": str(exc)[:500],
                    })

    save_summary(all_rows, out_dir, "hparam_sensitivity_summary.csv")


if __name__ == "__main__":
    main()
