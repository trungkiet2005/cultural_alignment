#!/usr/bin/env python3
"""Round-2 -- multi-seed confidence intervals on the headline Phi-4 run.

Reviewers ask for CI on the 19--24% MIS reduction claim. This script runs
Phi-4 SWA-DPBR with 3 seeds across the 20 paper countries, then computes the
mean ± 95% normal-CI over seeds per country and across the macro mean.

Seeds:   {42, 101, 2026} (overrideable via R2_SEEDS).
Countries: 20 paper countries (overrideable via R2_COUNTRIES).
Scenarios: 500 per country (overrideable via R2_N_SCENARIOS).

The IS proposal noise is drawn from ``torch.randn`` which is seeded via
:func:`src.model.setup_seeds`, so per-seed variability comes from:
    * the scenario sampling order inside load_multitp_dataset (seed-threaded)
    * the 2×K_HALF IS proposal noise
    * the dual-pass bootstrap variance

Kaggle (best on a long session):
    !python exp_paper/round2/phase4_big_sweeps/exp_r2_multiseed_phi4.py
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
    if not _os.path.isdir("/kaggle/working"):
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

os.environ.setdefault("MORAL_MODEL_BACKEND", os.environ.get("R2_BACKEND", "vllm"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from exp_paper.paper_countries import PAPER_20_COUNTRIES  # noqa: E402
from experiment_DM.exp24_dpbr_core import (  # noqa: E402
    BootstrapPriorState,
    PRIOR_STATE,
    patch_swa_runner_controller,
)
from src.baseline_runner import run_baseline_vanilla  # noqa: E402
from src.model import setup_seeds  # noqa: E402
from src.personas import SUPPORTED_COUNTRIES, build_country_personas  # noqa: E402
from src.swa_runner import run_country_experiment  # noqa: E402

MODEL_NAME = os.environ.get("R2_MODEL", "microsoft/phi-4")
N_SCEN = int(os.environ.get("R2_N_SCENARIOS", "500"))
SEEDS = [
    int(s.strip())
    for s in os.environ.get("R2_SEEDS", "42,101,2026").split(",")
    if s.strip()
]
COUNTRIES = (
    [c.strip() for c in os.environ["R2_COUNTRIES"].split(",") if c.strip()]
    if "R2_COUNTRIES" in os.environ
    else list(PAPER_20_COUNTRIES)
)

RESULTS_BASE = (
    "/kaggle/working/cultural_alignment/results/exp24_round2/multiseed_phi4"
    if on_kaggle()
    else str(Path(__file__).parent.parent / "results" / "exp24_round2" / "multiseed_phi4")
)
WVS_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"


def _row_from_alignment(method: str, country: str, seed: int, a: Dict, dt: float, n: int) -> Dict:
    return {
        "method":      method,
        "country":     country,
        "seed":        seed,
        "n_scenarios": n,
        "elapsed_sec": dt,
        "mis":         a.get("mis",         float("nan")),
        "jsd":         a.get("jsd",         float("nan")),
        "pearson_r":   a.get("pearson_r",   float("nan")),
        "spearman_rho": a.get("spearman_rho", float("nan")),
        "mae":         a.get("mae",         float("nan")),
        "rmse":        a.get("rmse",        float("nan")),
    }


def _ci95(x: np.ndarray) -> float:
    """Normal 95% half-width from seed variability (1.96 * sd/sqrt(n))."""
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    return float(1.96 * x.std(ddof=1) / np.sqrt(x.size))


def main() -> None:
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

    all_rows: List[Dict] = []
    try:
        for seed in SEEDS:
            os.environ["EXP24_SEED"] = str(seed)
            setup_seeds(seed)
            print(f"\n{'#' * 72}\n# SEED = {seed}\n{'#' * 72}")
            for ci, country in enumerate(COUNTRIES):
                if country not in SUPPORTED_COUNTRIES:
                    continue
                print(f"\n[seed={seed}] [{ci+1}/{len(COUNTRIES)}] {country}")
                scen = load_scenarios(cfg, country)
                personas = build_country_personas(country, wvs_path=WVS_PATH)

                # Vanilla baseline
                t0 = time.time()
                bl = run_baseline_vanilla(model, tokenizer, scen, country, cfg)
                dt = time.time() - t0
                all_rows.append(_row_from_alignment(
                    "vanilla", country, seed, bl["alignment"], dt, len(bl["results_df"]),
                ))

                # SWA-DPBR
                PRIOR_STATE.clear()
                PRIOR_STATE[country] = BootstrapPriorState()
                patch_swa_runner_controller()
                t0 = time.time()
                _, summary = run_country_experiment(
                    model, tokenizer, country, personas, scen, cfg,
                )
                dt = time.time() - t0
                all_rows.append(_row_from_alignment(
                    "swa_dpbr", country, seed, summary["alignment"], dt, summary.get("n_scenarios", N_SCEN),
                ))

                pd.DataFrame(all_rows).to_csv(out_dir / "multiseed_partial.csv", index=False)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "multiseed_all.csv", index=False)

    # Per-country mean ± CI over seeds, separately for vanilla and swa_dpbr.
    ci_rows: List[Dict] = []
    for (method, country), grp in df.groupby(["method", "country"]):
        rec = {"method": method, "country": country, "n_seeds": len(grp)}
        for m in ("mis", "jsd", "pearson_r", "mae", "rmse"):
            vals = grp[m].values.astype(np.float64)
            rec[f"mean_{m}"] = float(np.nanmean(vals))
            rec[f"ci95_{m}"] = _ci95(vals)
        ci_rows.append(rec)
    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(out_dir / "multiseed_per_country_ci.csv", index=False)

    # Macro mean ± CI: first macro-average over countries for each (seed, method),
    # then compute CI of the resulting per-seed macro values.
    macro = (df.groupby(["method", "seed"])[["mis", "jsd", "pearson_r", "mae", "rmse"]]
               .mean().reset_index())
    macro_ci: List[Dict] = []
    for method, grp in macro.groupby("method"):
        rec = {"method": method, "n_seeds": len(grp)}
        for m in ("mis", "jsd", "pearson_r", "mae", "rmse"):
            vals = grp[m].values.astype(np.float64)
            rec[f"mean_{m}"] = float(np.nanmean(vals))
            rec[f"ci95_{m}"] = _ci95(vals)
        macro_ci.append(rec)
    pd.DataFrame(macro_ci).to_csv(out_dir / "multiseed_macro_ci.csv", index=False)

    save_summary(all_rows, out_dir, "multiseed_all_rows.csv")
    print("\n[DONE] Multi-seed CI computation finished.")


if __name__ == "__main__":
    main()
