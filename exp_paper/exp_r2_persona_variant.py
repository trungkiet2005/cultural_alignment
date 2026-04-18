#!/usr/bin/env python3
"""Round-2 Reviewer W6 -- persona 4th-slot sensitivity.

Runs SWA-DPBR twice per country on Phi-4:
    * ``fourth=aggregate``   (current code reality; default in the paper)
    * ``fourth=utilitarian`` (original Figure 1 label)

Reports the head-to-head delta so the paper can (a) reconcile the
text/figure mismatch, and (b) quantify how much it matters.

Kaggle:
    !python exp_paper/exp_r2_persona_variant.py
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

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from experiment_DM.exp24_dpbr_core import (  # noqa: E402
    BootstrapPriorState,
    PRIOR_STATE,
    patch_swa_runner_controller,
)
from src.model import setup_seeds  # noqa: E402
from src.personas import SUPPORTED_COUNTRIES, build_country_personas  # noqa: E402
from src.swa_runner import run_country_experiment  # noqa: E402
from exp_paper.paper_countries import PAPER_20_COUNTRIES  # noqa: E402

MODEL_NAME = os.environ.get("R2_MODEL", "microsoft/phi-4")
N_SCEN = int(os.environ.get("R2_N_SCENARIOS", "250"))
COUNTRIES = (
    [c.strip() for c in os.environ["R2_COUNTRIES"].split(",") if c.strip()]
    if "R2_COUNTRIES" in os.environ
    else list(PAPER_20_COUNTRIES)
)
VARIANTS = ("aggregate", "utilitarian")

RESULTS_BASE = (
    "/kaggle/working/cultural_alignment/results/exp24_round2/persona_variant"
    if on_kaggle()
    else str(Path(__file__).parent.parent / "results" / "exp24_round2" / "persona_variant")
)
WVS_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"


def _run_variant(model, tokenizer, cfg, country, scen, fourth: str) -> Dict:
    PRIOR_STATE.clear()
    PRIOR_STATE[country] = BootstrapPriorState()
    patch_swa_runner_controller()

    personas = build_country_personas(country, wvs_path=WVS_PATH, fourth=fourth)

    t0 = time.time()
    results_df, summary = run_country_experiment(
        model, tokenizer, country, personas, scen, cfg,
    )
    dt = time.time() - t0

    a = summary["alignment"]
    return {
        "variant":    fourth,
        "country":    country,
        "n_scenarios": len(results_df),
        "elapsed_sec": dt,
        "mis":        a.get("mis",        float("nan")),
        "jsd":        a.get("jsd",        float("nan")),
        "pearson_r":  a.get("pearson_r",  float("nan")),
        "spearman_rho": a.get("spearman_rho", float("nan")),
        "mae":        a.get("mae",        float("nan")),
        "rmse":       a.get("rmse",       float("nan")),
        "flip_rate":  summary.get("flip_rate", float("nan")),
    }


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

    scen_cache: Dict[str, pd.DataFrame] = {}
    for c in COUNTRIES:
        if c not in SUPPORTED_COUNTRIES:
            continue
        scen_cache[c] = load_scenarios(cfg, c)

    rows: List[Dict] = []
    for variant in VARIANTS:
        print(f"\n{'#' * 72}\n# variant={variant}\n{'#' * 72}")
        for country in COUNTRIES:
            if country not in scen_cache:
                continue
            try:
                rows.append(_run_variant(model, tokenizer, cfg, country, scen_cache[country], variant))
                pd.DataFrame(rows).to_csv(out_dir / "persona_variant_partial.csv", index=False)
            except Exception as exc:
                print(f"[ERROR] variant={variant} {country}: {exc}")
                rows.append({"variant": variant, "country": country, "error": str(exc)[:500]})
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    save_summary(rows, out_dir, "persona_variant_summary.csv")


if __name__ == "__main__":
    main()
