#!/usr/bin/env python3
"""Round-2 Reviewer W5b -- no-cap / no-oversampling sensitivity.

Re-runs Phi-4 SWA-DPBR on the 20 paper countries with the per-category
sub-sampling cap disabled, so reviewers can see that the 19--24% MIS
reduction is not an artefact of balancing the scenario pool.

Note: the loader never *up*-samples scenarios beyond what the source CSV
contains -- the "oversampling" the reviewer refers to is the per-category
cap at 80 (see src/constants.py::MAX_SCENARIOS_PER_CATEGORY) plus the
deterministic hashlib fallback in :func:`src.data.parse_left_right`.
Dropping the cap + dumping the exact scenario ids reproduces the entire
preprocessing path.

Kaggle:
    !python exp_paper/round2/phase3_sensitivity/exp_r2_no_oversampling.py
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

from exp_paper.paper_countries import PAPER_20_COUNTRIES  # noqa: E402
from experiment_DM.exp24_dpbr_core import (  # noqa: E402
    BootstrapPriorState,
    PRIOR_STATE,
    patch_swa_runner_controller,
)
from src.constants import COUNTRY_LANG  # noqa: E402
from src.data import load_multitp_dataset  # noqa: E402
from src.model import setup_seeds  # noqa: E402
from src.personas import SUPPORTED_COUNTRIES, build_country_personas  # noqa: E402
from src.swa_runner import run_country_experiment  # noqa: E402

MODEL_NAME = os.environ.get("R2_MODEL", "microsoft/phi-4")
N_SCEN = int(os.environ.get("R2_N_SCENARIOS", "500"))
CAP = os.environ.get("R2_CAP_PER_CAT", "0").strip().lower() not in ("0", "false", "no", "off")
DUMP_IDS = os.environ.get("R2_DUMP_IDS", "1").strip().lower() not in ("0", "false", "no", "off")
COUNTRIES = (
    [c.strip() for c in os.environ["R2_COUNTRIES"].split(",") if c.strip()]
    if "R2_COUNTRIES" in os.environ
    else list(PAPER_20_COUNTRIES)
)

RESULTS_BASE = (
    "/kaggle/working/cultural_alignment/results/exp24_round2/no_oversampling"
    if on_kaggle()
    else str(Path(__file__).parent.parent / "results" / "exp24_round2" / "no_oversampling")
)
WVS_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"


def _load_scenarios_nocap(cfg, country: str, out_dir: Path) -> pd.DataFrame:
    lang = COUNTRY_LANG.get(country, "en")
    dump = str(out_dir / "scenario_ids" / f"{country}_{lang}.csv") if DUMP_IDS else ""
    df = load_multitp_dataset(
        data_base_path=cfg.multitp_data_path,
        lang=lang,
        translator=cfg.multitp_translator,
        suffix=cfg.multitp_suffix,
        n_scenarios=cfg.n_scenarios,
        cap_per_category=CAP,
        dump_ids_path=dump,
    )
    df = df.copy()
    df["lang"] = lang
    return df


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

    rows: List[Dict] = []
    try:
        for ci, country in enumerate(COUNTRIES):
            if country not in SUPPORTED_COUNTRIES:
                continue
            print(f"\n[{ci+1}/{len(COUNTRIES)}] no-cap  |  {country}")
            scen = _load_scenarios_nocap(cfg, country, out_dir)
            personas = build_country_personas(country, wvs_path=WVS_PATH)

            PRIOR_STATE.clear()
            PRIOR_STATE[country] = BootstrapPriorState()
            patch_swa_runner_controller()

            t0 = time.time()
            results_df, summary = run_country_experiment(
                model, tokenizer, country, personas, scen, cfg,
            )
            dt = time.time() - t0
            results_df.to_csv(out_dir / f"no_oversampling_swa_{country}.csv", index=False)

            a = summary["alignment"]
            rows.append({
                "method":      "swa_dpbr_no_cap",
                "country":     country,
                "n_scenarios": len(results_df),
                "elapsed_sec": dt,
                "mis":         a.get("mis",         float("nan")),
                "jsd":         a.get("jsd",         float("nan")),
                "pearson_r":   a.get("pearson_r",   float("nan")),
                "mae":         a.get("mae",         float("nan")),
                "rmse":        a.get("rmse",        float("nan")),
            })
            pd.DataFrame(rows).to_csv(out_dir / "no_oversampling_partial.csv", index=False)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_summary(rows, out_dir, "no_oversampling_summary.csv")


if __name__ == "__main__":
    main()
