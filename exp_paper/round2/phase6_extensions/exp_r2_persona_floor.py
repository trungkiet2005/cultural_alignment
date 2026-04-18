#!/usr/bin/env python3
"""Minority-protection safeguard: per-persona utility floor experiment.

The paper's §Broader impact discusses a \"per-persona floor\" that clamps
each persona's post-correction Prospect-Theory utility from below at
$-\\texttt{floor}$, preventing any single persona's preference from being
driven arbitrarily negative by a correction that favours the rest of the
panel. This script runs the actual safeguard experiment.

Controller hook: we added ``EXP24_PERSONA_FLOOR`` to
``experiment_DM/exp24_dpbr_core.py``. When > 0 it applies
    v_per_agent = torch.clamp(v_per_agent, min=-PERSONA_FLOOR)
inside the IS pass, everything else unchanged.

Sweep: floor $\\in \\{0.0, 0.5, 1.0, 2.0\\}$ (0 = off, matches default) on
Phi-4 × 3-country panel. Per scenario we additionally track the minimum
per-persona utility and its post-correction rank so the paper can quote
\"fraction of scenarios where the worst persona was protected\" as
evidence the safeguard works as intended.

Kaggle (~2 h on H100):
    !python exp_paper/round2/phase6_extensions/exp_r2_persona_floor.py

Env overrides:
    R2_MODEL         HF id (default: microsoft/phi-4)
    R2_COUNTRIES     comma ISO3 list (default: USA,VNM,DEU)
    R2_N_SCENARIOS   per-country (default: 250)
    R2_FLOOR_GRID    comma list (default: 0,0.5,1.0,2.0)
    R2_BACKEND       vllm (default) | hf_native | unsloth
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Self-bootstrap — works when copy-pasted into a fresh Kaggle notebook cell.
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

# IMPORTANT: import AFTER env is configured — PERSONA_FLOOR is captured at
# module-import time from the env var. We'll reassign it on the fly below.
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
FLOOR_GRID = [
    float(s.strip())
    for s in os.environ.get("R2_FLOOR_GRID", "0,0.5,1.0,2.0").split(",")
    if s.strip()
]

RESULTS_BASE = (
    "/kaggle/working/cultural_alignment/results/exp24_round2/persona_floor"
    if on_kaggle()
    else str(Path(__file__).parent.parent / "results" / "exp24_round2" / "persona_floor")
)
WVS_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"


def _reset_prior(country: str) -> None:
    PRIOR_STATE.clear()
    PRIOR_STATE[country] = BootstrapPriorState()


def _run_one_floor(model, tokenizer, cfg, country, scen, floor: float) -> Dict:
    """Run a single (floor, country) cell. ``floor == 0`` disables the clamp."""
    _snap_floor = _dpbr.PERSONA_FLOOR
    _dpbr.PERSONA_FLOOR = float(floor)
    _reset_prior(country)
    patch_swa_runner_controller()

    personas = build_country_personas(country, wvs_path=WVS_PATH)
    t0 = time.time()
    try:
        results_df, summary = run_country_experiment(
            model, tokenizer, country, personas, scen, cfg,
        )
    finally:
        _dpbr.PERSONA_FLOOR = _snap_floor

    # Per-scenario persona-reward spread — proxy for minority-impact magnitude.
    if "agent_reward_min" in results_df.columns and "agent_reward_max" in results_df.columns:
        worst = float(results_df["agent_reward_min"].mean())
        worst_std = float(results_df["agent_reward_min"].std(ddof=1)) \
                    if len(results_df) >= 2 else float("nan")
        best  = float(results_df["agent_reward_max"].mean())
        spread = float((results_df["agent_reward_max"] -
                        results_df["agent_reward_min"]).mean())
    else:
        worst = worst_std = best = spread = float("nan")

    a = summary["alignment"]
    return {
        "floor":            float(floor),
        "country":          country,
        "n_scenarios":      len(results_df),
        "elapsed_sec":      time.time() - t0,
        "mis":              a.get("mis", float("nan")),
        "jsd":              a.get("jsd", float("nan")),
        "pearson_r":        a.get("pearson_r", float("nan")),
        "flip_rate":        summary.get("flip_rate", float("nan")),
        "mean_worst_reward": worst,
        "std_worst_reward": worst_std,
        "mean_best_reward": best,
        "mean_persona_spread": spread,
    }


def main() -> None:
    setup_seeds(42)
    out_dir = Path(RESULTS_BASE)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(MODEL_NAME, RESULTS_BASE, COUNTRIES,
                    n_scenarios=N_SCEN, load_in_4bit=False)
    backend = os.environ.get("MORAL_MODEL_BACKEND", "vllm").strip().lower()
    model, tokenizer = load_model_timed(
        MODEL_NAME, backend=backend, load_in_4bit=False,
    )

    scen_cache: Dict[str, pd.DataFrame] = {}
    for c in COUNTRIES:
        if c not in SUPPORTED_COUNTRIES:
            print(f"[SKIP] {c}")
            continue
        scen_cache[c] = load_scenarios(cfg, c)

    rows: List[Dict] = []
    for floor in FLOOR_GRID:
        print(f"\n{'#' * 72}\n# persona_floor = {floor}\n{'#' * 72}")
        for country in COUNTRIES:
            if country not in scen_cache:
                continue
            try:
                row = _run_one_floor(model, tokenizer, cfg, country,
                                     scen_cache[country], floor)
                rows.append(row)
                pd.DataFrame(rows).to_csv(
                    out_dir / "persona_floor_partial.csv", index=False)
                a = row
                print(f"  ✓ floor={floor} {country}  MIS={a['mis']:.4f}  "
                      f"worst_r={a['mean_worst_reward']:+.4f}  "
                      f"spread={a['mean_persona_spread']:.4f}  "
                      f"({a['elapsed_sec']:.0f}s)")
            except Exception as exc:
                print(f"[ERROR] floor={floor} {country}: {exc}")
                rows.append({"floor": floor, "country": country,
                             "error": str(exc)[:500]})
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    save_summary(rows, out_dir, "persona_floor_summary.csv")


if __name__ == "__main__":
    main()
