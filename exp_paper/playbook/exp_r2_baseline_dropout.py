#!/usr/bin/env python3
"""Round-2 Baseline #1 -- MC-Dropout inference-time calibration.

Runs the :mod:`src.mc_dropout_runner` baseline on Phi-4 (14B) across the
20 paper countries, outputs per-country AMCE / MIS so it can be dropped into
Table~\ref{tab:main_macro_summary} as an additional baseline row.

Kaggle:
    !python exp_paper/playbook/exp_r2_baseline_dropout.py

Env overrides:
    R2_MC_T            number of MC passes (default: 8)
    R2_MC_P            dropout rate override (default: 0.10; None to keep model-default)
    R2_MODEL           HF id (default: microsoft/phi-4)
    R2_COUNTRIES       comma-separated ISO3 (default: all 20 paper countries)
    R2_N_SCENARIOS     per-country scenario count (default: 500)
    R2_BACKEND         hf_native (default, required for dropout hooks) | unsloth
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Self-bootstrap — works when copy-pasted into a fresh Kaggle notebook cell.
# Clones the repo on Kaggle if not already on sys.path, then adds it. Safe to
# run multiple times (idempotent: detects src/controller.py in cwd).
# ─────────────────────────────────────────────────────────────────────────────
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _kaggle_setup import bootstrap_offline, zip_outputs as _zip_outputs

bootstrap_offline()
# Set backend BEFORE paper_runtime is imported so install_paper_kaggle_deps()
# picks the correct pip branch (vLLM vs Unsloth).
_os.environ.setdefault("MORAL_MODEL_BACKEND", _os.environ.get("R2_BACKEND", "hf_native"))
import os
from pathlib import Path

from exp_paper._r2_common import (
    build_cfg,
    load_model_timed,
    on_kaggle,
    run_country_loop,
    save_summary,
)
from exp_paper.paper_runtime import configure_paper_env, install_paper_kaggle_deps  # noqa: E402

configure_paper_env()
from src.hf_env import apply_hf_credentials  # noqa: E402

apply_hf_credentials()
install_paper_kaggle_deps()

# MC-Dropout needs Python dropout hooks, which vLLM does not expose. Force
# HF-native unless the user explicitly asks for another backend.
from exp_paper.paper_countries import PAPER_20_COUNTRIES  # noqa: E402
from src.mc_dropout_runner import run_baseline_mc_dropout  # noqa: E402
from src.model import setup_seeds  # noqa: E402

MODEL_NAME = os.environ.get("R2_MODEL", "microsoft/phi-4")
N_SCEN = int(os.environ.get("R2_N_SCENARIOS", "500"))
T_SAMPLES = int(os.environ.get("R2_MC_T", "8"))
P_OVERRIDE = os.environ.get("R2_MC_P", "0.10")
DROPOUT_P = None if P_OVERRIDE in ("none", "None", "") else float(P_OVERRIDE)

COUNTRIES = (
    [c.strip() for c in os.environ["R2_COUNTRIES"].split(",") if c.strip()]
    if "R2_COUNTRIES" in os.environ
    else list(PAPER_20_COUNTRIES)
)

RESULTS_BASE = (
    "/kaggle/working/cultural_alignment/results/exp24_round2/mc_dropout"
    if on_kaggle()
    else str(Path(__file__).parent.parent / "results" / "exp24_round2" / "mc_dropout")
)


def main() -> None:
    setup_seeds(42)
    out_dir = Path(RESULTS_BASE)

    cfg = build_cfg(
        MODEL_NAME, RESULTS_BASE, COUNTRIES,
        n_scenarios=N_SCEN, load_in_4bit=False,
    )
    backend = os.environ.get("MORAL_MODEL_BACKEND", "hf_native").strip().lower()
    model, tokenizer = load_model_timed(
        MODEL_NAME, backend=backend, load_in_4bit=False,
    )

    def _runner(model, tokenizer, scen, country, cfg):
        return run_baseline_mc_dropout(
            model, tokenizer, scen, country, cfg,
            T=T_SAMPLES, dropout_p=DROPOUT_P,
        )

    rows = run_country_loop(
        model=model, tokenizer=tokenizer, cfg=cfg,
        countries=COUNTRIES, runner_fn=_runner,
        method_tag=f"mc_dropout_T{T_SAMPLES}_p{DROPOUT_P}", out_dir=out_dir,
    )
    save_summary(rows, out_dir, f"mc_dropout_summary_T{T_SAMPLES}_p{DROPOUT_P}.csv")


if __name__ == "__main__":
    main()
    try:
        _zip_outputs(RESULTS_BASE if 'RESULTS_BASE' in globals() else OUT_DIR if 'OUT_DIR' in globals() else '.')
    except Exception as _e:
        print(f'[ZIP] failed: {_e}')
