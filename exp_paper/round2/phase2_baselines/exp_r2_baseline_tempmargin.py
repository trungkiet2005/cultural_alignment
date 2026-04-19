#!/usr/bin/env python3
"""Round-2 Baseline #2 -- per-country temperature / margin scaling.

Fits a per-country scalar T_c (or m_c) on a 25% calibration split of MultiTP
for each of the 20 paper countries, applies it to the remaining 75% test
split, and reports the held-out MIS / r / JSD. Acts as a full standalone
baseline on the same 20-country grid as SWA-DPBR.

Kaggle:
    !python exp_paper/round2/phase2_baselines/exp_r2_baseline_tempmargin.py

Env overrides:
    R2_CALIB_METHOD  temperature | margin  (default: both -- runs two passes)
    R2_CAL_FRAC      calibration split fraction (default: 0.25)
    R2_MODEL         HF id (default: microsoft/phi-4)
    R2_COUNTRIES     comma ISO3 list (default: 20 paper countries)
    R2_N_SCENARIOS   per-country (default: 500)
    R2_BACKEND       vllm (default) | hf_native
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
from exp_paper.paper_countries import PAPER_20_COUNTRIES  # noqa: E402
from src.calibration_baselines import run_baseline_calibration_scaling  # noqa: E402
from src.model import setup_seeds  # noqa: E402

MODEL_NAME = os.environ.get("R2_MODEL", "microsoft/phi-4")
N_SCEN = int(os.environ.get("R2_N_SCENARIOS", "500"))
CAL_FRAC = float(os.environ.get("R2_CAL_FRAC", "0.25"))
METHOD = os.environ.get("R2_CALIB_METHOD", "both").lower()
COUNTRIES = (
    [c.strip() for c in os.environ["R2_COUNTRIES"].split(",") if c.strip()]
    if "R2_COUNTRIES" in os.environ
    else list(PAPER_20_COUNTRIES)
)

RESULTS_BASE = (
    "/kaggle/working/cultural_alignment/results/exp24_round2/tempmargin"
    if on_kaggle()
    else str(Path(__file__).parent.parent / "results" / "exp24_round2" / "tempmargin")
)


def _extras(method: str):
    def _fn(out):
        p = out.get("calib_params", {})
        return {
            "calib_method": method,
            "calib_value":  p.get("value", float("nan")),
            "cal_loss":     p.get("cal_loss", float("nan")),
            "n_cal":        p.get("n_cal", 0),
            "n_test":       p.get("n_test", 0),
        }
    return _fn


def _run_method(model, tokenizer, cfg, method: str, out_dir: Path):
    def _runner(m, t, scen, country, cfg):
        return run_baseline_calibration_scaling(
            m, t, scen, country, cfg, method=method, cal_frac=CAL_FRAC,
        )
    rows = run_country_loop(
        model=model, tokenizer=tokenizer, cfg=cfg,
        countries=COUNTRIES, runner_fn=_runner,
        method_tag=f"calib_{method}", out_dir=out_dir,
        row_extras_fn=_extras(method),
    )
    save_summary(rows, out_dir, f"calib_{method}_summary.csv")


def main() -> None:
    setup_seeds(42)
    cfg = build_cfg(
        MODEL_NAME, RESULTS_BASE, COUNTRIES,
        n_scenarios=N_SCEN, load_in_4bit=False,
    )
    backend = os.environ.get("MORAL_MODEL_BACKEND", "vllm").strip().lower()
    model, tokenizer = load_model_timed(
        MODEL_NAME, backend=backend, load_in_4bit=False,
    )

    out_dir = Path(RESULTS_BASE)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = ("temperature", "margin") if METHOD == "both" else (METHOD,)
    for m in methods:
        print(f"\n{'#' * 72}\n# Calibration-only baseline: method={m}\n{'#' * 72}")
        _run_method(model, tokenizer, cfg, m, out_dir)


if __name__ == "__main__":
    main()
