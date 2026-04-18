#!/usr/bin/env python3
"""Round-2 Baseline #3 -- DIFFPO-adapted binary-decision steering.

Implements the DIFFPO spirit (black-box, sentence-level alignment nudge)
adapted to MultiTP's binary A/B format:

    p_aligned = (1 - alpha) * p_vanilla + alpha * p_target

where p_target is built directly from the country's *public* human AMCE
(see src/diffpo_binary_baseline.py). The mixing weight alpha is fit per
country on a 25% calibration split, then applied to the 75% test split.

Because this baseline *consumes* the human AMCE at inference time, it has
strictly more information than SWA-DPBR -- the comparison is deliberately
favourable to the baseline.

Kaggle:
    !python exp_paper/exp_r2_baseline_diffpo.py
"""

from __future__ import annotations

import os
from pathlib import Path

from exp_paper._r2_common import (
    build_cfg,
    ensure_repo,
    load_model_timed,
    on_kaggle,
    run_country_loop,
    save_summary,
)

ensure_repo()

from exp_paper.paper_runtime import configure_paper_env, install_paper_kaggle_deps  # noqa: E402

configure_paper_env()
from src.hf_env import apply_hf_credentials  # noqa: E402

apply_hf_credentials()
install_paper_kaggle_deps()

os.environ.setdefault("MORAL_MODEL_BACKEND", os.environ.get("R2_BACKEND", "vllm"))

from exp_paper.paper_countries import PAPER_20_COUNTRIES  # noqa: E402
from src.diffpo_binary_baseline import run_baseline_diffpo_binary  # noqa: E402
from src.model import setup_seeds  # noqa: E402

MODEL_NAME = os.environ.get("R2_MODEL", "microsoft/phi-4")
N_SCEN = int(os.environ.get("R2_N_SCENARIOS", "500"))
CAL_FRAC = float(os.environ.get("R2_CAL_FRAC", "0.25"))
COUNTRIES = (
    [c.strip() for c in os.environ["R2_COUNTRIES"].split(",") if c.strip()]
    if "R2_COUNTRIES" in os.environ
    else list(PAPER_20_COUNTRIES)
)

RESULTS_BASE = (
    "/kaggle/working/cultural_alignment/results/exp24_round2/diffpo_binary"
    if on_kaggle()
    else str(Path(__file__).parent.parent / "results" / "exp24_round2" / "diffpo_binary")
)


def _extras(out):
    p = out.get("diffpo_params", {})
    return {
        "diffpo_alpha": p.get("alpha", float("nan")),
        "cal_loss":     p.get("cal_loss", float("nan")),
        "n_cal":        p.get("n_cal", 0),
        "n_test":       p.get("n_test", 0),
    }


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

    def _runner(m, t, scen, country, cfg):
        return run_baseline_diffpo_binary(
            m, t, scen, country, cfg, cal_frac=CAL_FRAC,
        )

    rows = run_country_loop(
        model=model, tokenizer=tokenizer, cfg=cfg,
        countries=COUNTRIES, runner_fn=_runner,
        method_tag="diffpo_binary", out_dir=out_dir,
        row_extras_fn=_extras,
    )
    save_summary(rows, out_dir, "diffpo_binary_summary.csv")


if __name__ == "__main__":
    main()
