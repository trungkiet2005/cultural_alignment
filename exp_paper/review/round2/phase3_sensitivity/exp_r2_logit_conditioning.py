#!/usr/bin/env python3
"""Round-2 Reviewer W10 -- logit-conditioning diagnostic.

Runs a vanilla forward pass per scenario for each of the 20 countries on a
target model, computes per-scenario decision-margin / decision-entropy /
|raw_logit_gap|, and saves both per-scenario and per-country aggregates so
we can show a scatter of "logit conditioning" vs "MIS improvement" in the
paper appendix.

Uses :func:`src.logit_conditioning.diagnose_country` (no personas, no IS;
this is purely a diagnostic about the base model).

Kaggle:
    !python exp_paper/review/round2/phase3_sensitivity/exp_r2_logit_conditioning.py

Env overrides:
    R2_MODEL          (default: microsoft/phi-4)
    R2_COUNTRIES      comma ISO3 list (default: 20 paper countries)
    R2_N_SCENARIOS    (default: 300 -- we only need enough for a stable mean)
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
import json
import os
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

from exp_paper.paper_countries import PAPER_20_COUNTRIES  # noqa: E402
from src.logit_conditioning import diagnose_country  # noqa: E402
from src.model import setup_seeds  # noqa: E402

MODEL_NAME = os.environ.get("R2_MODEL", "microsoft/phi-4")
N_SCEN = int(os.environ.get("R2_N_SCENARIOS", "300"))
COUNTRIES = (
    [c.strip() for c in os.environ["R2_COUNTRIES"].split(",") if c.strip()]
    if "R2_COUNTRIES" in os.environ
    else list(PAPER_20_COUNTRIES)
)

RESULTS_BASE = (
    "/kaggle/working/cultural_alignment/results/exp24_round2/logit_conditioning"
    if on_kaggle()
    else str(Path(__file__).parent.parent / "results" / "exp24_round2" / "logit_conditioning")
)


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
    per_scenario_frames: List[pd.DataFrame] = []
    for ci, country in enumerate(COUNTRIES):
        print(f"\n[{ci+1}/{len(COUNTRIES)}] {country}")
        scen = load_scenarios(cfg, country)
        out = diagnose_country(model, tokenizer, scen, country, cfg)
        per = out["results_df"]
        per_scenario_frames.append(per)
        per.to_csv(out_dir / f"logit_cond_per_scenario_{country}.csv", index=False)

        s = out["summary"]
        rows.append({
            "country":             country,
            "n_scenarios":         len(per),
            "mean_entropy":        s["mean_entropy"],
            "mean_margin":         s["mean_margin"],
            "median_margin":       s["median_margin"],
            "std_margin":          s["std_margin"],
            "mean_abs_gap":        s["mean_abs_gap"],
            "frac_margin_lt_0.1":  s["frac_margin_lt_0.1"],
            "frac_margin_gt_0.5":  s["frac_margin_gt_0.5"],
        })
        with open(out_dir / f"logit_cond_by_cat_{country}.json", "w", encoding="utf-8") as fh:
            json.dump(s["by_category"], fh, indent=2, default=str)

    save_summary(rows, out_dir, "logit_conditioning_per_country.csv")
    if per_scenario_frames:
        pd.concat(per_scenario_frames, ignore_index=True).to_csv(
            out_dir / "logit_cond_per_scenario_all.csv", index=False,
        )
        print(f"[SAVED] {out_dir / 'logit_cond_per_scenario_all.csv'}")


if __name__ == "__main__":
    main()
