#!/usr/bin/env python3
"""Dump the exact MultiTP scenario slice (Prompt list + seed) for every
country in the 20-country paper set.

Writes ``results/exp24_round2/scenario_ids/<iso3>_<lang>.csv`` containing
the five core columns per scenario (Prompt, phenomenon_category,
preferred_on_right, n_left, n_right). Combined with the seed (42 by default)
this lets reviewers reproduce the evaluation slice bit-for-bit without
needing to reload any model.

Usage:
    python exp_paper/round2/phase1_posthoc/dump_scenario_seeds.py
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

import os
from pathlib import Path

# _r2_bootstrap() above already put the repo root on sys.path.
from exp_paper.paper_countries import PAPER_20_COUNTRIES
from src.constants import COUNTRY_LANG
from src.data import load_multitp_dataset

MULTITP_PATH = os.environ.get(
    "R2_MULTITP_PATH",
    "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data",
)
N_SCEN = int(os.environ.get("R2_N_SCENARIOS", "500"))
SEED = int(os.environ.get("R2_SEED", "42"))
_DEFAULT_OUT = (
    "/kaggle/working/cultural_alignment/results/exp24_round2/scenario_ids"
    if os.path.isdir("/kaggle/input")
    else "results/exp24_round2/scenario_ids"
)
OUT_DIR = Path(os.environ.get("R2_OUT_DIR", _DEFAULT_OUT))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[DUMP] out_dir={OUT_DIR}  seed={SEED}  n={N_SCEN}")

    for country in PAPER_20_COUNTRIES:
        lang = COUNTRY_LANG.get(country, "en")
        out = OUT_DIR / f"{country}_{lang}.csv"
        try:
            load_multitp_dataset(
                data_base_path=MULTITP_PATH,
                lang=lang,
                translator="google",
                suffix="",
                n_scenarios=N_SCEN,
                seed=SEED,
                cap_per_category=True,
                dump_ids_path=str(out),
            )
        except Exception as exc:
            print(f"[ERROR] {country}/{lang}: {exc}")


if __name__ == "__main__":
    main()
