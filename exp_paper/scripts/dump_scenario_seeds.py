#!/usr/bin/env python3
"""Dump the exact MultiTP scenario slice (Prompt list + seed) for every
country in the 20-country paper set.

Writes ``results/exp24_round2/scenario_ids/<iso3>_<lang>.csv`` containing
the five core columns per scenario (Prompt, phenomenon_category,
preferred_on_right, n_left, n_right). Combined with the seed (42 by default)
this lets reviewers reproduce the evaluation slice bit-for-bit without
needing to reload any model.

Usage:
    python exp_paper/scripts/dump_scenario_seeds.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the repo root importable when running standalone from a nested dir.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from exp_paper.paper_countries import PAPER_20_COUNTRIES
from src.constants import COUNTRY_LANG
from src.data import load_multitp_dataset

MULTITP_PATH = os.environ.get(
    "R2_MULTITP_PATH",
    "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data",
)
N_SCEN = int(os.environ.get("R2_N_SCENARIOS", "500"))
SEED = int(os.environ.get("R2_SEED", "42"))
OUT_DIR = Path(os.environ.get(
    "R2_OUT_DIR",
    str(_REPO / "results" / "exp24_round2" / "scenario_ids"),
))


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
