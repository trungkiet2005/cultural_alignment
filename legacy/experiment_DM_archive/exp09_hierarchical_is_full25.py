#!/usr/bin/env python3
"""
EXP-09 full-25 countries runner.

Reuse the exact hierarchical IS pipeline from exp09_hierarchical_is.py
and only override country coverage + output paths.
"""

import experiment_DM.exp09_hierarchical_is as exp09
from src.constants import COUNTRY_FULL_NAMES


# Keep only the 25-country benchmark set (exclude legacy non-WVS targets).
_LEGACY_EXCLUDED = {"SAU", "FRA", "POL", "ZAF", "SWE"}
exp09.TARGET_COUNTRIES = [
    c for c in COUNTRY_FULL_NAMES.keys()
    if c not in _LEGACY_EXCLUDED
]

# Write to a separate result folder to avoid overriding EXP-09 5-country outputs.
exp09.EXP_NAME = "hierarchical_is_full25"
exp09.SWA_ROOT = f"/kaggle/working/cultural_alignment/results/{exp09.EXP_NAME}/swa"
exp09.CMP_ROOT = f"/kaggle/working/cultural_alignment/results/{exp09.EXP_NAME}/compare"


if __name__ == "__main__":
    print(f"[EXP-09] Running full-25 countries: {len(exp09.TARGET_COUNTRIES)} targets")
    exp09.main()
