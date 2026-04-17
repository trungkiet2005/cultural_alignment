#!/usr/bin/env python3
"""
EXP-24: print effective DPBR hyperparameters from environment.

Usage:
    python experiment_DM/exp24_ablation_env.py

Set EXP24_VAR_SCALE / EXP24_K_HALF in the shell *before* running experiments
so exp24_dpbr_core picks them up at import time. See docs/exp24_reproducibility.md.
"""

from __future__ import annotations

import os
import sys

# Ensure repo root on path when run as script
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from experiment_DM.exp24_dpbr_core import K_HALF, VAR_SCALE  # noqa: E402


def main() -> None:
    print("EXP-24 DPBR effective constants (read at import of exp24_dpbr_core):")
    print(f"  EXP24_VAR_SCALE env: {os.environ.get('EXP24_VAR_SCALE', '<unset>')}")
    print(f"  EXP24_K_HALF   env: {os.environ.get('EXP24_K_HALF', '<unset>')}")
    print(f"  VAR_SCALE  = {VAR_SCALE}")
    print(f"  K_HALF     = {K_HALF}")
    print(f"  Total IS K = {2 * K_HALF} (match EXP-09 K_samples=128)")
    print()
    print("Example (bash):  export EXP24_VAR_SCALE=0.08 && python exp_model/exp_24/exp_phi_4.py")


if __name__ == "__main__":
    main()
