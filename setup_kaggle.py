#!/usr/bin/env python3
"""Kaggle environment bootstrap: install dependencies and create directories."""

import os
import subprocess
from pathlib import Path


def _run(cmd: str, verbose: bool = False) -> int:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if verbose and r.stdout:
        print(r.stdout.strip())
    if r.returncode != 0 and r.stderr:
        print(r.stderr.strip())
    return r.returncode


def setup():
    """Install dependencies and create working directories on Kaggle."""
    if not os.path.exists("/kaggle/working"):
        print("[SETUP] Not running on Kaggle, skipping.")
        return

    print("[SETUP] Installing dependencies...")
    _run("pip install -q bitsandbytes scipy tqdm matplotlib seaborn")
    _run("pip install --upgrade --no-deps unsloth")
    _run("pip install -q unsloth_zoo")
    _run("pip install --quiet --no-deps --force-reinstall pyarrow")
    _run("pip install --quiet 'datasets>=3.4.1,<4.4.0'")

    work_dir = Path("/kaggle/working/SWA_MPPI")
    data_dir = work_dir / "data"
    results_dir = work_dir / "results"
    for d in [data_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"[SETUP] Working directory: {work_dir}")
    print("[SETUP] Done")


if __name__ == "__main__":
    setup()
