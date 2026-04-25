#!/usr/bin/env python3
"""
Paper sweep — Qwen2.5-7B-Instruct (HF native, bf16) — EXP-24 (DPBR), 20 countries
=====================================================================================

*** KAGGLE OFFLINE VERSION (No Internet) ***

Environment:
    GPU  : NVIDIA RTX 6000 (96 GB VRAM)
    Mode : Internet OFF — no git clone, no pip install from PyPI, no HF download

Setup:
    1. Upload the entire `cultural_alignment` project as a **Kaggle Dataset**
       (e.g. dataset slug: `your-username/cultural-alignment`).
       → Appears at: /kaggle/input/cultural-alignment/...

    2. Add the **model** as a Kaggle Model input.
       → /kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1

    3. Ensure the `multitp-data` dataset is also added as input.
       → /kaggle/input/mutltitp-data/data/data/...
       (Same dataset used by all other exp_paper scripts)

    4. In the Kaggle notebook, paste this entire file content into a cell and Run.
       Or: !python /kaggle/input/cultural-alignment/exp_paper/models/exp_paper_kaggle_qwen25_7b.py
"""

import gc
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  1. PATHS — Update these to match your Kaggle input slugs                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# --- Project repo (uploaded as Kaggle dataset) ---
PROJECT_DATASET_DIR = "/kaggle/input/cultural-alignment"

# --- Model weights (Kaggle Model input) ---
MODEL_LOCAL_PATH = "/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1"

# --- Data inputs (multitp-data dataset — MUST match _base_dpbr.py hardcoded paths) ---
# _base_dpbr.py hardcodes these paths; we monkey-patch them after import.
MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH     = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
HUMAN_AMCE_PATH   = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"

# --- Working directory (writable) ---
WORK_DIR = "/kaggle/working/cultural_alignment"

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  2. ENVIRONMENT SETUP (offline-safe — no pip, no git, no internet)         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
os.environ["UNSLOTH_DISABLE_AUTO_COMPILE"] = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Disable any HF Hub download attempts (model is local)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _on_kaggle() -> bool:
    return os.path.isdir("/kaggle/working")


def _setup_project() -> str:
    """
    Copy the project from the read-only Kaggle input to a writable working dir.
    Kaggle input datasets are mounted read-only, but the experiment writes
    results, CSVs, and flags — so we need a writable copy.
    """
    if os.path.isdir(WORK_DIR) and os.path.isfile(
        os.path.join(WORK_DIR, "src", "controller.py")
    ):
        print(f"[SETUP] Working dir already exists: {WORK_DIR}")
    else:
        if not os.path.isdir(PROJECT_DATASET_DIR):
            raise RuntimeError(
                f"[SETUP] Project dataset not found at {PROJECT_DATASET_DIR}\n"
                f"Make sure you uploaded the cultural_alignment repo as a Kaggle dataset "
                f"and update PROJECT_DATASET_DIR in this script."
            )
        print(f"[SETUP] Copying project from {PROJECT_DATASET_DIR} → {WORK_DIR} ...")
        shutil.copytree(PROJECT_DATASET_DIR, WORK_DIR, dirs_exist_ok=True)
        print(f"[SETUP] Copy done.")

    os.chdir(WORK_DIR)
    sys.path.insert(0, WORK_DIR)
    return WORK_DIR


def _install_offline_deps() -> None:
    """
    On Kaggle GPU images, most deps (torch, transformers, scipy, tqdm, etc.)
    are pre-installed. We only try quiet no-internet installs as fallback.
    """
    if not _on_kaggle():
        return
    for cmd in [
        "pip install -q --no-deps --no-index scipy tqdm sentencepiece protobuf 2>/dev/null || true",
    ]:
        subprocess.run(cmd, shell=True, check=False)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  3. BOOTSTRAP                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("=" * 70)
print("  KAGGLE OFFLINE — Qwen2.5-7B-Instruct (HF native, bf16)")
print("  RTX 6000 · 96 GB VRAM · No Internet")
print("=" * 70)

_setup_project()
_install_offline_deps()

# --- Validate model path ---
MODEL_LOCAL_PATH_RESOLVED = MODEL_LOCAL_PATH  # default

if not os.path.isdir(MODEL_LOCAL_PATH):
    # Try common Kaggle model input patterns
    _candidates = [
        MODEL_LOCAL_PATH,
        f"{MODEL_LOCAL_PATH}/transformers/default/1",
        f"{MODEL_LOCAL_PATH}/pytorch/default/1",
    ]
    _found = None
    for c in _candidates:
        if os.path.isdir(c) and os.path.isfile(os.path.join(c, "config.json")):
            _found = c
            break
    if _found:
        MODEL_LOCAL_PATH_RESOLVED = _found
    else:
        print(f"[WARNING] Model path not found: {MODEL_LOCAL_PATH}")
        print(f"[WARNING] Listing /kaggle/input/ to help you find it:")
        if os.path.isdir("/kaggle/input"):
            for p in sorted(Path("/kaggle/input").iterdir()):
                print(f"  {p}")
                if p.is_dir():
                    for sub in sorted(p.iterdir()):
                        print(f"    {sub}")
        raise RuntimeError(
            f"Model weights not found at {MODEL_LOCAL_PATH}.\n"
            f"Update MODEL_LOCAL_PATH in this script to the correct Kaggle input path."
        )
else:
    # Check if config.json is directly here or in a subdirectory
    if os.path.isfile(os.path.join(MODEL_LOCAL_PATH, "config.json")):
        MODEL_LOCAL_PATH_RESOLVED = MODEL_LOCAL_PATH
    else:
        # Search one level deeper
        _found = None
        for sub in Path(MODEL_LOCAL_PATH).rglob("config.json"):
            _found = str(sub.parent)
            break
        if _found:
            MODEL_LOCAL_PATH_RESOLVED = _found

print(f"[SETUP] Model path resolved: {MODEL_LOCAL_PATH_RESOLVED}")
print(f"[SETUP] Project dir: {WORK_DIR}")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  4. IMPORT & PATCH DATA PATHS                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Use the LOCAL model path (not a HuggingFace repo id)
MODEL_NAME = MODEL_LOCAL_PATH_RESOLVED
MODEL_SHORT = "kaggle_qwen25_7b_bf16"

from exp_paper.paper_countries import PAPER_20_COUNTRIES, RESULTS_BASE_EXP24_20C  # noqa: E402
from exp_model._base_dpbr import run_for_model  # noqa: E402

# Monkey-patch _base_dpbr's hardcoded data paths to match THIS notebook's inputs.
# These module-level constants are used inside _build_cfg() which is called by run_for_model().
import exp_model._base_dpbr as _dpbr  # noqa: E402

_dpbr.MULTITP_DATA_PATH = MULTITP_DATA_PATH
_dpbr.WVS_DATA_PATH = WVS_DATA_PATH
_dpbr.HUMAN_AMCE_PATH = HUMAN_AMCE_PATH
print(f"[SETUP] Data paths patched in _base_dpbr:")
print(f"  MULTITP : {_dpbr.MULTITP_DATA_PATH}")
print(f"  WVS     : {_dpbr.WVS_DATA_PATH}")
print(f"  AMCE    : {_dpbr.HUMAN_AMCE_PATH}")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  5. RUN EXPERIMENT                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    print(f"\n[RUN] Starting EXP-24 DPBR with local model: {MODEL_NAME}")
    print(f"[RUN] Countries: {PAPER_20_COUNTRIES}")
    print(f"[RUN] Results base: {RESULTS_BASE_EXP24_20C}")

    run_for_model(
        MODEL_NAME,
        MODEL_SHORT,
        load_in_4bit=False,       # bf16 full precision — RTX 6000 has 96 GB VRAM
        use_hf_native=True,       # HF transformers only (no Unsloth needed)
        target_countries=PAPER_20_COUNTRIES,
        results_base=RESULTS_BASE_EXP24_20C,
    )
