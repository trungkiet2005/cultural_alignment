#!/usr/bin/env python3
"""
EXP-24 DPBR — Phi-3-mini-4k-Instruct (HF upstream, bf16)
========================================================

Model  : microsoft/Phi-3-mini-4k-instruct
Load   : load_in_4bit=False  — Hugging Face ``AutoModelForCausalLM`` + bf16 (no Unsloth)
Method : Dual-Pass Bootstrap IS Reliability (DPBR)

Usage (Kaggle)
--------------
    !python exp_model/exp_24/hf_full/exp_phi3_mini_4k_instruct.py

Do not paste only the bottom of this file in a notebook: the bootstrap block below
must run first so the repo is cloned and ``exp_model`` is importable.

VRAM: ~8–10 GB bf16 — fits many single-GPU notebooks.
"""

# ============================================================
# Step 0: env bootstrap — MUST run before any ``exp_model`` import.
# On Kaggle, cwd is often /kaggle/working (no package root); we clone + sys.path.
# ============================================================
import os
import subprocess
import sys

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

REPO_URL = "https://github.com/trungkiet2005/cultural_alignment.git"
REPO_DIR_KAGGLE = "/kaggle/working/cultural_alignment"


def _on_kaggle() -> bool:
    return os.path.isdir("/kaggle/working")


def _ensure_repo() -> str:
    here = os.getcwd()
    if os.path.isfile(os.path.join(here, "src", "controller.py")):
        return here
    if not _on_kaggle():
        raise RuntimeError("Not on Kaggle and not inside the repo root.")
    if not os.path.isdir(REPO_DIR_KAGGLE):
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, REPO_DIR_KAGGLE], check=True
        )
    os.chdir(REPO_DIR_KAGGLE)
    sys.path.insert(0, REPO_DIR_KAGGLE)
    return REPO_DIR_KAGGLE


def _install_deps() -> None:
    if not _on_kaggle():
        return
    for cmd in [
        "pip install -q accelerate bitsandbytes scipy tqdm sentencepiece protobuf",
        'pip install --quiet "datasets>=3.4.1,<4.4.0"',
    ]:
        subprocess.run(cmd, shell=True, check=False)


_ensure_repo()
_install_deps()

# ============================================================
# Step 1: model + run
# ============================================================
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
MODEL_SHORT = "hf_phi3_mini_bf16"

from exp_model._base_dpbr import run_for_model  # noqa: E402

if __name__ == "__main__":
    run_for_model(MODEL_NAME, MODEL_SHORT, load_in_4bit=False, use_hf_native=True)
