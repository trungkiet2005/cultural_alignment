#!/usr/bin/env python3
"""
EXP-24 DPBR — Meta-Llama-3-8B-Instruct (HF upstream, bf16)
============================================================

Model  : meta-llama/Meta-Llama-3-8B-Instruct  (no Unsloth *_bnb_4bit suffix)
Load   : load_in_4bit=False  — Hugging Face ``AutoModelForCausalLM`` + bf16 (no Unsloth)
Method : Dual-Pass Bootstrap IS Reliability (DPBR) — same as other EXP-24

Usage (Kaggle)
--------------
    !python exp_model/exp_24/hf_full/exp_meta_llama3_8b_instruct.py

Requires Hugging Face access to gated Llama 3 weights (HF_TOKEN / Kaggle secret).

VRAM: ~16–20 GB bf16 on a single 80 GB GPU is typical; T4 16 GB will OOM.
"""

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

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_SHORT = "hf_llama3_8b_bf16"

from exp_model._base_dpbr import run_for_model  # noqa: E402

if __name__ == "__main__":
    run_for_model(MODEL_NAME, MODEL_SHORT, load_in_4bit=False, use_hf_native=True)
