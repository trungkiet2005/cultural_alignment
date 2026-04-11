#!/usr/bin/env python3
"""
Paper sweep — Qwen2.5-7B-Instruct (HF upstream, bf16) — EXP-24 (DPBR), 20 countries
=====================================================================================

Model : Qwen/Qwen2.5-7B-Instruct (HF native, no 4-bit; aligns with EXP-24-HF_QWEN25_7B_BF16)
Method: EXP-24 DPBR

Kaggle:
    !python exp_paper/exp_paper_hf_qwen25_7b_bf16.py

VRAM: ~15–18 GB bf16 on 80 GB–class GPUs.
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
        sys.path.insert(0, here)
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
from src.hf_env import apply_hf_credentials  # noqa: E402

_install_deps()
apply_hf_credentials()

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_SHORT = "hf_qwen25_7b_bf16"

from exp_paper.paper_countries import PAPER_20_COUNTRIES, RESULTS_BASE_EXP24_20C  # noqa: E402
from exp_model._base_dpbr import run_for_model  # noqa: E402

if __name__ == "__main__":
    run_for_model(
        MODEL_NAME,
        MODEL_SHORT,
        load_in_4bit=False,
        use_hf_native=True,
        target_countries=PAPER_20_COUNTRIES,
        results_base=RESULTS_BASE_EXP24_20C,
    )
