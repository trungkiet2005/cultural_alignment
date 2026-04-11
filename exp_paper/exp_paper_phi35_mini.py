#!/usr/bin/env python3
"""
Paper sweep — Phi-3.5-mini-instruct — EXP-24 (DPBR), 20 countries
=================================================================

Model : unsloth/Phi-3.5-mini-instruct
Method: EXP-24 DPBR

Kaggle:
    !python exp_paper/exp_paper_phi35_mini.py
"""

import os
import subprocess
import sys

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("UNSLOTH_DISABLE_AUTO_COMPILE", "1")
os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")

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
        "pip install -q bitsandbytes scipy tqdm",
        'pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer',
        "pip install --no-deps unsloth_zoo bitsandbytes accelerate peft trl triton unsloth",
        "pip install transformers==4.56.2",
        "pip install --no-deps trl==0.22.2",
    ]:
        subprocess.run(cmd, shell=True, check=False)


_ensure_repo()
_install_deps()

MODEL_NAME = "unsloth/Phi-3.5-mini-instruct"
MODEL_SHORT = "phi35_mini"

from exp_paper.paper_countries import PAPER_20_COUNTRIES, RESULTS_BASE_EXP24_20C  # noqa: E402
from exp_model._base_dpbr import run_for_model  # noqa: E402

if __name__ == "__main__":
    run_for_model(
        MODEL_NAME,
        MODEL_SHORT,
        target_countries=PAPER_20_COUNTRIES,
        results_base=RESULTS_BASE_EXP24_20C,
    )
