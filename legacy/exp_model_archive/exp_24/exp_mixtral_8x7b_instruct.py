#!/usr/bin/env python3
"""
EXP-24 Dual-Pass Bootstrap IS — Mixtral 8x7B Instruct v0.1 (4-bit)
====================================================================

Model  : unsloth/Mixtral-8x7B-Instruct-v0.1-unsloth-bnb-4bit
Profile: pypi  (bitsandbytes + PyPI unsloth)
Method : Dual-Pass Bootstrap IS Reliability (DPBR) — identical to EXP-24

Usage on Kaggle
---------------
    !python exp_model/exp_24/exp_mixtral_8x7b_instruct.py

VRAM: MoE 8x7B @ 4-bit still needs substantial GPU memory (often 24GB+ effective). Prefer A100 / H100.

Reference: widely used open MoE instruct model (Mistral family).
"""

import os, sys, subprocess

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("UNSLOTH_DISABLE_AUTO_COMPILE", "1")
os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")

REPO_URL        = "https://github.com/trungkiet2005/cultural_alignment.git"
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
        'pip install -q bitsandbytes scipy tqdm sentencepiece protobuf',
        'pip install --upgrade --no-deps unsloth',
        'pip install -q unsloth_zoo',
        'pip install --quiet "datasets>=3.4.1,<4.4.0"',
    ]:
        subprocess.run(cmd, shell=True, check=False)


_ensure_repo()
_install_deps()

MODEL_NAME  = "unsloth/Mixtral-8x7B-Instruct-v0.1-unsloth-bnb-4bit"
MODEL_SHORT = "mixtral_8x7b_instruct"

from exp_model._base_dpbr import run_for_model  # noqa: E402

if __name__ == "__main__":
    run_for_model(MODEL_NAME, MODEL_SHORT)
