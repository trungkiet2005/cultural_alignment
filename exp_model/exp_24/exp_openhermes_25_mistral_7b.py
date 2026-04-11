#!/usr/bin/env python3
"""
EXP-24 Dual-Pass Bootstrap IS — OpenHermes 2.5 Mistral 7B (4-bit)
================================================================

Model  : unsloth/OpenHermes-2.5-Mistral-7B-bnb-4bit
Profile: pypi  (bitsandbytes + PyPI unsloth)
Method : Dual-Pass Bootstrap IS Reliability (DPBR) — identical to EXP-24

Usage on Kaggle
---------------
    !python exp_model/exp_24/exp_openhermes_25_mistral_7b.py

VRAM: 7B @ 4-bit — fits comfortably on T4 / L4.

Note: Very common community chat-tuned Mistral-7B variant (often used in tutorials / Kaggle notebooks).
Chat template follows the model card (typically ChatML-style tokens).
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

MODEL_NAME  = "unsloth/OpenHermes-2.5-Mistral-7B-bnb-4bit"
MODEL_SHORT = "openhermes_25_mistral_7b"

from exp_model._base_dpbr import run_for_model  # noqa: E402

if __name__ == "__main__":
    run_for_model(MODEL_NAME, MODEL_SHORT)
