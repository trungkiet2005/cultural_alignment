#!/usr/bin/env python3
"""
Paper sweep — Llama-3.3-70B-Instruct (vLLM) — EXP-24 (DPBR), 20 countries
=========================================================================

Model : meta-llama/Llama-3.3-70B-Instruct (vLLM)
Method: EXP-24 DPBR (same as ``exp_model/exp_24/exp_llama33_70b.py``, extended countries).

Kaggle:
    !python exp_paper/exp_paper_llama33_70b.py
"""

import os
import subprocess
import sys

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("MORAL_MODEL_BACKEND", "vllm")
if os.path.isdir("/kaggle/working"):
    os.environ.setdefault("MORAL_VLLM_GPU_MEM", "0.95")

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
        'pip install -q "numpy<2.3"',
        "pip uninstall -y -q tensorflow tensorflow-cpu tf_keras 2>/dev/null || true",
        'pip install -q --upgrade "protobuf>=5.29.6,<6" "grpcio>=1.68" "googleapis-common-protos>=1.66"',
        "pip install -q scipy tqdm sentencepiece",
        "pip install -q vllm",
        'pip install --quiet "datasets>=3.4.1,<4.4.0"',
    ]:
        subprocess.run(cmd, shell=True, check=False)


_ensure_repo()
from src.hf_env import apply_hf_credentials  # noqa: E402

apply_hf_credentials()
_install_deps()

MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_SHORT = "llama33_70b"

from exp_paper.paper_countries import PAPER_20_COUNTRIES, RESULTS_BASE_EXP24_20C  # noqa: E402
from exp_model._base_dpbr import run_for_model  # noqa: E402

if __name__ == "__main__":
    run_for_model(
        MODEL_NAME,
        MODEL_SHORT,
        target_countries=PAPER_20_COUNTRIES,
        results_base=RESULTS_BASE_EXP24_20C,
    )
