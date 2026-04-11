#!/usr/bin/env python3
"""
Paper sweep — Gemma-3-270M-IT — EXP-24 (DPBR), 20 countries
============================================================

Model : google/gemma-3-270m-it (vLLM)
Method: EXP-24 DPBR

Kaggle:
    !python exp_paper/exp_paper_gemma3_270m.py
"""

import os
import subprocess
import sys

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("MORAL_MODEL_BACKEND", "vllm")
if os.path.isdir("/kaggle/working"):
    os.environ.setdefault("VLLM_GPU_MEMORY_UTILIZATION", "0.95")

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
    try:
        from kaggle_secrets import UserSecretsClient

        _hf = UserSecretsClient().get_secret("HF_TOKEN")
        if _hf:
            os.environ["HF_TOKEN"] = _hf
            os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf
    except Exception:
        pass
    for cmd in [
        'pip install -q "numpy<2.3"',
        "pip install -q scipy tqdm sentencepiece protobuf",
        "pip install -q vllm",
        'pip install --quiet "datasets>=3.4.1,<4.4.0"',
    ]:
        subprocess.run(cmd, shell=True, check=False)


_ensure_repo()
_install_deps()

MODEL_NAME = "google/gemma-3-270m-it"
MODEL_SHORT = "gemma3_270m"

from exp_paper.paper_countries import PAPER_20_COUNTRIES, RESULTS_BASE_EXP24_20C  # noqa: E402
from exp_model._base_dpbr import run_for_model  # noqa: E402

if __name__ == "__main__":
    run_for_model(
        MODEL_NAME,
        MODEL_SHORT,
        target_countries=PAPER_20_COUNTRIES,
        results_base=RESULTS_BASE_EXP24_20C,
    )
