#!/usr/bin/env python3
"""
EXP-24 DPBR — Qwen2.5-7B-Instruct (vLLM engine, bf16)
=====================================================

Model  : Qwen/Qwen2.5-7B-Instruct
Load   : ``vllm.LLM`` — next-token scores for A/B ids via ``generate`` + ``allowed_token_ids`` (see ``src.vllm_causal``)
Method : Dual-Pass Bootstrap IS Reliability (DPBR)

Other EXP-24 scripts use **Unsloth** by default; ``exp_24/hf_full`` uses HF ``AutoModel``;
this folder uses **vLLM** for throughput-oriented inference.

Usage (Kaggle, CUDA)
--------------------
    !python exp_model/exp_24/hf_vllm/exp_qwen25_7b_instruct.py

Env (optional)
--------------
    EXP24_VLLM_PREFLIGHT=1   — subprocess preflight loads model once before the main run (slow).
    VLLM_GPU_MEMORY_UTILIZATION=0.9
    VLLM_TP=1
    VLLM_ENFORCE_EAGER=1
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
        "pip install -q accelerate scipy tqdm sentencepiece protobuf",
        "pip install -q vllm",
        'pip install --quiet "datasets>=3.4.1,<4.4.0"',
    ]:
        subprocess.run(cmd, shell=True, check=False)


_ensure_repo()
_install_deps()

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_SHORT = "vllm_qwen25_7b_bf16"

from exp_model._base_dpbr import run_for_model  # noqa: E402

if __name__ == "__main__":
    run_for_model(MODEL_NAME, MODEL_SHORT, load_in_4bit=False, use_vllm=True)
