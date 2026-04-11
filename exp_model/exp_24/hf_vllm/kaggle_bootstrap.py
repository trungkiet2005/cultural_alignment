"""
Kaggle bootstrap for ``hf_vllm`` scripts: clone repo + install vLLM stack.

Requires a **CUDA** runtime compatible with the vLLM wheel (typically Linux;
see https://docs.vllm.ai/en/latest/getting_started/installation.html).

Do not import before ``sys.path`` includes the repo root.
"""

import os
import subprocess
import sys

REPO_URL = "https://github.com/trungkiet2005/cultural_alignment.git"
REPO_DIR_KAGGLE = "/kaggle/working/cultural_alignment"


def setup_torch_env() -> None:
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")


def on_kaggle() -> bool:
    return os.path.isdir("/kaggle/working")


def ensure_repo() -> str:
    setup_torch_env()
    here = os.getcwd()
    if os.path.isfile(os.path.join(here, "src", "controller.py")):
        return here
    if not on_kaggle():
        raise RuntimeError("Not on Kaggle and not inside the repo root.")
    if not os.path.isdir(REPO_DIR_KAGGLE):
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, REPO_DIR_KAGGLE],
            check=True,
        )
    os.chdir(REPO_DIR_KAGGLE)
    sys.path.insert(0, REPO_DIR_KAGGLE)
    return REPO_DIR_KAGGLE


def install_vllm_stack() -> None:
    """vLLM + helpers for HF hub and DPBR data paths."""
    if not on_kaggle():
        return
    for cmd in [
        "pip install -q accelerate scipy tqdm sentencepiece protobuf",
        "pip install -q vllm",
        'pip install --quiet "datasets>=3.4.1,<4.4.0"',
    ]:
        subprocess.run(cmd, shell=True, check=False)
