"""
Reference copy of the Kaggle bootstrap (clone repo + pip deps for HF-native runs).

Each ``exp_*.py`` in this folder inlines the same logic so a notebook can run
without ``from exp_model...`` before ``sys.path`` includes the repo root.

These scripts use **Hugging Face transformers only** (no Unsloth) — see
``src.model.load_model_hf_native`` and ``run_for_model(..., use_hf_native=True)``.

Do not import this module from notebook cells unless the repo is already on
``sys.path`` (e.g. after ``%cd /kaggle/working/cultural_alignment``).
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
    """Return repo root; clone on Kaggle if needed."""
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


def install_hf_inference_deps() -> None:
    """accelerate (device_map), bitsandbytes (optional 4-bit), datasets — no Unsloth."""
    if not on_kaggle():
        return
    for cmd in [
        "pip install -q accelerate bitsandbytes scipy tqdm sentencepiece protobuf",
        'pip install --quiet "datasets>=3.4.1,<4.4.0"',
    ]:
        subprocess.run(cmd, shell=True, check=False)


def install_pypi_unsloth() -> None:
    """Legacy: Unsloth stack for non-hf_full EXP-24 entrypoints. Not used by ``hf_full``."""
    if not on_kaggle():
        return
    os.environ.setdefault("UNSLOTH_DISABLE_AUTO_COMPILE", "1")
    os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
    for cmd in [
        "pip install -q bitsandbytes scipy tqdm sentencepiece protobuf",
        "pip install --upgrade --no-deps unsloth",
        "pip install -q unsloth_zoo",
        'pip install --quiet "datasets>=3.4.1,<4.4.0"',
    ]:
        subprocess.run(cmd, shell=True, check=False)
