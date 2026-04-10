#!/usr/bin/env python3
"""
EXP-24 Dual-Pass Bootstrap IS — Llama-3.3-70B-Instruct (4-bit)
==============================================================

Model  : unsloth/Llama-3.3-70B-Instruct-bnb-4bit
Profile: pypi  (pip stack aligned with Reference_Notebook_Model where noted)
Method : Dual-Pass Bootstrap IS Reliability (DPBR) — identical to EXP-24
Base   : EXP-09 Hierarchical IS  (SOTA MIS=0.3975)

Usage on Kaggle
---------------
    !python exp_model/exp_llama33_70b.py

Note: ref_* profiles pin transformers; use a fresh Kaggle session when switching families
(e.g. Phi/Llama 4.56.x vs Qwen3.5 5.2.x vs ref_git_tf55/ref_gemma4 5.5.x).
"""

# ============================================================
# Step 0: env bootstrap  (same pattern as experiment_DM/*.py)
# ============================================================
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

# ============================================================
# Step 1: model config
# ============================================================
MODEL_NAME  = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
MODEL_SHORT = "llama33_70b"

# ============================================================
# Step 2: run (all EXP-24 logic lives in the shared base)
# ============================================================
from exp_model._base_dpbr import run_for_model  # noqa: E402

if __name__ == "__main__":
    run_for_model(MODEL_NAME, MODEL_SHORT)
