#!/usr/bin/env python3
"""
EXP-24 Dual-Pass Bootstrap IS — Nemotron-3-Nano-30B-A3B
=======================================================

Model  : unsloth/Nemotron-3-Nano-30B-A3B
Profile: pypi  (pip stack aligned with Reference_Notebook_Model/new)
Notebook: Reference_Notebook_Model/new/unsloth-nemotron-3-nano-30b-a3b.ipynb
         (uses trust_remote_code=True + attn eager — mirrored in src/model.py load_model)
Method : Dual-Pass Bootstrap IS Reliability (DPBR) — identical to EXP-24
Base   : EXP-09 Hierarchical IS  (SOTA MIS=0.3975)

Usage on Kaggle
---------------
    !python exp_model/exp_24/exp_nemotron3_nano_30b.py

Note: Large MoE-class checkpoint — prefer a high-VRAM GPU (e.g. H100 80GB) or Unsloth docs for your tier.
Note: ref_* profiles pin transformers; use a fresh Kaggle session when switching families.
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
    # Notebook (new/unsloth-nemotron-3-nano-30b-a3b.ipynb) pins transformers==4.56.2.
    # Nemotron uses trust_remote_code=True (see src/model.py); aligning TF version avoids
    # remote-code / API mismatches when Kaggle default is 5.x.
    for cmd in [
        'pip install -q bitsandbytes scipy tqdm sentencepiece protobuf',
        'pip install --upgrade --no-deps unsloth',
        'pip install -q unsloth_zoo',
        'pip install -q "transformers==4.56.2"',
        'pip install --quiet "datasets>=3.4.1,<4.4.0"',
    ]:
        subprocess.run(cmd, shell=True, check=False)


_ensure_repo()
_install_deps()

# ============================================================
# Step 1: model config
# ============================================================
MODEL_NAME  = "unsloth/Nemotron-3-Nano-30B-A3B"
MODEL_SHORT = "nemotron3_nano_30b"

# ============================================================
# Step 2: run (all EXP-24 logic lives in the shared base)
# ============================================================
from exp_model._base_dpbr import run_for_model  # noqa: E402

if __name__ == "__main__":
    run_for_model(MODEL_NAME, MODEL_SHORT)
