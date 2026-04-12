#!/usr/bin/env python3
"""
Paper sweep — Gemma-2-9B-IT (HF google/, bf16) — EXP-24 (DPBR), 20 countries
============================================================================

Model : google/gemma-2-9b-it (HF native; aligns with EXP-24-HF_GEMMA2_9B_BF16)
Method: EXP-24 DPBR

Kaggle:
    !python exp_paper/exp_paper_hf_gemma2_9b_bf16.py

Access (required or you get 401 / GatedRepoError):

1. On https://huggingface.co/google/gemma-2-9b-it — accept the Gemma license.
2. Kaggle → Add-ons → Secrets → create **HF_TOKEN** with a read token from
   https://huggingface.co/settings/tokens (same account that accepted the license).

VRAM: ~18–22 GB bf16.
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
        "pip install -q accelerate bitsandbytes scipy tqdm sentencepiece protobuf",
        'pip install --quiet "datasets>=3.4.1,<4.4.0"',
    ]:
        subprocess.run(cmd, shell=True, check=False)


_ensure_repo()
from src.hf_env import apply_hf_credentials  # noqa: E402

# Load .env / Kaggle HF_TOKEN secret before pip so any hub-aware steps see credentials.
apply_hf_credentials()
_install_deps()

MODEL_NAME = "google/gemma-2-9b-it"
MODEL_SHORT = "hf_gemma2_9b_bf16"

from exp_paper.paper_countries import PAPER_20_COUNTRIES, RESULTS_BASE_EXP24_20C  # noqa: E402
from exp_model._base_dpbr import run_for_model  # noqa: E402

if __name__ == "__main__":
    _tok = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if _on_kaggle() and not _tok:
        raise SystemExit(
            "[HF] google/gemma-2-9b-it is gated. Add Kaggle → Add-ons → Secrets → HF_TOKEN "
            "(read token from https://huggingface.co/settings/tokens ) using the account that "
            "accepted the license at https://huggingface.co/google/gemma-2-9b-it"
        )
    run_for_model(
        MODEL_NAME,
        MODEL_SHORT,
        load_in_4bit=False,
        use_hf_native=True,
        target_countries=PAPER_20_COUNTRIES,
        results_base=RESULTS_BASE_EXP24_20C,
    )
