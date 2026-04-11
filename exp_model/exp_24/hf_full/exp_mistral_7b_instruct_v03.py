#!/usr/bin/env python3
"""
EXP-24 DPBR — Mistral-7B-Instruct-v0.3 (HF upstream, bf16)
===========================================================

Model  : mistralai/Mistral-7B-Instruct-v0.3
Load   : load_in_4bit=False  (full bf16 via Unsloth)
Method : Dual-Pass Bootstrap IS Reliability (DPBR)

Usage (Kaggle)
--------------
    !python exp_model/exp_24/hf_full/exp_mistral_7b_instruct_v03.py

VRAM: ~15–18 GB bf16; T4 16 GB may be tight or OOM.
"""

from exp_model.exp_24.hf_full.kaggle_bootstrap import ensure_repo, install_pypi_unsloth

ensure_repo()
install_pypi_unsloth()

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SHORT = "hf_mistral7b_v03_bf16"

from exp_model._base_dpbr import run_for_model  # noqa: E402

if __name__ == "__main__":
    run_for_model(MODEL_NAME, MODEL_SHORT, load_in_4bit=False)
