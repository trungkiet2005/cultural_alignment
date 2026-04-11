#!/usr/bin/env python3
"""
EXP-24 DPBR — Phi-3-mini-4k-Instruct (HF upstream, bf16)
========================================================

Model  : microsoft/Phi-3-mini-4k-instruct
Load   : load_in_4bit=False  (full bf16 via Unsloth)
Method : Dual-Pass Bootstrap IS Reliability (DPBR)

Usage (Kaggle)
--------------
    !python exp_model/exp_24/hf_full/exp_phi3_mini_4k_instruct.py

VRAM: ~8–10 GB bf16 — fits many single-GPU notebooks.
"""

from exp_model.exp_24.hf_full.kaggle_bootstrap import ensure_repo, install_pypi_unsloth

ensure_repo()
install_pypi_unsloth()

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
MODEL_SHORT = "hf_phi3_mini_bf16"

from exp_model._base_dpbr import run_for_model  # noqa: E402

if __name__ == "__main__":
    run_for_model(MODEL_NAME, MODEL_SHORT, load_in_4bit=False)
