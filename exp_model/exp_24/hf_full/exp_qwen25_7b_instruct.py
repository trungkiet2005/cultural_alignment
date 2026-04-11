#!/usr/bin/env python3
"""
EXP-24 DPBR — Qwen2.5-7B-Instruct (HF upstream, bf16)
======================================================

Model  : Qwen/Qwen2.5-7B-Instruct
Load   : load_in_4bit=False  (full bf16 via Unsloth)
Method : Dual-Pass Bootstrap IS Reliability (DPBR)

Usage (Kaggle)
--------------
    !python exp_model/exp_24/hf_full/exp_qwen25_7b_instruct.py

VRAM: ~15–18 GB bf16 on 80 GB class GPUs.
"""

from exp_model.exp_24.hf_full.kaggle_bootstrap import ensure_repo, install_pypi_unsloth

ensure_repo()
install_pypi_unsloth()

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_SHORT = "hf_qwen25_7b_bf16"

from exp_model._base_dpbr import run_for_model  # noqa: E402

if __name__ == "__main__":
    run_for_model(MODEL_NAME, MODEL_SHORT, load_in_4bit=False)
