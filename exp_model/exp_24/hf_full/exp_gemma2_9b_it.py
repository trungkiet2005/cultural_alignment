#!/usr/bin/env python3
"""
EXP-24 DPBR — Gemma-2-9B-IT (HF google/, bf16)
==============================================

Model  : google/gemma-2-9b-it  (upstream weights, not unsloth/*-bnb-4bit)
Load   : load_in_4bit=False  (full bf16 via Unsloth)
Method : Dual-Pass Bootstrap IS Reliability (DPBR)

Usage (Kaggle)
--------------
    !python exp_model/exp_24/hf_full/exp_gemma2_9b_it.py

Access: Gemma weights on HF may require accepting the license; set `HF_TOKEN` if prompted.

VRAM: ~18–22 GB bf16; use ≥40 GB GPU for headroom.
"""

from exp_model.exp_24.hf_full.kaggle_bootstrap import ensure_repo, install_pypi_unsloth

ensure_repo()
install_pypi_unsloth()

MODEL_NAME = "google/gemma-2-9b-it"
MODEL_SHORT = "hf_gemma2_9b_bf16"

from exp_model._base_dpbr import run_for_model  # noqa: E402

if __name__ == "__main__":
    run_for_model(MODEL_NAME, MODEL_SHORT, load_in_4bit=False)
