#!/usr/bin/env python3
"""
EXP-24 DPBR — Meta-Llama-3-8B-Instruct (HF upstream, bf16)
============================================================

Model  : meta-llama/Meta-Llama-3-8B-Instruct  (no Unsloth *_bnb_4bit suffix)
Load   : load_in_4bit=False  →  full bf16 via Unsloth FastLanguageModel
Method : Dual-Pass Bootstrap IS Reliability (DPBR) — same as other EXP-24

Usage (Kaggle)
--------------
    !python exp_model/exp_24/hf_full/exp_meta_llama3_8b_instruct.py

Requires Hugging Face access to gated Llama 3 weights:
    - Add `HF_TOKEN` / Kaggle secret with Meta license acceptance on huggingface.co

VRAM: ~16–20 GB bf16 on a single 80 GB GPU is typical; T4 16 GB will OOM.
"""

from exp_model.exp_24.hf_full.kaggle_bootstrap import ensure_repo, install_pypi_unsloth

ensure_repo()
install_pypi_unsloth()

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_SHORT = "hf_llama3_8b_bf16"

from exp_model._base_dpbr import run_for_model  # noqa: E402

if __name__ == "__main__":
    run_for_model(MODEL_NAME, MODEL_SHORT, load_in_4bit=False)
