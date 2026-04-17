"""EXP-24 runners using upstream Hugging Face model IDs with full bf16 (no 4-bit).

Each ``exp_*.py`` clones this repo on Kaggle and patches ``sys.path`` before
importing ``exp_model`` (same pattern as ``exp_llama3_8b.py``). Run the full
file with ``!python ...``; do not paste only the imports into a notebook cell.
"""
