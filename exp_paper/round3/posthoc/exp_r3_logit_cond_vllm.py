#!/usr/bin/env python3
"""Cross-model logit-conditioning — vLLM BF16 track (5 models).

Runs vanilla logit-conditioning on the 5 paper models that fit Kaggle's
~20 GB working disk as BF16 (everything except Llama-3.3-70B). Produces
``logit_cond_<model>.csv`` per model. Aggregation is done by
``exp_r3_logit_cond_aggregate.py`` after the Unsloth twin has also run.

Just copy-paste into a fresh Kaggle notebook and run — no env vars needed.

    !python exp_paper/round3/posthoc/exp_r3_logit_cond_vllm.py
"""

from __future__ import annotations

import os as _os, subprocess as _sp, sys as _sys

_REPO_URL = "https://github.com/trungkiet2005/cultural_alignment.git"
_REPO_DIR_KAGGLE = "/kaggle/working/cultural_alignment"


def _bootstrap() -> str:
    here = _os.getcwd()
    if _os.path.isfile(_os.path.join(here, "src", "controller.py")):
        if here not in _sys.path:
            _sys.path.insert(0, here)
        return here
    if not _os.path.isdir("/kaggle/input"):
        raise RuntimeError("Not on Kaggle and not inside the repo root.")
    if not _os.path.isdir(_REPO_DIR_KAGGLE):
        _sp.run(["git", "clone", "--depth", "1", _REPO_URL, _REPO_DIR_KAGGLE], check=True)
    _os.chdir(_REPO_DIR_KAGGLE)
    _sys.path.insert(0, _REPO_DIR_KAGGLE)
    return _REPO_DIR_KAGGLE


_bootstrap()
# Force vLLM for this track (must be set BEFORE importing paper_runtime).
_os.environ["MORAL_MODEL_BACKEND"] = "vllm"

import gc
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from exp_paper._r2_common import build_cfg, load_model_timed, load_scenarios, on_kaggle
from exp_paper.paper_runtime import configure_paper_env, install_paper_kaggle_deps

configure_paper_env()
from src.hf_env import apply_hf_credentials

apply_hf_credentials()

# 5 models that run as BF16 via vLLM.
MODELS: List[Tuple[str, str, str]] = [
    # (display_name,          HF id,                             dir_short)
    ("Phi-3.5-mini",         "microsoft/Phi-3.5-mini-instruct", "phi_3_5_mini"),
    ("Qwen2.5-7B",           "Qwen/Qwen2.5-7B-Instruct",        "qwen2_5_7b"),
    ("Qwen3-VL-8B",          "Qwen/Qwen3-VL-8B-Instruct",       "qwen3_vl_8b"),
    ("Phi-4",                "microsoft/phi-4",                 "phi_4"),
    ("Magistral-Small-2509", "mistralai/Magistral-Small-2509",  "magistral_small_2509"),
]

# Rough on-disk size (GB) for pre-flight disk check.
_MODEL_DISK_GB: Dict[str, float] = {
    "microsoft/Phi-3.5-mini-instruct":  8.0,
    "Qwen/Qwen2.5-7B-Instruct":         15.0,
    "Qwen/Qwen3-VL-8B-Instruct":        17.0,
    "microsoft/phi-4":                  28.0,
    "mistralai/Magistral-Small-2509":   48.0,
}
_DISK_SAFETY_GB = 4.0

from exp_paper.paper_countries import PAPER_20_COUNTRIES

N_SCEN = int(os.environ.get("R2_N_SCENARIOS", "300"))
COUNTRIES = (
    [c.strip() for c in os.environ["R2_COUNTRIES"].split(",") if c.strip()]
    if "R2_COUNTRIES" in os.environ
    else list(PAPER_20_COUNTRIES)
)

OUT_DIR = Path(
    "/kaggle/working/cultural_alignment/results/exp24_round3/logit_conditioning_cross_model"
    if on_kaggle()
    else "results/exp24_round3/logit_conditioning_cross_model"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _free_disk_gb(path: str = "/") -> float:
    try:
        return shutil.disk_usage(path).free / (1024 ** 3)
    except Exception:
        return float("inf")


def _purge_hf_cache(hf_id: str) -> None:
    roots = [
        os.environ.get("HF_HOME"),
        os.environ.get("HUGGINGFACE_HUB_CACHE"),
        os.path.expanduser("~/.cache/huggingface"),
        "/root/.cache/huggingface",
        "/kaggle/working/.cache/huggingface",
    ]
    dir_name = "models--" + hf_id.replace("/", "--")
    for root in roots:
        if not root:
            continue
        for sub in ("hub", ""):
            cand = os.path.join(root, sub, dir_name) if sub else os.path.join(root, dir_name)
            if os.path.isdir(cand):
                shutil.rmtree(cand, ignore_errors=True)
                print(f"  [cache] purged {cand}")


def _purge_all_hf_cache() -> None:
    roots = [
        os.environ.get("HF_HOME"),
        os.environ.get("HUGGINGFACE_HUB_CACHE"),
        os.path.expanduser("~/.cache/huggingface"),
        "/root/.cache/huggingface",
        "/kaggle/working/.cache/huggingface",
    ]
    for root in roots:
        if not root:
            continue
        hub = os.path.join(root, "hub")
        if os.path.isdir(hub):
            shutil.rmtree(hub, ignore_errors=True)
            print(f"  [cache] swept {hub}")


def _run_one(display_name: str, hf_id: str, dir_short: str) -> None:
    from src.logit_conditioning import diagnose_country
    from src.model import setup_seeds

    setup_seeds(42)

    need = _MODEL_DISK_GB.get(hf_id, 20.0) + _DISK_SAFETY_GB
    free = _free_disk_gb("/")
    if free < need:
        print(f"  [disk] free={free:.1f} GB < need≈{need:.1f} GB — sweeping HF cache")
        _purge_all_hf_cache()
        free = _free_disk_gb("/")
    if free < need:
        raise RuntimeError(
            f"Insufficient disk: free={free:.1f} GB, need≈{need:.1f} GB for {hf_id}."
        )
    print(f"  [disk] free={free:.1f} GB OK for {hf_id} (need≈{need:.1f} GB)")

    cfg = build_cfg(hf_id, str(OUT_DIR), COUNTRIES, n_scenarios=N_SCEN, load_in_4bit=False)
    model, tokenizer = load_model_timed(hf_id, backend="vllm", load_in_4bit=False)

    rows: List[Dict] = []
    for country in COUNTRIES:
        try:
            scen = load_scenarios(cfg, country)
            agg, _ = diagnose_country(model, tokenizer, country, scen, cfg)
            rows.append({
                "model":       display_name,
                "model_short": dir_short,
                "country":     country,
                **agg,
            })
            print(f"  ✓ {display_name} {country}  "
                  f"margin={agg.get('mean_margin', float('nan')):.3f}  "
                  f"entropy={agg.get('mean_entropy', float('nan')):.3f}")
        except Exception as exc:
            print(f"[error] {display_name} {country}: {exc}")
            rows.append({"model": display_name, "country": country, "error": str(exc)[:500]})

    out_path = OUT_DIR / f"logit_cond_{dir_short}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[saved] {out_path}")

    del model, tokenizer
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    _purge_hf_cache(hf_id)


def main() -> None:
    # Install vLLM deps once. Do NOT upgrade to unsloth mid-run — that breaks
    # pyarrow ABI and kills every subsequent vLLM load.
    install_paper_kaggle_deps()

    idx_env = os.environ.get("R2_MODEL_INDEX", "").strip()
    if idx_env:
        keep = {int(x) for x in idx_env.split(",") if x.strip().isdigit()}
        run_list = [m for i, m in enumerate(MODELS) if i in keep]
        print(f"[R2_MODEL_INDEX={idx_env}] running {[m[0] for m in run_list]}")
    else:
        run_list = list(MODELS)

    for (display_name, hf_id, dir_short) in run_list:
        print(f"\n{'#' * 72}\n# {display_name}  ({hf_id})\n{'#' * 72}")
        print(f"  [disk] free before load: {_free_disk_gb('/'):.1f} GB")
        try:
            _run_one(display_name, hf_id, dir_short)
        except Exception as exc:
            print(f"[error] {display_name}: {exc}")
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            _purge_hf_cache(hf_id)
            if _free_disk_gb("/") < 10.0:
                print("  [disk] <10 GB free after per-model purge — full sweep")
                _purge_all_hf_cache()
            print(f"  [disk] free after cleanup: {_free_disk_gb('/'):.1f} GB")

    print(f"\n[done] vLLM track finished. CSVs in {OUT_DIR}")
    print("Next: run exp_r3_logit_cond_unsloth.py (Llama-70B) in a separate notebook,")
    print("      then exp_r3_logit_cond_aggregate.py to produce the cross-model table.")


if __name__ == "__main__":
    main()
