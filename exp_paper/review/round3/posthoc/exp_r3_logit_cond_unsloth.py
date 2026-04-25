#!/usr/bin/env python3
"""Cross-model logit-conditioning — Unsloth 4-bit track (Llama-3.3-70B only).

Runs vanilla logit-conditioning on Llama-3.3-70B via Unsloth bnb-4bit
(~40 GB on disk) because the BF16 checkpoint (~140 GB) does not fit
Kaggle's working disk.

**Must be run in its own Kaggle notebook** — mixing vLLM + Unsloth in one
process upgrades transformers to 5.x mid-run and breaks pyarrow ABI.

Copy-paste into a fresh Kaggle notebook, then run:

    !python exp_paper/review/round3/posthoc/exp_r3_logit_cond_unsloth.py
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
# Force Unsloth BEFORE importing paper_runtime so install_paper_kaggle_deps()
# installs the unsloth stack instead of vLLM.
_os.environ["MORAL_MODEL_BACKEND"] = "unsloth"

import gc
import os
import shutil
from pathlib import Path
from typing import Dict, List

import pandas as pd

from exp_paper._r2_common import build_cfg, load_model_timed, load_scenarios, on_kaggle
from exp_paper.paper_runtime import configure_paper_env, install_paper_kaggle_deps

configure_paper_env()
from src.hf_env import apply_hf_credentials

apply_hf_credentials()

# Only Llama-3.3-70B lives on the Unsloth track.
DISPLAY_NAME = "Llama-3.3-70B"
HF_ID        = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
DIR_SHORT    = "llama_3_3_70b"
EST_DISK_GB  = 42.0  # 4-bit checkpoint size
SAFETY_GB    = 6.0

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


def main() -> None:
    # Install Unsloth stack ONCE (MORAL_MODEL_BACKEND=unsloth was set above).
    install_paper_kaggle_deps()

    # Import unsloth FIRST so its patches apply before transformers loads.
    try:
        import unsloth  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "unsloth import failed after install_paper_kaggle_deps(). "
            "Check the install log above."
        ) from exc

    from src.logit_conditioning import diagnose_country
    from src.model import setup_seeds

    setup_seeds(42)

    print(f"\n{'#' * 72}\n# {DISPLAY_NAME}  ({HF_ID})\n{'#' * 72}")
    need = EST_DISK_GB + SAFETY_GB
    free = _free_disk_gb("/")
    print(f"  [disk] free={free:.1f} GB  need≈{need:.1f} GB")
    if free < need:
        raise SystemExit(
            f"Insufficient disk: free={free:.1f} GB, need≈{need:.1f} GB. "
            f"Enable Kaggle persistent disk or free working dir."
        )

    cfg = build_cfg(HF_ID, str(OUT_DIR), COUNTRIES, n_scenarios=N_SCEN, load_in_4bit=True)
    model, tokenizer = load_model_timed(HF_ID, backend="unsloth", load_in_4bit=True)

    rows: List[Dict] = []
    for country in COUNTRIES:
        try:
            scen = load_scenarios(cfg, country)
            agg, _ = diagnose_country(model, tokenizer, country, scen, cfg)
            rows.append({
                "model":       DISPLAY_NAME,
                "model_short": DIR_SHORT,
                "country":     country,
                **agg,
            })
            print(f"  ✓ {DISPLAY_NAME} {country}  "
                  f"margin={agg.get('mean_margin', float('nan')):.3f}  "
                  f"entropy={agg.get('mean_entropy', float('nan')):.3f}")
        except Exception as exc:
            print(f"[error] {DISPLAY_NAME} {country}: {exc}")
            rows.append({"model": DISPLAY_NAME, "country": country, "error": str(exc)[:500]})

    out_path = OUT_DIR / f"logit_cond_{DIR_SHORT}.csv"
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
    _purge_hf_cache(HF_ID)

    print(f"\n[done] Unsloth track finished. CSV in {OUT_DIR}")
    print("Next: run exp_r3_logit_cond_aggregate.py (CPU-only) to produce the cross-model table.")


if __name__ == "__main__":
    main()
