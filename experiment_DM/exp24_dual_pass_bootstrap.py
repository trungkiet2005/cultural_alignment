#!/usr/bin/env python3
"""
EXP-24: Dual-Pass Bootstrap IS Reliability (DPBR)
==================================================
**Base**: EXP-09 (Hierarchical IS — SOTA MIS = 0.3975)

============================================================
THE BINARY ESS GUARD PROBLEM
============================================================
EXP-09 uses a hard binary ESS guard:
    delta_star = IS_result  if k_eff/K >= rho_eff = 0.10
    delta_star = 0          otherwise  (complete fallback)

This is a coarse reliability filter. The IS is either fully trusted or
fully discarded, with no gradation. Two problems:

1. **False positives**: k_eff/K = 0.11 (barely above threshold) → IS output
   is trusted despite being barely better than uniform weight → potentially
   noisy delta_star contaminates delta_opt_micro and the country prior.

2. **False negatives**: k_eff/K = 0.09 (just below threshold) → IS output
   is completely discarded even though it may have found a useful direction.
   The binary fallback wastes a real signal.

A better reliability measure: run the IS TWICE with independent noise, and
use the AGREEMENT between the two runs as a continuous reliability signal.

============================================================
EXP-24 INNOVATION: Dual-Pass Bootstrap IS Reliability
============================================================
Instead of K=128 samples in one pass, run TWO independent passes:
    Pass 1: eps_1 ~ N(0, σ²), K1=64 samples → delta_star_1
    Pass 2: eps_2 ~ N(0, σ²), K2=64 samples → delta_star_2

Total compute: K1 + K2 = 128 (identical to EXP-09, no extra cost).

Bootstrap reliability: disagreement between the two passes:
    bootstrap_var = (delta_star_1 - delta_star_2)²

Soft reliability weight:
    r = exp(-bootstrap_var / VAR_SCALE)     ∈ (0, 1]

Final IS output (soft-blended):
    delta_star_final = r · (delta_star_1 + delta_star_2) / 2
                     + (1 - r) · 0.0   (= fallback when disagree)
    delta_opt_micro  = anchor + delta_star_final

**Properties**:
- When both passes agree (bootstrap_var → 0):  r → 1.0 → use mean (maximally trusted)
- When both passes disagree (bootstrap_var large): r → 0 → delta_star_final → 0 (fallback)
- No binary threshold: continuous soft weighting replaces the hard ESS guard
- Works for BOTH passes' ESS independently: each pass still has its own ESS guard
  as a safety rail (if k_eff/K < rho_eff, that pass returns 0)

**Connection to bootstrap in statistics**:
Bootstrap variance estimation: resample the data (here: resample the IS noise)
and compute the estimator variance. If the variance is high, the estimator is
unreliable. EXP-24 uses this to soft-threshold IS outputs without a hard ESS guard.

**Why this helps Mistral**:
Mistral's IS is CONSISTENTLY UNRELIABLE (Pearson r < 0). With the dual-pass approach:
  - Pass 1 and Pass 2 will DISAGREE more often (both are noisy but independently noisy)
  - bootstrap_var will be persistently HIGH → r → 0 → delta_star_final → 0
  - This is equivalent to AUTOMATICALLY DISABLING the IS for Mistral (per-scenario)
  - The country prior takes over (from the global EMA, anchored by the few reliable scenarios)

Hyperparameters:
    K1 = K2 = 64    (half of EXP-09's K=128 per pass, same total compute)
    VAR_SCALE = 0.04 (scale for soft weight: at bootstrap_var=0.04 → r=exp(-1)≈0.37)

Usage on Kaggle
---------------
    !python experiment_DM/exp24_dual_pass_bootstrap.py
"""

# ============================================================================
# Step 0: env bootstrap
# ============================================================================
import os, sys, subprocess

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("UNSLOTH_DISABLE_AUTO_COMPILE", "1")

REPO_URL        = "https://github.com/trungkiet2005/cultural_alignment.git"
REPO_DIR_KAGGLE = "/kaggle/working/cultural_alignment"

def _on_kaggle(): return os.path.isdir("/kaggle/working")

def _ensure_repo():
    here = os.getcwd()
    if os.path.isfile(os.path.join(here, "src", "controller.py")):
        return here
    if not _on_kaggle():
        raise RuntimeError("Not on Kaggle and not inside the repo root.")
    if not os.path.isdir(REPO_DIR_KAGGLE):
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR_KAGGLE], check=True)
    os.chdir(REPO_DIR_KAGGLE)
    sys.path.insert(0, REPO_DIR_KAGGLE)
    return REPO_DIR_KAGGLE

def _install_deps():
    if not _on_kaggle(): return
    for cmd in [
        "pip install -q bitsandbytes scipy tqdm",
        "pip install --upgrade --no-deps unsloth",
        "pip install -q unsloth_zoo",
        "pip install --quiet 'datasets>=3.4.1,<4.4.0'",
    ]:
        subprocess.run(cmd, shell=True, check=False)

_REPO_DIR = _ensure_repo()
_install_deps()

# ============================================================================
# Step 1: imports
# ============================================================================
import gc, shutil
from pathlib import Path
from typing import List

import torch
try:
    torch._dynamo.config.disable = True
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass
import pandas as pd

from experiment_DM.exp24_dpbr_core import (
    BootstrapPriorState,
    Exp24DualPassController,
    K_HALF,
    PRIOR_STATE,
    VAR_SCALE,
    patch_swa_runner_controller,
)
from experiment_DM.exp_reporting import (
    CompareSpec, append_rows_csv, flatten_per_dim_alignment,
    print_alignment_table, print_metric_comparison, try_load_reference_comparison,
)
from src.config import SWAConfig, resolve_output_dir
from src.constants import COUNTRY_LANG
from src.model import setup_seeds, load_model
from src.data import load_multitp_dataset
from src.scenarios import generate_multitp_scenarios
from src.personas import build_country_personas, SUPPORTED_COUNTRIES
from src.swa_runner import run_country_experiment

# ============================================================================
# Step 2: configuration
# ============================================================================
EXP_ID   = "EXP-24"
EXP_NAME = "dual_pass_bootstrap"

MODELS: List[str] = [
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "unsloth/gemma-2-9b-it-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
]
TARGET_COUNTRIES: List[str] = ["USA", "CHN", "JPN", "DEU", "BRA"]
N_SCENARIOS: int = 500
BATCH_SIZE:  int = 1
SEED:        int = 42
LAMBDA_COOP: float = 0.70
# K_HALF, VAR_SCALE: experiment_DM.exp24_dpbr_core (single source of truth)

SWA_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/swa"
CMP_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/compare"
MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH     = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
HUMAN_AMCE_PATH   = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"

# Core DPBR implementation: experiment_DM.exp24_dpbr_core (shared with exp_model/_base_dpbr.py)


# ============================================================================
# Step 3: Runner
# ============================================================================
def _free_model_cache(model_name):
    safe = "models--" + model_name.replace("/", "--")
    for root in [os.environ.get("HF_HUB_CACHE"), os.environ.get("HF_HOME"),
                 os.path.expanduser("~/.cache/huggingface"), "/root/.cache/huggingface"]:
        if not root: continue
        hub_dir = root if os.path.basename(root.rstrip("/")) == "hub" else os.path.join(root, "hub")
        target  = os.path.join(hub_dir, safe)
        if os.path.isdir(target):
            try: shutil.rmtree(target); print(f"[CLEANUP] removed {target}")
            except Exception as e: print(f"[CLEANUP] error: {e}")

def _build_cfg(model_name):
    return SWAConfig(
        model_name=model_name, n_scenarios=N_SCENARIOS, batch_size=BATCH_SIZE,
        target_countries=list(TARGET_COUNTRIES), load_in_4bit=True, use_real_data=True,
        multitp_data_path=MULTITP_DATA_PATH, wvs_data_path=WVS_DATA_PATH,
        human_amce_path=HUMAN_AMCE_PATH, output_dir=SWA_ROOT,
        lambda_coop=LAMBDA_COOP, K_samples=128,
    )

def _load_scen(cfg, country):
    lang = COUNTRY_LANG.get(country, "en")
    if cfg.use_real_data:
        df = load_multitp_dataset(data_base_path=cfg.multitp_data_path, lang=lang,
                                  translator=cfg.multitp_translator, suffix=cfg.multitp_suffix,
                                  n_scenarios=cfg.n_scenarios)
    else:
        df = generate_multitp_scenarios(cfg.n_scenarios, lang=lang)
    df = df.copy(); df["lang"] = lang
    return df

def _run_model(model, tokenizer, model_name) -> List[dict]:
    cfg     = _build_cfg(model_name)
    out_dir = Path(SWA_ROOT) / resolve_output_dir("", model_name).strip("/\\")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(out_dir)
    print(f"\n{'#'*70}\n# {EXP_ID} [{model_name}] Dual-Pass Bootstrap IS\n{'#'*70}")

    rows = []
    for ci, country in enumerate(cfg.target_countries):
        if country not in SUPPORTED_COUNTRIES: continue
        PRIOR_STATE.clear()
        PRIOR_STATE[country] = BootstrapPriorState()
        print(f"\n[{ci+1}/{len(cfg.target_countries)}] {EXP_ID} {model_name} | {country} "
              f"(K_half={K_HALF}×2={K_HALF*2}, VAR_SCALE={VAR_SCALE})")

        scen = _load_scen(cfg, country)
        personas = build_country_personas(country, wvs_path=WVS_DATA_PATH)

        orig_init = Exp24DualPassController.__init__
        def patched_init(self, *a, country=country, **kw): orig_init(self, *a, country=country, **kw)
        Exp24DualPassController.__init__ = patched_init
        patch_swa_runner_controller()

        results_df, summary = run_country_experiment(model, tokenizer, country, personas, scen, cfg)
        Exp24DualPassController.__init__ = orig_init

        results_df.to_csv(out_dir / f"swa_results_{country}.csv", index=False)
        append_rows_csv(str(Path(CMP_ROOT) / "per_dim_breakdown.csv"),
                        flatten_per_dim_alignment(summary.get("per_dimension_alignment", {}),
                                                  model=model_name, method=f"{EXP_ID}_dual_pass",
                                                  country=country))
        ps  = PRIOR_STATE.get(country, BootstrapPriorState()).stats
        mea = lambda col: float(results_df[col].mean()) if col in results_df.columns else float("nan")

        rows.append({
            "model": model_name, "method": f"{EXP_ID}_dual_pass", "country": country,
            **{f"align_{k}": v for k, v in summary["alignment"].items()},
            "flip_rate": summary["flip_rate"], "n_scenarios": summary["n_scenarios"],
            "final_delta_country": ps["delta_country"], "final_alpha_h": ps["alpha_h"],
            "mean_reliability_r": mea("reliability_r"),
            "mean_bootstrap_var": mea("bootstrap_var"),
            "mean_ess_pass1": mea("ess_pass1"), "mean_ess_pass2": mea("ess_pass2"),
        })

        pda = summary.get("per_dimension_alignment", {})
        if pda:
            print(f"\n  ┌── Per-Dimension ({country}) ──")
            for dk, dd in sorted(pda.items()):
                hv, mv = dd.get("human", float("nan")), dd.get("model", float("nan"))
                print(f"  │  {dk:<25s}  human={hv:6.1f}  model={mv:6.1f}  err={mv-hv:+6.1f}pp")
            print(f"  └── MIS={summary['alignment']['mis']:.4f}  r={summary['alignment']['pearson_r']:+.3f}  "
                  f"Flip={summary['flip_rate']:.1%}")
            print(f"      reliability_r(avg)={mea('reliability_r'):.3f}  "
                  f"bootstrap_var(avg)={mea('bootstrap_var'):.4f}  "
                  f"ESS(p1={mea('ess_pass1'):.3f}, p2={mea('ess_pass2'):.3f})")

        torch.cuda.empty_cache(); gc.collect()
    return rows

def main():
    setup_seeds(SEED)
    for d in (SWA_ROOT, CMP_ROOT): Path(d).mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}\n  {EXP_ID}: {EXP_NAME.upper()}  (base: EXP-09)\n{'='*70}")
    print(f"[THEORY] Pass1+Pass2: K_half={K_HALF} each → same total K={K_HALF*2} as EXP-09")
    print(f"[THEORY] r = exp(-(δ*₁-δ*₂)² / {VAR_SCALE})  (soft reliability weight)")
    print(f"[THEORY] δ* = r · (δ*₁+δ*₂)/2  (replaces binary ESS guard)")
    print(f"[TARGET] MIS < 0.3800 | Mistral r > 0 | mean_reliability_r > 0.60")

    all_rows: List[dict] = []
    for mi, model_name in enumerate(MODELS):
        print(f"\n{'='*70}\n  MODEL {mi+1}/{len(MODELS)}: {model_name}\n{'='*70}")
        model, tokenizer = load_model(model_name, max_seq_length=2048, load_in_4bit=True)
        try:
            all_rows.extend(_run_model(model, tokenizer, model_name))
        finally:
            del model, tokenizer; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            _free_model_cache(model_name)
        pd.DataFrame(all_rows).to_csv(Path(CMP_ROOT) / "comparison.csv", index=False)

    cmp_df = pd.DataFrame(all_rows)
    cmp_df.to_csv(Path(CMP_ROOT) / "comparison.csv", index=False)
    print(f"\n{'#'*70}\n# {EXP_ID} FINAL REPORT\n{'#'*70}")
    print_alignment_table(cmp_df, title=f"{EXP_ID} RESULTS")
    for model_name in MODELS:
        m_df = cmp_df[cmp_df["model"] == model_name]
        if m_df.empty: continue
        short = model_name.split("/")[-1][:22]
        print(f"  {short:<22s}  MIS={m_df['align_mis'].mean():.4f}  "
              f"r={m_df['align_pearson_r'].mean():+.3f}  Flip={m_df['flip_rate'].mean():.1%}  "
              f"rel_r={m_df['mean_reliability_r'].mean():.3f}")
    print(f"\n  OVERALL MEAN MIS = {cmp_df['align_mis'].mean():.4f}  (EXP-09 SOTA: 0.3975)")
    ref = try_load_reference_comparison()
    if ref is not None:
        print_metric_comparison(ref, cmp_df, title=f"{EXP_ID} vs EXP-01",
                                spec=CompareSpec(metric_col="align_mis", ref_method="swa_ptis",
                                                 cur_method=f"{EXP_ID}_dual_pass"))
    from experiment_DM.exp_reporting import print_tracker_ready_report
    print_tracker_ready_report(
        cmp_df,
        exp_id=EXP_ID,
        cur_method=f"{EXP_ID}_dual_pass",
        per_dim_csv_path=str(Path(CMP_ROOT) / "per_dim_breakdown.csv"),
    )
    print(f"\n[{EXP_ID}] DONE — {CMP_ROOT}")

if __name__ == "__main__":
    main()
