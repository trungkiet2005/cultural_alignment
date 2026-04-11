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
    !python experiment/test_50_countries.py

For each model: runs **vanilla** (token-logit baseline) on every country in
``SWAConfig.target_countries`` (~50), then EXP-24 dual-pass, then prints MIS
comparison vs ``baseline_vanilla`` using ``compare/vanilla_reference.csv``.
Set ``SKIP_VANILLA = True`` to reuse a previously saved ``vanilla_reference.csv``.
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
from typing import Dict, List

import numpy as np
import torch
try:
    torch._dynamo.config.disable = True
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass
import torch.nn.functional as F
import pandas as pd

from experiment_DM.exp_reporting import (
    CompareSpec,
    DEFAULT_EXP01_REFERENCE_CSV_CANDIDATES,
    append_rows_csv,
    flatten_per_dim_alignment,
    print_alignment_table,
    print_metric_comparison,
    try_load_reference_comparison,
)
from src.baseline_runner import run_baseline_vanilla
from src.config import BaselineConfig, SWAConfig, resolve_output_dir
from src.constants import COUNTRY_LANG
from src.model import setup_seeds, load_model
from src.data import load_multitp_dataset
from src.scenarios import generate_multitp_scenarios
from src.personas import build_country_personas, SUPPORTED_COUNTRIES
from src.controller import ImplicitSWAController
import src.swa_runner as _swa_runner_mod
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
# Load canonical 50-country WVS target set from shared config instead of hardcoding.
TARGET_COUNTRIES: List[str] = list(SWAConfig().target_countries)
N_SCENARIOS: int = 500
BATCH_SIZE:  int = 1
SEED:        int = 42
LAMBDA_COOP: float = 0.70

# EXP-09 hyperparameters (unchanged)
N_WARMUP  = 50
DECAY_TAU = 100
BETA_EMA  = 0.10

# EXP-24 specific
K_HALF     = 64     # samples per pass (2 × K_HALF = 128 = EXP-09 total)
VAR_SCALE  = 0.04   # soft reliability: r = exp(-bootstrap_var / VAR_SCALE)

SWA_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/swa"
BASE_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/baseline"
CMP_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/compare"
# Full vanilla run for every (model × country) in TARGET_COUNTRIES; saved here for EXP-24 vs vanilla.
VANILLA_REF_CSV = Path(CMP_ROOT) / "vanilla_reference.csv"
# Set True only if vanilla_reference.csv already exists from a prior run (same models + countries).
SKIP_VANILLA: bool = False
MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH     = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
HUMAN_AMCE_PATH   = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"


# ============================================================================
# Step 3: EXP-09 Prior State  (identical)
# ============================================================================
class BootstrapPriorState:
    """Minimal EXP-09 CountryPriorState (no changes needed here)."""

    def __init__(self):
        self.delta_country = 0.0
        self.step          = 0
        self._history: List[float] = []

    def alpha_h(self) -> float:
        if self.step < N_WARMUP: return 0.0
        return 1.0 - np.exp(-(self.step - N_WARMUP) / DECAY_TAU)

    def update(self, delta_opt_micro: float) -> None:
        self.delta_country = (1.0 - BETA_EMA) * self.delta_country + BETA_EMA * delta_opt_micro
        self._history.append(delta_opt_micro)
        self.step += 1

    def apply_prior(self, delta_opt_micro: float) -> float:
        a = self.alpha_h()
        return a * self.delta_country + (1.0 - a) * delta_opt_micro

    @property
    def stats(self) -> Dict:
        return {
            "step": self.step, "delta_country": self.delta_country, "alpha_h": self.alpha_h(),
            "history_std": float(np.std(self._history)) if len(self._history) > 1 else 0.0,
        }


_PRIOR_STATE: Dict[str, BootstrapPriorState] = {}


# ============================================================================
# Step 4: Dual-Pass Bootstrap IS Controller
# ============================================================================
class Exp24DualPassController(ImplicitSWAController):
    """
    EXP-09 with Dual-Pass Bootstrap IS Reliability Filter.

    Two changes vs EXP-09:
    1. IS split into two independent passes (K_HALF=64 each, same total K=128)
    2. Hard ESS guard replaced by soft reliability weight:
           r = exp(-bootstrap_var / VAR_SCALE)
           delta_star = r · (delta_star_1 + delta_star_2) / 2
    Country prior update and prior mixing identical to EXP-09.
    """

    def __init__(self, *args, country_iso: str = "UNKNOWN", **kwargs):
        super().__init__(*args, country_iso=country_iso, **kwargs)
        self.country = country_iso

    def _get_prior(self) -> BootstrapPriorState:
        if self.country not in _PRIOR_STATE:
            _PRIOR_STATE[self.country] = BootstrapPriorState()
        return _PRIOR_STATE[self.country]

    def _pt_value(self, x: torch.Tensor) -> torch.Tensor:
        a, b, k = self.pt_alpha, self.pt_beta, self.pt_kappa
        return torch.where(x >= 0, x.abs().pow(a), -k * x.abs().pow(b))

    def _single_is_pass(
        self,
        delta_base: torch.Tensor,
        delta_agents: torch.Tensor,
        anchor: torch.Tensor,
        sigma: float,
        K: int,
        device: torch.device,
    ) -> tuple:
        """Run one IS pass. Returns (delta_star, k_eff_ratio)."""
        eps         = torch.randn(K, device=device) * sigma
        delta_tilde = anchor + eps

        dist_base_to_i = (delta_base - delta_agents).abs()
        dist_cand_to_i = (delta_tilde.unsqueeze(1) - delta_agents.unsqueeze(0)).abs()
        g_per_agent    = (dist_base_to_i.unsqueeze(0) - dist_cand_to_i) / sigma
        v_per_agent    = self._pt_value(g_per_agent)
        mean_v         = v_per_agent.mean(dim=1)

        g_cons = ((delta_base - anchor).abs() - (delta_tilde - anchor).abs()) / sigma
        v_cons = self._pt_value(g_cons)
        U      = (1.0 - self.lambda_coop) * mean_v + self.lambda_coop * v_cons
        w      = F.softmax(U / self.beta, dim=0)
        k_eff  = 1.0 / torch.sum(w * w).clamp_min(1e-12)
        ess_r  = float(k_eff.item()) / K

        delta_star = (torch.sum(w * eps) if ess_r >= self.rho_eff
                      else torch.zeros((), device=device))
        return delta_star, ess_r

    @torch.no_grad()
    def predict(self, user_query, preferred_on_right=True, phenomenon_category="default", lang="en"):
        db1, da1, logit_temp = self._extract_logit_gaps(user_query, phenomenon_category, lang)
        swapped_query, swap_changed = self._swap_positional_labels(user_query, lang)
        if swap_changed:
            db2, da2, _ = self._extract_logit_gaps(swapped_query, phenomenon_category, lang)
        else:
            db2, da2 = db1, da1
        delta_base   = (db1 - db2) / 2.0 if swap_changed else db1
        delta_agents = (da1 - da2) / 2.0 if swap_changed else da1

        sigma = max(
            float(delta_agents.std(unbiased=True).item()) if delta_agents.numel() >= 2 else 0.0,
            self.noise_std)
        anchor = delta_agents.mean()
        device = self.device

        # ── EXP-24: Dual-pass IS ───────────────────────────────────────────────
        ds1, ess1 = self._single_is_pass(delta_base, delta_agents, anchor, sigma, K_HALF, device)
        ds2, ess2 = self._single_is_pass(delta_base, delta_agents, anchor, sigma, K_HALF, device)

        bootstrap_var = float((ds1 - ds2).pow(2).item())
        r             = float(np.exp(-bootstrap_var / VAR_SCALE))         # soft reliability ∈ (0,1]
        delta_star    = r * (ds1 + ds2) / 2.0                             # soft-weighted mean
        # ── End EXP-24 change ─────────────────────────────────────────────────

        delta_opt_micro = float((anchor + delta_star).item())
        prior           = self._get_prior()
        delta_opt_final = prior.apply_prior(delta_opt_micro)
        prior.update(delta_opt_micro)
        st = prior.stats

        p_right = torch.sigmoid(torch.tensor(delta_opt_final / self.decision_temperature)).item()
        p_pref  = p_right if preferred_on_right else 1.0 - p_right
        variance = float(delta_agents.var(unbiased=True).item()) if delta_agents.numel() > 1 else 0.0

        def _p_pref_micro(d_s: torch.Tensor) -> float:
            dm = float((anchor + d_s).item())
            pr = torch.sigmoid(torch.tensor(dm / self.decision_temperature)).item()
            return pr if preferred_on_right else 1.0 - pr

        return {
            "p_right": p_right, "p_left": 1.0 - p_right, "p_spare_preferred": p_pref,
            "variance": variance, "sigma_used": sigma,
            "mppi_flipped": (float(anchor.item()) > 0) != (delta_opt_final > 0),
            "delta_z_norm": abs(delta_opt_final - float(anchor.item())),
            "delta_consensus": float(anchor.item()), "delta_opt": delta_opt_final,
            "delta_opt_micro": delta_opt_micro,
            "delta_star_1": float(ds1.item()), "delta_star_2": float(ds2.item()),
            "bootstrap_var": bootstrap_var, "reliability_r": r,
            "ess_pass1": ess1, "ess_pass2": ess2,
            "delta_country": st["delta_country"], "alpha_h": st["alpha_h"],
            "prior_step": st["step"],
            "logit_temp_used": logit_temp, "n_personas": delta_agents.numel(),
            "agent_decision_gaps": delta_agents.tolist(),
            "agent_rewards": (delta_agents - delta_base).tolist(),
            "p_spare_preferred_is_pass1_micro": _p_pref_micro(ds1),
            "p_spare_preferred_is_pass2_micro": _p_pref_micro(ds2),
            "positional_bias": 0.0,
        }


_swa_runner_mod.ImplicitSWAController = Exp24DualPassController


# ============================================================================
# Step 5: Runner
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


def _build_baseline_config(model_name: str) -> BaselineConfig:
    return BaselineConfig(
        model_name=model_name, n_scenarios=N_SCENARIOS, batch_size=BATCH_SIZE,
        target_countries=list(TARGET_COUNTRIES), load_in_4bit=True, use_real_data=True,
        multitp_data_path=MULTITP_DATA_PATH, wvs_data_path=WVS_DATA_PATH,
        human_amce_path=HUMAN_AMCE_PATH, output_dir=BASE_ROOT,
    )


def _run_baseline_for_model(model, tokenizer, model_name: str) -> List[dict]:
    cfg = _build_baseline_config(model_name)
    out_dir = Path(BASE_ROOT) / resolve_output_dir("", model_name).strip("/\\")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(out_dir)
    print(f"\n{'#'*70}\n# VANILLA BASELINE [{model_name}] -> {out_dir}\n{'#'*70}")

    rows: List[dict] = []
    for ci, country in enumerate(cfg.target_countries):
        if country not in SUPPORTED_COUNTRIES:
            continue
        print(f"\n[{ci+1}/{len(cfg.target_countries)}] Vanilla {model_name} | {country}")
        scen = _load_scen(cfg, country)
        bl = run_baseline_vanilla(model, tokenizer, scen, country, cfg)
        bl["results_df"].to_csv(out_dir / f"vanilla_results_{country}.csv", index=False)
        rows.append({
            "model": model_name, "method": "baseline_vanilla", "country": country,
            **{f"align_{k}": v for k, v in bl["alignment"].items()},
            "n_scenarios": len(bl["results_df"]),
        })
        torch.cuda.empty_cache()
        gc.collect()
    return rows

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
        _PRIOR_STATE.clear()
        _PRIOR_STATE[country] = BootstrapPriorState()
        print(f"\n[{ci+1}/{len(cfg.target_countries)}] {EXP_ID} {model_name} | {country} "
              f"(K_half={K_HALF}×2={K_HALF*2}, VAR_SCALE={VAR_SCALE})")

        scen = _load_scen(cfg, country)
        personas = build_country_personas(country, wvs_path=WVS_DATA_PATH)

        _swa_runner_mod.ImplicitSWAController = Exp24DualPassController

        results_df, summary = run_country_experiment(model, tokenizer, country, personas, scen, cfg)

        results_df.to_csv(out_dir / f"swa_results_{country}.csv", index=False)
        append_rows_csv(str(Path(CMP_ROOT) / "per_dim_breakdown.csv"),
                        flatten_per_dim_alignment(summary.get("per_dimension_alignment", {}),
                                                  model=model_name, method=f"{EXP_ID}_dual_pass",
                                                  country=country))
        ps  = _PRIOR_STATE.get(country, BootstrapPriorState()).stats
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
    for d in (SWA_ROOT, BASE_ROOT, CMP_ROOT):
        Path(d).mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}\n  {EXP_ID}: {EXP_NAME.upper()}  (base: EXP-09)\n{'='*70}")
    print(f"[THEORY] Pass1+Pass2: K_half={K_HALF} each → same total K={K_HALF*2} as EXP-09")
    print(f"[THEORY] r = exp(-(δ*₁-δ*₂)² / {VAR_SCALE})  (soft reliability weight)")
    print(f"[THEORY] δ* = r · (δ*₁+δ*₂)/2  (replaces binary ESS guard)")
    print(f"[TARGET] MIS < 0.3800 | Mistral r > 0 | mean_reliability_r > 0.60")

    all_rows: List[dict] = []
    vanilla_rows: List[dict] = []
    for mi, model_name in enumerate(MODELS):
        print(f"\n{'='*70}\n  MODEL {mi+1}/{len(MODELS)}: {model_name}\n{'='*70}")
        model, tokenizer = load_model(model_name, max_seq_length=2048, load_in_4bit=True)
        try:
            if not SKIP_VANILLA:
                vanilla_rows.extend(_run_baseline_for_model(model, tokenizer, model_name))
                pd.DataFrame(vanilla_rows).to_csv(VANILLA_REF_CSV, index=False)
            all_rows.extend(_run_model(model, tokenizer, model_name))
        finally:
            del model, tokenizer; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            _free_model_cache(model_name)
        pd.DataFrame(all_rows).to_csv(Path(CMP_ROOT) / "comparison.csv", index=False)

    if not SKIP_VANILLA and vanilla_rows:
        pd.DataFrame(vanilla_rows).to_csv(VANILLA_REF_CSV, index=False)
        print(f"\n[SAVE] vanilla reference ({len(vanilla_rows)} rows) -> {VANILLA_REF_CSV}")
    elif SKIP_VANILLA and not VANILLA_REF_CSV.is_file():
        print(f"\n[WARN] SKIP_VANILLA=True but missing {VANILLA_REF_CSV} — vs-vanilla table may be empty.")

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
    ref_candidates = (str(VANILLA_REF_CSV),) + tuple(DEFAULT_EXP01_REFERENCE_CSV_CANDIDATES)
    ref = try_load_reference_comparison(candidates=ref_candidates)
    if ref is not None and "method" in ref.columns:
        n_v = int((ref["method"] == "baseline_vanilla").sum())
        print(f"\n[REF] Loaded reference CSV with baseline_vanilla rows: {n_v}")
    if ref is not None:
        print_metric_comparison(
            ref,
            cmp_df,
            title=f"{EXP_ID} vs Vanilla (50-country ref)",
            spec=CompareSpec(
                metric_col="align_mis",
                ref_method="baseline_vanilla",
                cur_method=f"{EXP_ID}_dual_pass",
            ),
        )
    from experiment_DM.exp_reporting import print_tracker_ready_report
    print_tracker_ready_report(cmp_df, exp_id=EXP_ID,
                               per_dim_csv_path=str(Path(CMP_ROOT) / "per_dim_breakdown.csv"))
    print(f"\n[{EXP_ID}] DONE — {CMP_ROOT}")

if __name__ == "__main__":
    main()
