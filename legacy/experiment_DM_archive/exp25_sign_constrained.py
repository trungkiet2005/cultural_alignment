#!/usr/bin/env python3
"""
EXP-25: Sign-Constrained EMA with Dampening (SCED)
===================================================
**Base**: EXP-09 (Hierarchical IS — SOTA MIS = 0.3975)

============================================================
THE SIGN-INCONSISTENCY PROBLEM IN EXP-09
============================================================
The single most persistent failure mode across all experiments is
**Mistral's NEGATIVE Pearson r** in every country. This means the IS
corrections are systematically pointing in the WRONG direction.

The root cause, from tracker analysis:
  - Mistral's WVS personas are skewed toward egalitarian responses
  - For some countries (USA, BRA), humans prefer HIGHER SV / utilitarian
    choices, but Mistral's personas predict LOWER
  - The IS correction delta_star is positive (IS finds personas prefer higher)
  - But Mistral's base model is already TOO HIGH on some dimensions (overshoot)

So the IS sometimes corrects rightward (+) when the TRUE direction is leftward (-)
for Mistral, and vice versa. This is the ANTI-CORRELATION pattern.

In EXP-09, these anti-correlated IS outputs still:
1. Enter the country prior: delta_country accumulates in the wrong direction
2. Get mixed into delta_opt_final: alpha_h amplifies the wrong direction
3. Create flip% because delta_opt_micro and delta_country SIGN-CONFLICT

============================================================
EXP-25 INNOVATION: Sign-Constrained EMA + Soft Dampening
============================================================
Two targeted interventions:

**PART A: Sign-Constrained EMA Update**
After each scenario, before updating delta_country, check if delta_opt_micro
AGREES IN SIGN with the current delta_country:

    agree = sign(delta_opt_micro) == sign(delta_country)   [after warmup]
    beta_update = BETA_EMA             if agree
               = BETA_EMA * BETA_ANTI  otherwise  (anti-aligned: slower update)

This means:
  - Scenarios that AGREE with the current prior direction → normal update
  - Scenarios that OPPOSE the current prior direction → reduced impact (BETA_ANTI)
  - The prior becomes "resistant to reversal" → accumulated direction is more stable
  - For Mistral: if the prior has correctly found a direction and an anti-aligned
    scenario comes in, it doesn't derail the prior

**PART B: Dampened Output for Anti-Aligned Predictions**
Before applying the country prior mixing:
    anti_aligned = sign(delta_opt_micro) ≠ sign(delta_country)   [after warmup]
    damp_factor = DAMP_FACTOR   if anti_aligned   (e.g. 0.40: reduce by 60%)
             = 1.0             otherwise

    delta_opt_micro_damped = delta_opt_micro * damp_factor
    delta_opt_final = alpha_h · delta_country + (1-alpha_h) · delta_opt_micro_damped

This soft dampening:
  - Reduces the probability that an anti-aligned IS output crosses the decision
    boundary (direct Flip% reduction)
  - Does NOT zero out the IS (unlike the hard guard) — just reduces its influence
  - During warmup: damp_factor=1.0 (no dampening → EXP-09 behaviour)
  - After warmup: dampening activates only when direction disagrees

**Mathematical grounding**:
This is a instance of Bayesian coherence prior:
    P(delta_opt_final | delta_country, delta_opt_micro)
    ∝ P(delta_opt_micro | delta_opt_final) · P(delta_opt_final | delta_country)

When sign(delta_opt_micro) ≠ sign(delta_country), the likelihood is LOW
(the micro IS estimate is inconsistent with the prior). Dampening the micro IS
is equivalent to down-weighting this low-likelihood evidence — a natural
Bayesian operation.

**Key property**: reduces to EXP-09 exactly when:
  - During warmup: both BETA_ANTI and DAMP_FACTOR inactive
  - After warmup with all-aligned scenarios: normal beta update + no dampening

Hyperparameters:
    BETA_ANTI  = 0.30   (fraction of BETA_EMA to use for anti-aligned updates: 30%)
    DAMP_FACTOR = 0.40  (multiply anti-aligned delta_opt_micro by 0.40)
    ALPHA_H_MIN_DAMP = 0.20  (minimum alpha_h before dampening activates)

Usage on Kaggle
---------------
    !python experiment_DM/exp25_sign_constrained.py
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
from typing import Dict, List, Optional

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
    CompareSpec, append_rows_csv, flatten_per_dim_alignment,
    print_alignment_table, print_metric_comparison, try_load_reference_comparison,
)
from src.config import SWAConfig, resolve_output_dir
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
EXP_ID   = "EXP-25"
EXP_NAME = "sign_constrained"

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

# EXP-09 hyperparameters (unchanged)
N_WARMUP  = 50
DECAY_TAU = 100
BETA_EMA  = 0.10

# EXP-25 specific
BETA_ANTI        = 0.30   # update fraction for anti-aligned scenarios (30% of BETA_EMA)
DAMP_FACTOR      = 0.40   # output dampening for anti-aligned delta_opt_micro
ALPHA_H_MIN_DAMP = 0.20   # dampening only activates when alpha_h > this

SWA_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/swa"
CMP_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/compare"
MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH     = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
HUMAN_AMCE_PATH   = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"


# ============================================================================
# Step 3: Sign-Constrained Prior State
# ============================================================================
class SignConstrainedPriorState:
    """
    EXP-09 CountryPriorState with two modifications:
    A) Sign-constrained EMA: β_eff = β if aligned, β·BETA_ANTI if anti-aligned
    B) Dampening tracker: records n_anti_aligned for diagnostics
    """

    def __init__(self):
        self.delta_country  = 0.0
        self.step           = 0
        self._history:      List[float] = []
        self._n_anti:       int = 0
        self._n_aligned:    int = 0

    def alpha_h(self) -> float:
        if self.step < N_WARMUP: return 0.0
        return 1.0 - np.exp(-(self.step - N_WARMUP) / DECAY_TAU)

    def is_anti_aligned(self, delta_opt_micro: float) -> bool:
        """After warmup: anti-aligned if opposite sign to delta_country (and both non-zero)."""
        if self.step < N_WARMUP: return False
        if abs(self.delta_country) < 1e-6: return False
        return (delta_opt_micro * self.delta_country) < 0  # opposite signs

    def damp_factor(self, delta_opt_micro: float) -> float:
        """Return output dampening multiplier (1.0 during warmup or if aligned)."""
        ah = self.alpha_h()
        if ah < ALPHA_H_MIN_DAMP: return 1.0
        if self.is_anti_aligned(delta_opt_micro): return DAMP_FACTOR
        return 1.0

    def update(self, delta_opt_micro: float) -> float:
        """
        Sign-constrained EMA update.
        Returns beta_eff used (for diagnostics).
        """
        anti    = self.is_anti_aligned(delta_opt_micro)
        beta_eff = BETA_EMA * (BETA_ANTI if anti else 1.0)
        self.delta_country = (1.0 - beta_eff) * self.delta_country + beta_eff * delta_opt_micro
        self._history.append(delta_opt_micro)
        if anti: self._n_anti    += 1
        else:    self._n_aligned += 1
        self.step += 1
        return beta_eff

    def apply_prior(self, delta_opt_micro_damped: float) -> float:
        """EXP-09 prior mixing with DAMPED micro IS result."""
        a = self.alpha_h()
        return a * self.delta_country + (1.0 - a) * delta_opt_micro_damped

    @property
    def stats(self) -> Dict:
        return {
            "step": self.step, "delta_country": self.delta_country, "alpha_h": self.alpha_h(),
            "n_anti": self._n_anti, "n_aligned": self._n_aligned,
            "anti_rate": self._n_anti / max(1, self.step - N_WARMUP),
            "history_std": float(np.std(self._history)) if len(self._history) > 1 else 0.0,
        }


_PRIOR_STATE: Dict[str, SignConstrainedPriorState] = {}


# ============================================================================
# Step 4: Sign-Constrained Controller
# ============================================================================
class Exp25SignConstrainedController(ImplicitSWAController):
    """
    EXP-09 + Sign-Constrained EMA (Part A) + Soft Output Dampening (Part B).

    Two small changes vs EXP-09, both in the prior step:
    A) delta_opt_micro_damped = delta_opt_micro * damp_factor (Part B)
    B) prior.update(delta_opt_micro): uses β·BETA_ANTI if anti-aligned (Part A)
    C) prior.apply_prior(delta_opt_micro_damped): mixing with damped value

    The IS itself (perturbations, PT value, ESS_guard) is IDENTICAL to EXP-09.
    """

    def __init__(self, *args, country: str = "UNKNOWN", **kwargs):
        super().__init__(*args, **kwargs)
        self.country = country

    def _get_prior(self) -> SignConstrainedPriorState:
        if self.country not in _PRIOR_STATE:
            _PRIOR_STATE[self.country] = SignConstrainedPriorState()
        return _PRIOR_STATE[self.country]

    def _pt_value(self, x: torch.Tensor) -> torch.Tensor:
        a, b, k = self.pt_alpha, self.pt_beta, self.pt_kappa
        return torch.where(x >= 0, x.abs().pow(a), -k * x.abs().pow(b))

    @torch.no_grad()
    def predict(self, user_query, preferred_on_right=True, phenomenon_category="default", lang="en"):
        # ── Identical to EXP-09 up to delta_opt_micro ────────────────────────
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
        K, device = self.K, self.device

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

        delta_star = (torch.sum(w * eps) if float(k_eff.item()) / K >= self.rho_eff
                      else torch.zeros((), device=device))

        delta_opt_micro = float((anchor + delta_star).item())

        # ── EXP-25 CHANGES (A + B in prior step only) ────────────────────────
        prior    = self._get_prior()
        df       = prior.damp_factor(delta_opt_micro)          # Part B: compute damp
        delta_opt_micro_damped = delta_opt_micro * df           # Part B: apply damp
        delta_opt_final = prior.apply_prior(delta_opt_micro_damped)  # EXP-09 mix (damped)
        beta_eff = prior.update(delta_opt_micro)               # Part A: sign-constrained β
        # ── End EXP-25 change ─────────────────────────────────────────────────

        st = prior.stats
        p_right = torch.sigmoid(torch.tensor(delta_opt_final / self.decision_temperature)).item()
        p_pref  = p_right if preferred_on_right else 1.0 - p_right
        variance = float(delta_agents.var(unbiased=True).item()) if delta_agents.numel() > 1 else 0.0

        return {
            "p_right": p_right, "p_left": 1.0 - p_right, "p_spare_preferred": p_pref,
            "variance": variance, "sigma_used": sigma,
            "mppi_flipped": (float(anchor.item()) > 0) != (delta_opt_final > 0),
            "delta_z_norm": abs(delta_opt_final - float(anchor.item())),
            "delta_consensus": float(anchor.item()), "delta_opt": delta_opt_final,
            "delta_opt_micro": delta_opt_micro,
            "delta_opt_micro_damped": delta_opt_micro_damped,
            "damp_factor": df, "beta_eff_update": beta_eff,
            "is_anti_aligned": int(df < 1.0),
            "delta_country": st["delta_country"], "alpha_h": st["alpha_h"],
            "anti_rate": st["anti_rate"], "prior_step": st["step"],
            "ess_ratio": float(k_eff.item()) / K,
            "logit_temp_used": logit_temp, "n_personas": delta_agents.numel(),
            "agent_decision_gaps": delta_agents.tolist(),
            "agent_rewards": (delta_agents - delta_base).tolist(),
            "p_spare_preferred_pass1": p_pref, "p_spare_preferred_pass2": p_pref,
            "positional_bias": 0.0,
        }


_swa_runner_mod.ImplicitSWAController = Exp25SignConstrainedController


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
    print(f"\n{'#'*70}\n# {EXP_ID} [{model_name}] Sign-Constrained EMA + Dampening\n{'#'*70}")

    rows = []
    for ci, country in enumerate(cfg.target_countries):
        if country not in SUPPORTED_COUNTRIES: continue
        _PRIOR_STATE.clear()
        _PRIOR_STATE[country] = SignConstrainedPriorState()
        print(f"\n[{ci+1}/{len(cfg.target_countries)}] {EXP_ID} {model_name} | {country} "
              f"(β_anti={BETA_ANTI}×β, damp={DAMP_FACTOR}, α_h_min={ALPHA_H_MIN_DAMP})")

        scen = _load_scen(cfg, country)
        personas = build_country_personas(country, wvs_path=WVS_DATA_PATH)

        orig_init = Exp25SignConstrainedController.__init__
        def patched_init(self, *a, country=country, **kw): orig_init(self, *a, country=country, **kw)
        Exp25SignConstrainedController.__init__ = patched_init
        _swa_runner_mod.ImplicitSWAController = Exp25SignConstrainedController

        results_df, summary = run_country_experiment(model, tokenizer, country, personas, scen, cfg)
        Exp25SignConstrainedController.__init__ = orig_init

        results_df.to_csv(out_dir / f"swa_results_{country}.csv", index=False)
        append_rows_csv(str(Path(CMP_ROOT) / "per_dim_breakdown.csv"),
                        flatten_per_dim_alignment(summary.get("per_dimension_alignment", {}),
                                                  model=model_name, method=f"{EXP_ID}_sign_constrained",
                                                  country=country))
        ps  = _PRIOR_STATE.get(country, SignConstrainedPriorState()).stats
        mea = lambda col: float(results_df[col].mean()) if col in results_df.columns else float("nan")

        rows.append({
            "model": model_name, "method": f"{EXP_ID}_sign_constrained", "country": country,
            **{f"align_{k}": v for k, v in summary["alignment"].items()},
            "flip_rate": summary["flip_rate"], "n_scenarios": summary["n_scenarios"],
            "final_delta_country": ps["delta_country"], "final_alpha_h": ps["alpha_h"],
            "final_anti_rate": ps["anti_rate"],
            "n_anti": ps["n_anti"], "n_aligned": ps["n_aligned"],
            "mean_damp_factor": mea("damp_factor"),
            "mean_is_anti": mea("is_anti_aligned"),
            "mean_ess_ratio": mea("ess_ratio"),
        })

        pda = summary.get("per_dimension_alignment", {})
        if pda:
            print(f"\n  ┌── Per-Dimension ({country}) ──")
            for dk, dd in sorted(pda.items()):
                hv, mv = dd.get("human", float("nan")), dd.get("model", float("nan"))
                print(f"  │  {dk:<25s}  human={hv:6.1f}  model={mv:6.1f}  err={mv-hv:+6.1f}pp")
            print(f"  └── MIS={summary['alignment']['mis']:.4f}  r={summary['alignment']['pearson_r']:+.3f}  "
                  f"Flip={summary['flip_rate']:.1%}")
            print(f"      anti_rate={ps['anti_rate']:.2%}  "
                  f"n(anti={ps['n_anti']}, aligned={ps['n_aligned']})  "
                  f"δ_cty={ps['delta_country']:+.4f}  α_h={ps['alpha_h']:.3f}")

        torch.cuda.empty_cache(); gc.collect()
    return rows

def main():
    setup_seeds(SEED)
    for d in (SWA_ROOT, CMP_ROOT): Path(d).mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}\n  {EXP_ID}: {EXP_NAME.upper()}  (base: EXP-09)\n{'='*70}")
    print(f"[PART A] β_eff = β if aligned; β·{BETA_ANTI} if anti-aligned  (sign-constrained EMA)")
    print(f"[PART B] damp = {DAMP_FACTOR} if anti-aligned AND α_h>{ALPHA_H_MIN_DAMP}  (output dampening)")
    print(f"[THEORY] Bayesian coherence: anti-aligned evidence → lower likelihood → damped weight")
    print(f"[TARGET] MIS < 0.3800 | Mistral Pearson r > 0 | anti_rate diagnostic")

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
              f"anti_rate={m_df['final_anti_rate'].mean():.2%}")
    print(f"\n  OVERALL MEAN MIS = {cmp_df['align_mis'].mean():.4f}  (EXP-09 SOTA: 0.3975)")
    ref = try_load_reference_comparison()
    if ref is not None:
        print_metric_comparison(ref, cmp_df, title=f"{EXP_ID} vs EXP-01",
                                spec=CompareSpec(metric_col="align_mis", ref_method="swa_ptis",
                                                 cur_method=f"{EXP_ID}_sign_constrained"))
    from experiment_DM.exp_reporting import print_tracker_ready_report
    print_tracker_ready_report(cmp_df, exp_id=EXP_ID,
                               per_dim_csv_path=str(Path(CMP_ROOT) / "per_dim_breakdown.csv"))
    print(f"\n[{EXP_ID}] DONE — {CMP_ROOT}")

if __name__ == "__main__":
    main()
