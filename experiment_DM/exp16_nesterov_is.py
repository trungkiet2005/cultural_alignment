#!/usr/bin/env python3
"""
EXP-16: Nesterov Momentum IS with Progressive PT Sharpening (NMIS)
====================================================================

**Novelty** (paper §3.2 extension + tracker insights):

**Paper §3.2 (IS update)**: Each scenario is treated INDEPENDENTLY.
    δ* = Σ_k w_k · ε_k      (standard IS, per-scenario E[ε|U > threshold])

**Paper §3.3 (EXP-09 hierarchical prior)**: Added EMA scalar memory
    δ_country = (1 - β) δ_country + β δ_opt   (country prior updated after each scenario)

**EXP-16 Innovation: Nesterov IS with Momentum Lookahead**

Insight from optimization theory: Nesterov Accelerated Gradient (NAG) achieves
faster convergence than vanilla gradient descent by computing the gradient at a
"lookahead" point (current position + momentum vector) instead of the current
position. We apply this idea to the IS correction process:

**Momentum vector** m_t: running EMA of IS corrections (not the full delta, just δ*)
    m_t = β_m · m_{t-1} + (1 - β_m) · δ*_{t-1}

**Nesterov lookahead anchor**: Instead of sampling around anchor = δ̄ (mean persona delta),
sample around the Nesterov lookahead point:
    anchor_NAG = δ̄ + γ_m · m_t

The PT gains are then computed relative to this lookahead anchor, not the standard
mean. This provides:
  1. **Momentum stabilisation**: in the direction that the IS has been consistently
     correcting, the proposal is "nudged" in that direction → faster convergence
     to the true cultural signal
  2. **Natural anti-correlation fix**: when IS has been consistently making -δ*
     corrections (anti-aligned), the momentum m_t < 0 nudges the next proposal
     toward m_t direction, providing memory about the systematic correction direction
  3. **Reduced exploration waste**: instead of sampling symmetrically around δ̄,
     samples are concentrated where the IS evidence says the optimal correction is

**Progressive PT Sharpening**:
    PT loss aversion parameter κ_t is annealed:
        κ_t = κ_low + (κ_high - κ_low) · (1 - exp(-t / τ_sharp))
    After warmup: κ starts at κ_low (more exploratory) and grows toward κ_high
    (more conservative) as the run accumulates evidence.
    This creates a natural curriculum: early scenarios explore widely; late scenarios
    commit to the emerging cultural signal.
    Connection to EXP-14: EXP-14 uses DIRECTION-conditioned κ; EXP-16 uses
    TIME-conditioned κ (orthogonal dimensions).

**Mathematical grounding** (Nesterov IS in logit space):

Classical NAG update rule:
    y_{t+1} = x_t + β(x_t - x_{t-1})   (lookahead)
    x_{t+1} = y_{t+1} - α · ∇f(y_{t+1})   (gradient step from lookahead)

EXP-16 IS analogue:
    m_t = β_m · m_{t-1} + (1-β_m) · δ*_t          (momentum EMA)
    anchor_NAG = δ̄ + γ_m · m_t                       (Nesterov lookahead anchor)
    ε_k ~ N(0, σ²), δ̃_k = anchor_NAG + ε_k           (IS proposal)
    δ*_{t+1} = Σ_k w_k · ε_k                         (IS update from lookahead)
    δ_opt = anchor_NAG + δ*_{t+1}                     (final prediction)

This is precisely the IS version of NAG where "gradient" = IS correction.
When m_t → 0 (IS consistently produces zero correction): NAG → standard IS.
When m_t → constant (systematic correction in one direction): NAG accelerates
convergence in that direction by factor (1/(1-β_m)) in the early steps.

**Combination with EXP-09 country prior**:
The country prior (EMA of delta_opt) provides the MACRO signal ← EXP-09.
The momentum provides the MICRO acceleration ← EXP-16.
EXP-16 integrates both: final prediction uses the hierarchical prior from EXP-09,
but the IS proposal itself uses the Nesterov lookahead anchor.

**Expected improvements**:
  - Mistral: momentum direction converges to the consistent IS direction even
    when it's small; lookahead prevents each scenario from starting "cold"
  - BRA (weakest in all experiments): momentum accumulated from early accurate
    scenarios guides later ones more effectively
  - JSD: Nesterov IS has faster convergence to the true cultural signal
    → higher Pearson r, lower JSD per-country
  - Flip%: momentum damping reduces wild flips (analogous to momentum SGD)
  - Overall MIS target: < 0.3700 (vs EXP-09 SOTA = 0.3975)

Usage on Kaggle
---------------
    !python experiment_DM/exp16_nesterov_is.py
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
    if not _on_kaggle():
        return
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
    CompareSpec,
    append_rows_csv,
    flatten_per_dim_alignment,
    print_alignment_table,
    print_metric_comparison,
    try_load_reference_comparison,
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
# Step 2: experiment configuration
# ============================================================================
EXP_ID   = "EXP-16"
EXP_NAME = "nesterov_is"

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

# ── Nesterov Momentum IS hyperparameters ──────────────────────────────────
BETA_MOMENTUM = 0.90     # EMA decay for momentum vector m_t (high = stable)
GAMMA_NAG     = 0.70     # lookahead strength: anchor_NAG = δ̄ + γ · m_t
N_WARMUP      = 50       # scenarios before momentum is applied (= EXP-09)
DECAY_TAU     = 100      # country prior annealing (= EXP-09)
BETA_EMA      = 0.10     # country prior EMA (= EXP-09)

# ── Progressive PT Sharpening (time-conditioned κ annealing) ──────────────
KAPPA_LOW     = 1.80     # κ at the beginning of run (more exploratory)
KAPPA_HIGH    = 2.80     # κ at end of run (more conservative)
TAU_KAPPA     = 150      # annealing speed for κ: κ_t approaches KAPPA_HIGH
                         # κ_t = κ_low + (κ_high - κ_low)*(1 - exp(-t/τ))

# PT curvature (unchanged from paper)
PT_ALPHA  = 0.88
PT_BETA   = 0.88

SWA_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/swa"
CMP_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/compare"
MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/"
    "WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
)
HUMAN_AMCE_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"


# ============================================================================
# Step 3: Nesterov IS State
# ============================================================================
class NesterovISState:
    """
    Maintains IS state across scenarios for one (model, country) pair:
      1. Momentum vector m_t (EMA of per-scenario IS corrections δ*)
      2. Country prior δ_country (EMA of full delta_opt, identical to EXP-09)
      3. Step counter for:
         - Warmup (no momentum/prior during first N_WARMUP scenarios)
         - Progressive κ annealing (κ_t increases with step)
    """

    def __init__(self, beta_m: float = BETA_MOMENTUM, gamma_nag: float = GAMMA_NAG,
                 beta_ema: float = BETA_EMA, decay_tau: float = DECAY_TAU,
                 n_warmup: int = N_WARMUP):
        self.m           = 0.0    # momentum vector (scalar in logit space)
        self.delta_country = 0.0  # country prior EMA (from EXP-09)
        self.beta_m      = beta_m
        self.gamma_nag   = gamma_nag
        self.beta_ema    = beta_ema
        self.decay_tau   = decay_tau
        self.n_warmup    = n_warmup
        self.step        = 0
        self._delta_star_history: List[float] = []
        self._delta_opt_history:  List[float] = []

    def kappa_t(self) -> float:
        """
        Progressive κ annealing: κ starts at KAPPA_LOW and increases to KAPPA_HIGH.
        During warmup: always KAPPA_LOW (maximally exploratory).
        """
        if self.step < self.n_warmup:
            return KAPPA_LOW
        t = self.step - self.n_warmup
        # Annealing: fast at start, saturates at KAPPA_HIGH
        frac = 1.0 - np.exp(-t / TAU_KAPPA)
        return KAPPA_LOW + (KAPPA_HIGH - KAPPA_LOW) * frac

    def alpha_h(self) -> float:
        """Country prior annealing weight (identical to EXP-09)."""
        if self.step < self.n_warmup:
            return 0.0
        t = self.step - self.n_warmup
        return 1.0 - np.exp(-t / self.decay_tau)

    def nag_anchor_correction(self) -> float:
        """
        Returns γ · m_t (the Nesterov lookahead correction).
        Before warmup: return 0 (= pure persona consensus).
        """
        if self.step < self.n_warmup:
            return 0.0
        return self.gamma_nag * self.m

    def update(self, delta_star: float, delta_opt: float) -> None:
        """
        Update momentum and country prior after processing one scenario.
        delta_star : IS correction (δ* from IS step, NOT the full delta_opt)
        delta_opt  : final prediction delta (what gets passed to sigmoid)
        """
        # Force host scalars so numpy stats and CSV export never touch CUDA tensors.
        delta_star_f = float(delta_star)
        delta_opt_f = float(delta_opt)

        # Momentum update (bias-corrected EMA)
        self.m = self.beta_m * self.m + (1.0 - self.beta_m) * delta_star_f

        # Country prior update (identical to EXP-09)
        self.delta_country = ((1.0 - self.beta_ema) * self.delta_country
                              + self.beta_ema * delta_opt_f)
        self._delta_star_history.append(delta_star_f)
        self._delta_opt_history.append(delta_opt_f)
        self.step += 1

    def apply_country_prior(self, delta_opt_micro: float) -> float:
        """Mix country prior and micro IS result (identical to EXP-09 hierarchical IS)."""
        a = self.alpha_h()
        return float(a * self.delta_country + (1.0 - a) * float(delta_opt_micro))

    @property
    def stats(self) -> Dict:
        hs = self._delta_star_history
        ho = self._delta_opt_history
        return {
            "step":            self.step,
            "momentum":        self.m,
            "delta_country":   self.delta_country,
            "alpha_h":         self.alpha_h(),
            "kappa_t":         self.kappa_t(),
            "delta_star_std":  float(np.std(hs)) if len(hs) > 1 else 0.0,
            "delta_opt_std":   float(np.std(ho)) if len(ho) > 1 else 0.0,
        }


_NAG_STATE: Dict[str, NesterovISState] = {}


# ============================================================================
# Step 4: Nesterov IS Controller
# ============================================================================
class Exp16NesterovISController(ImplicitSWAController):
    """
    Nesterov Momentum IS with Progressive PT Sharpening (NMIS).

    Key modifications to the paper's IS loop:
    1. IS proposal centred at Nesterov lookahead:  anchor_NAG = δ̄ + γ·m_t
    2. PT value function with time-conditioned κ:  κ_t = annealed schedule
    3. Country prior applied AFTER IS (same as EXP-09)
    4. Momentum updated with per-scenario δ* (pure IS correction term)
    """

    def __init__(self, *args, country: str = "UNKNOWN", **kwargs):
        super().__init__(*args, **kwargs)
        self.country = country

    def _get_nag_state(self) -> NesterovISState:
        if self.country not in _NAG_STATE:
            _NAG_STATE[self.country] = NesterovISState()
        return _NAG_STATE[self.country]

    def _pt_value_progressive(self, x: torch.Tensor, kappa: float) -> torch.Tensor:
        """
        PT value function with progressive κ (time-conditioned, not direction-conditioned).
        x     : (K,) or (K, N) normalised gains
        kappa : scalar, from NesterovISState.kappa_t()
        """
        alpha, beta = PT_ALPHA, PT_BETA
        return torch.where(
            x >= 0,
            x.abs().pow(alpha),
            -kappa * x.abs().pow(beta),
        )

    @torch.no_grad()
    def predict(self, user_query, preferred_on_right=True, phenomenon_category="default", lang="en"):
        # ── Step 1: Positional debiasing (unchanged from paper) ──
        db1, da1, logit_temp = self._extract_logit_gaps(user_query, phenomenon_category, lang)
        swapped_query, swap_changed = self._swap_positional_labels(user_query, lang)
        if swap_changed:
            db2, da2, _ = self._extract_logit_gaps(swapped_query, phenomenon_category, lang)
        else:
            db2, da2 = db1, da1

        delta_base   = (db1 - db2) / 2.0 if swap_changed else db1
        delta_agents = (da1 - da2) / 2.0 if swap_changed else da1   # (N,)

        # ── Step 2: Fetch Nesterov state ──
        nag_state = self._get_nag_state()
        kappa_t   = nag_state.kappa_t()
        nag_corr  = nag_state.nag_anchor_correction()    # γ·m_t (0 during warmup)

        # ── Step 3: Nesterov lookahead anchor ──
        anchor_unif = delta_agents.mean()                             # standard consensus δ̄
        anchor_nag  = anchor_unif + float(nag_corr)                  # lookahead anchor
        anchor_nag_t = anchor_unif.new_tensor(anchor_nag)

        sigma = max(
            float(delta_agents.std(unbiased=True).item()) if delta_agents.numel() >= 2 else 0.0,
            self.noise_std
        )
        K, device = self.K, self.device

        # Sample around NESTEROV anchor (key innovation: not around plain mean!)
        eps         = torch.randn(K, device=device) * sigma
        delta_tilde = anchor_nag_t + eps                              # (K,) Nesterov proposals

        # ── Step 4: Per-agent gains (σ-normalised, relative to Nesterov anchor) ──
        dist_base_to_i = (delta_base - delta_agents).abs()                               # (N,)
        dist_cand_to_i = (delta_tilde.unsqueeze(1) - delta_agents.unsqueeze(0)).abs()    # (K, N)
        g_per_agent    = (dist_base_to_i.unsqueeze(0) - dist_cand_to_i) / sigma          # (K, N)

        # ── Step 5: Progressive PT value (time-conditioned κ) ──
        v_per_agent = self._pt_value_progressive(g_per_agent, kappa_t)   # (K, N)
        mean_v      = v_per_agent.mean(dim=1)                              # (K,)

        # Consensus utility (uses Nesterov anchor as consensus target, not plain mean)
        g_cons = ((delta_base - anchor_nag_t).abs() - (delta_tilde - anchor_nag_t).abs()) / sigma
        v_cons = self._pt_value_progressive(g_cons, kappa_t)

        U     = (1.0 - self.lambda_coop) * mean_v + self.lambda_coop * v_cons
        w     = F.softmax(U / self.beta, dim=0)
        k_eff = 1.0 / torch.sum(w * w).clamp_min(1e-12)

        delta_star_raw = float(
            torch.sum(w * eps).item()
            if float(k_eff.item()) / K >= self.rho_eff
            else 0.0
        )

        # ── Step 6: Micro IS result (from Nesterov anchor) ──
        delta_opt_micro = anchor_nag + delta_star_raw

        # ── Step 7: Country prior mixing (EXP-09 hierarchical approach) ──
        delta_opt_final = nag_state.apply_country_prior(delta_opt_micro)

        # ── Step 8: Update Nesterov state ──
        nag_state.update(delta_star_raw, delta_opt_micro)   # update with δ* (not δ_opt)
        nag_stats = nag_state.stats

        p_right = torch.sigmoid(
            anchor_unif.new_tensor(float(delta_opt_final) / self.decision_temperature)
        ).item()
        p_pref  = p_right if preferred_on_right else 1.0 - p_right
        variance = float(delta_agents.var(unbiased=True).item()) if delta_agents.numel() > 1 else 0.0

        return {
            "p_right": p_right, "p_left": 1.0 - p_right, "p_spare_preferred": p_pref,
            "variance": variance, "sigma_used": sigma,
            "mppi_flipped": (float(anchor_nag_t.item()) > 0) != (delta_opt_final > 0),
            "delta_z_norm": abs(delta_opt_final - float(anchor_nag_t.item())),
            "delta_consensus": float(anchor_nag_t.item()), "delta_opt": delta_opt_final,
            # NMIS diagnostics
            "delta_opt_micro": delta_opt_micro,
            "delta_star":      delta_star_raw,
            "momentum":        nag_stats["momentum"],
            "nag_correction":  nag_corr,
            "kappa_t":         kappa_t,
            "delta_country":   nag_stats["delta_country"],
            "alpha_h":         nag_stats["alpha_h"],
            "prior_step":      nag_stats["step"],
            "ess_ratio":       float(k_eff.item()) / K,
            "logit_temp_used": logit_temp,
            "n_personas":      delta_agents.numel(),
            "agent_decision_gaps": delta_agents.tolist(),
            "agent_rewards":       (delta_agents - delta_base).tolist(),
            "p_spare_preferred_pass1": p_pref,
            "p_spare_preferred_pass2": p_pref,
            "positional_bias": 0.0,
        }


_swa_runner_mod.ImplicitSWAController = Exp16NesterovISController


# ============================================================================
# Step 5: Runner
# ============================================================================
def _free_model_cache(model_name: str):
    safe = "models--" + model_name.replace("/", "--")
    for root in [os.environ.get("HF_HUB_CACHE"), os.environ.get("HF_HOME"),
                 os.path.expanduser("~/.cache/huggingface"), "/root/.cache/huggingface"]:
        if not root:
            continue
        hub_dir = root if os.path.basename(root.rstrip("/")) == "hub" else os.path.join(root, "hub")
        target  = os.path.join(hub_dir, safe)
        if os.path.isdir(target):
            try:
                shutil.rmtree(target)
                print(f"[CLEANUP] removed {target}")
            except Exception as e:
                print(f"[CLEANUP] error: {e}")


def _build_swa_config(model_name: str) -> SWAConfig:
    return SWAConfig(
        model_name=model_name,
        n_scenarios=N_SCENARIOS,
        batch_size=BATCH_SIZE,
        target_countries=list(TARGET_COUNTRIES),
        load_in_4bit=True,
        use_real_data=True,
        multitp_data_path=MULTITP_DATA_PATH,
        wvs_data_path=WVS_DATA_PATH,
        human_amce_path=HUMAN_AMCE_PATH,
        output_dir=SWA_ROOT,
        lambda_coop=LAMBDA_COOP,
        K_samples=128,
    )


def _load_country_scenarios(cfg: SWAConfig, country: str):
    lang = COUNTRY_LANG.get(country, "en")
    if cfg.use_real_data:
        df = load_multitp_dataset(
            data_base_path=cfg.multitp_data_path, lang=lang,
            translator=cfg.multitp_translator, suffix=cfg.multitp_suffix,
            n_scenarios=cfg.n_scenarios,
        )
    else:
        df = generate_multitp_scenarios(cfg.n_scenarios, lang=lang)
    df = df.copy()
    df["lang"] = lang
    return df


def _run_swa_for_model(model, tokenizer, model_name: str) -> List[dict]:
    cfg     = _build_swa_config(model_name)
    out_dir = Path(SWA_ROOT) / resolve_output_dir("", model_name).strip("/\\")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(out_dir)
    print(f"\n{'#'*70}\n# {EXP_ID} [{model_name}] Nesterov IS + Progressive PT Sharpening\n{'#'*70}")

    rows = []
    for ci, country in enumerate(cfg.target_countries):
        if country not in SUPPORTED_COUNTRIES:
            continue

        # Reset Nesterov state per country
        _NAG_STATE.clear()
        _NAG_STATE[country] = NesterovISState()
        print(f"\n[{ci+1}/{len(cfg.target_countries)}] {EXP_ID} {model_name} | {country} "
              f"(NMIS: β_m={BETA_MOMENTUM}, γ={GAMMA_NAG}, "
              f"κ: {KAPPA_LOW}→{KAPPA_HIGH} over τ={TAU_KAPPA})")

        scen     = _load_country_scenarios(cfg, country)
        personas = build_country_personas(country, wvs_path=WVS_DATA_PATH)
        print(f"  [PERSONAS] N={len(personas)}")

        orig_init = Exp16NesterovISController.__init__
        def patched_init(self, *a, country=country, **kw):
            orig_init(self, *a, country=country, **kw)
        Exp16NesterovISController.__init__ = patched_init
        _swa_runner_mod.ImplicitSWAController = Exp16NesterovISController

        results_df, summary = run_country_experiment(model, tokenizer, country, personas, scen, cfg)
        Exp16NesterovISController.__init__ = orig_init

        results_df.to_csv(out_dir / f"swa_results_{country}.csv", index=False)
        append_rows_csv(
            str(Path(CMP_ROOT) / "per_dim_breakdown.csv"),
            flatten_per_dim_alignment(
                summary.get("per_dimension_alignment", {}),
                model=model_name,
                method=f"{EXP_ID}_nesterov_is",
                country=country,
            ),
        )

        nag_stats    = _NAG_STATE.get(country, NesterovISState()).stats
        mean_ess     = (float(results_df["ess_ratio"].mean())
                        if "ess_ratio" in results_df.columns else float("nan"))
        mean_kappa   = (float(results_df["kappa_t"].mean())
                        if "kappa_t" in results_df.columns else float("nan"))
        mean_momentum = (float(results_df["momentum"].mean())
                         if "momentum" in results_df.columns else float("nan"))

        rows.append({
            "model":   model_name,
            "method":  f"{EXP_ID}_nesterov_is",
            "country": country,
            **{f"align_{k}": v for k, v in summary["alignment"].items()},
            "flip_rate":           summary["flip_rate"],
            "mean_latency_ms":     summary["mean_latency_ms"],
            "n_scenarios":         summary["n_scenarios"],
            "mean_ess_ratio":      mean_ess,
            "mean_kappa":          mean_kappa,
            "mean_momentum":       mean_momentum,
            "final_delta_country": nag_stats["delta_country"],
            "final_alpha_h":       nag_stats["alpha_h"],
            "final_kappa_t":       nag_stats["kappa_t"],
            "beta_momentum":       BETA_MOMENTUM,
            "gamma_nag":           GAMMA_NAG,
            "kappa_low":           KAPPA_LOW,
            "kappa_high":          KAPPA_HIGH,
        })

        # ── Detailed per-dimension log ──
        pda = summary.get("per_dimension_alignment", {})
        if pda:
            print(f"\n  ┌── Per-Dimension Alignment ({country}) ──")
            for dim_key, dim_data in sorted(pda.items()):
                hv  = dim_data.get("human", float("nan"))
                mv  = dim_data.get("model", float("nan"))
                err = dim_data.get("error", mv - hv)
                print(f"  │  {dim_key:<25s}  human={hv:6.1f}  model={mv:6.1f}  err={err:+6.1f}pp")
            print(f"  └── MIS={summary['alignment']['mis']:.4f}  "
                  f"JSD={summary['alignment']['jsd']:.4f}  "
                  f"r={summary['alignment']['pearson_r']:+.3f}  "
                  f"MAE={summary['alignment']['mae']:.2f}  Flip={summary['flip_rate']:.1%}")
            print(f"      m_final={nag_stats['momentum']:+.4f}  "
                  f"κ_final={nag_stats['kappa_t']:.3f}  "
                  f"δ_country={nag_stats['delta_country']:.4f}  "
                  f"ESS={mean_ess:.3f}  α_h={nag_stats['alpha_h']:.3f}")

        torch.cuda.empty_cache()
        gc.collect()

    return rows


def main():
    setup_seeds(SEED)
    for d in (SWA_ROOT, CMP_ROOT):
        Path(d).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  {EXP_ID}: {EXP_NAME.upper()} — Nesterov IS + Progressive PT Sharpening")
    print(f"{'='*70}")
    print(f"[CONFIG] β_momentum={BETA_MOMENTUM}, γ_NAG={GAMMA_NAG}")
    print(f"[CONFIG] κ schedule: {KAPPA_LOW} → {KAPPA_HIGH} over τ={TAU_KAPPA} steps")
    print(f"[CONFIG] Country prior: N_warmup={N_WARMUP}, decay_τ={DECAY_TAU}, β_ema={BETA_EMA}")
    print(f"[THEORY] anchor_NAG = δ̄ + γ·m_t  (Nesterov lookahead in logit space)")
    print(f"[THEORY] m_t = β_m·m_{{t-1}} + (1-β_m)·δ*_t  (momentum EMA on IS corrections)")
    print(f"[THEORY] κ_t = {KAPPA_LOW} + ({KAPPA_HIGH}-{KAPPA_LOW})·(1-exp(-t/{TAU_KAPPA}))  (progressive sharpening)")
    print(f"[TARGET] Mean MIS < 0.3700 | Mistral Pearson r > 0 | Flip% < 12%")

    all_rows: List[dict] = []
    for mi, model_name in enumerate(MODELS):
        print(f"\n{'='*70}\n  MODEL {mi+1}/{len(MODELS)}: {model_name}\n{'='*70}")
        model, tokenizer = load_model(model_name, max_seq_length=2048, load_in_4bit=True)
        try:
            all_rows.extend(_run_swa_for_model(model, tokenizer, model_name))
        finally:
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _free_model_cache(model_name)
        pd.DataFrame(all_rows).to_csv(Path(CMP_ROOT) / "comparison.csv", index=False)

    cmp_df = pd.DataFrame(all_rows)
    cmp_df.to_csv(Path(CMP_ROOT) / "comparison.csv", index=False)

    print(f"\n\n{'#'*70}")
    print(f"# {EXP_ID} FINAL REPORT — {EXP_NAME.upper()}")
    print(f"{'#'*70}")
    print_alignment_table(cmp_df, title=f"{EXP_ID} RESULTS — {EXP_NAME}")

    print(f"\n{'─'*70}")
    for model_name in MODELS:
        m_df = cmp_df[cmp_df["model"] == model_name]
        if m_df.empty:
            continue
        short      = model_name.split("/")[-1][:20]
        mis_mean   = m_df["align_mis"].mean()
        jsd_mean   = m_df["align_jsd"].mean()
        r_mean     = m_df["align_pearson_r"].mean()
        flip_mean  = m_df["flip_rate"].mean()
        kappa_mean = m_df["mean_kappa"].mean()
        mom_mean   = m_df["mean_momentum"].mean()
        print(f"  {short:<20s}  MIS={mis_mean:.4f}  JSD={jsd_mean:.4f}  "
              f"r={r_mean:+.3f}  Flip={flip_mean:.1%}  κ={kappa_mean:.3f}  m={mom_mean:+.4f}")

    overall_mis = cmp_df["align_mis"].mean()
    print(f"\n  OVERALL MEAN MIS = {overall_mis:.4f}  (EXP-09 SOTA: 0.3975)")

    ref = try_load_reference_comparison()
    if ref is not None:
        print_metric_comparison(
            ref, cmp_df,
            title=f"{EXP_ID} vs EXP-01 (reference) — MIS",
            spec=CompareSpec(
                metric_col="align_mis",
                ref_method="swa_ptis",
                cur_method=f"{EXP_ID}_nesterov_is",
            ),
        )

    # ── Paper-ready table ──
    print(f"\n{'─'*70}")
    print(f"  PAPER-READY TABLE (copy to tracker)")
    print(f"{'─'*70}")
    print(f"\n| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% | κ_final | m_final |")
    print(f"|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|:-------:|:-------:|")
    for _, row in cmp_df.iterrows():
        short = row["model"].split("/")[-1].split("-Instruct")[0].split("-instruct")[0]
        print(f"| {short} | {row['country']} | {row['align_mis']:.4f} | "
              f"{row['align_jsd']:.4f} | {row['align_pearson_r']:+.3f} | "
              f"{row['align_mae']:.2f} | {row['flip_rate']:.1%} | "
              f"{row.get('final_kappa_t', float('nan')):.3f} | "
              f"{row.get('mean_momentum', float('nan')):+.4f} |")

    from experiment_DM.exp_reporting import print_tracker_ready_report
    print_tracker_ready_report(
        cmp_df, exp_id=EXP_ID,
        per_dim_csv_path=str(Path(CMP_ROOT) / "per_dim_breakdown.csv"),
    )
    print(f"\n[{EXP_ID}] DONE — results under {CMP_ROOT}")


if __name__ == "__main__":
    main()
