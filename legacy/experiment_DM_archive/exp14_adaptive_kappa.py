#!/usr/bin/env python3
"""
EXP-14: Direction-Conditioned Adaptive Loss Aversion (DCAL)
============================================================

**Novelty** (derivation from paper + tracker analysis):

**Paper §3.2** establishes that the Prospect-Theory value function with κ=2.25
(loss-averse) is the "load-bearing ingredient". The canonical value is:
    v(x) = x^α              if x ≥ 0  (gain)
    v(x) = -κ · |x|^β      if x < 0  (loss)
with α=β=0.88, κ=2.25 **fixed**.

**Tracker diagnosis**: The worst failure mode observed is Mistral's *systematic
anti-correlation* (Pearson r < 0 across all 5 countries). When each scenario
is processed independently (EXP-01, EXP-09), the IS has no memory of whether
its corrections are systematically going in the wrong direction country-wide.
EXP-09's hierarchical prior (EMA delta_country) provides a macro-level signal,
but the PT value function still uses the SAME κ regardless of whether the
current scenario's IS is aligned with or opposed to this accumulated signal.

**EXP-14 Hypothesis**: The κ parameter should be DIRECTION-CONDITIONED:
  - When the IS correction delta_opt aligns WITH the emerging country prior
    (sign(delta_opt_micro) == sign(delta_country)): κ_low (e.g. 1.5) — less
    loss-aversion so the IS can push further in the confirmed direction
  - When the IS correction is ANTI-ALIGNED with the country prior:
    κ_high (e.g. 3.5) — stronger loss-aversion as a "brake" preventing the IS
    from doubling down in the wrong direction
Additionally, a **soft sign consistency** term penalises candidates that would
flip the prediction sign relative to the current country prior direction.

**Mathematical grounding** (paper §3.2 extension):

    κ_t = κ_base + γ · [sign(delta_star_t) ≠ sign(delta_country_t)]
                 · (1 - exp(-step / τ_kappa))

The step-dependent term means:
  - During warmup (step < N_warmup): κ = κ_base (no adaptation, pure exploration)
  - After warmup: κ increases by γ exactly when IS correction opposes country prior
  - At convergence (step → ∞): κ → κ_base + γ for anti-aligned scenarios

**Soft sign consistency penalty** added to collective utility:
    U_aug = U_total + λ_sign · v_sign(delta_tilde_k)
    v_sign(x) = -κ_sign · |x|^β     if sign(x) ≠ sign(delta_country)
                x^α                   otherwise

This design has three important properties:
1. During warmup (no country prior): κ = κ_base (= EXP-01 behaviour)
2. After warmup with aligned correction: κ < κ_base (more exploratory) 
3. After warmup with anti-aligned correction: κ > κ_base (more conservative)

**Expected improvements**:
  - Mistral: country prior direction signal causes IS to stop chasing anti-
    correlated directions → Pearson r should flip positive
  - SocialValue gap: aligned with EXP-03 personas, κ_low allows larger push
  - Flip%: κ_high for anti-aligned cases prevents wild flips (EXP-09 problem)
  - Mean MIS target: < 0.3800 (vs EXP-09 SOTA = 0.3975)

**Connection to EXP-09**: Builds on EXP-09's CountryPriorState (EMA + annealing)
but uses it to MODULATE the PT value function itself, not just post-hoc blend.
This closes the loop: the IS update now "knows" whether it's going in the
country-consistent direction and adjusts its loss-aversion accordingly.

Usage on Kaggle
---------------
    !python experiment_DM/exp14_adaptive_kappa.py
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
from typing import Dict, List, Optional, Tuple

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
EXP_ID   = "EXP-14"
EXP_NAME = "adaptive_kappa"

MODELS: List[str] = [
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "unsloth/gemma-2-9b-it-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
]
TARGET_COUNTRIES: List[str] = ["USA", "CHN", "JPN", "DEU", "BRA"]
N_SCENARIOS: int = 500
BATCH_SIZE:  int = 1
SEED:        int = 42

# ── DCAL hyperparameters ──────────────────────────────────────────────────────
PT_ALPHA     = 0.88   # unchanged from paper
PT_BETA      = 0.88   # unchanged from paper
KAPPA_BASE   = 2.25   # paper canonical value
KAPPA_LOW    = 1.50   # κ when IS correction is aligned with country prior
KAPPA_HIGH   = 3.50   # κ when IS correction is anti-aligned with country prior
GAMMA_KAPPA  = 0.0    # set runtime from step-dependent formula
TAU_KAPPA    = 80.0   # annealing timescale for κ adaptation

# Sign consistency penalty
LAMBDA_SIGN  = 0.30   # weight of sign-consistency term in U_total
KAPPA_SIGN   = 2.00   # strength of sign-consistency loss

# Country prior (same as EXP-09)
N_WARMUP    = 50
DECAY_TAU   = 100
BETA_EMA    = 0.10
LAMBDA_COOP = 0.70

SWA_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/swa"
CMP_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/compare"
MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/"
    "WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
)
HUMAN_AMCE_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"


# ============================================================================
# Step 3: Country Prior State  (identical to EXP-09 / EXP-10)
# ============================================================================
class CountryPriorState:
    """
    Running EMA country-level prior with annealing.
    Tracks both scalar delta and sign statistics for κ adaptation.
    """

    def __init__(self, beta: float = BETA_EMA, decay_tau: float = DECAY_TAU,
                 n_warmup: int = N_WARMUP):
        self.delta_country = 0.0
        self.beta          = beta
        self.decay_tau     = decay_tau
        self.n_warmup      = n_warmup
        self.step          = 0
        self._history: List[float] = []

    def alpha_h(self) -> float:
        if self.step < self.n_warmup:
            return 0.0
        t = self.step - self.n_warmup
        return 1.0 - np.exp(-t / self.decay_tau)

    def update(self, delta_opt: float) -> None:
        self.delta_country = (1.0 - self.beta) * self.delta_country + self.beta * delta_opt
        self._history.append(delta_opt)
        self.step += 1

    def apply_prior(self, delta_opt_micro: float) -> float:
        a = self.alpha_h()
        return a * self.delta_country + (1.0 - a) * delta_opt_micro

    def kappa_for(self, delta_candidate: float) -> float:
        """
        Return the adaptive κ for a given candidate direction.
        During warmup: return KAPPA_BASE (paper canonical).
        After warmup: KAPPA_LOW if aligned with country prior, else KAPPA_HIGH.
        """
        if self.step < self.n_warmup or abs(self.delta_country) < 1e-6:
            return KAPPA_BASE
        # Annealing weight: adaptation strength grows with step
        adapt_weight = 1.0 - np.exp(-max(0, self.step - self.n_warmup) / TAU_KAPPA)
        if delta_candidate * self.delta_country >= 0:  # same sign → aligned
            kappa = KAPPA_BASE + adapt_weight * (KAPPA_LOW - KAPPA_BASE)  # → KAPPA_LOW
        else:                                             # opposite sign → anti-aligned
            kappa = KAPPA_BASE + adapt_weight * (KAPPA_HIGH - KAPPA_BASE)  # → KAPPA_HIGH
        return float(np.clip(kappa, 0.5, 8.0))

    @property
    def stats(self) -> Dict:
        h = self._history
        return {
            "step": self.step,
            "delta_country": self.delta_country,
            "alpha_h": self.alpha_h(),
            "history_std": float(np.std(h)) if len(h) > 1 else 0.0,
        }


_COUNTRY_PRIOR_STATE: Dict[str, CountryPriorState] = {}


# ============================================================================
# Step 4: DCAL Controller
# ============================================================================
class Exp14AdaptiveKappaController(ImplicitSWAController):
    """
    Direction-Conditioned Adaptive Loss Aversion (DCAL):
    
    Modifies the PT value function used in IS to have a direction-conditioned κ:
      - κ_low  when candidate direction aligns with country prior  → more exploratory
      - κ_high when candidate direction opposes country prior      → more averse (brake)
    
    Additionally adds a soft sign consistency penalty to U_total.
    
    The country prior is updated via EMA (same as EXP-09) but used INSIDE the
    PT value function (before IS weight computation), not just post-hoc.
    """

    def __init__(self, *args, country: str = "UNKNOWN", **kwargs):
        super().__init__(*args, **kwargs)
        self.country = country

    def _get_prior_state(self) -> CountryPriorState:
        if self.country not in _COUNTRY_PRIOR_STATE:
            _COUNTRY_PRIOR_STATE[self.country] = CountryPriorState()
        return _COUNTRY_PRIOR_STATE[self.country]

    def _pt_value_adaptive(
        self,
        x: torch.Tensor,
        delta_country: float,
    ) -> torch.Tensor:
        """
        Vectorised PT value function with direction-conditioned κ.
        x: shape (K,) or (K, N) — normalised gains
        delta_country: scalar country prior (0 during warmup)
        """
        alpha, beta = PT_ALPHA, PT_BETA

        # Compute per-element κ: KAPPA_LOW if sign(x)==sign(delta_country), else KAPPA_HIGH
        # During warmup (delta_country≈0): always KAPPA_BASE
        if abs(delta_country) < 1e-6:
            kappa = KAPPA_BASE
            kappa_t = torch.full_like(x, kappa)
        else:
            # Annealing weight
            prior = _COUNTRY_PRIOR_STATE.get(self.country, CountryPriorState())
            adapt_w = 1.0 - np.exp(-max(0, prior.step - N_WARMUP) / TAU_KAPPA)
            k_low  = KAPPA_BASE + adapt_w * (KAPPA_LOW  - KAPPA_BASE)
            k_high = KAPPA_BASE + adapt_w * (KAPPA_HIGH - KAPPA_BASE)
            k_low  = float(np.clip(k_low,  0.5, 8.0))
            k_high = float(np.clip(k_high, 0.5, 8.0))

            same_sign = (x * delta_country) >= 0   # bool tensor
            kappa_t = torch.where(same_sign,
                                  torch.full_like(x, k_low),
                                  torch.full_like(x, k_high))

        gain = torch.where(x >= 0,
                           x.abs().pow(alpha),
                           -kappa_t * x.abs().pow(beta))
        return gain

    def _sign_consistency_value(
        self,
        delta_tilde: torch.Tensor,
        delta_country: float,
    ) -> torch.Tensor:
        """
        Soft sign consistency: reward candidates aligned with country prior,
        penalise anti-aligned candidates.
        Returns shape (K,) additive term to U_total.
        """
        if abs(delta_country) < 1e-6:
            return torch.zeros(delta_tilde.shape[0], device=delta_tilde.device)
        dc = delta_country
        # progress toward country prior direction: positive = same sign
        sign_progress = delta_tilde * float(np.sign(dc + 1e-9))  # (K,)
        # PT-like: positive progress = reward, negative = loss
        return torch.where(
            sign_progress >= 0,
            sign_progress.abs().pow(PT_ALPHA),
            -KAPPA_SIGN * sign_progress.abs().pow(PT_BETA),
        )

    @torch.no_grad()
    def predict(self, user_query, preferred_on_right=True, phenomenon_category="default", lang="en"):
        # ── Step 1: Two-pass positional debiasing (unchanged) ──
        db1, da1, logit_temp = self._extract_logit_gaps(user_query, phenomenon_category, lang)
        swapped_query, swap_changed = self._swap_positional_labels(user_query, lang)
        if swap_changed:
            db2, da2, _ = self._extract_logit_gaps(swapped_query, phenomenon_category, lang)
        else:
            db2, da2 = db1, da1

        delta_base   = (db1 - db2) / 2.0 if swap_changed else db1
        delta_agents = (da1 - da2) / 2.0 if swap_changed else da1

        # ── Step 2: Proposal setup ──
        sigma = max(
            float(delta_agents.std(unbiased=True).item()) if delta_agents.numel() >= 2 else 0.0,
            self.noise_std
        )
        anchor = delta_agents.mean()
        K, device = self.K, self.device

        eps          = torch.randn(K, device=device) * sigma
        delta_tilde  = anchor + eps               # (K,)

        # ── Step 3: Per-agent gains (σ-normalised) ──
        dist_base_to_i = (delta_base - delta_agents).abs()                     # (N,)
        dist_cand_to_i = (delta_tilde.unsqueeze(1) - delta_agents.unsqueeze(0)).abs()  # (K, N)
        g_per_agent    = (dist_base_to_i.unsqueeze(0) - dist_cand_to_i) / sigma        # (K, N)

        # ── Step 4: DCAL — direction-conditioned PT value ──
        prior_state    = self._get_prior_state()
        delta_country  = prior_state.delta_country  # scalar EMA prior

        v_per_agent = self._pt_value_adaptive(g_per_agent, delta_country)   # (K, N)
        mean_v      = v_per_agent.mean(dim=1)                                 # (K,)

        # Consensus utility (standard PT — not adaptive-κ for consensus term,
        # to keep the cooperative objective stable)
        g_cons = ((delta_base - anchor).abs() - (delta_tilde - anchor).abs()) / sigma
        v_cons = torch.where(g_cons >= 0,
                             g_cons.abs().pow(PT_ALPHA),
                             -torch.full_like(g_cons, KAPPA_BASE) * g_cons.abs().pow(PT_BETA))

        # ── Step 5: Sign consistency bonus (EXP-14 innovation) ──
        v_sign = self._sign_consistency_value(delta_tilde, delta_country)    # (K,)

        # Total utility (augmented)
        U = ((1.0 - self.lambda_coop) * mean_v
             + self.lambda_coop        * v_cons
             + LAMBDA_SIGN             * v_sign)

        w     = F.softmax(U / self.beta, dim=0)
        k_eff = 1.0 / torch.sum(w * w).clamp_min(1e-12)

        delta_star = (torch.sum(w * eps)
                      if float(k_eff.item()) / K >= self.rho_eff
                      else torch.zeros((), device=device))

        delta_opt_micro = float((anchor + delta_star).item())

        # ── Step 6: Apply country prior (EXP-09 approach) ──
        delta_opt_final = prior_state.apply_prior(delta_opt_micro)
        prior_state.update(delta_opt_micro)

        prior_stats = prior_state.stats

        # Compute effective κ used (for diagnostics)
        kappa_eff = prior_state.kappa_for(delta_opt_micro)

        p_right = torch.sigmoid(
            torch.tensor(delta_opt_final / self.decision_temperature)
        ).item()
        p_pref  = p_right if preferred_on_right else 1.0 - p_right
        variance = float(delta_agents.var(unbiased=True).item()) if delta_agents.numel() > 1 else 0.0

        return {
            "p_right": p_right, "p_left": 1.0 - p_right, "p_spare_preferred": p_pref,
            "variance": variance, "sigma_used": sigma,
            "mppi_flipped": (float(anchor.item()) > 0) != (delta_opt_final > 0),
            "delta_z_norm": abs(delta_opt_final - float(anchor.item())),
            "delta_consensus": float(anchor.item()), "delta_opt": delta_opt_final,
            # DCAL diagnostics
            "delta_opt_micro":  delta_opt_micro,
            "delta_country":    prior_stats["delta_country"],
            "alpha_h":          prior_stats["alpha_h"],
            "prior_step":       prior_stats["step"],
            "kappa_eff":        kappa_eff,
            "ess_ratio":        float(k_eff.item()) / K,
            "logit_temp_used":  logit_temp,
            "n_personas":       delta_agents.numel(),
            "agent_decision_gaps": delta_agents.tolist(),
            "agent_rewards":       (delta_agents - delta_base).tolist(),
            "p_spare_preferred_pass1": p_pref,
            "p_spare_preferred_pass2": p_pref,
            "positional_bias": 0.0,
        }


_swa_runner_mod.ImplicitSWAController = Exp14AdaptiveKappaController


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
    print(f"\n{'#'*70}\n# {EXP_ID} [{model_name}] Direction-Conditioned Adaptive κ\n{'#'*70}")

    rows = []
    for ci, country in enumerate(cfg.target_countries):
        if country not in SUPPORTED_COUNTRIES:
            continue

        # Reset country prior
        _COUNTRY_PRIOR_STATE.clear()
        _COUNTRY_PRIOR_STATE[country] = CountryPriorState()
        print(f"\n[{ci+1}/{len(cfg.target_countries)}] {EXP_ID} {model_name} | {country} "
              f"(DCAL: κ_low={KAPPA_LOW}, κ_high={KAPPA_HIGH}, λ_sign={LAMBDA_SIGN})")

        scen     = _load_country_scenarios(cfg, country)
        personas = build_country_personas(country, wvs_path=WVS_DATA_PATH)
        print(f"  [PERSONAS] N={len(personas)}")

        # Inject country at construction
        orig_init = Exp14AdaptiveKappaController.__init__
        def patched_init(self, *a, country=country, **kw):
            orig_init(self, *a, country=country, **kw)
        Exp14AdaptiveKappaController.__init__ = patched_init
        _swa_runner_mod.ImplicitSWAController = Exp14AdaptiveKappaController

        results_df, summary = run_country_experiment(model, tokenizer, country, personas, scen, cfg)
        Exp14AdaptiveKappaController.__init__ = orig_init

        results_df.to_csv(out_dir / f"swa_results_{country}.csv", index=False)
        append_rows_csv(
            str(Path(CMP_ROOT) / "per_dim_breakdown.csv"),
            flatten_per_dim_alignment(
                summary.get("per_dimension_alignment", {}),
                model=model_name,
                method=f"{EXP_ID}_adaptive_kappa",
                country=country,
            ),
        )

        prior_stats    = _COUNTRY_PRIOR_STATE.get(country, CountryPriorState()).stats
        mean_kappa_eff = (float(results_df["kappa_eff"].mean())
                          if "kappa_eff" in results_df.columns else float("nan"))
        mean_ess       = (float(results_df["ess_ratio"].mean())
                          if "ess_ratio" in results_df.columns else float("nan"))

        rows.append({
            "model":   model_name,
            "method":  f"{EXP_ID}_adaptive_kappa",
            "country": country,
            **{f"align_{k}": v for k, v in summary["alignment"].items()},
            "flip_rate":           summary["flip_rate"],
            "mean_latency_ms":     summary["mean_latency_ms"],
            "n_scenarios":         summary["n_scenarios"],
            "mean_kappa_eff":      mean_kappa_eff,
            "mean_ess_ratio":      mean_ess,
            "kappa_low":           KAPPA_LOW,
            "kappa_high":          KAPPA_HIGH,
            "lambda_sign":         LAMBDA_SIGN,
            "final_delta_country": prior_stats["delta_country"],
            "final_alpha_h":       prior_stats["alpha_h"],
            "history_std":         prior_stats["history_std"],
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
            print(f"      κ_eff={mean_kappa_eff:.3f}  ESS={mean_ess:.3f}  "
                  f"δ_country={prior_stats['delta_country']:.4f}  α_h={prior_stats['alpha_h']:.3f}")

        torch.cuda.empty_cache()
        gc.collect()

    return rows


def main():
    setup_seeds(SEED)
    for d in (SWA_ROOT, CMP_ROOT):
        Path(d).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  {EXP_ID}: {EXP_NAME.upper()} — Direction-Conditioned Adaptive Loss Aversion")
    print(f"{'='*70}")
    print(f"[CONFIG] κ_base={KAPPA_BASE}, κ_low={KAPPA_LOW}, κ_high={KAPPA_HIGH}")
    print(f"[CONFIG] τ_kappa={TAU_KAPPA}, λ_sign={LAMBDA_SIGN}, κ_sign={KAPPA_SIGN}")
    print(f"[CONFIG] Country prior: N_warmup={N_WARMUP}, decay_τ={DECAY_TAU}, β_ema={BETA_EMA}")
    print(f"[THEORY] v_adaptive(x) = x^α if x>=0; -κ_dir(x)·|x|^β if x<0")
    print(f"[THEORY] κ_dir = κ_low if aligned with δ_country else κ_high (after warmup)")
    print(f"[TARGET] Mean MIS < 0.3800 | Pearson r > 0 for Mistral | Flip% < 10%")

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

    # ── Final report ──
    print(f"\n\n{'#'*70}")
    print(f"# {EXP_ID} FINAL REPORT — {EXP_NAME.upper()}")
    print(f"{'#'*70}")
    print_alignment_table(cmp_df, title=f"{EXP_ID} RESULTS — {EXP_NAME}")

    print(f"\n{'─'*70}")
    for model_name in MODELS:
        m_df = cmp_df[cmp_df["model"] == model_name]
        if m_df.empty:
            continue
        short     = model_name.split("/")[-1][:20]
        mis_mean  = m_df["align_mis"].mean()
        jsd_mean  = m_df["align_jsd"].mean()
        r_mean    = m_df["align_pearson_r"].mean()
        flip_mean = m_df["flip_rate"].mean()
        kap_mean  = m_df["mean_kappa_eff"].mean()
        print(f"  {short:<20s}  MIS={mis_mean:.4f}  JSD={jsd_mean:.4f}  "
              f"r={r_mean:+.3f}  Flip={flip_mean:.1%}  κ_eff={kap_mean:.3f}")

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
                cur_method=f"{EXP_ID}_adaptive_kappa",
            ),
        )

    # ── Paper-ready table ──
    print(f"\n{'─'*70}")
    print(f"  PAPER-READY TABLE")
    print(f"{'─'*70}")
    print(f"\n| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% | κ_eff | δ_country |")
    print(f"|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|:-----:|:---------:|")
    for _, row in cmp_df.iterrows():
        short = row["model"].split("/")[-1].split("-Instruct")[0].split("-instruct")[0]
        print(f"| {short} | {row['country']} | {row['align_mis']:.4f} | "
              f"{row['align_jsd']:.4f} | {row['align_pearson_r']:+.3f} | "
              f"{row['align_mae']:.2f} | {row['flip_rate']:.1%} | "
              f"{row.get('mean_kappa_eff', float('nan')):.3f} | "
              f"{row.get('final_delta_country', float('nan')):.4f} |")

    from experiment_DM.exp_reporting import print_tracker_ready_report
    print_tracker_ready_report(
        cmp_df, exp_id=EXP_ID,
        per_dim_csv_path=str(Path(CMP_ROOT) / "per_dim_breakdown.csv"),
    )
    print(f"\n[{EXP_ID}] DONE — results under {CMP_ROOT}")


if __name__ == "__main__":
    main()
