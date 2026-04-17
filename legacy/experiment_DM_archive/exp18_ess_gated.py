#!/usr/bin/env python3
"""
EXP-18: ESS-Quality-Gated Prior Update (EGPU)
===============================================
**Base**: EXP-09 (Hierarchical IS — current SOTA, mean MIS = 0.3975)

============================================================
WHY EXP-09 IS THE BASE
============================================================
EXP-09 accumulates a country prior via EMA with fixed β=0.10:
    delta_country ← (1 - β) · delta_country + β · delta_opt_micro   [fixed β]

This means EVERY scenario's IS output contributes equally to the country prior,
regardless of whether that scenario had a RELIABLE IS estimate (high ESS) or
a COLLAPSED IS estimate (very few effective samples).

============================================================
WHAT THIS MISSES
============================================================
The paper (§3.2) introduces the ESS guard:
    k_eff = 1 / Σ w_k²  ∈ [1, K]
    δ* = 0  if k_eff/K < ρ_eff = 0.10   (hard safety rail)

This guard is a BINARY decision: either trust the IS or fall back to zero.
But there is a CONTINUOUS spectrum of IS quality between k_eff/K = 0.10
(barely trustworthy) and k_eff/K = 1.0 (uniformly weighted = maximum
exploration). EXP-09 treats both equally: once ESS > ρ_eff, the IS output
is accepted and triggers a full β=0.10 update of the country prior.

EXP-18 asks: **should a scenario with k_eff/K = 0.12 (barely trustworthy)
update the country prior as aggressively as a scenario with k_eff/K = 0.95?**

The answer is NO — and the fix is elementary:

    β_eff_t = β_base · ρ_t      where ρ_t = clamp(k_eff_t / K, ρ_floor, 1.0)
    delta_country ← (1 - β_eff_t) · delta_country + β_eff_t · delta_opt_micro

When k_eff/K = 0.10: β_eff = 0.10 × 0.10 = 0.010 → barely move the prior
When k_eff/K = 0.95: β_eff = 0.10 × 0.95 = 0.095 → almost normal update

This is the **ESS-Quality-Gated Prior Update (EGPU)**:
  - Low-quality IS scenarios (ESS collapse): contribute weakly to country prior
  - High-quality IS scenarios (uniform weights): contribute fully to country prior
  - Natural self-regularisation: when Mistral's IS collapses repeatedly (known
    failure), the country prior stays stable instead of accumulating bad signal

============================================================
ADDITIONAL INNOVATION: ESS-Adaptive Anchor Regularization
============================================================
EXP-05 showed ESS-adaptive anchor reg (α·anchor + (1-α)·base + δ*) produces
MIS=0.4174 (2nd best after EXP-09). EXP-18 integrates this DIRECTLY into EXP-09
before the hierarchical prior step:

    delta_opt_micro = α_reg · anchor + (1 - α_reg) · delta_base + delta_star
    where α_reg = clamp(k_eff/K, ρ_eff, 1.0)                     [from EXP-05]

    delta_country ← (1 - β_eff) · delta_country + β_eff · delta_opt_micro
    where β_eff = β_base · ρ_t                                    [EXP-18 new]

    delta_opt_final = alpha_h · delta_country + (1 - alpha_h) · delta_opt_micro
                                                                   [EXP-09 unchanged]

This stacks EXP-05 (anchor reg) and EXP-09 (hierarchical prior) with quality-gating,
but unlike EXP-10 (Grand Fusion) which also adds EXP-03 social personas, this keeps
the persona pool standard (4 WVS agents) to isolate the IS-side improvements.

============================================================
MATHEMATICAL GROUNDING
============================================================
ESS-gated EMA can be viewed as a RECURSIVE WEIGHTED LEAST SQUARES:

    delta_country_t = Σ_{i≤t} w_i · ρ_i · delta_opt_micro_i / Σ_{i≤t} w_i · ρ_i

where w_i = β(1-β)^(t-i) is the EMA discount and ρ_i = k_eff_i/K is the
scenario-level reliability weight. This is the IS-quality-weighted empirical Bayes
estimator: high-quality observations have more weight in the country prior.

At convergence (t→∞ with stationary IS quality), delta_country → E_ρ[delta_opt],
the ESS-weighted expectation — a better estimator of the true cultural signal than
the uniform EMA used in EXP-09.

============================================================
EXPECTED IMPROVEMENTS OVER EXP-09
============================================================
- Mean MIS target     : < 0.3800 (quality-gating removes bad IS contributions)
- Mistral: EXP-09 accumulates Mistral's collapsed IS outputs in the country prior,
  which then pulls all subsequent predictions the wrong way. EGPU clamps the β_eff
  during collapse episodes → prior stays stable → less damage.
- Flip% reduction: anchor reg (EXP-05 component) provides a smoother delta_opt_micro
  signal → less variance in the prior → fewer boundary crossings.
- Qwen: high ESS ensures strong updates → faster convergence to true cultural signal.

============================================================
HYPERPARAMETERS
============================================================
EXP-09 base:   N_warmup=50, decay_tau=100, lambda_coop=0.70, K=128, σ₀=0.30
EXP-18 new:    beta_base=0.10 (same as EXP-09!), rho_floor=0.10 (= ρ_eff)
               alpha_reg = clamp(k_eff/K, rho_eff, 1.0)  [EXP-05 formula]

The only difference from EXP-09 in the prior update:
    EXP-09:  β_eff = β_base = 0.10  (constant)
    EXP-18:  β_eff = β_base · max(k_eff/K, rho_floor)  (ESS-gated)

Usage on Kaggle
---------------
    !python experiment_DM/exp18_ess_gated.py
"""

# ============================================================================
# Step 0: env bootstrap  (identical to EXP-09)
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
# Step 2: experiment configuration
# ============================================================================
EXP_ID   = "EXP-18"
EXP_NAME = "ess_gated"

MODELS: List[str] = [
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "unsloth/gemma-2-9b-it-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
]
TARGET_COUNTRIES: List[str] = ["USA", "CHN", "JPN", "DEU", "BRA"]
N_SCENARIOS: int = 500
BATCH_SIZE:  int = 1
SEED:        int = 42

# ── EXP-09 hyperparameters (unchanged) ─────────────────────────────────────
N_WARMUP    = 50
DECAY_TAU   = 100
BETA_BASE   = 0.10     # same as EXP-09 — only gating mode changes
RHO_FLOOR   = 0.10     # = ρ_eff from paper (minimum ESS ratio to trust IS)
LAMBDA_COOP = 0.70

# ── EXP-18 specific ─────────────────────────────────────────────────────────
# ESS-Quality-Gated EMA: β_eff = BETA_BASE * max(k_eff/K, RHO_FLOOR)
# No new hyperparameters needed — reuses paper's ρ_eff as the floor.

SWA_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/swa"
CMP_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/compare"
MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/"
    "WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
)
HUMAN_AMCE_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"


# ============================================================================
# Step 3: ESS-Gated Country Prior State
# ============================================================================
class ESSGatedPriorState:
    """
    EXP-09 CountryPriorState with ESS-quality-gated EMA update.

    EXP-09:  β_eff = β_base = 0.10  (constant regardless of IS quality)
    EXP-18:  β_eff = β_base · max(k_eff/K, rho_floor)
              → reduces update when IS collapses (k_eff/K ≈ rho_floor)
              → full update when IS is high quality (k_eff/K ≈ 1.0)

    Tracks cumulative effective_beta for diagnostics.
    """

    def __init__(self):
        self.delta_country    = 0.0
        self.step             = 0
        self._history:        List[float] = []
        self._beta_eff_history: List[float] = []

    def alpha_h(self) -> float:
        """EXP-09 annealing weight (unchanged)."""
        if self.step < N_WARMUP:
            return 0.0
        t = self.step - N_WARMUP
        return 1.0 - np.exp(-t / DECAY_TAU)

    def update(self, delta_opt_micro: float, ess_ratio: float) -> float:
        """
        ESS-gated EMA update.
        ess_ratio = k_eff / K ∈ [0, 1]
        Returns β_eff used for this update (for diagnostics).
        """
        beta_eff = BETA_BASE * float(np.clip(ess_ratio, RHO_FLOOR, 1.0))
        self.delta_country = ((1.0 - beta_eff) * self.delta_country
                              + beta_eff * delta_opt_micro)
        self._history.append(delta_opt_micro)
        self._beta_eff_history.append(beta_eff)
        self.step += 1
        return beta_eff

    def apply_prior(self, delta_opt_micro: float) -> float:
        """Mix country prior with micro IS result (= EXP-09 logic, unchanged)."""
        a = self.alpha_h()
        return a * self.delta_country + (1.0 - a) * delta_opt_micro

    @property
    def stats(self) -> Dict:
        h   = self._history
        bh  = self._beta_eff_history
        return {
            "step":              self.step,
            "delta_country":     self.delta_country,
            "alpha_h":           self.alpha_h(),
            "history_std":       float(np.std(h))  if len(h)  > 1 else 0.0,
            "mean_beta_eff":     float(np.mean(bh)) if len(bh) > 0 else float("nan"),
            "min_beta_eff":      float(np.min(bh))  if len(bh) > 0 else float("nan"),
        }


_PRIOR_STATE: Dict[str, ESSGatedPriorState] = {}


# ============================================================================
# Step 4: ESS-Gated Controller  (extends EXP-09)
# ============================================================================
class Exp18ESSGatedController(ImplicitSWAController):
    """
    EXP-09 + ESS-Quality-Gated Prior Update + EXP-05 Anchor Regularization.

    Changes vs EXP-09:
    1. delta_opt_micro computed with ESS-adaptive anchor reg (EXP-05):
           delta_opt_micro = α_reg·anchor + (1-α_reg)·delta_base + delta_star
    2. Country prior update uses quality-gated β:
           β_eff = β_base · max(k_eff/K, rho_floor)
           delta_country ← (1 - β_eff)·delta_country + β_eff·delta_opt_micro

    Everything else (IS perturbations, PT value function, alpha_h annealing,
    prior mixing step) is identical to EXP-09.
    """

    def __init__(self, *args, country: str = "UNKNOWN", **kwargs):
        super().__init__(*args, **kwargs)
        self.country = country

    def _get_prior(self) -> ESSGatedPriorState:
        if self.country not in _PRIOR_STATE:
            _PRIOR_STATE[self.country] = ESSGatedPriorState()
        return _PRIOR_STATE[self.country]

    def _pt_value(self, x: torch.Tensor) -> torch.Tensor:
        a, b, k = self.pt_alpha, self.pt_beta, self.pt_kappa
        return torch.where(x >= 0, x.abs().pow(a), -k * x.abs().pow(b))

    @torch.no_grad()
    def predict(self, user_query, preferred_on_right=True, phenomenon_category="default", lang="en"):
        # ── Step 1: Standard EXP-09 IS (two-pass debias + PT-IS) ─────────────
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
            self.noise_std
        )
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

        U     = (1.0 - self.lambda_coop) * mean_v + self.lambda_coop * v_cons
        w     = F.softmax(U / self.beta, dim=0)
        k_eff = 1.0 / torch.sum(w * w).clamp_min(1e-12)

        ess_ratio  = float(k_eff.item()) / K
        delta_star = (torch.sum(w * eps)
                      if ess_ratio >= self.rho_eff
                      else torch.zeros((), device=device))

        # ── Step 2: EXP-05 Anchor Regularization (INTEGRATED INTO EXP-09) ────
        # alpha_reg = clamp(k_eff/K, rho_eff, 1.0)  → from EXP-05
        alpha_reg       = float(np.clip(ess_ratio, self.rho_eff, 1.0))
        anchor_f        = float(anchor.item())
        base_f          = float(delta_base.item())
        star_f          = float(delta_star.item())
        anchor_div      = abs(anchor_f - base_f)

        # Anchor-regularized micro IS (EXP-05):  α·anchor + (1-α)·base + δ*
        delta_opt_micro = alpha_reg * anchor_f + (1.0 - alpha_reg) * base_f + star_f

        # ── Step 3: EXP-18 ESS-Gated Prior Update ─────────────────────────────
        prior    = self._get_prior()
        beta_eff = prior.update(delta_opt_micro, ess_ratio)    # gated update!

        # ── Step 4: EXP-09 Hierarchical Prior Mixing (unchanged) ──────────────
        delta_opt_final = prior.apply_prior(delta_opt_micro)
        stats           = prior.stats

        p_right = torch.sigmoid(
            torch.tensor(delta_opt_final / self.decision_temperature)
        ).item()
        p_pref  = p_right if preferred_on_right else 1.0 - p_right
        variance = float(delta_agents.var(unbiased=True).item()) if delta_agents.numel() > 1 else 0.0

        return {
            "p_right": p_right, "p_left": 1.0 - p_right, "p_spare_preferred": p_pref,
            "variance": variance, "sigma_used": sigma,
            "mppi_flipped": (anchor_f > 0) != (delta_opt_final > 0),
            "delta_z_norm": abs(delta_opt_final - anchor_f),
            "delta_consensus": anchor_f, "delta_opt": delta_opt_final,
            # EXP-18 diagnostics
            "delta_opt_micro": delta_opt_micro,
            "delta_country":   stats["delta_country"],
            "alpha_h":         stats["alpha_h"],
            "prior_step":      stats["step"],
            "ess_ratio":       ess_ratio,
            "beta_eff":        beta_eff,
            "alpha_reg":       alpha_reg,
            "anchor_divergence": anchor_div,
            "logit_temp_used": logit_temp,
            "n_personas":      delta_agents.numel(),
            "agent_decision_gaps": delta_agents.tolist(),
            "agent_rewards":       (delta_agents - delta_base).tolist(),
            "p_spare_preferred_pass1": p_pref,
            "p_spare_preferred_pass2": p_pref,
            "positional_bias": 0.0,
        }


_swa_runner_mod.ImplicitSWAController = Exp18ESSGatedController


# ============================================================================
# Step 5: Runner  (identical structure to EXP-09)
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


def _build_swa_config(model_name):
    return SWAConfig(
        model_name=model_name, n_scenarios=N_SCENARIOS, batch_size=BATCH_SIZE,
        target_countries=list(TARGET_COUNTRIES), load_in_4bit=True, use_real_data=True,
        multitp_data_path=MULTITP_DATA_PATH, wvs_data_path=WVS_DATA_PATH,
        human_amce_path=HUMAN_AMCE_PATH, output_dir=SWA_ROOT,
        lambda_coop=LAMBDA_COOP, K_samples=128,
    )


def _load_country_scenarios(cfg, country):
    lang = COUNTRY_LANG.get(country, "en")
    if cfg.use_real_data:
        df = load_multitp_dataset(
            data_base_path=cfg.multitp_data_path, lang=lang,
            translator=cfg.multitp_translator, suffix=cfg.multitp_suffix,
            n_scenarios=cfg.n_scenarios,
        )
    else:
        df = generate_multitp_scenarios(cfg.n_scenarios, lang=lang)
    df = df.copy(); df["lang"] = lang
    return df


def _run_swa_for_model(model, tokenizer, model_name) -> List[dict]:
    cfg     = _build_swa_config(model_name)
    out_dir = Path(SWA_ROOT) / resolve_output_dir("", model_name).strip("/\\")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(out_dir)
    print(f"\n{'#'*70}\n# {EXP_ID} [{model_name}] ESS-Quality-Gated Prior + Anchor Reg\n{'#'*70}")

    rows = []
    for ci, country in enumerate(cfg.target_countries):
        if country not in SUPPORTED_COUNTRIES: continue

        _PRIOR_STATE.clear()
        _PRIOR_STATE[country] = ESSGatedPriorState()
        print(f"\n[{ci+1}/{len(cfg.target_countries)}] {EXP_ID} {model_name} | {country} "
              f"(β_base={BETA_BASE}, β_eff=β·ρ, anchor_reg=ESS-adaptive)")

        scen     = _load_country_scenarios(cfg, country)
        personas = build_country_personas(country, wvs_path=WVS_DATA_PATH)
        print(f"  [PERSONAS] N={len(personas)} (standard WVS pool, same as EXP-09)")

        orig_init = Exp18ESSGatedController.__init__
        def patched_init(self, *a, country=country, **kw):
            orig_init(self, *a, country=country, **kw)
        Exp18ESSGatedController.__init__ = patched_init
        _swa_runner_mod.ImplicitSWAController = Exp18ESSGatedController

        results_df, summary = run_country_experiment(model, tokenizer, country, personas, scen, cfg)
        Exp18ESSGatedController.__init__ = orig_init

        results_df.to_csv(out_dir / f"swa_results_{country}.csv", index=False)
        append_rows_csv(
            str(Path(CMP_ROOT) / "per_dim_breakdown.csv"),
            flatten_per_dim_alignment(
                summary.get("per_dimension_alignment", {}),
                model=model_name, method=f"{EXP_ID}_ess_gated", country=country,
            ),
        )

        ps           = _PRIOR_STATE.get(country, ESSGatedPriorState()).stats
        mean_ess     = float(results_df["ess_ratio"].mean())  if "ess_ratio" in results_df.columns else float("nan")
        mean_beta    = float(results_df["beta_eff"].mean())   if "beta_eff" in results_df.columns else float("nan")
        mean_alpha_r = float(results_df["alpha_reg"].mean())  if "alpha_reg" in results_df.columns else float("nan")
        mean_anch_d  = float(results_df["anchor_divergence"].mean()) if "anchor_divergence" in results_df.columns else float("nan")

        rows.append({
            "model": model_name, "method": f"{EXP_ID}_ess_gated", "country": country,
            **{f"align_{k}": v for k, v in summary["alignment"].items()},
            "flip_rate": summary["flip_rate"],
            "mean_latency_ms": summary["mean_latency_ms"],
            "n_scenarios": summary["n_scenarios"],
            "final_delta_country": ps["delta_country"],
            "final_alpha_h": ps["alpha_h"],
            "history_std": ps["history_std"],
            "mean_ess_ratio": mean_ess,
            "mean_beta_eff": mean_beta,
            "mean_alpha_reg": mean_alpha_r,
            "mean_anchor_div": mean_anch_d,
        })

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
                  f"Flip={summary['flip_rate']:.1%}")
            print(f"      β_eff={mean_beta:.4f}  ESS={mean_ess:.3f}  "
                  f"α_reg={mean_alpha_r:.3f}  anchor_div={mean_anch_d:.4f}  "
                  f"δ_cty={ps['delta_country']:+.4f}  α_h={ps['alpha_h']:.3f}")

        torch.cuda.empty_cache(); gc.collect()
    return rows


def main():
    setup_seeds(SEED)
    for d in (SWA_ROOT, CMP_ROOT):
        Path(d).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  {EXP_ID}: {EXP_NAME.upper()}")
    print(f"  Base: EXP-09 Hierarchical IS (SOTA MIS = 0.3975)")
    print(f"{'='*70}")
    print(f"[CONFIG] β_base={BETA_BASE}, ρ_floor={RHO_FLOOR}")
    print(f"[KEY]    β_eff = β_base · max(k_eff/K, ρ_floor)  (ESS-quality-gated EMA)")
    print(f"[KEY]    δ_opt_micro = α_reg·anchor + (1-α_reg)·base + δ*  (EXP-05 anchor reg)")
    print(f"[KEY]    δ_opt_final = α_h·δ_country + (1-α_h)·δ_opt_micro  (EXP-09 hier prior)")
    print(f"[CHANGE] vs EXP-09: (1) anchor reg before hier prior; (2) β_eff = β·ρ")
    print(f"[TARGET] MIS < 0.3800 | Flip% < 12% | Mistral r > 0")

    all_rows: List[dict] = []
    for mi, model_name in enumerate(MODELS):
        print(f"\n{'='*70}\n  MODEL {mi+1}/{len(MODELS)}: {model_name}\n{'='*70}")
        model, tokenizer = load_model(model_name, max_seq_length=2048, load_in_4bit=True)
        try:
            all_rows.extend(_run_swa_for_model(model, tokenizer, model_name))
        finally:
            del model, tokenizer; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            _free_model_cache(model_name)
        pd.DataFrame(all_rows).to_csv(Path(CMP_ROOT) / "comparison.csv", index=False)

    cmp_df = pd.DataFrame(all_rows)
    cmp_df.to_csv(Path(CMP_ROOT) / "comparison.csv", index=False)

    print(f"\n{'#'*70}\n# {EXP_ID} FINAL REPORT\n{'#'*70}")
    print_alignment_table(cmp_df, title=f"{EXP_ID} RESULTS — {EXP_NAME}")

    print(f"\n{'─'*70}")
    for model_name in MODELS:
        m_df = cmp_df[cmp_df["model"] == model_name]
        if m_df.empty: continue
        short = model_name.split("/")[-1][:20]
        print(f"  {short:<20s}  MIS={m_df['align_mis'].mean():.4f}  "
              f"JSD={m_df['align_jsd'].mean():.4f}  "
              f"r={m_df['align_pearson_r'].mean():+.3f}  "
              f"Flip={m_df['flip_rate'].mean():.1%}  "
              f"β_eff={m_df['mean_beta_eff'].mean():.4f}")
    print(f"\n  OVERALL MEAN MIS = {cmp_df['align_mis'].mean():.4f}  "
          f"(EXP-09 SOTA: 0.3975)")

    ref = try_load_reference_comparison()
    if ref is not None:
        print_metric_comparison(
            ref, cmp_df, title=f"{EXP_ID} vs EXP-01 — MIS",
            spec=CompareSpec(metric_col="align_mis", ref_method="swa_ptis",
                             cur_method=f"{EXP_ID}_ess_gated"),
        )

    print(f"\n{'─'*70}\n  PAPER-READY TABLE\n{'─'*70}")
    print(f"\n| Model | Country | MIS ↓ | JSD ↓ | r ↑ | MAE ↓ | Flip% | β_eff | ESS |")
    print(f"|:------|:-------:|:-----:|:-----:|:---:|:-----:|:-----:|:-----:|:---:|")
    for _, row in cmp_df.iterrows():
        short = row["model"].split("/")[-1].split("-Instruct")[0].split("-instruct")[0]
        print(f"| {short} | {row['country']} | {row['align_mis']:.4f} | "
              f"{row['align_jsd']:.4f} | {row['align_pearson_r']:+.3f} | "
              f"{row['align_mae']:.2f} | {row['flip_rate']:.1%} | "
              f"{row.get('mean_beta_eff', float('nan')):.4f} | "
              f"{row.get('mean_ess_ratio', float('nan')):.3f} |")

    from experiment_DM.exp_reporting import print_tracker_ready_report
    print_tracker_ready_report(cmp_df, exp_id=EXP_ID,
                               per_dim_csv_path=str(Path(CMP_ROOT) / "per_dim_breakdown.csv"))
    print(f"\n[{EXP_ID}] DONE — {CMP_ROOT}")


if __name__ == "__main__":
    main()
