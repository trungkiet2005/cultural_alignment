#!/usr/bin/env python3
"""
EXP-20: Variance-Adaptive Alpha Annealing (VAAA)
=================================================
**Base**: EXP-09 (Hierarchical IS — SOTA MIS = 0.3975)

============================================================
THE PROBLEM WITH EXP-09's FIXED ANNEALING SCHEDULE
============================================================
EXP-09 uses a purely TIME-based annealing for alpha_h:

    alpha_h(t) = 1 - exp(-(t - N_warmup) / DECAY_TAU)

This schedule is BLIND to the quality of the running estimate.
It locks in the country prior at a fixed rate regardless of whether
the accumulated delta_opt history is:
  - LOW VARIANCE (IS consistently gives the same direction → confident estimate)
  - HIGH VARIANCE (IS is all over the place → uncertain estimate)

**Concrete failure in EXP-09**:
  - BRA (worst country): IS corrections have high std (~0.3-0.4) throughout run
  - By step 150 (past warmup): alpha_h ≈ 0.78 → 78% weight on an uncertain prior
  - This pulls predictions toward the noisy prior → systematic error amplification
  - Meanwhile, JPN (best): low std → high alpha_h is fine because the prior IS reliable

============================================================
EXP-20 INNOVATION: Variance-Adaptive Alpha
============================================================
Modulate alpha_h by the CONFIDENCE (inverse variance) of the running estimate:

    sigma_hist_t  = std(delta_opt[max(0,t-W) : t])  (rolling window std)
    confidence_t  = exp(-sigma_hist_t / SIGMA_CONF_SCALE)  ∈ (0, 1]

    alpha_h_eff_t = alpha_h_base(t) * confidence_t

When sigma_hist is LOW (e.g. 0.05): confidence ≈ 1.0 → alpha_h_eff ≈ alpha_h_base
When sigma_hist is HIGH (e.g. 0.40): confidence ≈ 0.26 → alpha_h_eff much smaller

The final prior mixing is:
    delta_opt = alpha_h_eff · delta_country + (1 - alpha_h_eff) · delta_opt_micro

**Mathematical grounding** (Bayesian model averaging):
In Bayesian regression: posterior weight ∝ likelihood(data | model) / Z
The likelihood of the country prior model = exp(-MSE / (2σ²)) where σ = sigma_hist
Hence: alpha_h_eff ∝ exp(-sigma_hist² / 2) ≈ exp(-sigma_hist / scale) for small σ_hist

This is approximate empirical Bayes: the prior's influence is proportional to
its precision (1/variance) relative to the micro IS estimates.

**Rolling window** W=30 captures recent variance, not cumulative:
- Allows sigma_hist to DECREASE over time as IS converges → alpha_h_eff recovers
- Prevents permanent confidence suppression from noisy early scenarios
- W=30 ≈ 10% of 310 scenarios, balances recency vs stability

**Key property: reduces to EXP-09** when sigma_hist → 0:
    confidence → 1.0, alpha_h_eff → alpha_h_base  (pure EXP-09 behaviour)

**Target fixes**:
  - BRA: noisy IS → low confidence → weaker prior → less systematic error
  - Mistral: high variance collapses → prior disabled → falls back to stable micro IS
  - JPN/Qwen (already good): low variance → confidence ≈ 1 → EXP-09 unchanged

Hyperparameters:
    SIGMA_CONF_SCALE = 0.20   (scale: at sigma=0.20, confidence=exp(-1)≈0.37)
    WINDOW_SIZE = 30          (rolling variance window in scenarios)

Usage on Kaggle
---------------
    !python experiment_DM/exp20_variance_adaptive_alpha.py
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
from collections import deque
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
# Step 2: configuration
# ============================================================================
EXP_ID   = "EXP-20"
EXP_NAME = "variance_adaptive_alpha"

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

# EXP-20 specific
SIGMA_CONF_SCALE = 0.20   # exp(-sigma_hist / scale): at 0.20 → confidence=0.37
WINDOW_SIZE      = 30     # rolling window for variance estimate

SWA_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/swa"
CMP_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/compare"
MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH     = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
HUMAN_AMCE_PATH   = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"


# ============================================================================
# Step 3: Variance-Adaptive Prior State
# ============================================================================
class VarianceAdaptivePriorState:
    """
    EXP-09 CountryPriorState + variance-modulated alpha_h.

    alpha_h_eff = alpha_h_base(t) * confidence_t
    confidence_t = exp(-rolling_std / SIGMA_CONF_SCALE)
    rolling_std = std(delta_opt[-W:])   (sliding window of W scenarios)
    """

    def __init__(self):
        self.delta_country = 0.0
        self.step          = 0
        self._window: deque = deque(maxlen=WINDOW_SIZE)
        self._all:    List[float] = []

    def _rolling_std(self) -> float:
        if len(self._window) < 2:
            return 0.0
        return float(np.std(list(self._window)))

    def confidence(self) -> float:
        """exp(-rolling_std / scale) ∈ (0, 1]."""
        return float(np.exp(-self._rolling_std() / SIGMA_CONF_SCALE))

    def alpha_h_base(self) -> float:
        """EXP-09 time-based annealing (unchanged)."""
        if self.step < N_WARMUP:
            return 0.0
        t = self.step - N_WARMUP
        return 1.0 - np.exp(-t / DECAY_TAU)

    def alpha_h_eff(self) -> float:
        """Effective alpha = base * confidence."""
        return self.alpha_h_base() * self.confidence()

    def update(self, delta_opt_micro: float) -> None:
        self.delta_country = (1.0 - BETA_EMA) * self.delta_country + BETA_EMA * delta_opt_micro
        self._window.append(delta_opt_micro)
        self._all.append(delta_opt_micro)
        self.step += 1

    def apply_prior(self, delta_opt_micro: float) -> float:
        a = self.alpha_h_eff()
        return a * self.delta_country + (1.0 - a) * delta_opt_micro

    @property
    def stats(self) -> Dict:
        return {
            "step":          self.step,
            "delta_country": self.delta_country,
            "alpha_h_base":  self.alpha_h_base(),
            "alpha_h_eff":   self.alpha_h_eff(),
            "confidence":    self.confidence(),
            "rolling_std":   self._rolling_std(),
            "history_std":   float(np.std(self._all)) if len(self._all) > 1 else 0.0,
        }


_PRIOR_STATE: Dict[str, VarianceAdaptivePriorState] = {}


# ============================================================================
# Step 4: Controller
# ============================================================================
class Exp20VarAlphaController(ImplicitSWAController):
    """EXP-09 with variance-adaptive alpha_h (single component change)."""

    def __init__(self, *args, country: str = "UNKNOWN", **kwargs):
        super().__init__(*args, **kwargs)
        self.country = country

    def _get_prior(self) -> VarianceAdaptivePriorState:
        if self.country not in _PRIOR_STATE:
            _PRIOR_STATE[self.country] = VarianceAdaptivePriorState()
        return _PRIOR_STATE[self.country]

    def _pt_value(self, x: torch.Tensor) -> torch.Tensor:
        a, b, k = self.pt_alpha, self.pt_beta, self.pt_kappa
        return torch.where(x >= 0, x.abs().pow(a), -k * x.abs().pow(b))

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
        prior           = self._get_prior()
        delta_opt_final = prior.apply_prior(delta_opt_micro)   # variance-modulated
        prior.update(delta_opt_micro)
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
            "delta_country": st["delta_country"],
            "alpha_h_base":  st["alpha_h_base"],
            "alpha_h_eff":   st["alpha_h_eff"],
            "confidence":    st["confidence"],
            "rolling_std":   st["rolling_std"],
            "prior_step":    st["step"],
            "ess_ratio":     float(k_eff.item()) / K,
            "logit_temp_used": logit_temp,
            "n_personas": delta_agents.numel(),
            "agent_decision_gaps": delta_agents.tolist(),
            "agent_rewards": (delta_agents - delta_base).tolist(),
            "p_spare_preferred_pass1": p_pref,
            "p_spare_preferred_pass2": p_pref,
            "positional_bias": 0.0,
        }


_swa_runner_mod.ImplicitSWAController = Exp20VarAlphaController


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
    print(f"\n{'#'*70}\n# {EXP_ID} [{model_name}] Variance-Adaptive Alpha\n{'#'*70}")

    rows = []
    for ci, country in enumerate(cfg.target_countries):
        if country not in SUPPORTED_COUNTRIES: continue
        _PRIOR_STATE.clear()
        _PRIOR_STATE[country] = VarianceAdaptivePriorState()
        print(f"\n[{ci+1}/{len(cfg.target_countries)}] {EXP_ID} {model_name} | {country} "
              f"(σ_scale={SIGMA_CONF_SCALE}, W={WINDOW_SIZE})")

        scen = _load_scen(cfg, country)
        personas = build_country_personas(country, wvs_path=WVS_DATA_PATH)

        orig_init = Exp20VarAlphaController.__init__
        def patched_init(self, *a, country=country, **kw): orig_init(self, *a, country=country, **kw)
        Exp20VarAlphaController.__init__ = patched_init
        _swa_runner_mod.ImplicitSWAController = Exp20VarAlphaController

        results_df, summary = run_country_experiment(model, tokenizer, country, personas, scen, cfg)
        Exp20VarAlphaController.__init__ = orig_init

        results_df.to_csv(out_dir / f"swa_results_{country}.csv", index=False)
        append_rows_csv(str(Path(CMP_ROOT) / "per_dim_breakdown.csv"),
                        flatten_per_dim_alignment(summary.get("per_dimension_alignment", {}),
                                                  model=model_name, method=f"{EXP_ID}_var_alpha",
                                                  country=country))
        ps  = _PRIOR_STATE.get(country, VarianceAdaptivePriorState()).stats
        mea = lambda col: float(results_df[col].mean()) if col in results_df.columns else float("nan")

        rows.append({
            "model": model_name, "method": f"{EXP_ID}_var_alpha", "country": country,
            **{f"align_{k}": v for k, v in summary["alignment"].items()},
            "flip_rate": summary["flip_rate"], "n_scenarios": summary["n_scenarios"],
            "final_delta_country": ps["delta_country"],
            "final_alpha_h_base": ps["alpha_h_base"], "final_alpha_h_eff": ps["alpha_h_eff"],
            "final_confidence": ps["confidence"], "final_rolling_std": ps["rolling_std"],
            "mean_ess_ratio": mea("ess_ratio"), "mean_confidence": mea("confidence"),
            "mean_rolling_std": mea("rolling_std"),
        })

        pda = summary.get("per_dimension_alignment", {})
        if pda:
            print(f"\n  ┌── Per-Dimension ({country}) ──")
            for dk, dd in sorted(pda.items()):
                hv, mv = dd.get("human", float("nan")), dd.get("model", float("nan"))
                print(f"  │  {dk:<25s}  human={hv:6.1f}  model={mv:6.1f}  err={mv-hv:+6.1f}pp")
            print(f"  └── MIS={summary['alignment']['mis']:.4f}  r={summary['alignment']['pearson_r']:+.3f}  "
                  f"Flip={summary['flip_rate']:.1%}")
            print(f"      α_base={ps['alpha_h_base']:.3f}  α_eff={ps['alpha_h_eff']:.3f}  "
                  f"conf={ps['confidence']:.3f}  roll_σ={ps['rolling_std']:.4f}")

        torch.cuda.empty_cache(); gc.collect()
    return rows


def main():
    setup_seeds(SEED)
    for d in (SWA_ROOT, CMP_ROOT): Path(d).mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}\n  {EXP_ID}: {EXP_NAME.upper()}  (base: EXP-09)\n{'='*70}")
    print(f"[THEORY] alpha_h_eff = alpha_h_base(t) * exp(-rolling_std / {SIGMA_CONF_SCALE})")
    print(f"[THEORY] rolling_std = std(delta_opt[-{WINDOW_SIZE}:])")
    print(f"[TARGET] MIS < 0.3800 | BRA improved | Flip% < 12%")

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
              f"conf={m_df['mean_confidence'].mean():.3f}")
    print(f"\n  OVERALL MEAN MIS = {cmp_df['align_mis'].mean():.4f}  (EXP-09 SOTA: 0.3975)")

    ref = try_load_reference_comparison()
    if ref is not None:
        print_metric_comparison(ref, cmp_df, title=f"{EXP_ID} vs EXP-01",
                                spec=CompareSpec(metric_col="align_mis", ref_method="swa_ptis",
                                                 cur_method=f"{EXP_ID}_var_alpha"))
    from experiment_DM.exp_reporting import print_tracker_ready_report
    print_tracker_ready_report(cmp_df, exp_id=EXP_ID,
                               per_dim_csv_path=str(Path(CMP_ROOT) / "per_dim_breakdown.csv"))
    print(f"\n[{EXP_ID}] DONE — {CMP_ROOT}")


if __name__ == "__main__":
    main()
