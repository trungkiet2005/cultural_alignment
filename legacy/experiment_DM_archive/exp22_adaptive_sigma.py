#!/usr/bin/env python3
"""
EXP-22: Adaptive IS Sigma from Country History (AISH)
======================================================
**Base**: EXP-09 (Hierarchical IS — SOTA MIS = 0.3975)

============================================================
THE FIXED-SIGMA PROBLEM IN EXP-09
============================================================
EXP-09's IS proposal sigma:
    σ = max(std(delta_agents), σ₀ = 0.30)

This is based ONLY on the persona spread in the CURRENT scenario.
The σ₀=0.30 floor is the paper's "minimum exploration guarantee"
(§3.2: "ensures exploration even at consensus").

**Two failure modes from fixed σ**:

1. **EARLY OVER-EXPLORATION** (BRA, Mistral):
   EXP-09's σ is HIGH when personas disagree (as they often do for Mistral).
   With σ=0.40+ and K=128, the IS samples cover [-0.8, +0.8] — a very wide
   range. Most samples are far from the likely optimal correction. This wastes
   the IS budget on implausible proposals → low ESS → delta_star → 0 (ESS guard
   kicks in) → corruption of the country prior with delta_opt_micro ≈ anchor.

2. **LATE UNDER-EXPLORATION** (JPN, Qwen — already good, converged):
   Once the country IS corrections have consistently converged toward a stable
   direction, σ should NARROW to exploit this knowledge. The fixed σ=0.30 floor
   keeps exploring even when the model is already near-optimal.

============================================================
EXP-22 INNOVATION: History-Adaptive Sigma
============================================================
Augment the scenario-level sigma with a HISTORY-BASED component:

    sigma_hist_t = std(delta_opt[max(0, t-W) : t])  (rolling IS history std)

    sigma_eff_t = max(
        std(delta_agents),                          # scenario-level (EXP-09)
        sigma_hist_t * HIST_SIGMA_SCALE + sigma_0,  # history-informed + floor
    )

    Then CLIP: sigma_eff_t = clip(sigma_eff_t, SIGMA_MIN, SIGMA_MAX)

**Interpretation**:
- HIGH sigma_hist (IS corrections are volatile → uncertain country signal):
  large σ_eff → more exploration → likely helps find correct direction
  (analogous to high temperature in simulated annealing)

- LOW sigma_hist (IS corrections converging → confident signal):
  smaller σ_eff (bounded by floor and current persona spread)
  (analogous to cooling in simulated annealing)

**Concrete effect on failure modes**:

- BRA/Mistral (sigma_hist HIGH throughout): sigma_eff stays large → but
  the HISTORY signal tells us the IS is uncertain, so a DIFFERENT clipping
  logic applies: when sigma_hist > SIGMA_HIST_COLLAPSE, FALL BACK to σ₀
  (safety rail: very high history variance means IS is unreliable, don't expand)

- JPN/Qwen after 100 scenarios (sigma_hist LOW ≈ 0.05):
  sigma_eff ≈ max(delta_agents_std, 0.05 * SCALE + 0.30) ≈ delta_agents_std
  → σ is purely driven by current scenario (more precise IS for stable regions)

**Mathematical grounding** (Adaptive Bandwidth IS):
Standard importance sampling theory: the optimal proposal distribution
q*(x) ∝ |f(x)| · p(x) minimises MC variance. For Gaussian proposals:
q*(x) ~ N(μ*, σ*²) where σ* is the RMS of the target function.
sigma_hist approximates this σ* empirically from observed IS corrections.

Hardware note: sigma_eff < sigma_agents preserves higher persona-informed ESS.
sigma_eff > sigma_agents expands exploration when history says we need it.

Hyperparameters (EXP-09 unchanged: N_WARMUP, DECAY_TAU, BETA_EMA):
    WINDOW_SIZE       = 30    (rolling window for sigma_hist)
    HIST_SIGMA_SCALE  = 0.50  (weight of history sigma vs floor)
    SIGMA_MIN         = 0.15  (absolute minimum sigma — below σ₀=0.30 floor)
    SIGMA_MAX         = 0.80  (absolute maximum sigma — prevents explosion)
    SIGMA_HIST_COLLAPSE = 0.50  (if sigma_hist > this: fall back to σ₀, IS is unreliable)

Usage on Kaggle
---------------
    !python experiment_DM/exp22_adaptive_sigma.py
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
EXP_ID   = "EXP-22"
EXP_NAME = "adaptive_sigma"

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

# EXP-22 specific
WINDOW_SIZE          = 30    # rolling window for sigma_hist
HIST_SIGMA_SCALE     = 0.50  # weight of history sigma in sigma_eff
SIGMA_MIN            = 0.15  # absolute minimum (below paper's σ₀=0.30 floor)
SIGMA_MAX            = 0.80  # absolute maximum (prevent explosion)
SIGMA_HIST_COLLAPSE  = 0.50  # safety: if sigma_hist > this, use σ₀ fallback

SWA_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/swa"
CMP_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/compare"
MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH     = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
HUMAN_AMCE_PATH   = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"

SIGMA_0 = 0.30  # paper canonical σ₀ floor


# ============================================================================
# Step 3: Adaptive Sigma Prior State
# ============================================================================
class AdaptiveSigmaPriorState:
    """
    EXP-09 CountryPriorState + rolling sigma_hist tracker.

    Tracks:
        delta_country: EMA of IS corrections (identical to EXP-09)
        sigma_hist: rolling std of delta_opt in the last W scenarios
    
    Provides:
        sigma_eff(sigma_agents): history-augmented IS sigma
    """

    def __init__(self):
        self.delta_country = 0.0
        self.step          = 0
        self._window: deque = deque(maxlen=WINDOW_SIZE)
        self._history: List[float] = []

    def sigma_hist(self) -> float:
        """Rolling std of IS corrections over the last W scenarios."""
        if len(self._window) < 2:
            return 0.0
        return float(np.std(list(self._window)))

    def sigma_eff(self, sigma_agents: float) -> float:
        """
        History-adaptive sigma:
            sh = sigma_hist()
            if sh > SIGMA_HIST_COLLAPSE: return sigma_0 (IS unreliable → use paper floor)
            else: return clip(max(sigma_agents, sh*scale + sigma_0), SIGMA_MIN, SIGMA_MAX)
        """
        sh = self.sigma_hist()
        if sh > SIGMA_HIST_COLLAPSE:
            # IS correction history is too volatile → fall back to paper σ₀
            return float(np.clip(sigma_agents, SIGMA_MIN, SIGMA_MAX))
        return float(np.clip(
            max(sigma_agents, sh * HIST_SIGMA_SCALE + SIGMA_0),
            SIGMA_MIN, SIGMA_MAX
        ))

    def alpha_h(self) -> float:
        if self.step < N_WARMUP: return 0.0
        return 1.0 - np.exp(-(self.step - N_WARMUP) / DECAY_TAU)

    def update(self, delta_opt_micro: float) -> None:
        self.delta_country = (1.0 - BETA_EMA) * self.delta_country + BETA_EMA * delta_opt_micro
        self._window.append(delta_opt_micro)
        self._history.append(delta_opt_micro)
        self.step += 1

    def apply_prior(self, delta_opt_micro: float) -> float:
        a = self.alpha_h()
        return a * self.delta_country + (1.0 - a) * delta_opt_micro

    @property
    def stats(self) -> Dict:
        return {
            "step": self.step,
            "delta_country": self.delta_country,
            "alpha_h": self.alpha_h(),
            "sigma_hist": self.sigma_hist(),
            "history_std": float(np.std(self._history)) if len(self._history) > 1 else 0.0,
        }


_PRIOR_STATE: Dict[str, AdaptiveSigmaPriorState] = {}


# ============================================================================
# Step 4: Adaptive Sigma Controller
# ============================================================================
class Exp22AdaptiveSigmaController(ImplicitSWAController):
    """
    EXP-09 Hierarchical IS with history-adaptive sigma.

    Single change vs EXP-09:
        sigma = max(std(delta_agents), σ₀)          [EXP-09]
        sigma = prior.sigma_eff(std(delta_agents))    [EXP-22]

    Country prior update and prior mixing identical to EXP-09.
    """

    def __init__(self, *args, country: str = "UNKNOWN", **kwargs):
        super().__init__(*args, **kwargs)
        self.country = country

    def _get_prior(self) -> AdaptiveSigmaPriorState:
        if self.country not in _PRIOR_STATE:
            _PRIOR_STATE[self.country] = AdaptiveSigmaPriorState()
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

        # ── EXP-22 CHANGE: history-adaptive sigma ────────────────────────────
        sigma_agents = (float(delta_agents.std(unbiased=True).item())
                        if delta_agents.numel() >= 2 else 0.0)
        prior        = self._get_prior()
        sigma        = prior.sigma_eff(sigma_agents)         # adaptive (not fixed floor)
        # ── End EXP-22 change ─────────────────────────────────────────────────

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
        delta_opt_final = prior.apply_prior(delta_opt_micro)
        prior.update(delta_opt_micro)
        st = prior.stats

        p_right = torch.sigmoid(torch.tensor(delta_opt_final / self.decision_temperature)).item()
        p_pref  = p_right if preferred_on_right else 1.0 - p_right
        variance = float(delta_agents.var(unbiased=True).item()) if delta_agents.numel() > 1 else 0.0

        return {
            "p_right": p_right, "p_left": 1.0 - p_right, "p_spare_preferred": p_pref,
            "variance": variance, "sigma_used": sigma,  # now adaptive!
            "sigma_agents": sigma_agents,               # original persona-based sigma
            "sigma_hist": st["sigma_hist"],             # history-based component
            "mppi_flipped": (float(anchor.item()) > 0) != (delta_opt_final > 0),
            "delta_z_norm": abs(delta_opt_final - float(anchor.item())),
            "delta_consensus": float(anchor.item()), "delta_opt": delta_opt_final,
            "delta_opt_micro": delta_opt_micro,
            "delta_country": st["delta_country"],
            "alpha_h": st["alpha_h"],
            "prior_step": st["step"],
            "ess_ratio": float(k_eff.item()) / K,
            "logit_temp_used": logit_temp,
            "n_personas": delta_agents.numel(),
            "agent_decision_gaps": delta_agents.tolist(),
            "agent_rewards": (delta_agents - delta_base).tolist(),
            "p_spare_preferred_pass1": p_pref,
            "p_spare_preferred_pass2": p_pref,
            "positional_bias": 0.0,
        }


_swa_runner_mod.ImplicitSWAController = Exp22AdaptiveSigmaController


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
    print(f"\n{'#'*70}\n# {EXP_ID} [{model_name}] Adaptive Sigma from History\n{'#'*70}")

    rows = []
    for ci, country in enumerate(cfg.target_countries):
        if country not in SUPPORTED_COUNTRIES: continue
        _PRIOR_STATE.clear()
        _PRIOR_STATE[country] = AdaptiveSigmaPriorState()
        print(f"\n[{ci+1}/{len(cfg.target_countries)}] {EXP_ID} {model_name} | {country} "
              f"(W={WINDOW_SIZE}, hist_scale={HIST_SIGMA_SCALE}, "
              f"σ∈[{SIGMA_MIN},{SIGMA_MAX}], collapse>{SIGMA_HIST_COLLAPSE})")

        scen = _load_scen(cfg, country)
        personas = build_country_personas(country, wvs_path=WVS_DATA_PATH)

        orig_init = Exp22AdaptiveSigmaController.__init__
        def patched_init(self, *a, country=country, **kw): orig_init(self, *a, country=country, **kw)
        Exp22AdaptiveSigmaController.__init__ = patched_init
        _swa_runner_mod.ImplicitSWAController = Exp22AdaptiveSigmaController

        results_df, summary = run_country_experiment(model, tokenizer, country, personas, scen, cfg)
        Exp22AdaptiveSigmaController.__init__ = orig_init

        results_df.to_csv(out_dir / f"swa_results_{country}.csv", index=False)
        append_rows_csv(str(Path(CMP_ROOT) / "per_dim_breakdown.csv"),
                        flatten_per_dim_alignment(summary.get("per_dimension_alignment", {}),
                                                  model=model_name, method=f"{EXP_ID}_adaptive_sigma",
                                                  country=country))
        ps  = _PRIOR_STATE.get(country, AdaptiveSigmaPriorState()).stats
        mea = lambda col: float(results_df[col].mean()) if col in results_df.columns else float("nan")

        rows.append({
            "model": model_name, "method": f"{EXP_ID}_adaptive_sigma", "country": country,
            **{f"align_{k}": v for k, v in summary["alignment"].items()},
            "flip_rate": summary["flip_rate"], "n_scenarios": summary["n_scenarios"],
            "final_delta_country": ps["delta_country"], "final_alpha_h": ps["alpha_h"],
            "final_sigma_hist": ps["sigma_hist"],
            "mean_sigma_used": mea("sigma_used"),
            "mean_sigma_agents": mea("sigma_agents"),
            "mean_sigma_hist": mea("sigma_hist"),
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
            print(f"      σ_eff(avg)={mea('sigma_used'):.4f}  σ_agents={mea('sigma_agents'):.4f}  "
                  f"σ_hist={ps['sigma_hist']:.4f}  ESS={mea('ess_ratio'):.3f}")

        torch.cuda.empty_cache(); gc.collect()
    return rows


def main():
    setup_seeds(SEED)
    for d in (SWA_ROOT, CMP_ROOT): Path(d).mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}\n  {EXP_ID}: {EXP_NAME.upper()}  (base: EXP-09)\n{'='*70}")
    print(f"[THEORY] σ_eff = clip(max(σ_agents, σ_hist·{HIST_SIGMA_SCALE}+{SIGMA_0}), {SIGMA_MIN},{SIGMA_MAX})")
    print(f"[THEORY] σ_hist = std(delta_opt[-{WINDOW_SIZE}:])  (rolling IS correction std)")
    print(f"[THEORY] Collapse safety: σ_hist>{SIGMA_HIST_COLLAPSE} → fall back to σ_agents (IS unreliable)")
    print(f"[TARGET] MIS < 0.3800 | ESS ↑ over EXP-09 | BRA improved")

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
              f"σ_eff={m_df['mean_sigma_used'].mean():.4f}")
    print(f"\n  OVERALL MEAN MIS = {cmp_df['align_mis'].mean():.4f}  (EXP-09 SOTA: 0.3975)")

    ref = try_load_reference_comparison()
    if ref is not None:
        print_metric_comparison(ref, cmp_df, title=f"{EXP_ID} vs EXP-01",
                                spec=CompareSpec(metric_col="align_mis", ref_method="swa_ptis",
                                                 cur_method=f"{EXP_ID}_adaptive_sigma"))
    from experiment_DM.exp_reporting import print_tracker_ready_report
    print_tracker_ready_report(cmp_df, exp_id=EXP_ID,
                               per_dim_csv_path=str(Path(CMP_ROOT) / "per_dim_breakdown.csv"))
    print(f"\n[{EXP_ID}] DONE — {CMP_ROOT}")


if __name__ == "__main__":
    main()
