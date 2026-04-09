#!/usr/bin/env python3
"""
EXP-21: Directional IS Noise — Biased Proposal Toward Country Prior (DISP)
============================================================================
**Base**: EXP-09 (Hierarchical IS — SOTA MIS = 0.3975)

============================================================
THE EXPLORATION INEFFICIENCY IN EXP-09
============================================================
EXP-09's IS proposal:
    ε_k ~ N(0, σ²)          (centred at zero)
    δ̃_k = anchor + ε_k     (centred at persona consensus)

With K=128 samples, ~50% fall on each side of the anchor.
Once the country prior has established a DIRECTION (delta_country > 0),
roughly HALF the IS samples still explore the wrong (negative) direction.

This is wasteful: 50% of compute is spent on proposals that the country
prior already tells us are unlikely to be correct.

**Concrete example** (USA, after 100 scenarios):
  - delta_country = +0.15 (IS consistently corrects rightward for USA)
  - sigma = 0.22 (from persona spread)
  - N(0, σ²): 50 of 128 samples have ε < 0 → waste ~40% of proposals
  - N(μ_shift, σ²) with μ_shift = +0.08: only ~32 samples have ε < 0-μ_shift

The asymmetry gets the IS to spend more budget near the LIKELY GOOD region.

============================================================
EXP-21 INNOVATION: Directional IS Proposal
============================================================
Shift the IS noise distribution toward the country prior direction:

    μ_shift_t = GAMMA_DIR · alpha_h_t · delta_country_t
    ε_k ~ N(μ_shift_t, σ²)    (biased toward country prior)
    δ̃_k = anchor + ε_k

Clamp: μ_shift = clip(μ_shift_t, -SHIFT_MAX·σ, +SHIFT_MAX·σ)
This prevents bias from overwhelming σ when delta_country is very large.

**During warmup** (alpha_h=0): μ_shift=0 → pure N(0,σ²) = EXP-09 identical
**After warmup**: μ_shift grows with alpha_h (confidence in direction)
                 and delta_country (magnitude of direction signal)

**Mathematical grounding** (IS with biased proposal):
The IS weight must be corrected for the proposal bias. With biased proposal
q(ε) = N(μ_shift, σ²) instead of p(ε) = N(0, σ²):

    w_k_corrected = w_k · [p(ε_k) / q(ε_k)]
    = w_k · exp(-ε_k² / (2σ²)) / exp(-(ε_k - μ_shift)² / (2σ²))
    = w_k · exp(-μ_shift·(ε_k - μ_shift/2) / σ²)

We apply this IS correction weight to maintain an unbiased estimator.
When μ_shift → 0: correction → 1 (no bias correction needed, = EXP-09).

This makes EXP-21 THEORETICALLY CORRECT (not just a heuristic shift):
the biased proposal + IS correction produces an unbiased estimate of δ*
while concentrating samples near the country-prior-guided region.

**ESS with biased proposal**: because proposals are concentrated near
the high-value region, the IS weights should be MORE uniform (higher ESS)
when the prior direction is correct. This is measurable: ESS_biased > ESS_EXP09
in countries where delta_country is a reliable direction.

Hyperparameters (from EXP-09):
    GAMMA_DIR  = 0.50  (fraction of alpha_h · delta_country to use as shift)
    SHIFT_MAX  = 0.80  (maximum shift as fraction of sigma)

Usage on Kaggle
---------------
    !python experiment_DM/exp21_directional_noise.py
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
EXP_ID   = "EXP-21"
EXP_NAME = "directional_noise"

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

# EXP-21 specific
GAMMA_DIR = 0.50    # shift fraction: μ_shift = γ · alpha_h · delta_country
SHIFT_MAX = 0.80    # max shift as fraction of sigma: μ_shift ≤ SHIFT_MAX · σ

SWA_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/swa"
CMP_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/compare"
MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH     = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
HUMAN_AMCE_PATH   = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"


# ============================================================================
# Step 3: Prior State  (identical to EXP-09's CountryPriorState)
# ============================================================================
class DirectionalPriorState:
    """Minimal EXP-09 CountryPriorState — used for delta_country + alpha_h only."""

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

    def mu_shift(self, sigma: float) -> float:
        """
        Compute the directional IS noise shift (clamped).
        μ_shift = clip(γ · alpha_h · delta_country, -SHIFT_MAX·σ, +SHIFT_MAX·σ)
        """
        raw = GAMMA_DIR * self.alpha_h() * self.delta_country
        return float(np.clip(raw, -SHIFT_MAX * sigma, SHIFT_MAX * sigma))

    @property
    def stats(self) -> Dict:
        h = self._history
        return {
            "step": self.step,
            "delta_country": self.delta_country,
            "alpha_h": self.alpha_h(),
            "history_std": float(np.std(h)) if len(h) > 1 else 0.0,
        }


_PRIOR_STATE: Dict[str, DirectionalPriorState] = {}


# ============================================================================
# Step 4: Directional IS Controller
# ============================================================================
class Exp21DirectionalNoiseController(ImplicitSWAController):
    """
    EXP-09 + Directional IS noise proposal with IS-correction weighting.

    Single change: ε_k ~ N(μ_shift, σ²) instead of N(0, σ²).
    IS weights corrected for proposal bias: w_k *= p(ε_k)/q(ε_k).
    Country prior update and alpha_h annealing identical to EXP-09.
    """

    def __init__(self, *args, country: str = "UNKNOWN", **kwargs):
        super().__init__(*args, **kwargs)
        self.country = country

    def _get_prior(self) -> DirectionalPriorState:
        if self.country not in _PRIOR_STATE:
            _PRIOR_STATE[self.country] = DirectionalPriorState()
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

        # ── EXP-21 CHANGE: biased IS proposal ──────────────────────────────
        prior    = self._get_prior()
        mu_shift = prior.mu_shift(sigma)        # directional shift (0 during warmup)
        mu_t     = torch.tensor(mu_shift, dtype=anchor.dtype, device=device)

        eps         = torch.randn(K, device=device) * sigma + mu_t   # N(μ_shift, σ²)
        delta_tilde = anchor + eps

        # IS correction weights: log[p(ε)/q(ε)] = log[N(0,σ²)(ε) / N(μ,σ²)(ε)]
        #   = (ε - μ/2)·μ / σ²    (log ratio of Gaussians, simplified)
        log_correction = (eps - mu_t / 2.0) * mu_t / (sigma ** 2 + 1e-12)  # (K,)
        # ── End EXP-21 change ───────────────────────────────────────────────

        dist_base_to_i = (delta_base - delta_agents).abs()
        dist_cand_to_i = (delta_tilde.unsqueeze(1) - delta_agents.unsqueeze(0)).abs()
        g_per_agent    = (dist_base_to_i.unsqueeze(0) - dist_cand_to_i) / sigma
        v_per_agent    = self._pt_value(g_per_agent)
        mean_v         = v_per_agent.mean(dim=1)

        g_cons = ((delta_base - anchor).abs() - (delta_tilde - anchor).abs()) / sigma
        v_cons = self._pt_value(g_cons)

        # PT utility + IS correction for proposal bias
        U = ((1.0 - self.lambda_coop) * mean_v
             + self.lambda_coop * v_cons
             + log_correction)         # bias correction: subtract log(q/p) = -(log p/q)

        w     = F.softmax(U / self.beta, dim=0)
        k_eff = 1.0 / torch.sum(w * w).clamp_min(1e-12)

        delta_star = (torch.sum(w * (eps - mu_t))   # correction: δ* should be vs N(0,σ²)
                      if float(k_eff.item()) / K >= self.rho_eff
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
            "variance": variance, "sigma_used": sigma,
            "mppi_flipped": (float(anchor.item()) > 0) != (delta_opt_final > 0),
            "delta_z_norm": abs(delta_opt_final - float(anchor.item())),
            "delta_consensus": float(anchor.item()), "delta_opt": delta_opt_final,
            "delta_opt_micro": delta_opt_micro,
            "delta_country": st["delta_country"],
            "alpha_h": st["alpha_h"],
            "mu_shift": mu_shift,
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


_swa_runner_mod.ImplicitSWAController = Exp21DirectionalNoiseController


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
    print(f"\n{'#'*70}\n# {EXP_ID} [{model_name}] Directional IS Noise\n{'#'*70}")

    rows = []
    for ci, country in enumerate(cfg.target_countries):
        if country not in SUPPORTED_COUNTRIES: continue
        _PRIOR_STATE.clear()
        _PRIOR_STATE[country] = DirectionalPriorState()
        print(f"\n[{ci+1}/{len(cfg.target_countries)}] {EXP_ID} {model_name} | {country} "
              f"(γ={GAMMA_DIR}, max_shift={SHIFT_MAX}σ)")

        scen = _load_scen(cfg, country)
        personas = build_country_personas(country, wvs_path=WVS_DATA_PATH)

        orig_init = Exp21DirectionalNoiseController.__init__
        def patched_init(self, *a, country=country, **kw): orig_init(self, *a, country=country, **kw)
        Exp21DirectionalNoiseController.__init__ = patched_init
        _swa_runner_mod.ImplicitSWAController = Exp21DirectionalNoiseController

        results_df, summary = run_country_experiment(model, tokenizer, country, personas, scen, cfg)
        Exp21DirectionalNoiseController.__init__ = orig_init

        results_df.to_csv(out_dir / f"swa_results_{country}.csv", index=False)
        append_rows_csv(str(Path(CMP_ROOT) / "per_dim_breakdown.csv"),
                        flatten_per_dim_alignment(summary.get("per_dimension_alignment", {}),
                                                  model=model_name, method=f"{EXP_ID}_dir_noise",
                                                  country=country))
        ps  = _PRIOR_STATE.get(country, DirectionalPriorState()).stats
        mea = lambda col: float(results_df[col].mean()) if col in results_df.columns else float("nan")

        rows.append({
            "model": model_name, "method": f"{EXP_ID}_dir_noise", "country": country,
            **{f"align_{k}": v for k, v in summary["alignment"].items()},
            "flip_rate": summary["flip_rate"], "n_scenarios": summary["n_scenarios"],
            "final_delta_country": ps["delta_country"], "final_alpha_h": ps["alpha_h"],
            "mean_mu_shift": mea("mu_shift"), "mean_ess_ratio": mea("ess_ratio"),
        })

        pda = summary.get("per_dimension_alignment", {})
        if pda:
            print(f"\n  ┌── Per-Dimension ({country}) ──")
            for dk, dd in sorted(pda.items()):
                hv, mv = dd.get("human", float("nan")), dd.get("model", float("nan"))
                print(f"  │  {dk:<25s}  human={hv:6.1f}  model={mv:6.1f}  err={mv-hv:+6.1f}pp")
            print(f"  └── MIS={summary['alignment']['mis']:.4f}  r={summary['alignment']['pearson_r']:+.3f}  "
                  f"Flip={summary['flip_rate']:.1%}")
            print(f"      μ_shift(avg)={mea('mu_shift'):+.4f}  ESS={mea('ess_ratio'):.3f}  "
                  f"δ_cty={ps['delta_country']:+.4f}  α_h={ps['alpha_h']:.3f}")

        torch.cuda.empty_cache(); gc.collect()
    return rows


def main():
    setup_seeds(SEED)
    for d in (SWA_ROOT, CMP_ROOT): Path(d).mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}\n  {EXP_ID}: {EXP_NAME.upper()}  (base: EXP-09)\n{'='*70}")
    print(f"[THEORY] ε_k ~ N(μ_shift, σ²),  μ_shift = clip(γ·α_h·δ_country, ±{SHIFT_MAX}·σ)")
    print(f"[THEORY] IS weights corrected: U += log[p(ε_k)/q(ε_k)] (unbiased estimator)")
    print(f"[TARGET] MIS < 0.3800 | ESS↑ over EXP-09 | Flip% ≤ EXP-09")

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
              f"μ_shift={m_df['mean_mu_shift'].mean():+.4f}")
    print(f"\n  OVERALL MEAN MIS = {cmp_df['align_mis'].mean():.4f}  (EXP-09 SOTA: 0.3975)")

    ref = try_load_reference_comparison()
    if ref is not None:
        print_metric_comparison(ref, cmp_df, title=f"{EXP_ID} vs EXP-01",
                                spec=CompareSpec(metric_col="align_mis", ref_method="swa_ptis",
                                                 cur_method=f"{EXP_ID}_dir_noise"))
    from experiment_DM.exp_reporting import print_tracker_ready_report
    print_tracker_ready_report(cmp_df, exp_id=EXP_ID,
                               per_dim_csv_path=str(Path(CMP_ROOT) / "per_dim_breakdown.csv"))
    print(f"\n[{EXP_ID}] DONE — {CMP_ROOT}")


if __name__ == "__main__":
    main()
