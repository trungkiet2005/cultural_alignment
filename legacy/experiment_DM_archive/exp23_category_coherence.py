#!/usr/bin/env python3
"""
EXP-23: Category-Coherence Regularization (CCR)
================================================
**Base**: EXP-09 (Hierarchical IS — SOTA MIS = 0.3975)

============================================================
THE INTRA-CATEGORY INCONSISTENCY PROBLEM
============================================================
EXP-09 processes each scenario independently. The country prior
(delta_country, scalar EMA) provides cross-scenario signal, but it
aggregates ALL categories into one scalar — so it cannot prevent
intra-category wild swings.

Specifically: within the same country-category pair (e.g., USA-SocialValue),
sequential predictions can swing wildly because:
  - Each scenario's persona spread is different
  - The IS proposals are independently sampled
  - No mechanism enforces that "if scenario 42 (SocialValue) predicted +0.30,
    scenario 43 (SocialValue) should probably also be near +0.30"

**Empirical evidence from EXP-09 diagnostics**:
  - Flip% = 14-18% (very high): many consecutive predictions cross the boundary
  - Per-category MSE is HIGH even when mean direction is correct
  - This inflates JSD (even correct means can have high distributional error
    if per-scenario variance is high)

============================================================
EXP-23 INNOVATION: Category-Running Mean Coherence
============================================================
Maintain a running mean delta_opt PER CATEGORY:
    mu_cat[dim]  ← (1-α_cat) · mu_cat[dim] + α_cat · delta_opt_micro_t
    n_cat[dim]   stores number of scenarios processed for that category

After computing delta_opt_micro from IS, apply coherence regularization:
    coherence_weight(t) = alpha_h · LAMBDA_COH
    delta_opt_micro_coh = (1 - coherence_weight) · delta_opt_micro
                        + coherence_weight · mu_cat[dim]

This is DIFFERENT from EXP-19 (per-dim prior):
  - EXP-19: uses mu[dim] as a SOURCE OF COUNTRY DIRECTION (replaces delta_country)
  - EXP-23: uses mu[dim] as a COHERENCE CONSTRAINT (blends into the prediction)
  - EXP-19 feeds forward into the PRIOR MIXING step
  - EXP-23 corrects BEFORE the prior mixing step (at the IS output level)

The coherence weight grows with alpha_h: no coherence during warmup (pure IS),
increasing coherence as we accumulate category-specific evidence.

**Why this reduces Flip%**:
If mu_cat["SocialValue"] = +0.20 and the latest IS gives delta_opt_micro = -0.05
(a flip!), the coherence regularization pulls it toward mu_cat → +0.14 (less likely
to be below the decision boundary if previous SV predictions were strongly positive).

**Why this helps JSD**:
The distributional error JSD measures deviation between predicted and human
PROPORTIONS across scenarios. High within-category variance → high JSD even
with correct mean. Coherence regularization reduces within-category variance.

Hyperparameters:
    ALPHA_CAT = 0.08   (slow EMA for category mean, changes less than EXP-09's 0.10)
    LAMBDA_COH = 0.35  (coherence blend weight at full alpha_h)
    N_WARMUP_CAT = 15  (per-category warmup: after 15 scenarios in this dim, activate)

Usage on Kaggle
---------------
    !python experiment_DM/exp23_category_coherence.py
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
EXP_ID   = "EXP-23"
EXP_NAME = "category_coherence"

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

# EXP-23 specific
ALPHA_CAT    = 0.08   # EMA rate for per-category running mean
LAMBDA_COH   = 0.35   # coherence weight at full alpha_h
N_WARMUP_CAT = 15     # per-category warmup (activates coherence after N cat-scenarios)

MULTITP_DIMS = ["Species", "Gender", "Age", "Fitness", "SocialValue", "Utilitarianism", "default"]

SWA_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/swa"
CMP_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/compare"
MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH     = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
HUMAN_AMCE_PATH   = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"


# ============================================================================
# Step 3: Category-Coherence Prior State
# ============================================================================
class CategoryCoherencePriorState:
    """
    EXP-09 CountryPriorState + per-category running mean for coherence.

    Two-level state:
        delta_country:  global EMA (identical to EXP-09, scalar)
        mu_cat[dim]:    per-category EMA (6 scalars for intra-cat coherence)
        n_cat[dim]:     per-category scenario count (for warmup tracking)
    """

    def __init__(self):
        self.delta_country = 0.0
        self.step          = 0
        self.mu_cat:  Dict[str, float] = {d: 0.0 for d in MULTITP_DIMS}
        self.n_cat:   Dict[str, int]   = {d: 0   for d in MULTITP_DIMS}
        self._history: List[float] = []

    def _dim_key(self, category: str) -> str:
        for d in MULTITP_DIMS:
            if d.lower() in category.lower():
                return d
        return "default"

    def alpha_h(self) -> float:
        if self.step < N_WARMUP: return 0.0
        return 1.0 - np.exp(-(self.step - N_WARMUP) / DECAY_TAU)

    def coherence_weight(self, dim: str) -> float:
        """Active only after both global warmup AND per-category warmup."""
        if self.step < N_WARMUP: return 0.0
        if self.n_cat.get(dim, 0) < N_WARMUP_CAT: return 0.0
        return self.alpha_h() * LAMBDA_COH

    def apply_coherence(self, delta_opt_micro: float, category: str) -> float:
        """Blend IS output with category running mean (coherence step)."""
        dim = self._dim_key(category)
        cw  = self.coherence_weight(dim)
        return (1.0 - cw) * delta_opt_micro + cw * self.mu_cat[dim]

    def update(self, delta_opt_micro: float, category: str) -> None:
        """Update global prior AND per-category running mean."""
        dim = self._dim_key(category)
        # Global EMA (EXP-09 identical)
        self.delta_country = (1.0 - BETA_EMA) * self.delta_country + BETA_EMA * delta_opt_micro
        # Per-category EMA
        self.mu_cat[dim]   = (1.0 - ALPHA_CAT) * self.mu_cat[dim] + ALPHA_CAT * delta_opt_micro
        self.n_cat[dim]    = self.n_cat.get(dim, 0) + 1
        self._history.append(delta_opt_micro)
        self.step += 1

    def apply_prior(self, delta_opt_micro: float) -> float:
        """EXP-09 prior mixing (unchanged, uses global delta_country)."""
        a = self.alpha_h()
        return a * self.delta_country + (1.0 - a) * delta_opt_micro

    @property
    def stats(self) -> Dict:
        dims = [d for d in MULTITP_DIMS if d != "default"]
        return {
            "step": self.step,
            "delta_country": self.delta_country,
            "alpha_h": self.alpha_h(),
            "mu_cat": {d: self.mu_cat.get(d, 0.0) for d in dims},
            "n_cat":  {d: self.n_cat.get(d,  0)   for d in dims},
        }


_PRIOR_STATE: Dict[str, CategoryCoherencePriorState] = {}


# ============================================================================
# Step 4: Controller
# ============================================================================
class Exp23CategoryCoherenceController(ImplicitSWAController):
    """
    EXP-09 + Category-Coherence Regularization.

    Pipeline vs EXP-09:
        [EXP-09] IS → delta_opt_micro → apply_prior → delta_opt_final
        [EXP-23] IS → delta_opt_micro → apply_coherence → delta_opt_coh
                     → apply_prior (using global delta_country) → delta_opt_final

    The coherence step blends the IS output toward the per-category running mean
    before the global country prior is applied.
    """

    def __init__(self, *args, country: str = "UNKNOWN", **kwargs):
        super().__init__(*args, **kwargs)
        self.country = country

    def _get_prior(self) -> CategoryCoherencePriorState:
        if self.country not in _PRIOR_STATE:
            _PRIOR_STATE[self.country] = CategoryCoherencePriorState()
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

        # ── EXP-23 CHANGE: apply coherence before country prior ───────────────
        prior = self._get_prior()
        delta_opt_coh   = prior.apply_coherence(delta_opt_micro, phenomenon_category)
        delta_opt_final = prior.apply_prior(delta_opt_coh)          # global EXP-09 prior
        prior.update(delta_opt_coh, phenomenon_category)             # update with coherence-adjusted val
        # ── End EXP-23 change ─────────────────────────────────────────────────

        st = prior.stats
        cw = prior.coherence_weight(prior._dim_key(phenomenon_category))

        p_right = torch.sigmoid(torch.tensor(delta_opt_final / self.decision_temperature)).item()
        p_pref  = p_right if preferred_on_right else 1.0 - p_right
        variance = float(delta_agents.var(unbiased=True).item()) if delta_agents.numel() > 1 else 0.0

        return {
            "p_right": p_right, "p_left": 1.0 - p_right, "p_spare_preferred": p_pref,
            "variance": variance, "sigma_used": sigma,
            "mppi_flipped": (float(anchor.item()) > 0) != (delta_opt_final > 0),
            "delta_z_norm": abs(delta_opt_final - float(anchor.item())),
            "delta_consensus": float(anchor.item()), "delta_opt": delta_opt_final,
            "delta_opt_micro": delta_opt_micro, "delta_opt_coh": delta_opt_coh,
            "delta_country": st["delta_country"], "alpha_h": st["alpha_h"],
            "coherence_weight": cw, "prior_step": st["step"],
            "ess_ratio": float(k_eff.item()) / K,
            "logit_temp_used": logit_temp, "n_personas": delta_agents.numel(),
            "agent_decision_gaps": delta_agents.tolist(),
            "agent_rewards": (delta_agents - delta_base).tolist(),
            "p_spare_preferred_pass1": p_pref, "p_spare_preferred_pass2": p_pref,
            "positional_bias": 0.0,
        }


_swa_runner_mod.ImplicitSWAController = Exp23CategoryCoherenceController


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
    print(f"\n{'#'*70}\n# {EXP_ID} [{model_name}] Category-Coherence Regularization\n{'#'*70}")

    rows = []
    for ci, country in enumerate(cfg.target_countries):
        if country not in SUPPORTED_COUNTRIES: continue
        _PRIOR_STATE.clear()
        _PRIOR_STATE[country] = CategoryCoherencePriorState()
        print(f"\n[{ci+1}/{len(cfg.target_countries)}] {EXP_ID} {model_name} | {country} "
              f"(α_cat={ALPHA_CAT}, λ_coh={LAMBDA_COH}, N_warmup_cat={N_WARMUP_CAT})")

        scen = _load_scen(cfg, country)
        personas = build_country_personas(country, wvs_path=WVS_DATA_PATH)

        orig_init = Exp23CategoryCoherenceController.__init__
        def patched_init(self, *a, country=country, **kw): orig_init(self, *a, country=country, **kw)
        Exp23CategoryCoherenceController.__init__ = patched_init
        _swa_runner_mod.ImplicitSWAController = Exp23CategoryCoherenceController

        results_df, summary = run_country_experiment(model, tokenizer, country, personas, scen, cfg)
        Exp23CategoryCoherenceController.__init__ = orig_init

        results_df.to_csv(out_dir / f"swa_results_{country}.csv", index=False)
        append_rows_csv(str(Path(CMP_ROOT) / "per_dim_breakdown.csv"),
                        flatten_per_dim_alignment(summary.get("per_dimension_alignment", {}),
                                                  model=model_name, method=f"{EXP_ID}_cat_coherence",
                                                  country=country))
        ps  = _PRIOR_STATE.get(country, CategoryCoherencePriorState()).stats
        mea = lambda col: float(results_df[col].mean()) if col in results_df.columns else float("nan")

        rows.append({
            "model": model_name, "method": f"{EXP_ID}_cat_coherence", "country": country,
            **{f"align_{k}": v for k, v in summary["alignment"].items()},
            "flip_rate": summary["flip_rate"], "n_scenarios": summary["n_scenarios"],
            "final_delta_country": ps["delta_country"], "final_alpha_h": ps["alpha_h"],
            "mean_coherence_weight": mea("coherence_weight"), "mean_ess_ratio": mea("ess_ratio"),
            **{f"mu_cat_{d}": ps["mu_cat"].get(d, 0.0) for d in ps["mu_cat"]},
        })

        pda = summary.get("per_dimension_alignment", {})
        if pda:
            print(f"\n  ┌── Per-Dimension ({country}) ──")
            for dk, dd in sorted(pda.items()):
                hv, mv = dd.get("human", float("nan")), dd.get("model", float("nan"))
                print(f"  │  {dk:<25s}  human={hv:6.1f}  model={mv:6.1f}  err={mv-hv:+6.1f}pp")
            print(f"  └── MIS={summary['alignment']['mis']:.4f}  r={summary['alignment']['pearson_r']:+.3f}  "
                  f"Flip={summary['flip_rate']:.1%}")
            print(f"      coh_w(avg)={mea('coherence_weight'):.3f}  "
                  f"mu_cat(SV)={ps['mu_cat'].get('SocialValue', 0.0):+.4f}  "
                  f"n_SV={ps['n_cat'].get('SocialValue', 0)}")

        torch.cuda.empty_cache(); gc.collect()
    return rows

def main():
    setup_seeds(SEED)
    for d in (SWA_ROOT, CMP_ROOT): Path(d).mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}\n  {EXP_ID}: {EXP_NAME.upper()}  (base: EXP-09)\n{'='*70}")
    print(f"[THEORY] delta_opt_coh = (1-cw)·delta_opt_micro + cw·mu_cat[dim]")
    print(f"[THEORY] cw = alpha_h · {LAMBDA_COH}  (active after cat warmup={N_WARMUP_CAT})")
    print(f"[TARGET] MIS < 0.3800 | Flip% < 10% | JSD ↓ (lower within-cat variance)")

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
              f"JSD={m_df['align_jsd'].mean():.4f}  r={m_df['align_pearson_r'].mean():+.3f}  "
              f"Flip={m_df['flip_rate'].mean():.1%}")
    print(f"\n  OVERALL MEAN MIS = {cmp_df['align_mis'].mean():.4f}  (EXP-09 SOTA: 0.3975)")
    ref = try_load_reference_comparison()
    if ref is not None:
        print_metric_comparison(ref, cmp_df, title=f"{EXP_ID} vs EXP-01",
                                spec=CompareSpec(metric_col="align_mis", ref_method="swa_ptis",
                                                 cur_method=f"{EXP_ID}_cat_coherence"))
    from experiment_DM.exp_reporting import print_tracker_ready_report
    print_tracker_ready_report(cmp_df, exp_id=EXP_ID,
                               per_dim_csv_path=str(Path(CMP_ROOT) / "per_dim_breakdown.csv"))
    print(f"\n[{EXP_ID}] DONE — {CMP_ROOT}")

if __name__ == "__main__":
    main()
