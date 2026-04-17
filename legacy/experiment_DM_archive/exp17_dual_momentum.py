#!/usr/bin/env python3
"""
EXP-17: Dual-Momentum Hierarchical Prior (DMHP)
================================================
**Base**: EXP-09 (Hierarchical IS — current SOTA, mean MIS = 0.3975)

============================================================
WHY EXP-09 IS THE BASE
============================================================
EXP-09 achieved the best overall mean MIS (0.3975) by introducing a
country-level EMA prior that stabilises per-scenario IS drift:

    delta_country ← (1 - β) · delta_country + β · delta_opt_micro   [EXP-09]
    delta_opt = alpha_h · delta_country + (1 - alpha_h) · delta_opt_micro

The single EMA (β=0.10) is a compromise: slow enough to be stable but
potentially too sluggish to capture fast-changing cultural signals within a run.

============================================================
WHAT EXP-09 STILL GETS WRONG (tracker analysis)
============================================================
1. **Flip% = 14–18%** — far higher than EXP-01 (1–2%). The single-scalar
   prior can pull predictions across the decision boundary when early
   scenarios are noisy and the prior is not yet settled.
2. **Mistral Pearson r < 0** in all 5 countries — the single EMA accumulates
   noise from bad early scenarios. By step ~50 (alpha_h activating), if the
   first 50 Mistral corrections were systematically in the wrong direction,
   delta_country points the wrong way for the entire subsequent run.
3. **Qwen DEU regression** in EXP-09 vs EXP-05 — the EMA prior overshoots
   DEU's subtle cultural signal.

============================================================
EXP-17 INNOVATION: Dual-Momentum Prior (Adam-style)
============================================================
Borrow from Adam optimiser's motivation: using a FAST EMA (high β) to capture
recent gradients and a SLOW EMA (low β) to capture the long-run trend solves
the bias-variance tradeoff that a single EMA cannot.

    m_fast_t = (1 - β_fast) · m_fast_{t-1} + β_fast · delta_opt_micro
    m_slow_t = (1 - β_slow) · m_slow_{t-1} + β_slow · delta_opt_micro

    delta_country_t = (1 - λ_t) · m_fast_t + λ_t · m_slow_t

where λ_t is annealed from 0 → λ_max:
    λ_t = λ_max · (1 - exp(-(t - N_warmup) / τ_blend))

**Interpretation**:
- Early (t = N_warmup): λ_t ≈ 0 → delta_country ≈ m_fast → responsive to
  recent corrections, minimal noise accumulation from early bad scenarios.
- Late (large t): λ_t → λ_max → delta_country blends fast+slow → stable
  long-run trend with dampened high-frequency noise.

**Why this fixes the problems**:
- Flip%: fast EMA is more responsive than the single β=0.10 → earlier
  lock-in to the correct direction → fewer boundary crossings.
- Mistral: slow EMA accumulates over many scenarios → even if 50 early
  scenarios are noisy, the slow EMA averages them out; the blend shifts
  to slow only after the fast EMA has stabilised.
- DEU overshoot: slow arm dampens overshoots in slow-convergence countries.

**Connection to Adam optimizer (for paper §3.3)**:
Adam's first moment: m_1 = β_1 · m_1 + (1-β_1) · g  [Eq. 3, Kingma&Ba 2015]
Adam's second moment: not used (we don't normalise by variance)
But the motivation for dual timescales is identical:
  "the combination of fast and slow gradient statistics produces a better
  estimate of the true gradient direction than either alone."

Here "gradient" = IS correction direction (delta_opt_micro), and
"better estimate" = lower MIS.

**Hyperparameters from EXP-09 unchanged**:
    N_WARMUP = 50, DECAY_TAU = 100 (alpha_h annealing)
    lambda_coop = 0.70, K = 128, σ₀ = 0.30 (all paper defaults)

**New hyperparameters**:
    β_fast  = 0.20  (fast EMA, 2× EXP-09's β=0.10)
    β_slow  = 0.03  (slow EMA, 3× slower than EXP-09)
    λ_max   = 0.50  (equal blend at convergence)
    τ_blend = 80    (blend annealing timescale)

**Expected improvements**:
    Mean MIS target  : < 0.3700 (vs EXP-09 SOTA 0.3975)
    Flip%            : < 10% (vs EXP-09's 14–18%)
    Mistral Pearson r: > 0 (at least in 3/5 countries)
    Qwen DEU: improve vs EXP-09 regression

Usage on Kaggle
---------------
    !python experiment_DM/exp17_dual_momentum.py
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
EXP_ID   = "EXP-17"
EXP_NAME = "dual_momentum"

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
LAMBDA_COOP = 0.70

# ── EXP-17: Dual-Momentum hyperparameters ───────────────────────────────────
BETA_FAST    = 0.20    # fast EMA — 2× faster than EXP-09's β=0.10
BETA_SLOW    = 0.03    # slow EMA — ~3× slower than EXP-09
LAMBDA_MAX   = 0.50    # maximum blend toward slow arm (at convergence)
TAU_BLEND    = 80      # annealing timescale for blend weight λ_t

SWA_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/swa"
CMP_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/compare"
MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/"
    "WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
)
HUMAN_AMCE_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"


# ============================================================================
# Step 3: Dual-Momentum Country Prior State
# ============================================================================
class DualMomentumPriorState:
    """
    Dual-EMA country prior (Adam-style: fast arm + slow arm).

    State:
        m_fast  = fast EMA of delta_opt_micro   (β = BETA_FAST)
        m_slow  = slow EMA of delta_opt_micro   (β = BETA_SLOW)
        delta_country = (1 - λ_t) · m_fast + λ_t · m_slow

    Annealing:
        alpha_h  = 1 - exp(-(step - N_warmup) / DECAY_TAU)  [EXP-09 unchanged]
        λ_t      = LAMBDA_MAX · (1 - exp(-(step - N_warmup) / TAU_BLEND))
    """

    def __init__(self):
        self.m_fast        = 0.0
        self.m_slow        = 0.0
        self.step          = 0
        self._history: List[float] = []

    def _lambda_t(self) -> float:
        """Blend weight toward slow arm (0 during warmup, → LAMBDA_MAX)."""
        if self.step < N_WARMUP:
            return 0.0
        t = self.step - N_WARMUP
        return LAMBDA_MAX * (1.0 - np.exp(-t / TAU_BLEND))

    def delta_country(self) -> float:
        """Blended dual-momentum country prior."""
        lam = self._lambda_t()
        return (1.0 - lam) * self.m_fast + lam * self.m_slow

    def alpha_h(self) -> float:
        """EXP-09 annealing weight (unchanged)."""
        if self.step < N_WARMUP:
            return 0.0
        t = self.step - N_WARMUP
        return 1.0 - np.exp(-t / DECAY_TAU)

    def apply_prior(self, delta_opt_micro: float) -> float:
        """Mix dual-momentum prior with micro IS result (= EXP-09 apply_prior logic)."""
        a  = self.alpha_h()
        dc = self.delta_country()
        return a * dc + (1.0 - a) * delta_opt_micro

    def update(self, delta_opt_micro: float) -> None:
        """Update both EMA arms after a scenario."""
        self.m_fast = (1.0 - BETA_FAST) * self.m_fast + BETA_FAST * delta_opt_micro
        self.m_slow = (1.0 - BETA_SLOW) * self.m_slow + BETA_SLOW * delta_opt_micro
        self._history.append(delta_opt_micro)
        self.step += 1

    @property
    def stats(self) -> Dict:
        h = self._history
        return {
            "step":          self.step,
            "m_fast":        self.m_fast,
            "m_slow":        self.m_slow,
            "delta_country": self.delta_country(),
            "lambda_t":      self._lambda_t(),
            "alpha_h":       self.alpha_h(),
            "history_std":   float(np.std(h)) if len(h) > 1 else 0.0,
        }


_PRIOR_STATE: Dict[str, DualMomentumPriorState] = {}


# ============================================================================
# Step 4: Dual-Momentum Controller  (extends EXP-09)
# ============================================================================
class Exp17DualMomentumController(ImplicitSWAController):
    """
    Hierarchical IS with Dual-Momentum Country Prior.

    Identical to EXP-09's Exp09HierarchicalController EXCEPT:
    - CountryPriorState replaced with DualMomentumPriorState (fast + slow EMA)
    - delta_country = (1-λ)·m_fast + λ·m_slow  (adaptive blend)
    - Everything else (IS perturbations, PT value, ESS guard) unchanged.
    """

    def __init__(self, *args, country: str = "UNKNOWN", **kwargs):
        super().__init__(*args, **kwargs)
        self.country = country

    def _get_prior(self) -> DualMomentumPriorState:
        if self.country not in _PRIOR_STATE:
            _PRIOR_STATE[self.country] = DualMomentumPriorState()
        return _PRIOR_STATE[self.country]

    def _pt_value(self, x: torch.Tensor) -> torch.Tensor:
        a, b, k = self.pt_alpha, self.pt_beta, self.pt_kappa
        return torch.where(x >= 0, x.abs().pow(a), -k * x.abs().pow(b))

    @torch.no_grad()
    def predict(self, user_query, preferred_on_right=True, phenomenon_category="default", lang="en"):
        # ── Standard EXP-09 IS pipeline ──────────────────────────────────────
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

        delta_star = (torch.sum(w * eps)
                      if float(k_eff.item()) / K >= self.rho_eff
                      else torch.zeros((), device=device))

        # ── EXP-17 CHANGE: use dual-momentum prior instead of single EMA ─────
        delta_opt_micro = float((anchor + delta_star).item())
        prior           = self._get_prior()
        delta_opt_final = prior.apply_prior(delta_opt_micro)   # blended prior
        prior.update(delta_opt_micro)
        stats = prior.stats

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
            # EXP-17 diagnostics
            "delta_opt_micro": delta_opt_micro,
            "delta_country":   stats["delta_country"],
            "m_fast":          stats["m_fast"],
            "m_slow":          stats["m_slow"],
            "lambda_t":        stats["lambda_t"],
            "alpha_h":         stats["alpha_h"],
            "prior_step":      stats["step"],
            "ess_ratio":       float(k_eff.item()) / K,
            "logit_temp_used": logit_temp,
            "n_personas":      delta_agents.numel(),
            "agent_decision_gaps": delta_agents.tolist(),
            "agent_rewards":       (delta_agents - delta_base).tolist(),
            "p_spare_preferred_pass1": p_pref,
            "p_spare_preferred_pass2": p_pref,
            "positional_bias": 0.0,
        }


_swa_runner_mod.ImplicitSWAController = Exp17DualMomentumController


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
    print(f"\n{'#'*70}\n# {EXP_ID} [{model_name}] Dual-Momentum Hierarchical Prior\n{'#'*70}")

    rows = []
    for ci, country in enumerate(cfg.target_countries):
        if country not in SUPPORTED_COUNTRIES: continue

        _PRIOR_STATE.clear()
        _PRIOR_STATE[country] = DualMomentumPriorState()
        print(f"\n[{ci+1}/{len(cfg.target_countries)}] {EXP_ID} {model_name} | {country} "
              f"(β_fast={BETA_FAST}, β_slow={BETA_SLOW}, λ_max={LAMBDA_MAX}, τ={TAU_BLEND})")

        scen     = _load_country_scenarios(cfg, country)
        personas = build_country_personas(country, wvs_path=WVS_DATA_PATH)
        print(f"  [PERSONAS] N={len(personas)} (standard WVS pool, same as EXP-09)")

        orig_init = Exp17DualMomentumController.__init__
        def patched_init(self, *a, country=country, **kw):
            orig_init(self, *a, country=country, **kw)
        Exp17DualMomentumController.__init__ = patched_init
        _swa_runner_mod.ImplicitSWAController = Exp17DualMomentumController

        results_df, summary = run_country_experiment(model, tokenizer, country, personas, scen, cfg)
        Exp17DualMomentumController.__init__ = orig_init

        results_df.to_csv(out_dir / f"swa_results_{country}.csv", index=False)
        append_rows_csv(
            str(Path(CMP_ROOT) / "per_dim_breakdown.csv"),
            flatten_per_dim_alignment(
                summary.get("per_dimension_alignment", {}),
                model=model_name, method=f"{EXP_ID}_dual_momentum", country=country,
            ),
        )

        ps        = _PRIOR_STATE.get(country, DualMomentumPriorState()).stats
        mean_ess  = float(results_df["ess_ratio"].mean()) if "ess_ratio" in results_df.columns else float("nan")
        mean_lam  = float(results_df["lambda_t"].mean())  if "lambda_t" in results_df.columns else float("nan")

        rows.append({
            "model": model_name, "method": f"{EXP_ID}_dual_momentum", "country": country,
            **{f"align_{k}": v for k, v in summary["alignment"].items()},
            "flip_rate": summary["flip_rate"],
            "mean_latency_ms": summary["mean_latency_ms"],
            "n_scenarios": summary["n_scenarios"],
            "final_delta_country": ps["delta_country"],
            "final_m_fast": ps["m_fast"], "final_m_slow": ps["m_slow"],
            "final_lambda_t": ps["lambda_t"], "final_alpha_h": ps["alpha_h"],
            "history_std": ps["history_std"],
            "mean_ess_ratio": mean_ess, "mean_lambda_t": mean_lam,
            "beta_fast": BETA_FAST, "beta_slow": BETA_SLOW,
            "lambda_max": LAMBDA_MAX, "tau_blend": TAU_BLEND,
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
            print(f"      m_fast={ps['m_fast']:+.4f}  m_slow={ps['m_slow']:+.4f}  "
                  f"δ_cty={ps['delta_country']:+.4f}  λ={ps['lambda_t']:.3f}  "
                  f"α_h={ps['alpha_h']:.3f}")

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
    print(f"[CONFIG] β_fast={BETA_FAST}, β_slow={BETA_SLOW}")
    print(f"[CONFIG] λ_max={LAMBDA_MAX}, τ_blend={TAU_BLEND}")
    print(f"[THEORY] δ_country = (1-λ)·m_fast + λ·m_slow")
    print(f"[THEORY] δ_opt = α_h · δ_country + (1-α_h) · δ_opt_micro  [EXP-09 identical]")
    print(f"[CHANGE] ONLY: CountryPriorState → DualMomentumPriorState (2 EMA arms)")
    print(f"[TARGET] MIS < 0.3700 | Flip% < 10% | Mistral r > 0")

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
              f"Flip={m_df['flip_rate'].mean():.1%}")
    print(f"\n  OVERALL MEAN MIS = {cmp_df['align_mis'].mean():.4f}  "
          f"(EXP-09 SOTA: 0.3975)")

    ref = try_load_reference_comparison()
    if ref is not None:
        print_metric_comparison(
            ref, cmp_df, title=f"{EXP_ID} vs EXP-01 — MIS",
            spec=CompareSpec(metric_col="align_mis", ref_method="swa_ptis",
                             cur_method=f"{EXP_ID}_dual_momentum"),
        )

    print(f"\n{'─'*70}\n  PAPER-READY TABLE\n{'─'*70}")
    print(f"\n| Model | Country | MIS ↓ | JSD ↓ | r ↑ | MAE ↓ | Flip% | λ_t | δ_cty |")
    print(f"|:------|:-------:|:-----:|:-----:|:---:|:-----:|:-----:|:---:|:-----:|")
    for _, row in cmp_df.iterrows():
        short = row["model"].split("/")[-1].split("-Instruct")[0].split("-instruct")[0]
        print(f"| {short} | {row['country']} | {row['align_mis']:.4f} | "
              f"{row['align_jsd']:.4f} | {row['align_pearson_r']:+.3f} | "
              f"{row['align_mae']:.2f} | {row['flip_rate']:.1%} | "
              f"{row.get('final_lambda_t', float('nan')):.3f} | "
              f"{row.get('final_delta_country', float('nan')):+.4f} |")

    from experiment_DM.exp_reporting import print_tracker_ready_report
    print_tracker_ready_report(cmp_df, exp_id=EXP_ID,
                               per_dim_csv_path=str(Path(CMP_ROOT) / "per_dim_breakdown.csv"))
    print(f"\n[{EXP_ID}] DONE — {CMP_ROOT}")


if __name__ == "__main__":
    main()
