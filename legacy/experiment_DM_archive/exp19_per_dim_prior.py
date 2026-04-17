#!/usr/bin/env python3
"""
EXP-19: Per-Dimension Hierarchical Prior (PDHP)
================================================
**Base**: EXP-09 (Hierarchical IS — current SOTA, mean MIS = 0.3975)

============================================================
WHY EXP-09 IS THE BASE
============================================================
EXP-09 maintains ONE scalar country prior delta_country:

    delta_country ← (1-β)·delta_country + β·delta_opt_micro   [ALL scenarios]
    delta_opt = alpha_h·delta_country + (1-alpha_h)·delta_opt_micro

Every scenario — regardless of whether it tests Species, SocialValue,
Age, Fitness, Gender, or Utilitarianism — updates the SAME prior.

============================================================
THE CORE PROBLEM: DIMENSION-SPECIFIC FAILURE MODES
============================================================
From tracker per-dimension analysis (EXP-09 results):

| Model   | Country | Worst dim         | |err| (pp) |
|---------|---------|-------------------|-----------|
| Qwen    | USA     | SocialValue_High  | 28.7      |
| Qwen    | DEU     | Species_Humans    | 28.4      |
| Gemma   | USA     | Age_Young         | 27.3      |
| Gemma   | JPN     | Utilitarianism    | 27.2      |
| Mistral | USA     | Utilitarianism    | 30.8      |
| Mistral | DEU     | Species_Humans    | 30.7      |

These dimension-specific errors persist in EXP-09 because the SINGLE
scalar prior cannot differentiate:
  - A strong positive IS correction on SocialValue_High (good → update up)
  - A strong negative IS correction on Species_Humans (bad → don't update)

When the SV prior builds up correctly and Species doesn't, mixing them in
a single scalar AVERAGES away the structure. The Species errors pollute
the SocialValue prior and vice versa.

============================================================
EXP-19 INNOVATION: Per-Dimension Hierarchical Prior (PDHP)
============================================================
Maintain SIX separate country priors, one per MultiTP moral dimension:

    delta_country[dim] ← (1-β)·delta_country[dim] + β·delta_opt_micro
    where "dim" = category of the CURRENT scenario

For a Species scenario:    only delta_country["Species"]    gets updated
For a SocialValue scenario: only delta_country["SocialValue"] gets updated

Final prediction:
    delta_opt = alpha_h[dim] · delta_country[dim] + (1-alpha_h[dim]) · delta_opt_micro
    where alpha_h[dim] = 1 - exp(-(step[dim] - N_warmup) / DECAY_TAU)

Each dimension has its own step counter — warming up at its OWN rate
based on how many scenarios of that type have been processed.

This is the key novelty: each dimension independently learns its own
country-level correction signal, without cross-contamination.

============================================================
MATHEMATICAL GROUNDING
============================================================
EXP-09 prior is a scalar empirical Bayes estimator:
    delta_country ≈ E_{P(x|c)}[delta_opt(x)]    (averaged over all scenarios)

PDHP prior is a CONDITIONAL empirical Bayes estimator:
    delta_country[d] ≈ E_{P(x|c, dim(x)=d)}[delta_opt(x)]

The conditional estimator naturally captures:
    - Systematic OVER-correction in dim Species (Mistral DEU: 30.7pp)
    - Systematic UNDER-correction in dim SocialValue (Qwen USA: 28.7pp)

Because delta_country[Species] < 0 (IS over-corrects toward humans) AND
delta_country[SocialValue] > 0 (IS under-corrects, needs bigger push),
the per-dimension prior can apply opposite corrections simultaneously —
something impossible with a single scalar.

The six priors are independent, so there is NO cross-dimension contamination.

Warmup is per-dimension: with 310 scenarios and 6 dimensions, each dim
gets ~50 scenarios → reaches N_warmup=30 (dimension-specific, smaller
than EXP-09's N_warmup=50 since per-dim data is sparser).

============================================================
DIM-STEP ANNEALING
============================================================
    n_step[dim] = number of scenarios of type dim processed
    alpha_h[dim] = 0                                if n_step[dim] < N_WARMUP_DIM
                = 1 - exp(-(n_step[dim]-N_WARMUP_DIM) / DECAY_TAU)  otherwise

This ensures that:
  - Rare dimensions (few scenarios): alpha_h stays low → mostly micro IS
  - Common dimensions: alpha_h grows naturally → prior kicks in
  - Same behaviour as EXP-09 but per-dimension

============================================================
EXPECTED IMPROVEMENTS OVER EXP-09
============================================================
- SocialValue: dedicated SV prior builds up correctly, not diluted by
  Species/Age corrections → SV error < 20pp
- Mistral Species: Species prior captures the consistent over-correction
  direction → MIS improves for DEU where Species_Humans is the worst dim
- Pearson r: dimension-specific corrections better align the RANKING
  of dimensions → higher r across all models
- Flip%: dimension-specific prior is MORE targeted → less variance in
  predictions across the decision boundary
- Mean MIS target: < 0.3700 (most ambitious after EXP-16)

============================================================
HYPERPARAMETERS (vs EXP-09)
============================================================
EXP-09:  N_WARMUP=50, DECAY_TAU=100, BETA_EMA=0.10 (all scenarios)
EXP-19:  N_WARMUP_DIM=25, DECAY_TAU=100, BETA_EMA=0.10 (per-dimension)

N_WARMUP_DIM=25 (half of EXP-09's 50): with ~50-80 scenarios per dim
in a 310-scenario run, dim-specific warmup of 25 gives ~30-55 "active"
annealing steps per dimension — similar effective learning horizon.

Usage on Kaggle
---------------
    !python experiment_DM/exp19_per_dim_prior.py
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
EXP_ID   = "EXP-19"
EXP_NAME = "per_dim_prior"

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

# ── EXP-09 hyperparameters (unchanged) ─────────────────────────────────────
DECAY_TAU = 100
BETA_EMA  = 0.10

# ── EXP-19 specific ─────────────────────────────────────────────────────────
N_WARMUP_DIM = 25    # per-dimension warmup (< EXP-09's 50, adapted for sparser signals)

# MultiTP dimension labels (matched against phenomenon_category field)
MULTITP_DIMS = [
    "Species", "Gender", "Age", "Fitness", "SocialValue", "Utilitarianism",
]
_DIM_FALLBACK = "default"  # catch-all for unlabelled scenarios

SWA_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/swa"
CMP_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/compare"
MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/"
    "WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
)
HUMAN_AMCE_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"


# ============================================================================
# Step 3: Per-Dimension Country Prior State
# ============================================================================
class PerDimPriorState:
    """
    Six independent EMA country priors — one per MultiTP moral dimension.

    Each dimension maintains its own:
        delta_country[dim]  : EMA of IS corrections for dim-specific scenarios
        step[dim]           : number of dim scenarios processed (for warmup)

    update(delta_opt_micro, dim) → updates only delta_country[dim]
    apply_prior(delta_opt_micro, dim) → mixes dim-specific prior

    During dim warmup (step[dim] < N_WARMUP_DIM): alpha_h[dim]=0 → pure micro IS
    After warmup: alpha_h[dim] grows toward 1.0 via EXP-09-style annealing.
    """

    def __init__(self):
        self.delta_country: Dict[str, float] = {d: 0.0 for d in MULTITP_DIMS + [_DIM_FALLBACK]}
        self.n_step:        Dict[str, int]   = {d: 0   for d in MULTITP_DIMS + [_DIM_FALLBACK]}
        self._history:      Dict[str, List[float]] = {d: [] for d in MULTITP_DIMS + [_DIM_FALLBACK]}

    def _dim_key(self, category: str) -> str:
        """Map phenomenon_category to a tracked dimension key."""
        for d in MULTITP_DIMS:
            if d.lower() in category.lower():
                return d
        return _DIM_FALLBACK

    def alpha_h(self, dim: str) -> float:
        """Per-dimension annealing weight (identical formula to EXP-09 but per-dim)."""
        n = self.n_step.get(dim, 0)
        if n < N_WARMUP_DIM:
            return 0.0
        t = n - N_WARMUP_DIM
        return 1.0 - np.exp(-t / DECAY_TAU)

    def update(self, delta_opt_micro: float, category: str) -> str:
        """Update ONLY the prior for the given category's dimension."""
        dim = self._dim_key(category)
        self.delta_country[dim] = (
            (1.0 - BETA_EMA) * self.delta_country[dim] + BETA_EMA * delta_opt_micro
        )
        self._history[dim].append(delta_opt_micro)
        self.n_step[dim] = self.n_step.get(dim, 0) + 1
        return dim

    def apply_prior(self, delta_opt_micro: float, category: str) -> float:
        """Mix dim-specific prior with micro IS result."""
        dim = self._dim_key(category)
        a   = self.alpha_h(dim)
        dc  = self.delta_country[dim]
        return a * dc + (1.0 - a) * delta_opt_micro

    @property
    def stats(self) -> Dict:
        """Summary statistics for all dimensions."""
        return {
            "steps":    {d: self.n_step.get(d, 0)          for d in MULTITP_DIMS},
            "deltas":   {d: self.delta_country.get(d, 0.0) for d in MULTITP_DIMS},
            "alpha_hs": {d: self.alpha_h(d)                for d in MULTITP_DIMS},
            "stds":     {d: (float(np.std(self._history[d])) if len(self._history[d]) > 1 else 0.0)
                         for d in MULTITP_DIMS},
        }

    def dominant_dim_str(self) -> str:
        """Return the dimension with the largest |delta_country| for diagnostics."""
        if not any(self.n_step.values()):
            return "none"
        best = max(MULTITP_DIMS, key=lambda d: abs(self.delta_country.get(d, 0.0)))
        return f"{best}({self.delta_country[best]:+.4f})"


_PRIOR_STATE: Dict[str, PerDimPriorState] = {}


# ============================================================================
# Step 4: Per-Dimension Prior Controller  (extends EXP-09)
# ============================================================================
class Exp19PerDimController(ImplicitSWAController):
    """
    EXP-09 Hierarchical IS with per-dimension country priors.

    The ONLY change vs EXP-09:
        CountryPriorState (single scalar EMA) → PerDimPriorState (6 scalar EMAs)

    Each call to predict() passes phenomenon_category → selects the correct dim,
    updates only that dimension's prior, and applies only that dimension's prior
    in the final mix.

    Everything else — IS perturbations, PT value function, positional debiasing,
    ESS guard, lambda_coop — is IDENTICAL to EXP-09.
    """

    def __init__(self, *args, country: str = "UNKNOWN", **kwargs):
        super().__init__(*args, **kwargs)
        self.country = country

    def _get_prior(self) -> PerDimPriorState:
        if self.country not in _PRIOR_STATE:
            _PRIOR_STATE[self.country] = PerDimPriorState()
        return _PRIOR_STATE[self.country]

    def _pt_value(self, x: torch.Tensor) -> torch.Tensor:
        a, b, k = self.pt_alpha, self.pt_beta, self.pt_kappa
        return torch.where(x >= 0, x.abs().pow(a), -k * x.abs().pow(b))

    @torch.no_grad()
    def predict(self, user_query, preferred_on_right=True, phenomenon_category="default", lang="en"):
        # ── Steps 1-4: Identical to EXP-09 ──────────────────────────────────
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

        # Standard micro IS result (same as EXP-09)
        delta_opt_micro = float((anchor + delta_star).item())

        # ── EXP-19 CHANGE: per-dimension prior instead of single scalar ───────
        prior = self._get_prior()
        delta_opt_final = prior.apply_prior(delta_opt_micro, phenomenon_category)
        dim_used        = prior.update(delta_opt_micro, phenomenon_category)

        # Diagnostics
        ps           = prior.stats
        alpha_h_dim  = ps["alpha_hs"][dim_used] if dim_used in ps["alpha_hs"] else 0.0
        delta_cty_dim = ps["deltas"][dim_used]   if dim_used in ps["deltas"]   else 0.0

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
            # EXP-19 diagnostics
            "delta_opt_micro":  delta_opt_micro,
            "dim_used":         dim_used,
            "delta_country_dim": delta_cty_dim,
            "alpha_h_dim":      alpha_h_dim,
            "prior_step_dim":   prior.n_step.get(dim_used, 0),
            "ess_ratio":        float(k_eff.item()) / K,
            "logit_temp_used":  logit_temp,
            "n_personas":       delta_agents.numel(),
            "agent_decision_gaps": delta_agents.tolist(),
            "agent_rewards":       (delta_agents - delta_base).tolist(),
            "p_spare_preferred_pass1": p_pref,
            "p_spare_preferred_pass2": p_pref,
            "positional_bias": 0.0,
        }


_swa_runner_mod.ImplicitSWAController = Exp19PerDimController


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
    print(f"\n{'#'*70}\n# {EXP_ID} [{model_name}] Per-Dimension Hierarchical Prior\n{'#'*70}")

    rows = []
    for ci, country in enumerate(cfg.target_countries):
        if country not in SUPPORTED_COUNTRIES: continue

        # Reset per-dimension state for each (model, country)
        _PRIOR_STATE.clear()
        _PRIOR_STATE[country] = PerDimPriorState()
        print(f"\n[{ci+1}/{len(cfg.target_countries)}] {EXP_ID} {model_name} | {country} "
              f"(6 independent dim priors, N_warmup_dim={N_WARMUP_DIM})")

        scen     = _load_country_scenarios(cfg, country)
        personas = build_country_personas(country, wvs_path=WVS_DATA_PATH)
        print(f"  [PERSONAS] N={len(personas)} (standard WVS pool, same as EXP-09)")

        orig_init = Exp19PerDimController.__init__
        def patched_init(self, *a, country=country, **kw):
            orig_init(self, *a, country=country, **kw)
        Exp19PerDimController.__init__ = patched_init
        _swa_runner_mod.ImplicitSWAController = Exp19PerDimController

        results_df, summary = run_country_experiment(model, tokenizer, country, personas, scen, cfg)
        Exp19PerDimController.__init__ = orig_init

        results_df.to_csv(out_dir / f"swa_results_{country}.csv", index=False)
        append_rows_csv(
            str(Path(CMP_ROOT) / "per_dim_breakdown.csv"),
            flatten_per_dim_alignment(
                summary.get("per_dimension_alignment", {}),
                model=model_name, method=f"{EXP_ID}_per_dim_prior", country=country,
            ),
        )

        ps       = _PRIOR_STATE.get(country, PerDimPriorState()).stats
        dominant = _PRIOR_STATE.get(country, PerDimPriorState()).dominant_dim_str()
        mean_ess = float(results_df["ess_ratio"].mean()) if "ess_ratio" in results_df.columns else float("nan")

        rows.append({
            "model": model_name, "method": f"{EXP_ID}_per_dim_prior", "country": country,
            **{f"align_{k}": v for k, v in summary["alignment"].items()},
            "flip_rate": summary["flip_rate"],
            "mean_latency_ms": summary["mean_latency_ms"],
            "n_scenarios": summary["n_scenarios"],
            "mean_ess_ratio": mean_ess,
            "dominant_dim": dominant,
            # Per-dimension final priors (for paper table)
            **{f"delta_cty_{d}": ps["deltas"].get(d, 0.0)   for d in MULTITP_DIMS},
            **{f"alpha_h_{d}":   ps["alpha_hs"].get(d, 0.0) for d in MULTITP_DIMS},
            **{f"n_step_{d}":    ps["steps"].get(d, 0)       for d in MULTITP_DIMS},
        })

        # ── Per-dimension prior diagnostics ──
        print(f"\n  ┌── Per-Dimension Alignment ({country}) ──")
        pda = summary.get("per_dimension_alignment", {})
        for dim_key, dim_data in sorted(pda.items()):
            hv  = dim_data.get("human", float("nan"))
            mv  = dim_data.get("model", float("nan"))
            err = dim_data.get("error", mv - hv)
            print(f"  │  {dim_key:<25s}  human={hv:6.1f}  model={mv:6.1f}  err={err:+6.1f}pp")
        print(f"  └── MIS={summary['alignment']['mis']:.4f}  "
              f"JSD={summary['alignment']['jsd']:.4f}  "
              f"r={summary['alignment']['pearson_r']:+.3f}  "
              f"Flip={summary['flip_rate']:.1%}")

        print(f"\n  ┌── Per-Dimension Prior State ({country}) ──")
        for d in MULTITP_DIMS:
            n   = ps["steps"].get(d, 0)
            dc  = ps["deltas"].get(d, 0.0)
            ah  = ps["alpha_hs"].get(d, 0.0)
            std = ps["stds"].get(d, 0.0)
            print(f"  │  {d:<14s}  step={n:3d}  δ_cty={dc:+.4f}  α_h={ah:.3f}  σ={std:.4f}")
        print(f"  └── dominant={dominant}  ESS={mean_ess:.3f}")

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
    print(f"[CONFIG] N_warmup_dim={N_WARMUP_DIM}, DECAY_TAU={DECAY_TAU}, BETA_EMA={BETA_EMA}")
    print(f"[DIMS]   {MULTITP_DIMS}")
    print(f"[THEORY] δ_country[dim] ← (1-β)·δ_country[dim] + β·δ_opt_micro  (dim-specific)")
    print(f"[THEORY] α_h[dim] = 1 - exp(-(n_step[dim] - N_warmup_dim) / τ)  (per-dim warmup)")
    print(f"[THEORY] δ_opt = α_h[dim]·δ_country[dim] + (1-α_h[dim])·δ_opt_micro")
    print(f"[CHANGE] ONLY: single scalar EMA → 6 independent dim-specific EMAs")
    print(f"[TARGET] MIS < 0.3700 | SocialValue err < 20pp | Mistral r > 0")

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
                             cur_method=f"{EXP_ID}_per_dim_prior"),
        )
        print_metric_comparison(
            ref, cmp_df, title=f"{EXP_ID} vs EXP-01 — JSD",
            spec=CompareSpec(metric_col="align_jsd", ref_method="swa_ptis",
                             cur_method=f"{EXP_ID}_per_dim_prior"),
        )

    print(f"\n{'─'*70}\n  PAPER-READY TABLE\n{'─'*70}")
    print(f"\n| Model | Country | MIS ↓ | JSD ↓ | r ↑ | MAE ↓ | Flip% | dominant_dim |")
    print(f"|:------|:-------:|:-----:|:-----:|:---:|:-----:|:-----:|:------------:|")
    for _, row in cmp_df.iterrows():
        short = row["model"].split("/")[-1].split("-Instruct")[0].split("-instruct")[0]
        print(f"| {short} | {row['country']} | {row['align_mis']:.4f} | "
              f"{row['align_jsd']:.4f} | {row['align_pearson_r']:+.3f} | "
              f"{row['align_mae']:.2f} | {row['flip_rate']:.1%} | "
              f"{row.get('dominant_dim', 'N/A')} |")

    # ── Per-dimension final prior table (novel diagnostic) ──
    print(f"\n{'─'*70}\n  PER-DIMENSION COUNTRY PRIORS (final state)\n{'─'*70}")
    print(f"| Model | Country | Species | Gender | Age | Fitness | SocialValue | Util |")
    print(f"|:------|:-------:|:-------:|:------:|:---:|:-------:|:-----------:|:----:|")
    for _, row in cmp_df.iterrows():
        short = row["model"].split("/")[-1].split("-Instruct")[0].split("-instruct")[0]
        vals  = [f"{row.get(f'delta_cty_{d}', 0.0):+.3f}" for d in MULTITP_DIMS]
        print(f"| {short} | {row['country']} | {' | '.join(vals)} |")

    from experiment_DM.exp_reporting import print_tracker_ready_report
    print_tracker_ready_report(cmp_df, exp_id=EXP_ID,
                               per_dim_csv_path=str(Path(CMP_ROOT) / "per_dim_breakdown.csv"))
    print(f"\n[{EXP_ID}] DONE — {CMP_ROOT}")


if __name__ == "__main__":
    main()
