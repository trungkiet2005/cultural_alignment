#!/usr/bin/env python3
"""
EXP-09: Hierarchical IS with Country-Level Prior (Two-Level Bayesian IS)
========================================================================

**Motivation** (from paper §6 + conclusion + ablation analysis):

The paper's ablation (Table 3) shows that removing the utilitarian persona drops
JSD by 43.5% and Pearson r by 0.283. This is a massive single-component effect.
The conclusion attributes this to the utilitarian persona "anchoring the ensemble
along the numerosity dimension."

However, this is currently achieved by including 1 country-INVARIANT utilitarian
in a pool of 3 country-SPECIFIC WVS agents (N=4 total). This design has a tension:
  - The utilitarian agent is crucial but country-invariant
  - The WVS agents are country-specific but all skew egalitarian (SocialValue fail)

**Root cause of per-country inconsistency**:
The IS update is scenario-LOCAL. Each scenario is treated independently, with no
mechanism to ensure that the country's OVERALL preference profile is respected.
This means the IS can make good scenario-level adjustments that are inconsistent
at the country level (e.g., each individual scenario slightly overestimates female
preference, but no single scenario's IS correction catches this systematic drift).

**EXP-09 Fix: Two-Level Hierarchical IS**

**Level 1 (Country Prior)**: After processing the first N_WARMUP scenarios for a
country, compute the running average delta_opt across all scenarios → this becomes
the country-level prior delta_country. This is a summary of "what correction the
IS has been applying on average" for this country.

**Level 2 (Scenario Micro)**: For each subsequent scenario, the IS update is:
    delta_opt = delta_country + alpha_h * delta_star_micro
    alpha_h = exp(-step / decay_tau)  (annealing: early scenarios explore freely,
                                        later scenarios anchor to the country prior)

This creates a hierarchical Bayesian structure:
    P(delta | x, c) = P(x | delta) * P(delta | delta_country)
where P(delta | delta_country) = N(delta_country, sigma_h^2) is the country prior.

**Mathematical grounding (for paper §3.3 extension)**:

This is the IS version of hierarchical Bayes:
    Micro update: delta_star ~ IS optimal for scenario x
    Prior update: delta_country = (1-beta) * delta_country_prev + beta * delta_opt_current

The annealing parameter alpha_h controls how quickly the country prior is trusted:
    alpha_h = 1 fully trusts the country prior (ignores per-scenario variation)
    alpha_h = 0 is standard EXP-01 (no country prior)

In the limit as N_scenarios → ∞, if the model is well-calibrated:
    delta_country → E[delta_opt | c]
    delta_opt → delta_country  (self-consistent fixed point)

This is the hierarchical inference analog of empirical Bayes.

**Expected results**:
  - Reduces per-country JSD variance (more consistent country-level alignment)
  - Improves countries where the IS makes inconsistent decisions (Brazil, France)
  - Improves overall Pearson r by better aligning the ranking of dimensions

Usage on Kaggle
---------------
    !python experiment/exp09_hierarchical_is.py
"""

# ============================================================================
# Step 0: env
# ============================================================================
import os, sys, subprocess

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("UNSLOTH_DISABLE_AUTO_COMPILE", "1")

REPO_URL = "https://github.com/trungkiet2005/cultural_alignment.git"
REPO_DIR_KAGGLE = "/kaggle/working/cultural_alignment"

def _on_kaggle(): return os.path.isdir("/kaggle/working")

def _ensure_repo():
    here = os.getcwd()
    if os.path.isfile(os.path.join(here, "src", "controller.py")):
        return here
    if not _on_kaggle():
        raise RuntimeError("Not on Kaggle, not inside the repo.")
    if not os.path.isdir(REPO_DIR_KAGGLE):
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR_KAGGLE], check=True)
    os.chdir(REPO_DIR_KAGGLE)
    sys.path.insert(0, REPO_DIR_KAGGLE)
    return REPO_DIR_KAGGLE

def _install_deps():
    if not _on_kaggle(): return
    for c in [
        "pip install -q bitsandbytes scipy tqdm",
        "pip install --upgrade --no-deps unsloth",
        "pip install -q unsloth_zoo",
        "pip install --quiet 'datasets>=3.4.1,<4.4.0'",
    ]:
        subprocess.run(c, shell=True, check=False)

_REPO_DIR = _ensure_repo()
_install_deps()

# ============================================================================
# Step 2: imports
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
# Step 3: experiment configuration
# ============================================================================
EXP_ID   = "EXP-09"
EXP_NAME = "hierarchical_is"

MODELS: List[str] = [
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "unsloth/gemma-2-9b-it-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
]
TARGET_COUNTRIES: List[str] = ["USA", "CHN", "JPN", "DEU", "BRA"]
N_SCENARIOS: int = 500
BATCH_SIZE: int = 1
SEED: int = 42

# ============================================================================
# Hierarchical IS hyperparameters
# ============================================================================
N_WARMUP    = 50     # number of scenarios before activating country prior
DECAY_TAU   = 100    # annealing decay constant (alpha_h = exp(-step/tau))
BETA_EMA    = 0.1    # EMA decay rate for country prior update
SIGMA_H     = 0.5    # prior width on the country prior N(delta_country, sigma_h^2)

SWA_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/swa"
CMP_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/compare"
MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/"
    "WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
)
HUMAN_AMCE_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"


# ============================================================================
# Step 4: Country-Prior State (shared across scenarios within a country run)
# ============================================================================
class CountryPriorState:
    """
    Maintains the running country-level prior delta_country using EMA.
    Thread-unsafe (single-threaded run only, which is our case).

    Update rule:
        delta_country = (1 - beta) * delta_country + beta * delta_opt

    Annealing rule for alpha_h (how much the current scenario trusts the prior):
        alpha_h = 1 - exp(-step / decay_tau)  [starts at 0, approaches 1]

    This means early scenarios are treated as pure local micro-IS,
    and later scenarios increasingly defer to the emerging country prior.
    """

    def __init__(
        self,
        init_value: float = 0.0,
        beta:       float = BETA_EMA,
        decay_tau:  float = DECAY_TAU,
        n_warmup:   int   = N_WARMUP,
    ):
        self.delta_country = init_value
        self.beta          = beta
        self.decay_tau     = decay_tau
        self.n_warmup      = n_warmup
        self.step          = 0
        self._history: List[float] = []

    def alpha_h(self) -> float:
        """
        Annealing mixing weight for country prior.
        0 during warmup, gradually increases to 1 after N_WARMUP scenarios.
        """
        if self.step < self.n_warmup:
            return 0.0
        t = self.step - self.n_warmup
        return 1.0 - np.exp(-t / self.decay_tau)

    def update(self, delta_opt: float) -> None:
        """Update country prior EMA with the latest scenario's delta_opt."""
        self.delta_country = (1.0 - self.beta) * self.delta_country + self.beta * delta_opt
        self._history.append(delta_opt)
        self.step += 1

    def apply_prior(self, delta_opt_micro: float) -> float:
        """
        Mix micro IS result with country prior.
        Returns: alpha_h * delta_country + (1 - alpha_h) * delta_opt_micro
        """
        a = self.alpha_h()
        return a * self.delta_country + (1.0 - a) * delta_opt_micro

    @property
    def stats(self) -> Dict:
        h = self._history
        return {
            "step":          self.step,
            "delta_country": self.delta_country,
            "alpha_h":       self.alpha_h(),
            "history_std":   float(np.std(h)) if len(h) > 1 else 0.0,
        }


# Global state: reset per country per model
_COUNTRY_PRIOR_STATE: Dict[str, CountryPriorState] = {}


# ============================================================================
# Step 5: Hierarchical IS Controller
# ============================================================================
class Exp09HierarchicalController(ImplicitSWAController):
    """
    Two-Level Hierarchical IS:
      Level 1 (Macro): Country-level prior delta_country (EMA over processed scenarios)
      Level 2 (Micro): Standard PT-IS correction per scenario

    Final prediction: delta_opt = apply_prior(delta_opt_micro)
    Where apply_prior mixes country prior with micro update based on step.
    """

    def __init__(self, *args, country: str = "UNKNOWN", **kwargs):
        super().__init__(*args, **kwargs)
        self.country = country

    def _get_prior_state(self) -> CountryPriorState:
        if self.country not in _COUNTRY_PRIOR_STATE:
            _COUNTRY_PRIOR_STATE[self.country] = CountryPriorState()
        return _COUNTRY_PRIOR_STATE[self.country]

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

        U = (1.0 - self.lambda_coop) * mean_v + self.lambda_coop * v_cons
        w = F.softmax(U / self.beta, dim=0)

        k_eff      = 1.0 / torch.sum(w * w).clamp_min(1e-12)
        delta_star = torch.sum(w * eps) if float(k_eff.item()) / K >= self.rho_eff else torch.zeros((), device=device)

        # =========================================================
        # EXP-09 KEY CHANGE: Apply Hierarchical Country Prior
        # delta_opt_micro = anchor + delta_star  (standard IS result)
        # delta_opt_hier  = apply_prior(delta_opt_micro)
        #                 = alpha_h * delta_country + (1-alpha_h) * delta_opt_micro
        # =========================================================
        delta_opt_micro = float((anchor + delta_star).item())
        prior_state     = self._get_prior_state()
        delta_opt_final = prior_state.apply_prior(delta_opt_micro)

        # Update country prior with this scenario's micro result
        prior_state.update(delta_opt_micro)

        prior_stats = prior_state.stats

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
            "delta_opt_micro": delta_opt_micro, "delta_country": prior_stats["delta_country"],
            "alpha_h": prior_stats["alpha_h"], "prior_step": prior_stats["step"],
            "logit_temp_used": logit_temp, "n_personas": delta_agents.numel(),
            "agent_decision_gaps": delta_agents.tolist(),
            "agent_rewards": (delta_agents - delta_base).tolist(),
            "p_spare_preferred_pass1": p_pref, "p_spare_preferred_pass2": p_pref,
            "positional_bias": 0.0,
        }


_swa_runner_mod.ImplicitSWAController = Exp09HierarchicalController


# ============================================================================
# Step 6: Runner with country-state reset
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
        lambda_coop=0.7, K_samples=128,
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
    cfg = _build_swa_config(model_name)
    out_dir = Path(SWA_ROOT) / resolve_output_dir("", model_name).strip("/\\")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(out_dir)
    print(f"\n{'#'*70}\n# {EXP_ID} [{model_name}] Hierarchical IS\n{'#'*70}")

    rows = []
    for ci, country in enumerate(cfg.target_countries):
        if country not in SUPPORTED_COUNTRIES: continue

        # CRITICAL: Reset country prior state for each country x model combination
        _COUNTRY_PRIOR_STATE.clear()
        _COUNTRY_PRIOR_STATE[country] = CountryPriorState()
        print(f"\n[{ci+1}/{len(cfg.target_countries)}] {EXP_ID} {model_name} | {country} (prior reset)")

        scen     = _load_country_scenarios(cfg, country)
        personas = build_country_personas(country, wvs_path=WVS_DATA_PATH)

        # Inject country into controller at construction time via monkey-patch
        orig_init = Exp09HierarchicalController.__init__

        def patched_init(self, *args, country=country, **kwargs):
            orig_init(self, *args, country=country, **kwargs)

        Exp09HierarchicalController.__init__ = patched_init
        _swa_runner_mod.ImplicitSWAController = Exp09HierarchicalController

        results_df, summary = run_country_experiment(model, tokenizer, country, personas, scen, cfg)
        Exp09HierarchicalController.__init__ = orig_init  # restore

        results_df.to_csv(out_dir / f"swa_results_{country}.csv", index=False)

        # Get final country prior state stats
        prior_stats = _COUNTRY_PRIOR_STATE.get(country, CountryPriorState()).stats

        rows.append({
            "model": model_name, "method": f"{EXP_ID}_hierarchical_is",
            "country": country,
            **{f"align_{k}": v for k, v in summary["alignment"].items()},
            "flip_rate": summary["flip_rate"],
            "mean_latency_ms": summary["mean_latency_ms"],
            "n_scenarios": summary["n_scenarios"],
            "final_delta_country": prior_stats["delta_country"],
            "final_alpha_h": prior_stats["alpha_h"],
            "history_std": prior_stats["history_std"],
            "n_warmup": N_WARMUP,
            "decay_tau": DECAY_TAU,
            "beta_ema": BETA_EMA,
        })
        torch.cuda.empty_cache(); gc.collect()
    return rows


def main():
    setup_seeds(SEED)
    for d in (SWA_ROOT, CMP_ROOT):
        Path(d).mkdir(parents=True, exist_ok=True)

    print(f"\n[{EXP_ID}] {EXP_NAME}")
    print(f"[CONFIG] N_WARMUP={N_WARMUP}, DECAY_TAU={DECAY_TAU}, BETA_EMA={BETA_EMA}")
    print(f"[CONFIG] alpha_h = 1 - exp(-step / {DECAY_TAU}) after warmup")
    print(f"[CONFIG] delta_opt = alpha_h * delta_country + (1-alpha_h) * delta_opt_micro")

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
    print(f"\n[{EXP_ID}] DONE. Key: delta_country convergence + final alpha_h per country.")
    print(cmp_df[["model", "country", "align_mis", "final_delta_country", "final_alpha_h"]].to_string())


if __name__ == "__main__":
    main()
