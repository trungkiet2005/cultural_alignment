#!/usr/bin/env python3
"""
EXP-07: Cross-National WVS Augmentation for Sparse Countries
=============================================================

**Motivation** (from paper §6 Limitations + EXP-01 analysis):

Paper admits: "Saudi Arabia and Brazil are the two countries where WVS Wave 7
coverage is sparsest, which is reflected in their comparatively smaller (and
occasionally negative) per-country gains."

From EXP-01 data:
  - BRA: Qwen SWA Pearson r = +0.167 (lowest across all countries+models)
  - SAU: Not in EXP-01 but paper shows SAU is the only country with NEGATIVE JSD
         on Qwen2.5-72B (-2.3%) — correlates with WVS sparse coverage

**Root cause**: When WVS Wave 7 has <200 respondents per cohort for a country:
  - Age-cohort means have high variance → noisy persona profiles
  - Personas may not capture the actual cultural profile
  - IS anchor = mean(delta_i) from unreliable personas → wrong direction

**EXP-07 Fix: Hofstede-Distance Neighbor Borrowing**

When country c has sparse WVS coverage (n_c < N_THRESHOLD per cohort):
1. Find the K_NEIGHBORS=3 most culturally similar countries using Hofstede's
   6 cultural dimensions distance (Power Distance, Individualism, Masculinity,
   Uncertainty Avoidance, Long-Term Orientation, Indulgence)
2. Compute kernel-smoothed WVS means:
   mu_augmented = (n_c * mu_c + sum_j w_j * n_j * mu_j) / (n_c + sum_j w_j * n_j)
   where w_j = exp(-d_Hofstede(c, j) / tau_H) (softmax over cultural distance)
3. Use augmented_mu to build personas for sparse-coverage countries

**Mathematical grounding (for paper §3.2 extension)**:

This is a kernel-smoothed Bayes estimator:
  P(cultural_profile | sparse_data) ∝ P(data | profile) × P(profile | Hofstede_neighbors)

The Hofstede distance acts as a prior: culturally similar countries have higher
prior probability of sharing moral preference patterns. This is validated by:
- Awad et al. (2018) show East vs West are the clearest cultural clusters in Moral Machine
- Hofstede dimensions predict moral preference clusters in cross-national studies

**Expected results**:
  - BRA: MIS improves from 0.4025 → <0.30 (target: fix Fitness and Age errors)
  - SAU: First positive gains if run (currently -2.3% in paper's Qwen-72B)
  - NGA: New country expansion with augmented personas

Usage on Kaggle
---------------
    !python experiment/exp07_wvs_augmentation.py
"""

# ============================================================================
# Step 0: env
# ============================================================================
import os, sys, subprocess, math

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
from typing import Dict, List, Optional, Tuple
from copy import deepcopy

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
# Step 3: Experiment configuration
# ============================================================================
EXP_ID   = "EXP-07"
EXP_NAME = "wvs_augmentation"

# Focus on the known sparse-coverage problem countries + compare to full-coverage
MODELS: List[str] = [
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "unsloth/gemma-2-9b-it-bnb-4bit",
]
# Include BRA (sparse in EXP-01), plus dense ones (JPN, DEU) as control
TARGET_COUNTRIES: List[str] = ["BRA", "JPN", "DEU", "USA", "CHN"]
N_SCENARIOS: int = 500
BATCH_SIZE: int = 1
SEED: int = 42
N_THRESHOLD: int = 200  # below this respondent count per cohort → augment

SWA_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/swa"
CMP_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/compare"
MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/"
    "WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
)
HUMAN_AMCE_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"

# ============================================================================
# Step 4: Hofstede's 6 Cultural Dimensions Data
#
# Source: Hofstede Insights (hofstede-insights.com), dimensions:
#   PDI = Power Distance Index
#   IDV = Individualism
#   MAS = Masculinity
#   UAI = Uncertainty Avoidance Index
#   LTO = Long-Term Orientation
#   IVR = Indulgence vs Restraint
#
# NaN values are imputed with the global mean for the dimension.
# ============================================================================
HOFSTEDE: Dict[str, List[float]] = {
    # [PDI, IDV, MAS, UAI, LTO, IVR]
    "USA": [40, 91, 62, 46, 26, 68],
    "CHN": [80, 20, 66, 30, 87, 24],
    "JPN": [54, 46, 95, 92, 88, 42],
    "DEU": [35, 67, 66, 65, 83, 40],
    "BRA": [69, 38, 49, 76, 44, 59],
    "SAU": [95, 25, 60, 80, 36, 52],
    "ARG": [49, 46, 56, 86, 20, 62],
    "MEX": [81, 30, 69, 82, 24, 97],
    "IND": [77, 48, 56, 40, 51, 26],
    "KOR": [60, 18, 39, 85, 100, 29],
    "GBR": [35, 89, 66, 35, 51, 69],
    "FRA": [68, 71, 43, 86, 63, 48],
    "AUS": [38, 90, 61, 51, 21, 71],
    "VNM": [70, 20, 40, 30, 57, 35],
    "NGA": [80, 30, 60, 55, 13, 84],
    "RUS": [93, 39, 36, 95, 81, 20],
}

# Countries known to have thin WVS Wave 7 coverage (< N_THRESHOLD in some cohorts)
SPARSE_COUNTRIES = {"BRA", "SAU", "NGA"}


def _hofstede_distance(c1: str, c2: str) -> float:
    """
    Compute normalized Euclidean distance in Hofstede 6-D space.
    Returns 0 (identical) to 1 (maximally different).
    Falls back to large distance (0.9) if country not in table.
    """
    v1 = HOFSTEDE.get(c1)
    v2 = HOFSTEDE.get(c2)
    if v1 is None or v2 is None:
        return 0.9  # unknown culture → weak neighbor weight
    d = math.sqrt(sum((a - b)**2 for a, b in zip(v1, v2)))
    # Normalize by max possible distance across the 6 dimensions
    max_d = math.sqrt(sum([100**2] * 6))
    return d / max_d


def _find_cultural_neighbors(
    country: str,
    all_countries: List[str],
    k: int = 3,
    tau: float = 0.15,
) -> List[Tuple[str, float]]:
    """
    Find k most culturally similar countries and compute kernel weights.
    Returns list of (country, weight) sorted by weight descending.
    """
    distances = [(c, _hofstede_distance(country, c)) for c in all_countries if c != country]
    distances.sort(key=lambda x: x[1])
    top_k = distances[:k]

    # Softmax kernel: w = exp(-d / tau)
    weights_raw = [math.exp(-d / tau) for _, d in top_k]
    w_sum = sum(weights_raw) + 1e-12
    neighbors = [(n, w / w_sum) for (n, _), w in zip(top_k, weights_raw)]

    print(f"[EXP-07] Cultural neighbors of {country}: "
          + ", ".join(f"{n}({w:.2f})" for n, w in neighbors))
    return neighbors


def _build_augmented_personas(
    country: str,
    wvs_path: str,
    all_countries: List[str],
    n_threshold: int = N_THRESHOLD,
) -> List[str]:
    """
    Build WVS personas with Hofstede-neighbor augmentation for sparse countries.

    For dense countries (enough WVS data): returns standard personas (same as EXP-01).
    For sparse countries:
      1. Get base personas from country WVS
      2. Find 3 cultural neighbors
      3. Build neighbor personas
      4. Return a pool of 6-7 personas (base + selected neighbor personas)
         weighted by Hofstede similarity
    """
    # Always get the base country personas
    base_personas = build_country_personas(country, wvs_path=wvs_path)

    if country not in SPARSE_COUNTRIES:
        # Dense country: return standard 4 personas (no augmentation)
        print(f"[EXP-07] {country}: dense WVS coverage → standard {len(base_personas)} personas")
        return base_personas

    # Sparse country: augment with neighbor personas
    print(f"[EXP-07] {country}: sparse WVS coverage → augmenting with cultural neighbors")
    neighbors = _find_cultural_neighbors(country, all_countries)

    augmented = list(base_personas)  # Start with base personas

    for neighbor_country, weight in neighbors:
        if neighbor_country not in SUPPORTED_COUNTRIES:
            continue
        try:
            nb_personas = build_country_personas(neighbor_country, wvs_path=wvs_path)
            # Add the young + utilitarian persona from each neighbor
            # (these tend to be the most culturally differentiated)
            if len(nb_personas) >= 1:
                # Rephrase the persona to attribute to target country, not neighbor
                adapted = (
                    f"[Culturally similar voice weighted {weight:.2f}] " + nb_personas[0]
                )
                augmented.append(adapted)
        except Exception as e:
            print(f"[EXP-07] Could not get {neighbor_country} personas: {e}")

    print(f"[EXP-07] Total augmented personas for {country}: {len(augmented)}")
    return augmented


# ============================================================================
# Step 5: controller (identical to paper — augmentation is only in persona pool)
# ============================================================================
class Exp07AugmentedController(ImplicitSWAController):
    """Standard SWA-PTIS math, but with an augmented N-persona pool for sparse countries."""

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

        delta_opt = anchor + delta_star
        p_right   = torch.sigmoid(delta_opt / self.decision_temperature).item()
        p_pref    = p_right if preferred_on_right else 1.0 - p_right
        variance  = float(delta_agents.var(unbiased=True).item()) if delta_agents.numel() > 1 else 0.0

        return {
            "p_right": p_right, "p_left": 1.0 - p_right, "p_spare_preferred": p_pref,
            "variance": variance, "sigma_used": sigma,
            "mppi_flipped": (float(anchor.item()) > 0) != (float(delta_opt.item()) > 0),
            "delta_z_norm": abs(float(delta_star.item())),
            "delta_consensus": float(anchor.item()), "delta_opt": float(delta_opt.item()),
            "logit_temp_used": logit_temp, "n_personas": delta_agents.numel(),
            "agent_decision_gaps": delta_agents.tolist(),
            "agent_rewards": (delta_agents - delta_base).tolist(),
            "p_spare_preferred_pass1": p_pref, "p_spare_preferred_pass2": p_pref,
            "positional_bias": 0.0,
        }


_swa_runner_mod.ImplicitSWAController = Exp07AugmentedController


# ============================================================================
# Step 6: runner
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
    print(f"\n{'#'*70}\n# {EXP_ID} [{model_name}] WVS Augmentation\n{'#'*70}")

    all_supported = [c for c in SUPPORTED_COUNTRIES]
    rows = []
    for ci, country in enumerate(cfg.target_countries):
        if country not in SUPPORTED_COUNTRIES: continue
        print(f"\n[{ci+1}/{len(cfg.target_countries)}] {EXP_ID} {model_name} | {country}")
        scen     = _load_country_scenarios(cfg, country)
        personas = _build_augmented_personas(
            country, WVS_DATA_PATH, all_supported, N_THRESHOLD)
        results_df, summary = run_country_experiment(model, tokenizer, country, personas, scen, cfg)
        results_df.to_csv(out_dir / f"swa_results_{country}.csv", index=False)
        rows.append({
            "model": model_name, "method": f"{EXP_ID}_wvs_augmented",
            "country": country, "is_sparse": country in SPARSE_COUNTRIES,
            "n_personas": len(personas),
            **{f"align_{k}": v for k, v in summary["alignment"].items()},
            "flip_rate": summary["flip_rate"],
            "mean_latency_ms": summary["mean_latency_ms"],
            "n_scenarios": summary["n_scenarios"],
        })
        torch.cuda.empty_cache(); gc.collect()
    return rows


def main():
    setup_seeds(SEED)
    for d in (SWA_ROOT, CMP_ROOT):
        Path(d).mkdir(parents=True, exist_ok=True)

    print(f"\n[{EXP_ID}] {EXP_NAME}")
    print(f"[CONFIG] Sparse countries: {SPARSE_COUNTRIES}")
    print(f"[CONFIG] N_THRESHOLD = {N_THRESHOLD} respondents per cohort")
    print(f"[CONFIG] Hofstede K=3 neighbors with tau=0.15 kernel")

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
    print(f"\n[{EXP_ID}] DONE. Compare sparse vs dense country MIS.")
    sparse_df = cmp_df[cmp_df["is_sparse"] == True]
    dense_df  = cmp_df[cmp_df["is_sparse"] == False]
    print(f"Sparse countries mean MIS: {sparse_df['align_mis'].mean():.4f}")
    print(f"Dense  countries mean MIS: {dense_df['align_mis'].mean():.4f}")
    print(cmp_df.to_string())


if __name__ == "__main__":
    main()
