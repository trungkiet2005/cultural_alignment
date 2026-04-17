#!/usr/bin/env python3
"""
EXP-06: Adaptive Sigma via Per-Model Entropy Calibration
=========================================================

**Motivation** (from paper §7 Conclusion + EXP-01 failure analysis):

The paper explicitly names "adaptive σ tied to per-model entropy" as the top
future direction. EXP-01 confirmed WHY this matters: Qwen-32B achieves 0%
improvement because its fixed σ=0.3 perturbations are at a completely different
scale than the collapsed decision logits (IS behaves as noise).

**Root cause of fixed-σ failure**:
Let H_model = entropy of [p(A), p(B)] at the decision token = -sum(p * log p).
  - Well-spread model (70B, H~0.6 nats): σ=0.3 ≈ 0.5 * p(B)-p(A) → meaningful exploration
  - Collapsed model (32B, H~0.1 nats): σ=0.3 >> decision gap → random walk

**EXP-06 Fix: Entropy-Adaptive Proposal Standard Deviation**

Define H_ref = reference entropy of a "well-calibrated" model (estimated from
the training set or fixed to ln(2)/2 ≈ 0.35 nats, representing the 0.5/0.5 split).

    sigma_k = clamp(sigma_prior × (H_model / H_ref), sigma_floor, sigma_ceil)

Where:
  - sigma_prior = 0.3 (paper's fixed value, used when H_model = H_ref)
  - H_model = mean entropy of the base model on the current batch of scenarios
  - H_ref = 0.35 nats (calibrated reference)
  - sigma_floor = 0.1 (minimum IS exploration)
  - sigma_ceil = 1.5 (maximum IS exploration, prevents instability)

**Mathematical justification (for paper §3.5)**:

The optimal proposal σ* for IS minimises asymptotic IS variance:
    Var(IS) ∝ E_p[(f(ε))² / q*(ε)] / K

For a Gaussian proposal matching the logit-gap distribution:
    σ*_IS ∝ stddev(delta_i) ∝ sqrt(H_model) 

(High entropy ↔ more spread in the decision logits ↔ wider IS exploration needed)

The entropy-calibrated sigma:
    sigma_k = sigma_prior × (H_model / H_ref)
is the first-order approximation to σ*_IS, where sigma_prior at H_ref is the
"paper baseline" and we scale linearly with entropy ratio.

This means:
  - Qwen-32B (H~0.1): sigma ≈ 0.3 × (0.1/0.35) ≈ 0.086 → small, respects the model
  - Qwen-72B (H~0.5): sigma ≈ 0.3 × (0.5/0.35) ≈ 0.43 → wider, more signal
  - Mistral (H~0.3): sigma ≈ 0.3 × (0.3/0.35) ≈ 0.26 → close to paper baseline

**Expected results**:
  - Qwen-32B: improvement from ~0% → +15% (by preventing overshooting)
  - All models: more consistent per-country performance
  - Mistral/non-English: sigma auto-shrinks when multilingual collapse detected

Usage on Kaggle
---------------
    !python experiment/exp06_adaptive_sigma.py
"""

# ============================================================================
# Step 0: env vars
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
        raise RuntimeError("Not on Kaggle and not inside the repo.")
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
import gc, shutil, math
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
    CompareSpec,
    append_rows_csv,
    flatten_per_dim_alignment,
    print_alignment_table,
    print_metric_comparison,
    try_load_reference_comparison,
)

from src.config import SWAConfig, BaselineConfig, resolve_output_dir
from src.constants import COUNTRY_LANG
from src.model import setup_seeds, load_model
from src.data import load_multitp_dataset
from src.scenarios import generate_multitp_scenarios
from src.personas import build_country_personas, SUPPORTED_COUNTRIES
from src.controller import ImplicitSWAController
import src.swa_runner as _swa_runner_mod
from src.swa_runner import run_country_experiment
from src.baseline_runner import run_baseline_vanilla

# ============================================================================
# Step 3: experiment configuration
# ============================================================================
EXP_ID   = "EXP-06"
EXP_NAME = "adaptive_sigma"

MODELS: List[str] = [
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "unsloth/gemma-2-9b-it-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
]
TARGET_COUNTRIES: List[str] = ["USA", "CHN", "JPN", "DEU", "BRA"]
N_SCENARIOS: int = 500
BATCH_SIZE: int = 1
SEED: int = 42

SWA_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/swa"
CMP_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/compare"

MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/"
    "WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
)
HUMAN_AMCE_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"
)

# ============================================================================
# Step 4: Entropy-Adaptive Sigma — per-model calibration constants
#
# H_ref = ln(2)/2 ≈ 0.347 nats = entropy of a 50/50 binary Bernoulli
#   (represents a perfectly uncertain model at the decision token)
# sigma_prior = 0.3 = paper baseline (correct at H_model = H_ref)
# sigma_floor = 0.05 (never fully collapse; prevents IS degeneracy)
# sigma_ceil  = 1.5 (never explore too widely; prevents random walk)
# ============================================================================
H_REF         = math.log(2) / 2   # 0.347 nats — reference entropy
SIGMA_PRIOR   = 0.3               # paper's fixed value (correct at H_ref)
SIGMA_FLOOR   = 0.05              # minimum IS exploration
SIGMA_CEIL    = 1.5               # maximum IS exploration


def _decision_token_entropy(logit_a: float, logit_b: float) -> float:
    """
    Compute binary entropy H = -p log p - (1-p) log(1-p) at the decision tokens.
    Input: raw unnormalised logits for tokens A and B.
    Output: entropy in nats, in [0, ln(2)] ≈ [0, 0.693].
    """
    log_z     = math.log(math.exp(logit_a) + math.exp(logit_b) + 1e-12)
    p_a       = math.exp(logit_a - log_z)
    p_b       = math.exp(logit_b - log_z)
    h         = -(p_a * math.log(p_a + 1e-12) + p_b * math.log(p_b + 1e-12))
    return float(h)


def _adaptive_sigma(
    delta_agents: torch.Tensor,
    base_log_a: float,
    base_log_b: float,
    sigma_prior: float = SIGMA_PRIOR,
    h_ref:       float = H_REF,
    floor:       float = SIGMA_FLOOR,
    ceil:        float = SIGMA_CEIL,
) -> float:
    """
    Compute entropy-calibrated sigma for the IS proposal.

    Formula:
        H_model = binary_entropy(logit_a, logit_b)
        sigma_adaptive = sigma_prior * (H_model / H_ref)
        sigma = clamp(max(sigma_adaptive, per-agent-std), floor, ceil)

    The first max() ensures we never shrink BELOW the empirical agent spread
    (which is an unbiased estimator of the true signal spread when N≥2).
    """
    # Entropy of the base model at decision tokens
    h_model = _decision_token_entropy(base_log_a, base_log_b)

    # Entropy-calibrated sigma (scale sigma_prior linearly with entropy ratio)
    sigma_entropy = sigma_prior * (h_model / h_ref) if h_ref > 0 else sigma_prior

    # Per-agent empirical std (the EXP-01 adaptive component)
    sigma_agents = float(
        delta_agents.std(unbiased=True).item()
    ) if delta_agents.numel() >= 2 else 0.0

    # Take the max (most informative) of the two signals, then clamp
    sigma = float(np.clip(max(sigma_entropy, sigma_agents), floor, ceil))
    return sigma


# ============================================================================
# Step 5: EXP-06 Controller
# ============================================================================
class Exp06AdaptiveSigmaController(ImplicitSWAController):
    """
    SWA-PTIS with entropy-calibrated proposal sigma.

    Key change vs EXP-01 (paper):
        sigma = max(sigma_entropy, sigma_agents)  [entropy-calibrated]
    vs EXP-01:
        sigma = max(sigma_agents, sigma_floor)    [floor-only]

    The PT math, anchor, ESS gate are IDENTICAL to paper.
    This isolates adaptive sigma as the sole variable.
    """

    def _pt_value(self, x: torch.Tensor) -> torch.Tensor:
        a, b, k = self.pt_alpha, self.pt_beta, self.pt_kappa
        return torch.where(x >= 0, x.abs().pow(a), -k * x.abs().pow(b))

    @torch.no_grad()
    def predict(
        self,
        user_query: str,
        preferred_on_right: bool = True,
        phenomenon_category: str = "default",
        lang: str = "en",
    ) -> Dict:
        # Pass 1
        db1, da1, logit_temp = self._extract_logit_gaps(user_query, phenomenon_category, lang)
        swapped_query, swap_changed = self._swap_positional_labels(user_query, lang)
        if swap_changed:
            db2, da2, _ = self._extract_logit_gaps(swapped_query, phenomenon_category, lang)
        else:
            db2, da2 = db1, da1

        delta_base   = (db1 - db2) / 2.0 if swap_changed else db1
        delta_agents = (da1 - da2) / 2.0 if swap_changed else da1

        # =========================================================
        # EXP-06 KEY CHANGE: Entropy-Adaptive Sigma
        # In EXP-01: sigma = max(std(delta_agents), sigma_floor=0.3)
        # In EXP-06: sigma = adaptive_sigma(H_model, delta_agents)
        #            using H_model = entropy of base decision logits
        # =========================================================
        # Extract base model raw logits for entropy computation.
        # db1 = (z_b - z_a) / T_cat.  We need z_a, z_b separately.
        # Approximation: z_a = -db1*T_cat/2, z_b = +db1*T_cat/2 (symmetric)
        # This is only used for H_model estimation — small error is fine.
        half_db1 = float(db1.item()) * logit_temp / 2.0
        sigma = _adaptive_sigma(
            delta_agents,
            base_log_a=-half_db1,
            base_log_b=+half_db1,
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
            "p_right": p_right,
            "p_left": 1.0 - p_right,
            "p_spare_preferred": p_pref,
            "variance": variance,
            "sigma_used": sigma,
            "sigma_entropy": _decision_token_entropy(-half_db1, +half_db1),
            "mppi_flipped": (float(anchor.item()) > 0) != (float(delta_opt.item()) > 0),
            "delta_z_norm": abs(float(delta_star.item())),
            "delta_consensus": float(anchor.item()),
            "delta_opt": float(delta_opt.item()),
            "logit_temp_used": logit_temp,
            "agent_decision_gaps": delta_agents.tolist(),
            "agent_rewards": (delta_agents - delta_base).tolist(),
            "p_spare_preferred_pass1": p_pref,
            "p_spare_preferred_pass2": p_pref,
            "positional_bias": 0.0,
        }


_swa_runner_mod.ImplicitSWAController = Exp06AdaptiveSigmaController


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
            try:
                shutil.rmtree(target)
                print(f"[CLEANUP] removed {target}")
            except Exception as e:
                print(f"[CLEANUP] error: {e}")


def _build_swa_config(model_name):
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
        lambda_coop=0.7,
        K_samples=128,
    )


def _load_country_scenarios(cfg, country):
    lang = COUNTRY_LANG.get(country, "en")
    if cfg.use_real_data:
        df = load_multitp_dataset(
            data_base_path=cfg.multitp_data_path,
            lang=lang, translator=cfg.multitp_translator,
            suffix=cfg.multitp_suffix, n_scenarios=cfg.n_scenarios,
        )
    else:
        df = generate_multitp_scenarios(cfg.n_scenarios, lang=lang)
    df = df.copy()
    df["lang"] = lang
    return df


def _run_swa_for_model(model, tokenizer, model_name) -> List[dict]:
    cfg = _build_swa_config(model_name)
    out_dir = Path(SWA_ROOT) / resolve_output_dir("", model_name).strip("/\\")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(out_dir)
    print(f"\n{'#'*70}\n# {EXP_ID} [{model_name}] Adaptive Sigma\n{'#'*70}")

    rows = []
    for ci, country in enumerate(cfg.target_countries):
        if country not in SUPPORTED_COUNTRIES: continue
        print(f"\n[{ci+1}/{len(cfg.target_countries)}] {EXP_ID} {model_name} | {country}")
        scen     = _load_country_scenarios(cfg, country)
        personas = build_country_personas(country, wvs_path=WVS_DATA_PATH)
        results_df, summary = run_country_experiment(model, tokenizer, country, personas, scen, cfg)
        results_df.to_csv(out_dir / f"swa_results_{country}.csv", index=False)
        append_rows_csv(
            str(Path(CMP_ROOT) / "per_dim_breakdown.csv"),
            flatten_per_dim_alignment(
                summary.get("per_dimension_alignment", {}),
                model=model_name,
                method=f"{EXP_ID}_adaptive_sigma",
                country=country,
            ),
        )

        mean_sigma = float(results_df["sigma_used"].mean()) if "sigma_used" in results_df.columns else float("nan")
        mean_h     = float(results_df["sigma_entropy"].mean()) if "sigma_entropy" in results_df.columns else float("nan")

        rows.append({
            "model": model_name, "method": f"{EXP_ID}_adaptive_sigma", "country": country,
            **{f"align_{k}": v for k, v in summary["alignment"].items()},
            "flip_rate": summary["flip_rate"],
            "mean_latency_ms": summary["mean_latency_ms"],
            "n_scenarios": summary["n_scenarios"],
            "mean_sigma": mean_sigma,
            "mean_entropy": mean_h,
            "sigma_prior": SIGMA_PRIOR,
            "h_ref": H_REF,
        })
        torch.cuda.empty_cache(); gc.collect()
    return rows


def main():
    setup_seeds(SEED)
    for d in (SWA_ROOT, CMP_ROOT):
        Path(d).mkdir(parents=True, exist_ok=True)

    print(f"\n[{EXP_ID}] {EXP_NAME}")
    print(f"[CONFIG] sigma = clamp(max(sigma_entropy, sigma_agents), {SIGMA_FLOOR}, {SIGMA_CEIL})")
    print(f"[CONFIG] sigma_entropy = {SIGMA_PRIOR:.2f} * (H_model / {H_REF:.3f})")
    print(f"[CONFIG] All other params identical to EXP-01 for clean ablation")

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
    print_alignment_table(cmp_df, title=f"{EXP_ID} RESULTS — {EXP_NAME}")

    ref = try_load_reference_comparison()
    if ref is not None:
        print_metric_comparison(
            ref,
            cmp_df,
            title=f"{EXP_ID} vs EXP-01 (reference) — MIS",
            spec=CompareSpec(
                metric_col="align_mis",
                ref_method="swa_ptis",
                cur_method=f"{EXP_ID}_adaptive_sigma",
            ),
        )
        print_metric_comparison(
            ref,
            cmp_df,
            title=f"{EXP_ID} vs EXP-01 (reference) — JSD",
            spec=CompareSpec(
                metric_col="align_jsd",
                ref_method="swa_ptis",
                cur_method=f"{EXP_ID}_adaptive_sigma",
            ),
        )

    print(f"\n[{EXP_ID}] DONE — results under {CMP_ROOT}")


if __name__ == "__main__":
    main()
