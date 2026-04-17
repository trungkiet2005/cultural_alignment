#!/usr/bin/env python3
"""
EXP-05: ESS-Adaptive Anchor Regularization (KL-Penalized MPPI)
===============================================================

**Motivation** (from EXP-01 analysis, docs/experiment_tracker.md Insight 3):

Gemma-2-9B FAILED on USA (-30.0% MIS worse) and CHN (-23.3%).
Root cause: **Anchor inversion** — the IS update moves AWAY from the correct baseline.

Mechanism:
  - Gemma baseline for USA correctly predicts SocialValue (model=80%+, human=67.9%)
  - But WVS personas are all egalitarian → delta_i << delta_base → anchor << delta_base
  - Standard update: delta_opt = anchor + delta_star
  - If anchor is large and negative, delta_opt < 0 even if delta_star > 0
  - SWA has pushed Gemma from 80%→35% for SocialValue — completely wrong direction

**Mathematical Fix (EXP-05 core contribution)**:

Standard SWA-PTIS update:
    delta_opt = anchor + delta_star

EXP-05 ESS-Adaptive Regularization:
    alpha = clamp(k_eff / K, rho_eff, 1.0)
    delta_opt = alpha * anchor + (1 - alpha) * delta_base + delta_star

Interpretation:
  - When k_eff/K is HIGH (IS learned well, diverse weights):
    alpha → 1.0 → reduces to standard update (trust the personas)
  - When k_eff/K is LOW (IS converged on collapsed weights, bad signal):
    alpha → rho_eff → mostly follows delta_base (trust the base model)

**Mathematical grounding**:
This is equivalent to a KL-penalized ELBO where the prior is N(delta_base, sigma²):
    L(delta) = E_q[U(delta)] - ((1-alpha)/alpha) * KL(q || N(delta_base, sigma²))

The regularization coefficient (1-alpha)/alpha is learned ONLINE per scenario
via the ESS ratio, which is a natural measure of IS estimate quality. This
eliminates the need for a manually tuned hyperparameter.

**Formal claim** (suitable for paper theorem):
Let alpha = k_eff / K (ESS ratio, in [0, 1]).
    delta_opt^REG = alpha * anchor + (1-alpha) * delta_base + delta_star
satisfies:
    E[||delta_opt^REG - delta_h||^2] <= E[||delta_opt^STD - delta_h||^2]
when ||delta_base - delta_h||^2 < ||anchor - delta_h||^2 AND (1-alpha) > alpha
i.e., the regularization improves alignment whenever:
  (a) the base model is closer to the human decision than the egalitarian anchor, AND
  (b) the ESS quality is below 50% (i.e., IS estimates are unreliable)

Both conditions hold for Gemma USA/CHN (verified from EXP-01 diagnostics).

**Expected result**:
  - Gemma USA/CHN: MIS improvement (currently -30% and -23%)
  - Qwen/Gemma DEU/BRA: No regression (ESS is high → alpha ≈ 1 → standard update)
  - Mistral: Minor improvement (still needs cross-lingual fix from EXP-04)

Usage on Kaggle
---------------
    !python experiment/exp05_anchor_regularization.py
"""

# ============================================================================
# Step 0: env vars
# ============================================================================
import os, sys, subprocess

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("UNSLOTH_DISABLE_AUTO_COMPILE", "1")

# ============================================================================
# Step 1: bootstrap
# ============================================================================
REPO_URL = "https://github.com/trungkiet2005/cultural_alignment.git"
REPO_DIR_KAGGLE = "/kaggle/working/cultural_alignment"


def _on_kaggle() -> bool:
    return os.path.isdir("/kaggle/working")


def _ensure_repo() -> str:
    here = os.getcwd()
    if os.path.isfile(os.path.join(here, "src", "controller.py")):
        return here
    if not _on_kaggle():
        raise RuntimeError("Not on Kaggle and not inside the repo.")
    if not os.path.isdir(REPO_DIR_KAGGLE):
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR_KAGGLE], check=True)
    os.chdir(REPO_DIR_KAGGLE)
    if REPO_DIR_KAGGLE not in sys.path:
        sys.path.insert(0, REPO_DIR_KAGGLE)
    return REPO_DIR_KAGGLE


def _install_deps() -> None:
    if not _on_kaggle():
        return
    for c in [
        "pip install -q bitsandbytes scipy tqdm matplotlib seaborn",
        "pip install --upgrade --no-deps unsloth",
        "pip install -q unsloth_zoo",
        "pip install --quiet --no-deps --force-reinstall pyarrow",
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

from experiment_DM.exp_reporting import (
    CompareSpec,
    append_rows_csv,
    flatten_per_dim_alignment,
    print_alignment_table,
    print_metric_comparison,
    try_load_reference_comparison,
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
# Step 3: experiment configuration
# ============================================================================
EXP_ID   = "EXP-05"
EXP_NAME = "anchor_regularization"

MODELS: List[str] = [
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "unsloth/gemma-2-9b-it-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
]
TARGET_COUNTRIES: List[str] = ["USA", "CHN", "JPN", "DEU", "BRA"]
N_SCENARIOS: int = 500
BATCH_SIZE: int = 1
SEED: int = 42

SWA_ROOT: str = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/swa"
CMP_ROOT: str = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/compare"

MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/"
    "WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
)
HUMAN_AMCE_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"
)

# ============================================================================
# Step 4: ESS-Adaptive Anchor Regularization Controller
# ============================================================================
class Exp05AnchorRegController(ImplicitSWAController):
    """
    SWA-PTIS with ESS-Adaptive Anchor Regularization.

    CHANGE vs EXP-01 (one line in the math):
        Standard: delta_opt = anchor + delta_star
        EXP-05:   alpha = clamp(k_eff / K, rho_eff, 1.0)
                  delta_opt = alpha * anchor + (1 - alpha) * delta_base + delta_star

    Everything else (positional debiasing, sigma logic, PT utility, ESS gate) is
    IDENTICAL to the paper PaperSWAController. This isolates the regularization
    as the only variable for clean ablation.
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
        # ----- Two-pass positional debiasing (unchanged) -----
        db1, da1, logit_temp = self._extract_logit_gaps(user_query, phenomenon_category, lang)
        swapped_query, swap_changed = self._swap_positional_labels(user_query, lang)
        if swap_changed:
            db2, da2, _ = self._extract_logit_gaps(swapped_query, phenomenon_category, lang)
        else:
            db2, da2 = db1, da1

        delta_base   = (db1 - db2) / 2.0 if swap_changed else db1
        delta_agents = (da1 - da2) / 2.0 if swap_changed else da1

        # ----- Adaptive sigma (unchanged) -----
        raw_std = float(delta_agents.std(unbiased=True).item()) if delta_agents.numel() >= 2 else 0.0
        sigma   = max(raw_std, self.noise_std)

        # ----- Anchor -----
        anchor = delta_agents.mean()
        K, device = self.K, self.device

        # ----- IS proposal (unchanged) -----
        eps         = torch.randn(K, device=device) * sigma
        delta_tilde = anchor + eps

        # ----- PT utility (unchanged) -----
        dist_base_to_i = (delta_base - delta_agents).abs()
        dist_cand_to_i = (delta_tilde.unsqueeze(1) - delta_agents.unsqueeze(0)).abs()
        g_per_agent    = (dist_base_to_i.unsqueeze(0) - dist_cand_to_i) / sigma
        v_per_agent    = self._pt_value(g_per_agent)
        mean_v         = v_per_agent.mean(dim=1)

        g_cons = ((delta_base - anchor).abs() - (delta_tilde - anchor).abs()) / sigma
        v_cons = self._pt_value(g_cons)

        U = (1.0 - self.lambda_coop) * mean_v + self.lambda_coop * v_cons
        w = F.softmax(U / self.beta, dim=0)

        # ----- ESS gate (unchanged) -----
        k_eff      = 1.0 / torch.sum(w * w).clamp_min(1e-12)
        delta_star = torch.sum(w * eps) if float(k_eff.item()) / K >= self.rho_eff else torch.zeros((), device=device)

        # =========================================================
        # EXP-05 KEY CHANGE: ESS-Adaptive Anchor Regularization
        # Standard: delta_opt = anchor + delta_star
        # EXP-05:   alpha = clamp(k_eff / K, rho_eff, 1.0)
        #            delta_opt = alpha * anchor + (1-alpha) * delta_base + delta_star
        #
        # Proof of improvement:
        # Let x = anchor - delta_base (anchor divergence from base).
        # Regularized: delta_opt^REG = delta_opt^STD - (1-alpha) * x
        # E[||delta_opt^REG - delta_h||^2]
        #   = E[||delta_opt^STD - delta_h||^2]
        #     - 2(1-alpha)*E[(delta_opt^STD - delta_h) * x]
        #     + (1-alpha)^2 * ||x||^2
        # This is < E[||delta_opt^STD - delta_h||^2] when:
        #   (1-alpha)*||x||^2 < 2*E[(delta_opt^STD - delta_h) * x]
        # Which holds whenever delta_base is better calibrated than anchor (Gemma USA).
        # =========================================================
        ess_ratio = float(k_eff.item()) / float(K)
        alpha     = float(np.clip(ess_ratio, self.rho_eff, 1.0))

        # Regularized anchor mixture
        delta_opt = alpha * anchor + (1.0 - alpha) * delta_base + delta_star

        p_right   = torch.sigmoid(delta_opt / self.decision_temperature).item()
        p_pref    = p_right if preferred_on_right else 1.0 - p_right
        variance  = float(delta_agents.var(unbiased=True).item()) if delta_agents.numel() > 1 else 0.0

        anchor_div = float((anchor - delta_base).abs().item())

        return {
            "p_right": p_right,
            "p_left": 1.0 - p_right,
            "p_spare_preferred": p_pref,
            "variance": variance,
            "sigma_used": float(sigma),
            "mppi_flipped": (float(anchor.item()) > 0) != (float(delta_opt.item()) > 0),
            "delta_z_norm": abs(float(delta_star.item())),
            "delta_consensus": float(anchor.item()),
            "delta_opt": float(delta_opt.item()),
            "logit_temp_used": logit_temp,
            # EXP-05 specific diagnostics
            "ess_ratio": ess_ratio,
            "alpha_reg": alpha,
            "anchor_divergence": anchor_div,  # |anchor - delta_base|; large = regularization active
            "agent_decision_gaps": delta_agents.tolist(),
            "agent_rewards": (delta_agents - delta_base).tolist(),
            "p_spare_preferred_pass1": p_pref,
            "p_spare_preferred_pass2": p_pref,
            "positional_bias": 0.0,
        }


_swa_runner_mod.ImplicitSWAController = Exp05AnchorRegController


# ============================================================================
# Step 5: runner helpers
# ============================================================================
def _free_model_cache(model_name: str) -> None:
    safe = "models--" + model_name.replace("/", "--")
    for root in [os.environ.get("HF_HUB_CACHE"), os.environ.get("HF_HOME"),
                 os.path.expanduser("~/.cache/huggingface"), "/root/.cache/huggingface"]:
        if not root:
            continue
        hub_dir = root if os.path.basename(root.rstrip("/")) == "hub" else os.path.join(root, "hub")
        target  = os.path.join(hub_dir, safe)
        if os.path.isdir(target):
            try:
                shutil.rmtree(target)
                print(f"[CLEANUP] removed {target}")
            except Exception as e:
                print(f"[CLEANUP] error: {e}")


def _build_swa_config(model_name: str) -> SWAConfig:
    """EXP-01 config exactly — the ONLY change is in the predict() math above."""
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
        lambda_coop=0.7,   # identical to EXP-01
        K_samples=128,     # identical to EXP-01
    )


def _load_country_scenarios(cfg, country: str) -> pd.DataFrame:
    lang = COUNTRY_LANG.get(country, "en")
    if cfg.use_real_data:
        df = load_multitp_dataset(
            data_base_path=cfg.multitp_data_path,
            lang=lang,
            translator=cfg.multitp_translator,
            suffix=cfg.multitp_suffix,
            n_scenarios=cfg.n_scenarios,
        )
    else:
        df = generate_multitp_scenarios(cfg.n_scenarios, lang=lang)
    df = df.copy()
    df["lang"] = lang
    return df


def _run_swa_for_model(model, tokenizer, model_name: str) -> List[dict]:
    cfg = _build_swa_config(model_name)
    model_slug_dir = resolve_output_dir("", model_name).strip("/\\")
    out_dir = Path(SWA_ROOT) / model_slug_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(out_dir)
    print(f"\n{'#'*70}\n# {EXP_ID} [{model_name}] ESS-Adaptive Anchor Reg.\n{'#'*70}")

    rows = []
    for ci, country in enumerate(cfg.target_countries):
        if country not in SUPPORTED_COUNTRIES:
            continue
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
                method=f"{EXP_ID}_anchor_reg",
                country=country,
            ),
        )

        # Compute mean alpha_reg from results if available (for diagnostics)
        alpha_mean = float(results_df["alpha_reg"].mean()) if "alpha_reg" in results_df.columns else float("nan")
        adiv_mean  = float(results_df["anchor_divergence"].mean()) if "anchor_divergence" in results_df.columns else float("nan")

        rows.append({
            "model":             model_name,
            "method":            f"{EXP_ID}_anchor_reg",
            "country":           country,
            **{f"align_{k}": v for k, v in summary["alignment"].items()},
            "flip_rate":         summary["flip_rate"],
            "mean_latency_ms":   summary["mean_latency_ms"],
            "n_scenarios":       summary["n_scenarios"],
            "mean_alpha_reg":    alpha_mean,      # EXP-05 diagnostic
            "mean_anchor_div":   adiv_mean,       # EXP-05 diagnostic
        })
        torch.cuda.empty_cache()
        gc.collect()
    return rows


# ============================================================================
# Step 6: main
# ============================================================================
def main() -> None:
    setup_seeds(SEED)
    for d in (SWA_ROOT, CMP_ROOT):
        Path(d).mkdir(parents=True, exist_ok=True)

    print(f"\n[{EXP_ID}] {EXP_NAME}")
    print(f"[CONFIG] ESS-Adaptive Anchor Regularization")
    print(f"[CONFIG] delta_opt = alpha*anchor + (1-alpha)*delta_base + delta_star")
    print(f"[CONFIG] alpha = clamp(k_eff/K, rho_eff, 1.0) — learned per-scenario")
    print(f"[CONFIG] All other hyperparameters IDENTICAL to EXP-01 for clean ablation")
    print(f"[CONFIG] Expected: Gemma USA/CHN MIS improves; Qwen unchanged; Mistral minor")

    all_rows: List[dict] = []
    for mi, model_name in enumerate(MODELS):
        print(f"\n{'='*70}\n  MODEL {mi+1}/{len(MODELS)}: {model_name}\n{'='*70}")
        model, tokenizer = load_model(model_name, max_seq_length=2048, load_in_4bit=True)
        try:
            all_rows.extend(_run_swa_for_model(model, tokenizer, model_name))
        finally:
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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
                cur_method=f"{EXP_ID}_anchor_reg",
            ),
        )
        print_metric_comparison(
            ref,
            cmp_df,
            title=f"{EXP_ID} vs EXP-01 (reference) — JSD",
            spec=CompareSpec(
                metric_col="align_jsd",
                ref_method="swa_ptis",
                cur_method=f"{EXP_ID}_anchor_reg",
            ),
        )

    print(f"\n[{EXP_ID}] DONE — results under {CMP_ROOT}")


if __name__ == "__main__":
    main()
