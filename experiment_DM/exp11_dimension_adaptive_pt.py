#!/usr/bin/env python3
"""
EXP-11: Dimension-Decoupled Adaptive Prospect Theory
=====================================================

**Motivation** (novel theoretical contribution for NeurIPS):

The core SWA-PTIS paper uses FIXED Prospect Theory parameters (α=β=0.88, κ=2.25)
across ALL six moral dimensions. This is a strong assumption: it implies that human
loss aversion is identical whether the moral trade-off involves Species (human vs.
animal), SocialValue (executive vs. homeless), or Utilitarianism (more vs. fewer lives).

Behavioral economics research shows that loss aversion is DOMAIN-SPECIFIC:
  - Kahneman & Tversky (1979): original calibration on monetary gambles
  - Weber & Johnson (2009): risk preferences differ across health, financial,
    ethical, and social domains
  - Fehr & Schmidt (1999): inequity aversion ≠ loss aversion in social contexts

**The SocialValue failure is a PT parameter problem:**

The tracker shows SocialValue_High is the #1 error (~27pp). Why? The personas
(all egalitarian) produce a NEGATIVE anchor for SocialValue. With κ=2.25,
the PT value function HEAVILY penalizes any candidate that moves AWAY from this
wrong anchor (loss aversion). The IS update is trapped: it cannot explore enough
because every candidate that corrects the SocialValue bias incurs a 2.25x loss
penalty on one or more egalitarian personas.

**Fix: dimension-specific κ (loss aversion modulation)**

For dimensions where the anchor is known to be biased (SocialValue), REDUCE κ
to allow the IS to explore more aggressively. For dimensions where the anchor
is already good (Species, Gender), INCREASE κ to prevent over-correction.

This is theoretically novel:
  "Heterogeneous Prospect Theory parameters across moral domains,
   calibrated to the per-dimension anchor quality"

**Mathematical formulation:**

    v_d(x) = { x^α_d        if x ≥ 0
             { -κ_d · |x|^β_d  if x < 0

where d ∈ {Species, Gender, Age, Fitness, SocialValue, Utilitarianism}.

The dimension-specific parameters are set based on the STRUCTURAL properties
of the persona pool for that dimension:

    κ_d = κ_base · (1 + γ · sign_quality_d)

where sign_quality_d measures whether the persona anchor tends to agree (+)
or disagree (-) with human ground truth for dimension d. Higher sign_quality
→ higher κ (trust the anchor, penalize deviations). Lower sign_quality →
lower κ (distrust the anchor, allow exploration).

For this experiment, we use FIXED dimension-specific κ derived from the
EXP-01 per-dimension error analysis:
  - SocialValue: anchor is systematically wrong → κ = 1.25 (low loss aversion)
  - Species: anchor is good → κ = 3.00 (high loss aversion, prevent over-correction)
  - Gender: anchor is moderate → κ = 2.25 (standard)
  - Age: anchor is moderate-poor → κ = 1.75
  - Fitness: anchor is moderate → κ = 2.25 (standard)
  - Utilitarianism: anchor quality varies → κ = 2.00

**Additionally: dimension-specific σ (proposal width)**

Dimensions where the anchor is wrong need WIDER proposals to find the correct
correction. Dimensions where it's right need NARROW proposals to avoid over-shooting.

    σ_d = σ_base · scaling_d

**Expected results:**
  - SocialValue gap reduced (lower κ allows bigger corrections)
  - Species/Gender stability maintained (higher κ prevents over-correction)
  - Better per-dimension JSD uniformity
  - Novel theoretical contribution for paper §6 extension

Usage on Kaggle
---------------
    !python experiment_DM/exp11_dimension_adaptive_pt.py
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
EXP_ID   = "EXP-11"
EXP_NAME = "dimension_adaptive_pt"

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
# Dimension-Specific PT Parameters (KEY INNOVATION)
# ============================================================================
# Derived from EXP-01 per-dimension error analysis:
#   - SocialValue: systematic -27pp anchor bias → need LOW κ (allow exploration)
#   - Species: anchor generally correct → HIGH κ (prevent over-correction)
#   - Gender: moderate anchor quality → standard κ
#   - Age: moderate-poor anchor → slightly lower κ
#   - Fitness: moderate anchor → standard κ
#   - Utilitarianism: varies by country → slightly lower κ

DIMENSION_PT_PARAMS: Dict[str, Dict[str, float]] = {
    "Species":        {"kappa": 3.00, "alpha": 0.88, "beta": 0.88, "sigma_scale": 0.8},
    "Gender":         {"kappa": 2.25, "alpha": 0.88, "beta": 0.88, "sigma_scale": 1.0},
    "Age":            {"kappa": 1.75, "alpha": 0.88, "beta": 0.88, "sigma_scale": 1.2},
    "Fitness":        {"kappa": 2.25, "alpha": 0.88, "beta": 0.88, "sigma_scale": 1.0},
    "SocialValue":    {"kappa": 1.25, "alpha": 0.85, "beta": 0.90, "sigma_scale": 1.5},
    "Utilitarianism": {"kappa": 2.00, "alpha": 0.88, "beta": 0.88, "sigma_scale": 1.1},
}
DEFAULT_PT_PARAMS = {"kappa": 2.25, "alpha": 0.88, "beta": 0.88, "sigma_scale": 1.0}

SWA_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/swa"
CMP_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/compare"
MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/"
    "WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
)
HUMAN_AMCE_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"


# ============================================================================
# Step 5: Dimension-Adaptive PT Controller
# ============================================================================
class Exp11DimAdaptiveController(ImplicitSWAController):
    """
    Dimension-Decoupled Adaptive Prospect Theory.

    Key change: the PT value function v(x) uses dimension-specific parameters
    (κ_d, α_d, β_d) instead of global fixed values. The proposal width σ is
    also scaled per dimension.

    This captures the intuition that moral risk preferences are domain-specific:
    loss aversion for Species decisions ≠ loss aversion for SocialValue decisions.
    """

    def _pt_value_dim(self, x: torch.Tensor, kappa: float, alpha: float, beta: float) -> torch.Tensor:
        """Dimension-specific Prospect Theory value function."""
        return torch.where(x >= 0, x.abs().pow(alpha), -kappa * x.abs().pow(beta))

    @torch.no_grad()
    def predict(self, user_query, preferred_on_right=True, phenomenon_category="default", lang="en"):
        # ── Look up dimension-specific PT parameters ──
        dim_params = DIMENSION_PT_PARAMS.get(phenomenon_category, DEFAULT_PT_PARAMS)
        kappa_d     = dim_params["kappa"]
        alpha_d     = dim_params["alpha"]
        beta_d      = dim_params["beta"]
        sigma_scale = dim_params["sigma_scale"]

        # ── Standard two-pass debiasing ──
        db1, da1, logit_temp = self._extract_logit_gaps(user_query, phenomenon_category, lang)
        swapped_query, swap_changed = self._swap_positional_labels(user_query, lang)
        if swap_changed:
            db2, da2, _ = self._extract_logit_gaps(swapped_query, phenomenon_category, lang)
        else:
            db2, da2 = db1, da1

        delta_base   = (db1 - db2) / 2.0 if swap_changed else db1
        delta_agents = (da1 - da2) / 2.0 if swap_changed else da1

        # ── Dimension-scaled proposal width ──
        base_sigma = max(
            float(delta_agents.std(unbiased=True).item()) if delta_agents.numel() >= 2 else 0.0,
            self.noise_std
        )
        sigma = base_sigma * sigma_scale  # dimension-specific scaling

        anchor = delta_agents.mean()
        K, device = self.K, self.device

        eps         = torch.randn(K, device=device) * sigma
        delta_tilde = anchor + eps

        # ── Per-agent gain (standard) ──
        dist_base_to_i = (delta_base - delta_agents).abs()
        dist_cand_to_i = (delta_tilde.unsqueeze(1) - delta_agents.unsqueeze(0)).abs()
        g_per_agent    = (dist_base_to_i.unsqueeze(0) - dist_cand_to_i) / sigma

        # ── DIMENSION-SPECIFIC PT value function ──
        v_per_agent = self._pt_value_dim(g_per_agent, kappa_d, alpha_d, beta_d)
        mean_v      = v_per_agent.mean(dim=1)

        g_cons = ((delta_base - anchor).abs() - (delta_tilde - anchor).abs()) / sigma
        v_cons = self._pt_value_dim(g_cons, kappa_d, alpha_d, beta_d)

        U = (1.0 - self.lambda_coop) * mean_v + self.lambda_coop * v_cons
        w = F.softmax(U / self.beta, dim=0)

        k_eff      = 1.0 / torch.sum(w * w).clamp_min(1e-12)
        delta_star = torch.sum(w * eps) if float(k_eff.item()) / K >= self.rho_eff else torch.zeros((), device=device)

        delta_opt = float((anchor + delta_star).item())

        p_right = torch.sigmoid(
            torch.tensor(delta_opt / self.decision_temperature)
        ).item()
        p_pref  = p_right if preferred_on_right else 1.0 - p_right
        variance = float(delta_agents.var(unbiased=True).item()) if delta_agents.numel() > 1 else 0.0

        return {
            "p_right": p_right, "p_left": 1.0 - p_right, "p_spare_preferred": p_pref,
            "variance": variance, "sigma_used": sigma,
            "mppi_flipped": (float(anchor.item()) > 0) != (delta_opt > 0),
            "delta_z_norm": abs(delta_opt - float(anchor.item())),
            "delta_consensus": float(anchor.item()), "delta_opt": delta_opt,
            # EXP-11 diagnostics
            "dim_category": phenomenon_category,
            "kappa_d": kappa_d, "alpha_d": alpha_d, "beta_d": beta_d,
            "sigma_scale": sigma_scale, "sigma_base": base_sigma,
            "ess_ratio": float(k_eff.item()) / K,
            "logit_temp_used": logit_temp, "n_personas": delta_agents.numel(),
            "agent_decision_gaps": delta_agents.tolist(),
            "agent_rewards": (delta_agents - delta_base).tolist(),
            "p_spare_preferred_pass1": p_pref, "p_spare_preferred_pass2": p_pref,
            "positional_bias": 0.0,
        }


_swa_runner_mod.ImplicitSWAController = Exp11DimAdaptiveController


# ============================================================================
# Step 6: Runner
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
    print(f"\n{'#'*70}\n# {EXP_ID} [{model_name}] Dimension-Adaptive PT\n{'#'*70}")

    rows = []
    for ci, country in enumerate(cfg.target_countries):
        if country not in SUPPORTED_COUNTRIES: continue
        print(f"\n[{ci+1}/{len(cfg.target_countries)}] {EXP_ID} {model_name} | {country}")

        # Print dimension-specific PT params
        print(f"  [DIM-PT] Dimension-specific parameters:")
        for dim, params in sorted(DIMENSION_PT_PARAMS.items()):
            print(f"    {dim:<18s}  κ={params['kappa']:.2f}  α={params['alpha']:.2f}  "
                  f"β={params['beta']:.2f}  σ_scale={params['sigma_scale']:.1f}")

        scen     = _load_country_scenarios(cfg, country)
        personas = build_country_personas(country, wvs_path=WVS_DATA_PATH)
        _swa_runner_mod.ImplicitSWAController = Exp11DimAdaptiveController

        results_df, summary = run_country_experiment(model, tokenizer, country, personas, scen, cfg)

        results_df.to_csv(out_dir / f"swa_results_{country}.csv", index=False)
        append_rows_csv(
            str(Path(CMP_ROOT) / "per_dim_breakdown.csv"),
            flatten_per_dim_alignment(
                summary.get("per_dimension_alignment", {}),
                model=model_name,
                method=f"{EXP_ID}_dim_adaptive_pt",
                country=country,
            ),
        )

        # Compute per-dimension κ usage statistics from results
        dim_kappa_stats = {}
        if "kappa_d" in results_df.columns and "dim_category" in results_df.columns:
            for dim in DIMENSION_PT_PARAMS:
                dim_mask = results_df["dim_category"] == dim
                if dim_mask.any():
                    dim_kappa_stats[dim] = {
                        "count": int(dim_mask.sum()),
                        "mean_ess": float(results_df.loc[dim_mask, "ess_ratio"].mean()) if "ess_ratio" in results_df.columns else float("nan"),
                    }

        rows.append({
            "model": model_name, "method": f"{EXP_ID}_dim_adaptive_pt",
            "country": country,
            **{f"align_{k}": v for k, v in summary["alignment"].items()},
            "flip_rate": summary["flip_rate"],
            "mean_latency_ms": summary["mean_latency_ms"],
            "n_scenarios": summary["n_scenarios"],
        })

        # ── Detailed per-dimension log ──
        pda = summary.get("per_dimension_alignment", {})
        if pda:
            print(f"\n  ┌── Per-Dimension Alignment ({country}) ──")
            for dim_key, dim_data in sorted(pda.items()):
                human_val = dim_data.get("human", float("nan"))
                model_val = dim_data.get("model", float("nan"))
                err       = dim_data.get("error", model_val - human_val)
                # Find which PT category this maps to
                base_dim = dim_key.split("_")[0] if "_" in dim_key else dim_key
                pt = DIMENSION_PT_PARAMS.get(base_dim, DEFAULT_PT_PARAMS)
                kst = dim_kappa_stats.get(base_dim, {})
                print(f"  │  {dim_key:<25s}  human={human_val:6.1f}  model={model_val:6.1f}  "
                      f"err={err:+6.1f}pp  (κ={pt['kappa']:.2f}, σ×{pt['sigma_scale']:.1f}, "
                      f"n={kst.get('count', '?')}, ESS={kst.get('mean_ess', float('nan')):.3f})")
            print(f"  └── MIS={summary['alignment']['mis']:.4f}  JSD={summary['alignment']['jsd']:.4f}  "
                  f"r={summary['alignment']['pearson']:.3f}  MAE={summary['alignment']['mae']:.2f}  "
                  f"Flip={summary['flip_rate']:.1%}")

        torch.cuda.empty_cache(); gc.collect()
    return rows


def main():
    setup_seeds(SEED)
    for d in (SWA_ROOT, CMP_ROOT):
        Path(d).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  {EXP_ID}: {EXP_NAME.upper()}")
    print(f"  Novel: Heterogeneous PT parameters across moral dimensions")
    print(f"{'='*70}")
    print(f"[CONFIG] Dimension-specific κ (loss aversion):")
    for dim, params in sorted(DIMENSION_PT_PARAMS.items()):
        print(f"  {dim:<18s}  κ={params['kappa']:.2f}  α={params['alpha']:.2f}  "
              f"β={params['beta']:.2f}  σ_scale={params['sigma_scale']:.1f}")
    print(f"[THEORY] v_d(x) = x^α_d (gain) or -κ_d·|x|^β_d (loss)")
    print(f"[THEORY] Low κ → allow exploration (SocialValue)")
    print(f"[THEORY] High κ → prevent over-correction (Species)")

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

    # ── Final comprehensive report ──
    print(f"\n\n{'#'*70}")
    print(f"# {EXP_ID} FINAL REPORT — {EXP_NAME.upper()}")
    print(f"{'#'*70}")
    print_alignment_table(cmp_df, title=f"{EXP_ID} RESULTS — {EXP_NAME}")

    # ── Aggregate statistics ──
    print(f"\n{'─'*70}")
    print(f"  AGGREGATE STATISTICS")
    print(f"{'─'*70}")
    for model_name in MODELS:
        m_df = cmp_df[cmp_df["model"] == model_name]
        if m_df.empty: continue
        short = model_name.split("/")[-1][:20]
        print(f"  {short:<20s}  MIS={m_df['align_mis'].mean():.4f}  "
              f"JSD={m_df['align_jsd'].mean():.4f}  "
              f"r={m_df['align_pearson'].mean():+.3f}  "
              f"MAE={m_df['align_mae'].mean():.2f}  "
              f"Flip={m_df['flip_rate'].mean():.1%}")

    overall_mis = cmp_df["align_mis"].mean()
    print(f"\n  OVERALL MEAN MIS = {overall_mis:.4f}  (EXP-01 baseline: 0.4269)")

    # ── Reference comparison ──
    ref = try_load_reference_comparison()
    if ref is not None:
        for metric, label in [("align_mis", "MIS"), ("align_jsd", "JSD")]:
            print_metric_comparison(
                ref, cmp_df,
                title=f"{EXP_ID} vs EXP-01 (reference) — {label}",
                spec=CompareSpec(
                    metric_col=metric,
                    ref_method="swa_ptis",
                    cur_method=f"{EXP_ID}_dim_adaptive_pt",
                ),
            )

    # ── Paper-ready table ──
    print(f"\n{'─'*70}")
    print(f"  PAPER-READY TABLE (copy to tracker)")
    print(f"{'─'*70}")
    print(f"\n| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |")
    print(f"|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|")
    for _, row in cmp_df.iterrows():
        short = row["model"].split("/")[-1].split("-Instruct")[0].split("-instruct")[0]
        print(f"| {short} | {row['country']} | {row['align_mis']:.4f} | "
              f"{row['align_jsd']:.4f} | {row['align_pearson']:+.3f} | "
              f"{row['align_mae']:.2f} | {row['flip_rate']:.1%} |")

    # ── Dimension-specific κ sensitivity analysis ──
    print(f"\n{'─'*70}")
    print(f"  DIMENSION-SPECIFIC κ SENSITIVITY (for paper table)")
    print(f"{'─'*70}")
    print(f"  | Dimension | κ_d | σ_scale | Rationale |")
    print(f"  |:----------|:---:|:-------:|:----------|")
    rationales = {
        "Species": "Anchor correct; high κ prevents over-correction",
        "Gender": "Moderate anchor; standard κ",
        "Age": "Moderate-poor anchor; lower κ allows exploration",
        "Fitness": "Moderate anchor; standard κ",
        "SocialValue": "Systematic -27pp bias; low κ enables correction",
        "Utilitarianism": "Country-dependent anchor; slightly lower κ",
    }
    for dim, params in sorted(DIMENSION_PT_PARAMS.items()):
        print(f"  | {dim:<18s} | {params['kappa']:.2f} | {params['sigma_scale']:.1f}x | "
              f"{rationales.get(dim, '')} |")

    print(f"\n[{EXP_ID}] DONE — results under {CMP_ROOT}")


if __name__ == "__main__":
    main()
