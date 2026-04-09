#!/usr/bin/env python3
"""
EXP-15: Online Persona Credibility Reweighting (OPCR)
======================================================

**Novelty** (grounded in paper + tracker insights):

**Paper §3.1**: SWA-PTIS computes anchor = (1/N) Σ δ_i — all N personas share
equal weight. This is an egalitarian average regardless of how well each
persona's predictions align with what the country actually prefers.

**Tracker diagnosis**:
  - EXP-09 shows: Pearson r for Mistral is ALWAYS NEGATIVE (all 5 countries).
    This means Mistral's persona pool systematically predicts the WRONG direction.
    Equal weighting means every "wrong" persona pulls the anchor the wrong way.
  - SocialValue gap (27pp): the WVS egalitarian personas persistently under-assign
    SocialValue. Within a country run we accumulate 310-500 scenario results,
    but the anchor weighting never learns that the SocialValue-aligned persona
    (P4 in EXP-03) should get upweighted for SocialValue scenarios.
  - Qwen JPN excels (MIS=0.2802) while Qwen DEU regresses in EXP-09: different
    personas have heterogeneous predictive quality per country.

**EXP-15 Innovation: Online Persona Credibility Reweighting (OPCR)**

After each scenario, we compute a credibility score c_i for each persona i:
    c_i_t = (1 - α_cred) · c_i_{t-1} + α_cred · agree_i_t

where agree_i_t measures how well persona i's direction δ_i aligns with the
MOST RECENT prediction outcome (captured by the sign of delta_opt):
    agree_i_t = exp(-|δ_i - delta_opt| / σ_cred)  ∈ (0, 1]

The credibility scores are then converted to weights via temperature-softmax:
    w_i = softmax(c_i / T_cred)

The anchor becomes a CREDIBILITY-WEIGHTED mean:
    anchor_cred = Σ w_i · δ_i

This is equivalent to online variational inference where the "posterior" over
persona reliability is updated with each new observation (scenario outcome).

**Mathematical grounding (extension of paper §3.2)**:

Let θ_i be the "calibration quality" of persona i for country c.
We model: δ_i ~ N(δ_true, σ_i²)  where σ_i is unknown.
Online estimate: σ̂_i² ≈ Var_rolling(δ_i - delta_opt)   (scenario-level error)
MLE weight: w_i ∝ 1/σ̂_i²   ≡ credibility score 1/c_i if c_i ≈ σ̂_i²

EXP-15 uses EMA on agreement rather than direct variance for stability.

**Category-conditioned credibility** (for SocialValue fix):
  - Maintain SEPARATE credibility scores c_i[category] per (persona, category)
  - SocialValue scenarios update only c_i["SocialValue"]
  - Result: utilitarian personas get high c_i["SocialValue"] after a few SV
    scenarios; WVS egalitarian personas get lower c_i["SocialValue"]
  - This is compatible with existing category routing (EXP-06b design)

**Per-scenario credibility update rule**:
    c_i[cat]_t+1 = (1-α_cred) · c_i[cat]_t + α_cred · agree_i(cat)_t

where:
    agree_i(cat)_t = exp(-|delta_i - delta_opt| / sigma_agree)

The weight matrix w_i[cat] = softmax(c_i[cat] / T_cred)
produces the credibility-weighted anchor for the next scenario in that category.

**Expected gains**:
  - Mistral: wrong personas eventually get c_i → 0; dominant persona becomes
    country-aligned over run → Pearson r should flip positive
  - SocialValue: utilitarian-adjacent personas get c↑ in SocialValue category
  - Qwen/Gemma: already-good persona just fine-tunes its advantage weighting
  - Overall MIS target: < 0.3800

**Design choices**:
  - Warmup: first N_WARMUP scenarios use uniform weights (= EXP-01 behaviour)
  - α_cred = 0.15: slower than EXP-09 β_ema (0.10) to avoid over-fitting early
  - T_cred = 0.5: moderate softmax sharpness (1.0 = uniform, ~0.0 = argmax)
  - Category bucket: uses scenario["phenomenon_category"] field (already in pipeline)

Usage on Kaggle
---------------
    !python experiment_DM/exp15_persona_credibility.py
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
    if not _on_kaggle():
        return
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
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

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
# Step 2: experiment configuration
# ============================================================================
EXP_ID   = "EXP-15"
EXP_NAME = "persona_credibility"

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

# ── OPCR hyperparameters ───────────────────────────────────────────────────
N_WARMUP      = 40      # scenarios before credibility weighting activates
ALPHA_CRED    = 0.15    # EMA decay for credibility update per scenario
T_CRED        = 0.50    # softmax temperature for weight sharpness
SIGMA_AGREE   = 0.30    # scale parameter for agreement function (= σ₀ floor)
USE_CATEGORY  = True    # if True, maintain per-category credibility scores

# PT parameters (paper canonical, fixed)
PT_ALPHA  = 0.88
PT_BETA   = 0.88
PT_KAPPA  = 2.25

SWA_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/swa"
CMP_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/compare"
MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/"
    "WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
)
HUMAN_AMCE_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"

# MultiTP categories
ALL_CATEGORIES = ["Species", "Gender", "Age", "Fitness", "SocialValue", "Utilitarianism", "default"]


# ============================================================================
# Step 3: Persona Credibility State
# ============================================================================
class PersonaCredibilityState:
    """
    Online credibility tracker for N personas.

    Maintains per-category EMA credibility scores and converts them to
    softmax weights for the credibility-weighted anchor computation.

    State variables:
        c[cat][i]  ∈ (0,1]  EMA credibility score for persona i in category cat
        step       ∈ ℤ≥0    number of scenarios processed (for warmup check)
    """

    def __init__(self, n_personas: int, alpha: float = ALPHA_CRED,
                 t_cred: float = T_CRED, n_warmup: int = N_WARMUP,
                 sigma_agree: float = SIGMA_AGREE, use_category: bool = USE_CATEGORY):
        self.n_personas   = n_personas
        self.alpha        = alpha
        self.t_cred       = t_cred
        self.n_warmup     = n_warmup
        self.sigma_agree  = sigma_agree
        self.use_category = use_category
        self.step         = 0

        # c[cat] = np.array of shape (n_personas,), initialised to 1/N (uniform)
        init = np.ones(n_personas) / n_personas
        if use_category:
            self.c: Dict[str, np.ndarray] = {cat: init.copy() for cat in ALL_CATEGORIES}
        else:
            self.c = {"default": init.copy()}

        # History for diagnostics
        self._weight_history: List[np.ndarray] = []

    def _agreement(self, delta_i: np.ndarray, delta_opt: float) -> np.ndarray:
        """
        Compute per-persona agreement with the current delta_opt outcome.
        agree_i = exp(-|delta_i - delta_opt| / SIGMA_AGREE)  ∈ (0, 1]
        """
        return np.exp(-np.abs(delta_i - delta_opt) / (self.sigma_agree + 1e-8))

    def update(self, delta_agents: np.ndarray, delta_opt: float,
               category: str = "default") -> None:
        """
        Update credibility scores for the given category after a scenario.

        delta_agents : shape (N,) — raw per-persona delta values for scenario
        delta_opt    : final IS prediction delta for this scenario
        category     : MultiTP category label (e.g. 'SocialValue')
        """
        if self.step < self.n_warmup:
            self.step += 1
            return  # no updates during warmup

        agree    = self._agreement(delta_agents, delta_opt)
        cat_key  = category if (self.use_category and category in self.c) else "default"
        old_c    = self.c[cat_key]
        new_c    = (1.0 - self.alpha) * old_c + self.alpha * agree
        self.c[cat_key] = np.clip(new_c, 1e-4, 1.0)
        self.step += 1
        self._weight_history.append(self.weights(cat_key))

    def weights(self, category: str = "default") -> np.ndarray:
        """
        Return softmax-normalised credibility weights for a given category.
        During warmup: uniform weights (= vanilla EXP-01 anchor).
        """
        if self.step <= self.n_warmup:
            return np.ones(self.n_personas) / self.n_personas
        cat_key = category if (self.use_category and category in self.c) else "default"
        logits  = self.c[cat_key] / self.t_cred   # shape (N,)
        logits  = logits - logits.max()            # numerical stability
        w       = np.exp(logits)
        return w / w.sum()

    @property
    def stats(self) -> Dict:
        default_w = self.weights("default")
        return {
            "step":             self.step,
            "entropy_default":  float(-np.sum(default_w * np.log(default_w + 1e-8))),
            "max_weight":       float(default_w.max()),
            "min_weight":       float(default_w.min()),
            "weight_vec":       default_w.tolist(),
        }


# Global credibility state: reset per (model, country)
_CRED_STATE: Dict[str, PersonaCredibilityState] = {}


# ============================================================================
# Step 4: OPCR Controller
# ============================================================================
class Exp15PersonaCredibilityController(ImplicitSWAController):
    """
    Online Persona Credibility Reweighting (OPCR).

    Key modification:
        anchor = Σ w_i(category) · δ_i
    where w_i are credibility-weighted softmax probabilities updated online.

    All other components (PT-IS, positional debiasing, ESS guard) are unchanged.
    """

    def __init__(self, *args, country: str = "UNKNOWN", n_personas: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.country   = country
        self.n_personas = n_personas

    def _get_cred_state(self) -> PersonaCredibilityState:
        key = self.country
        if key not in _CRED_STATE:
            _CRED_STATE[key] = PersonaCredibilityState(n_personas=self.n_personas)
        return _CRED_STATE[key]

    def _pt_value(self, x: torch.Tensor) -> torch.Tensor:
        a, b, k = PT_ALPHA, PT_BETA, PT_KAPPA
        return torch.where(x >= 0, x.abs().pow(a), -k * x.abs().pow(b))

    @torch.no_grad()
    def predict(self, user_query, preferred_on_right=True, phenomenon_category="default", lang="en"):
        # ── Step 1: Positional debiasing (unchanged from paper) ──
        db1, da1, logit_temp = self._extract_logit_gaps(user_query, phenomenon_category, lang)
        swapped_query, swap_changed = self._swap_positional_labels(user_query, lang)
        if swap_changed:
            db2, da2, _ = self._extract_logit_gaps(swapped_query, phenomenon_category, lang)
        else:
            db2, da2 = db1, da1

        delta_base   = (db1 - db2) / 2.0 if swap_changed else db1
        delta_agents = (da1 - da2) / 2.0 if swap_changed else da1  # (N,)

        # ── Step 2: OPCR — credibility-weighted anchor ──
        cred_state = self._get_cred_state()
        c_weights  = cred_state.weights(phenomenon_category)          # np (N,)
        c_weights_t = torch.tensor(c_weights, dtype=delta_agents.dtype, device=delta_agents.device)

        # Credibility-weighted anchor (vs uniform mean in paper)
        anchor_cred = (c_weights_t * delta_agents).sum()              # scalar tensor
        # Standard uniform anchor (for consensus utility, unchanged)
        anchor_unif = delta_agents.mean()

        # ── Step 3: IS proposal setup ──
        sigma = max(
            float(delta_agents.std(unbiased=True).item()) if delta_agents.numel() >= 2 else 0.0,
            self.noise_std
        )
        K, device = self.K, self.device

        # Sample around CREDIBILITY-WEIGHTED anchor (key innovation)
        eps         = torch.randn(K, device=device) * sigma
        delta_tilde = anchor_cred + eps                                # (K,)

        # ── Step 4: PT-IS (standard, uses credibility-weighted anchor as reference) ──
        dist_base_to_i = (delta_base - delta_agents).abs()                       # (N,)
        dist_cand_to_i = (delta_tilde.unsqueeze(1) - delta_agents.unsqueeze(0)).abs()  # (K, N)
        g_per_agent    = (dist_base_to_i.unsqueeze(0) - dist_cand_to_i) / sigma  # (K, N)
        v_per_agent    = self._pt_value(g_per_agent)                              # (K, N)

        # Apply credibility weights INSIDE the mean (key: not all personas equally valuable)
        # v_weighted = Σ w_i · v(g_i,k)   (credibility-weighted average of PT values)
        c_weights_k = c_weights_t.unsqueeze(0)                        # (1, N) for broadcasting
        mean_v      = (c_weights_k * v_per_agent).sum(dim=1)          # (K,)

        # Consensus utility uses uniform anchor (stability)
        g_cons = ((delta_base - anchor_unif).abs() - (delta_tilde - anchor_unif).abs()) / sigma
        v_cons = self._pt_value(g_cons)

        U = (1.0 - self.lambda_coop) * mean_v + self.lambda_coop * v_cons
        w = F.softmax(U / self.beta, dim=0)

        k_eff      = 1.0 / torch.sum(w * w).clamp_min(1e-12)
        delta_star = (torch.sum(w * eps)
                      if float(k_eff.item()) / K >= self.rho_eff
                      else torch.zeros((), device=device))

        delta_opt = float((anchor_cred + delta_star).item())

        # ── Step 5: Update credibility state ──
        da_np = delta_agents.float().cpu().numpy().astype(float)
        cred_state.update(da_np, delta_opt, category=phenomenon_category)
        cred_stats = cred_state.stats

        p_right = torch.sigmoid(
            torch.tensor(delta_opt / self.decision_temperature)
        ).item()
        p_pref  = p_right if preferred_on_right else 1.0 - p_right
        variance = float(delta_agents.var(unbiased=True).item()) if delta_agents.numel() > 1 else 0.0

        return {
            "p_right": p_right, "p_left": 1.0 - p_right, "p_spare_preferred": p_pref,
            "variance": variance, "sigma_used": sigma,
            "mppi_flipped": (float(anchor_cred.item()) > 0) != (delta_opt > 0),
            "delta_z_norm": abs(delta_opt - float(anchor_cred.item())),
            "delta_consensus": float(anchor_cred.item()), "delta_opt": delta_opt,
            # OPCR diagnostics
            "cred_entropy":  cred_stats["entropy_default"],
            "cred_max_w":    cred_stats["max_weight"],
            "cred_min_w":    cred_stats["min_weight"],
            "cred_step":     cred_stats["step"],
            "ess_ratio":     float(k_eff.item()) / K,
            "logit_temp_used": logit_temp,
            "n_personas":    delta_agents.numel(),
            "agent_decision_gaps": delta_agents.tolist(),
            "agent_rewards":       (delta_agents - delta_base).tolist(),
            "p_spare_preferred_pass1": p_pref,
            "p_spare_preferred_pass2": p_pref,
            "positional_bias": 0.0,
        }


_swa_runner_mod.ImplicitSWAController = Exp15PersonaCredibilityController


# ============================================================================
# Step 5: Runner
# ============================================================================
def _free_model_cache(model_name: str):
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
        lambda_coop=LAMBDA_COOP,
        K_samples=128,
    )


def _load_country_scenarios(cfg: SWAConfig, country: str):
    lang = COUNTRY_LANG.get(country, "en")
    if cfg.use_real_data:
        df = load_multitp_dataset(
            data_base_path=cfg.multitp_data_path, lang=lang,
            translator=cfg.multitp_translator, suffix=cfg.multitp_suffix,
            n_scenarios=cfg.n_scenarios,
        )
    else:
        df = generate_multitp_scenarios(cfg.n_scenarios, lang=lang)
    df = df.copy()
    df["lang"] = lang
    return df


def _run_swa_for_model(model, tokenizer, model_name: str) -> List[dict]:
    cfg     = _build_swa_config(model_name)
    out_dir = Path(SWA_ROOT) / resolve_output_dir("", model_name).strip("/\\")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(out_dir)
    print(f"\n{'#'*70}\n# {EXP_ID} [{model_name}] Online Persona Credibility Reweighting\n{'#'*70}")

    rows = []
    for ci, country in enumerate(cfg.target_countries):
        if country not in SUPPORTED_COUNTRIES:
            continue

        # Reset credibility state per country
        _CRED_STATE.clear()
        personas = build_country_personas(country, wvs_path=WVS_DATA_PATH)
        _CRED_STATE[country] = PersonaCredibilityState(n_personas=len(personas))
        print(f"\n[{ci+1}/{len(cfg.target_countries)}] {EXP_ID} {model_name} | {country} "
              f"(OPCR: N={len(personas)}, α={ALPHA_CRED}, T={T_CRED}, cat={'ON' if USE_CATEGORY else 'OFF'})")

        scen = _load_country_scenarios(cfg, country)

        # Inject country + n_personas at construction
        orig_init = Exp15PersonaCredibilityController.__init__
        def patched_init(self, *a, country=country, n_personas=len(personas), **kw):
            orig_init(self, *a, country=country, n_personas=n_personas, **kw)
        Exp15PersonaCredibilityController.__init__ = patched_init
        _swa_runner_mod.ImplicitSWAController = Exp15PersonaCredibilityController

        results_df, summary = run_country_experiment(model, tokenizer, country, personas, scen, cfg)
        Exp15PersonaCredibilityController.__init__ = orig_init

        results_df.to_csv(out_dir / f"swa_results_{country}.csv", index=False)
        append_rows_csv(
            str(Path(CMP_ROOT) / "per_dim_breakdown.csv"),
            flatten_per_dim_alignment(
                summary.get("per_dimension_alignment", {}),
                model=model_name,
                method=f"{EXP_ID}_persona_credibility",
                country=country,
            ),
        )

        cred_stats   = _CRED_STATE.get(country, PersonaCredibilityState(n_personas=4)).stats
        mean_ess     = (float(results_df["ess_ratio"].mean())
                        if "ess_ratio" in results_df.columns else float("nan"))
        mean_entropy = (float(results_df["cred_entropy"].mean())
                        if "cred_entropy" in results_df.columns else float("nan"))

        rows.append({
            "model":   model_name,
            "method":  f"{EXP_ID}_persona_credibility",
            "country": country,
            **{f"align_{k}": v for k, v in summary["alignment"].items()},
            "flip_rate":          summary["flip_rate"],
            "mean_latency_ms":    summary["mean_latency_ms"],
            "n_scenarios":        summary["n_scenarios"],
            "mean_ess_ratio":     mean_ess,
            "mean_cred_entropy":  mean_entropy,
            "final_max_weight":   cred_stats["max_weight"],
            "alpha_cred":         ALPHA_CRED,
            "t_cred":             T_CRED,
            "use_category":       USE_CATEGORY,
        })

        # ── Detailed per-dimension log ──
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
                  f"MAE={summary['alignment']['mae']:.2f}  Flip={summary['flip_rate']:.1%}")
            print(f"      ESS={mean_ess:.3f}  cred_entropy={mean_entropy:.3f}  "
                  f"max_w={cred_stats['max_weight']:.3f}  "
                  f"weights={[f'{w:.2f}' for w in cred_stats['weight_vec']]}")

        torch.cuda.empty_cache()
        gc.collect()

    return rows


def main():
    setup_seeds(SEED)
    for d in (SWA_ROOT, CMP_ROOT):
        Path(d).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  {EXP_ID}: {EXP_NAME.upper()} — Online Persona Credibility Reweighting")
    print(f"{'='*70}")
    print(f"[CONFIG] N_warmup={N_WARMUP}, α_cred={ALPHA_CRED}, T_cred={T_CRED}")
    print(f"[CONFIG] σ_agree={SIGMA_AGREE}, use_category={USE_CATEGORY}")
    print(f"[THEORY] anchor = Σ w_i(cat)·δ_i  where w_i ∝ exp(c_i/T_cred)")
    print(f"[THEORY] c_i ← (1-α)c_i + α·exp(-|δ_i - δ_opt|/σ_agree)  (online EMA)")
    print(f"[TARGET] Mean MIS < 0.3800 | Mistral Pearson r > 0 | Flip% < 12%")

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

    print(f"\n\n{'#'*70}")
    print(f"# {EXP_ID} FINAL REPORT — {EXP_NAME.upper()}")
    print(f"{'#'*70}")
    print_alignment_table(cmp_df, title=f"{EXP_ID} RESULTS — {EXP_NAME}")

    print(f"\n{'─'*70}")
    for model_name in MODELS:
        m_df = cmp_df[cmp_df["model"] == model_name]
        if m_df.empty:
            continue
        short     = model_name.split("/")[-1][:20]
        mis_mean  = m_df["align_mis"].mean()
        jsd_mean  = m_df["align_jsd"].mean()
        r_mean    = m_df["align_pearson_r"].mean()
        flip_mean = m_df["flip_rate"].mean()
        print(f"  {short:<20s}  MIS={mis_mean:.4f}  JSD={jsd_mean:.4f}  "
              f"r={r_mean:+.3f}  Flip={flip_mean:.1%}")

    overall_mis = cmp_df["align_mis"].mean()
    print(f"\n  OVERALL MEAN MIS = {overall_mis:.4f}  (EXP-09 SOTA: 0.3975)")

    ref = try_load_reference_comparison()
    if ref is not None:
        print_metric_comparison(
            ref, cmp_df,
            title=f"{EXP_ID} vs EXP-01 (reference) — MIS",
            spec=CompareSpec(
                metric_col="align_mis",
                ref_method="swa_ptis",
                cur_method=f"{EXP_ID}_persona_credibility",
            ),
        )

    # ── Paper-ready table ──
    print(f"\n{'─'*70}")
    print(f"  PAPER-READY TABLE (copy to tracker)")
    print(f"{'─'*70}")
    print(f"\n| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% | cred_entropy |")
    print(f"|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|:------------:|")
    for _, row in cmp_df.iterrows():
        short = row["model"].split("/")[-1].split("-Instruct")[0].split("-instruct")[0]
        print(f"| {short} | {row['country']} | {row['align_mis']:.4f} | "
              f"{row['align_jsd']:.4f} | {row['align_pearson_r']:+.3f} | "
              f"{row['align_mae']:.2f} | {row['flip_rate']:.1%} | "
              f"{row.get('mean_cred_entropy', float('nan')):.3f} |")

    from experiment_DM.exp_reporting import print_tracker_ready_report
    print_tracker_ready_report(
        cmp_df, exp_id=EXP_ID,
        per_dim_csv_path=str(Path(CMP_ROOT) / "per_dim_breakdown.csv"),
    )
    print(f"\n[{EXP_ID}] DONE — results under {CMP_ROOT}")


if __name__ == "__main__":
    main()
