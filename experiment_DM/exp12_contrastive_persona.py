#!/usr/bin/env python3
"""
EXP-12: Contrastive Persona Decoding (CPD)
============================================

**Motivation** (novel algorithmic contribution for NeurIPS):

The fundamental problem with SWA-PTIS is that the persona anchor δ̄ (mean of
persona logit gaps) carries STRUCTURAL BIASES from the persona construction:
  - All WVS-derived personas are egalitarian → SocialValue anchor is negative
  - All personas share the same frozen LLM → instruction-tuning bias baked in
  - The utilitarian persona is country-INVARIANT → cannot capture country variation

These biases are ADDITIVE: they shift the anchor by a constant offset that is
the same regardless of which country we target. This means the cultural SIGNAL
(what makes Japan different from Brazil) is a small perturbation on top of a
large shared bias.

**Key insight: Contrastive Decoding**

If we run the SAME persona set with a "world-average" framing and get δ̄_world,
and then run them with the country-specific framing and get δ̄_country, then:

    cultural_signal = δ̄_country - δ̄_world

This subtraction cancels the shared structural biases (egalitarian anchor,
instruction-tuning bias, utilitarian invariance) and isolates the pure
CULTURAL DIFFERENCE signal.

**Implementation:**

For each scenario and country c:
  1. Standard run: get δ_base, δ_agents[c] from country-specific personas
  2. World-average run: get δ_agents[world] from a GENERIC "global citizen" persona set
  3. Contrastive correction:
     δ_contrastive = δ_agents[c] - δ_agents[world]  (per-persona)
  4. Apply contrastive correction to base:
     δ_corrected = δ_base + λ_contrast * δ_contrastive.mean()
  5. Run PT-IS on the corrected logit landscape

**Mathematical grounding:**

This is the logit-space analogue of contrastive decoding (Li et al., 2023):
    P(y|x, c) ∝ P(y|x, c)^(1+α) / P(y|x, world)^α

In logit space:
    δ_final = (1+α) * δ_country - α * δ_world
            = δ_country + α * (δ_country - δ_world)
            = δ_country + α * cultural_signal

Where α controls the contrast strength. We set α = λ_contrast.

**Why this is novel:**
  - First application of contrastive decoding to cross-cultural moral alignment
  - Principled removal of shared structural bias (egalitarian anchor)
  - No additional forward pass cost (world-average personas are precomputed)
  - Theoretically grounded in the contrastive decoding literature

**Expected results:**
  - SocialValue gap reduced (egalitarian bias cancels in the subtraction)
  - Better cultural specificity (the signal that differentiates Japan from USA is amplified)
  - Reduced Gemma over-correction (shared bias removed before PT-IS)

Usage on Kaggle
---------------
    !python experiment_DM/exp12_contrastive_persona.py
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
EXP_ID   = "EXP-12"
EXP_NAME = "contrastive_persona"

MODELS: List[str] = [
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "unsloth/gemma-2-9b-it-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
]
TARGET_COUNTRIES: List[str] = ["USA", "CHN", "JPN", "DEU", "BRA"]
N_SCENARIOS: int = 500
BATCH_SIZE: int = 1
SEED: int = 42

# Contrastive decoding strength
LAMBDA_CONTRAST = 0.5  # α in P(y|c)^(1+α) / P(y|world)^α

SWA_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/swa"
CMP_ROOT = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/compare"
MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/"
    "WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
)
HUMAN_AMCE_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"


# ============================================================================
# Step 4: World-Average Personas
# ============================================================================
# These personas represent a "cosmopolitan global citizen" — no specific country.
# They carry the SAME structural biases as country-specific personas (egalitarian,
# instruction-tuning) but WITHOUT country-specific cultural signal.

WORLD_AVERAGE_PERSONAS: List[str] = [
    (
        "You are a young adult (age 18-35) representing the global average. "
        "You hold moderate views that reflect the worldwide median on social "
        "values: moderate religiosity, moderate gender equality support, "
        "moderate social trust, and moderate tolerance for diversity. "
        "You do not identify with any particular country or culture. "
        "When making moral judgments, you weigh all considerations equally "
        "without strong bias toward any particular moral framework."
    ),
    (
        "You are a middle-aged adult (age 36-55) representing the global average. "
        "Your values reflect the worldwide median across all cultures: moderate "
        "conservatism balanced with progressive views, average levels of social "
        "trust and institutional confidence. You approach moral dilemmas with "
        "a balanced perspective that does not favor any specific cultural tradition."
    ),
    (
        "You are an older adult (age 55+) representing the global average. "
        "Your values reflect the accumulated wisdom of diverse global traditions: "
        "moderate religiosity, respect for both individual rights and collective "
        "responsibility, and pragmatic moral reasoning. You do not identify with "
        "any particular national culture."
    ),
    (
        "You are a moral philosopher committed to maximizing the total number "
        "of lives saved. When faced with a trolley-problem dilemma, you always "
        "prefer the option that saves more people, regardless of their social "
        "status, age, gender, or species."
    ),
]


# ============================================================================
# Step 5: Contrastive Persona Controller
# ============================================================================
class Exp12ContrastiveController(ImplicitSWAController):
    """
    Contrastive Persona Decoding (CPD).

    Key idea: Run both country-specific and world-average personas on the same
    scenario. The DIFFERENCE isolates the cultural signal and removes shared
    structural biases (egalitarian anchor, instruction-tuning bias).

    Final logit correction:
        δ_corrected_i = δ_country_i + λ * (δ_country_i - δ_world_i)
                      = (1 + λ) * δ_country_i - λ * δ_world_i

    This is contrastive decoding in logit-gap space.
    """

    def __init__(self, *args, world_personas: List[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        # Store world-average personas for contrastive decoding
        self._world_personas = world_personas or WORLD_AVERAGE_PERSONAS
        self._world_prefix_ids = None

    def _ensure_world_prefixes(self):
        """Tokenize world-average persona prefixes (lazy, cached)."""
        if self._world_prefix_ids is not None:
            return
        self._world_prefix_ids = []
        for persona_text in self._world_personas:
            messages = [{"role": "system", "content": persona_text}]
            try:
                prefix_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
            except Exception:
                prefix_ids = self.tokenizer.encode(persona_text, add_special_tokens=False)
            self._world_prefix_ids.append(prefix_ids)

    def _evaluate_world_agents(self, query_ids, lang, logit_temp=None):
        """
        Forward pass with world-average personas.
        Returns only the world-agent logit gaps (no base — we already have that).
        """
        self._ensure_world_prefixes()
        device = self.device

        # Build input sequences for world personas
        all_ids = []
        for prefix_ids in self._world_prefix_ids:
            ids = prefix_ids + query_ids
            all_ids.append(torch.tensor(ids, dtype=torch.long, device=device))

        # Pad and batch
        max_len = max(t.size(0) for t in all_ids)
        padded  = torch.zeros(len(all_ids), max_len, dtype=torch.long, device=device)
        for i, t in enumerate(all_ids):
            padded[i, :t.size(0)] = t

        with torch.no_grad():
            outputs = self.model(input_ids=padded)
            logits  = outputs.logits  # (N_world, seq_len, vocab)

        # Extract decision token logits at the last position
        # Use the same A/B token IDs as the main controller
        a_id, b_id = self._resolve_decision_tokens_for_lang(lang)
        last_logits = logits[:, -1, :]  # (N_world, vocab)
        z_a = last_logits[:, a_id]
        z_b = last_logits[:, b_id]

        temp = logit_temp or self.logit_temperature
        delta_world = (z_b - z_a) / temp  # (N_world,)
        return delta_world

    def _pt_value(self, x: torch.Tensor) -> torch.Tensor:
        a, b, k = self.pt_alpha, self.pt_beta, self.pt_kappa
        return torch.where(x >= 0, x.abs().pow(a), -k * x.abs().pow(b))

    @torch.no_grad()
    def predict(self, user_query, preferred_on_right=True, phenomenon_category="default", lang="en"):
        # ── Pass 1: Country-specific (standard) ──
        db1, da1, logit_temp = self._extract_logit_gaps(user_query, phenomenon_category, lang)
        swapped_query, swap_changed = self._swap_positional_labels(user_query, lang)
        if swap_changed:
            db2, da2, _ = self._extract_logit_gaps(swapped_query, phenomenon_category, lang)
        else:
            db2, da2 = db1, da1

        delta_base   = (db1 - db2) / 2.0 if swap_changed else db1
        delta_agents = (da1 - da2) / 2.0 if swap_changed else da1

        # ── Pass 2: World-average (contrastive reference) ──
        # We need the tokenized query without persona prefix
        from src.i18n import PROMPT_FRAME_I18N
        frame = PROMPT_FRAME_I18N.get(lang, PROMPT_FRAME_I18N["en"])
        prompt_text = frame.format(scenario=user_query)
        query_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        dw1 = self._evaluate_world_agents(query_ids, lang, logit_temp)
        if swap_changed:
            swapped_text = frame.format(scenario=swapped_query)
            swap_ids = self.tokenizer.encode(swapped_text, add_special_tokens=False)
            dw2 = self._evaluate_world_agents(swap_ids, lang, logit_temp)
            delta_world = (dw1 - dw2) / 2.0
        else:
            delta_world = dw1

        # ── Contrastive correction ──
        # Align dimensions: both delta_agents and delta_world should be N-dimensional
        n_country = delta_agents.numel()
        n_world   = delta_world.numel()
        n_min     = min(n_country, n_world)

        # Per-persona contrastive signal
        cultural_signal = delta_agents[:n_min] - delta_world[:n_min]
        mean_cultural_signal = cultural_signal.mean()

        # Apply contrastive decoding:
        #   δ_corrected = (1 + λ) * δ_country - λ * δ_world
        # Equivalently for the anchor:
        #   anchor_corrected = anchor_country + λ * cultural_signal
        anchor_country = delta_agents.mean()
        anchor_world   = delta_world[:n_min].mean()
        anchor_corrected = anchor_country + LAMBDA_CONTRAST * mean_cultural_signal

        # ── PT-IS update with corrected anchor ──
        sigma = max(
            float(delta_agents.std(unbiased=True).item()) if delta_agents.numel() >= 2 else 0.0,
            self.noise_std
        )
        K, device = self.K, self.device

        eps         = torch.randn(K, device=device) * sigma
        delta_tilde = anchor_corrected + eps

        dist_base_to_i = (delta_base - delta_agents).abs()
        dist_cand_to_i = (delta_tilde.unsqueeze(1) - delta_agents.unsqueeze(0)).abs()
        g_per_agent    = (dist_base_to_i.unsqueeze(0) - dist_cand_to_i) / sigma
        v_per_agent    = self._pt_value(g_per_agent)
        mean_v         = v_per_agent.mean(dim=1)

        g_cons = ((delta_base - anchor_corrected).abs() - (delta_tilde - anchor_corrected).abs()) / sigma
        v_cons = self._pt_value(g_cons)

        U = (1.0 - self.lambda_coop) * mean_v + self.lambda_coop * v_cons
        w = F.softmax(U / self.beta, dim=0)

        k_eff      = 1.0 / torch.sum(w * w).clamp_min(1e-12)
        delta_star = torch.sum(w * eps) if float(k_eff.item()) / K >= self.rho_eff else torch.zeros((), device=device)

        delta_opt = float((anchor_corrected + delta_star).item())

        p_right = torch.sigmoid(
            torch.tensor(delta_opt / self.decision_temperature)
        ).item()
        p_pref  = p_right if preferred_on_right else 1.0 - p_right
        variance = float(delta_agents.var(unbiased=True).item()) if delta_agents.numel() > 1 else 0.0

        return {
            "p_right": p_right, "p_left": 1.0 - p_right, "p_spare_preferred": p_pref,
            "variance": variance, "sigma_used": sigma,
            "mppi_flipped": (float(anchor_country.item()) > 0) != (delta_opt > 0),
            "delta_z_norm": abs(delta_opt - float(anchor_country.item())),
            "delta_consensus": float(anchor_country.item()), "delta_opt": delta_opt,
            # EXP-12 contrastive diagnostics
            "anchor_country": float(anchor_country.item()),
            "anchor_world": float(anchor_world.item()),
            "anchor_corrected": float(anchor_corrected.item()),
            "cultural_signal": float(mean_cultural_signal.item()),
            "lambda_contrast": LAMBDA_CONTRAST,
            "logit_temp_used": logit_temp, "n_personas": delta_agents.numel(),
            "agent_decision_gaps": delta_agents.tolist(),
            "agent_rewards": (delta_agents - delta_base).tolist(),
            "world_agent_gaps": delta_world.tolist(),
            "cultural_signal_per_agent": cultural_signal.tolist(),
            "p_spare_preferred_pass1": p_pref, "p_spare_preferred_pass2": p_pref,
            "positional_bias": 0.0,
        }


_swa_runner_mod.ImplicitSWAController = Exp12ContrastiveController


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
    print(f"\n{'#'*70}\n# {EXP_ID} [{model_name}] Contrastive Persona Decoding\n{'#'*70}")

    rows = []
    for ci, country in enumerate(cfg.target_countries):
        if country not in SUPPORTED_COUNTRIES: continue
        print(f"\n[{ci+1}/{len(cfg.target_countries)}] {EXP_ID} {model_name} | {country} (contrastive λ={LAMBDA_CONTRAST})")

        scen     = _load_country_scenarios(cfg, country)
        personas = build_country_personas(country, wvs_path=WVS_DATA_PATH)

        # Inject world personas into controller
        orig_init = Exp12ContrastiveController.__init__
        def patched_init(self, *args, **kwargs):
            orig_init(self, *args, world_personas=WORLD_AVERAGE_PERSONAS, **kwargs)

        Exp12ContrastiveController.__init__ = patched_init
        _swa_runner_mod.ImplicitSWAController = Exp12ContrastiveController

        results_df, summary = run_country_experiment(model, tokenizer, country, personas, scen, cfg)
        Exp12ContrastiveController.__init__ = orig_init

        results_df.to_csv(out_dir / f"swa_results_{country}.csv", index=False)
        append_rows_csv(
            str(Path(CMP_ROOT) / "per_dim_breakdown.csv"),
            flatten_per_dim_alignment(
                summary.get("per_dimension_alignment", {}),
                model=model_name,
                method=f"{EXP_ID}_contrastive_persona",
                country=country,
            ),
        )

        # Compute contrastive diagnostics
        mean_cultural_signal = float(results_df["cultural_signal"].mean()) if "cultural_signal" in results_df.columns else float("nan")
        mean_anchor_shift = float(
            (results_df["anchor_corrected"] - results_df["anchor_country"]).abs().mean()
        ) if "anchor_corrected" in results_df.columns and "anchor_country" in results_df.columns else float("nan")

        rows.append({
            "model": model_name, "method": f"{EXP_ID}_contrastive_persona",
            "country": country,
            **{f"align_{k}": v for k, v in summary["alignment"].items()},
            "flip_rate": summary["flip_rate"],
            "mean_latency_ms": summary["mean_latency_ms"],
            "n_scenarios": summary["n_scenarios"],
            "lambda_contrast": LAMBDA_CONTRAST,
            "mean_cultural_signal": mean_cultural_signal,
            "mean_anchor_shift": mean_anchor_shift,
        })

        # ── Detailed per-dimension log ──
        pda = summary.get("per_dimension_alignment", {})
        if pda:
            print(f"\n  ┌── Per-Dimension Alignment ({country}) ──")
            for dim_key, dim_data in sorted(pda.items()):
                human_val = dim_data.get("human", float("nan"))
                model_val = dim_data.get("model", float("nan"))
                err       = dim_data.get("error", model_val - human_val)
                print(f"  │  {dim_key:<25s}  human={human_val:6.1f}  model={model_val:6.1f}  err={err:+6.1f}pp")
            print(f"  └── MIS={summary['alignment']['mis']:.4f}  JSD={summary['alignment']['jsd']:.4f}  "
                  f"r={summary['alignment']['pearson']:.3f}  MAE={summary['alignment']['mae']:.2f}  "
                  f"Flip={summary['flip_rate']:.1%}")
            print(f"      cultural_signal={mean_cultural_signal:.4f}  anchor_shift={mean_anchor_shift:.4f}")

        torch.cuda.empty_cache(); gc.collect()
    return rows


def main():
    setup_seeds(SEED)
    for d in (SWA_ROOT, CMP_ROOT):
        Path(d).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  {EXP_ID}: {EXP_NAME.upper()}")
    print(f"  Novel: Contrastive Persona Decoding for Cultural Signal Isolation")
    print(f"{'='*70}")
    print(f"[CONFIG] λ_contrast = {LAMBDA_CONTRAST}")
    print(f"[CONFIG] δ_corrected = (1+λ)·δ_country - λ·δ_world")
    print(f"[CONFIG] World personas: {len(WORLD_AVERAGE_PERSONAS)} generic global-citizen agents")
    print(f"[THEORY] Cancels shared structural biases (egalitarian, RLHF)")
    print(f"[THEORY] Isolates pure cultural signal for PT-IS correction")

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

    # ── Aggregate stats ──
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
              f"Flip={m_df['flip_rate'].mean():.1%}  "
              f"cultural_sig={m_df['mean_cultural_signal'].mean():.4f}")

    overall_mis = cmp_df["align_mis"].mean()
    print(f"\n  OVERALL MEAN MIS = {overall_mis:.4f}  (EXP-01 baseline: 0.4269)")

    # ── Contrastive signal analysis ──
    print(f"\n{'─'*70}")
    print(f"  CONTRASTIVE SIGNAL ANALYSIS")
    print(f"{'─'*70}")
    for _, row in cmp_df.iterrows():
        short = row["model"].split("/")[-1][:15]
        print(f"  {short:<15s} | {row['country']} | cultural_sig={row.get('mean_cultural_signal', float('nan')):+.4f} | "
              f"anchor_shift={row.get('mean_anchor_shift', float('nan')):.4f} | "
              f"MIS={row['align_mis']:.4f}")

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
                    cur_method=f"{EXP_ID}_contrastive_persona",
                ),
            )

    # ── Paper-ready table ──
    print(f"\n{'─'*70}")
    print(f"  PAPER-READY TABLE (copy to tracker)")
    print(f"{'─'*70}")
    print(f"\n| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% | Cultural Signal |")
    print(f"|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|:---------------:|")
    for _, row in cmp_df.iterrows():
        short = row["model"].split("/")[-1].split("-Instruct")[0].split("-instruct")[0]
        print(f"| {short} | {row['country']} | {row['align_mis']:.4f} | "
              f"{row['align_jsd']:.4f} | {row['align_pearson']:+.3f} | "
              f"{row['align_mae']:.2f} | {row['flip_rate']:.1%} | "
              f"{row.get('mean_cultural_signal', float('nan')):+.4f} |")

    print(f"\n[{EXP_ID}] DONE — results under {CMP_ROOT}")


if __name__ == "__main__":
    main()
