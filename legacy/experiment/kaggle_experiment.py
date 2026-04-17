#!/usr/bin/env python3
"""
Kaggle experiment runner: SWA-PTIS (paper version) for cultural alignment.

What this script does on Kaggle
-------------------------------
1. Bootstraps the env: git clones the repo (if not already) and pip-installs
   Unsloth + bitsandbytes + a few small deps.
2. For each LLM in MODELS (Qwen 7B, Gemma 2 9B, Mistral 7B — all 4-bit):
     a. Loads it once via Unsloth in 4-bit quantisation.
     b. Runs the vanilla LLM baseline on TARGET_COUNTRIES.
     c. Runs SWA-PTIS (paper pipeline) on the same countries.
     d. Removes the model's HF cache so the next model has free disk.
3. Aggregates everything into a single comparison CSV under `CMP_ROOT/`:
     model x country  ->  baseline MIS  vs  SWA MIS, with delta and improv %.

The SWA-PTIS math (linear A<->B positional debias, persona-mean anchor,
K-sample importance sampling with Prospect-Theory utility + ESS gate) lives
in the `PaperSWAController` class below. Persona LLM I/O is inherited
unchanged from `src.controller.ImplicitSWAController`.

Usage on Kaggle
---------------
Edit the constants in "Step 3: experiment configuration" (MODELS,
TARGET_COUNTRIES, N_SCENARIOS, SKIP_BASELINE, ...) and then run:

Option A — upload only this single file and let it bootstrap itself:
    !python kaggle_experiment.py

Option B — clone the repo first, then run:
    !git clone https://github.com/trungkiet2005/cultural_alignment.git
    %cd cultural_alignment
    !python kaggle_experiment.py
"""

# ============================================================================
# Step 0: env vars MUST be set before any torch import
# ============================================================================
import os
import sys
import subprocess

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("UNSLOTH_DISABLE_AUTO_COMPILE", "1")


# ============================================================================
# Step 1: bootstrap (Kaggle only) — git clone + pip install
# ============================================================================
REPO_URL = "https://github.com/trungkiet2005/cultural_alignment.git"
REPO_DIR_KAGGLE = "/kaggle/working/cultural_alignment"


def _on_kaggle() -> bool:
    return os.path.isdir("/kaggle/working")


def _ensure_repo() -> str:
    """Make sure we are inside the cultural_alignment repo and return its path."""
    here = os.getcwd()
    if os.path.isfile(os.path.join(here, "src", "controller.py")):
        return here
    if not _on_kaggle():
        raise RuntimeError(
            "Not running on Kaggle and not inside the cultural_alignment repo. "
            "cd into the repo or run on Kaggle."
        )
    if not os.path.isdir(REPO_DIR_KAGGLE):
        print(f"[BOOTSTRAP] git clone {REPO_URL} -> {REPO_DIR_KAGGLE}")
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, REPO_DIR_KAGGLE],
            check=True,
        )
    else:
        print(f"[BOOTSTRAP] Repo already at {REPO_DIR_KAGGLE}")
    os.chdir(REPO_DIR_KAGGLE)
    if REPO_DIR_KAGGLE not in sys.path:
        sys.path.insert(0, REPO_DIR_KAGGLE)
    return REPO_DIR_KAGGLE


def _install_deps() -> None:
    if not _on_kaggle():
        return
    print("[BOOTSTRAP] Installing dependencies (Unsloth + bitsandbytes + ...)")
    cmds = [
        "pip install -q bitsandbytes scipy tqdm matplotlib seaborn",
        "pip install --upgrade --no-deps unsloth",
        "pip install -q unsloth_zoo",
        "pip install --quiet --no-deps --force-reinstall pyarrow",
        "pip install --quiet 'datasets>=3.4.1,<4.4.0'",
    ]
    for c in cmds:
        subprocess.run(c, shell=True, check=False)


_REPO_DIR = _ensure_repo()
_install_deps()


# ============================================================================
# Step 2: now safe to import torch and project code
# ============================================================================
import gc
import shutil
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

# Project imports (require _ensure_repo() to have run)
from src.config import (
    BaselineConfig, SWAConfig,
    resolve_output_dir,
)
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
# Step 3: experiment configuration  (EDIT HERE)
# ============================================================================

# 3 LLMs in 4-bit (Unsloth pre-quantised). Order = run order.
MODELS: List[str] = [
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "unsloth/gemma-2-9b-it-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
]

# 5 culturally distinct countries (Western Anglo / Confucian / East Asian /
# Western European / Latin American). All have WVS-7 personas + MultiTP data.
TARGET_COUNTRIES: List[str] = ["USA", "CHN", "JPN", "DEU", "BRA"]

# Lower this for fast iteration on the math; raise for paper-grade numbers.
N_SCENARIOS: int = 500

# Inference batch size for the baseline runner. 0 = auto-detect by free VRAM.
# Use 1 to be safe on small Kaggle GPUs / large models.
BATCH_SIZE: int = 1

# Output roots — three separate dirs so baseline / SWA methods / comparison
# CSVs don't stomp on each other. Each gets one subdir per model.
BASE_ROOT: str = "/kaggle/working/cultural_alignment/results/baseline"
SWA_ROOT:  str = "/kaggle/working/cultural_alignment/results/swa_mppi"
CMP_ROOT:  str = "/kaggle/working/cultural_alignment/results/compare"

# Set to True to skip the vanilla LLM baseline (saves time when iterating
# on math only).
SKIP_BASELINE: bool = False

# Set to True to use synthetic scenarios instead of the real MultiTP data.
USE_SYNTHETIC_DATA: bool = False

# RNG seed.
SEED: int = 42

# Kaggle dataset paths — change here if you uploaded the data elsewhere.
MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/"
    "WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
)
HUMAN_AMCE_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"
)


# ============================================================================
# Step 4: PaperSWAController — published SWA-PTIS pipeline (math layer)
# ============================================================================
# Inherits ALL persona LLM I/O from src.controller.ImplicitSWAController:
#     forward batched eval over base + N personas, A<->B label swap,
#     per-language A/B answer-token resolution. Those parts are not touched.
#
# Overrides predict() to apply the published math:
#     1. Linear positional debias            (db1 - db2) / 2
#     2. Adaptive sigma (floor noise_std)    std(delta_agents)
#     3. Persona-mean anchor                 delta_agents.mean()
#     4. K-sample importance sampling with Prospect-Theory utility
#        v(x) = x^alpha           if x >= 0
#             = -kappa * |x|^beta if x <  0
#     5. ESS gate — reject the IS update if k_eff/K < rho_eff
# ============================================================================
class PaperSWAController(ImplicitSWAController):
    """Drop-in replacement for ImplicitSWAController that runs the paper's
    SWA-PTIS update inside `predict()`. Patched into src.swa_runner once at
    import time so `run_country_experiment` picks it up automatically.
    """

    def _pt_value(self, x: torch.Tensor) -> torch.Tensor:
        """Prospect-Theory value function (loss-averse, concave on gains)."""
        a, b, k = self.pt_alpha, self.pt_beta, self.pt_kappa
        return torch.where(x >= 0,
                           x.abs().pow(a),
                           -k * x.abs().pow(b))

    @torch.no_grad()
    def predict(
        self,
        user_query: str,
        preferred_on_right: bool = True,
        phenomenon_category: str = "default",
        lang: str = "en",
    ) -> Dict:
        # ----- Persona LLM I/O (inherited, fixed) -----
        # Pass 1: original A/B ordering.
        db1, da1, logit_temp = self._extract_logit_gaps(
            user_query, phenomenon_category, lang)
        # Pass 2: A<->B label swap (only if the prompt actually has A/B literals).
        swapped_query, swap_changed = self._swap_positional_labels(
            user_query, lang)
        if swap_changed:
            db2, da2, _ = self._extract_logit_gaps(
                swapped_query, phenomenon_category, lang)
        else:
            db2, da2 = db1, da1

        # ----- 1. Linear positional debias -----
        if swap_changed:
            delta_base = (db1 - db2) / 2.0
            delta_agents = (da1 - da2) / 2.0
        else:
            delta_base = db1
            delta_agents = da1

        # ----- 2. Adaptive proposal sigma (floored at noise_std) -----
        if delta_agents.numel() < 2:
            sigma = self.noise_std
        else:
            std = float(delta_agents.std(unbiased=True).item())
            sigma = max(std, self.noise_std)

        # ----- 3. Persona-mean anchor -----
        anchor = delta_agents.mean()

        # ----- 4. Prospect-Theory + Importance Sampling -----
        K = self.K
        device = self.device
        eps = torch.randn(K, device=device) * sigma
        delta_tilde = anchor + eps                                     # (K,)

        dist_base_to_i = (delta_base - delta_agents).abs()             # (N,)
        dist_cand_to_i = (delta_tilde.unsqueeze(1)
                          - delta_agents.unsqueeze(0)).abs()           # (K, N)
        g_per_agent = (dist_base_to_i.unsqueeze(0) - dist_cand_to_i) / sigma

        v_per_agent = self._pt_value(g_per_agent)                      # (K, N)
        mean_v = v_per_agent.mean(dim=1)                               # (K,)

        g_cons = ((delta_base - anchor).abs()
                  - (delta_tilde - anchor).abs()) / sigma
        v_cons = self._pt_value(g_cons)

        U = (1.0 - self.lambda_coop) * mean_v + self.lambda_coop * v_cons
        w = F.softmax(U / self.beta, dim=0)

        # ----- 5. ESS gate -----
        k_eff = 1.0 / torch.sum(w * w).clamp_min(1e-12)
        if float(k_eff.item()) / float(K) < self.rho_eff:
            delta_star = torch.zeros((), device=device)
        else:
            delta_star = torch.sum(w * eps)

        delta_opt = anchor + delta_star
        variance = (float(delta_agents.var(unbiased=True).item())
                    if delta_agents.numel() > 1 else 0.0)

        # ----- Pack diagnostics in the format swa_runner expects -----
        delta_consensus_f = float(anchor.item())
        delta_opt_f = float(delta_opt.item())
        delta_star_f = float(delta_star.item())

        p_right = torch.sigmoid(delta_opt / self.decision_temperature).item()
        p_pref = p_right if preferred_on_right else 1.0 - p_right

        return {
            "p_right": p_right,
            "p_left": 1.0 - p_right,
            "p_spare_preferred": p_pref,
            "variance": variance,
            "sigma_used": float(sigma),
            "mppi_flipped": (delta_consensus_f > 0) != (delta_opt_f > 0),
            "delta_z_norm": abs(delta_star_f),
            "delta_consensus": delta_consensus_f,
            "delta_opt": delta_opt_f,
            "logit_temp_used": logit_temp,
            "agent_decision_gaps": delta_agents.tolist(),
            "agent_rewards": (delta_agents - delta_base).tolist(),
            "p_spare_preferred_pass1": p_pref,
            "p_spare_preferred_pass2": p_pref,
            "positional_bias": 0.0,
        }


# Patch the runner once so run_country_experiment uses our PaperSWAController.
_swa_runner_mod.ImplicitSWAController = PaperSWAController


def _dir_size_gb(path: str) -> float:
    total = 0
    for dirpath, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(dirpath, f)
            try:
                if not os.path.islink(fp):
                    total += os.path.getsize(fp)
            except OSError:
                pass
    return total / 1e9


def _free_model_cache(model_name: str) -> None:
    """Delete the HuggingFace hub cache directory for one model so the next
    model has free disk for its download.

    Only touches the HF hub cache (`<cache>/hub/models--<owner>--<name>/`).
    Does NOT touch /kaggle/working — your results / outputs are safe.
    """
    safe = "models--" + model_name.replace("/", "--")
    candidate_roots = [
        os.environ.get("HF_HUB_CACHE"),
        os.environ.get("HF_HOME"),
        os.path.expanduser("~/.cache/huggingface"),
        "/root/.cache/huggingface",
    ]
    seen = set()
    for root in candidate_roots:
        if not root:
            continue
        # HF_HUB_CACHE points directly at the hub dir; the others need /hub appended.
        hub_dir = root if os.path.basename(root.rstrip("/")) == "hub" \
            else os.path.join(root, "hub")
        target = os.path.join(hub_dir, safe)
        if target in seen:
            continue
        seen.add(target)
        if os.path.isdir(target):
            try:
                size_gb = _dir_size_gb(target)
                shutil.rmtree(target)
                print(f"[CLEANUP] removed {target}  ({size_gb:.2f} GB freed)")
            except Exception as e:
                print(f"[CLEANUP] failed to remove {target}: {e}")


# ============================================================================
# Step 6: per-model experiment driver (baseline + SWA)
# ============================================================================
def _build_swa_config(model_name: str) -> SWAConfig:
    return SWAConfig(
        model_name=model_name,
        n_scenarios=N_SCENARIOS,
        batch_size=BATCH_SIZE,
        target_countries=list(TARGET_COUNTRIES),
        load_in_4bit=True,
        use_real_data=not USE_SYNTHETIC_DATA,
        multitp_data_path=MULTITP_DATA_PATH,
        wvs_data_path=WVS_DATA_PATH,
        human_amce_path=HUMAN_AMCE_PATH,
        output_dir=SWA_ROOT,
    )


def _build_baseline_config(model_name: str) -> BaselineConfig:
    return BaselineConfig(
        model_name=model_name,
        n_scenarios=N_SCENARIOS,
        batch_size=BATCH_SIZE,
        target_countries=list(TARGET_COUNTRIES),
        load_in_4bit=True,
        use_real_data=not USE_SYNTHETIC_DATA,
        multitp_data_path=MULTITP_DATA_PATH,
        wvs_data_path=WVS_DATA_PATH,
        human_amce_path=HUMAN_AMCE_PATH,
        output_dir=BASE_ROOT,
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


def _run_baseline_for_model(
    model, tokenizer, model_name: str,
) -> List[dict]:
    cfg = _build_baseline_config(model_name)
    model_slug_dir = resolve_output_dir("", model_name).strip("/\\")
    out_dir = Path(BASE_ROOT) / model_slug_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(out_dir)
    print(f"\n{'#'*70}\n# BASELINE [{model_name}] -> {out_dir}\n{'#'*70}")

    rows = []
    for ci, country in enumerate(cfg.target_countries):
        if country not in SUPPORTED_COUNTRIES:
            print(f"[SKIP] {country} not in SUPPORTED_COUNTRIES")
            continue
        print(f"\n[{ci+1}/{len(cfg.target_countries)}] Baseline {model_name} | {country}")
        scen = _load_country_scenarios(cfg, country)
        bl = run_baseline_vanilla(model, tokenizer, scen, country, cfg)
        bl["results_df"].to_csv(out_dir / f"vanilla_results_{country}.csv", index=False)
        rows.append({
            "model":   model_name,
            "method":  "baseline_vanilla",
            "country": country,
            **{f"align_{k}": v for k, v in bl["alignment"].items()},
            "n_scenarios": len(bl["results_df"]),
        })
        torch.cuda.empty_cache()
        gc.collect()
    return rows


def _run_swa_for_model(
    model, tokenizer, model_name: str,
) -> List[dict]:
    """Run the paper SWA-PTIS pipeline on every TARGET_COUNTRY for one model."""
    cfg = _build_swa_config(model_name)
    model_slug_dir = resolve_output_dir("", model_name).strip("/\\")
    out_dir = Path(SWA_ROOT) / model_slug_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(out_dir)
    print(f"\n{'#'*70}\n# SWA-PTIS [{model_name}] -> {out_dir}\n{'#'*70}")

    rows = []
    for ci, country in enumerate(cfg.target_countries):
        if country not in SUPPORTED_COUNTRIES:
            print(f"[SKIP] {country} not in SUPPORTED_COUNTRIES")
            continue
        print(f"\n[{ci+1}/{len(cfg.target_countries)}] SWA {model_name} | {country}")
        scen = _load_country_scenarios(cfg, country)
        personas = build_country_personas(country, wvs_path=cfg.wvs_data_path)
        results_df, summary = run_country_experiment(
            model, tokenizer, country, personas, scen, cfg,
        )
        results_df.to_csv(out_dir / f"swa_results_{country}.csv", index=False)
        rows.append({
            "model":   model_name,
            "method":  "swa_ptis",
            "country": country,
            **{f"align_{k}": v for k, v in summary["alignment"].items()},
            "flip_rate":      summary["flip_rate"],
            "mean_latency_ms": summary["mean_latency_ms"],
            "n_scenarios":    summary["n_scenarios"],
        })
        torch.cuda.empty_cache()
        gc.collect()
    return rows


# ============================================================================
# Step 7: top-level main
# ============================================================================
def main() -> None:
    setup_seeds(SEED)
    for d in (BASE_ROOT, SWA_ROOT, CMP_ROOT):
        Path(d).mkdir(parents=True, exist_ok=True)

    print(f"\n[CONFIG] models   : {MODELS}")
    print(f"[CONFIG] countries: {TARGET_COUNTRIES}")
    print(f"[CONFIG] n_scenarios per country: {N_SCENARIOS}")
    print(f"[CONFIG] batch_size      : {BATCH_SIZE}")
    print(f"[CONFIG] base_root       : {BASE_ROOT}")
    print(f"[CONFIG] swa_root        : {SWA_ROOT}")
    print(f"[CONFIG] cmp_root        : {CMP_ROOT}")
    print(f"[CONFIG] skip_baseline   : {SKIP_BASELINE}")
    print(f"[CONFIG] use_synthetic   : {USE_SYNTHETIC_DATA}")

    all_rows: List[dict] = []

    for mi, model_name in enumerate(MODELS):
        print(f"\n{'='*70}")
        print(f"  MODEL {mi+1}/{len(MODELS)}: {model_name}")
        print(f"{'='*70}")
        model, tokenizer = load_model(
            model_name, max_seq_length=2048, load_in_4bit=True,
        )
        try:
            if not SKIP_BASELINE:
                all_rows.extend(_run_baseline_for_model(
                    model, tokenizer, model_name,
                ))
            all_rows.extend(_run_swa_for_model(
                model, tokenizer, model_name,
            ))
        finally:
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Free disk by removing this model's HF hub cache so the next
            # model has room to download. Does NOT touch /kaggle/working.
            _free_model_cache(model_name)

        # Incremental save so a mid-run crash doesn't lose finished rows.
        cmp_path = Path(CMP_ROOT) / "comparison.csv"
        pd.DataFrame(all_rows).to_csv(cmp_path, index=False)
        print(f"[SAVE] partial comparison -> {cmp_path}  ({len(all_rows)} rows)")

    # Final aggregated table
    cmp_path = Path(CMP_ROOT) / "comparison.csv"
    cmp_df = pd.DataFrame(all_rows)
    cmp_df.to_csv(cmp_path, index=False)
    print(f"\n[SAVE] FINAL comparison -> {cmp_path}  ({len(cmp_df)} rows)")

    # ----- MIS comparison: per-model baseline-vs-SWA tables -----
    _print_mis_comparison(cmp_df)

    print(f"\n[DONE] artefacts under:")
    print(f"  baseline -> {BASE_ROOT}")
    print(f"  swa      -> {SWA_ROOT}")
    print(f"  compare  -> {CMP_ROOT}")


def _print_mis_comparison(cmp_df: pd.DataFrame) -> None:
    """Print a per-country MIS improvement table in the paper's format,
    one block per model — baseline_vanilla vs swa_ptis.

    Only MIS is shown (paper's primary L2 metric, lower = better). The footer
    lists macro/micro averages and the SWA win count. A single CSV
    (`mis_comparison.csv`) is saved with every row for downstream analysis.
    """
    if cmp_df.empty or "method" not in cmp_df.columns:
        return
    if "align_mis" not in cmp_df.columns:
        print("\n[MIS] align_mis column not found — skipping MIS comparison.")
        return
    methods_present = set(cmp_df["method"])
    if "baseline_vanilla" not in methods_present or "swa_ptis" not in methods_present:
        print("\n[MIS] Need both baseline_vanilla and swa_ptis rows — "
              "skipping MIS comparison.")
        return

    out_rows: List[dict] = []
    width = 72

    for model_name in cmp_df["model"].drop_duplicates().tolist():
        mdf = cmp_df[cmp_df["model"] == model_name]
        baseline = (mdf[mdf["method"] == "baseline_vanilla"]
                    .drop_duplicates("country")
                    .set_index("country")["align_mis"])
        swa = (mdf[mdf["method"] == "swa_ptis"]
               .drop_duplicates("country")
               .set_index("country")["align_mis"])
        common = [c for c in baseline.index if c in swa.index]
        if not common:
            continue
        b = baseline.loc[common].astype(float)
        s = swa.loc[common].astype(float)

        print(f"\n{'='*width}")
        print(f"  MIS (PAPER-ALIGNED) IMPROVEMENT  —  L2 misalignment, lower=better")
        print(f"  MODEL: {model_name}")
        print(f"{'='*width}")
        print(f"   country   baseline        swa      delta     improv %")

        wins = 0
        for country in common:
            bv = float(b.loc[country])
            sv = float(s.loc[country])
            delta = bv - sv  # positive = SWA is better
            imp_pct = (100.0 * delta / bv) if bv != 0 else 0.0
            arrow = "↓" if delta > 0 else ("↑" if delta < 0 else "=")
            d_sign = "+" if delta >= 0 else ""
            i_sign = "+" if imp_pct >= 0 else ""
            print(f"   {country:>7}   {bv:8.4f}   {sv:8.4f}   "
                  f"{d_sign}{delta:7.4f}   {i_sign}{imp_pct:6.2f}% {arrow}")
            if delta > 0:
                wins += 1
            out_rows.append({
                "model":        model_name,
                "country":      country,
                "baseline_mis": bv,
                "swa_mis":      sv,
                "delta":        delta,
                "improve_pct":  imp_pct,
            })

        mean_b = float(b.mean())
        mean_s = float(s.mean())
        abs_red = mean_b - mean_s
        macro_pct = (100.0 * abs_red / mean_b) if mean_b != 0 else 0.0
        per_country_pct = (100.0 * (b - s) / b.replace(0, np.nan))
        micro_pct = float(per_country_pct.mean())
        n = len(common)

        d_sign = "+" if abs_red >= 0 else ""
        ma_sign = "+" if macro_pct >= 0 else ""
        mi_sign = "+" if micro_pct >= 0 else ""
        print(f"{'-'*width}")
        print(f"  Mean baseline MIS:            {mean_b:.4f}")
        print(f"  Mean SWA-PTIS MIS:            {mean_s:.4f}")
        print(f"  Absolute reduction:           {d_sign}{abs_red:.4f}")
        print(f"  Improvement on the means:     {ma_sign}{macro_pct:.2f}%   (macro-average)")
        print(f"  Mean per-country improvement: {mi_sign}{micro_pct:.2f}%   (micro-average)")
        print(f"  SWA wins (MIS lower):         {wins}/{n} countries")
        print(f"{'='*width}")

    if out_rows:
        out_path = Path(CMP_ROOT) / "mis_comparison.csv"
        pd.DataFrame(out_rows).to_csv(out_path, index=False)
        print(f"\n[SAVE] MIS comparison -> {out_path}  ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
