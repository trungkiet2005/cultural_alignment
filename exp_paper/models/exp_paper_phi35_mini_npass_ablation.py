#!/usr/bin/env python3
"""
Paper sweep — Phi-3.5-mini-instruct — N-pass DPBR ablation, 20 countries
========================================================================

Question: the paper uses dual-pass (N=2). What about N = 1, 3, 4, 5, 6, …?

This script answers that without paying 6× GPU cost. It runs the model **once**
across all 20 countries and caches per-scenario debiased logit gaps, then
replays the cache through arbitrary N-pass DPBR variants in pure numpy.

Generalised DPBR rule (matches paper exactly at N=2):
    bootstrap_var = 2 · Var_ddof1(δ*₁,…,δ*ₙ)        # = (a-b)² when N=2
    r             = exp(-bootstrap_var / VAR_SCALE)
    δ*            = r · mean(δ*₁,…,δ*ₙ)
At N=1: r = 1, δ* = δ*₁.

Offline Kaggle setup (no internet — repo + datasets pre-loaded as Kaggle inputs):
    PROJECT_DATASET_DIR = "/kaggle/input/notebooks/foundnotkiet/git-moral/cultural-alignment"
    PROJECT_DATASET_DIR_ALT = "/kaggle/input/notebooks/foundnotkiet/git-moral/cultural_alignment"
    MULTITP_DATA_PATH   = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
    WVS_DATA_PATH       = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
    HUMAN_AMCE_PATH     = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"
    WORK_DIR            = "/kaggle/working/cultural_alignment"

Run:
    !python exp_paper/models/exp_paper_phi35_mini_npass_ablation.py
"""

import os
import shutil
import sys

# ─── Offline Kaggle paths (do NOT git-clone; copy from /kaggle/input) ─────────
PROJECT_DATASET_DIR     = "/kaggle/input/notebooks/foundnotkiet/git-moral/cultural-alignment"
PROJECT_DATASET_DIR_ALT = "/kaggle/input/notebooks/foundnotkiet/git-moral/cultural_alignment"
MULTITP_DATA_PATH       = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH           = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
HUMAN_AMCE_PATH         = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"
WORK_DIR                = "/kaggle/working/cultural_alignment"


def _on_kaggle() -> bool:
    return os.path.isdir("/kaggle/working")


def _ensure_repo_offline() -> str:
    """Use a local repo checkout if already inside one; otherwise copy the
    pre-staged Kaggle dataset directory into /kaggle/working. Never hits the
    network — this script must run with internet disabled.
    """
    here = os.getcwd()
    if os.path.isfile(os.path.join(here, "src", "controller.py")):
        sys.path.insert(0, here)
        return here

    if not _on_kaggle():
        raise RuntimeError(
            "Not on Kaggle and not inside a repo root with src/controller.py. "
            "Set cwd to the repo root before running."
        )

    src_dir = next(
        (p for p in (PROJECT_DATASET_DIR, PROJECT_DATASET_DIR_ALT)
         if os.path.isdir(os.path.join(p, "src"))),
        None,
    )
    if src_dir is None:
        raise RuntimeError(
            "Repo not found in /kaggle/input. Expected one of:\n"
            f"  {PROJECT_DATASET_DIR}\n  {PROJECT_DATASET_DIR_ALT}\n"
            "Mount the Kaggle dataset that contains the cultural_alignment repo."
        )

    if not os.path.isdir(WORK_DIR):
        print(f"[OFFLINE] Copying repo: {src_dir} → {WORK_DIR}")
        shutil.copytree(src_dir, WORK_DIR)
    else:
        print(f"[OFFLINE] Reusing existing {WORK_DIR}")

    os.chdir(WORK_DIR)
    sys.path.insert(0, WORK_DIR)
    return WORK_DIR


_ensure_repo_offline()

# ─── Import paper runtime; skip pip install when offline ─────────────────────
from exp_paper.paper_runtime import configure_paper_env  # noqa: E402

configure_paper_env()

# Optional pip step — only when explicitly enabled (default: off, no internet).
if os.environ.get("EXP_OFFLINE_INSTALL_DEPS", "0").strip() == "1":
    from exp_paper.paper_runtime import install_paper_kaggle_deps  # noqa: E402
    install_paper_kaggle_deps()
else:
    print("[OFFLINE] skipping pip install (set EXP_OFFLINE_INSTALL_DEPS=1 to enable)")

from src.hf_env import apply_hf_credentials  # noqa: E402

apply_hf_credentials()

# ─── Stdlib + heavy deps (after sys.path is set) ─────────────────────────────
import gc                                                     # noqa: E402
import json                                                   # noqa: E402
import time                                                   # noqa: E402
from pathlib import Path                                      # noqa: E402
from typing import Dict, List, Sequence, Tuple                # noqa: E402

import numpy as np                                            # noqa: E402
import pandas as pd                                           # noqa: E402
import torch                                                  # noqa: E402
from tqdm.auto import tqdm                                    # noqa: E402

try:
    torch._dynamo.config.disable = True
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

from src.amce import (                                        # noqa: E402
    compute_alignment_metrics,
    compute_amce_from_preferences,
    compute_per_dimension_alignment,
    compute_utilitarianism_slope,
    load_human_amce,
)
from src.baseline_runner import run_baseline_vanilla          # noqa: E402
from src.config import SWAConfig, resolve_output_dir          # noqa: E402
from src.constants import COUNTRY_LANG                        # noqa: E402
from src.data import load_multitp_dataset                     # noqa: E402
from src.model import load_model, setup_seeds                 # noqa: E402
from src.personas import SUPPORTED_COUNTRIES, build_country_personas  # noqa: E402
from src.scenarios import generate_multitp_scenarios          # noqa: E402

from experiment_DM.exp24_dpbr_core import (                   # noqa: E402
    BootstrapPriorState,
    K_HALF,
    PRIOR_STATE,
    VAR_SCALE,
)
from experiment_DM.exp_npass_dpbr import (                    # noqa: E402
    LoggingDPBRController,
    cache_path_for,
    load_logit_cache,
    npass_bootstrap_var,
    replay_country_npass,
    save_logit_cache,
)
from exp_paper.paper_countries import (                       # noqa: E402
    PAPER_20_COUNTRIES,
    RESULTS_BASE_EXP24_20C,
)


# ─── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME   = "microsoft/Phi-3.5-mini-instruct"
MODEL_SHORT  = "phi35_mini"
EXP_BASE     = "EXP-24-NPASS"

N_SCENARIOS  = 500
BATCH_SIZE   = 1
SEED         = 42
LAMBDA_COOP  = 0.70

# Default ablation grid. Override with EXP24_NPASS_GRID="1,2,3,4,5,6,8,12".
N_PASSES_GRID: List[int] = [
    int(v) for v in os.environ.get("EXP24_NPASS_GRID", "1,2,3,4,5,6").split(",") if v.strip()
]


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _build_cfg(model_name: str, swa_root: str, target_countries: Sequence[str]) -> SWAConfig:
    return SWAConfig(
        model_name=model_name,
        n_scenarios=N_SCENARIOS,
        batch_size=BATCH_SIZE,
        target_countries=list(target_countries),
        load_in_4bit=True,
        use_real_data=True,
        multitp_data_path=MULTITP_DATA_PATH,
        wvs_data_path=WVS_DATA_PATH,
        human_amce_path=HUMAN_AMCE_PATH,
        output_dir=swa_root,
        lambda_coop=LAMBDA_COOP,
        K_samples=128,
    )


def _load_scen(cfg: SWAConfig, country: str) -> pd.DataFrame:
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


def _row_meta(idx: int, row: pd.Series, lang: str) -> Dict:
    return {
        "scenario_idx": int(idx),
        "Prompt": row.get("Prompt", row.get("prompt", "")),
        "phenomenon_category": row.get("phenomenon_category", "default"),
        "this_group_name": row.get("this_group_name", "Unknown"),
        "preferred_on_right": int(bool(row.get("preferred_on_right", 1))),
        "n_left": int(row.get("n_left", 1)),
        "n_right": int(row.get("n_right", 1)),
        "lang": lang,
    }


def _summarise_rows(results_df: pd.DataFrame, country: str, cfg: SWAConfig,
                    method: str, model_name: str) -> Dict:
    """Compute alignment metrics + per-dimension breakdown for a results_df."""
    model_amce = compute_amce_from_preferences(results_df)
    human_amce = load_human_amce(cfg.human_amce_path, country)
    alignment  = compute_alignment_metrics(model_amce, human_amce)
    per_dim    = compute_per_dimension_alignment(model_amce, human_amce)
    util       = compute_utilitarianism_slope(results_df)

    def _mea(c: str) -> float:
        return float(results_df[c].mean()) if c in results_df.columns and len(results_df) else float("nan")

    flip_rate = float(results_df["mppi_flipped"].mean()) if "mppi_flipped" in results_df.columns else float("nan")

    return {
        "model": model_name,
        "method": method,
        "country": country,
        **{f"align_{k}": v for k, v in alignment.items()},
        "flip_rate": flip_rate,
        "n_scenarios": len(results_df),
        "mean_reliability_r": _mea("reliability_r"),
        "mean_bootstrap_var": _mea("bootstrap_var"),
        "mean_ess_min": _mea("ess_min"),
        "mean_ess_anchor_alpha": _mea("ess_anchor_alpha"),
        "mean_positional_bias": _mea("positional_bias"),
        "utilitarianism_slope_hat": float(util.get("slope_hat", float("nan"))),
        "utilitarianism_slope_n": int(util.get("n_obs", 0) or 0),
        "per_dim_alignment": per_dim,
    }


# ─── PHASE 1: extraction (run model once, cache logits per country) ──────────
def run_extraction_for_country(
    model, tokenizer, country: str, personas: List[str],
    scen: pd.DataFrame, cfg: SWAConfig, out_dir: Path,
) -> Tuple[pd.DataFrame, Dict, pd.DataFrame, Dict]:
    """Run vanilla baseline + standard DPBR (N=2) with logit logging.

    Returns
    -------
    (vanilla_results_df, vanilla_summary, dpbr_results_df, dpbr_summary).
    Side effect: writes the per-scenario logit cache parquet next to ``out_dir``.
    """
    lang = COUNTRY_LANG.get(country, "en")
    scen = scen.copy()
    if "lang" not in scen.columns:
        scen["lang"] = lang

    # 1) Vanilla baseline (no personas, no IS).
    print(f"  [VANILLA] {country} …")
    bl = run_baseline_vanilla(model, tokenizer, scen, country, cfg)
    bl["results_df"].to_csv(out_dir / f"vanilla_results_{country}.csv", index=False)
    vanilla_summary = {
        "model": cfg.model_name, "method": "baseline_vanilla", "country": country,
        **{f"align_{k}": v for k, v in bl["alignment"].items()},
        "flip_rate": float("nan"), "n_scenarios": len(bl["results_df"]),
    }

    # 2) DPBR-with-logging (N=2, paper-canonical). Custom loop so we can attach
    #    per-scenario metadata to the controller's cache records.
    PRIOR_STATE.clear()
    PRIOR_STATE[country] = BootstrapPriorState()
    controller = LoggingDPBRController(
        model=model, tokenizer=tokenizer, personas=personas,
        lambda_coop=cfg.lambda_coop, alpha_ctl=cfg.alpha_ctl,
        K_samples=cfg.K_samples, noise_std=cfg.noise_std,
        temperature=cfg.temperature, logit_temperature=cfg.logit_temperature,
        category_logit_temperatures=cfg.category_logit_temperatures,
        pt_alpha=cfg.pt_alpha, pt_beta=cfg.pt_beta, pt_kappa=cfg.pt_kappa,
        decision_temperature=cfg.decision_temperature,
        assistant_lang=lang, country_iso=country,
    )

    rows: List[Dict] = []
    print(f"  [DPBR-N2 + cache] {country} …")
    for idx, row in tqdm(scen.iterrows(), total=len(scen), desc=f"DPBR/log [{country}]"):
        prompt = row.get("Prompt", row.get("prompt", ""))
        if not prompt:
            continue
        meta = _row_meta(int(idx), row, lang)
        controller.attach_meta(meta)
        t0 = time.time()
        pred = controller.predict(
            prompt,
            preferred_on_right=bool(meta["preferred_on_right"]),
            phenomenon_category=meta["phenomenon_category"],
            lang=lang,
        )
        latency = time.time() - t0
        rewards = np.asarray(pred["agent_rewards"], dtype=float)
        rows.append({
            "country": country, "scenario_idx": int(idx),
            "Prompt": prompt,
            "phenomenon_category": meta["phenomenon_category"],
            "this_group_name": meta["this_group_name"],
            "preferred_on_right": meta["preferred_on_right"],
            "n_left": meta["n_left"], "n_right": meta["n_right"],
            "p_left": float(pred["p_left"]),
            "p_right": float(pred["p_right"]),
            "p_spare_preferred": float(pred["p_spare_preferred"]),
            "delta_consensus": float(pred["delta_consensus"]),
            "delta_opt": float(pred["delta_opt"]),
            "delta_opt_micro": float(pred.get("delta_opt_micro", pred["delta_opt"])),
            "reliability_r": float(pred.get("reliability_r", float("nan"))),
            "bootstrap_var": float(pred.get("bootstrap_var", float("nan"))),
            "ess_pass1": float(pred.get("ess_pass1", float("nan"))),
            "ess_pass2": float(pred.get("ess_pass2", float("nan"))),
            "ess_anchor_alpha": float(pred.get("ess_anchor_alpha", float("nan"))),
            "positional_bias": float(pred.get("positional_bias", 0.0)),
            "logit_temp_used": float(pred["logit_temp_used"]),
            "mppi_flipped": bool(pred["mppi_flipped"]),
            "delta_z_norm": float(pred["delta_z_norm"]),
            "mppi_variance": float(pred["variance"]),
            "agent_reward_mean": float(rewards.mean()) if rewards.size else 0.0,
            "agent_reward_std": float(rewards.std()) if rewards.size else 0.0,
            "latency_ms": latency * 1000.0,
        })

    dpbr_results_df = pd.DataFrame(rows)
    dpbr_results_df.to_csv(out_dir / f"swa_results_{country}.csv", index=False)
    dpbr_summary = _summarise_rows(
        dpbr_results_df, country, cfg, method=f"{EXP_BASE}_dual_pass", model_name=cfg.model_name,
    )

    # 3) Persist the logit cache (one parquet per country).
    cache_df = pd.DataFrame(controller._cache_records)
    save_logit_cache(out_dir, country, cache_df)
    print(f"  [CACHE] saved {len(cache_df)} rows → {cache_path_for(out_dir, country)}")

    # Free the controller (model / tokenizer survive the call).
    del controller
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return bl["results_df"], vanilla_summary, dpbr_results_df, dpbr_summary


# ─── PHASE 2: replay (no model — pure numpy) ─────────────────────────────────
def run_replay_for_n(
    out_dir: Path, countries: Sequence[str], cfg: SWAConfig, n_passes: int,
) -> List[Dict]:
    """Replay each country's cached logits through ``n_passes`` IS passes."""
    rows: List[Dict] = []
    method = f"{EXP_BASE}_n{n_passes}"
    for ci, country in enumerate(countries):
        cache_df = load_logit_cache(out_dir, country)
        if cache_df is None or cache_df.empty:
            print(f"  [N={n_passes}] {country}: no cache, skipping")
            continue
        print(f"  [N={n_passes}] [{ci+1}/{len(countries)}] replay {country} ({len(cache_df)} scenarios)")
        results_df = replay_country_npass(
            cache_df, country_iso=country, n_passes=n_passes,
            K_per_pass=K_HALF, var_scale=VAR_SCALE,
            lambda_coop=cfg.lambda_coop,
            pt_alpha=cfg.pt_alpha, pt_beta=cfg.pt_beta, pt_kappa=cfg.pt_kappa,
            beta_temp=cfg.temperature, rho_eff=0.1,
            decision_temperature=cfg.decision_temperature,
            noise_std_floor=cfg.noise_std,
            seed=SEED,
        )
        results_df.to_csv(out_dir / f"npass{n_passes}_results_{country}.csv", index=False)
        summary = _summarise_rows(results_df, country, cfg, method=method, model_name=cfg.model_name)
        rows.append(summary)
    return rows


# ─── Main orchestrator ───────────────────────────────────────────────────────
def main() -> None:
    setup_seeds(SEED)
    countries = list(PAPER_20_COUNTRIES)

    rb       = f"{RESULTS_BASE_EXP24_20C}_npass_ablation"
    swa_root = f"{rb}/{MODEL_SHORT}/swa"
    cmp_root = f"{rb}/{MODEL_SHORT}/compare"
    for d in (swa_root, cmp_root):
        Path(d).mkdir(parents=True, exist_ok=True)

    cfg     = _build_cfg(MODEL_NAME, swa_root, target_countries=countries)
    out_dir = Path(swa_root) / resolve_output_dir("", MODEL_NAME).strip("/\\")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(out_dir)

    print(f"\n{'='*72}")
    print(f"  {EXP_BASE}: N-pass DPBR ablation [{MODEL_NAME}]")
    print(f"{'='*72}")
    print(f"[GRID] N values        = {N_PASSES_GRID}")
    print(f"[CONFIG] K_per_pass    = {K_HALF}")
    print(f"[CONFIG] VAR_SCALE     = {VAR_SCALE}")
    print(f"[CONFIG] λ_coop        = {LAMBDA_COOP}")
    print(f"[CONFIG] N_scenarios   = {N_SCENARIOS}  ×  {len(countries)} countries")
    print(f"[CONFIG] cache dir     = {out_dir}")
    print(f"[OFFLINE] MultiTP      = {MULTITP_DATA_PATH}")
    print(f"[OFFLINE] WVS          = {WVS_DATA_PATH}")
    print(f"[OFFLINE] human AMCE   = {HUMAN_AMCE_PATH}")

    # ── PHASE 1: load model once, sweep all countries ──────────────────────
    skip_extraction = all(cache_path_for(out_dir, c).is_file() for c in countries)
    extraction_rows: List[Dict] = []
    if skip_extraction and os.environ.get("EXP24_NPASS_FORCE_EXTRACT", "0") != "1":
        print("\n[PHASE-1] all per-country logit caches present → skipping extraction.")
        print("          (set EXP24_NPASS_FORCE_EXTRACT=1 to recompute)")
    else:
        print("\n[PHASE-1] extracting logits (one-time GPU sweep across 20 countries)")
        model, tokenizer = load_model(MODEL_NAME, max_seq_length=2048, load_in_4bit=True)
        try:
            for ci, country in enumerate(countries):
                if country not in SUPPORTED_COUNTRIES:
                    print(f"[SKIP] unsupported country: {country}")
                    continue
                print(f"\n[{ci+1}/{len(countries)}] {EXP_BASE} | {country}")
                scen     = _load_scen(cfg, country)
                personas = build_country_personas(country, wvs_path=WVS_DATA_PATH)
                _, vanilla_summary, _, dpbr_summary = run_extraction_for_country(
                    model, tokenizer, country, personas, scen, cfg, out_dir,
                )
                extraction_rows.extend([vanilla_summary, dpbr_summary])
                torch.cuda.empty_cache()
                gc.collect()
        finally:
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # Persist phase-1 summaries.
        if extraction_rows:
            pd.DataFrame([{k: v for k, v in r.items() if k != "per_dim_alignment"}
                          for r in extraction_rows]).to_csv(
                Path(cmp_root) / "phase1_vanilla_and_dpbr_n2.csv", index=False,
            )
        print(f"\n[PHASE-1] DONE — caches written under {out_dir}")

    # ── PHASE 2: replay each N from cached logits ──────────────────────────
    print("\n[PHASE-2] replaying cached logits for each N (no GPU required)")
    all_replay_rows: List[Dict] = []
    for n in N_PASSES_GRID:
        print(f"\n  ── N = {n} ──")
        all_replay_rows.extend(run_replay_for_n(out_dir, countries, cfg, n_passes=n))

    if not all_replay_rows:
        print("[PHASE-2] no replay rows produced — aborting")
        return

    # Strip per-dim dict before CSV (keep it for printout below).
    csv_rows = [{k: v for k, v in r.items() if k != "per_dim_alignment"} for r in all_replay_rows]
    replay_df = pd.DataFrame(csv_rows)
    replay_df.to_csv(Path(cmp_root) / "npass_ablation_per_country.csv", index=False)

    # Aggregate per-N summary across the 20 countries.
    agg = (replay_df
           .groupby("method")
           .agg(mean_mis=("align_mis", "mean"),
                mean_pearson_r=("align_pearson_r", "mean"),
                mean_jsd=("align_jsd", "mean"),
                mean_mae=("align_mae", "mean"),
                mean_flip_rate=("flip_rate", "mean"),
                mean_reliability_r=("mean_reliability_r", "mean"),
                mean_bootstrap_var=("mean_bootstrap_var", "mean"),
                n_countries=("country", "nunique"))
           .reset_index())
    # Sort by N (extracted from method name).
    agg["n_passes"] = agg["method"].str.extract(r"_n(\d+)").astype(int)
    agg = agg.sort_values("n_passes").reset_index(drop=True)
    agg.to_csv(Path(cmp_root) / "npass_ablation_summary.csv", index=False)

    # ── Final report ───────────────────────────────────────────────────────
    print(f"\n{'#'*72}\n# {EXP_BASE} FINAL REPORT — {MODEL_NAME}\n{'#'*72}")
    print(f"\nPer-country detail:  {Path(cmp_root) / 'npass_ablation_per_country.csv'}")
    print(f"Aggregated summary:  {Path(cmp_root) / 'npass_ablation_summary.csv'}")
    print(f"\n{'N':>3}  {'MIS':>8}  {'r':>8}  {'JSD':>8}  {'MAE':>8}  "
          f"{'flip%':>7}  {'rel_r':>8}  {'bvar':>9}")
    print(f"{'-'*3}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*9}")
    for _, r in agg.iterrows():
        print(f"{int(r['n_passes']):>3}  "
              f"{r['mean_mis']:>8.4f}  {r['mean_pearson_r']:>+8.3f}  "
              f"{r['mean_jsd']:>8.4f}  {r['mean_mae']:>8.2f}  "
              f"{r['mean_flip_rate']*100:>6.1f}%  "
              f"{r['mean_reliability_r']:>8.3f}  {r['mean_bootstrap_var']:>9.4f}")

    # Highlight best N by MIS (lower = better).
    best = agg.sort_values("mean_mis").iloc[0]
    print(f"\n[BEST] N={int(best['n_passes'])}: MIS={best['mean_mis']:.4f}, "
          f"r={best['mean_pearson_r']:+.3f} (averaged over {int(best['n_countries'])} countries)")

    print(f"\n[{EXP_BASE}] DONE — {cmp_root}")


if __name__ == "__main__":
    main()
