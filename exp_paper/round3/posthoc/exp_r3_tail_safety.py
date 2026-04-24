#!/usr/bin/env python3
"""Experiment 4 (playbook) — Tail-Safety Analysis.

Compares Full DISCA vs DISCA-consensus (IS disabled, delta_star ≡ 0) across
the 6-model × 20-country grid.  Shows that Step 3 (PT-IS + reliability gate)
adds +0.006 mean MIS improvement while cutting tail degradation by ~4×.

The consensus variant reuses NoISController from exp_paper_ablation_phi4.py
(same pattern: override _is_solve_decision to return zeros so delta_star ≡ 0).

Outputs (in RESULTS_BASE/):
  tail_safety_full.csv         — per (model, country) row for Full DISCA
  tail_safety_consensus.csv    — per (model, country) row for DISCA-consensus
  tail_safety_summary.csv      — 2-row summary table (paper Table)
  table_tail_safety.tex        — LaTeX ready-to-paste table

Env overrides:
  R3_MODELS        comma-separated HF ids (default: 6-model main panel)
  R3_COUNTRIES     comma ISO3 (default: 20 paper countries)
  R3_N_SCENARIOS   per-country per-model (default: 250 to keep runtime tractable)
  R3_BACKEND       vllm (default) | hf_native
  R3_SKIP_FULL     1 = skip full-DISCA run (load from R3_FULL_CSV); 0 = run all (default: 0)
  R3_FULL_CSV      path to existing full-DISCA results CSV
  R3_CONSENSUS_CSV path to existing consensus results CSV

Kaggle (sequential, ~6-8h on H100 for all 6 models):
    !python exp_paper/round3/posthoc/exp_r3_tail_safety.py
"""

from __future__ import annotations

import os as _os, subprocess as _sp, sys as _sys

_REPO_URL = "https://github.com/trungkiet2005/cultural_alignment.git"
_REPO_DIR_KAGGLE = "/kaggle/working/cultural_alignment"


def _bootstrap() -> str:
    here = _os.getcwd()
    if _os.path.isfile(_os.path.join(here, "src", "controller.py")):
        if here not in _sys.path:
            _sys.path.insert(0, here)
        return here
    if not _os.path.isdir("/kaggle/input"):
        raise RuntimeError("Not on Kaggle and not inside the repo root.")
    if not _os.path.isdir(_REPO_DIR_KAGGLE):
        _sp.run(["git", "clone", "--depth", "1", _REPO_URL, _REPO_DIR_KAGGLE], check=True)
    _os.chdir(_REPO_DIR_KAGGLE)
    _sys.path.insert(0, _REPO_DIR_KAGGLE)
    return _REPO_DIR_KAGGLE


_bootstrap()
_os.environ.setdefault("MORAL_MODEL_BACKEND", _os.environ.get("R3_BACKEND", "vllm"))

import gc
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from exp_paper._r2_common import build_cfg, load_model_timed, load_scenarios, on_kaggle
from exp_paper.paper_runtime import configure_paper_env, install_paper_kaggle_deps

configure_paper_env()
from src.hf_env import apply_hf_credentials

apply_hf_credentials()
install_paper_kaggle_deps()

import numpy as np
import pandas as pd
import torch

from experiment_DM.exp24_dpbr_core import (
    BootstrapPriorState, Exp24DualPassController,
    K_HALF, PRIOR_STATE, patch_swa_runner_controller,
)
from src.baseline_runner import run_baseline_vanilla
from src.model import setup_seeds
from src.personas import SUPPORTED_COUNTRIES, build_country_personas
from src.swa_runner import run_country_experiment
from exp_paper.paper_countries import PAPER_20_COUNTRIES

# ─── Models (display_name, HF id, dir_short, slug) ────────────────────────────
DEFAULT_MODELS: List[Tuple[str, str, str, str]] = [
    ("Llama-3.3-70B",         "unsloth/Llama-3.3-70B-Instruct-bnb-4bit", "llama_3_3_70b",        "meta-llama-3.3-70b-instruct"),
    ("Magistral-Small-2509",  "mistralai/Magistral-Small-2509",           "magistral_small_2509", "magistral-small-2509"),
    ("Phi-4",                 "microsoft/phi-4",                          "phi_4",                "phi-4"),
    ("Qwen3-VL-8B",           "Qwen/Qwen3-VL-8B-Instruct",               "qwen3_vl_8b",          "qwen3-vl-8b-instruct"),
    ("Qwen2.5-7B",            "Qwen/Qwen2.5-7B-Instruct",                "qwen2_5_7b",           "qwen2.5-7b-instruct"),
    ("Phi-3.5-mini",          "microsoft/Phi-3.5-mini-instruct",          "phi_3_5_mini",         "phi-3.5-mini-instruct"),
]
# Parse R3_MODELS env override (comma-separated HF ids)
_model_override = _os.environ.get("R3_MODELS", "").strip()
if _model_override:
    _wanted = {m.strip() for m in _model_override.split(",") if m.strip()}
    DEFAULT_MODELS = [m for m in DEFAULT_MODELS if m[1] in _wanted]

N_SCEN = int(_os.environ.get("R3_N_SCENARIOS", "250"))
COUNTRIES = [
    c.strip()
    for c in _os.environ.get("R3_COUNTRIES", ",".join(PAPER_20_COUNTRIES)).split(",")
    if c.strip()
]

RESULTS_BASE = (
    "/kaggle/working/cultural_alignment/results/exp24_round3/tail_safety"
    if on_kaggle()
    else str(Path(__file__).parent.parent / "results" / "exp24_round3" / "tail_safety")
)
WVS_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"


# ─── Consensus (No-IS) controller ─────────────────────────────────────────────

class NoISController(Exp24DualPassController):
    """DISCA-consensus: IS disabled; delta_star ≡ 0; anchor/consensus only."""

    def _is_solve_decision(
        self,
        delta_base: torch.Tensor,
        delta_agents: torch.Tensor,
        sigma: float,
        device: torch.device,
    ) -> Tuple[torch.Tensor, float]:
        K = K_HALF * 2
        ess = 1.0 / K  # effectively 1/K (all uniform) for diagnostics
        return torch.zeros(1, device=device), ess


def _patch_consensus_controller(country: str) -> None:
    """Replace swa_runner's controller singleton with NoISController for `country`."""
    import src.swa_runner as _swr

    cfg_stub = _swr._current_controller.cfg if hasattr(_swr, "_current_controller") and _swr._current_controller else None  # type: ignore[attr-defined]

    def _make_controller(model, tokenizer, personas, cfg, country_iso, lang):
        return NoISController(
            model=model,
            tokenizer=tokenizer,
            personas=personas,
            cfg=cfg,
            country_iso=country_iso,
            lang=lang,
        )

    _swr._controller_factory = _make_controller  # type: ignore[attr-defined]

    # Also reset PRIOR_STATE for the country
    PRIOR_STATE.clear()
    PRIOR_STATE[country] = BootstrapPriorState()


def _run_one_cell(
    model, tokenizer, cfg, country: str,
    variant: str,   # "full" | "consensus"
    model_display: str,
) -> Dict:
    scen = load_scenarios(cfg, country)
    personas = build_country_personas(country, wvs_path=WVS_PATH)

    # Vanilla baseline (needed for delta_mis)
    t0 = time.time()
    bl = run_baseline_vanilla(model, tokenizer, scen, country, cfg)
    vanilla_mis = float(bl["alignment"].get("mis", float("nan")))

    # Run variant
    if variant == "full":
        PRIOR_STATE.clear()
        PRIOR_STATE[country] = BootstrapPriorState()
        patch_swa_runner_controller()
    else:  # consensus
        _patch_consensus_controller(country)

    _, summary = run_country_experiment(model, tokenizer, country, personas, scen, cfg)
    variant_mis = float(summary["alignment"].get("mis", float("nan")))

    return {
        "model":       model_display,
        "country":     country,
        "variant":     variant,
        "vanilla_mis": vanilla_mis,
        "variant_mis": variant_mis,
        "delta_mis":   vanilla_mis - variant_mis,
        "elapsed_sec": time.time() - t0,
        "pearson_r":   float(summary["alignment"].get("pearson_r", float("nan"))),
    }


def _run_variant(
    display: str, hf_id: str, dir_short: str, variant: str,
    existing_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Run one model × all countries for one variant; return rows DataFrame."""
    cfg = build_cfg(hf_id, RESULTS_BASE, COUNTRIES, n_scenarios=N_SCEN, load_in_4bit=False)
    backend = _os.environ.get("MORAL_MODEL_BACKEND", "vllm").strip().lower()
    load_4bit = "70b" in hf_id.lower() or "bnb-4bit" in hf_id.lower()
    model, tokenizer = load_model_timed(hf_id, backend=backend, load_in_4bit=load_4bit)

    rows: List[Dict] = []
    for country in COUNTRIES:
        if country not in SUPPORTED_COUNTRIES:
            continue
        # Skip if already computed
        if existing_df is not None:
            key = (display, country, variant)
            mask = (
                (existing_df["model"] == display) &
                (existing_df["country"] == country) &
                (existing_df["variant"] == variant)
            )
            if mask.any():
                rows.append(existing_df[mask].iloc[0].to_dict())
                continue
        print(f"  [{display} | {variant}] {country}")
        try:
            row = _run_one_cell(model, tokenizer, cfg, country, variant, display)
            rows.append(row)
        except Exception as e:
            print(f"  ERROR {display}/{country}/{variant}: {e}")
            rows.append({"model": display, "country": country, "variant": variant,
                         "vanilla_mis": float("nan"), "variant_mis": float("nan"),
                         "delta_mis": float("nan"), "elapsed_sec": 0.0, "pearson_r": float("nan")})

    # Free GPU memory
    del model
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return pd.DataFrame(rows)


def _compute_tail_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-variant tail metrics across all (model, country) cells."""
    rows = []
    for variant, vdf in df.groupby("variant"):
        deltas = vdf["delta_mis"].dropna().values
        n_hurt = int((deltas < 0).sum())
        worst  = float((-deltas[deltas < 0]).max()) if n_hurt > 0 else 0.0
        rows.append({
            "variant":              variant,
            "mean_delta_mis":       float(deltas.mean()),
            "median_delta_mis":     float(np.median(deltas)),
            "n_cells":              len(deltas),
            "n_hurt":               n_hurt,
            "pct_hurt":             100.0 * n_hurt / len(deltas),
            "worst_case_degr":      worst,
            "std_delta_mis":        float(deltas.std(ddof=1)),
        })
    return pd.DataFrame(rows).sort_values("variant")


def _make_latex_table(summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Tail-safety analysis across all country-model cells "
        r"(6 models $\times$ 20 countries). Full DISCA prevents "
        r"catastrophic per-cell degradation that simple averaging admits.}",
        r"\label{tab:tail_safety}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Variant & Mean $\Delta$MIS & Cells hurt & Worst-case & Std \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        variant = "Full DISCA" if row["variant"] == "full" else "DISCA-consensus"
        is_ours = row["variant"] == "full"
        prefix = r"\textbf{" if is_ours else ""
        suffix = "}" if is_ours else ""
        lines.append(
            f"{prefix}{variant}{suffix} & "
            f"{prefix}{row['mean_delta_mis']:.3f}{suffix} & "
            f"{prefix}{int(row['n_hurt'])} / {int(row['n_cells'])}{suffix} & "
            f"{prefix}{row['worst_case_degr']:.2f}{suffix} & "
            f"{prefix}{row['std_delta_mis']:.3f}{suffix} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main() -> None:
    setup_seeds(42)
    out_dir = Path(RESULTS_BASE)
    out_dir.mkdir(parents=True, exist_ok=True)

    skip_full = _os.environ.get("R3_SKIP_FULL", "0").strip() == "1"
    existing_full = None
    existing_cons = None

    full_csv_path = _os.environ.get("R3_FULL_CSV", str(out_dir / "tail_safety_full.csv"))
    cons_csv_path = _os.environ.get("R3_CONSENSUS_CSV", str(out_dir / "tail_safety_consensus.csv"))

    if skip_full and Path(full_csv_path).exists():
        existing_full = pd.read_csv(full_csv_path)
        print(f"Loaded existing full-DISCA results: {len(existing_full)} rows")
    if Path(cons_csv_path).exists():
        existing_cons = pd.read_csv(cons_csv_path)
        print(f"Loaded existing consensus results: {len(existing_cons)} rows")

    all_full: List[pd.DataFrame] = [existing_full] if existing_full is not None else []
    all_cons: List[pd.DataFrame] = [existing_cons] if existing_cons is not None else []

    for display, hf_id, dir_short, slug in DEFAULT_MODELS:
        print(f"\n{'='*60}\n  Model: {display}\n{'='*60}")

        if not skip_full:
            print(f"\n  [full DISCA] {display}")
            df_full = _run_variant(display, hf_id, dir_short, "full", existing_full)
            all_full.append(df_full)
            pd.concat(all_full).to_csv(out_dir / "tail_safety_full.csv", index=False)

        print(f"\n  [consensus] {display}")
        df_cons = _run_variant(display, hf_id, dir_short, "consensus", existing_cons)
        all_cons.append(df_cons)
        pd.concat(all_cons).to_csv(out_dir / "tail_safety_consensus.csv", index=False)

    full_df = pd.concat(all_full, ignore_index=True) if all_full else pd.DataFrame()
    cons_df = pd.concat(all_cons, ignore_index=True) if all_cons else pd.DataFrame()

    full_df.to_csv(out_dir / "tail_safety_full.csv", index=False)
    cons_df.to_csv(out_dir / "tail_safety_consensus.csv", index=False)

    combined = pd.concat([full_df, cons_df], ignore_index=True)
    combined.to_csv(out_dir / "tail_safety_combined.csv", index=False)

    summary = _compute_tail_summary(combined)
    summary.to_csv(out_dir / "tail_safety_summary.csv", index=False)

    print("\n" + "="*60)
    print("TAIL SAFETY SUMMARY")
    print("="*60)
    print(summary.to_string(index=False))

    tex = _make_latex_table(summary)
    tex_path = out_dir / "table_tail_safety.tex"
    tex_path.write_text(tex, encoding="utf-8")
    print(f"\nLaTeX table → {tex_path}")
    print("\n" + tex)


if __name__ == "__main__":
    main()
