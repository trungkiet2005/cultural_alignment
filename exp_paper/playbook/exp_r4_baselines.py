#!/usr/bin/env python3
"""Experiment 5 (playbook) — Strong Baselines on the 20-country grid.

Runs five baselines + comparison vs DISCA on a single backbone (Phi-4 by
default) across the paper 20-country panel:
  * Vanilla                        (no calibration, no personas)
  * WVS Prompt (B2)                (textual WVS-grounded prompt; prompt-only)
  * MC-Dropout                     (uncertainty via dropout sampling)
  * Temp Scaling   [uses AMCE]     (per-country T fit on cal split)
  * DiffPO-binary  [uses AMCE]     (alpha calibration toward human AMCE)

The first three are "fair" baselines (no human AMCE leak). The last two
ARE oracle baselines that consume the human target — they bound what
any AMCE-informed method can do.

Outputs (in RESULTS_BASE/):
  baseline_country_mis.csv         — long-form (country, method, mis)
  baseline_summary.csv             — per-method aggregates (mean MIS, wins vs vanilla)
  detail/<country>_<method>.csv    — full per-scenario results

If R4_DISCA_CSV is set (or default file present), DISCA is included in the
final summary as a row labelled "DISCA (ours)".

Defends against:
  R3: "Baselines are weak. How does DISCA compare to oracle methods?"

Env overrides:
  R4_MODEL        HF id (default: microsoft/phi-4)
  R4_COUNTRIES    comma ISO3 list (default: PAPER_20_COUNTRIES)
  R4_N_SCENARIOS  per-country (default: 500)
  R4_BACKEND      vllm (default) | hf_native
  R4_DISCA_CSV    path to per-country DISCA results (cols: country, mis,
                  optionally vanilla_mis). If absent, summary omits DISCA row.

Kaggle:
    !python exp_paper/playbook/exp_r4_baselines.py
"""

from __future__ import annotations

import os as _os
import subprocess as _sp
import sys as _sys

_REPO_URL = "https://github.com/trungkiet2005/cultural_alignment.git"
_REPO_DIR_KAGGLE = "/kaggle/working/cultural_alignment"


def _bootstrap() -> str:
    here = _os.getcwd()
    if _os.path.isfile(_os.path.join(here, "src", "controller.py")):
        if here not in _sys.path:
            _sys.path.insert(0, here)
        return here
    if not _os.path.isdir("/kaggle/input"):
        raise RuntimeError("Not on Kaggle and not inside repo root.")
    if not _os.path.isdir(_REPO_DIR_KAGGLE):
        _sp.run(["git", "clone", "--depth", "1", _REPO_URL, _REPO_DIR_KAGGLE], check=True)
    _os.chdir(_REPO_DIR_KAGGLE)
    _sys.path.insert(0, _REPO_DIR_KAGGLE)
    return _REPO_DIR_KAGGLE


_bootstrap()
_os.environ.setdefault("MORAL_MODEL_BACKEND", _os.environ.get("R4_BACKEND", "vllm"))

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from exp_paper._r2_common import build_cfg, load_model_timed, load_scenarios, on_kaggle
from exp_paper.paper_countries import PAPER_20_COUNTRIES
from exp_paper.paper_runtime import configure_paper_env, install_paper_kaggle_deps

configure_paper_env()
from src.hf_env import apply_hf_credentials

apply_hf_credentials()
install_paper_kaggle_deps()

from src.baseline_runner import run_baseline_vanilla
from src.calibration_baselines import run_baseline_calibration_scaling
from src.diffpo_binary_baseline import run_baseline_diffpo_binary
from src.mc_dropout_runner import run_baseline_mc_dropout
from src.model import setup_seeds
from src.personas import SUPPORTED_COUNTRIES
from src.prompt_baseline_runner import run_prompt_baseline_country

MODEL_NAME = _os.environ.get("R4_MODEL", "microsoft/phi-4")
N_SCEN = int(_os.environ.get("R4_N_SCENARIOS", "500"))
COUNTRIES = (
    [c.strip() for c in _os.environ.get("R4_COUNTRIES", "").split(",") if c.strip()]
    or list(PAPER_20_COUNTRIES)
)

WVS_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
HUMAN_AMCE_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"
RESULTS_BASE = (
    "/kaggle/working/cultural_alignment/results/exp24_round4/baselines"
    if on_kaggle()
    else str(Path(__file__).parent.parent / "results" / "exp24_round4" / "baselines")
)
DISCA_CSV = _os.environ.get("R4_DISCA_CSV", "").strip()

METHODS = [
    "Vanilla",
    "WVS Prompt",
    "MC-Dropout",
    "Temp Scaling (uses AMCE)",
    "DiffPO-binary (uses AMCE)",
]


def _run_one(method: str, model, tokenizer, scenario_df, country, cfg) -> Tuple[float, pd.DataFrame]:
    if method == "Vanilla":
        out = run_baseline_vanilla(model, tokenizer, scenario_df, country, cfg)
    elif method == "WVS Prompt":
        out = run_prompt_baseline_country(
            model, tokenizer, scenario_df, country, cfg,
            baseline="B2",
            wvs_csv_path=WVS_PATH,
            human_amce_path=HUMAN_AMCE_PATH,
        )
    elif method == "MC-Dropout":
        out = run_baseline_mc_dropout(model, tokenizer, scenario_df, country, cfg)
    elif method == "Temp Scaling (uses AMCE)":
        out = run_baseline_calibration_scaling(
            model, tokenizer, scenario_df, country, cfg, method="temperature",
        )
    elif method == "DiffPO-binary (uses AMCE)":
        out = run_baseline_diffpo_binary(model, tokenizer, scenario_df, country, cfg)
    else:
        raise ValueError(f"Unknown method: {method}")

    align = out.get("alignment") or out.get("summary") or {}
    mis = float(align.get("mis", np.nan))
    res_df = out.get("results_df", pd.DataFrame()).copy()
    if "country" not in res_df.columns:
        res_df["country"] = country
    if "method" not in res_df.columns:
        res_df["method"] = method
    return mis, res_df


def main() -> None:
    setup_seeds(42)
    out_dir = Path(RESULTS_BASE)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_dir = out_dir / "detail"
    detail_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(MODEL_NAME, RESULTS_BASE, COUNTRIES,
                    n_scenarios=N_SCEN, load_in_4bit=False)
    backend = _os.environ.get("MORAL_MODEL_BACKEND", "vllm").strip().lower()
    model, tokenizer = load_model_timed(MODEL_NAME, backend=backend, load_in_4bit=False)

    rows: List[Dict] = []
    for country in COUNTRIES:
        if country not in SUPPORTED_COUNTRIES:
            print(f"[SKIP] {country} not in SUPPORTED_COUNTRIES")
            continue
        print(f"\n{'='*60}\n  COUNTRY = {country}\n{'='*60}")
        scen = load_scenarios(cfg, country)
        for method in METHODS:
            print(f"  -> {method}")
            try:
                mis, res_df = _run_one(method, model, tokenizer, scen, country, cfg)
            except Exception as exc:
                print(f"     [ERROR] {method} on {country}: {exc!r}")
                rows.append({"country": country, "method": method, "mis": float("nan")})
                continue
            rows.append({"country": country, "method": method, "mis": mis})
            tag = (method.lower().replace(" ", "_").replace("(", "")
                                .replace(")", "").replace("-", "_"))
            res_df.to_csv(detail_dir / f"{country}_{tag}.csv", index=False)
            # incremental
            pd.DataFrame(rows).to_csv(out_dir / "baseline_country_mis_partial.csv", index=False)

    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(out_dir / "baseline_country_mis.csv", index=False)
    print(f"\nSaved {len(raw_df)} rows -> {out_dir/'baseline_country_mis.csv'}")

    # Wins-vs-vanilla per method
    vanilla_map = {
        str(r["country"]): float(r["mis"])
        for _, r in raw_df[raw_df["method"] == "Vanilla"].iterrows()
        if np.isfinite(float(r["mis"]))
    }
    summary_rows: List[Dict] = []
    for method, mdf in raw_df.groupby("method"):
        wins = sum(
            1 for _, r in mdf.iterrows()
            if (str(r["country"]) in vanilla_map
                and np.isfinite(float(r["mis"]))
                and float(r["mis"]) < vanilla_map[str(r["country"])])
        )
        summary_rows.append({
            "method": method,
            "uses_amce": "Yes" if "uses AMCE" in method else "No",
            "mean_mis": float(mdf["mis"].dropna().mean()),
            "wins_vs_vanilla": int(wins),
            "n_countries": int(mdf["mis"].notna().sum()),
        })

    # Optionally fold in DISCA
    disca_path = Path(DISCA_CSV) if DISCA_CSV else (out_dir / "disca_results.csv")
    if disca_path.is_file():
        d = pd.read_csv(disca_path)
        mis_col = "disca_mis" if "disca_mis" in d.columns else "mis"
        if mis_col in d.columns:
            disca_mean = float(d[mis_col].mean())
            wins_disca = -1
            if "vanilla_mis" in d.columns:
                wins_disca = int((d[mis_col] < d["vanilla_mis"]).sum())
            elif vanilla_map:
                wins_disca = sum(
                    1 for _, r in d.iterrows()
                    if (str(r.get("country", "")) in vanilla_map
                        and float(r[mis_col]) < vanilla_map[str(r["country"])])
                )
            summary_rows.append({
                "method": "DISCA (ours)",
                "uses_amce": "No",
                "mean_mis": disca_mean,
                "wins_vs_vanilla": wins_disca,
                "n_countries": int(len(d)),
            })

    summary_df = pd.DataFrame(summary_rows).sort_values("mean_mis").reset_index(drop=True)
    summary_df.to_csv(out_dir / "baseline_summary.csv", index=False)

    print("\n" + "-" * 70)
    print("  Strong Baselines — RESULT  (sorted by mean MIS, lower = better)")
    print("-" * 70)
    disp = summary_df.copy()
    disp["mean_mis"] = disp["mean_mis"].round(4)
    print(disp.to_string(index=False))
    if "DISCA (ours)" in summary_df["method"].values:
        rank = int(summary_df.index[summary_df["method"] == "DISCA (ours)"][0]) + 1
        print(f"\n  DISCA rank: #{rank} of {len(summary_df)}")
    print(f"\n  saved -> {out_dir}")


if __name__ == "__main__":
    main()
