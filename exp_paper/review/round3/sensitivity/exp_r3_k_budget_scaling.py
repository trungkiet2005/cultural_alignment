#!/usr/bin/env python3
"""K-budget scaling curve for the dual-pass IS estimator.

Sweeps the per-pass IS sample budget ``K_half`` and measures macro MIS,
flip rate, and reliability-weight statistics on a 3-country Phi-4 panel.
The total IS budget per scenario is ``2 * K_half``; the released default
is ``K_half=64`` (total 128). This experiment answers the question
"is K=128 the right operating point?" with a curve showing where MIS
plateaus and where the variance of the reliability gate stabilises.

Kaggle (~1-1.5h on H100):
    !python exp_paper/review/round3/sensitivity/exp_r3_k_budget_scaling.py

Env overrides:
    R2_MODEL          HF id (default: microsoft/phi-4)
    R2_COUNTRIES      comma ISO3 list (default: USA,VNM,DEU)
    R2_N_SCENARIOS    per-country (default: 250)
    R2_K_GRID         comma list of K_half values (default: 8,16,32,64,128,192)
    R2_BACKEND        vllm (default) | hf_native
"""

from __future__ import annotations

import os as _os, subprocess as _sp, sys as _sys

_REPO_URL = "https://github.com/trungkiet2005/cultural_alignment.git"
_REPO_DIR_KAGGLE = "/kaggle/working/cultural_alignment"


def _r2_bootstrap() -> str:
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


_r2_bootstrap()
_os.environ.setdefault("MORAL_MODEL_BACKEND", _os.environ.get("R2_BACKEND", "vllm"))

import gc
import os
import time
from pathlib import Path
from typing import Dict, List

from exp_paper._r2_common import (
    build_cfg,
    load_model_timed,
    load_scenarios,
    on_kaggle,
    save_summary,
)
from exp_paper.paper_runtime import configure_paper_env, install_paper_kaggle_deps  # noqa: E402

configure_paper_env()
from src.hf_env import apply_hf_credentials  # noqa: E402

apply_hf_credentials()
install_paper_kaggle_deps()

import pandas as pd  # noqa: E402
import torch  # noqa: E402

# IMPORTANT: K_HALF is captured at import time from env; we reassign on the fly.
from experiment_DM import exp24_dpbr_core as _dpbr  # noqa: E402
from experiment_DM.exp24_dpbr_core import (  # noqa: E402
    BootstrapPriorState,
    PRIOR_STATE,
    patch_swa_runner_controller,
)
from src.model import setup_seeds  # noqa: E402
from src.personas import SUPPORTED_COUNTRIES, build_country_personas  # noqa: E402
from src.swa_runner import run_country_experiment  # noqa: E402

MODEL_NAME = os.environ.get("R2_MODEL", "microsoft/phi-4")
N_SCEN = int(os.environ.get("R2_N_SCENARIOS", "250"))
COUNTRIES = [c.strip() for c in os.environ.get("R2_COUNTRIES", "USA,VNM,DEU").split(",") if c.strip()]
K_GRID = [int(s.strip()) for s in os.environ.get("R2_K_GRID", "8,16,32,64,128,192").split(",") if s.strip()]

RESULTS_BASE = (
    "/kaggle/working/cultural_alignment/results/exp24_round3/k_budget_scaling"
    if on_kaggle()
    else str(Path(__file__).parent.parent / "results" / "exp24_round2" / "k_budget_scaling")
)
WVS_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"


def _reset_prior(country: str) -> None:
    PRIOR_STATE.clear()
    PRIOR_STATE[country] = BootstrapPriorState()


def _run_one_K(model, tokenizer, cfg, country, scen, K_half: int) -> Dict:
    """Run one (K_half, country) cell. Total IS budget = 2 * K_half."""
    snap = _dpbr.K_HALF
    _dpbr.K_HALF = int(K_half)
    _reset_prior(country)
    patch_swa_runner_controller()
    personas = build_country_personas(country, wvs_path=WVS_PATH)
    t0 = time.time()
    try:
        results_df, summary = run_country_experiment(
            model, tokenizer, country, personas, scen, cfg)
    finally:
        _dpbr.K_HALF = snap

    a = summary["alignment"]
    rel = (
        float(results_df["reliability_r"].mean())
        if "reliability_r" in results_df.columns else float("nan")
    )
    rel_std = (
        float(results_df["reliability_r"].std(ddof=1))
        if "reliability_r" in results_df.columns and len(results_df) >= 2
        else float("nan")
    )
    return {
        "K_half":               int(K_half),
        "K_total":              int(2 * K_half),
        "country":              country,
        "n_scenarios":          len(results_df),
        "elapsed_sec":          time.time() - t0,
        "mis":                  a.get("mis", float("nan")),
        "jsd":                  a.get("jsd", float("nan")),
        "pearson_r":            a.get("pearson_r", float("nan")),
        "flip_rate":            summary.get("flip_rate", float("nan")),
        "mean_reliability_r":   rel,
        "std_reliability_r":    rel_std,
    }


def main() -> None:
    setup_seeds(42)
    out_dir = Path(RESULTS_BASE)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(MODEL_NAME, RESULTS_BASE, COUNTRIES,
                    n_scenarios=N_SCEN, load_in_4bit=False)
    backend = os.environ.get("MORAL_MODEL_BACKEND", "vllm").strip().lower()
    model, tokenizer = load_model_timed(MODEL_NAME, backend=backend, load_in_4bit=False)

    scen_cache: Dict[str, pd.DataFrame] = {}
    for c in COUNTRIES:
        if c not in SUPPORTED_COUNTRIES:
            print(f"[skip] {c}")
            continue
        scen_cache[c] = load_scenarios(cfg, c)

    rows: List[Dict] = []
    for K_half in K_GRID:
        print(f"\n{'#' * 72}\n# K_half = {K_half}  (total IS budget = {2*K_half})\n{'#' * 72}")
        for country in COUNTRIES:
            if country not in scen_cache:
                continue
            try:
                row = _run_one_K(model, tokenizer, cfg, country, scen_cache[country], K_half)
                rows.append(row)
                pd.DataFrame(rows).to_csv(out_dir / "k_budget_partial.csv", index=False)
                print(f"  ✓ K_half={K_half:>4d} {country}  MIS={row['mis']:.4f}  "
                      f"flip={row['flip_rate']:.3f}  rel={row['mean_reliability_r']:.3f}  "
                      f"({row['elapsed_sec']:.0f}s)")
            except Exception as exc:
                print(f"[error] K_half={K_half} {country}: {exc}")
                rows.append({"K_half": K_half, "country": country,
                             "error": str(exc)[:500]})
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    save_summary(rows, out_dir, "k_budget_summary.csv")
    _build_curve_table(rows, out_dir)
    _zip_outputs(out_dir, "round3_sensitivity_k_budget_scaling")


def _build_curve_table(rows: List[Dict], out_dir: Path) -> None:
    """Macro across countries → one line per K_half."""
    df = pd.DataFrame([r for r in rows if "mis" in r and r.get("mis") == r.get("mis")])
    if df.empty:
        return
    macro = df.groupby("K_half").agg(
        n_countries=("country", "nunique"),
        mean_mis=("mis", "mean"),
        std_mis=("mis", "std"),
        mean_flip=("flip_rate", "mean"),
        mean_rel_r=("mean_reliability_r", "mean"),
        mean_elapsed=("elapsed_sec", "mean"),
    ).reset_index()
    macro["K_total"] = 2 * macro["K_half"]
    macro.to_csv(out_dir / "k_budget_macro.csv", index=False)
    print(f"\n[saved] {out_dir / 'k_budget_macro.csv'}")
    print(macro.round(4).to_string(index=False))

    # Paper-ready LaTeX table
    lines = [
        r"\begin{tabular}{rrcccc}\toprule",
        r"$K_\text{half}$ & $K_\text{total}$ & Macro MIS $\downarrow$ & Flip rate & "
        r"$\overline{r}$ (gate) & sec/scen \\\midrule",
    ]
    for _, row in macro.iterrows():
        lines.append(
            f"{int(row['K_half'])} & {int(row['K_total'])} & "
            f"{row['mean_mis']:.4f} & {row['mean_flip']:.3f} & "
            f"{row['mean_rel_r']:.3f} & {row['mean_elapsed']/250:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "k_budget_table.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {out_dir / 'k_budget_table.tex'}")


def _zip_outputs(out_dir: Path, label: str) -> None:
    import shutil
    dest_base = (
        Path("/kaggle/working")
        if os.path.isdir("/kaggle/input")
        else out_dir.parent.parent / "download"
    )
    dest_base.mkdir(parents=True, exist_ok=True)
    zip_path = shutil.make_archive(
        str(dest_base / label), "zip",
        root_dir=str(out_dir.parent),
        base_dir=out_dir.name,
    )
    print(f"[zip] {zip_path}")


if __name__ == "__main__":
    main()
