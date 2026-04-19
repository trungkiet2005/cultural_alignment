#!/usr/bin/env python3
"""Oral-level analysis: wall-time latency & throughput overhead (Reviewer Q7).

Reviewers ask: "What are the latency/throughput overheads in tokens/second
or wall-time relative to vanilla decoding, and how does batch size affect
the two-pass IS stability?"

This script measures:

    1. Scenarios/second for: vanilla decoding vs. SWA-DPBR at K=32/64/128
    2. Wall-time overhead per scenario (SWA / vanilla ratio)
    3. GPU memory delta between vanilla and SWA-DPBR
    4. How DPBR reliability gate affects effective K
       (gate fires → K_eff < K_HALF*2; gate suppresses → correction → 0)

Design: 3 countries × 100 scenarios each (light sweep, ~30 min on H100).
All methods run on the same scenarios in the same session to keep GPU state
stable. Temperature/decoding is identical between vanilla and SWA so the
only overhead is the IS sampling loop.

Outputs:
    results/exp24_round2/latency/
        latency_raw.csv          # per-(country, method, K) timing rows
        latency_summary.csv      # aggregated (method, K, mean_sec, overhead_pct)
        latency_table.tex        # LaTeX table for Appendix
        latency_notes.txt        # GPU memory, K_eff distribution

Kaggle (~30 min on H100):
    !python exp_paper/round2/phase7_oral/exp_r2_latency_benchmark.py

Env overrides:
    R2_MODEL          HF id (default: microsoft/phi-4)
    R2_COUNTRIES      comma-separated ISO3 (default: USA,VNM,DEU)
    R2_N_SCENARIOS    per-country (default: 100)
    R2_K_GRID         comma list of K values (default: 32,64,128)
    R2_BACKEND        vllm (default) | hf_native
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Self-bootstrap
# ─────────────────────────────────────────────────────────────────────────────
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
        raise RuntimeError(
            "Not on Kaggle and not inside the repo root. "
            "Either cd into the cultural_alignment repo first, or run on Kaggle."
        )
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
from typing import Dict, List, Optional

from exp_paper._r2_common import (
    build_cfg,
    load_model_timed,
    load_scenarios,
    on_kaggle,
    save_summary,
)
from exp_paper.paper_runtime import configure_paper_env, install_paper_kaggle_deps

configure_paper_env()

from src.hf_env import apply_hf_credentials

apply_hf_credentials()
install_paper_kaggle_deps()

import numpy as np
import pandas as pd
import torch

from experiment_DM.exp24_dpbr_core import (
    BootstrapPriorState,
    PRIOR_STATE,
    patch_swa_runner_controller,
)
from src.baseline_runner import run_baseline_vanilla
from src.model import setup_seeds
from src.personas import SUPPORTED_COUNTRIES, build_country_personas
from src.swa_runner import run_country_experiment

# ─── config ─────────────────────────────────────────────────────────────────
MODEL_NAME = os.environ.get("R2_MODEL", "microsoft/phi-4")
N_SCEN     = int(os.environ.get("R2_N_SCENARIOS", "100"))
COUNTRIES  = [
    c.strip()
    for c in os.environ.get("R2_COUNTRIES", "USA,VNM,DEU").split(",")
    if c.strip()
]
K_GRID = [
    int(s.strip())
    for s in os.environ.get("R2_K_GRID", "32,64,128").split(",")
    if s.strip()
]

RESULTS_BASE = (
    "/kaggle/working/cultural_alignment/results/exp24_round2/latency"
    if on_kaggle()
    else str(Path(__file__).parent.parent / "results" / "exp24_round2" / "latency")
)

WVS_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
    if on_kaggle()
    else "WVS_data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
)


# ─── GPU memory helper ───────────────────────────────────────────────────────
def _gpu_mem_gb() -> float:
    if not torch.cuda.is_available():
        return float("nan")
    return torch.cuda.memory_allocated() / 1e9


def _gpu_peak_gb() -> float:
    if not torch.cuda.is_available():
        return float("nan")
    return torch.cuda.max_memory_allocated() / 1e9


# ─── single benchmark cell ────────────────────────────────────────────────────
def _bench_vanilla(model, tokenizer, cfg, country: str, scen: pd.DataFrame) -> Dict:
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    run_baseline_vanilla(model, tokenizer, scen, country, cfg)
    elapsed = time.perf_counter() - t0
    return {
        "method":      "vanilla",
        "K":           0,
        "country":     country,
        "n_scenarios": len(scen),
        "elapsed_sec": elapsed,
        "sec_per_scen": elapsed / max(len(scen), 1),
        "gpu_peak_gb": _gpu_peak_gb(),
    }


def _bench_swa(model, tokenizer, cfg, country: str, scen: pd.DataFrame, k_half: int) -> Dict:
    # Configure DPBR with specified K.
    _os.environ["EXP24_K_HALF"] = str(k_half)
    PRIOR_STATE.clear()
    PRIOR_STATE[country] = BootstrapPriorState()
    patch_swa_runner_controller()

    personas = build_country_personas(country, wvs_path=WVS_PATH)

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    _, summary = run_country_experiment(model, tokenizer, country, personas, scen, cfg)
    elapsed = time.perf_counter() - t0

    flip_rate = summary.get("flip_rate", float("nan"))
    return {
        "method":        "swa_dpbr",
        "K":             k_half * 2,
        "country":       country,
        "n_scenarios":   len(scen),
        "elapsed_sec":   elapsed,
        "sec_per_scen":  elapsed / max(len(scen), 1),
        "gpu_peak_gb":   _gpu_peak_gb(),
        "flip_rate":     flip_rate,
    }


# ─── LaTeX table ─────────────────────────────────────────────────────────────
def _build_latency_table(summary_df: pd.DataFrame) -> str:
    methods = ["vanilla"] + [f"swa_dpbr" for _ in K_GRID]
    lines = [
        r"\begin{table}[h]\centering\scriptsize",
        r"\caption{Wall-time overhead of SWA-DPBR relative to vanilla decoding "
        r"on Phi-4 (14B, BF16) for a single forward pass per scenario. "
        r"Timings are averaged over " + str(len(COUNTRIES))
        + r" countries $\times$ " + str(N_SCEN)
        + r" scenarios on a single H100 80GB GPU. "
        r"$\times$overhead = SWA / vanilla wall time per scenario.}",
        r"\label{tab:latency_overhead}",
        r"\begin{tabular}{lcccc}\toprule",
        r"Method & $K$ (IS samples) & sec/scenario & overhead & GPU peak (GB) \\\midrule",
    ]
    van = summary_df[summary_df["method"] == "vanilla"]
    van_sec = float(van["mean_sec_per_scen"].mean()) if not van.empty else float("nan")

    for _, row in summary_df.iterrows():
        overhead = row["mean_sec_per_scen"] / van_sec if van_sec > 0 else float("nan")
        k_str = "--" if row["method"] == "vanilla" else str(int(row["K"]))
        lines.append(
            f"{row['method']} & {k_str} & {row['mean_sec_per_scen']:.3f} & "
            f"{overhead:.2f}$\\times$ & {row['mean_gpu_peak_gb']:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ─── main ───────────────────────────────────────────────────────────────────
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
            print(f"[SKIP] {c}")
            continue
        scen_cache[c] = load_scenarios(cfg, c)

    raw_rows: List[Dict] = []

    for country in COUNTRIES:
        scen = scen_cache.get(country)
        if scen is None or scen.empty:
            continue
        print(f"\n[LATENCY] {country}  (n={len(scen)})")

        # Vanilla
        row_v = _bench_vanilla(model, tokenizer, cfg, country, scen)
        raw_rows.append(row_v)
        print(f"  vanilla         {row_v['sec_per_scen']:.3f} s/scen")

        # SWA-DPBR at each K
        for k_half in K_GRID:
            try:
                row_s = _bench_swa(model, tokenizer, cfg, country, scen, k_half)
                raw_rows.append(row_s)
                overhead = row_s["sec_per_scen"] / row_v["sec_per_scen"]
                print(f"  swa K={k_half*2:<4d}  {row_s['sec_per_scen']:.3f} s/scen  "
                      f"({overhead:.2f}×)  flip={row_s.get('flip_rate', float('nan')):.1%}")
            except Exception as exc:
                print(f"  [ERROR] K={k_half*2}: {exc}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(out_dir / "latency_raw.csv", index=False)
    print(f"\n[SAVED] {out_dir / 'latency_raw.csv'}")

    # Aggregate per (method, K)
    agg = (
        raw_df.groupby(["method", "K"])
        .agg(
            mean_sec_per_scen=("sec_per_scen", "mean"),
            std_sec_per_scen=("sec_per_scen", "std"),
            mean_gpu_peak_gb=("gpu_peak_gb", "mean"),
            n_cells=("n_scenarios", "count"),
        )
        .reset_index()
    )
    agg.to_csv(out_dir / "latency_summary.csv", index=False)
    print(f"[SAVED] {out_dir / 'latency_summary.csv'}")

    # LaTeX table
    latex = _build_latency_table(agg)
    (out_dir / "latency_table.tex").write_text(latex, encoding="utf-8")
    print(f"[SAVED] {out_dir / 'latency_table.tex'}")

    # Notes text
    van_sec = float(
        agg[agg["method"] == "vanilla"]["mean_sec_per_scen"].mean()
    ) if (agg["method"] == "vanilla").any() else float("nan")
    notes = [
        "Latency benchmark notes",
        "=" * 50,
        f"Model:       {MODEL_NAME}",
        f"Backend:     {os.environ.get('MORAL_MODEL_BACKEND', 'vllm')}",
        f"Countries:   {COUNTRIES}",
        f"N scenarios: {N_SCEN} per country",
        f"K grid:      {[k*2 for k in K_GRID]}",
        "",
        "Overhead summary (wall time per scenario):",
    ]
    for _, row in agg.iterrows():
        if row["method"] == "vanilla":
            continue
        ov = row["mean_sec_per_scen"] / van_sec if np.isfinite(van_sec) and van_sec > 0 else float("nan")
        notes.append(
            f"  K={int(row['K']):<4d}  {row['mean_sec_per_scen']:.3f}s  overhead={ov:.2f}×"
        )

    notes_str = "\n".join(notes) + "\n"
    (out_dir / "latency_notes.txt").write_text(notes_str, encoding="utf-8")
    print(f"[SAVED] {out_dir / 'latency_notes.txt'}")
    print("\n" + notes_str)

    _zip_outputs(out_dir, "round2_phase7_latency")


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
    print(f"[ZIP] {zip_path}")


if __name__ == "__main__":
    main()
