#!/usr/bin/env python3
"""Single global T_cat ablation (Reviewer Q4).

Replaces the per-category temperatures
(Species=4.0 / Gender=3.5 / others=1.5) with a single global value
T_cat in {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0} and re-runs SWA-DPBR on a
3-country panel to measure how much the per-category schedule actually
buys vs.\ a single tuned scalar.

Kaggle (Phi-4 14B x 3 countries x 7 T_cat values ~ 1.5h on H100):
    !python exp_paper/round3/sensitivity/exp_r3_global_tcat.py

Env overrides:
    R2_MODEL          HF id (default: microsoft/phi-4)
    R2_COUNTRIES      comma ISO3 list (default: USA,VNM,DEU)
    R2_N_SCENARIOS    per-country (default: 250)
    R2_TCAT_GRID      comma list of T_cat values (default: 1.0,1.5,2.0,2.5,3.0,3.5,4.0)
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
TCAT_GRID = [
    float(s.strip())
    for s in os.environ.get("R2_TCAT_GRID", "1.0,1.5,2.0,2.5,3.0,3.5,4.0").split(",")
    if s.strip()
]

RESULTS_BASE = (
    "/kaggle/working/cultural_alignment/results/exp24_round3/global_tcat"
    if on_kaggle()
    else str(Path(__file__).parent.parent / "results" / "exp24_round2" / "global_tcat")
)
WVS_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"

CAT_KEYS = ["Species", "Gender", "Age", "Fitness", "SocialValue", "Utilitarianism"]
DEFAULT_CAT_TEMPS = {"Species": 4.0, "Gender": 3.5, "Age": 1.5,
                     "Fitness": 1.5, "SocialValue": 1.5, "Utilitarianism": 1.5}


def _reset_prior(country: str) -> None:
    PRIOR_STATE.clear()
    PRIOR_STATE[country] = BootstrapPriorState()


def _set_global_tcat(cfg, value: float) -> Dict[str, float]:
    """Override cfg.category_logit_temperatures to a single global value.

    Returns the snapshot so it can be restored afterwards.
    """
    snap = dict(cfg.category_logit_temperatures)
    cfg.category_logit_temperatures = {k: float(value) for k in CAT_KEYS}
    return snap


def _run_one_tcat(model, tokenizer, cfg, country: str,
                  scen: pd.DataFrame, T_cat: float) -> Dict:
    snap = _set_global_tcat(cfg, T_cat)
    _reset_prior(country)
    patch_swa_runner_controller()
    personas = build_country_personas(country, wvs_path=WVS_PATH)
    t0 = time.time()
    try:
        results_df, summary = run_country_experiment(
            model, tokenizer, country, personas, scen, cfg)
    finally:
        cfg.category_logit_temperatures = snap

    a = summary["alignment"]
    return {
        "T_cat":            float(T_cat),
        "country":          country,
        "n_scenarios":      len(results_df),
        "elapsed_sec":      time.time() - t0,
        "mis":              a.get("mis", float("nan")),
        "jsd":              a.get("jsd", float("nan")),
        "pearson_r":        a.get("pearson_r", float("nan")),
        "flip_rate":        summary.get("flip_rate", float("nan")),
    }


def _run_default(model, tokenizer, cfg, country: str, scen: pd.DataFrame) -> Dict:
    """Per-category default schedule (4.0 / 3.5 / 1.5) — the released config."""
    snap = dict(cfg.category_logit_temperatures)
    cfg.category_logit_temperatures = dict(DEFAULT_CAT_TEMPS)
    _reset_prior(country)
    patch_swa_runner_controller()
    personas = build_country_personas(country, wvs_path=WVS_PATH)
    t0 = time.time()
    try:
        results_df, summary = run_country_experiment(
            model, tokenizer, country, personas, scen, cfg)
    finally:
        cfg.category_logit_temperatures = snap

    a = summary["alignment"]
    return {
        "T_cat":            -1.0,  # sentinel meaning "per-category default"
        "country":          country,
        "n_scenarios":      len(results_df),
        "elapsed_sec":      time.time() - t0,
        "mis":              a.get("mis", float("nan")),
        "jsd":              a.get("jsd", float("nan")),
        "pearson_r":        a.get("pearson_r", float("nan")),
        "flip_rate":        summary.get("flip_rate", float("nan")),
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

    # First: the per-category default (released config) as a baseline reference.
    print(f"\n{'#' * 72}\n# Per-category default (Species=4.0 / Gender=3.5 / others=1.5)\n{'#' * 72}")
    for country in COUNTRIES:
        if country not in scen_cache:
            continue
        try:
            row = _run_default(model, tokenizer, cfg, country, scen_cache[country])
            rows.append(row)
            pd.DataFrame(rows).to_csv(out_dir / "global_tcat_partial.csv", index=False)
            print(f"  ✓ default {country}  MIS={row['mis']:.4f}  flip={row['flip_rate']:.3f}")
        except Exception as exc:
            print(f"[error] default {country}: {exc}")
            rows.append({"T_cat": -1.0, "country": country, "error": str(exc)[:500]})

    # Then: each global T_cat value.
    for T in TCAT_GRID:
        print(f"\n{'#' * 72}\n# global T_cat = {T}\n{'#' * 72}")
        for country in COUNTRIES:
            if country not in scen_cache:
                continue
            try:
                row = _run_one_tcat(model, tokenizer, cfg, country, scen_cache[country], T)
                rows.append(row)
                pd.DataFrame(rows).to_csv(out_dir / "global_tcat_partial.csv", index=False)
                print(f"  ✓ T={T:.2f} {country}  MIS={row['mis']:.4f}  flip={row['flip_rate']:.3f}")
            except Exception as exc:
                print(f"[error] T={T} {country}: {exc}")
                rows.append({"T_cat": T, "country": country, "error": str(exc)[:500]})

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    save_summary(rows, out_dir, "global_tcat_summary.csv")
    _build_curve(rows, out_dir)
    _zip_outputs(out_dir, "round3_sensitivity_global_tcat")


def _build_curve(rows: List[Dict], out_dir: Path) -> None:
    df = pd.DataFrame([r for r in rows if "mis" in r and r.get("mis") == r.get("mis")])
    if df.empty:
        return
    macro = df.groupby("T_cat").agg(
        n_countries=("country", "nunique"),
        mean_mis=("mis", "mean"),
        std_mis=("mis", "std"),
        mean_flip=("flip_rate", "mean"),
    ).reset_index()
    macro.to_csv(out_dir / "global_tcat_macro.csv", index=False)
    print(f"\n[saved] {out_dir / 'global_tcat_macro.csv'}")
    print(macro.round(4).to_string(index=False))

    # LaTeX table (default row first, then sorted T_cat).
    lines = [
        r"\begin{tabular}{lccc}\toprule",
        r"$T_{\text{cat}}$ schedule & Macro MIS $\downarrow$ & Std (across countries) & Flip rate \\\midrule",
    ]
    default_row = macro[macro["T_cat"] == -1.0]
    other_rows  = macro[macro["T_cat"] != -1.0].sort_values("T_cat")
    if not default_row.empty:
        r = default_row.iloc[0]
        lines.append(
            r"per-category default (4.0 / 3.5 / 1.5) & "
            f"\\textbf{{{r['mean_mis']:.4f}}} & {r['std_mis']:.4f} & {r['mean_flip']:.3f} \\\\"
        )
        lines.append(r"\midrule")
    for _, r in other_rows.iterrows():
        lines.append(
            f"global $T_\\text{{cat}} = {r['T_cat']:.2f}$ & "
            f"{r['mean_mis']:.4f} & {r['std_mis']:.4f} & {r['mean_flip']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "global_tcat_table.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {out_dir / 'global_tcat_table.tex'}")


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
