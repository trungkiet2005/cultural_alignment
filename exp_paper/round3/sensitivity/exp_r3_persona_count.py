#!/usr/bin/env python3
"""Persona-count ablation: N \\in {2, 3, 4, 5, 6} (Reviewer Q3).

Tests whether the released 4-persona panel is the right operating
point. Each N-variant slices or extends the standard 4-persona list
(``src.personas.build_country_personas``):

  * N=2  young + older only (no middle cohort, no aggregate)
  * N=3  three age cohorts only (no aggregate)
  * N=4  full default panel (3 cohorts + aggregate)            -- baseline
  * N=5  default + a country-invariant utilitarian anchor
  * N=6  default + utilitarian anchor + a deontological anchor

The deontological anchor is a manually crafted single-line persona
that prioritises rule-following / harm-avoidance regardless of
country, used purely to test "do more personas help, even if they are
not WVS-grounded?".

Kaggle (Phi-4 14B x 3 countries x 5 N values ~ 1.5h on H100):
    !python exp_paper/round3/sensitivity/exp_r3_persona_count.py

Env overrides:
    R2_MODEL          HF id (default: microsoft/phi-4)
    R2_COUNTRIES      comma ISO3 list (default: USA,VNM,DEU)
    R2_N_SCENARIOS    per-country (default: 250)
    R2_N_GRID         comma list of N values (default: 2,3,4,5,6)
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
from src.constants import COUNTRY_FULL_NAMES  # noqa: E402
from src.model import setup_seeds  # noqa: E402
from src.personas import SUPPORTED_COUNTRIES, build_country_personas  # noqa: E402
from src.swa_runner import run_country_experiment  # noqa: E402


MODEL_NAME = os.environ.get("R2_MODEL", "microsoft/phi-4")
N_SCEN = int(os.environ.get("R2_N_SCENARIOS", "250"))
COUNTRIES = [c.strip() for c in os.environ.get("R2_COUNTRIES", "USA,VNM,DEU").split(",") if c.strip()]
N_GRID = [int(s.strip()) for s in os.environ.get("R2_N_GRID", "2,3,4,5,6").split(",") if s.strip()]

RESULTS_BASE = (
    "/kaggle/working/cultural_alignment/results/exp24_round3/persona_count"
    if on_kaggle()
    else str(Path(__file__).parent.parent / "results" / "exp24_round2" / "persona_count")
)
WVS_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"


# ─── extra persona templates (non-WVS, country-invariant) ────────────────────
_UTILITARIAN_PERSONA = (
    "You are a strict utilitarian: when judging a moral dilemma, you choose "
    "the option that maximises the total number of lives saved or the total "
    "well-being, regardless of the identities of the people involved. Apply "
    "this principle directly to the scenario."
)
_DEONTOLOGICAL_PERSONA = (
    "You are a strict deontologist: when judging a moral dilemma, you "
    "choose based on duty, rights, and rules. Avoid actively causing harm "
    "even if doing so would produce a better aggregate outcome. Apply this "
    "principle directly to the scenario."
)


def _slice_personas(country: str, N: int) -> List[str]:
    """Return a length-N persona list derived from the default 4-persona panel.

    The default panel order from ``build_country_personas`` is:
        [young, middle, older, aggregate]
    """
    full = build_country_personas(country, wvs_path=WVS_PATH)
    if len(full) < 3:
        # Fallback paths (e.g., no WVS coverage) returned <3 — bail out.
        return full
    if N == 2:
        # Endpoints only: young + older.
        return [full[0], full[2]]
    if N == 3:
        # Three age cohorts, no aggregate.
        return full[:3]
    if N == 4:
        return full
    if N == 5:
        return full + [_UTILITARIAN_PERSONA]
    if N == 6:
        return full + [_UTILITARIAN_PERSONA, _DEONTOLOGICAL_PERSONA]
    raise ValueError(f"Unsupported N={N}")


def _reset_prior(country: str) -> None:
    PRIOR_STATE.clear()
    PRIOR_STATE[country] = BootstrapPriorState()


def _run_one_N(model, tokenizer, cfg, country: str,
               scen: pd.DataFrame, N: int) -> Dict:
    personas = _slice_personas(country, N)
    _reset_prior(country)
    patch_swa_runner_controller()
    t0 = time.time()
    results_df, summary = run_country_experiment(
        model, tokenizer, country, personas, scen, cfg)
    a = summary["alignment"]
    return {
        "N":                len(personas),
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
    for N in N_GRID:
        print(f"\n{'#' * 72}\n# N = {N} personas\n{'#' * 72}")
        for country in COUNTRIES:
            if country not in scen_cache:
                continue
            try:
                row = _run_one_N(model, tokenizer, cfg, country, scen_cache[country], N)
                rows.append(row)
                pd.DataFrame(rows).to_csv(out_dir / "persona_count_partial.csv", index=False)
                print(f"  ✓ N={N} {country}  MIS={row['mis']:.4f}  flip={row['flip_rate']:.3f}  "
                      f"({row['elapsed_sec']:.0f}s)")
            except Exception as exc:
                print(f"[error] N={N} {country}: {exc}")
                rows.append({"N": N, "country": country, "error": str(exc)[:500]})
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    save_summary(rows, out_dir, "persona_count_summary.csv")
    _build_curve(rows, out_dir)
    _zip_outputs(out_dir, "round3_sensitivity_persona_count")


def _build_curve(rows: List[Dict], out_dir: Path) -> None:
    df = pd.DataFrame([r for r in rows if "mis" in r and r.get("mis") == r.get("mis")])
    if df.empty:
        return
    macro = df.groupby("N").agg(
        n_countries=("country", "nunique"),
        mean_mis=("mis", "mean"),
        std_mis=("mis", "std"),
        mean_flip=("flip_rate", "mean"),
        mean_elapsed=("elapsed_sec", "mean"),
    ).reset_index()
    macro.to_csv(out_dir / "persona_count_macro.csv", index=False)
    print(f"\n[saved] {out_dir / 'persona_count_macro.csv'}")
    print(macro.round(4).to_string(index=False))

    lines = [
        r"\begin{tabular}{rcccc}\toprule",
        r"$N$ & Macro MIS $\downarrow$ & Std (across countries) & Flip rate & sec/scenario \\\midrule",
    ]
    for _, row in macro.iterrows():
        n_countries = max(int(row["n_countries"]), 1)
        # Approximate per-scenario time: elapsed / N_SCEN.
        per_scen = row["mean_elapsed"] / max(N_SCEN, 1)
        bold_open  = r"\textbf{" if int(row["N"]) == 4 else ""
        bold_close = r"}"        if int(row["N"]) == 4 else ""
        lines.append(
            f"{bold_open}{int(row['N'])}{bold_close} & "
            f"{bold_open}{row['mean_mis']:.4f}{bold_close} & "
            f"{row['std_mis']:.4f} & {row['mean_flip']:.3f} & {per_scen:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "persona_count_table.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {out_dir / 'persona_count_table.tex'}")


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
