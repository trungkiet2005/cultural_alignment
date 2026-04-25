#!/usr/bin/env python3
"""Round-2 Reviewer W2 -- ablation breadth beyond Phi-4 / USA.

Runs the same 6-row ablation (Full / No-IS / Always-on / No-debias /
No-persona / No-prior) from :mod:`exp_paper.ablation.exp_paper_ablation_phi4` but on
a grid of additional models × countries so reviewers can see that the
load-bearing components are not Phi-4 / USA specific.

Default grid (env-overridable):
    models    = ["microsoft/phi-4", "Qwen/Qwen2.5-7B-Instruct", "microsoft/Phi-3.5-mini-instruct"]
    countries = ["USA", "JPN", "VNM"]

This is 3 × 3 × 6 = 54 runs. We stream partial CSV after each cell so a
mid-session crash keeps the results we already have.

Kaggle:
    !python exp_paper/review/round2/phase4_big_sweeps/exp_r2_ablation_breadth.py
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Self-bootstrap — works when copy-pasted into a fresh Kaggle notebook cell.
# Clones the repo on Kaggle if not already on sys.path, then adds it. Safe to
# run multiple times (idempotent: detects src/controller.py in cwd).
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


# Set backend BEFORE paper_runtime is imported so install_paper_kaggle_deps()
# picks the correct pip branch (vLLM vs Unsloth).
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

# Reuse the 5 ablation controller subclasses from the Phi-4 ablation script.
from exp_paper.ablation.exp_paper_ablation_phi4 import (  # noqa: E402
    ABLATION_SPECS,
    AblationSpec,
    NoPersonaController,
    NoPriorController,
    _collect_row,
    _reset_prior_state,
    _run_ablation_country,
)
from experiment_DM.exp24_dpbr_core import (  # noqa: E402
    BootstrapPriorState,
    PRIOR_STATE,
)
from src.model import setup_seeds  # noqa: E402
from src.personas import SUPPORTED_COUNTRIES, build_country_personas  # noqa: E402

MODELS: List[str] = [
    s.strip() for s in os.environ.get(
        "R2_BREADTH_MODELS",
        "microsoft/phi-4,Qwen/Qwen2.5-7B-Instruct,microsoft/Phi-3.5-mini-instruct",
    ).split(",")
    if s.strip()
]
COUNTRIES: List[str] = [
    s.strip() for s in os.environ.get("R2_BREADTH_COUNTRIES", "USA,JPN,VNM").split(",")
    if s.strip()
]
N_SCEN = int(os.environ.get("R2_N_SCENARIOS", "300"))

RESULTS_BASE = (
    "/kaggle/working/cultural_alignment/results/exp24_round2/ablation_breadth"
    if on_kaggle()
    else str(Path(__file__).parent.parent / "results" / "exp24_round2" / "ablation_breadth")
)
WVS_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"


def _run_one_model(model_name: str, out_dir: Path) -> List[Dict]:
    setup_seeds(42)
    short = model_name.split("/")[-1].lower().replace("_", "-")
    print(f"\n{'#' * 80}\n# Ablation model: {model_name}\n{'#' * 80}")

    cfg = build_cfg(
        model_name, str(out_dir), COUNTRIES,
        n_scenarios=N_SCEN, load_in_4bit=False,
    )
    backend = os.environ.get("MORAL_MODEL_BACKEND", "vllm").strip().lower()
    model, tokenizer = load_model_timed(
        model_name, backend=backend, load_in_4bit=False,
    )

    rows: List[Dict] = []
    try:
        for country in COUNTRIES:
            if country not in SUPPORTED_COUNTRIES:
                print(f"[SKIP] {country}")
                continue
            scen = load_scenarios(cfg, country)
            personas = build_country_personas(country, wvs_path=WVS_PATH)

            for spec in ABLATION_SPECS:
                print(f"\n[{short}|{country}|{spec.row_label}]  {spec.description}")
                _reset_prior_state(country)
                torch.cuda.empty_cache()
                gc.collect()
                t0 = time.time()
                try:
                    results_df, summary = _run_ablation_country(
                        spec, model, tokenizer, country, personas, scen, cfg,
                    )
                except Exception as exc:
                    print(f"[ERROR] {model_name} {country} {spec.row_label}: {exc}")
                    rows.append({
                        "model": model_name, "ablation": spec.row_label,
                        "country": country, "error": str(exc)[:500],
                    })
                    continue
                dt = time.time() - t0
                row = _collect_row(spec, country, results_df, summary, dt)
                row["model"] = model_name
                rows.append(row)
                pd.DataFrame(rows).to_csv(
                    out_dir / f"ablation_breadth_{short}_partial.csv", index=False,
                )
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def main() -> None:
    out_dir = Path(RESULTS_BASE)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []
    for mname in MODELS:
        try:
            all_rows.extend(_run_one_model(mname, out_dir))
        except Exception as exc:
            print(f"[ERROR] model={mname}: {exc}")
            all_rows.append({"model": mname, "error": str(exc)[:500]})
        pd.DataFrame(all_rows).to_csv(out_dir / "ablation_breadth_partial.csv", index=False)

    save_summary(all_rows, out_dir, "ablation_breadth_summary.csv")


if __name__ == "__main__":
    main()
