#!/usr/bin/env python3
"""Round-2 Reviewer W4b -- reliability-gate audit.

Reads the per-country ``swa_results_<COUNTRY>.csv`` written by the main Phi-4
paper run and emits a regime table answering "how often are both passes
high-ESS yet disagree, and what does the gate do in that case?".

This script does NOT re-run the model; it's post-hoc over existing CSVs.

Kaggle:
    !python exp_paper/round2/phase1_posthoc/exp_r2_reliability_audit.py

Env overrides:
    R2_SWA_DIR   where to find swa_results_*.csv (default: Kaggle Phi-4 path)
    R2_ESS_HI    threshold on min(ESS) (default: 0.40)
    R2_VAR_HI    threshold on bootstrap_var (default: 0.04)
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
    if not _os.path.isdir("/kaggle/working"):
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

import glob
import os
from pathlib import Path

from exp_paper._r2_common import ensure_repo, on_kaggle, save_summary
from src.config import model_slug  # noqa: E402
from src.dpbr_reliability_audit import audit_paths  # noqa: E402

MODEL_NAME = os.environ.get("R2_MODEL", "microsoft/phi-4")
MODEL_SHORT = os.environ.get("R2_MODEL_SHORT", "phi_4")
DEFAULT_SWA_DIR = (
    f"/kaggle/working/cultural_alignment/results/exp24_paper_20c/"
    f"{MODEL_SHORT}/swa/{model_slug(MODEL_NAME)}"
)
SWA_DIR = os.environ.get("R2_SWA_DIR", DEFAULT_SWA_DIR)
ESS_HI = float(os.environ.get("R2_ESS_HI", "0.40"))
VAR_HI = float(os.environ.get("R2_VAR_HI", "0.04"))

RESULTS_BASE = (
    "/kaggle/working/cultural_alignment/results/exp24_round2/reliability_audit"
    if on_kaggle()
    else str(Path(__file__).parent.parent / "results" / "exp24_round2" / "reliability_audit")
)


def main() -> None:
    paths = sorted(glob.glob(os.path.join(SWA_DIR, "swa_results_*.csv")))
    if not paths:
        raise SystemExit(
            f"No swa_results_*.csv under {SWA_DIR!r}. "
            "Run the main Phi-4 paper sweep first."
        )

    print(f"[AUDIT] Found {len(paths)} per-country CSVs under {SWA_DIR}")
    print(f"[AUDIT] ess_hi={ESS_HI}  var_hi={VAR_HI}")
    df = audit_paths(paths, ess_hi=ESS_HI, var_hi=VAR_HI)
    out_dir = Path(RESULTS_BASE)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "reliability_audit_per_country.csv"
    df.to_csv(csv_path, index=False)
    print(f"[SAVED] {csv_path}")

    # Cross-country means for the headline numbers.
    num_cols = [c for c in df.columns if c != "country" and c != "error"
                and df[c].dtype.kind in ("f", "i")]
    if num_cols:
        mean_row = {"country": "MEAN"} | {c: float(df[c].mean()) for c in num_cols}
        import pandas as pd
        mean_df = pd.DataFrame([mean_row])
        mean_csv = out_dir / "reliability_audit_mean.csv"
        mean_df.to_csv(mean_csv, index=False)
        print(f"[SAVED] {mean_csv}")

        # Log the table the paper will quote
        print("\n--- Regime breakdown (aggregated across countries) ---")
        for k in ["HighESS_LowDisagree", "HighESS_HighDisagree",
                  "LowESS_LowDisagree",  "LowESS_HighDisagree"]:
            n = float(df[f"{k}_count"].sum())
            frac = n / float(df["n_total"].sum())
            mean_r = float(df[f"{k}_mean_r"].mean())
            mean_flip = float(df[f"{k}_mean_flip"].mean())
            mean_shrink = float(df[f"{k}_mean_shrink"].mean())
            print(f"  {k:<22s}  n={int(n):5d} ({frac*100:5.1f}%)  "
                  f"mean_r={mean_r:.3f}  flip={mean_flip*100:4.1f}%  "
                  f"shrink={mean_shrink*100:4.1f}%")


if __name__ == "__main__":
    main()
