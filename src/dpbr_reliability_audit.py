"""DPBR reliability-audit: post-hoc diagnostics for the dual-pass gate.

Round 2 reviewer W4b asks:

    "Did you observe regimes where both passes have high ESS yet disagree
    substantially -- if so, how often, and what are typical outcomes after
    gating?"

This module answers that from any SWA-DPBR per-scenario results CSV (the
``swa_results_<COUNTRY>.csv`` files written by
:func:`src.swa_runner.run_country_experiment`, which already include the
columns ``ess_pass1``, ``ess_pass2``, ``bootstrap_var``, ``reliability_r``,
``delta_star_1``, ``delta_star_2``).

Two audits are produced:

1. **Regime counts.** Partition every scenario into four regimes based on
   two thresholds ``ess_hi`` and ``var_hi``:

        HighESS_LowDisagree  (ESS_min ≥ ess_hi, bootstrap_var < var_hi)
        HighESS_HighDisagree (ESS_min ≥ ess_hi, bootstrap_var ≥ var_hi)
        LowESS_LowDisagree   (ESS_min < ess_hi, bootstrap_var < var_hi)
        LowESS_HighDisagree  (ESS_min < ess_hi, bootstrap_var ≥ var_hi)

   Report: count, mean reliability_r, mean |delta_star|, mean consensus
   reversal rate (``mppi_flipped``).

2. **Gating counterfactual.** Under each scenario's reliability weight ``r``,
   how much did the gate shrink the average of the two IS passes? Report
   ``(delta_star_1 + delta_star_2) / 2`` vs ``r * (...) / 2`` to give a
   "would-have-applied-vs-actually-applied" ratio.

Entry point :func:`audit_country` reads one CSV and returns a dict of
reported numbers. :func:`audit_paths` aggregates across many CSVs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _finite_mean(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float("nan")


def audit_country(
    csv_path: str,
    *,
    ess_hi: float = 0.40,
    var_hi: float = 0.04,
) -> Dict[str, float]:
    """Audit one per-scenario SWA-DPBR CSV.

    Parameters
    ----------
    csv_path
        Path to ``swa_results_<COUNTRY>.csv``.
    ess_hi
        Threshold on ``min(ess_pass1, ess_pass2)`` to call a scenario
        "high-ESS". Default 0.40 (i.e., ≥40% effective-sample retention).
    var_hi
        Threshold on ``bootstrap_var`` to call a scenario "high-disagreement".
        Default 0.04 = the paper's ``VAR_SCALE`` (i.e., half-life r=1/e).
    """
    df = pd.read_csv(csv_path)
    req = {"ess_pass1", "ess_pass2", "bootstrap_var", "reliability_r",
           "delta_star_1", "delta_star_2"}
    if not req.issubset(df.columns):
        missing = req - set(df.columns)
        raise ValueError(f"CSV {csv_path!r} missing columns: {missing}")

    ess_min = np.minimum(df["ess_pass1"].values, df["ess_pass2"].values)
    bvar    = df["bootstrap_var"].values
    r       = df["reliability_r"].values
    ds1     = df["delta_star_1"].values
    ds2     = df["delta_star_2"].values
    flipped = df.get("mppi_flipped", pd.Series([False] * len(df))).values

    # "Would apply" vs "actually applied" magnitudes.
    avg_unrel = 0.5 * (ds1 + ds2)           # what we would have applied if r≡1
    actual    = r * avg_unrel               # what DPBR actually applies
    shrink    = 1.0 - (np.abs(actual) / np.maximum(np.abs(avg_unrel), 1e-12))

    regimes = {
        "HighESS_LowDisagree":  (ess_min >= ess_hi) & (bvar <  var_hi),
        "HighESS_HighDisagree": (ess_min >= ess_hi) & (bvar >= var_hi),
        "LowESS_LowDisagree":   (ess_min <  ess_hi) & (bvar <  var_hi),
        "LowESS_HighDisagree":  (ess_min <  ess_hi) & (bvar >= var_hi),
    }
    out: Dict[str, float] = {
        "country":  Path(csv_path).stem.replace("swa_results_", ""),
        "n_total":  len(df),
        "ess_hi":   ess_hi,
        "var_hi":   var_hi,
        "mean_r":   _finite_mean(r),
        "mean_bvar": _finite_mean(bvar),
        "mean_ess_min": _finite_mean(ess_min),
        "mean_shrink_abs": _finite_mean(shrink),
        "flip_rate": float(flipped.mean()) if flipped.size else float("nan"),
    }
    for name, mask in regimes.items():
        n = int(mask.sum())
        out[f"{name}_count"] = n
        out[f"{name}_frac"]  = float(n / max(1, len(df)))
        out[f"{name}_mean_r"]       = _finite_mean(r[mask])       if n else float("nan")
        out[f"{name}_mean_flip"]    = float(flipped[mask].mean()) if n else float("nan")
        out[f"{name}_mean_shrink"]  = _finite_mean(shrink[mask])  if n else float("nan")
        out[f"{name}_mean_avg_ds"]  = _finite_mean(avg_unrel[mask]) if n else float("nan")
        out[f"{name}_mean_actual"]  = _finite_mean(actual[mask])    if n else float("nan")
    return out


def audit_paths(
    csv_paths: List[str],
    *,
    ess_hi: float = 0.40,
    var_hi: float = 0.04,
) -> pd.DataFrame:
    """Audit a list of per-country CSVs; concatenate into one DataFrame."""
    rows: List[Dict[str, float]] = []
    for p in csv_paths:
        try:
            rows.append(audit_country(p, ess_hi=ess_hi, var_hi=var_hi))
        except Exception as exc:
            rows.append({"country": Path(p).stem, "error": str(exc)[:500]})
    return pd.DataFrame(rows)
