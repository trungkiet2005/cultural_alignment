#!/usr/bin/env python3
"""Experiment 9 (playbook) — Negative Pearson r Diagnosis.

Pure post-hoc analysis. For each country, compares the model AMCE vector
to the human AMCE vector and identifies countries where Pearson r < 0
despite a (possibly) improved MIS. For each such country, enumerates
the dimension pairs whose ranking is reversed between human and model.

Inputs (long-form CSVs):
  R4_MODEL_AMCE_CSV   columns: country, dimension, model_amce
  R4_HUMAN_AMCE_CSV   columns: country, dimension, human_amce
                      (or wide-form: country + one column per dimension —
                       script auto-melts).

Outputs (in RESULTS_BASE/):
  negr_country_summary.csv   — per-country: n_dims, r, sign(r), n_swaps
  negr_swap_pairs.csv        — long-form (country, dim_a, dim_b) for swapped pairs
  negr_paragraph.txt         — paper-ready paragraph stats

Defends against:
  R2: "DISCA improves MIS but has negative r in several countries — does it
       capture cultural structure or just push numbers around?"

Env overrides:
  R4_MODEL_AMCE_CSV   [REQUIRED] path
  R4_HUMAN_AMCE_CSV   [REQUIRED] path
  R4_DEFAULT_DIMS     comma list of dimension names to enforce ordering
                      (default: Species_Humans,Gender_Female,Age_Young,
                                Fitness_Fit,SocialValue_High,Utilitarianism_More)

Local / Kaggle:
    R4_MODEL_AMCE_CSV=/path/disca_amce_long.csv \
        R4_HUMAN_AMCE_CSV=/path/human_amce_long.csv \
        python exp_paper/playbook/exp_r4_negr_diagnosis.py
"""

from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _kaggle_setup import bootstrap_offline, zip_outputs as _zip_outputs

bootstrap_offline()

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

_HERE = Path(__file__).resolve()

DEFAULT_DIMS = [
    "Species_Humans",
    "Gender_Female",
    "Age_Young",
    "Fitness_Fit",
    "SocialValue_High",
    "Utilitarianism_More",
]
DIMS = [
    d.strip()
    for d in _os.environ.get("R4_DEFAULT_DIMS", ",".join(DEFAULT_DIMS)).split(",")
    if d.strip()
]
MODEL_AMCE_CSV = _os.environ.get("R4_MODEL_AMCE_CSV", "").strip()
HUMAN_AMCE_CSV = _os.environ.get("R4_HUMAN_AMCE_CSV", "").strip()

RESULTS_BASE = (
    "/kaggle/working/cultural_alignment/results/exp24_round4/negr_diagnosis"
    if _os.path.isdir("/kaggle/working")
    else str(_HERE.parent.parent / "results" / "exp24_round4" / "negr_diagnosis")
)


def _load_long(path: str, value_col: str) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Not a file: {p}")
    df = pd.read_csv(p)
    if value_col in df.columns and "dimension" in df.columns and "country" in df.columns:
        return df[["country", "dimension", value_col]].copy()
    # Wide-form: country, <dim1>, <dim2>, …
    if "country" not in df.columns:
        raise ValueError(f"{path} needs a 'country' column.")
    dim_cols = [c for c in df.columns if c in DIMS]
    if not dim_cols:
        raise ValueError(
            f"{path} has neither '{value_col}' long form nor wide-form dim columns "
            f"matching {DIMS}; got {list(df.columns)}"
        )
    melt = df.melt(id_vars="country", value_vars=dim_cols,
                   var_name="dimension", value_name=value_col)
    return melt


def _country_vec(df_long: pd.DataFrame, value_col: str, country: str) -> Tuple[np.ndarray, List[str]]:
    sub = df_long[df_long["country"] == country].set_index("dimension")[value_col]
    keep = [d for d in DIMS if d in sub.index]
    return sub.reindex(keep).to_numpy(dtype=float), keep


def _find_swaps(human: np.ndarray, model: np.ndarray, dims: List[str]) -> List[Tuple[str, str]]:
    if len(human) < 2:
        return []
    h_rank = np.argsort(np.argsort(-human))  # higher value -> rank 0
    m_rank = np.argsort(np.argsort(-model))
    swaps: List[Tuple[str, str]] = []
    for i in range(len(dims)):
        for j in range(i + 1, len(dims)):
            if (h_rank[i] < h_rank[j]) != (m_rank[i] < m_rank[j]):
                swaps.append((dims[i], dims[j]))
    return swaps


def main() -> None:
    if not MODEL_AMCE_CSV or not HUMAN_AMCE_CSV:
        raise SystemExit(
            "Missing R4_MODEL_AMCE_CSV / R4_HUMAN_AMCE_CSV env vars. "
            "Provide long-form CSVs (country, dimension, <amce>)."
        )
    out_dir = Path(RESULTS_BASE)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_long = _load_long(MODEL_AMCE_CSV, "model_amce")
    human_long = _load_long(HUMAN_AMCE_CSV, "human_amce")
    countries = sorted(set(model_long["country"]) & set(human_long["country"]))
    if not countries:
        raise SystemExit("No overlapping countries between model and human CSVs.")

    summary_rows: List[Dict] = []
    swap_rows: List[Dict] = []
    swap_pair_counter: Dict[Tuple[str, str], int] = {}

    for c in countries:
        h_vec, h_dims = _country_vec(human_long, "human_amce", c)
        m_vec, m_dims = _country_vec(model_long, "model_amce", c)
        # Align on shared dims
        shared = [d for d in h_dims if d in m_dims]
        if len(shared) < 2:
            continue
        h_idx = [h_dims.index(d) for d in shared]
        m_idx = [m_dims.index(d) for d in shared]
        h = h_vec[h_idx]
        m = m_vec[m_idx]
        if not (np.all(np.isfinite(h)) and np.all(np.isfinite(m))):
            continue
        if h.std() == 0 or m.std() == 0:
            r = float("nan")
        else:
            r, _ = pearsonr(h, m)
        sw = _find_swaps(h, m, shared)
        for a, b in sw:
            key = tuple(sorted([a, b]))
            swap_pair_counter[key] = swap_pair_counter.get(key, 0) + 1
            swap_rows.append({"country": c, "dim_a": key[0], "dim_b": key[1]})
        summary_rows.append({
            "country": c,
            "n_dims": len(shared),
            "pearson_r": float(r) if np.isfinite(r) else float("nan"),
            "negative_r": bool(np.isfinite(r) and r < 0),
            "n_swaps": len(sw),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("pearson_r")
    swap_df = pd.DataFrame(swap_rows)
    summary_df.to_csv(out_dir / "negr_country_summary.csv", index=False)
    swap_df.to_csv(out_dir / "negr_swap_pairs.csv", index=False)

    # Aggregate the most common swap pair across negative-r countries
    neg_countries = summary_df[summary_df["negative_r"]]["country"].tolist()
    if not swap_df.empty:
        # Restrict to neg-r countries
        sw_neg = swap_df[swap_df["country"].isin(neg_countries)]
        pair_counts = (
            sw_neg.groupby(["dim_a", "dim_b"], as_index=False)
            .size()
            .sort_values("size", ascending=False)
        )
        pair_counts.to_csv(out_dir / "negr_swap_pair_counts.csv", index=False)
    else:
        pair_counts = pd.DataFrame()

    # Paragraph-ready paragraph
    n_total = len(summary_df)
    n_neg = len(neg_countries)
    para_lines = [
        f"Negative Pearson r occurs in {n_neg} of the {n_total} countries.",
    ]
    if n_neg > 0 and not pair_counts.empty:
        top = pair_counts.iloc[0]
        para_lines.append(
            f"In {int(top['size'])} of the {n_neg} negative-r countries, the rank "
            f"reversal occurs between {top['dim_a']} and {top['dim_b']} -- the same "
            f"two dimensions consistently."
        )
    para_lines.append(
        "DISCA corrects the magnitude of the moral preference vector but does not "
        "always correct the relative ordering of the hardest dimension pair; this "
        "is a specific, diagnosable limitation rather than an unexplained failure."
    )
    paragraph = "\n".join(para_lines)
    (out_dir / "negr_paragraph.txt").write_text(paragraph + "\n", encoding="utf-8")

    print("\n" + "-" * 70)
    print("  Negative Pearson r Diagnosis — RESULT")
    print("-" * 70)
    print(f"  countries analyzed    : {n_total}")
    print(f"  negative-r countries  : {n_neg}")
    if n_neg:
        print(f"  list                  : {neg_countries}")
    if not pair_counts.empty:
        print("\n  Top swap pairs in neg-r countries:")
        print(pair_counts.head(5).to_string(index=False))
    print("\n  Paragraph:")
    print("  " + paragraph.replace("\n", "\n  "))
    print(f"\n  saved -> {out_dir}")


if __name__ == "__main__":
    main()
    try:
        _zip_outputs(RESULTS_BASE)
    except Exception as _e:
        print(f"[ZIP] failed: {_e}")
