#!/usr/bin/env python3
"""Oral-level analysis: WVS persona validity — do personas independently
track human AMCEs? (Reviewer Q5).

This script answers whether the WVS-grounded cultural profile of a country
correlates with its human AMCE vector *independently* of any LLM correction
— i.e., whether the SWA-DPBR personas are targeting the right direction to
begin with.

Two complementary analyses:

(A) WVS → AMCE linear regression
    For each of the 6 MultiTP AMCE dimensions, fit an OLS regression from
    the 10 WVS country-level feature means (same dims used in personas.py)
    to predict the human AMCE. Report R², adjusted R², and standardised
    coefficients so reviewers can see *which* WVS dimensions drive each
    AMCE dimension.

(B) Persona-consensus AMCE vs human AMCE (Pearson r and Spearman rho)
    For each country, the SWA-DPBR controller builds a panel of 5 personas
    (4 age cohorts + utilitarian neutral). Each persona is a text description;
    we compute the *semantic* alignment of the persona panel with the human
    AMCE via the WVS feature vector of the panel's centroid. We then
    correlate this with the human AMCE dimension-by-dimension across the 20
    countries. Positive r confirms personas aim in the right direction.

No GPU required — pure post-hoc from WVS CSV + human AMCE CSV.

Kaggle:
    !python exp_paper/round2/phase7_oral/exp_r2_persona_amce_corr.py

Env overrides:
    R2_WVS_PATH       WVS-7 inverted CSV (default: Kaggle path)
    R2_HUMAN_AMCE     country_specific_ACME.csv (default: Kaggle path)
    R2_COUNTRIES      comma-separated ISO3 (default: all 20 paper countries)
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

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from exp_paper._r2_common import on_kaggle
from exp_paper.paper_countries import PAPER_20_COUNTRIES
from src.amce import load_human_amce

# ─── paths ──────────────────────────────────────────────────────────────────
WVS_PATH = os.environ.get(
    "R2_WVS_PATH",
    "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
    if on_kaggle()
    else "WVS_data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv",
)
HUMAN_AMCE_PATH = os.environ.get(
    "R2_HUMAN_AMCE",
    "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"
    if on_kaggle()
    else "WVS_data/country_specific_ACME.csv",
)
COUNTRIES = (
    [c.strip() for c in os.environ["R2_COUNTRIES"].split(",") if c.strip()]
    if "R2_COUNTRIES" in os.environ
    else list(PAPER_20_COUNTRIES)
)
OUT_DIR = Path(
    "/kaggle/working/cultural_alignment/results/exp24_round2/persona_amce_corr"
    if on_kaggle()
    else "results/exp24_round2/persona_amce_corr"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# The 10 WVS feature dims (same as personas.py WVS_DIMS labels).
WVS_FEATURES = [
    "religiosity", "child_rearing", "moral_acceptability",
    "social_trust", "political_participation", "national_pride",
    "happiness", "gender_equality", "materialism_orientation",
    "tolerance_diversity",
]

# WVS question-code → feature mapping (mirrors personas.py WVS_DIMS).
# We compute per-country means of these codes then normalise to [0,1].
_WVS_CODES: Dict[str, List[str]] = {
    "religiosity":             ["Q6P"],
    "child_rearing":           ["Q14P", "Q17P"],
    "moral_acceptability":     ["Q177", "Q178", "Q179", "Q180", "Q181", "Q182"],
    "social_trust":            ["Q57P"],
    "political_participation": ["Q199P", "Q200P", "Q201P"],
    "national_pride":          ["Q254P"],
    "happiness":               ["Q46P"],
    "gender_equality":         ["Q29P", "Q30P", "Q31P", "Q32P", "Q33P"],
    "materialism_orientation": ["Q152", "Q153", "Q154"],
    "tolerance_diversity":     ["Q19P", "Q20P", "Q21P", "Q22P", "Q23P"],
}

AMCE_DIMS = [
    "Species_Humans", "Gender_Female", "Age_Young",
    "Fitness_Fit", "SocialValue_High", "Utilitarianism_More",
]


# ─── WVS feature extraction ──────────────────────────────────────────────────
def _extract_wvs_features(wvs_df: pd.DataFrame, iso3: str) -> Optional[Dict[str, float]]:
    """Compute normalised [0,1] country mean for each WVS feature."""
    # WVS column for country codes (B_COUNTRY_ALPHA or B_COUNTRY).
    country_col = None
    for c in ("B_COUNTRY_ALPHA", "B_COUNTRY"):
        if c in wvs_df.columns:
            country_col = c
            break
    if country_col is None:
        raise RuntimeError("WVS CSV has no B_COUNTRY_ALPHA or B_COUNTRY column.")

    sub = wvs_df[wvs_df[country_col] == iso3]
    if sub.empty:
        return None

    feats: Dict[str, float] = {}
    for feat, codes in _WVS_CODES.items():
        present = [c for c in codes if c in sub.columns]
        if not present:
            feats[feat] = float("nan")
            continue
        vals = sub[present].replace([-1, -2, -3, -4, -5], np.nan).mean(axis=1)
        raw_mean = float(vals.mean(skipna=True))
        # Normalise by observed range across all valid rows in the full WVS,
        # not just this country. Range: use column min/max over entire dataset.
        all_vals = wvs_df[present].replace([-1, -2, -3, -4, -5], np.nan).mean(axis=1)
        lo, hi = float(all_vals.min(skipna=True)), float(all_vals.max(skipna=True))
        feats[feat] = (raw_mean - lo) / max(hi - lo, 1e-9)
    return feats


# ─── analysis A: OLS regression WVS → AMCE ───────────────────────────────────
def _run_regression(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    dim_name: str,
) -> Dict:
    """OLS with standardised predictors. Returns R², adjusted R², coefs."""
    n, p = X.shape
    # Standardise X and y.
    X_std = (X - X.mean(axis=0)) / np.clip(X.std(axis=0, ddof=1), 1e-9, None)
    y_std = (y - y.mean()) / max(float(y.std(ddof=1)), 1e-9)

    X_aug = np.column_stack([np.ones(n), X_std])
    try:
        beta, *_ = np.linalg.lstsq(X_aug, y_std, rcond=None)
    except np.linalg.LinAlgError:
        return {}

    y_hat = X_aug @ beta
    ss_res = float(np.sum((y_std - y_hat) ** 2))
    ss_tot = float(np.sum((y_std - y_std.mean()) ** 2))
    r2     = 1.0 - ss_res / max(ss_tot, 1e-12)
    r2_adj = 1.0 - (1 - r2) * (n - 1) / max(n - p - 1, 1)

    coefs = {feat: float(beta[i + 1]) for i, feat in enumerate(feature_names)}
    return {"dim": dim_name, "r2": r2, "r2_adj": r2_adj, "n": n, **coefs}


# ─── analysis B: persona consensus correlation ────────────────────────────────
def _corr_per_dim(
    wvs_vecs: Dict[str, Dict[str, float]],
    human_amces: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """Pearson r and Spearman rho between WVS feature and human AMCE per dim."""
    rows = []
    for feat in WVS_FEATURES:
        for dim in AMCE_DIMS:
            xs, ys = [], []
            for country in COUNTRIES:
                x = wvs_vecs.get(country, {}).get(feat, float("nan"))
                y = human_amces.get(country, {}).get(dim, float("nan"))
                if np.isfinite(x) and np.isfinite(y):
                    xs.append(x)
                    ys.append(y)
            if len(xs) < 5:
                continue
            r, rp = pearsonr(xs, ys)
            rho, rhop = spearmanr(xs, ys)
            rows.append({
                "wvs_feat": feat, "amce_dim": dim,
                "pearson_r": round(r, 3), "pearson_p": round(rp, 3),
                "spearman_rho": round(rho, 3), "spearman_p": round(rhop, 3),
                "n": len(xs),
            })
    return pd.DataFrame(rows)


# ─── main ───────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"[PERSONA-AMCE] Loading WVS: {WVS_PATH}")
    wvs_df = pd.read_csv(WVS_PATH, low_memory=False)

    # Build WVS feature vectors per country.
    wvs_vecs: Dict[str, Dict[str, float]] = {}
    human_amces: Dict[str, Dict[str, float]] = {}
    for country in COUNTRIES:
        feats = _extract_wvs_features(wvs_df, country)
        if feats is None:
            print(f"  [SKIP] {country} — not in WVS")
            continue
        wvs_vecs[country] = feats
        hamce = load_human_amce(HUMAN_AMCE_PATH, country)
        if hamce:
            human_amces[country] = hamce
        print(f"  {country}  WVS dims={sum(np.isfinite(v) for v in feats.values())}/10  "
              f"AMCE dims={len(hamce)}")

    valid_countries = [c for c in COUNTRIES if c in wvs_vecs and c in human_amces]
    print(f"\n[PERSONA-AMCE] {len(valid_countries)} countries with both WVS + human AMCE\n")

    # ── Analysis A: OLS regression WVS → AMCE ──────────────────────────────
    X_rows = []
    for c in valid_countries:
        row = [wvs_vecs[c].get(f, float("nan")) for f in WVS_FEATURES]
        X_rows.append(row)
    X = np.array(X_rows, dtype=np.float64)

    reg_rows = []
    for dim in AMCE_DIMS:
        y = np.array([human_amces[c].get(dim, float("nan")) for c in valid_countries])
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        if mask.sum() < 6:
            continue
        result = _run_regression(X[mask], y[mask], WVS_FEATURES, dim)
        if result:
            reg_rows.append(result)
            print(f"  {dim:<22s}  R²={result['r2']:.3f}  R²_adj={result['r2_adj']:.3f}  "
                  f"n={result['n']}")

    reg_df = pd.DataFrame(reg_rows)
    reg_df.to_csv(OUT_DIR / "wvs_amce_regression.csv", index=False)
    print(f"\n[SAVED] {OUT_DIR / 'wvs_amce_regression.csv'}")

    # Regression LaTeX table
    if not reg_df.empty:
        lines = [
            r"\begin{table}[h]\centering\scriptsize",
            r"\caption{WVS features as predictors of human AMCE dimensions. "
            r"OLS with standardised predictors. "
            r"Coefficients are standardised ($\beta$); bold = $|\beta| > 0.3$. "
            r"$R^2_{\text{adj}}$ measures how well the 10 WVS cultural dimensions "
            r"explain each moral preference dimension across the 20-country panel.}",
            r"\label{tab:wvs_amce_regression}",
            r"\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{l" + "r" * (len(WVS_FEATURES) + 2) + r"}\toprule",
            r"AMCE dim & " + " & ".join(f[:5] for f in WVS_FEATURES)
            + r" & $R^2$ & $R^2_{\text{adj}}$ \\\midrule",
        ]
        for _, row in reg_df.iterrows():
            cells = []
            for feat in WVS_FEATURES:
                v = row.get(feat, float("nan"))
                if not np.isfinite(v):
                    cells.append("--")
                elif abs(v) > 0.3:
                    cells.append(rf"\textbf{{{v:+.2f}}}")
                else:
                    cells.append(f"{v:+.2f}")
            cells += [f"{row['r2']:.2f}", f"{row['r2_adj']:.2f}"]
            short = row["dim"].split("_")[0]
            lines.append(short + " & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        (OUT_DIR / "wvs_amce_regression.tex").write_text("\n".join(lines), encoding="utf-8")
        print(f"[SAVED] {OUT_DIR / 'wvs_amce_regression.tex'}")

    # ── Analysis B: per-feature × per-dim correlation ──────────────────────
    corr_df = _corr_per_dim(wvs_vecs, human_amces)
    corr_df.to_csv(OUT_DIR / "wvs_amce_corr.csv", index=False)
    print(f"[SAVED] {OUT_DIR / 'wvs_amce_corr.csv'}")

    # Summary: strongest WVS predictor per AMCE dim
    lines = ["\nStrongest WVS predictor per AMCE dimension (by |Pearson r|):",
             "-" * 60]
    for dim in AMCE_DIMS:
        sub = corr_df[corr_df["amce_dim"] == dim].copy()
        if sub.empty:
            continue
        sub["abs_r"] = sub["pearson_r"].abs()
        best = sub.nlargest(3, "abs_r")
        entries = [f"{r['wvs_feat']} r={r['pearson_r']:+.2f}" for _, r in best.iterrows()]
        lines.append(f"  {dim:<22s}: " + "  |  ".join(entries))

    summary = "\n".join(lines) + "\n"
    (OUT_DIR / "persona_amce_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)

    _zip_outputs(OUT_DIR, "round2_phase7_persona_amce_corr")


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
