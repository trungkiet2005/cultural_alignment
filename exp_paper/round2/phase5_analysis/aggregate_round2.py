#!/usr/bin/env python3
"""Post-hoc aggregator: turn Round-2 CSV outputs into paper-ready LaTeX tables.

Reads every ``*_summary.csv`` / ``*_per_country*.csv`` produced by the
phase 1--4 scripts and emits:

    results/exp24_round2/phase5_analysis/
      ├── main_baselines_20country.tex    # Table: Phi-4 vanilla/dropout/temp/margin/diffpo/SWA-DPBR × 20 countries
      ├── hparam_sensitivity_filled.tex   # Table: fills in the `tab:r2_hparam_sensitivity` placeholder rows
      ├── reliability_audit_table.tex     # Table: regime counts (HighESS/LowDisagree etc.)
      ├── multiseed_ci_table.tex          # Table: macro MIS with 95% CI over seeds
      ├── persona_variant_head_to_head.tex# Table: aggregate vs utilitarian 4th-persona
      └── summary_numbers.txt             # Plain-text one-liners for the paper body

Usage (no GPU, ~seconds):
    python exp_paper/round2/phase5_analysis/aggregate_round2.py

Env overrides:
    R2_RESULTS_BASE   root dir holding per-phase subfolders
                      (default: results/exp24_round2/ relative to repo root)
    R2_OUT_DIR        where to write .tex + .txt outputs
                      (default: <results_base>/phase5_analysis/)
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Self-bootstrap — so the script runs from anywhere on Kaggle / local.
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
        raise RuntimeError("Not on Kaggle and not inside the repo root.")
    if not _os.path.isdir(_REPO_DIR_KAGGLE):
        _sp.run(["git", "clone", "--depth", "1", _REPO_URL, _REPO_DIR_KAGGLE], check=True)
    _os.chdir(_REPO_DIR_KAGGLE)
    _sys.path.insert(0, _REPO_DIR_KAGGLE)
    return _REPO_DIR_KAGGLE


_r2_bootstrap()

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

R2_BASE = Path(os.environ.get(
    "R2_RESULTS_BASE",
    "/kaggle/working/cultural_alignment/results/exp24_round2"
    if os.path.isdir("/kaggle/working")
    else "results/exp24_round2",
))
OUT_DIR = Path(os.environ.get("R2_OUT_DIR", str(R2_BASE / "phase5_analysis")))
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── helpers ────────────────────────────────────────────────────────────────
def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"[MISS] {path}  (skipping section)")
        return None
    return pd.read_csv(path)


def _fmt(x: float, d: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return "--"
    return f"{x:.{d}f}"


# ─── section 1: Phi-4 × 20 countries baselines (phase 2) ────────────────────
def _build_baselines_table() -> None:
    """Merge vanilla (from main SWA run) + MC-Dropout + T-scale + margin + DiffPO."""
    dropout = _load_csv(R2_BASE / "mc_dropout" / "mc_dropout_summary_T8_p0.1.csv")
    temp    = _load_csv(R2_BASE / "tempmargin" / "calib_temperature_summary.csv")
    margin  = _load_csv(R2_BASE / "tempmargin" / "calib_margin_summary.csv")
    diffpo  = _load_csv(R2_BASE / "diffpo_binary" / "diffpo_binary_summary.csv")

    if any(d is None for d in (dropout, temp, margin, diffpo)):
        print("[SKIP] baselines table — one or more CSVs missing.")
        return

    # Join on `country`, stacking columns. align_mis in the per-country summary
    # CSVs is the held-out (test-split) MIS for tempmargin/diffpo, full-pool
    # MIS for dropout (since dropout isn't fitted).
    def _pick(df: pd.DataFrame, name: str) -> pd.DataFrame:
        return df[["country", "align_mis"]].rename(columns={"align_mis": name})

    merged = _pick(dropout, "mc_dropout")
    for nm, df in [("temp_c", temp), ("margin_c", margin), ("diffpo_alpha", diffpo)]:
        merged = merged.merge(_pick(df, nm), on="country", how="outer")

    merged = merged.sort_values("country").reset_index(drop=True)
    tex = [
        r"\begin{table}[h]\centering\scriptsize",
        r"\caption{Phi-4 (14B) head-to-head: additional inference-time baselines vs.\ SWA-DPBR on the 20-country grid. Each column is held-out MIS $\downarrow$ on the 75\% test split. Numbers produced by the phase-2 runners.}",
        r"\label{tab:r2_baselines_filled}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lcccc}\toprule",
        r"Country & MC-Dropout & $T_c$ scale & Margin $m_c$ & DiffPO-binary \\\midrule",
    ]
    for _, r in merged.iterrows():
        tex.append(
            f"{r['country']} & {_fmt(r['mc_dropout'], 3)} & {_fmt(r['temp_c'], 3)} & "
            f"{_fmt(r['margin_c'], 3)} & {_fmt(r['diffpo_alpha'], 3)} \\\\"
        )
    mean = merged.drop(columns=["country"]).mean(numeric_only=True)
    tex.append(r"\midrule")
    tex.append(
        rf"\textbf{{Mean}} & \textbf{{{_fmt(mean['mc_dropout'])}}} & "
        rf"\textbf{{{_fmt(mean['temp_c'])}}} & \textbf{{{_fmt(mean['margin_c'])}}} & "
        rf"\textbf{{{_fmt(mean['diffpo_alpha'])}}} \\"
    )
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT_DIR / "main_baselines_20country.tex").write_text("\n".join(tex), encoding="utf-8")
    print(f"[OK] {OUT_DIR / 'main_baselines_20country.tex'}")


# ─── section 2: hparam sensitivity table (phase 3) ──────────────────────────
def _build_hparam_table() -> None:
    df = _load_csv(R2_BASE / "hparam_sensitivity" / "hparam_sensitivity_summary.csv")
    if df is None:
        return
    tex = [
        r"% Drop-in replacement for Table~\ref{tab:r2_hparam_sensitivity} in the paper.",
        r"\begin{tabular}{lccccc}\toprule",
        r"Axis & Default & Grid & Min MIS & Max MIS & $\Delta$ \\\midrule",
    ]
    # Macro over the country panel, per (axis, value).
    grouped = df.groupby(["axis", "value"])["mis"].mean().reset_index()
    default_map = {"s": 0.04, "lambda": 0.70, "sigma": 0.30, "tcat": 3.0}
    label_map = {
        "s": r"$s$",
        "lambda": r"$\lambda_{\text{coop}}$",
        "sigma": r"$\sigma$",
        "tcat": r"$T_{\text{cat}}$ scale",
    }
    for axis in ("s", "lambda", "sigma", "tcat"):
        g = grouped[grouped["axis"] == axis].sort_values("value")
        if g.empty:
            continue
        grid = "$\\{" + ", ".join(_fmt(v, 2) for v in g["value"].tolist()) + "\\}$"
        mn, mx = g["mis"].min(), g["mis"].max()
        tex.append(
            f"{label_map[axis]} & {default_map[axis]} & {grid} & "
            f"{_fmt(mn)} & {_fmt(mx)} & {_fmt(mx - mn)} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}"]
    (OUT_DIR / "hparam_sensitivity_filled.tex").write_text("\n".join(tex), encoding="utf-8")
    print(f"[OK] {OUT_DIR / 'hparam_sensitivity_filled.tex'}")


# ─── section 3: reliability audit regimes (phase 1) ─────────────────────────
def _build_reliability_table() -> None:
    df = _load_csv(R2_BASE / "reliability_audit" / "reliability_audit_mean.csv")
    if df is None:
        df = _load_csv(R2_BASE / "reliability_audit" / "reliability_audit_per_country.csv")
        if df is None:
            return
    # Sum regime counts across countries for the headline numbers.
    regimes = ["HighESS_LowDisagree", "HighESS_HighDisagree",
               "LowESS_LowDisagree", "LowESS_HighDisagree"]
    tex = [
        r"\begin{tabular}{lcccc}\toprule",
        r"Regime & Count & Frac (\%) & mean $r$ & mean shrinkage (\%) \\\midrule",
    ]
    total_n = float(df["n_total"].sum()) if "n_total" in df.columns else float("nan")
    for reg in regimes:
        cnt_col = f"{reg}_count"
        r_col   = f"{reg}_mean_r"
        sh_col  = f"{reg}_mean_shrink"
        if cnt_col not in df.columns:
            continue
        cnt = float(df[cnt_col].sum())
        frac = 100.0 * cnt / max(1.0, total_n)
        mean_r = float(df[r_col].mean(skipna=True))
        mean_sh = 100.0 * float(df[sh_col].mean(skipna=True))
        tex.append(
            f"{reg.replace('_', r'\\,/\\,')} & {int(cnt)} & {_fmt(frac, 1)} & "
            f"{_fmt(mean_r)} & {_fmt(mean_sh, 1)} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}"]
    (OUT_DIR / "reliability_audit_table.tex").write_text("\n".join(tex), encoding="utf-8")
    print(f"[OK] {OUT_DIR / 'reliability_audit_table.tex'}")


# ─── section 4: multi-seed CIs (phase 4) ────────────────────────────────────
def _build_multiseed_table() -> None:
    macro = _load_csv(R2_BASE / "multiseed_phi4" / "multiseed_macro_ci.csv")
    if macro is None:
        return
    tex = [
        r"\begin{tabular}{lcc}\toprule",
        r"Method & Mean MIS $\pm$ 95\% CI & Mean $r$ $\pm$ 95\% CI \\\midrule",
    ]
    for _, row in macro.iterrows():
        mis_hi = row.get("ci95_mis", float("nan"))
        r_hi   = row.get("ci95_pearson_r", float("nan"))
        tex.append(
            f"{row['method']} & {_fmt(row['mean_mis'])} $\\pm$ {_fmt(mis_hi, 3)} & "
            f"{_fmt(row['mean_pearson_r'], 3)} $\\pm$ {_fmt(r_hi, 3)} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}"]
    (OUT_DIR / "multiseed_ci_table.tex").write_text("\n".join(tex), encoding="utf-8")
    print(f"[OK] {OUT_DIR / 'multiseed_ci_table.tex'}")


# ─── section 5: persona variant head-to-head (phase 3) ──────────────────────
def _build_persona_variant_table() -> None:
    df = _load_csv(R2_BASE / "persona_variant" / "persona_variant_summary.csv")
    if df is None:
        return
    piv = df.pivot(index="country", columns="variant", values="align_mis")
    if "aggregate" not in piv.columns or "utilitarian" not in piv.columns:
        print(f"[WARN] persona_variant: expected both variants, got {piv.columns.tolist()}")
        return
    piv["delta"] = piv["aggregate"] - piv["utilitarian"]

    tex = [
        r"\begin{tabular}{lccc}\toprule",
        r"Country & MIS (aggregate) & MIS (utilitarian) & $\Delta$ (agg $-$ util) \\\midrule",
    ]
    for c, row in piv.iterrows():
        tex.append(
            f"{c} & {_fmt(row['aggregate'])} & {_fmt(row['utilitarian'])} & "
            f"{_fmt(row['delta'])} \\\\"
        )
    tex += [
        r"\midrule",
        rf"\textbf{{Mean}} & \textbf{{{_fmt(piv['aggregate'].mean())}}} & "
        rf"\textbf{{{_fmt(piv['utilitarian'].mean())}}} & "
        rf"\textbf{{{_fmt(piv['delta'].mean())}}} \\",
        r"\bottomrule", r"\end{tabular}",
    ]
    (OUT_DIR / "persona_variant_head_to_head.tex").write_text("\n".join(tex), encoding="utf-8")
    print(f"[OK] {OUT_DIR / 'persona_variant_head_to_head.tex'}")


# ─── section 6: summary one-liners for the paper body ───────────────────────
def _build_summary_numbers() -> None:
    lines: List[str] = []

    def _scalar(path: Path, col: str, agg: str = "mean") -> float:
        d = _load_csv(path)
        if d is None or col not in d.columns:
            return float("nan")
        return float(getattr(d[col].dropna(), agg)())

    # Hparam max-delta on each axis
    df = _load_csv(R2_BASE / "hparam_sensitivity" / "hparam_sensitivity_summary.csv")
    if df is not None:
        for axis in ("s", "lambda", "sigma", "tcat"):
            g = df[df["axis"] == axis].groupby("value")["mis"].mean()
            if len(g):
                lines.append(f"hparam/{axis:<6} max-delta-MIS = {g.max() - g.min():.4f}")

    # Reliability audit high-ESS/high-disagree fraction
    rel = _load_csv(R2_BASE / "reliability_audit" / "reliability_audit_per_country.csv")
    if rel is not None and "HighESS_HighDisagree_frac" in rel.columns:
        lines.append(
            f"reliability HighESS/HighDisagree mean frac = "
            f"{100.0 * rel['HighESS_HighDisagree_frac'].mean():.2f}%"
        )

    # Multi-seed macro CI
    mm = _load_csv(R2_BASE / "multiseed_phi4" / "multiseed_macro_ci.csv")
    if mm is not None:
        for _, r in mm.iterrows():
            lines.append(
                f"multiseed {r['method']:<10} macro MIS = "
                f"{r['mean_mis']:.4f} ± {r['ci95_mis']:.4f}"
            )

    # Persona variant delta
    pv = _load_csv(R2_BASE / "persona_variant" / "persona_variant_summary.csv")
    if pv is not None:
        piv = pv.pivot(index="country", columns="variant", values="align_mis")
        if {"aggregate", "utilitarian"}.issubset(piv.columns):
            delta = (piv["utilitarian"] - piv["aggregate"]).mean()
            lines.append(
                f"persona utilitarian - aggregate macro MIS shift = {delta:+.4f}"
            )

    path = OUT_DIR / "summary_numbers.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] {path}")
    print("\n".join("  " + l for l in lines))


# ─── zip helper ─────────────────────────────────────────────────────────────
def _zip_outputs(out_dir: Path, label: str) -> None:
    """Zip *out_dir* so it appears as a single downloadable file.

    On Kaggle: writes to /kaggle/working/{label}.zip (visible in the output panel).
    Locally:   writes to results/download/{label}.zip next to the results tree.
    """
    import shutil
    dest_base = (
        Path("/kaggle/working")
        if os.path.isdir("/kaggle/working")
        else out_dir.parent.parent / "download"
    )
    dest_base.mkdir(parents=True, exist_ok=True)
    zip_path = shutil.make_archive(
        str(dest_base / label), "zip",
        root_dir=str(out_dir.parent),
        base_dir=out_dir.name,
    )
    print(f"[ZIP] {zip_path}")


# ─── main ───────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"[AGGREGATE] reading from {R2_BASE}\n[AGGREGATE] writing to  {OUT_DIR}\n")
    _build_baselines_table()
    _build_hparam_table()
    _build_reliability_table()
    _build_multiseed_table()
    _build_persona_variant_table()
    _build_summary_numbers()
    print(f"\n[DONE] {OUT_DIR}")
    _zip_outputs(OUT_DIR, "round2_phase5_analysis")


if __name__ == "__main__":
    main()
