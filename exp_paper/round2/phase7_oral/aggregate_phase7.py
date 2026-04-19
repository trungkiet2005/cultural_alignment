#!/usr/bin/env python3
"""Phase 7 aggregator: merge per_dim_mis + persona_amce_corr + prism + latency
into paper-ready LaTeX tables and one summary text block.

No GPU needed — pure post-hoc over phase 7 CSVs.

Kaggle:
    !python exp_paper/round2/phase7_oral/aggregate_phase7.py

Env overrides:
    R2_RESULTS_BASE   root of results/exp24_round2/ (Kaggle default auto-set)
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

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Windows consoles default to cp1252; rebind stdout to UTF-8 so the
# summary text (which uses ², →, en-dash) prints without UnicodeEncodeError.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from exp_paper._r2_common import on_kaggle

R2_BASE = Path(os.environ.get(
    "R2_RESULTS_BASE",
    "/kaggle/working/cultural_alignment/results/exp24_round2"
    if on_kaggle()
    else "results/exp24_round2",
))
OUT_DIR = R2_BASE / "phase7_oral"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIMS = [
    "Species_Humans", "Gender_Female", "Age_Young",
    "Fitness_Fit", "SocialValue_High", "Utilitarianism_More",
]
DIM_SHORT = {
    "Species_Humans":    "Species",
    "Gender_Female":     "Gender",
    "Age_Young":         "Age",
    "Fitness_Fit":       "Fitness",
    "SocialValue_High":  "SocVal",
    "Utilitarianism_More": "Util",
}


def _load(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"[MISS] {path}")
        return None
    return pd.read_csv(path)


def _fmt(x: float, d: int = 3) -> str:
    if x is None or not np.isfinite(float(x)):
        return "--"
    return f"{float(x):.{d}f}"


# ─── section 1: per-dim macro summary ────────────────────────────────────────
def _section_per_dim() -> None:
    macro = _load(R2_BASE / "per_dim_mis" / "per_dim_macro.csv")
    if macro is None:
        return
    row = macro[macro["country"] == "MACRO"].iloc[0]

    lines = [
        r"\begin{tabular}{lcccc}\toprule",
        r"Dimension & Vanilla MAE & SWA-DPBR MAE & $\Delta$ MAE & Gain (\%) \\\midrule",
    ]
    for dim in DIMS:
        short = DIM_SHORT[dim]
        v = row.get(f"{dim}_van", float("nan"))
        s = row.get(f"{dim}_swa", float("nan"))
        d = row.get(f"{dim}_delta", float("nan"))
        p = row.get(f"{dim}_pct", float("nan"))
        pstr = f"{float(p):+.1f}" if np.isfinite(p) else "--"
        lines.append(f"{short} & {_fmt(v,3)} & {_fmt(s,3)} & {_fmt(d,3)} & {pstr} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT_DIR / "per_dim_macro_table.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] per_dim_macro_table.tex")


# ─── section 2: WVS regression R² summary ────────────────────────────────────
def _section_wvs_regression() -> None:
    reg = _load(R2_BASE / "persona_amce_corr" / "wvs_amce_regression.csv")
    if reg is None:
        return
    lines = [
        r"\begin{tabular}{lcc}\toprule",
        r"AMCE dimension & $R^2$ & $R^2_{\text{adj}}$ \\\midrule",
    ]
    for _, row in reg.iterrows():
        lines.append(
            f"{row['dim'].split('_')[0]} & {_fmt(row['r2'],3)} & {_fmt(row['r2_adj'],3)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT_DIR / "wvs_regression_r2.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wvs_regression_r2.tex")


# ─── section 3: PRISM vs SWA-DPBR macro comparison ───────────────────────────
def _section_prism() -> None:
    prism = _load(R2_BASE / "prism_baseline" / "prism_summary.csv")
    if prism is None:
        return
    if "mis" not in prism.columns:
        print(f"[SKIP] prism_summary.csv has no 'mis' column")
        return
    # ``mis`` may be all-NaN if the PRISM run aborted before any scenario
    # produced metrics. In that case skip the macro section gracefully.
    mis_numeric = pd.to_numeric(prism["mis"], errors="coerce")
    valid = prism[mis_numeric.notna() & np.isfinite(mis_numeric)]
    if valid.empty:
        print(f"[SKIP] prism_summary.csv has no valid MIS rows (all NaN — re-run prism_baseline)")
        return
    mean_mis  = float(valid["mis"].mean())
    mean_r    = float(valid["pearson_r"].mean(skipna=True))
    lines = [
        r"\begin{tabular}{lcc}\toprule",
        r"Method & Macro MIS $\downarrow$ & Macro $r$ $\uparrow$ \\\midrule",
        rf"Vanilla (no correction) & \multicolumn{{1}}{{c}}{{see Table~\ref{{tab:main}}}} & -- \\",
        rf"PRISM prompting (ours) & {mean_mis:.4f} & {mean_r:.3f} \\",
        rf"SWA-DPBR (ours) & \multicolumn{{1}}{{c}}{{see Table~\ref{{tab:main}}}} & -- \\",
        r"\bottomrule", r"\end{tabular}",
    ]
    (OUT_DIR / "prism_macro_compare.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] prism_macro_compare.tex  (PRISM mean MIS={mean_mis:.4f}  r={mean_r:.3f})")


# ─── section 4: latency table ────────────────────────────────────────────────
def _section_latency() -> None:
    lat = _load(R2_BASE / "latency" / "latency_summary.csv")
    if lat is None:
        lat = _load(R2_BASE / "latency" / "latency_raw.csv")
        if lat is None:
            return
    vanilla_sec = float(lat[lat["method"] == "vanilla"]["mean_sec_per_scen"].mean()) \
        if (lat["method"] == "vanilla").any() else float("nan")

    lines = [
        r"\begin{tabular}{llcc}\toprule",
        r"Method & $K$ & sec/scenario & Overhead \\\midrule",
    ]
    for _, row in lat.iterrows():
        ov = float(row["mean_sec_per_scen"]) / vanilla_sec \
            if np.isfinite(vanilla_sec) and vanilla_sec > 0 else float("nan")
        k_str = "--" if row["method"] == "vanilla" else str(int(row["K"]))
        ov_str = f"{ov:.2f}$\\times$" if np.isfinite(ov) else "--"
        lines.append(
            f"{row['method']} & {k_str} & {_fmt(row['mean_sec_per_scen'],3)} & {ov_str} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT_DIR / "latency_compact.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] latency_compact.tex")


# ─── master summary ──────────────────────────────────────────────────────────
def _section_summary() -> None:
    lines = ["Phase 7 (oral-level) summary numbers", "=" * 55]

    # per-dim top winner/loser
    macro = _load(R2_BASE / "per_dim_mis" / "per_dim_macro.csv")
    if macro is not None:
        m = macro[macro["country"] == "MACRO"].iloc[0]
        pcts = {dim: float(m.get(f"{dim}_pct", float("nan"))) for dim in DIMS}
        best = max(pcts, key=lambda d: pcts[d] if np.isfinite(pcts[d]) else -999)
        worst = min(pcts, key=lambda d: pcts[d] if np.isfinite(pcts[d]) else 999)
        lines.append(f"Per-dim best gain:  {DIM_SHORT[best]} ({pcts[best]:+.1f}%)")
        lines.append(f"Per-dim worst:      {DIM_SHORT[worst]} ({pcts[worst]:+.1f}%)")

    # WVS R² range
    reg = _load(R2_BASE / "persona_amce_corr" / "wvs_amce_regression.csv")
    if reg is not None and not reg.empty:
        lines.append(
            f"WVS → AMCE R² range: {float(reg['r2'].min()):.2f} – {float(reg['r2'].max()):.2f}"
        )

    # PRISM
    prism = _load(R2_BASE / "prism_baseline" / "prism_summary.csv")
    if prism is not None:
        valid = prism[prism["mis"].apply(lambda x: np.isfinite(float(x)) if pd.notna(x) else False)]
        if not valid.empty:
            lines.append(f"PRISM macro MIS:    {float(valid['mis'].mean()):.4f}")

    summary = "\n".join(lines) + "\n"
    (OUT_DIR / "phase7_summary.txt").write_text(summary, encoding="utf-8")
    print(f"\n[DONE] {OUT_DIR}\n\n" + summary)


# ─── zip ────────────────────────────────────────────────────────────────────
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


# ─── main ───────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"[PHASE7-AGG] reading from {R2_BASE}\n"
          f"[PHASE7-AGG] writing to   {OUT_DIR}\n")
    _section_per_dim()
    _section_wvs_regression()
    _section_prism()
    _section_latency()
    _section_summary()
    _zip_outputs(OUT_DIR, "round2_phase7_oral_aggregated")


if __name__ == "__main__":
    main()
