#!/usr/bin/env python3
"""Emit LaTeX for per-country MIS + SWA-DPBR JSD/r from exp_paper/tracker.md.

Generates:
  _generated_percountry_main.tex  – paper checkpoints (see FULL_CFG; currently one wide table)
  _generated_percountry_app.tex    – empty placeholder if all models fit in main text
  _generated_percountry_all.tex    – combined parts (for convenience)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

TRACKER = Path(__file__).resolve().parents[1] / "tracker.md"

# Paper table: six checkpoints with strong macro MIS gains in the 20-country sweep
# (see exp_paper/Paper_New/SWA_DPBR/paper_revised.tex). Ordered by *nominal* parameter
# count descending (not by gain). Full 16-model grid remains in tracker.md.
FULL_CFG: List[Tuple[str, str, str]] = [
    ("EXP-24-LLAMA33_70B",              "Llama-3.3-70B",                  "Llama-3.3-70B"),
    ("EXP-24-MAGISTRAL_SMALL_2509",     "Magistral-Small-2509",           "Magistral-Sml (24B)"),
    ("EXP-24-PHI_4",                    "Phi-4",                          "Phi-4 (14B)"),
    ("EXP-24-QWEN3_VL_8B",              "Qwen3-VL-8B",                    "Qwen3-VL-8B"),
    ("EXP-24-HF_QWEN25_7B_BF16",        "Qwen2.5-7B",                     "Qwen2.5-7B"),
    ("EXP-24-PHI35_MINI",               "Phi-3.5-mini",                   "Phi-3.5-mini (3.8B)"),
]

# One wide table in the main paper (six models).
CHUNK = 6

# Geographic region for each ISO-3 country code in the 20-country panel.
# Sorted display order follows REGION_ORDER below.
REGION: Dict[str, str] = {
    "ARG": "Americas",
    "BRA": "Americas",
    "COL": "Americas",
    "MEX": "Americas",
    "USA": "Americas",
    "DEU": "Europe",
    "GBR": "Europe",
    "ROU": "Europe",
    "SRB": "Europe",
    "CHN": "E.~Asia",
    "JPN": "E.~Asia",
    "IDN": "SE.~Asia",
    "MMR": "SE.~Asia",
    "MYS": "SE.~Asia",
    "THA": "SE.~Asia",
    "VNM": "SE.~Asia",
    "BGD": "S.~Asia",
    "KGZ": "C.~Asia",
    "IRN": "W.~Asia",
    "ETH": "Africa",
}

REGION_ORDER = ["Americas", "Europe", "E.~Asia", "SE.~Asia", "S.~Asia", "C.~Asia", "W.~Asia", "Africa"]


def _region_sort_key(iso: str) -> Tuple[int, str]:
    reg = REGION.get(iso, "ZZZ")
    try:
        ri = REGION_ORDER.index(reg)
    except ValueError:
        ri = 99
    return (ri, iso)


@dataclass
class Row:
    mis_v: float
    mis_s: float
    mis_pct: float
    jsd_s: float
    r_s: float


def _f(s: str) -> float:
    return float(s.strip().replace("+", ""))


def _pct(s: str) -> float:
    m = re.search(r"([+-]?\d+(?:\.\d+)?)\s*%", s.replace("*", ""))
    if not m:
        return float("nan")
    return float(m.group(1))


def _block(text: str, title: str) -> str:
    i = text.index(title)
    j = text.find("\n####", i + 1)
    if j < 0:
        j = len(text)
    return text[i:j]


def _full_metrics(block: str, model_col: str) -> Dict[str, Tuple[float, float, float]]:
    """country -> MIS_s, JSD_s, r_s"""
    out: Dict[str, Tuple[float, float, float]] = {}
    for line in block.splitlines():
        if not line.strip().startswith("|"):
            continue
        p = [x.strip() for x in line.split("|")]
        if len(p) < 7:
            continue
        if p[1] != model_col or p[2] == "Country":
            continue
        cty = p[2]
        if not re.match(r"^[A-Z]{3}$", cty):
            continue
        mis_s = _f(p[3])
        jsd_s = _f(p[4])
        rr = p[5].replace("+", "").strip()
        r_s = _f(rr) if rr else float("nan")
        out[cty] = (mis_s, jsd_s, r_s)
    return out


def _vs_vanilla(block: str, model_col: str) -> Dict[str, Tuple[float, float, float]]:
    """country -> MIS_v, MIS_s, mis_pct"""
    out: Dict[str, Tuple[float, float, float]] = {}
    for line in block.splitlines():
        if not line.strip().startswith("|"):
            continue
        p = [x.strip() for x in line.split("|")]
        if len(p) < 8:
            continue
        if p[1] != model_col or p[2] == "Country":
            continue
        cty = p[2]
        if not re.match(r"^[A-Z]{3}$", cty):
            continue
        mis_v = _f(p[3])
        mis_s = _f(p[4])
        pct = _pct(p[6])
        out[cty] = (mis_v, mis_s, pct)
    return out


def _merge(
    run: str,
    model_col: str,
    text: str,
    countries: List[str],
) -> Dict[str, Row]:
    fm = _full_metrics(_block(text, f"#### {run} Full Metrics"), model_col)
    vm = _vs_vanilla(_block(text, f"#### {run} vs Vanilla (MIS)"), model_col)
    rows: Dict[str, Row] = {}
    for c in countries:
        if c not in fm or c not in vm:
            raise KeyError(f"{run} missing {c}: full={c in fm} van={c in vm}")
        mis_s1, jsd, rr = fm[c]
        mis_v, mis_s2, pct = vm[c]
        if abs(mis_s1 - mis_s2) > 0.001:
            raise ValueError(f"{run} {c} MIS mismatch {mis_s1} {mis_s2}")
        rows[c] = Row(mis_v, mis_s2, pct, jsd, rr)
    return rows


def _fmt_small(x: float) -> str:
    s = f"{x:.3f}"
    if x < 0:
        return "$-$" + s[1:]
    if s.startswith("0."):
        return s[1:]
    return s


def _fmt_pct_tex(x: float) -> str:
    if x != x:
        return "---"
    if x >= 0:
        return f"\\gain{{+{x:.1f}}}"
    return f"\\loss{{{x:.1f}}}"


def _emit_table(
    cfg: List[Tuple[str, str, str]],
    merged: Dict[str, Dict[str, Row]],
    countries: List[str],
    part: int,
    n_parts: int,
    float_spec: str = "t",
) -> str:
    n_models = len(cfg)
    lines: List[str] = []
    lines.append(rf"\begin{{table*}}[{float_spec}]")
    if n_parts > 1:
        title = (
            r"\textbf{Per-country SWA-DPBR results (20 countries), "
            + f"part {part} of {n_parts}."
            + r"} "
        )
    else:
        title = r"\textbf{Per-country SWA-DPBR results (20 countries).} "
    cap = (
        r"\caption{"
        + title
        + r"Countries are grouped by geographic region (shaded rows = region boundary). "
        r"Vanilla and SWA-DPBR MIS ($\downarrow$), relative MIS improvement (\%), "
        r"and SWA-DPBR JSD ($\downarrow$) / Pearson $r$ ($\uparrow$). "
        r"Models in the same order as Table~\ref{tab:main_macro_summary} "
        r"(nominal parameter count, descending).}"
    )
    lines.append(cap)
    label = f"tab:percountry_p{part}"
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\centering\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{1.5pt}")
    lines.append(r"\resizebox{\textwidth}{!}{%")

    # 2 label cols (ISO + Region) + 5 data cols per model
    col_spec = "@{}ll" + " ccccc" * n_models + "@{}"
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Top header: blank (ISO), blank (Region), then multicolumn per model
    # Model i occupies cols 3+5i .. 7+5i  (1-indexed)
    hdr_cells = ["", ""]
    for _, _, h in cfg:
        hdr_cells.append(f"\\multicolumn{{5}}{{c}}{{\\textbf{{{h}}}}}")
    lines.append(" & ".join(hdr_cells) + r" \\")

    # Cmidrule: starts at col 3, width 5 per model
    cmid = "".join(
        rf"\cmidrule(lr){{{3 + 5 * i}-{7 + 5 * i}}}" for i in range(n_models)
    )
    lines.append(cmid)

    # Sub-header
    sub_cells = [r"\textbf{ISO}", r"\textbf{Region}"]
    for _ in cfg:
        sub_cells.extend([
            r"MIS$_v$",
            r"MIS$_s$",
            r"\%$\Delta_{\mathrm{MIS}}$",
            r"JSD$_s$",
            r"$r_s$",
        ])
    lines.append(" & ".join(sub_cells) + r" \\")
    lines.append(r"\midrule")

    # Data rows, inserting \midrule between regions
    prev_region: Optional[str] = None
    for c in countries:
        reg = REGION.get(c, "")
        if prev_region is not None and reg != prev_region:
            lines.append(r"\midrule")
        prev_region = reg

        cells = [c, reg]
        for run, _, _ in cfg:
            r = merged[run][c]
            cells.append(_fmt_small(r.mis_v))
            cells.append(_fmt_small(r.mis_s))
            cells.append(_fmt_pct_tex(r.mis_pct))
            cells.append(_fmt_small(r.jsd_s))
            cells.append(_fmt_small(r.r_s))
        lines.append(" & ".join(cells) + r" \\")

    # Mean row
    lines.append(r"\midrule")
    mean_cells: List[str] = [r"\textbf{Mean}", ""]
    for run, _, _ in cfg:
        rs = list(merged[run].values())
        mv = sum(x.mis_v for x in rs) / len(rs)
        ms = sum(x.mis_s for x in rs) / len(rs)
        mp = sum(x.mis_pct for x in rs) / len(rs)
        jd = sum(x.jsd_s for x in rs) / len(rs)
        rr = sum(x.r_s for x in rs) / len(rs)
        mean_cells.append(f"\\textbf{{{_fmt_small(mv)}}}")
        mean_cells.append(f"\\textbf{{{_fmt_small(ms)}}}")
        mean_cells.append(f"\\textbf{{{_fmt_pct_tex(mp)}}}")
        mean_cells.append(f"\\textbf{{{_fmt_small(jd)}}}")
        mean_cells.append(f"\\textbf{{{_fmt_small(rr)}}}")
    lines.append(" & ".join(mean_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(
        r"\vspace{3pt}\footnotesize \textbf{Legend.} "
        r"MIS$_v$/MIS$_s$: vanilla vs.\ SWA-DPBR $\ell_2$ misalignment; "
        r"\%$\Delta_{\mathrm{MIS}}$: relative improvement in MIS; "
        r"JSD$_s$ and $r_s$: Jensen--Shannon distance and Pearson correlation for SWA-DPBR only. "
        r"\textbf{Region groups:} Americas (ARG, BRA, COL, MEX, USA); "
        r"Europe (DEU, GBR, ROU, SRB); "
        r"E.\,Asia (CHN, JPN); SE.\,Asia (IDN, MMR, MYS, THA, VNM); "
        r"S.\,Asia (BGD); C.\,Asia (KGZ); W.\,Asia (IRN); Africa (ETH)."
    )
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def main() -> None:
    text = TRACKER.read_text(encoding="utf-8")
    ref_run, ref_col, _ = FULL_CFG[0]
    raw_countries = sorted(
        _full_metrics(_block(text, f"#### {ref_run} Full Metrics"), ref_col).keys()
    )
    # Sort by region then alphabetically within region
    countries = sorted(raw_countries, key=_region_sort_key)

    merged: Dict[str, Dict[str, Row]] = {}
    for run, mc, _ in FULL_CFG:
        merged[run] = _merge(run, mc, text, countries)

    chunks: List[List[Tuple[str, str, str]]] = [
        FULL_CFG[i: i + CHUNK] for i in range(0, len(FULL_CFG), CHUNK)
    ]
    n_parts = len(chunks)

    out_dir = Path(__file__).resolve().parents[1] / "Paper_New" / "SWA_DPBR"

    # Parts 1-2: go into main paper body
    main_parts = [
        _emit_table(chunks[i], merged, countries, i + 1, n_parts, float_spec="tp")
        for i in range(min(2, n_parts))
    ]
    main_file = out_dir / "_generated_percountry_main.tex"
    main_file.write_text("".join(main_parts), encoding="utf-8")
    print(f"Wrote {main_file}")

    # Parts 3-4: go into appendix (empty when all models fit in main text)
    app_parts = [
        _emit_table(chunks[i], merged, countries, i + 1, n_parts, float_spec="tp")
        for i in range(2, n_parts)
    ]
    app_file = out_dir / "_generated_percountry_app.tex"
    if app_parts:
        app_file.write_text("".join(app_parts), encoding="utf-8")
    else:
        app_file.write_text(
            "% All per-country columns appear in the main paper (_generated_percountry_main.tex).\n",
            encoding="utf-8",
        )
    print(f"Wrote {app_file}")

    # Keep the combined file for backwards compat
    all_parts = [
        _emit_table(chunks[i], merged, countries, i + 1, n_parts, float_spec="tp")
        for i in range(n_parts)
    ]
    legacy = out_dir / "_generated_percountry_all.tex"
    legacy.write_text("".join(all_parts), encoding="utf-8")
    print(f"Wrote {legacy}")


if __name__ == "__main__":
    main()
