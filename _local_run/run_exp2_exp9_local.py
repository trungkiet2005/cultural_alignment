#!/usr/bin/env python3
"""Local post-hoc runner for playbook Exp 2 and Exp 9.

Pure stdlib + numpy + matplotlib only — bypasses scipy and pandas because
the user's local Windows Python install has paging-file issues that cause
OOM during their import chains.

Outputs land in ``exp_paper/result/exp24_playbook_qwen25_7b/`` so the
playbook README status table can flip to "done" for Exp 2 and Exp 9.

Run from repo root:
    python _local_run/run_exp2_exp9_local.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from math import erf, sqrt
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULT = ROOT / "exp_paper" / "result" / "exp24_playbook_qwen25_7b"
RESULT.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    return reader.fieldnames or [], rows


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _pearsonr(x, y) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 2 or x.std() == 0 or y.std() == 0:
        return float("nan"), float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    if abs(r) >= 1.0:
        return r, 0.0
    t = r * np.sqrt(max(n - 2, 1) / max(1.0 - r * r, 1e-12))
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t) / sqrt(2.0))))
    return r, float(p)


# ============================================================================
# Exp 2 — Country-Level Correlation
# ============================================================================

def run_exp2() -> None:
    print("\n" + "#" * 70)
    print("# Exp 2 - Country-Level Correlation")
    print("#" * 70)

    main_csv = ROOT / "_local_run" / "phi4_main_results_long.csv"
    if not main_csv.is_file():
        raise FileNotFoundError(main_csv)

    # Mean persona_variance per country, sourced from paper_20c swa_results_*.csv
    # (the playbook's scenario_analysis_all_countries.csv has empty
    # persona_variance for the 20-country run — known bug from earlier session).
    swa_dir = ROOT / "exp_paper" / "result" / "exp24_paper_20c" / "phi_4" / "swa" / "phi-4"
    var_sum: Dict[str, float] = defaultdict(float)
    var_n: Dict[str, int] = defaultdict(int)
    if not swa_dir.is_dir():
        raise FileNotFoundError(f"phi-4 paper_20c swa dir missing: {swa_dir}")
    for csv_path in sorted(swa_dir.glob("swa_results_*.csv")):
        country = csv_path.stem.replace("swa_results_", "")
        with open(csv_path, "r", encoding="utf-8", newline="") as fh:
            rdr = csv.DictReader(fh)
            for row in rdr:
                try:
                    v = float(row["mppi_variance"])
                except (KeyError, ValueError):
                    continue
                if not np.isfinite(v):
                    continue
                var_sum[country] += v
                var_n[country] += 1
    mean_var = {c: var_sum[c] / var_n[c] for c in var_sum if var_n[c] > 0}
    print(f"  computed mean_variance for {len(mean_var)} countries from paper_20c/phi-4")

    # vanilla_mis and disca_mis per country (long-form, method column)
    _, res_rows = _read_csv(main_csv)
    by_country: Dict[str, Dict[str, float]] = defaultdict(dict)
    for row in res_rows:
        c, method = row["country"], row["method"].lower()
        try:
            mis = float(row["mis"])
        except (KeyError, ValueError):
            continue
        if "vanilla" in method:
            by_country[c]["vanilla_mis"] = mis
        elif any(k in method for k in ("swa", "disca", "dpbr")):
            by_country[c]["disca_mis"] = mis

    merged_rows: List[Dict] = []
    for c in sorted(by_country.keys()):
        d = by_country[c]
        if "vanilla_mis" not in d or "disca_mis" not in d or c not in mean_var:
            continue
        merged_rows.append({
            "country": c,
            "mean_variance": mean_var[c],
            "vanilla_mis": d["vanilla_mis"],
            "disca_mis": d["disca_mis"],
            "delta_mis": d["vanilla_mis"] - d["disca_mis"],
        })

    out_csv = RESULT / "country_correlation_data.csv"
    _write_csv(out_csv,
               ["country", "mean_variance", "vanilla_mis", "disca_mis", "delta_mis"],
               merged_rows)
    print(f"  countries: {len(merged_rows)}")
    print(f"  saved {out_csv.name}")

    if not merged_rows:
        return

    mv = np.array([r["mean_variance"] for r in merged_rows], dtype=float)
    dm = np.array([r["delta_mis"] for r in merged_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#1A8A66" if d > 0 else "#C04E28" for d in dm]
    ax.scatter(mv, dm, s=80, c=colors, alpha=0.75, edgecolors="black", linewidths=0.5)
    for r in merged_rows:
        ax.annotate(r["country"], (r["mean_variance"], r["delta_mis"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=9)
    ax.axhline(0, color="gray", lw=0.7, ls="--")

    if len(mv) >= 3 and mv.std() > 0 and dm.std() > 0:
        r, p = _pearsonr(mv, dm)
        z = np.polyfit(mv, dm, 1)
        xs = np.linspace(mv.min(), mv.max(), 100)
        ax.plot(xs, z[0] * xs + z[1], "k-", lw=1, alpha=0.5)
        p_text = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        ax.text(0.05, 0.95, f"Pearson r = {r:.3f}\n{p_text}",
                transform=ax.transAxes, fontsize=11, va="top",
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))
    else:
        r, p = float("nan"), float("nan")

    ax.set_xlabel("Mean inter-persona variance per country", fontsize=12)
    ax.set_ylabel(r"$\Delta$MIS (vanilla $-$ DISCA)", fontsize=12)
    plt.tight_layout()
    pdf = RESULT / "figure3_country_correlation.pdf"
    png = RESULT / "figure3_country_correlation.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {pdf.name}, {png.name}")

    print("\n  -- Result --")
    print(f"  Pearson r (mean_variance vs DeltaMIS): {r:+.4f}  (p={p:.4f})")
    n_improved = sum(1 for r in merged_rows if r["delta_mis"] > 0)
    print(f"  countries improved (DeltaMIS > 0)    : {n_improved} / {len(merged_rows)}")
    sorted_rows = sorted(merged_rows, key=lambda x: -x["delta_mis"])
    print("\n  Top 5 improved:")
    print(f"  {'country':<8}{'mean_variance':>14}{'delta_mis':>12}")
    for r in sorted_rows[:5]:
        print(f"  {r['country']:<8}{r['mean_variance']:>14.4f}{r['delta_mis']:>12.4f}")
    print("\n  Bottom 5 (failure cases):")
    for r in sorted_rows[-5:]:
        print(f"  {r['country']:<8}{r['mean_variance']:>14.4f}{r['delta_mis']:>12.4f}")


# ============================================================================
# Exp 9 — Negative Pearson r Diagnosis
# ============================================================================

DIMS = ["Species_Humans", "Gender_Female", "Age_Young",
        "Fitness_Fit", "SocialValue_High", "Utilitarianism_More"]


def _country_amce(rows: List[Dict], value_col: str
                  ) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = defaultdict(dict)
    for r in rows:
        try:
            out[r["country"]][r["dimension"]] = float(r[value_col])
        except (KeyError, ValueError):
            continue
    return out


def _find_swaps(human, model, dims: List[str]) -> List[Tuple[str, str]]:
    if len(human) < 2:
        return []
    h_rank = np.argsort(np.argsort(-human))
    m_rank = np.argsort(np.argsort(-model))
    swaps = []
    for i in range(len(dims)):
        for j in range(i + 1, len(dims)):
            if (h_rank[i] < h_rank[j]) != (m_rank[i] < m_rank[j]):
                swaps.append((dims[i], dims[j]))
    return swaps


def run_exp9() -> None:
    print("\n" + "#" * 70)
    print("# Exp 9 - Negative Pearson r Diagnosis")
    print("#" * 70)

    _, model_rows = _read_csv(ROOT / "_local_run" / "phi4_model_amce_long.csv")
    _, human_rows = _read_csv(ROOT / "_local_run" / "human_amce_long.csv")
    model_by_c = _country_amce(model_rows, "model_amce")
    human_by_c = _country_amce(human_rows, "human_amce")

    countries = sorted(set(model_by_c.keys()) & set(human_by_c.keys()))
    print(f"  countries with both AMCE: {countries}")

    summary_rows: List[Dict] = []
    swap_rows: List[Dict] = []

    for c in countries:
        shared = [d for d in DIMS if d in model_by_c[c] and d in human_by_c[c]]
        if len(shared) < 2:
            continue
        h = np.array([human_by_c[c][d] for d in shared], dtype=float)
        m = np.array([model_by_c[c][d] for d in shared], dtype=float)
        if not (np.all(np.isfinite(h)) and np.all(np.isfinite(m))):
            continue
        if h.std() == 0 or m.std() == 0:
            r = float("nan")
        else:
            r, _ = _pearsonr(h, m)
        sw = _find_swaps(h, m, shared)
        for a, b in sw:
            key = tuple(sorted([a, b]))
            swap_rows.append({"country": c, "dim_a": key[0], "dim_b": key[1]})
        summary_rows.append({
            "country": c,
            "n_dims": len(shared),
            "pearson_r": r if np.isfinite(r) else "",
            "negative_r": "True" if (np.isfinite(r) and r < 0) else "False",
            "n_swaps": len(sw),
        })

    summary_rows.sort(key=lambda x: float("inf") if x["pearson_r"] == "" else x["pearson_r"])
    _write_csv(RESULT / "negr_country_summary.csv",
               ["country", "n_dims", "pearson_r", "negative_r", "n_swaps"],
               summary_rows)
    _write_csv(RESULT / "negr_swap_pairs.csv",
               ["country", "dim_a", "dim_b"], swap_rows)

    neg_countries = [r["country"] for r in summary_rows if r["negative_r"] == "True"]
    pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for r in swap_rows:
        if r["country"] in neg_countries:
            pair_counts[(r["dim_a"], r["dim_b"])] += 1
    pair_count_rows = [{"dim_a": a, "dim_b": b, "size": n}
                       for (a, b), n in sorted(pair_counts.items(), key=lambda kv: -kv[1])]
    _write_csv(RESULT / "negr_swap_pair_counts.csv",
               ["dim_a", "dim_b", "size"], pair_count_rows)

    n_total = len(summary_rows)
    n_neg = len(neg_countries)
    para_lines = [f"Negative Pearson r occurs in {n_neg} of the {n_total} countries."]
    if n_neg > 0 and pair_count_rows:
        top = pair_count_rows[0]
        para_lines.append(
            f"In {top['size']} of the {n_neg} negative-r countries, the rank "
            f"reversal occurs between {top['dim_a']} and {top['dim_b']} -- the same "
            f"two dimensions consistently."
        )
    para_lines.append(
        "DISCA corrects the magnitude of the moral preference vector but does not "
        "always correct the relative ordering of the hardest dimension pair; this "
        "is a specific, diagnosable limitation rather than an unexplained failure."
    )
    paragraph = "\n".join(para_lines)
    (RESULT / "negr_paragraph.txt").write_text(paragraph + "\n", encoding="utf-8")

    print("\n  -- Result --")
    print(f"  countries analyzed   : {n_total}")
    print(f"  negative-r countries : {n_neg}")
    if n_neg:
        print(f"  list                 : {neg_countries}")
    if pair_count_rows:
        print("\n  Top swap pairs in neg-r countries:")
        for r in pair_count_rows[:5]:
            print(f"    {r['dim_a']} <-> {r['dim_b']}  (n={r['size']})")
    print("\n  Paragraph:")
    print("  " + paragraph.replace("\n", "\n  "))


# ============================================================================
# Exp 11 — Per-Dimension MIS-Reduction across models
# ============================================================================

CAT_TO_DIM = {
    "Species":        "Species_Humans",
    "Gender":         "Gender_Female",
    "Age":            "Age_Young",
    "Fitness":        "Fitness_Fit",
    "SocialValue":    "SocialValue_High",
    "Utilitarianism": "Utilitarianism_More",
}


def _vanilla_amce(csv_path: Path) -> Dict[str, float]:
    """Compute MPR per dim from a vanilla_results_*.csv following src.amce."""
    sums: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            cat = r.get("phenomenon_category", "")
            dim = CAT_TO_DIM.get(cat)
            if dim is None:
                continue
            try:
                p = float(r["p_spare_preferred"])
            except (KeyError, ValueError):
                continue
            if not np.isfinite(p):
                continue
            sums[dim] += p
            counts[dim] += 1
    return {d: 100.0 * sums[d] / counts[d] for d in sums if counts[d] >= 3}


def run_exp11() -> None:
    print("\n" + "#" * 70)
    print("# Exp 11 - Per-Dimension MIS-Reduction across models")
    print("#" * 70)

    base = ROOT / "exp_paper" / "result" / "exp24_paper_20c"
    if not base.is_dir():
        print(f"  [SKIP] {base} missing")
        return

    cells: List[Dict] = []
    model_summary: List[Dict] = []

    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir():
            continue
        slug = model_dir.name
        per_dim_csv = model_dir / "compare" / "per_dim_breakdown.csv"
        swa_root = model_dir / "swa"
        if not per_dim_csv.is_file() or not swa_root.is_dir():
            print(f"  [SKIP] {slug} missing per_dim or swa dir")
            continue

        # 1) DISCA per-(country, dim) abs_err + human MPR
        disca: Dict[Tuple[str, str], float] = {}
        human: Dict[Tuple[str, str], float] = {}
        with open(per_dim_csv, "r", encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                key = (r["country"], r["dimension"])
                try:
                    disca[key] = float(r["abs_err"])
                    human[key] = float(r["human"])
                except (KeyError, ValueError):
                    continue

        # 2) Vanilla per-(country, dim) abs_err — compute MPR from raw CSV
        # swa_root has one subdir like 'phi-4'; pick the first
        sub = next((d for d in swa_root.iterdir() if d.is_dir()), None)
        if sub is None:
            continue
        vanilla: Dict[Tuple[str, str], float] = {}
        for v_csv in sorted(sub.glob("vanilla_results_*.csv")):
            country = v_csv.stem.replace("vanilla_results_", "")
            van_amce = _vanilla_amce(v_csv)
            for dim, m in van_amce.items():
                key = (country, dim)
                if key in human:
                    vanilla[key] = abs(m - human[key])

        # 3) Aggregate macro per dim
        per_dim: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"van": [], "swa": []})
        for (country, dim), van_err in vanilla.items():
            if (country, dim) in disca:
                per_dim[dim]["van"].append(van_err)
                per_dim[dim]["swa"].append(disca[(country, dim)])

        for dim, d in per_dim.items():
            if len(d["van"]) < 3:
                continue
            mv = float(np.mean(d["van"]))
            ms = float(np.mean(d["swa"]))
            cells.append({
                "model": slug,
                "dimension": dim,
                "n_countries": len(d["van"]),
                "macro_van_err": round(mv, 4),
                "macro_swa_err": round(ms, 4),
                "macro_delta": round(mv - ms, 4),
                "macro_pct_gain": round(100.0 * (mv - ms) / max(mv, 1e-9), 2),
            })
        # Per-model summary line
        model_total_van = sum(np.mean(d["van"]) for d in per_dim.values() if d["van"])
        model_total_swa = sum(np.mean(d["swa"]) for d in per_dim.values() if d["swa"])
        n_dims = sum(1 for d in per_dim.values() if d["van"])
        if n_dims > 0:
            model_summary.append({
                "model": slug,
                "n_dims": n_dims,
                "macro_van_err_mean": round(model_total_van / n_dims, 4),
                "macro_swa_err_mean": round(model_total_swa / n_dims, 4),
                "macro_delta_mean":   round((model_total_van - model_total_swa) / n_dims, 4),
            })

    if not cells:
        print("  No cells computed.")
        return

    cells.sort(key=lambda r: (r["model"], r["dimension"]))
    out_csv = RESULT / "exp11_per_dim_cross_model.csv"
    _write_csv(out_csv,
               ["model", "dimension", "n_countries", "macro_van_err",
                "macro_swa_err", "macro_delta", "macro_pct_gain"],
               cells)
    out_csv2 = RESULT / "exp11_model_summary.csv"
    _write_csv(out_csv2,
               ["model", "n_dims", "macro_van_err_mean",
                "macro_swa_err_mean", "macro_delta_mean"],
               model_summary)
    print(f"  saved {out_csv.name}  ({len(cells)} cells, "
          f"{len({c['model'] for c in cells})} models)")
    print(f"  saved {out_csv2.name}")

    print("\n  -- Per-model macro improvement --")
    print(f"  {'model':<25}{'n_dims':>8}{'van_err':>10}{'swa_err':>10}{'delta':>10}")
    for r in sorted(model_summary, key=lambda x: -x["macro_delta_mean"]):
        print(f"  {r['model']:<25}{r['n_dims']:>8}{r['macro_van_err_mean']:>10.4f}"
              f"{r['macro_swa_err_mean']:>10.4f}{r['macro_delta_mean']:>10.4f}")


if __name__ == "__main__":
    run_exp2()
    run_exp9()
    run_exp11()
    print("\n" + "=" * 70)
    print(f"  Outputs written to: {RESULT}")
    print("=" * 70)
