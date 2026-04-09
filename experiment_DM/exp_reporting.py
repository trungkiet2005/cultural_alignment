"""Shared reporting helpers for experiment_DM scripts.

Goals:
  - Print compact paper-style summary tables (MIS/JSD/Pearson/Cosine/MAE/Flip%).
  - Print experiment-vs-reference comparisons (delta + improvement%).
  - Persist per-dimension breakdowns (from swa_runner summary) to CSV.

This module is intentionally dependency-light: only pandas + stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


DEFAULT_EXP01_REFERENCE_CSV_CANDIDATES: Sequence[str] = (
    # Default output root for experiment/kaggle_experiment.py
    "/kaggle/working/cultural_alignment/results/compare/comparison.csv",
    # Alternate: some experiments set CMP_ROOT explicitly under results/compare
    "/kaggle/working/cultural_alignment/results/compare/comparison.csv",
)


def try_load_reference_comparison(
    candidates: Sequence[str] = DEFAULT_EXP01_REFERENCE_CSV_CANDIDATES,
) -> Optional[pd.DataFrame]:
    """Best-effort load of an EXP-01 reference comparison.csv.

    Returns None if not found / unreadable.
    """
    for p in candidates:
        try:
            path = Path(p)
            if path.is_file():
                return pd.read_csv(path)
        except Exception:
            continue
    return None


def _fmt(x: float, nd: int = 4) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "nan"


def print_alignment_table(
    df: pd.DataFrame,
    *,
    title: str,
    sort_cols: Sequence[str] = ("model", "country"),
) -> None:
    """Print a compact per-row alignment table."""
    if df is None or df.empty:
        print(f"\n[{title}] (no rows)")
        return

    cols = list(df.columns)
    for req in ("model", "country"):
        if req not in cols:
            print(f"\n[{title}] Missing column: {req}")
            return

    show = df.copy()
    for c in sort_cols:
        if c in show.columns:
            show = show.sort_values(list(sort_cols))
            break

    def _get(row, key):
        return row.get(key, float("nan"))

    print("\n" + "=" * 88)
    print(f"  {title}")
    print("=" * 88)
    print(f"{'Model':<45} {'Country':<6} {'MIS':>8} {'JSD':>8} {'Pearson r':>10} {'Cos':>8} {'MAE':>8} {'Flip%':>7}")
    print("-" * 88)
    for _, row in show.iterrows():
        model = str(_get(row, "model"))[-42:]
        ctry = str(_get(row, "country"))
        mis = _get(row, "align_mis")
        jsd = _get(row, "align_jsd")
        pr = _get(row, "align_pearson_r")
        cs = _get(row, "align_cosine_sim")
        mae = _get(row, "align_mae")
        flip = _get(row, "flip_rate")
        print(
            f"{model:<45} {ctry:<6} "
            f"{_fmt(mis):>8} {_fmt(jsd):>8} {float(pr):>10.3f} {float(cs):>8.3f} {float(mae):>8.2f} {100.0*float(flip):>6.2f}%"
        )
    print("=" * 88)


@dataclass(frozen=True)
class CompareSpec:
    """Specification for comparing a 'current' dataframe to a reference dataframe."""

    metric_col: str = "align_mis"
    # Label columns that must exist in both dfs and uniquely identify rows.
    key_cols: Tuple[str, str] = ("model", "country")
    # Optional filter on method col for each df.
    ref_method: Optional[str] = None
    cur_method: Optional[str] = None
    method_col: str = "method"
    # delta = ref - cur (positive means improvement when lower-is-better)
    delta_is_ref_minus_cur: bool = True
    lower_is_better: bool = True


def print_metric_comparison(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    *,
    title: str,
    spec: CompareSpec,
) -> None:
    """Print paper-style delta/improvement table comparing ref vs current.

    Output format matches the per-model blocks printed by `experiment/kaggle_experiment.py`:
      - One block per model
      - Per-country rows
      - Macro/micro averages + win count footer per model
    """
    if ref_df is None or cur_df is None or ref_df.empty or cur_df.empty:
        print(f"\n[{title}] (missing ref/current rows)")
        return
    for k in spec.key_cols:
        if k not in ref_df.columns or k not in cur_df.columns:
            print(f"\n[{title}] Missing key col {k} in ref/current")
            return
    if spec.metric_col not in ref_df.columns or spec.metric_col not in cur_df.columns:
        print(f"\n[{title}] Missing metric col {spec.metric_col} in ref/current")
        return

    r = ref_df.copy()
    c = cur_df.copy()

    if spec.ref_method and spec.method_col in r.columns:
        r = r[r[spec.method_col] == spec.ref_method]
    if spec.cur_method and spec.method_col in c.columns:
        c = c[c[spec.method_col] == spec.cur_method]

    r = r.drop_duplicates(list(spec.key_cols)).set_index(list(spec.key_cols))
    c = c.drop_duplicates(list(spec.key_cols)).set_index(list(spec.key_cols))

    common = r.index.intersection(c.index)
    if len(common) == 0:
        print(f"\n[{title}] No overlapping (model,country) between ref and current.")
        return

    r_m = r.loc[common, spec.metric_col].astype(float)
    c_m = c.loc[common, spec.metric_col].astype(float)
    if spec.delta_is_ref_minus_cur:
        delta = r_m - c_m
    else:
        delta = c_m - r_m

    # Improvement % defined relative to reference value.
    improv_pct = (delta / r_m.replace(0, pd.NA)) * 100.0

    # Print one block per model for readability (avoids mixing countries across models).
    models = sorted(set(common.get_level_values(0).tolist()))
    for model in models:
        idx = [t for t in common.tolist() if t[0] == model]
        if not idx:
            continue
        rr = r_m.loc[idx]
        cc = c_m.loc[idx]
        dd = delta.loc[idx]
        ii = improv_pct.loc[idx]

        print("\n" + "=" * 72)
        print(f"  {title}")
        print(f"  MODEL: {str(model)}")
        print("=" * 72)
        print(f"   {'country':>7}   {'ref':>10}   {'cur':>10}   {'delta':>9}   {'improv%':>8}   {'win':>3}")

        wins = 0
        for (_, country), rv in rr.items():
            cv = float(cc.loc[(model, country)])
            dv = float(dd.loc[(model, country)])
            iv = float(ii.loc[(model, country)]) if pd.notna(ii.loc[(model, country)]) else float("nan")

            win = dv > 0 if spec.lower_is_better else dv < 0
            wins += int(win)
            wmark = "✅" if win else "❌"
            d_sign = "+" if dv >= 0 else ""
            i_sign = "+" if iv >= 0 else ""
            print(
                f"   {country:>7}   {rv:10.4f}   {cv:10.4f}   "
                f"{d_sign}{dv:9.4f}   {i_sign}{iv:7.2f}%   {wmark:>3}"
            )

        mean_r = float(rr.mean())
        mean_c = float(cc.mean())
        mean_delta = float((rr - cc).mean()) if spec.delta_is_ref_minus_cur else float((cc - rr).mean())
        macro_pct = (mean_delta / mean_r) * 100.0 if mean_r != 0 else float("nan")
        micro_pct = float(ii.mean())
        n = len(idx)

        d_sign = "+" if mean_delta >= 0 else ""
        ma_sign = "+" if macro_pct >= 0 else ""
        mi_sign = "+" if micro_pct >= 0 else ""
        print("-" * 72)
        print(f"  Mean ref: {mean_r:.4f}")
        print(f"  Mean cur: {mean_c:.4f}")
        print(f"  Absolute delta (mean): {d_sign}{mean_delta:.4f}")
        print(f"  Improvement on means (macro): {ma_sign}{macro_pct:.2f}%")
        print(f"  Mean per-row improvement (micro): {mi_sign}{micro_pct:.2f}%")
        print(f"  Wins: {wins}/{n}")
        print("=" * 72)


def flatten_per_dim_alignment(
    per_dim: Dict[str, Dict[str, float]],
    *,
    model: str,
    method: str,
    country: str,
) -> List[dict]:
    """Convert swa_runner 'per_dimension_alignment' dict into CSV rows."""
    rows: List[dict] = []
    if not per_dim:
        return rows
    for dim, d in per_dim.items():
        rows.append(
            {
                "model": model,
                "method": method,
                "country": country,
                "dimension": dim,
                "human": float(d.get("human", float("nan"))),
                "model_mpr": float(d.get("model", float("nan"))),
                "abs_err": float(d.get("abs_err", float("nan"))),
            }
        )
    # sorted view (helpful downstream)
    rows.sort(key=lambda r: (r["model"], r["country"], -float(r.get("abs_err", 0.0))))
    return rows



# ── EXP-01 Vanilla Baseline MIS (hardcoded from tracker, stable reference) ──
VANILLA_MIS: Dict[Tuple[str, str], float] = {
    # (model_key, country) → vanilla MIS
    ("qwen", "USA"): 0.4559, ("qwen", "CHN"): 0.4646, ("qwen", "JPN"): 0.4208,
    ("qwen", "DEU"): 0.4398, ("qwen", "BRA"): 0.5111,
    ("gemma", "USA"): 0.4647, ("gemma", "CHN"): 0.3679, ("gemma", "JPN"): 0.4530,
    ("gemma", "DEU"): 0.4170, ("gemma", "BRA"): 0.4490,
    ("mistral", "USA"): 0.5706, ("mistral", "CHN"): 0.4569, ("mistral", "JPN"): 0.3429,
    ("mistral", "DEU"): 0.4909, ("mistral", "BRA"): 0.4144,
}

EXP01_SWA_MIS: Dict[Tuple[str, str], float] = {
    ("qwen", "USA"): 0.3677, ("qwen", "CHN"): 0.4078, ("qwen", "JPN"): 0.2802,
    ("qwen", "DEU"): 0.3424, ("qwen", "BRA"): 0.4025,
    ("gemma", "USA"): 0.6038, ("gemma", "CHN"): 0.4536, ("gemma", "JPN"): 0.4667,
    ("gemma", "DEU"): 0.3289, ("gemma", "BRA"): 0.3655,
    ("mistral", "USA"): 0.5984, ("mistral", "CHN"): 0.5067, ("mistral", "JPN"): 0.3502,
    ("mistral", "DEU"): 0.4942, ("mistral", "BRA"): 0.4362,
}


def _model_key(model_name: str) -> str:
    low = model_name.lower()
    if "qwen" in low: return "qwen"
    if "gemma" in low: return "gemma"
    if "mistral" in low: return "mistral"
    if "llama" in low: return "llama"
    return "unknown"


def print_tracker_ready_report(
    cmp_df: pd.DataFrame,
    *,
    exp_id: str,
    per_dim_csv_path: Optional[str] = None,
) -> None:
    """
    Print a comprehensive tracker-ready report with ALL info needed for
    copy-pasting into tracker.md. Includes:
      1. Full metrics table (MIS, JSD, Pearson r, MAE, Flip%)
      2. Vanilla comparison with Δ and ✅/❌ markers
      3. EXP-01 SWA comparison with Δ and ✅/❌ markers
      4. Per-model summary (Win Rate, Macro Δ%, Micro Δ%)
      5. Per-dimension worst errors (from CSV if available)
    """
    if cmp_df is None or cmp_df.empty:
        print(f"\n[{exp_id}] No data for tracker report.")
        return

    sep = "─" * 80

    # ════════════════════════════════════════════════════════════════
    # 1. FULL METRICS TABLE (paper-ready)
    # ════════════════════════════════════════════════════════════════
    print(f"\n\n{'█'*80}")
    print(f"  {exp_id} — TRACKER-READY REPORT (copy everything below)")
    print(f"{'█'*80}")

    print(f"\n#### {exp_id} Full Metrics\n")
    print(f"| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |")
    print(f"|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|")
    for _, row in cmp_df.sort_values(["model", "country"]).iterrows():
        short = row["model"].split("/")[-1].split("-Instruct")[0].split("-instruct")[0]
        print(f"| {short} | {row['country']} | {row['align_mis']:.4f} | "
              f"{row['align_jsd']:.4f} | {row.get('align_pearson_r', float('nan')):+.3f} | "
              f"{row.get('align_mae', float('nan')):.2f} | {row['flip_rate']:.1%} |")

    # ════════════════════════════════════════════════════════════════
    # 2. VANILLA COMPARISON (Δ + win markers)
    # ════════════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print(f"#### {exp_id} vs Vanilla (MIS)\n")
    print(f"| Model | Country | Vanilla MIS | {exp_id} MIS | Δ | Improv% | Win? |")
    print(f"|:------|:-------:|:-----------:|:{'─'*len(exp_id)}---:|:-:|:-------:|:----:|")

    model_names = cmp_df["model"].unique()
    for model_name in sorted(model_names):
        m_df = cmp_df[cmp_df["model"] == model_name]
        mk = _model_key(model_name)
        short = model_name.split("/")[-1].split("-Instruct")[0].split("-instruct")[0]
        wins, total = 0, 0
        vanilla_vals, exp_vals = [], []
        for _, row in m_df.sort_values("country").iterrows():
            country = row["country"]
            v = VANILLA_MIS.get((mk, country), float("nan"))
            e = row["align_mis"]
            delta = v - e
            improv = (delta / v * 100) if v > 0 else float("nan")
            win = delta > 0
            wins += int(win)
            total += 1
            vanilla_vals.append(v)
            exp_vals.append(e)
            mark = "✅" if win else "❌"
            print(f"| {short} | {country} | {v:.4f} | {e:.4f} | "
                  f"{'+' if delta>=0 else ''}{delta:.4f} | "
                  f"**{'+' if improv>=0 else ''}{improv:.2f}%** | {mark} |")

        if vanilla_vals:
            mean_v = sum(vanilla_vals) / len(vanilla_vals)
            mean_e = sum(exp_vals) / len(exp_vals)
            macro_d = mean_v - mean_e
            macro_p = (macro_d / mean_v * 100) if mean_v > 0 else 0
            print(f"\n- **{short}** Win Rate: **{wins}/{total}** | "
                  f"Vanilla={mean_v:.4f} → {exp_id}={mean_e:.4f} | "
                  f"Macro Δ: **{'+' if macro_p>=0 else ''}{macro_p:.2f}%**")

    # ════════════════════════════════════════════════════════════════
    # 3. EXP-01 SWA COMPARISON (Δ + win markers)
    # ════════════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print(f"#### {exp_id} vs EXP-01 SWA-PTIS (MIS)\n")
    print(f"| Model | Country | EXP-01 MIS | {exp_id} MIS | Δ | Improv% | Win? |")
    print(f"|:------|:-------:|:----------:|:{'─'*len(exp_id)}---:|:-:|:-------:|:----:|")

    for model_name in sorted(model_names):
        m_df = cmp_df[cmp_df["model"] == model_name]
        mk = _model_key(model_name)
        short = model_name.split("/")[-1].split("-Instruct")[0].split("-instruct")[0]
        wins, total = 0, 0
        ref_vals, exp_vals = [], []
        for _, row in m_df.sort_values("country").iterrows():
            country = row["country"]
            r = EXP01_SWA_MIS.get((mk, country), float("nan"))
            e = row["align_mis"]
            delta = r - e
            improv = (delta / r * 100) if r > 0 else float("nan")
            win = delta > 0
            wins += int(win)
            total += 1
            ref_vals.append(r)
            exp_vals.append(e)
            mark = "✅" if win else "❌"
            print(f"| {short} | {country} | {r:.4f} | {e:.4f} | "
                  f"{'+' if delta>=0 else ''}{delta:.4f} | "
                  f"{'+' if improv>=0 else ''}{improv:.2f}% | {mark} |")

        if ref_vals:
            mean_r = sum(ref_vals) / len(ref_vals)
            mean_e = sum(exp_vals) / len(exp_vals)
            macro_d = mean_r - mean_e
            macro_p = (macro_d / mean_r * 100) if mean_r > 0 else 0
            print(f"\n- **{short}** Win Rate: **{wins}/{total}** | "
                  f"EXP-01={mean_r:.4f} → {exp_id}={mean_e:.4f} | "
                  f"Macro Δ: **{'+' if macro_p>=0 else ''}{macro_p:.2f}%**")

    # ════════════════════════════════════════════════════════════════
    # 4. OVERALL LEADERBOARD ENTRY
    # ════════════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print(f"#### {exp_id} Leaderboard Entry\n")
    overall_mis = cmp_df["align_mis"].mean()
    overall_jsd = cmp_df["align_jsd"].mean()
    overall_r   = cmp_df.get("align_pearson_r", pd.Series(dtype=float)).mean()
    overall_mae = cmp_df.get("align_mae", pd.Series(dtype=float)).mean()
    overall_flip = cmp_df["flip_rate"].mean()
    n_models    = cmp_df["model"].nunique()
    n_countries = cmp_df["country"].nunique()
    print(f"| Rank | Method | Coverage | Mean MIS ↓ | Notes |")
    print(f"|:---:|:-------|:--------:|-----------:|:------|")
    print(f"| ? | **{exp_id}** | {n_models} models × {n_countries} countries | "
          f"**{overall_mis:.4f}** | MIS↓ JSD={overall_jsd:.4f} r={overall_r:+.3f} |")

    # ════════════════════════════════════════════════════════════════
    # 5. PER-DIMENSION WORST ERRORS (from CSV if available)
    # ════════════════════════════════════════════════════════════════
    if per_dim_csv_path:
        try:
            dim_df = pd.read_csv(per_dim_csv_path)
            if not dim_df.empty and "abs_err" in dim_df.columns:
                print(f"\n{sep}")
                print(f"#### {exp_id} Per-Dimension Worst Errors\n")
                print(f"| Model | Country | Worst Dim | Human | Model | |err| (pp) |")
                print(f"|:------|:-------:|:----------|:-----:|:-----:|:---------:|")

                # Get worst dimension per (model, country)
                for (model, country), grp in dim_df.groupby(["model", "country"]):
                    worst = grp.loc[grp["abs_err"].idxmax()]
                    short = str(model).split("/")[-1].split("-Instruct")[0].split("-instruct")[0]
                    print(f"| {short} | {country} | {worst['dimension']} | "
                          f"{worst.get('human', float('nan')):.1f} | "
                          f"{worst.get('model_mpr', float('nan')):.1f} | "
                          f"**{worst['abs_err']:.1f}** |")
        except Exception as e:
            print(f"\n  [per-dim] Could not load {per_dim_csv_path}: {e}")

    print(f"\n{'█'*80}")
    print(f"  END {exp_id} TRACKER-READY REPORT")
    print(f"{'█'*80}")


def append_rows_csv(path: str, rows: Iterable[dict]) -> None:
    """Append rows to a CSV (create if not exists)."""
    rows = list(rows)
    if not rows:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if out.is_file():
        df.to_csv(out, mode="a", header=False, index=False)
    else:
        df.to_csv(out, mode="w", header=True, index=False)

