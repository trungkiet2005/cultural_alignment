"""Baseline vs SWA-PTIS comparison plots (1:1 port from main.py)."""

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_baseline_comparison(swa_summaries, vanilla_metrics, output_dir):
    """fig8: side-by-side bars — Vanilla vs SWA-MPPI.

    Includes MIS (paper-aligned headline metric) plus JSD / Pearson r / MAE.
    """
    countries = [s["country"] for s in swa_summaries]
    metrics = ["mis", "jsd", "pearson_r", "mae"]
    metric_labels = ["MIS \u2193 (paper)", "JSD \u2193", "Pearson r \u2191", "MAE \u2193"]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    x = np.arange(len(countries)); width = 0.35
    for ax, metric, label in zip(axes, metrics, metric_labels):
        swa_vals = [s["alignment"].get(metric, np.nan) for s in swa_summaries]
        vanilla_vals = [vanilla_metrics.get(c, {}).get(metric, np.nan) for c in countries]
        ax.bar(x - width / 2, vanilla_vals, width, label='Vanilla LLM', color='#BDBDBD', edgecolor='white')
        ax.bar(x + width / 2, swa_vals, width, label='SWA-PTIS v3', color='#2196F3', edgecolor='white')
        ax.set_xlabel("Country", fontsize=11); ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(countries, rotation=45, ha='right')
        ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, "fig8_baseline_comparison.pdf")
    plt.savefig(path, bbox_inches='tight'); plt.savefig(path.replace('.pdf', '.png'))
    plt.show(); plt.close()
    print(f"[FIG 8] Saved -> {path}")


def plot_comparison_table(all_summaries, vanilla_metrics, output_dir):
    """Publication-quality comparison table: Vanilla LLM vs SWA-PTIS v3."""

    metrics = [
        ("MIS \u2193",       "mis",          ".4f", True),   # paper-aligned headline
        ("JSD \u2193",       "jsd",          ".4f", True),
        ("Pearson r \u2191", "pearson_r",    ".4f", False),
        ("Cosine \u2191",    "cosine_sim",   ".4f", False),
        ("Spearman \u03c1 \u2191", "spearman_rho", ".4f", False),
        ("MAE \u2193",       "mae",          ".2f", True),
        ("RMSE \u2193",      "rmse",         ".2f", True),
    ]

    columns = ["Country"]
    for label, _, _, _ in metrics:
        short = label.split()[0]
        columns += [f"Van. {short}", f"SWA {short}", f"\u0394 {short}"]
    columns.append("Improv. MIS%")

    rows = []
    for s in all_summaries:
        c = s["country"]
        swa_a = s["alignment"]
        van_a = s.get("baseline_alignment", vanilla_metrics.get(c, {}))
        row = [c]
        for label, key, fmt, lower_better in metrics:
            v_val = van_a.get(key, np.nan)
            s_val = swa_a.get(key, np.nan)
            delta = s_val - v_val
            row.append(f"{v_val:{fmt}}")
            row.append(f"{s_val:{fmt}}")
            row.append(f"{delta:+{fmt}}")
        # Headline improvement: paper-aligned MIS (lower is better).
        v_mis = van_a.get("mis", np.nan)
        s_mis = swa_a.get("mis", np.nan)
        if v_mis and not np.isnan(v_mis) and v_mis > 0:
            improv = (v_mis - s_mis) / v_mis * 100
            row.append(f"{improv:+.1f}%")
        else:
            row.append("\u2014")
        rows.append(row)

    # Mean row
    mean_row = ["Mean"]
    for label, key, fmt, lower_better in metrics:
        v_vals, s_vals = [], []
        for s in all_summaries:
            c = s["country"]
            van_a = s.get("baseline_alignment", vanilla_metrics.get(c, {}))
            v = van_a.get(key, np.nan)
            sv = s["alignment"].get(key, np.nan)
            if not np.isnan(v): v_vals.append(v)
            if not np.isnan(sv): s_vals.append(sv)
        mv = np.mean(v_vals) if v_vals else np.nan
        ms = np.mean(s_vals) if s_vals else np.nan
        md = ms - mv
        mean_row.append(f"{mv:{fmt}}")
        mean_row.append(f"{ms:{fmt}}")
        mean_row.append(f"{md:+{fmt}}")
    v_miss = [s.get("baseline_alignment", vanilla_metrics.get(s["country"], {})).get("mis", np.nan)
              for s in all_summaries]
    s_miss = [s["alignment"].get("mis", np.nan) for s in all_summaries]
    v_miss = [x for x in v_miss if not np.isnan(x)]
    s_miss = [x for x in s_miss if not np.isnan(x)]
    if v_miss and s_miss:
        mean_improv = (np.mean(v_miss) - np.mean(s_miss)) / np.mean(v_miss) * 100
        mean_row.append(f"{mean_improv:+.1f}%")
    else:
        mean_row.append("\u2014")
    rows.append(mean_row)

    n_cols = len(columns)
    fig, ax = plt.subplots(figsize=(max(24, n_cols * 1.6), 0.55 * len(rows) + 2.5))
    ax.axis('off')

    table = ax.table(cellText=rows, colLabels=columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)

    for j in range(n_cols):
        cell = table[0, j]
        cell.set_text_props(color='white', fontweight='bold', fontsize=8)
        col_name = columns[j]
        if col_name.startswith("Van."):
            cell.set_facecolor('#9E9E9E')
        elif col_name.startswith("SWA"):
            cell.set_facecolor('#2196F3')
        elif col_name.startswith("\u0394") or col_name.startswith("Improv"):
            cell.set_facecolor('#FF9800')
        else:
            cell.set_facecolor('#424242')

    delta_col_indices = [j for j, col in enumerate(columns)
                         if col.startswith("\u0394") or col.startswith("Improv")]
    for i in range(1, len(rows) + 1):
        is_mean = (i == len(rows))
        for j in range(n_cols):
            cell = table[i, j]
            if is_mean:
                cell.set_facecolor('#E3F2FD')
                cell.set_text_props(fontweight='bold')
            elif i % 2 == 0:
                cell.set_facecolor('#FAFAFA')
            else:
                cell.set_facecolor('white')
            if j in delta_col_indices:
                txt = rows[i - 1][j]
                try:
                    val = float(txt.rstrip('%'))
                    metric_idx = (j - 1) // 3
                    if metric_idx < len(metrics):
                        _, _, _, lower_better = metrics[metric_idx]
                        improved = (val < 0) if lower_better else (val > 0)
                    elif "Improv" in columns[j]:
                        improved = val > 0
                    else:
                        improved = val > 0
                    if improved:
                        cell.set_text_props(color='#2E7D32')
                    elif abs(val) > 0.001:
                        cell.set_text_props(color='#C62828')
                except (ValueError, IndexError):
                    pass

    ax.set_title("Table: Vanilla LLM vs SWA-PTIS v3 \u2014 Cross-Cultural Alignment Comparison",
                 fontsize=14, fontweight='bold', pad=20)
    path = os.path.join(output_dir, "fig_comparison_table.pdf")
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'))
    plt.show(); plt.close()
    print(f"[COMPARISON TABLE] Saved -> {path}")

    # LaTeX
    latex_path = os.path.join(output_dir, "table_comparison.tex")
    with open(latex_path, 'w') as f:
        f.write("\\begin{table*}[t]\n\\centering\n")
        f.write("\\caption{Vanilla LLM vs SWA-PTIS v3: Cross-Cultural Alignment Comparison}\n")
        f.write("\\label{tab:comparison}\n\\scriptsize\n")
        col_spec = "l" + "rrr" * len(metrics) + "r"
        f.write("\\begin{tabular}{" + col_spec + "}\n")
        f.write("\\toprule\n")
        header1_parts = [""]
        for label, _, _, _ in metrics:
            header1_parts.append(f"\\multicolumn{{3}}{{c}}{{{label}}}")
        header1_parts.append("")
        f.write(" & ".join(header1_parts) + " \\\\\n")
        sub_parts = ["Country"]
        for _ in metrics:
            sub_parts += ["Van.", "SWA", "$\\Delta$"]
        sub_parts.append("Improv.\\%")
        # Build cmidrules dynamically: each metric occupies 3 columns starting at col 2.
        cmidrules = "".join(
            f"\\cmidrule(lr){{{2 + 3*i}-{4 + 3*i}}}" for i in range(len(metrics))
        )
        f.write(cmidrules + "\n")
        f.write(" & ".join(sub_parts) + " \\\\\n")
        f.write("\\midrule\n")
        for row in rows[:-1]:
            cleaned = [r.replace('%', '\\%') for r in row]
            f.write(" & ".join(cleaned) + " \\\\\n")
        f.write("\\midrule\n")
        mean_cleaned = [r.replace('%', '\\%') for r in rows[-1]]
        mean_cleaned[0] = "\\textbf{Mean}"
        f.write(" & ".join(mean_cleaned) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    print(f"[COMPARISON TABLE] Saved LaTeX -> {latex_path}")
