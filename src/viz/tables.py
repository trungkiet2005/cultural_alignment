"""Results table visualizations (baseline and SWA-MPPI variants)."""

import os

import numpy as np
import matplotlib.pyplot as plt

from src.viz.style import BASELINE_COLOR, SWA_COLOR


def plot_results_table(all_summaries, output_dir, mode="baseline"):
    """
    Render a publication-quality results table as a figure and LaTeX file.

    Parameters
    ----------
    all_summaries : list[dict]
        Per-country summary dicts produced by the experiment pipeline.
    output_dir : str
        Directory to write output files.
    mode : str
        ``"baseline"`` for 7-column vanilla LLM table,
        ``"swa"`` for SWA-MPPI table (adds Flip %, Latency).
    """
    if mode == "swa":
        _plot_swa_results_table(all_summaries, output_dir)
    else:
        _plot_baseline_results_table(all_summaries, output_dir)


def _plot_baseline_results_table(all_summaries, output_dir):
    """8-column baseline results table (MIS as paper-aligned headline metric)."""
    columns = ["Country", "MIS \u2193", "JSD", "Cosine", "Pearson r",
               "Spearman \u03c1", "MAE", "RMSE"]
    rows = []
    for s in all_summaries:
        a = s["alignment"]
        rows.append([
            s["country"],
            f"{a.get('mis', np.nan):.4f}",
            f"{a.get('jsd', np.nan):.4f}",
            f"{a.get('cosine_sim', np.nan):.4f}",
            f"{a.get('pearson_r', np.nan):.4f}",
            f"{a.get('spearman_rho', np.nan):.4f}",
            f"{a.get('mae', np.nan):.2f}",
            f"{a.get('rmse', np.nan):.2f}",
        ])

    # Mean row
    numeric_cols = list(range(1, len(columns)))
    mean_row = ["Mean"] + ["\u2014"] * (len(columns) - 1)
    for ci in numeric_cols:
        vals = []
        for r in rows:
            try: vals.append(float(r[ci].rstrip('%')))
            except: pass
        if vals:
            fmt = ".4f" if float(rows[0][ci]) < 10 else ".2f"
            mean_row[ci] = f"{np.mean(vals):{fmt}}"
    rows.append(mean_row)

    header_color = BASELINE_COLOR
    mean_bg = '#E0E0E0'
    title = "Baseline Vanilla LLM Cross-Cultural Alignment Results"
    label = "tab:baseline_results"

    _render_table_figure(rows, columns, header_color, mean_bg, title, output_dir)
    _render_table_latex(rows, columns, title, label, output_dir)


def _plot_swa_results_table(all_summaries, output_dir):
    """SWA-MPPI results table (MIS as paper-aligned headline metric)."""
    columns = ["Country", "MIS \u2193", "JSD", "Cosine", "Pearson r",
               "Spearman \u03c1", "MAE", "RMSE", "Flip %", "Latency (ms)"]
    rows = []
    for s in all_summaries:
        a = s["alignment"]
        rows.append([
            s["country"],
            f"{a.get('mis', np.nan):.4f}",
            f"{a.get('jsd', np.nan):.4f}",
            f"{a.get('cosine_sim', np.nan):.4f}",
            f"{a.get('pearson_r', np.nan):.4f}",
            f"{a.get('spearman_rho', np.nan):.4f}",
            f"{a.get('mae', np.nan):.2f}",
            f"{a.get('rmse', np.nan):.2f}",
            f"{s.get('flip_rate', 0):.1%}",
            f"{s['mean_latency_ms']:.1f}",
        ])

    # Mean row — skip column 8 ("Flip %", percent-formatted) for the numeric average.
    numeric_cols = [1, 2, 3, 4, 5, 6, 7, 9]
    mean_row = ["Mean"] + ["\u2014"] * (len(columns) - 1)
    for ci in numeric_cols:
        vals = []
        for r in rows:
            try: vals.append(float(r[ci].rstrip('%')))
            except: pass
        if vals:
            fmt = ".4f" if float(rows[0][ci]) < 10 else ".2f"
            mean_row[ci] = f"{np.mean(vals):{fmt}}"
    rows.append(mean_row)

    header_color = SWA_COLOR
    mean_bg = '#E3F2FD'
    title = "Table 1: SWA-MPPI v3 Cross-Cultural Alignment Results"
    label = "tab:results"

    _render_table_figure(rows, columns, header_color, mean_bg, title, output_dir,
                         fig_width=20)
    _render_table_latex(rows, columns,
                        "SWA-MPPI v3 Cross-Cultural Alignment Results",
                        label, output_dir)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _render_table_figure(rows, columns, header_color, mean_bg, title,
                         output_dir, fig_width=16):
    """Render a matplotlib table figure and save as PDF + PNG."""
    fig, ax = plt.subplots(figsize=(fig_width, 0.5 * len(rows) + 2))
    ax.axis('off')
    table = ax.table(cellText=rows, colLabels=columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.8)
    for j in range(len(columns)):
        cell = table[0, j]; cell.set_facecolor(header_color)
        cell.set_text_props(color='white', fontweight='bold')
    for i in range(1, len(rows) + 1):
        for j in range(len(columns)):
            cell = table[i, j]
            if i == len(rows):
                cell.set_facecolor(mean_bg)
                cell.set_text_props(fontweight='bold')
            elif i % 2 == 0:
                cell.set_facecolor('#F5F5F5')
            else:
                cell.set_facecolor('white')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    path = os.path.join(output_dir, "fig6_results_table.pdf")
    plt.savefig(path, bbox_inches='tight'); plt.savefig(path.replace('.pdf', '.png'))
    plt.show(); plt.close()
    print(f"[FIG 6] Saved -> {path}")


def _render_table_latex(rows, columns, caption, label, output_dir):
    """Write a LaTeX booktabs table file."""
    latex_path = os.path.join(output_dir, "table1_results.tex")
    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n\\small\n")
        f.write("\\begin{tabular}{l" + "c" * (len(columns) - 1) + "}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(columns) + " \\\\\n\\midrule\n")
        for row in rows[:-1]:
            f.write(" & ".join(row) + " \\\\\n")
        f.write("\\midrule\n")
        f.write(" & ".join(rows[-1]).replace("Mean", "\\textbf{Mean}") + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"[TABLE] Saved LaTeX -> {latex_path}")
