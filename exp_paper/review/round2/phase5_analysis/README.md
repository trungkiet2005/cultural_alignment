# Phase 5 — post-hoc analyses over phase 1–4 outputs (no GPU)

These scripts consume the CSV outputs from phases 1–4 and produce
**paper-ready LaTeX tables + summary numbers**. Run after phase 1–4 have
completed.

| Script | Consumes | Produces |
|---|---|---|
| `aggregate_round2.py` | every `*_summary.csv` + per-country CSV from phases 1–4 | 5 `.tex` tables + `summary_numbers.txt` (filling the paper's placeholder tables) |
| `logit_gain_correlation.py` | `logit_conditioning_per_country.csv` + main Phi-4 `comparison.csv` | `margin_vs_gain_scatter.csv` + `margin_vs_gain_summary.txt` (W10 correlation, Pearson $r$ + Spearman $\rho$) |
| `wvs_dimension_matrix.py` | `wvs_dropout_summary.csv` | `wvs_dim_impact_matrix.{csv,tex}` (W9 10×6 causal coupling matrix) |

## Run

```bash
# From anywhere (self-bootstrap handles repo + sys.path).
python exp_paper/review/round2/phase5_analysis/aggregate_round2.py
python exp_paper/review/round2/phase5_analysis/logit_gain_correlation.py
python exp_paper/review/round2/phase5_analysis/wvs_dimension_matrix.py
```

## Outputs

All under `results/exp24_round2/phase5_analysis/`:

- `main_baselines_20country.tex` — 20-country head-to-head (Phi-4 vanilla vs
  MC-Dropout / per-country T-scale / margin-scale / DiffPO-binary)
- `hparam_sensitivity_filled.tex` — drop-in replacement for the paper's
  `\ref{tab:r2_hparam_sensitivity}` with actual min/max MIS rows
- `reliability_audit_table.tex` — scenario counts by (ESS × disagreement)
  regime, with mean reliability weight $r$ and shrinkage
- `multiseed_ci_table.tex` — Phi-4 macro MIS with 95% normal CI over 3 seeds
- `persona_variant_head_to_head.tex` — aggregate vs utilitarian 4th persona
  on the 20-country grid, with per-country and macro delta
- `summary_numbers.txt` — one-line summaries for pasting into prose
- `margin_vs_gain_scatter.csv` + `margin_vs_gain_summary.txt` — W10 scatter
  data and fit statistics
- `wvs_dim_impact_matrix.{csv,tex}` — W9 10×6 coupling matrix with top-3
  WVS→MultiTP couplings per row bolded

## Env overrides

- `R2_RESULTS_BASE` (default: `/kaggle/working/.../results/exp24_round2` on
  Kaggle, `results/exp24_round2` locally) — root holding phase-1..4 folders
- `R2_MAIN_COMPARISON` — path to main Phi-4 `comparison.csv` for the
  correlation script (default: the Kaggle main-run path)
- `R2_OUT_DIR` — where to write phase-5 outputs (default: `<R2_BASE>/phase5_analysis/`)
