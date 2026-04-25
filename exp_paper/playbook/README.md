# DISCA Playbook Experiments

This folder groups every script that implements the **DISCA experiment
playbook** ([`experiment_playbook.md`](experiment_playbook.md)).
It covers the experiments flagged as missing after Round 3 — the central
disagreement-correction evidence (Exp 1–2), the Step-3 tail-safety
defence (Exp 4), the 3×3 ablation generalisation grid (Exp 6), and the
reliability-weight distribution figure (Exp 10) — plus the all-in-one
`exp_paper_disca_playbook_qwen25_7b.py` runner that executes Exp 1–12
on Qwen2.5-7B in a single script.

All scripts share the same **Kaggle offline-mode** bootstrap (no internet
required — Kaggle competition kernels). Copy-paste a single `!python ...`
line into a fresh Kaggle notebook cell and the script copies the repo
from a Kaggle input dataset to `/kaggle/working/cultural_alignment`,
sets HF/Transformers offline mode, runs the experiment, and **zips
the output to `/kaggle/working/<exp_name>.zip`** for one-click download.

Outputs land in `results/exp24_round4/<experiment>/` (per-experiment) or
`results/exp24_playbook_qwen25_7b/` (all-in-one runner).

### Required Kaggle inputs

Attach to your notebook (these are the candidates the bootstrap looks for):
- repo: `notebooks/foundnotkiet/git-moral/cultural-alignment` (notebook output)
        or any path under `/kaggle/input/cultural-alignment/`
- model weights as a Kaggle model dataset (varies per experiment)
- multitp data: `/kaggle/input/datasets/trungkiet/mutltitp-data/`

## Status — what's done, what's left

> Last verified: 2026-04-25. Update this table whenever you ship a run.
> "Done" = output files exist under `exp_paper/result/exp24_playbook_qwen25_7b/`
> or another directly-corresponding folder under `exp_paper/result/`.

| Exp | Status | Output evidence | Cost |
|---|---|---|---|
| 1  Disagreement-Correction | ✅ done   | `figure2_scenario_correlation.{pdf,png}`, `scenario_analysis*.csv` | — |
| 2  Country-Level Correlation | ✅ done (local) | `figure3_country_correlation.{pdf,png}`, `country_correlation_data.csv` | — (computed locally from paper_20c/phi-4) |
| 3  Multi-Seed CI | ✅ done   | `exp3_country_stats.csv`, `exp3_macro_stats.csv`, `exp3_seed_country_mis.csv` | — |
| 4  Tail-Safety | ❌ TODO | (no `tail_safety/` output) | **GPU heavy** — 6 models × 20 countries × 2 variants |
| 5  Strong Baselines | ✅ done (component-wise) | `result/{mc_dropout,tempmargin,diffpo_binary,round3_baselines_prompts}/` | — (re-run `exp_r4_baselines.py` if you want the unified summary table) |
| 6  3×3 Ablation Grid | ✅ done   | `exp6_ablation_grid.csv`, `exp6_ablation_grid_delta.csv` (also `result/ablation_breadth/`) | — |
| 7  Predictive Failure | ❌ TODO | (no `failure_features.csv` / `failure_regression.csv`) | GPU vanilla pass + regression — needs `R4_DISCA_CSV` |
| 8  N-Persona Sensitivity | ✅ done   | `exp8_n_persona_sensitivity.{csv,pdf,png}` (also `result/round3_sensitivity_persona_count/`) | — |
| 9  Negative-r Diagnosis | ✅ done (local) | `negr_country_summary.csv`, `negr_swap_pairs.csv`, `negr_paragraph.txt` | — (Phi-4 has 0 / 20 negative-r countries) |
| 10 Reliability Distribution | ✅ done   | `exp10_reliability_distribution.{pdf,png}`, `exp10_reliability_stats.csv` | — |
| 11 Per-Dimension Breakdown | ✅ done (local) | `exp11_per_dim_cross_model.csv` (66 cells = 11 models × 6 dims), `exp11_model_summary.csv` | — (computed locally from `result/exp24_paper_20c/`) |
| 12 WVS Dimension Dropout | ✅ done   | `exp12_wvs_dropout_{raw,summary}.csv` (also `result/wvs_dropout/`) | — |

**2 remaining experiments:** 4, 7. Both need GPU; see "Recommended next runs"
below. Exp 2 / 9 / 11 were ran locally via [`_local_run/run_exp2_exp9_local.py`](../../_local_run/run_exp2_exp9_local.py)
(stdlib-only, bypasses scipy/pandas import-time DLL issues on the user's box).

## Layout

```
playbook/
├── experiment_playbook.md                   # The playbook itself (instructions per Exp)
├── exp_paper_disca_playbook_qwen25_7b.py    # All-in-one runner: Exp 1–12 on Qwen2.5-7B
│
│ ───── Exp 1 — Disagreement-Correction (Figure 2) ────────────────
├── exp_r4_scenario_logging.py               # per-scenario log + Pearson r scatter
│
│ ───── Exp 2 — Country-Level Correlation (Figure 3) ──────────────
├── exp_r4_country_correlation.py            # per-country mean variance vs ΔMIS
│
│ ───── Exp 3 — Multi-Seed Confidence Intervals ───────────────────
├── exp_r4_multiseed.py                      # 5-country multi-seed (Phi-4 default)
├── exp_r2_multiseed_phi4.py                 # full 20-country multi-seed (Phi-4)
│
│ ───── Exp 4 — Tail-Safety (Step 3 defense) ──────────────────────
├── exp_r4_tail_safety.py                    # Full DISCA vs consensus across 6×20 cells
│
│ ───── Exp 5 — Strong Baselines ──────────────────────────────────
├── exp_r4_baselines.py                      # all-in-one: vanilla / WVS / MC-dropout / Temp / DiffPO
├── exp_r2_baseline_dropout.py               # MC-Dropout standalone (Phi-4 × 20)
├── exp_r2_baseline_diffpo.py                # DiffPO-binary standalone
├── exp_r2_baseline_tempmargin.py            # per-country temperature/margin scaling
├── exp_r3_baseline_prompts.py               # B1/B2/B3/B4 prompt-prefix baselines (incl. WVS prompt)
│
│ ───── Exp 6 — 3×3 Ablation Grid ─────────────────────────────────
├── exp_r4_ablation_3x3.py                   # 5 variants × 3 models × 3 countries
├── exp_r2_ablation_breadth.py               # 6-row ablation across more (model, country) pairs
│
│ ───── Exp 7 — Predictive Failure Model ──────────────────────────
├── exp_r4_failure_prediction.py             # regress ΔMIS on margin/entropy/vanilla_mis
│
│ ───── Exp 8 — N-Persona Sensitivity ─────────────────────────────
├── exp_r3_persona_count.py                  # N ∈ {2..6}
│
│ ───── Exp 9 — Negative Pearson r Diagnosis ──────────────────────
├── exp_r4_negr_diagnosis.py                 # rank-swap analysis on negative-r countries
├── exp_r2_rank_agreement.py                 # Kendall τ / Spearman ρ / mean rank-error
│
│ ───── Exp 10 — Reliability Weight Distribution ──────────────────
├── exp_r4_reliability_dist.py               # histogram + CDF
├── exp_r2_reliability_audit.py              # post-hoc audit table from main run
│
│ ───── Exp 11 — Per-Dimension Improvement ────────────────────────
├── exp_r3_per_dim_cross_model.py            # 6-models × 6-dimensions matrix
├── exp_r2_per_dim_mis.py                    # per-dim MIS decomposition for one model
│
│ ───── Exp 12 — WVS Dimension Dropout ────────────────────────────
├── exp_r2_wvs_dropout.py                    # leave-one-WVS-dim-out
└── wvs_dimension_matrix.py                  # post-hoc 10×6 WVS-vs-MultiTP impact matrix
```

## What each script defends

| Playbook Exp | Reviewer attack it preempts | Primary script(s) |
|---|---|---|
| 1  | "The central claim 'disagreement is the signal' is asserted but never demonstrated." | `exp_r4_scenario_logging.py` |
| 2  | "Method effectiveness varies across countries for no explained reason." | `exp_r4_country_correlation.py` |
| 3  | "Single-seed results. Where are the error bars?" | `exp_r4_multiseed.py`, `exp_r2_multiseed_phi4.py` |
| 4  | "Step 3 contributes only +0.006 MIS. It's a marginal component." | `exp_r4_tail_safety.py` |
| 5  | "Baselines are weak. How does DISCA compare to oracle methods?" | `exp_r4_baselines.py`, `exp_r2_baseline_*`, `exp_r3_baseline_prompts.py` |
| 6  | "Ablation is on one model and one country. Does the hierarchy generalise?" | `exp_r4_ablation_3x3.py`, `exp_r2_ablation_breadth.py` |
| 7  | "When does DISCA fail? Can we predict it?" | `exp_r4_failure_prediction.py` |
| 8  | "Why 4 personas? Is it the right operating point?" | `exp_r3_persona_count.py` |
| 9  | "Negative Pearson r despite better MIS — does DISCA capture cultural structure?" | `exp_r4_negr_diagnosis.py`, `exp_r2_rank_agreement.py` |
| 10 | "The self-regulation claim is vague. Show me the gate actually activates." | `exp_r4_reliability_dist.py`, `exp_r2_reliability_audit.py` |
| 11 | "Where do the gains come from? Is the pattern backbone-dependent?" | `exp_r3_per_dim_cross_model.py`, `exp_r2_per_dim_mis.py` |
| 12 | "Which WVS dim drives which moral dim? Show causal coupling." | `exp_r2_wvs_dropout.py`, `wvs_dimension_matrix.py` |

## Recommended next runs (only the 2 remaining experiments)

Both need GPU. Each script auto-zips its output to
`/kaggle/working/<exp_name>.zip` for one-click download.

```bash
# Exp 7 — GPU vanilla forward pass per country + regression (~30 min on H100).
R4_DISCA_CSV=/kaggle/input/.../disca_phi4_20c.csv \
    python exp_paper/playbook/exp_r4_failure_prediction.py

# Exp 4 — heaviest. 6 models × 20 countries × {full, consensus} variants.
python exp_paper/playbook/exp_r4_tail_safety.py
```

**Priority** if you only have time for one: **Exp 4** (tail-safety) — it
defends Step 3, the harshest Reviewer 2 criticism.

## Env overrides (common)

| Var | Meaning | Default |
|---|---|---|
| `R4_MODEL`        | HF id of the model to run | `microsoft/phi-4` |
| `R4_MODELS`       | Comma list of HF ids (tail-safety, 3×3) | panel defaults |
| `R4_COUNTRIES`    | Comma ISO3 list | per-script defaults |
| `R4_N_SCENARIOS`  | Per-country scenarios | 250–500 |
| `R4_BACKEND`      | `vllm` (default) or `hf_native` | `vllm` |
| `EXP24_VAR_SCALE` | Dual-pass reliability scale | `0.04` |
| `EXP24_K_HALF`    | IS samples per pass | `64` |

See each script's top docstring for script-specific env vars.
