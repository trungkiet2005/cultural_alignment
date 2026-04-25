# DISCA Playbook Experiments

This folder groups every script that implements the **DISCA experiment
playbook** ([`experiment_playbook.md`](experiment_playbook.md)).
It covers the experiments flagged as missing after Round 3 — the central
disagreement-correction evidence (Exp 1–2), the Step-3 tail-safety
defence (Exp 4), the 3×3 ablation generalisation grid (Exp 6), and the
reliability-weight distribution figure (Exp 10) — plus the all-in-one
`exp_paper_disca_playbook_qwen25_7b.py` runner that executes Exp 1–12
on Qwen2.5-7B in a single script.

All scripts share the same Kaggle self-bootstrap pattern as `round2/`
and `round3/` — copy-paste a single `!python ...` line into a fresh
Kaggle notebook cell and the script clones the repo, sets up the
environment, and runs. Outputs land in
`results/exp24_round4/<experiment>/` (per-experiment) or
`results/exp24_playbook_qwen25_7b/` (all-in-one runner).

## Layout

```
playbook/
├── experiment_playbook.md                   # The playbook itself (instructions per Exp)
├── exp_paper_disca_playbook_qwen25_7b.py    # All-in-one runner: Exp 1–12 on Qwen2.5-7B
├── exp_r4_scenario_logging.py               # Exp 1  — Figure 2 disagreement-vs-correction scatter
├── exp_r4_country_correlation.py            # Exp 2  — Figure 3 country-level variance vs ΔMIS
├── exp_r4_multiseed.py                      # Exp 3  — Multi-seed CI (mean ± std)
├── exp_r4_tail_safety.py                    # Exp 4  — Full DISCA vs consensus across 6×20 cells
├── exp_r4_baselines.py                      # Exp 5  — Strong baselines (vanilla, WVS, MC-dropout, Temp, DiffPO)
├── exp_r4_ablation_3x3.py                   # Exp 6  — 5 variants × 3 models × 3 countries
├── exp_r4_failure_prediction.py             # Exp 7  — Predictive failure regression on vanilla features
├── exp_r3_persona_count.py                  # Exp 8  — N-persona sensitivity (N ∈ {2..6})
├── exp_r4_negr_diagnosis.py                 # Exp 9  — Negative-r diagnosis (rank swap analysis)
├── exp_r4_reliability_dist.py               # Exp 10 — Reliability weight histogram + CDF
├── exp_r3_per_dim_cross_model.py            # Exp 11 — Per-dimension MIS-reduction across models
├── exp_r2_wvs_dropout.py                    # Exp 12 — WVS dimension leave-one-out
└── wvs_dimension_matrix.py                  # Exp 12 — Post-hoc 10×6 impact matrix
```

## What each script defends

| Script | Playbook Exp | Reviewer attack it preempts |
|---|---|---|
| `exp_r4_scenario_logging.py`    | 1  | "The central claim 'disagreement is the signal' is asserted but never demonstrated." |
| `exp_r4_country_correlation.py` | 2  | "Method effectiveness varies across countries for no explained reason." |
| `exp_r4_multiseed.py`           | 3  | "Single-seed results. Where are the error bars?" |
| `exp_r4_tail_safety.py`         | 4  | "Step 3 contributes only +0.006 MIS. It's a marginal component." |
| `exp_r4_baselines.py`           | 5  | "Baselines are weak. How does DISCA compare to oracle methods?" |
| `exp_r4_ablation_3x3.py`        | 6  | "Ablation is on one model and one country. Does the hierarchy generalise?" |
| `exp_r4_failure_prediction.py`  | 7  | "When does DISCA fail? Can we predict it?" |
| `exp_r3_persona_count.py`       | 8  | "Why 4 personas? Is it the right operating point?" |
| `exp_r4_negr_diagnosis.py`      | 9  | "Negative Pearson r despite better MIS — does DISCA capture cultural structure?" |
| `exp_r4_reliability_dist.py`    | 10 | "The self-regulation claim is vague. Show me the gate actually activates." |
| `exp_r3_per_dim_cross_model.py` | 11 | "Where do the gains come from? Is the pattern backbone-dependent?" |
| `exp_r2_wvs_dropout.py` + `wvs_dimension_matrix.py` | 12 | "Which WVS dim drives which moral dim? Show causal coupling." |

## Run order (recommended)

1. `exp_r4_scenario_logging.py` — produces `scenario_analysis.csv` consumed by #2 and #5.
2. `exp_r4_country_correlation.py` — consumes Exp 1 CSV + main 20-country results.
3. `exp_r4_tail_safety.py` — heaviest; needs 6 × 20 GPU cells.
4. `exp_r4_ablation_3x3.py` — 3 × 3 × 5 variants, medium cost.
5. `exp_r4_reliability_dist.py` — pure post-hoc if Exp 1 CSV exists.

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
