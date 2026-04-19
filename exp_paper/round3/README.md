# Round 3 — Reviewer-Driven Experiments

This folder contains the **seven new experiments added to address the
Round-3 ICLR review** (see [`../Review/Round3.md`](../Review/Round3.md))
plus three impact-driven experiments suggested by an internal paper
audit. All scripts share the same Kaggle-self-bootstrap pattern as
`round2/` — copy-paste a single `!python ...` line into a fresh Kaggle
notebook cell and the script clones the repo, sets up the environment,
and runs.

Outputs land in `results/exp24_round3/<experiment>/` and are zipped to
`/kaggle/working/round3_*.zip` for download.

## Layout

```
round3/
├── baselines/          # New comparator baselines (Round-3 reviewer asks)
│   └── exp_r3_baseline_swai.py
├── sensitivity/        # Method-internal hyperparameter / panel sweeps
│   ├── exp_r3_k_budget_scaling.py
│   ├── exp_r3_persona_count.py
│   └── exp_r3_global_tcat.py
└── posthoc/            # Pure post-hoc analyses (no GPU needed)
    ├── exp_r3_per_dim_cross_model.py
    ├── exp_r3_logit_conditioning_cross_model.py
    └── exp_r3_pre_run_triage.py
```

## Experiment ↔ paper-section ↔ reviewer-Q map

| Script | Paper appendix | Reviewer Q (Round 3) |
|---|---|---|
| `baselines/exp_r3_baseline_swai.py` | `app:r2_swai` | "missing SWAI comparator" |
| `sensitivity/exp_r3_k_budget_scaling.py` | `app:r2_k_budget` | (paper-impact, no specific Q) |
| `sensitivity/exp_r3_persona_count.py` | `app:r2_persona_count` | Q3 ("N=3 / N=6 robustness") |
| `sensitivity/exp_r3_global_tcat.py` | `app:r2_global_tcat` | Q4 ("single global T_cat") |
| `posthoc/exp_r3_per_dim_cross_model.py` | `app:r2_per_dim_cross_model` | (paper-impact, no specific Q) |
| `posthoc/exp_r3_logit_conditioning_cross_model.py` | `app:r2_logit_cond_cross_model` | (paper-impact, extends architectural-failure claim) |
| `posthoc/exp_r3_pre_run_triage.py` | `app:r2_triage` | Q9 ("pre-run criterion to predict low-gain regimes") |

## Compute budget

| Script | Backend | Estimated cost |
|---|---|---|
| `baselines/exp_r3_baseline_swai.py` | vLLM (any) | ~1.5h H100, Phi-4 × 20 countries |
| `sensitivity/exp_r3_k_budget_scaling.py` | vLLM | ~1.5h H100, Phi-4 × 3 countries × 6 K values |
| `sensitivity/exp_r3_persona_count.py` | vLLM | ~1.5h H100, Phi-4 × 3 countries × 5 N values |
| `sensitivity/exp_r3_global_tcat.py` | vLLM | ~2h H100, Phi-4 × 3 countries × 8 T_cat values (incl. default) |
| `posthoc/exp_r3_per_dim_cross_model.py` | CPU only | ~1 min (post-hoc on existing CSVs) |
| `posthoc/exp_r3_logit_conditioning_cross_model.py` | vLLM (or `--aggregate-only` CPU) | ~2-3h for full GPU sweep, <1 min aggregate-only |
| `posthoc/exp_r3_pre_run_triage.py` | CPU only | ~1 min (post-hoc on existing CSVs) |

The four post-hoc / aggregate-only runs (`per_dim_cross_model`,
`logit_conditioning_cross_model --aggregate-only`, `pre_run_triage`)
need no GPU and can be re-run after each main per-model experiment
completes to refresh the cross-model tables in the paper.
