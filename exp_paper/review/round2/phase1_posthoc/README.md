# Phase 1 — post-hoc analyses (no GPU, seconds to minutes)

Run these **first** on any session. They don't need model weights — they
just read existing per-country CSVs from the main Phi-4 paper run and
produce aggregate tables the paper cites.

| Script | Reviewer | What it does | Requires |
|---|---|---|---|
| `exp_r2_reliability_audit.py`  | W4b | Partitions scenarios into {high/low ESS} × {high/low disagreement} regimes and reports gate shrinkage per regime. | `results/exp24_paper_20c/phi_4/swa/phi-4/swa_results_*.csv` from the main Phi-4 run. |
| `exp_r2_rank_agreement.py`     | W7  | Kendall τ / Spearman ρ / mean rank-error for SWA-DPBR vs vanilla per country; produces the (r, MIS) scatter table. | Same CSVs as above (+ `vanilla_results_*.csv`). |
| `dump_scenario_seeds.py`       | W5a | Deterministically writes the exact per-country scenario list (Prompt + 4 columns) under `results/exp24_round2/scenario_ids/`. | MultiTP dataset path. |

## Run

```python
# All three at once (minutes on CPU):
!python exp_paper/review/round2/phase1_posthoc/exp_r2_reliability_audit.py
!python exp_paper/review/round2/phase1_posthoc/exp_r2_rank_agreement.py
!python exp_paper/review/round2/phase1_posthoc/dump_scenario_seeds.py
```

## Output

- `results/exp24_round2/reliability_audit/` — `reliability_audit_per_country.csv`, `reliability_audit_mean.csv`
- `results/exp24_round2/rank_agreement/`    — `rank_agreement_per_country.csv`, `r_vs_mis_scatter.csv`
- `results/exp24_round2/scenario_ids/`      — `<ISO3>_<lang>.csv` × 20
