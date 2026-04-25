# Round-2 Results

Per-experiment output from the NeurIPS 2026 round-2 reviewer-response runs.
Each subfolder is the artifact set of a single experiment script under
[`exp_paper/review/round2/`](../). All originally produced on Kaggle H100, then
unzipped and flattened here for archival.

## Index

| Folder | Source script | Phase | What's inside |
|---|---|---|---|
| `ablation_breadth/`     | `phase4_big_sweeps/exp_r2_ablation_breadth.py`    | 4 | Multi-model ablation (PT-IS / debias / persona / prior off) macro CSVs |
| `diffpo_binary/`        | `phase2_baselines/exp_r2_baseline_diffpo.py`      | 2 | DiffPO-binary baseline per-country results + summary |
| `hparam_sensitivity/`   | `phase3_sensitivity/exp_r2_hparam_sensitivity.py` | 3 | Sensitivity to s, λ_coop, σ, T_cat — long-format summary |
| `latency/`              | `phase7_oral/exp_r2_latency_benchmark.py`         | 7 | Wall-time overhead (vanilla vs SWA at K=32/64/128) + LaTeX table |
| `logit_conditioning/`   | `phase3_sensitivity/exp_r2_logit_conditioning.py` | 3 | Per-country base-model decision-margin diagnostics (W10) |
| `mc_dropout/`           | `phase2_baselines/exp_r2_baseline_dropout.py`     | 2 | MC-Dropout T=8, p=0.10 baseline per-country |
| `multiseed_phi4/`       | `phase4_big_sweeps/exp_r2_multiseed_phi4.py`      | 4 | Phi-4 SWA-DPBR seeds {42, 101, 2026} → 95% CI macro stats |
| `no_oversampling/`      | `phase3_sensitivity/exp_r2_no_oversampling.py`    | 3 | Ablation: SWA-DPBR with raw scenario distribution |
| `persona_amce_corr/`    | `phase7_oral/exp_r2_persona_amce_corr.py`         | 7 | WVS → human-AMCE OLS R² + Pearson per WVS×AMCE pair (Q5) |
| `persona_floor/`        | `phase6_extensions/exp_r2_persona_floor.py`       | 6 | Per-persona utility floor sweep {0, 0.5, 1.0, 2.0} |
| `persona_variant/`      | `phase3_sensitivity/exp_r2_persona_variant.py`    | 3 | Aggregate vs utilitarian-only persona panel head-to-head |
| `phase5_analysis/`      | `phase5_analysis/aggregate_round2.py`             | 5 | Post-hoc summary numbers (only `summary_numbers.txt` survived) |
| `prism_baseline/`       | `phase7_oral/exp_r2_prism_baseline.py`            | 7 | PRISM cultural-prompting baseline (mostly partial — re-run advised) |
| `scenario_ids/`         | `phase1_posthoc/dump_scenario_seeds.py`           | 1 | Frozen scenario IDs per country (reproducibility seed) |
| `tempmargin/`           | `phase2_baselines/exp_r2_baseline_tempmargin.py`  | 2 | Per-country temperature `T_c` and margin `m_c` calibration |
| `wvs_dropout/`          | `phase3_sensitivity/exp_r2_wvs_dropout.py`        | 3 | Leave-one-WVS-dim-out summary (W9 causal coupling) |

## Re-running the analysis aggregators

These archives are the inputs to the post-hoc aggregators. To regenerate
LaTeX tables for the paper, point the env var to this directory:

```bash
R2_RESULTS_BASE="$(pwd)" python exp_paper/review/round2/phase5_analysis/aggregate_round2.py
R2_RESULTS_BASE="$(pwd)" python exp_paper/review/round2/phase5_analysis/wvs_dimension_matrix.py
R2_RESULTS_BASE="$(pwd)" python exp_paper/review/round2/phase5_analysis/logit_gain_correlation.py
R2_RESULTS_BASE="$(pwd)" python exp_paper/review/round2/phase7_oral/aggregate_phase7.py
```

The phase 5/7 aggregators auto-discover sibling folders under `R2_RESULTS_BASE`.

## Provenance

Each folder corresponds to a single Kaggle session zip downloaded at
session-end on 2026-04-19. Originally archived as 12 large repo-dump zips
plus 5 small phase-script zips; flattened to per-experiment folders here
with all source zips removed. Reconstruct the original Kaggle layout by
prepending `cultural_alignment/results/exp24_round2/` to any path below.
