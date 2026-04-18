# Round-2 reviewer response scripts

Each script is **self-contained** — paste the entire file content into a
fresh Kaggle notebook cell and it will bootstrap the repo for you. See
`../Review/Round2_tracker.md` for the mapping of every reviewer comment
(W1 … W10) to script / paper section.

## Phases (pick whichever matches your Kaggle session length)

| Phase | When to run | Scripts | Needs GPU? |
|---|---|---|---|
| [phase1_posthoc/](phase1_posthoc/)    | Fast post-hoc analyses over **existing** SWA-DPBR CSVs. Run first to validate plumbing. | `exp_r2_reliability_audit.py`, `exp_r2_rank_agreement.py`, `dump_scenario_seeds.py` | No (minutes) |
| [phase2_baselines/](phase2_baselines/) | Three inference-time baselines on the 20-country Phi-4 grid. | `exp_r2_baseline_dropout.py`, `exp_r2_baseline_tempmargin.py`, `exp_r2_baseline_diffpo.py` | Yes (≈2–3h each) |
| [phase3_sensitivity/](phase3_sensitivity/) | Hyperparameter / persona / WVS / preprocessing sensitivity + logit diagnostic. | `exp_r2_logit_conditioning.py`, `exp_r2_persona_variant.py`, `exp_r2_hparam_sensitivity.py`, `exp_r2_wvs_dropout.py`, `exp_r2_no_oversampling.py` | Yes (≈4–6h combined) |
| [phase4_big_sweeps/](phase4_big_sweeps/) | Expensive: multi-model ablation breadth + multi-seed CI. | `exp_r2_ablation_breadth.py`, `exp_r2_multiseed_phi4.py` | Yes (≈6–8h each) |
| [phase5_analysis/](phase5_analysis/)  | Post-hoc aggregation over phase 1–4 CSVs → paper-ready LaTeX tables + W9 causal matrix + W10 margin-vs-gain correlation. | `aggregate_round2.py`, `logit_gain_correlation.py`, `wvs_dimension_matrix.py` | No (seconds) |
| [phase6_extensions/](phase6_extensions/) | Per-persona floor minority-protection safeguard (the §Broader impact claim, quantitatively backed). | `exp_r2_persona_floor.py` | Yes (≈2h) |

## Hardware target

- **Kaggle H100 80GB × 1** — default. All scripts use BF16 + vLLM
  (or `hf_native` for the dropout baseline, which needs Python dropout hooks).
- **Do not add Llama-3.3-70B** to `R2_BREADTH_MODELS` on a single H100 —
  use `VLLM_TENSOR_PARALLEL_SIZE=2` on a 2×H100 session instead.

## Kaggle boot (all 3 patterns work)

```python
# Pattern A — paste the whole script content into a cell
# (bootstrap at top auto-clones repo on first run)

# Pattern B — clone then invoke
!git clone --depth 1 https://github.com/trungkiet2005/cultural_alignment.git /kaggle/working/cultural_alignment
%cd /kaggle/working/cultural_alignment
!python exp_paper/round2/phase2_baselines/exp_r2_baseline_dropout.py

# Pattern C — just invoke; bootstrap clones if needed
!python exp_paper/round2/phase2_baselines/exp_r2_baseline_dropout.py
```

## Output layout

All phases write to `results/exp24_round2/<study>/`. On Kaggle that becomes
`/kaggle/working/cultural_alignment/results/exp24_round2/<study>/`, which
you can download as a zip at session end. Each long-running script streams a
`*_partial.csv` after every cell so a mid-session kill keeps progress.

## Common env overrides

| Var | Default | Purpose |
|---|---|---|
| `R2_MODEL`        | `microsoft/phi-4` | HF id of target model |
| `R2_COUNTRIES`    | 20 paper countries | Comma-separated ISO3 list |
| `R2_N_SCENARIOS`  | per-script (250–500) | Scenarios per country |
| `R2_BACKEND`      | `vllm` | `vllm` or `hf_native` (dropout needs the latter) |
| `R2_SEEDS`        | `42,101,2026` | Phase 4 multi-seed list |
