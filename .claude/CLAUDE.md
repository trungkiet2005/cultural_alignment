# Cultural Alignment — SWA-DPBR

## What This Project Is

Research codebase for a **NeurIPS 2026** submission: **SWA-DPBR** (Socially-Weighted Alignment with Dual-Pass Bootstrap Reliability) — a train-free, inference-time method to steer frozen LLMs' moral choices toward country-specific human preferences, without finetuning or per-country reward models.

Uses MultiTP dataset (40M+ human responses, 233 countries, 6 moral dimensions) in trolley-problem format. Core idea: use **within-country disagreement** among WVS-derived cultural personas as a steering signal, converted into logit-level corrections via Prospect Theory importance sampling.

## Repository Structure

```
src/                        # Core library
  config.py                 # BaselineConfig, SWAConfig dataclasses
  constants.py              # Characters, categories, country mappings (45 countries)
  data.py                   # MultiTP dataset loading & balancing
  model.py                  # Model loading (Unsloth 4-bit / HF native / vLLM)
  scenarios.py              # Multilingual moral dilemma prompt construction
  personas.py               # WVS Wave 7 persona generation (4 age cohorts + utilitarian)
  persona_i18n.py           # Persona translations (25+ languages)
  i18n.py                   # Scenario frame translations
  controller.py             # ImplicitSWAController (importance sampling engine)
  amce.py                   # AMCE computation & alignment metrics (MIS, JSD, Pearson r)
  baseline_runner.py        # Vanilla LLM baseline
  swa_runner.py             # SWA-DPBR experiment runner
  vllm_*.py                 # vLLM backend integration
  viz/                      # Radar charts, heatmaps, dendrograms, bar charts

exp_paper/                  # NeurIPS 2026 paper track (SWA-DPBR)
  Paper_New/SWA_DPBR/       # LaTeX paper, compiled PDF, generated tables
  Review/Round1.md          # Reviewer feedback
  tracker.md                # Full results log (all models x countries)
  exp_paper_*.py            # Per-model experiment scripts (22 models)
  exp_paper_ablation_phi4.py
  exp_persona_debiasing_sweep.py
  paper_countries.py        # 20 target countries + paths
  paper_runtime.py          # Kaggle env setup (vLLM / Unsloth)

open_ended/                 # BLEnD benchmark track (open-ended QA)
  main.py                   # SWA for QA on Phi-4 14B
  baseline_open_ended.py    # Vanilla greedy baseline
  swa_open_ended.py         # SWA-PTIS variant for QA
  tune_swa_mppi.py          # Hyperparameter tuning (3-phase)
  utils.py                  # 20+ model paths, country-language maps
  data/                     # BLEnD dataset (annotations, prompts, questions)
  evaluation/               # SEM-B/SEM-W scoring, MC evaluation pipeline

WVS_data/                   # World Values Survey Wave 7 (gitignored, ~190MB)
docs/                       # Gap analysis, reproducibility notes
```

## Tech Stack

- **Python 3.10+**, PyTorch, Transformers, Unsloth (4-bit), vLLM (BF16)
- **LaTeX** (NeurIPS 2026 template) — compile with `latexmk -pdf paper_revised.tex`
- **Kaggle** — primary compute environment; scripts auto-detect `/kaggle/working`
- **25+ languages** supported (scenario frames, personas, character verbalization)

## Key Metrics

- **MIS** (Misalignment Score) — L2 distance between model AMCE and human AMCE (lower = better)
- **JSD** (Jensen-Shannon Divergence) — distributional alignment
- **Pearson r** — correlation between model and human AMCEs
- **SEM-B / SEM-W** — Soft Exact Match for open-ended QA (BLEnD)

## Models

Primary: Phi-4 (14B), Llama-3.3-70B, Qwen2.5-7B, Qwen3-VL-8B, Magistral-Small-2509, Phi-3.5-mini.
Extended landscape: 16 additional models. Backends: Unsloth (4-bit for >=70B), vLLM (BF16 for <70B).

## Key Findings

- **19-24% MIS reduction** across strongest models (train-free)
- **Positional debiasing is load-bearing** (ablation: -0.282 correlation drop without it)
- **Personas are load-bearing** (ablation: -0.183 correlation drop)
- Calibration competes with scale: Phi-4 (14B) beats Llama-3.3-70B in absolute MIS
- Gains are geographically broad, not concentrated in Western countries

## Conventions

- Country codes are **ISO 3166-1 alpha-3** (USA, VNM, DEU, GBR, etc.)
- Experiment scripts follow pattern: `exp_paper_{model}.py` calling `run_for_model()`
- Results go to `results/exp24_paper_20c/{model}/{country}/`
- The performance tracker is at `docs/performance_tracker.md` — "update the tracker" means prepend a new dated entry
- `exp_paper/tracker.md` is the full experiment log with per-model per-country metrics

## Common Commands

```bash
# Run baseline for a model
python run_baseline.py --model unsloth/Phi-4 --countries USA VNM DEU

# Run SWA-DPBR
python run_swa_mppi.py --model unsloth/Phi-4 --countries USA VNM DEU

# Run specific paper experiment (on Kaggle)
python exp_paper/exp_paper_phi_4.py

# Run ablation study
python exp_paper/exp_paper_ablation_phi4.py

# Compile paper
cd exp_paper/Paper_New/SWA_DPBR && latexmk -pdf paper_revised.tex

# BLEnD baseline
python open_ended/baseline_open_ended.py

# BLEnD SWA
python open_ended/main.py
```

## Review Status

NeurIPS 2026 Round 1: **lean accept**, contingent on:
- Confidence intervals / multi-seed reporting
- Per-dimension MIS deltas
- Comparison to attention-dropout calibration
- Exact hierarchical prior specification
