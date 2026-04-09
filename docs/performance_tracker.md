# Performance Tracker — SWA-PTIS vs Vanilla Baseline

Chronological log of cultural-alignment experiments. Each run records the
config, per-model MIS tables, and aggregated stats so we can see how method
changes (math fixes, persona prompts, scenario data) move the numbers over
time. Append new entries to the **top**.

Primary metric: **MIS** = L2 misalignment vs human AMCE (paper-aligned, lower
is better). The CSV behind every run is saved at
`results/compare/mis_comparison.csv` on the Kaggle box that ran it.

---

## Run 2026-04-09 — first 3-model sweep on the post-σ-normalisation pipeline

**Commit:** `756d389` (vendor SWA_MPPI_paper) on top of `ca6b206`
(σ-normalise Prospect-Theory input — fixes T_cat cross-dim bias)
**Script:** [experiment/kaggle_experiment.py](../experiment/kaggle_experiment.py)
**Config:**
- Models (4-bit Unsloth): Qwen2.5-7B-Instruct, Gemma-2-9B-it, Mistral-7B-Instruct-v0.3
- Countries: USA, CHN, JPN, DEU, BRA
- `N_SCENARIOS = 500`, `BATCH_SIZE = 1`, `SEED = 42`
- Real MultiTP data, WVS-7 personas
- Method: paper SWA-PTIS (linear A↔B debias → persona-mean anchor → PT-IS solver with ESS gate)

### Per-model MIS tables

```
========================================================================
  MIS (PAPER-ALIGNED) IMPROVEMENT  —  L2 misalignment, lower=better
  MODEL: unsloth/Qwen2.5-7B-Instruct-bnb-4bit
========================================================================
   country   baseline        swa      delta     improv %
       USA     0.4559     0.3677    +0.0882     +19.34% ↓
       CHN     0.4646     0.4078    +0.0568     +12.22% ↓
       JPN     0.4208     0.2802    +0.1405     +33.40% ↓
       DEU     0.4398     0.3424    +0.0974     +22.15% ↓
       BRA     0.5111     0.4025    +0.1086     +21.26% ↓
------------------------------------------------------------------------
  Mean baseline MIS:            0.4584
  Mean SWA-PTIS MIS:            0.3601
  Absolute reduction:           +0.0983
  Improvement on the means:     +21.45%   (macro-average)
  Mean per-country improvement: +21.68%   (micro-average)
  SWA wins (MIS lower):         5/5 countries
========================================================================

========================================================================
  MIS (PAPER-ALIGNED) IMPROVEMENT  —  L2 misalignment, lower=better
  MODEL: unsloth/gemma-2-9b-it-bnb-4bit
========================================================================
   country   baseline        swa      delta     improv %
       USA     0.4647     0.6038    -0.1391     -29.95% ↑
       CHN     0.3679     0.4536    -0.0857     -23.28% ↑
       JPN     0.4530     0.4667    -0.0136      -3.01% ↑
       DEU     0.4170     0.3289    +0.0882     +21.14% ↓
       BRA     0.4490     0.3655    +0.0834     +18.58% ↓
------------------------------------------------------------------------
  Mean baseline MIS:            0.4303
  Mean SWA-PTIS MIS:            0.4437
  Absolute reduction:           -0.0134
  Improvement on the means:      -3.11%   (macro-average)
  Mean per-country improvement:  -3.30%   (micro-average)
  SWA wins (MIS lower):         2/5 countries
========================================================================

========================================================================
  MIS (PAPER-ALIGNED) IMPROVEMENT  —  L2 misalignment, lower=better
  MODEL: unsloth/mistral-7b-instruct-v0.3-bnb-4bit
========================================================================
   country   baseline        swa      delta     improv %
       USA     0.5706     0.5984    -0.0278      -4.87% ↑
       CHN     0.4569     0.5067    -0.0498     -10.90% ↑
       JPN     0.3429     0.3502    -0.0073      -2.12% ↑
       DEU     0.4909     0.4942    -0.0033      -0.67% ↑
       BRA     0.4144     0.4362    -0.0217      -5.25% ↑
------------------------------------------------------------------------
  Mean baseline MIS:            0.4551
  Mean SWA-PTIS MIS:            0.4771
  Absolute reduction:           -0.0220
  Improvement on the means:      -4.83%   (macro-average)
  Mean per-country improvement:  -4.76%   (micro-average)
  SWA wins (MIS lower):         0/5 countries
========================================================================
```

### Aggregated summary (3 models × 5 countries = 15 cells)

| Model                  | Mean baseline | Mean SWA | Δ MIS    | Improv % | Wins |
| ---------------------- | ------------: | -------: | -------: | -------: | :--: |
| Qwen2.5-7B-Instruct    | 0.4584        | 0.3601   | +0.0983  | +21.45%  | 5/5  |
| Gemma-2-9B-it          | 0.4303        | 0.4437   | -0.0134  |  -3.11%  | 2/5  |
| Mistral-7B-Instruct    | 0.4551        | 0.4771   | -0.0220  |  -4.83%  | 0/5  |
| **All 3 (macro)**      | **0.4479**    | **0.4270** | **+0.0210** | **+4.50%** | **7/15** |

### Per-country across-model summary (Δ MIS = baseline − SWA)

| Country | Qwen      | Gemma     | Mistral   | Mean Δ    | Models helped |
| :-----: | --------: | --------: | --------: | --------: | :-----------: |
| USA     | +0.0882   | -0.1391   | -0.0278   | -0.0263   | 1/3           |
| CHN     | +0.0568   | -0.0857   | -0.0498   | -0.0262   | 1/3           |
| JPN     | +0.1405   | -0.0136   | -0.0073   | +0.0399   | 1/3           |
| DEU     | +0.0974   | +0.0882   | -0.0033   | +0.0608   | 2/3           |
| BRA     | +0.1086   | +0.0834   | -0.0217   | +0.0568   | 2/3           |

### Observations

- **Qwen is the clean win** — 5/5 countries, ≈+21% mean MIS improvement. The
  paper pipeline does what it says on the tin here.
- **Mistral is uniformly slightly negative** — 0/5 wins, -4.8% mean. The
  damage is small per-country (-0.7% to -10.9%) but consistent. Suggests SWA
  is flipping borderline-correct vanilla decisions for Mistral, possibly
  because Mistral's persona-conditional logits don't separate the way the
  PT-IS update assumes.
- **Gemma is bimodal** — wins big on DEU/BRA (≈+20%), loses big on USA/CHN
  (-30% / -23%). JPN essentially flat. Worth digging into whether Gemma's
  personas for USA/CHN are degenerate (e.g. all 4 personas give the same
  delta_agents → adaptive σ collapses to noise floor).
- **DEU and BRA are SWA's friends** across the board (helped on 2/3 models,
  flat on the third). USA and CHN are the hostile pair (helped on 1/3,
  hurt big on Gemma).
- **Macro across all models is +4.5%** — net positive but modest. The Qwen
  result is carrying the average; Mistral and Gemma are dragging it down.

### TODO / next runs to compare against

- [ ] Run with `N_SCENARIOS = 1000` to check whether the Mistral regression
      is a small-sample artefact.
- [ ] Inspect `swa_results_<country>.csv` for Gemma USA/CHN — check
      `variance` and `sigma_used` to see if persona disagreement collapsed.
- [ ] Try a per-model `decision_temperature` sweep — Mistral's softmax may
      be sharper/flatter than Qwen's, breaking the shared `beta`.
