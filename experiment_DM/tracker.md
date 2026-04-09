# SWA-PTIS — experiment_DM Tracker

> **Folder purpose**: Custom experiments built on top of the paper baseline (`experiment/kaggle_experiment.py`).  
> Each script is **self-contained** and Kaggle-ready (auto-bootstrap + pip-install).  
> Read the `Motivation` docstring at the top of each file for full design rationale.
>
> **Run order on Kaggle H100**: EXP-07 → EXP-06 (routing) → EXP-03 → EXP-05 → EXP-04  
> **Primary metric**: MIS = L2 misalignment vs human AMCE ↓. Secondary: JSD ↓, Pearson r ↑.  
> **Completed runs**: EXP-01 ✅ (2026-04-09) | EXP-02 ✅ (2026-04-09) | EXP-07a ✅ (2026-04-09) | EXP-09 ✅ (2026-04-09)

---

## File Index

| File | EXP ID | Status | Key Innovation | Fixes |
|:-----|:------:|:------:|:--------------|:------|
| `exp02_expanded_personas.py` | EXP-02 | ✅ DONE | 8 personas (urban/rural split) | More coverage |
| `exp03_socialvalue_personas.py` | EXP-03 | ✅ DONE | Social-utility gradient personas (P4/P5) | SocialValue underestimation |
| `exp04_mistral_crosslingual.py` | EXP-04 | ✅ DONE | English persona override + σ₀=0.8 + K=512 + T=0.5 | Mistral variance collapse |
| `exp05_anchor_regularization.py` | EXP-05 | ✅ DONE | ESS-adaptive anchor α·anchor + (1-α)·base | Gemma over-correction |
| `exp06_adaptive_sigma.py` | EXP-06a | 🟡 READY | Per-scenario adaptive σ (entropy-based) | Qwen32B logit collapse |
| `exp06_category_routing.py` | EXP-06b | ✅ DONE | Per-category expert persona pools (6 panels) | Dim-level anchor bias |
| `exp07_best_config_sweep.py` | EXP-07 | 🟡 READY | **Combined EXP-03+04+05+06, 15 countries** | ALL |
| `exp07_wvs_augmentation.py` | EXP-07a | ✅ DONE | Hofstede-neighbor persona augmentation (sparse WVS) | Sparse WVS coverage |
| `exp08_category_routing.py` | EXP-08 | 🟡 READY | Extended category routing (8 panels) | Precision routing |
| `exp09_hierarchical_is.py` | EXP-09 | ✅ DONE | Hierarchical IS with country prior (EMA + annealing) | IS variance / stability |

---

## Key Tables (paper-ready)

These are the **most important** tables for fast decision-making and for paper insertion.

### Leaderboard (Mean MIS ↓)

| Rank | Method | Coverage | Mean MIS ↓ | Notes |
|:---:|:-------|:--------:|-----------:|:------|
| 1 | **EXP-09 Hierarchical IS** | 3 models × 5 countries | **0.3975** | Strong overall; Mistral still shows negative correlation risk in some countries |
| 2 | **EXP-05 Anchor regularization** | 3 models × 5 countries | **0.4174** | Big gain on Qwen (esp. JPN/DEU); Gemma improves vs EXP-01 on USA/CHN but still not “fixed” |
| 3 | EXP-01 SWA-PTIS (4-agent) | 3 models × 5 countries | 0.4269 | Strong for Qwen; harms Gemma/Mistral in some countries |
| 4 | EXP-06b Category routing | 3 models × 5 countries | 0.4269 | Practically identical to EXP-01 (no measurable gain) |
| 5 | EXP-02 Expanded personas (8-agent) | 3 models × 5 countries | 0.4304 | Improves Qwen; regresses Gemma/Mistral |
| 6 | EXP-03 SocialValue personas | 3 models × 5 countries | 0.4413 | Strong gains on Qwen SocialValue (USA/CHN/JPN), but Gemma/Mistral remain problematic and anti-correlation persists in some countries |
| 7 | EXP-04 Mistral cross-lingual | 1 model × 5 countries | 0.4463* | *Only Mistral was run (Qwen/Gemma unchanged vs EXP-01); mixed outcome (CHN/DEU/JPN improved, USA/BRA regressed) |
| 8 | EXP-07a WVS augmentation | 2 models × 5 countries | 0.4031* | *Not directly comparable (missing Mistral) |

> Mean MIS computed as the simple average over the reported (model,country) rows for that method.

### Big Table — MIS vs Vanilla (all methods)

Reference point: **EXP-01 Vanilla** (raw LLM, no personas).  
Notation: `delta = MIS_vanilla - MIS_method` so **positive delta = method improved**.

| Model | Country | Vanilla MIS | EXP-01 SWA | Δ | Improv% | EXP-02 (8-agent) | Δ | Improv% | EXP-03 (SV personas) | Δ | Improv% | EXP-04 (Mistral xling) | Δ | Improv% | EXP-05 (anchor reg) | Δ | Improv% | EXP-07a (WVS aug) | Δ | Improv% | EXP-09 (hier IS) | Δ | Improv% |
|:------|:-------:|------------:|-----------:|--:|--------:|-----------------:|--:|--------:|---------------------:|--:|--------:|--------------------:|--:|--------:|------------------:|--:|--------:|-----------------:|--:|--------:|-----------------:|--:|--------:|
| Qwen2.5-7B | USA | 0.4559 | 0.3677 | +0.0882 | +19.34% | 0.3496 | +0.1063 | +23.31% | **0.2491** | **+0.2069** | **+45.38%** | — | — | — | 0.3628 | +0.0931 | +20.43% | 0.3687 | +0.0872 | +19.13% | 0.3538 | +0.1021 | +22.40% |
| Qwen2.5-7B | CHN ⚠️ | 0.4646 | 0.4078 | +0.0568 | +12.22% | 0.3680 | +0.0966 | +20.79% | 0.2930 | +0.1717 | +36.95% | — | — | — | 0.3791 | +0.0855 | +18.40% | 0.4094 | +0.0552 | +11.89% | 0.3526 | +0.1121 | +24.12% |
| Qwen2.5-7B | JPN | 0.4208 | 0.2802 | +0.1405 | +33.40% | 0.2808 | +0.1400 | +33.26% | 0.2925 | +0.1283 | +30.49% | — | — | — | **0.2493** | **+0.1714** | **+40.72%** | **0.2801** | +0.1407 | +33.44% | 0.3392 | +0.0816 | +19.39% |
| Qwen2.5-7B | DEU | 0.4398 | 0.3424 | +0.0974 | +22.15% | 0.3895 | +0.0503 | +11.43% | 0.3827 | +0.0571 | +12.99% | — | — | — | **0.3140** | **+0.1259** | **+28.61%** | 0.3444 | +0.0954 | +21.69% | 0.4262 | +0.0136 | +3.09% |
| Qwen2.5-7B | BRA | 0.5111 | 0.4025 | +0.1086 | +21.26% | 0.3904 | +0.1207 | +23.62% | 0.4041 | +0.1070 | +20.94% | — | — | — | 0.4493 | +0.0618 | +12.09% | 0.3792 | +0.1319 | +25.81% | 0.3546 | +0.1565 | +30.62% |
| Gemma-2-9B | USA | 0.4647 | 0.6038 | -0.1391 | -29.95% | 0.6073 | -0.1426 | -30.68% | 0.5497 | -0.0850 | -18.30% | — | — | — | 0.5599 | -0.0952 | -20.48% | 0.6067 | -0.1420 | -30.55% | 0.4922 | -0.0275 | -5.91% |
| Gemma-2-9B | CHN ⚠️ | 0.3679 | 0.4536 | -0.0857 | -23.28% | 0.4095 | -0.0416 | -11.30% | 0.6321 | -0.2642 | -71.81% | — | — | — | 0.4002 | -0.0323 | -8.78% | 0.4517 | -0.0838 | -22.78% | 0.3592 | +0.0087 | +2.36% |
| Gemma-2-9B | JPN | 0.4530 | 0.4667 | -0.0136 | -3.01% | 0.5730 | -0.1200 | -26.50% | 0.5265 | -0.0735 | -16.24% | — | — | — | 0.5012 | -0.0482 | -10.63% | 0.4616 | -0.0086 | -1.90% | 0.4411 | +0.0119 | +2.63% |
| Gemma-2-9B | DEU | 0.4170 | **0.3289** | +0.0882 | +21.14% | 0.3418 | +0.0752 | +18.03% | 0.3948 | +0.0222 | +5.33% | — | — | — | 0.3420 | +0.0750 | +17.98% | **0.3286** | +0.0884 | +21.20% | 0.3653 | +0.0517 | +12.39% |
| Gemma-2-9B | BRA | 0.4490 | 0.3655 | +0.0834 | +18.58% | 0.3873 | +0.0617 | +13.74% | 0.4406 | +0.0084 | +1.87% | — | — | — | **0.3446** | **+0.1045** | **+23.27%** | 0.4002 | +0.0488 | +10.86% | 0.3438 | +0.1052 | +23.43% |
| Mistral-7B | USA | 0.5706 | 0.5984 | -0.0278 | -4.87% | 0.6368 | -0.0661 | -11.59% | 0.6666 | -0.0960 | -16.83% | 0.6303 | -0.0597 | -10.46% | — | — | — | 0.5266 | +0.0440 | +7.72% |
| Mistral-7B | CHN ⚠️ | 0.4569 | 0.5067 | -0.0498 | -10.90% | 0.5053 | -0.0484 | -10.59% | **0.3091** | **+0.1478** | **+32.35%** | 0.4764 | -0.0195 | -4.26% | — | — | — | 0.4099 | +0.0470 | +10.29% |
| Mistral-7B | JPN | 0.3429 | 0.3502 | -0.0073 | -2.12% | 0.3508 | -0.0079 | -2.30% | 0.3221 | +0.0208 | +6.06% | **0.3442** | -0.0013 | -0.37% | — | — | — | 0.3273 | +0.0156 | +4.55% |
| Mistral-7B | DEU | 0.4909 | 0.4942 | -0.0033 | -0.67% | 0.5106 | -0.0197 | -4.01% | 0.4091 | +0.0818 | +16.66% | 0.4889 | +0.0020 | +0.41% | — | — | — | 0.4634 | +0.0275 | +5.60% |
| Mistral-7B | BRA | 0.4144 | 0.4362 | -0.0217 | -5.25% | 0.4447 | -0.0303 | -7.32% | 0.5246 | -0.1102 | -26.59% | 0.4195 | -0.0051 | -1.22% | — | — | — | 0.4138 | +0.0006 | +0.13% |

### Big Table — Full metrics (most complete we have)

This is the “copy-paste into paper” table format with **max available metrics**.  
For Vanilla, only MIS is tracked here (we did not log vanilla JSD/Pearson/MAE in EXP-01).

#### EXP-01 SWA-PTIS (full metrics)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Qwen2.5-7B | USA | 0.3677 | 0.0759 | +0.639 | 9.75 | 1.9% |
| Qwen2.5-7B | CHN ⚠️ | 0.4078 | 0.0956 | +0.418 | 13.54 | 1.3% |
| Qwen2.5-7B | JPN | 0.2802 | 0.0553 | +0.406 | 9.97 | 0.6% |
| Qwen2.5-7B | DEU | 0.3424 | 0.0558 | +0.461 | 10.43 | 2.6% |
| Qwen2.5-7B | BRA | 0.4025 | 0.0942 | +0.167 | 14.30 | 0.6% |
| Gemma-2-9B | USA | 0.6038 | 0.1108 | +0.630 | 22.31 | 0.6% |
| Gemma-2-9B | CHN ⚠️ | 0.4536 | 0.1034 | +0.777 | 14.68 | 1.3% |
| Gemma-2-9B | JPN | 0.4667 | 0.0794 | +0.332 | 15.85 | 1.6% |
| Gemma-2-9B | DEU | 0.3289 | 0.0683 | +0.795 | 10.13 | 1.3% |
| Gemma-2-9B | BRA | 0.3655 | 0.0749 | +0.272 | 14.09 | 3.9% |
| Mistral-7B | USA | 0.5984 | 0.1303 | -0.570 | 21.61 | 0.6% |
| Mistral-7B | CHN ⚠️ | 0.5067 | 0.1050 | -0.682 | 17.41 | 0.3% |
| Mistral-7B | JPN | 0.3502 | 0.0765 | -0.905 | 12.46 | 1.3% |
| Mistral-7B | DEU | 0.4942 | 0.1060 | -0.957 | 17.18 | 1.0% |
| Mistral-7B | BRA | 0.4362 | 0.0947 | -0.665 | 14.02 | 0.3% |

#### EXP-02 Expanded Personas (8-agent, full metrics)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Qwen2.5-7B | USA | 0.3496 | 0.0636 | +0.625 | 10.30 | 0.3% |
| Qwen2.5-7B | CHN | 0.3680 | 0.0842 | +0.343 | 12.56 | 0.6% |
| Qwen2.5-7B | JPN | 0.2808 | 0.0488 | +0.366 | 9.29 | 0.3% |
| Qwen2.5-7B | DEU | 0.3895 | 0.0592 | +0.256 | 12.53 | 1.3% |
| Qwen2.5-7B | BRA | 0.3904 | 0.0880 | -0.010 | 13.20 | 0.3% |
| Gemma-2-9B | USA | 0.6073 | 0.1098 | +0.557 | 22.12 | 0.6% |
| Gemma-2-9B | CHN | 0.4095 | 0.0860 | +0.709 | 14.25 | 1.0% |
| Gemma-2-9B | JPN | 0.5730 | 0.1122 | +0.117 | 18.99 | 0.6% |
| Gemma-2-9B | DEU | 0.3418 | 0.0595 | +0.684 | 11.79 | 1.3% |
| Gemma-2-9B | BRA | 0.3873 | 0.0753 | +0.074 | 14.93 | 2.9% |
| Mistral-7B | USA | 0.6368 | 0.1398 | -0.619 | 23.10 | 1.0% |
| Mistral-7B | CHN | 0.5053 | 0.1049 | -0.665 | 17.60 | 0.3% |
| Mistral-7B | JPN | 0.3508 | 0.0767 | -0.911 | 12.74 | 0.0% |
| Mistral-7B | DEU | 0.5106 | 0.1094 | -0.962 | 18.08 | 0.3% |
| Mistral-7B | BRA | 0.4447 | 0.0973 | -0.696 | 14.39 | 0.3% |

#### EXP-03 SocialValue-Targeted Personas (full metrics)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Qwen2.5-7B | USA | **0.2491** | 0.0388 | +0.704 | **7.56** | 0.0% |
| Qwen2.5-7B | CHN ⚠️ | 0.2930 | 0.0614 | +0.271 | 11.24 | 3.2% |
| Qwen2.5-7B | JPN | 0.2925 | 0.0388 | +0.527 | 10.37 | 1.9% |
| Qwen2.5-7B | DEU | 0.3827 | 0.0569 | +0.004 | 12.77 | 1.3% |
| Qwen2.5-7B | BRA | 0.4041 | 0.0870 | -0.327 | 15.55 | 1.9% |
| Gemma-2-9B | USA | 0.5497 | 0.1199 | +0.397 | 17.63 | 1.6% |
| Gemma-2-9B | CHN ⚠️ | 0.6321 | 0.1622 | +0.445 | 23.05 | 0.6% |
| Gemma-2-9B | JPN | 0.5265 | 0.1050 | +0.171 | 16.39 | 1.0% |
| Gemma-2-9B | DEU | 0.3948 | 0.0760 | +0.420 | 10.86 | 2.6% |
| Gemma-2-9B | BRA | 0.4406 | 0.0993 | -0.235 | 16.38 | 3.5% |
| Mistral-7B | USA | 0.6500 | 0.1476 | -0.565 | 22.31 | 0.6% |
| Mistral-7B | CHN ⚠️ | 0.4925 | 0.1040 | -0.553 | 17.61 | 0.3% |
| Mistral-7B | JPN | 0.3341 | 0.0733 | -0.871 | 12.23 | 0.3% |
| Mistral-7B | DEU | 0.4947 | 0.1060 | -0.942 | 18.85 | 0.0% |
| Mistral-7B | BRA | 0.4434 | 0.0966 | -0.687 | 13.39 | 0.6% |

#### EXP-07a WVS Augmentation (full metrics; Qwen+Gemma only)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Qwen2.5-7B | BRA | 0.3792 | 0.0855 | +0.105 | 13.64 | 1.0% |
| Qwen2.5-7B | JPN | 0.2801 | 0.0551 | +0.406 | 9.93 | 1.0% |
| Qwen2.5-7B | DEU | 0.3444 | 0.0559 | +0.460 | 10.55 | 2.3% |
| Qwen2.5-7B | USA | 0.3687 | 0.0763 | +0.631 | 9.72 | 2.3% |
| Qwen2.5-7B | CHN ⚠️ | 0.4094 | 0.0956 | +0.422 | 13.66 | 0.0% |
| Gemma-2-9B | BRA | 0.4002 | 0.0799 | -0.136 | 14.58 | 1.9% |
| Gemma-2-9B | JPN | 0.4616 | 0.0784 | +0.339 | 15.70 | 1.6% |
| Gemma-2-9B | DEU | 0.3286 | 0.0673 | +0.796 | 10.19 | 0.6% |
| Gemma-2-9B | USA | 0.6067 | 0.1112 | +0.626 | 22.45 | 0.3% |
| Gemma-2-9B | CHN ⚠️ | 0.4517 | 0.1024 | +0.782 | 14.48 | 0.3% |

#### EXP-04 Mistral Cross-Lingual (English personas + σ₀=0.8 + K=512 + T=0.5; Mistral-only)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Mistral-7B | USA | 0.6666 | 0.1538 | -0.339 | 23.33 | 0.3% |
| Mistral-7B | CHN ⚠️ | **0.3091** | 0.0676 | **+0.573** | 10.45 | 0.0% |
| Mistral-7B | JPN | 0.3221 | 0.0745 | +0.461 | 10.65 | 0.3% |
| Mistral-7B | DEU | 0.4091 | **0.0574** | +0.227 | 13.09 | 0.3% |
| Mistral-7B | BRA | 0.5246 | 0.1153 | -0.339 | 20.34 | 1.0% |

#### EXP-09 Hierarchical IS (full metrics)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Qwen2.5-7B | USA | 0.3538 | 0.0505 | +0.685 | 12.23 | 17.7% |
| Qwen2.5-7B | CHN ⚠️ | 0.3526 | 0.0681 | +0.349 | 11.35 | 18.1% |
| Qwen2.5-7B | JPN | 0.3392 | 0.0445 | +0.388 | 11.27 | 14.8% |
| Qwen2.5-7B | DEU | 0.4262 | 0.0526 | +0.357 | 14.56 | 16.5% |
| Qwen2.5-7B | BRA | 0.3546 | 0.0662 | +0.058 | 11.69 | 14.2% |
| Gemma-2-9B | USA | 0.4922 | 0.0556 | +0.675 | 18.32 | 11.9% |
| Gemma-2-9B | CHN ⚠️ | 0.3592 | 0.0494 | +0.784 | 12.78 | 12.9% |
| Gemma-2-9B | JPN | 0.4411 | 0.0565 | +0.346 | 15.56 | 13.5% |
| Gemma-2-9B | DEU | 0.3653 | 0.0461 | +0.772 | 13.19 | 15.8% |
| Gemma-2-9B | BRA | 0.3438 | 0.0493 | +0.336 | 11.49 | 17.7% |
| Mistral-7B | USA | 0.5266 | 0.1031 | -0.569 | 19.76 | 14.5% |
| Mistral-7B | CHN ⚠️ | 0.4099 | 0.0837 | -0.612 | 16.25 | 17.7% |
| Mistral-7B | JPN | 0.3273 | 0.0603 | -0.644 | 12.29 | 16.1% |
| Mistral-7B | DEU | 0.4634 | 0.0837 | -0.884 | 16.06 | 17.7% |
| Mistral-7B | BRA | 0.4138 | 0.0677 | -0.710 | 13.34 | 16.8% |

> EXP-09 has the best mean MIS overall, but it comes with much higher flip% than EXP-01/02 and persistent SocialValue/Species errors.

#### EXP-05 ESS-Adaptive Anchor Regularization (full metrics)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Qwen2.5-7B | USA | 0.3628 | 0.0674 | +0.610 | 10.20 | 6.1% |
| Qwen2.5-7B | CHN ⚠️ | 0.3791 | 0.0885 | +0.372 | 12.35 | 5.2% |
| Qwen2.5-7B | JPN | **0.2493** | **0.0442** | +0.543 | **8.70** | 8.7% |
| Qwen2.5-7B | DEU | **0.3140** | 0.0520 | +0.445 | 10.38 | 17.1% |
| Qwen2.5-7B | BRA | 0.4493 | 0.0965 | -0.218 | 16.67 | 8.1% |
| Gemma-2-9B | USA | 0.5599 | 0.0921 | +0.550 | 19.96 | 4.8% |
| Gemma-2-9B | CHN ⚠️ | 0.4002 | 0.0744 | +0.770 | 12.73 | 13.2% |
| Gemma-2-9B | JPN | 0.5012 | 0.0832 | +0.167 | 17.27 | 7.1% |
| Gemma-2-9B | DEU | 0.3420 | 0.0532 | +0.717 | 11.52 | 11.3% |
| Gemma-2-9B | BRA | **0.3446** | 0.0641 | +0.091 | 12.83 | 16.8% |
| Mistral-7B | USA | 0.6303 | 0.1385 | -0.653 | 23.25 | 5.5% |
| Mistral-7B | CHN ⚠️ | 0.4764 | 0.1003 | -0.657 | 17.12 | 1.6% |
| Mistral-7B | JPN | 0.3442 | 0.0755 | -0.944 | 12.31 | 3.2% |
| Mistral-7B | DEU | 0.4889 | 0.1042 | -0.984 | 17.39 | 4.5% |
| Mistral-7B | BRA | 0.4195 | 0.0822 | -0.689 | 13.51 | 5.8% |

> Note: the run printed `mean_alpha_reg` / `mean_anchor_div` as NaN in `comparison.csv`, indicating those columns were not present in `results_df` (diagnostic logging bug in EXP-05 script).

### Leaderboard (per-dimension “worst gaps” snapshot)

This table is meant for the paper’s diagnostic narrative (“where alignment fails”).

| Method | Model | Country | Worst dim | |err| (pp) | Notes |
|:------|:------|:-------:|:----------|----------:|:------|
| EXP-02 | Qwen | USA | SocialValue_High | 30.4 | SocialValue is consistently top error |
| EXP-07a | Qwen | BRA | SocialValue_High | 27.5 | augmentation helps MIS but not SocialValue |
| EXP-09 | Qwen | USA | SocialValue_High | 28.7 | high flip% (17.7%) despite better MIS |
| EXP-09 | Gemma | USA | Age_Young | 27.3 | age + social-value underestimation |
| EXP-09 | Gemma | DEU | SocialValue_High | 25.4 | SocialValue remains dominant |
| EXP-09 | Mistral | USA | Utilitarianism_More | 30.8 | strongest anti-correlation risk cluster |
| EXP-09 | Mistral | DEU | Species_Humans | 30.7 | high flip% (17.7%) + negative r |
| EXP-09 | Mistral | BRA | Age_Young | 30.0 | high flip% (16.8%) + age underestimation |

> Full per-dimension tables are stored as `CMP_ROOT/per_dim_breakdown.csv` for each run.

---

## Failure Modes Being Addressed

### Insight 1 — SocialValue Underestimation (ALL models)
- **Target file**: `exp03_socialvalue_personas.py`, `exp06_category_routing.py`
- EXP-02 confirmed: mean SocialValue error = **27–30pp** across all models/countries
  - Qwen: SV err ≈ 20–30pp | Gemma: SV err ≈ 17–29pp | Mistral: SV err ≈ 1–17pp
- EXP-02 urban/rural personas did NOT fix SocialValue → 8 agents still under-assign social value
- Root cause confirmed: WVS personas are structurally egalitarian → anchor < 0 for SocialValue.
- EXP-03 ran a **social-utility persona augmentation** (5 personas: 3 WVS + 2 targeted P4/P5):
  - **Qwen**: SocialValue_High error drops meaningfully (e.g., USA 27.0pp → 14.1pp; CHN 20–25pp → 5.5pp).
  - **Gemma/Mistral**: SocialValue improves in some countries but large Age/Utilitarianism errors remain; anti-correlation persists.

### Insight 2 — Mistral Variance Collapse (CONFIRMED & WORSENED in EXP-02)
- **Target file**: `exp04_mistral_crosslingual.py`
- EXP-02 results: JPN variance = **0.065** (collapsed!), Pearson r = **-0.911** (anti-correlated!)
- DEU: Pearson r = **-0.962**, Spearman ρ = **-0.943** → severe ranking inversion
- BRA: Pearson r = **-0.696**, Age_Young err = **37.8pp**
- Urban/rural modulation had no effect on SentencePiece collapse
- EXP-04 ran the planned mitigation (English-only personas + σ₀=0.8 + K=512 + T=0.5) with **mixed** results:
  - **CHN/DEU/JPN**: Pearson r became **positive** and MIS improved substantially vs EXP-01/02.
  - **USA/BRA**: Pearson r remained **negative** and MIS regressed (still unstable / collapse warnings observed).

### Insight 3 — Gemma Over-Correction (Partially improved vs EXP-01)
- **Target file**: `exp05_anchor_regularization.py`
- EXP-02: Gemma DEU MIS = 0.342 (best!), BRA MIS = 0.387 — modest improvement
- But USA MIS = 0.607, JPN MIS = 0.573 remain high
- Specific failure: JPN Utilitarianism_More error = **44.7pp** (model=24.0, human=68.7)
- Root cause: egalitarian anchor << delta_base → over-correction persists with 8 agents
- Fix needed: ESS-adaptive anchor regularization (EXP-05)

### Insight 4 — EXP-02 Net Impact vs EXP-01 (Urban/Rural Axis)
- See the **paper-style MIS comparison blocks** in the EXP-02 section below (format matched to `experiment/kaggle_experiment.py`).
- **Conclusion**: EXP-02 yields a small net gain for Qwen on average, but regresses for Gemma and Mistral. Urban/rural axis alone does **not** reliably reduce MIS; model-specific failure modes remain the dominant drivers (EXP-04, EXP-05).
- **Correction note**: Earlier Insight 4 accidentally compared EXP-02 Qwen JPN to **Mistral** EXP-01 JPN (0.3502) instead of **Qwen** EXP-01 JPN (0.2802). Fixed in the tables below.

---

## Results Log

### EXP-01 — Paper Baseline SWA-PTIS 4-Agent (✅ Completed 2026-04-09)

**Config**: N=4 personas (WVS young/middle/older + utilitarian), λ_coop=0.70, K=128, σ₀=0.30, seed=42, n_scenarios=310 (after quality filter). Both vanilla baseline AND SWA-PTIS run for all 3 models.

#### EXP-01 Vanilla Baseline MIS (no persona, raw LLM)

| Model | Country | Vanilla MIS |
|:------|:-------:|:-----------:|
| Qwen2.5-7B | USA | 0.4559 |
| Qwen2.5-7B | CHN ⚠️ | 0.4646 |
| Qwen2.5-7B | JPN | 0.4208 |
| Qwen2.5-7B | DEU | 0.4398 |
| Qwen2.5-7B | BRA | 0.5111 |
| Gemma-2-9B | USA | 0.4647 |
| Gemma-2-9B | CHN ⚠️ | 0.3679 |
| Gemma-2-9B | JPN | 0.4530 |
| Gemma-2-9B | DEU | 0.4170 |
| Gemma-2-9B | BRA | **0.4490** |
| Mistral-7B | USA | **0.5706** |
| Mistral-7B | CHN ⚠️ | **0.4569** |
| Mistral-7B | JPN | **0.3429** |
| Mistral-7B | DEU | **0.4909** |
| Mistral-7B | BRA | **0.4144** |

> ⚠️ CHN uses Afrikaans fallback data — bug in `data.py`.

#### EXP-01 SWA-PTIS MIS (4-agent, paper pipeline)

| Model | Country | SWA MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:---------:|:-----:|:-----------:|:-----:|:-----:|
| Qwen2.5-7B | USA | **0.3677** | 0.0759 | +0.639 | 9.75 | 1.9% |
| Qwen2.5-7B | CHN ⚠️ | 0.4078 | 0.0956 | +0.418 | 13.54 | 1.3% |
| Qwen2.5-7B | JPN | **0.2802** | 0.0553 | +0.406 | 9.97 | 0.6% |
| Qwen2.5-7B | DEU | 0.3424 | 0.0558 | +0.461 | 10.43 | 2.6% |
| Qwen2.5-7B | BRA | 0.4025 | 0.0942 | +0.167 | 14.30 | 0.6% |
| Gemma-2-9B | USA | 0.6038 | 0.1108 | +0.630 | 22.31 | 0.6% |
| Gemma-2-9B | CHN ⚠️ | 0.4536 | 0.1034 | +0.777 | 14.68 | 1.3% |
| Gemma-2-9B | JPN | 0.4667 | 0.0794 | +0.332 | 15.85 | 1.6% |
| Gemma-2-9B | DEU | **0.3289** | 0.0683 | +0.795 | 10.13 | 1.3% |
| Gemma-2-9B | BRA | 0.3655 | 0.0749 | +0.272 | 14.09 | 3.9% |
| Mistral-7B | USA | 0.5984 | 0.1303 | -0.570 | 21.61 | 0.6% |
| Mistral-7B | CHN ⚠️ | 0.5067 | 0.1050 | -0.682 | 17.41 | 0.3% |
| Mistral-7B | JPN | 0.3502 | 0.0765 | **-0.905** | 12.46 | 1.3% |
| Mistral-7B | DEU | 0.4942 | 0.1060 | **-0.957** | 17.18 | 1.0% |
| Mistral-7B | BRA | 0.4362 | 0.0947 | -0.665 | 14.02 | 0.3% |

#### EXP-01 MIS Comparison: Vanilla → SWA-PTIS

| Model | Country | Vanilla MIS | SWA MIS | Δ | Improv% | Win? |
|:------|:-------:|:-----------:|:-------:|:-:|:-------:|:----:|
| Qwen | USA | 0.4559 | 0.3677 | +0.0882 | **+19.34%** | ✅ |
| Qwen | CHN ⚠️ | 0.4646 | 0.4078 | +0.0568 | **+12.22%** | ✅ |
| Qwen | JPN | 0.4208 | 0.2802 | +0.1405 | **+33.40%** | ✅ |
| Qwen | DEU | 0.4398 | 0.3424 | +0.0974 | **+22.15%** | ✅ |
| Qwen | BRA | 0.5111 | 0.4025 | +0.1086 | **+21.26%** | ✅ |
| Gemma | USA | 0.4647 | 0.6038 | -0.1391 | -29.95% | ❌ |
| Gemma | CHN ⚠️ | 0.3679 | 0.4536 | -0.0857 | -23.28% | ❌ |
| Gemma | JPN | 0.4530 | 0.4667 | -0.0136 | -3.01% | ❌ |
| Gemma | DEU | 0.4170 | 0.3289 | +0.0882 | **+21.14%** | ✅ |
| Gemma | BRA | 0.4490 | 0.3655 | +0.0834 | **+18.58%** | ✅ |
| Mistral | USA | 0.5706 | 0.5984 | -0.0278 | -4.87% | ❌ |
| Mistral | CHN ⚠️ | 0.4569 | 0.5067 | -0.0498 | -10.90% | ❌ |
| Mistral | JPN | 0.3429 | 0.3502 | -0.0073 | -2.12% | ❌ |
| Mistral | DEU | 0.4909 | 0.4942 | -0.0033 | -0.67% | ❌ |
| Mistral | BRA | 0.4144 | 0.4362 | -0.0217 | -5.25% | ❌ |

#### EXP-01 Summary (SWA vs Vanilla)

| Model | Win Rate | Baseline MIS | SWA MIS | Macro Δ% | Micro Δ% |
|:------|:--------:|:------------:|:-------:|:--------:|:--------:|
| Qwen2.5-7B | **5/5** | 0.4584 | 0.3601 | **+21.45%** | **+21.68%** |
| Gemma-2-9B | 2/5 | 0.4303 | 0.4437 | **-3.11%** | **-3.30%** |
| Mistral-7B | 0/5 | 0.4551 | 0.4771 | **-4.83%** | **-4.76%** |

#### EXP-01 Notable Per-Dimension Errors (Qwen SWA)

| Country | Worst Dim | Human | Model | Error |
|:-------:|:---------:|:-----:|:-----:|:-----:|
| USA | SocialValue_High | 67.9 | 33.5 | **34.5pp** |
| CHN ⚠️ | SocialValue_High | 66.7 | 35.3 | **31.4pp** |
| JPN | SocialValue_High | 65.9 | 47.9 | **18.0pp** |
| DEU | SocialValue_High | 64.7 | 42.4 | **22.3pp** |
| BRA | SocialValue_High | 66.3 | 37.4 | **28.9pp** |

> SocialValue underestimation present in ALL countries even for best-performing Qwen.

---

### EXP-02 — Expanded Personas 8-Agent (✅ Completed 2026-04-09)

**Config**: N=8 personas (3 WVS base + urban-young + rural-young + urban-older + utilitarian + global-citizen), λ_coop=0.75, K=256, seed=42, n_scenarios=310 (after quality filter from 500 target).

#### EXP-02 Raw Results

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | Cosine ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:--------:|:-----:|:-----:|
| Qwen2.5-7B | USA | 0.3496 | 0.0636 | +0.625 | +0.625 | 10.30 | 0.3% |
| Qwen2.5-7B | CHN | 0.3680 | 0.0842 | +0.343 | +0.343 | 12.56 | 0.6% |
| Qwen2.5-7B | JPN | **0.2808** | **0.0488** | +0.366 | +0.366 | 9.29 | 0.3% |
| Qwen2.5-7B | DEU | 0.3895 | 0.0592 | +0.256 | +0.256 | 12.53 | 1.3% |
| Qwen2.5-7B | BRA | 0.3904 | 0.0880 | -0.010 | -0.010 | 13.20 | 0.3% |
| Gemma-2-9B | USA | 0.6073 | 0.1098 | +0.557 | +0.557 | 22.12 | 0.6% |
| Gemma-2-9B | CHN | 0.4095 | 0.0860 | +0.709 | +0.709 | 14.25 | 1.0% |
| Gemma-2-9B | JPN | 0.5730 | 0.1122 | +0.117 | +0.117 | 18.99 | 0.6% |
| Gemma-2-9B | DEU | **0.3418** | 0.0595 | +0.684 | +0.684 | 11.79 | 1.3% |
| Gemma-2-9B | BRA | 0.3873 | 0.0753 | +0.074 | +0.074 | 14.93 | 2.9% |
| Mistral-7B | USA | 0.6368 | 0.1398 | -0.619 | -0.619 | 23.10 | 1.0% |
| Mistral-7B | CHN | 0.5053 | 0.1049 | -0.665 | -0.665 | 17.60 | 0.3% |
| Mistral-7B | JPN | 0.3508 | 0.0767 | **-0.911** | **-0.911** | 12.74 | 0.0% |
| Mistral-7B | DEU | 0.5106 | 0.1094 | **-0.962** | **-0.962** | 18.08 | 0.3% |
| Mistral-7B | BRA | 0.4447 | 0.0973 | -0.696 | -0.696 | 14.39 | 0.3% |

#### EXP-02 Per-Dimension Worst Errors (notable findings)

| Model | Country | Worst Dim | Human | Model | Error |
|:------|:-------:|:---------:|:-----:|:-----:|:-----:|
| All | All | SocialValue_High | ~66% | ~40% | **~27pp** |
| Gemma | JPN | Utilitarianism_More | 68.7 | 24.0 | **44.7pp** |
| Mistral | DEU | Species_Humans | 82.4 | 49.4 | **33.0pp** |
| Mistral | USA | Utilitarianism_More | 76.6 | 39.9 | **36.7pp** |
| Qwen | BRA | Age_Young | 73.6 | 49.9 | **23.7pp** |

#### EXP-02 vs EXP-01 Comparison (Qwen, MIS) — corrected

Below is the **paper-style MIS improvement printout** (same layout as `experiment/kaggle_experiment.py`), comparing EXP-01 SWA (4-agent) → EXP-02 SWA (8-agent).  
Notation: `delta = MIS_exp01 - MIS_exp02` so **positive delta = EXP-02 improved**.

##### Qwen2.5-7B — MIS improvement (EXP-01 → EXP-02)

| country | exp01_mis | exp02_mis | delta | improv % | win |
|:------:|----------:|----------:|------:|---------:|:---:|
| USA | 0.3677 | 0.3496 | +0.0181 | +4.92% | ✅ |
| CHN ⚠️ | 0.4078 | 0.3680 | +0.0398 | +9.76% | ✅ |
| JPN | 0.2802 | 0.2808 | -0.0006 | -0.21% | ❌ |
| DEU | 0.3424 | 0.3895 | -0.0471 | -13.76% | ❌ |
| BRA | 0.4025 | 0.3904 | +0.0121 | +3.01% | ✅ |

- Mean exp01 MIS: 0.3601  
- Mean exp02 MIS: 0.3557  
- Absolute reduction: **+0.0045**  
- Improvement on the means (macro-average): **+1.24%**  
- Mean per-country improvement (micro-average): **+0.74%**  
- Wins (MIS lower): **3/5**

##### Gemma-2-9B — MIS improvement (EXP-01 → EXP-02)

| country | exp01_mis | exp02_mis | delta | improv % | win |
|:------:|----------:|----------:|------:|---------:|:---:|
| USA | 0.6038 | 0.6073 | -0.0035 | -0.58% | ❌ |
| CHN ⚠️ | 0.4536 | 0.4095 | +0.0441 | +9.72% | ✅ |
| JPN | 0.4667 | 0.5730 | -0.1063 | -22.78% | ❌ |
| DEU | 0.3289 | 0.3418 | -0.0129 | -3.92% | ❌ |
| BRA | 0.3655 | 0.3873 | -0.0218 | -5.96% | ❌ |

- Mean exp01 MIS: 0.4437  
- Mean exp02 MIS: 0.4638  
- Absolute reduction: **-0.0201**  
- Improvement on the means (macro-average): **-4.53%**  
- Mean per-country improvement (micro-average): **-4.70%**  
- Wins (MIS lower): **1/5**

##### Mistral-7B — MIS improvement (EXP-01 → EXP-02)

| country | exp01_mis | exp02_mis | delta | improv % | win |
|:------:|----------:|----------:|------:|---------:|:---:|
| USA | 0.5984 | 0.6368 | -0.0384 | -6.41% | ❌ |
| CHN ⚠️ | 0.5067 | 0.5053 | +0.0014 | +0.28% | ✅ |
| JPN | 0.3502 | 0.3508 | -0.0006 | -0.17% | ❌ |
| DEU | 0.4942 | 0.5106 | -0.0164 | -3.32% | ❌ |
| BRA | 0.4362 | 0.4447 | -0.0085 | -1.95% | ❌ |

- Mean exp01 MIS: 0.4771  
- Mean exp02 MIS: 0.4896  
- Absolute reduction: **-0.0125**  
- Improvement on the means (macro-average): **-2.62%**  
- Mean per-country improvement (micro-average): **-2.31%**  
- Wins (MIS lower): **1/5**

> **Hypothesis check (Confucian cluster)**: For Qwen, JPN MIS is essentially unchanged (0.2802 → 0.2808), so the “urban–rural expanded pool improves Confucian countries” hypothesis is **not supported** by MIS here.

---

### EXP-06b — Category-Routed Persona Pools (✅ Completed 2026-04-09)

**Script**: `exp06_category_routing.py`  
**Idea**: route 6 category-specific persona pools (Species/Gender/Age/Fitness/SocialValue/Utilitarianism) while keeping EXP-01 hyperparameters unchanged.

#### EXP-06b Raw Results (all 3 models × 5 countries)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | Cosine ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:--------:|:-----:|:-----:|
| Qwen2.5-7B | USA | 0.3677 | 0.0759 | +0.639 | +0.639 | 9.75 | 1.9% |
| Qwen2.5-7B | CHN ⚠️ | 0.4078 | 0.0956 | +0.418 | +0.418 | 13.54 | 1.3% |
| Qwen2.5-7B | JPN | 0.2802 | 0.0553 | +0.406 | +0.406 | 9.97 | 0.6% |
| Qwen2.5-7B | DEU | 0.3424 | 0.0558 | +0.461 | +0.461 | 10.43 | 2.6% |
| Qwen2.5-7B | BRA | 0.4025 | 0.0942 | +0.167 | +0.167 | 14.30 | 0.6% |
| Gemma-2-9B | USA | 0.6038 | 0.1108 | +0.630 | +0.630 | 22.31 | 0.6% |
| Gemma-2-9B | CHN ⚠️ | 0.4536 | 0.1034 | +0.777 | +0.777 | 14.68 | 1.3% |
| Gemma-2-9B | JPN | 0.4667 | 0.0794 | +0.332 | +0.332 | 15.85 | 1.6% |
| Gemma-2-9B | DEU | 0.3289 | 0.0683 | +0.795 | +0.795 | 10.13 | 1.3% |
| Gemma-2-9B | BRA | 0.3655 | 0.0749 | +0.272 | +0.272 | 14.09 | 3.9% |
| Mistral-7B | USA | 0.5984 | 0.1303 | -0.570 | -0.570 | 21.61 | 0.6% |
| Mistral-7B | CHN ⚠️ | 0.5067 | 0.1050 | -0.682 | -0.682 | 17.41 | 0.3% |
| Mistral-7B | JPN | 0.3502 | 0.0765 | -0.905 | -0.905 | 12.46 | 1.3% |
| Mistral-7B | DEU | 0.4942 | 0.1060 | -0.957 | -0.957 | 17.18 | 1.0% |
| Mistral-7B | BRA | 0.4362 | 0.0947 | -0.665 | -0.665 | 14.02 | 0.3% |

#### EXP-06b vs EXP-01 SWA (MIS)

| Model | Mean MIS (EXP-01) | Mean MIS (EXP-06b) | Delta |
|:------|-------------------:|-------------------:|------:|
| Qwen2.5-7B | 0.3601 | 0.3601 | +0.0000 |
| Gemma-2-9B | 0.4437 | 0.4437 | +0.0000 |
| Mistral-7B | 0.4771 | 0.4771 | +0.0000 |
| **Overall (15 rows)** | **0.4269** | **0.4269** | **+0.0000** |

#### EXP-06b key takeaways

- Category-routing run completed successfully, but **aggregate metrics are effectively identical to EXP-01**.
- The expected SocialValue gain (**27pp -> <10pp**) did **not** materialize in this run.
- Most likely action item: verify that routed personas are actually used inside the logits path (prefix build / persona dispatch coupling), not just logged.

---

### EXP-05 — ESS-Adaptive Anchor Regularization (✅ Completed 2026-04-09)

**Script**: `exp05_anchor_regularization.py`  
**Idea**: regularize toward the base model when ESS is low:
\[
\delta_{\text{opt}}=\alpha\cdot \text{anchor}+(1-\alpha)\cdot \delta_{\text{base}}+\delta_\star,\quad
\alpha=\mathrm{clip}(\mathrm{ESS}/K,\rho_{\text{eff}},1)
\]

#### EXP-05 Raw Results (all 3 models × 5 countries)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | Cosine ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:--------:|:-----:|:-----:|
| Qwen2.5-7B | USA | 0.3628 | 0.0674 | +0.610 | +0.610 | 10.20 | 6.1% |
| Qwen2.5-7B | CHN ⚠️ | 0.3791 | 0.0885 | +0.372 | +0.372 | 12.35 | 5.2% |
| Qwen2.5-7B | JPN | **0.2493** | **0.0442** | +0.543 | +0.543 | **8.70** | 8.7% |
| Qwen2.5-7B | DEU | **0.3140** | 0.0520 | +0.445 | +0.445 | 10.38 | 17.1% |
| Qwen2.5-7B | BRA | 0.4493 | 0.0965 | -0.218 | -0.218 | 16.67 | 8.1% |
| Gemma-2-9B | USA | 0.5599 | 0.0921 | +0.550 | +0.550 | 19.96 | 4.8% |
| Gemma-2-9B | CHN ⚠️ | 0.4002 | 0.0744 | +0.770 | +0.770 | 12.73 | 13.2% |
| Gemma-2-9B | JPN | 0.5012 | 0.0832 | +0.167 | +0.167 | 17.27 | 7.1% |
| Gemma-2-9B | DEU | 0.3420 | 0.0532 | +0.717 | +0.717 | 11.52 | 11.3% |
| Gemma-2-9B | BRA | **0.3446** | 0.0641 | +0.091 | +0.091 | 12.83 | 16.8% |
| Mistral-7B | USA | 0.6303 | 0.1385 | -0.653 | -0.653 | 23.25 | 5.5% |
| Mistral-7B | CHN ⚠️ | 0.4764 | 0.1003 | -0.657 | -0.657 | 17.12 | 1.6% |
| Mistral-7B | JPN | 0.3442 | 0.0755 | -0.944 | -0.944 | 12.31 | 3.2% |
| Mistral-7B | DEU | 0.4889 | 0.1042 | -0.984 | -0.984 | 17.39 | 4.5% |
| Mistral-7B | BRA | 0.4195 | 0.0822 | -0.689 | -0.689 | 13.51 | 5.8% |

#### EXP-05 key takeaways

- **Mean MIS (overall 15 rows)**: **0.4174** (2nd best after EXP-09).
- **Qwen**: big wins on JPN/DEU; BRA regresses vs EXP-01.
- **Gemma**: reduces the worst failures (USA/CHN) but not fully fixed.
- **Mistral**: still anti-correlated in some countries; EXP-04 helps CHN/DEU/JPN but does **not** fully fix USA/BRA.
- **Diagnostics missing**: `mean_alpha_reg` / `mean_anchor_div` are **NaN** in `comparison.csv` → need to ensure `results_df` contains `alpha_reg` / `anchor_divergence` columns so run-level means can be computed.

---

### EXP-04 — Mistral Cross-Lingual English Override + Variance Floor Fix (✅ Completed 2026-04-09)

**Script**: `exp04_mistral_crosslingual.py`  
**Idea**: Mitigate Mistral’s multilingual variance collapse by forcing **English personas** (for all countries) and increasing exploration/robustness (σ₀=0.8, K=512, decision T=0.5).

**Run config (this run)**:
- Model: Mistral-7B (Unsloth 4-bit)
- Countries: USA, CHN ⚠️, JPN, DEU, BRA
- n_scenarios: 310 (after quality filter)
- Personas: forced English for all 5 countries (`persona_lang=en`)

#### EXP-04 Raw Results (Mistral-only, 5 countries)

| Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| USA | 0.6666 | 0.1538 | -0.339 | 23.33 | 0.3% |
| CHN ⚠️ | **0.3091** | 0.0676 | **+0.573** | **10.45** | 0.0% |
| JPN | 0.3221 | 0.0745 | +0.461 | 10.65 | 0.3% |
| DEU | 0.4091 | **0.0574** | +0.227 | 13.09 | 0.3% |
| BRA | 0.5246 | 0.1153 | -0.339 | 20.34 | 1.0% |

#### EXP-04 vs EXP-01 SWA-PTIS (Mistral MIS)

`delta = MIS_EXP-01 - MIS_EXP-04` (positive = EXP-04 improved)

| Country | EXP-01 MIS | EXP-04 MIS | Δ | Outcome |
|:-------:|-----------:|-----------:|--:|:--------|
| USA | 0.5984 | 0.6666 | -0.0682 | worse |
| CHN ⚠️ | 0.5067 | **0.3091** | **+0.1976** | win |
| JPN | 0.3502 | 0.3221 | +0.0281 | win |
| DEU | 0.4942 | 0.4091 | +0.0851 | win |
| BRA | 0.4362 | 0.5246 | -0.0884 | worse |

**Key takeaways**:
- **Correlation**: Sign flipped to **positive** for CHN/JPN/DEU, but **still negative** for USA/BRA.
- **Not fully solved**: Many per-scenario logs still emitted “variance collapsed” warnings → English personas + σ₀ floor reduces the worst collapses in some countries but doesn’t reliably stabilize all.

---

### EXP-07/08 — results pending Kaggle run

| EXP | Model | Country | Baseline MIS | EXP MIS | Δ | Status |
|:----|:-----:|:-------:|:------------:|:-------:|:-:|:------:|
| 07 | All | All 15 | — | — | — | ⏳ |

---

### EXP-07a — WVS Augmentation for Sparse Countries (✅ Completed 2026-04-09)

**Script**: `exp07_wvs_augmentation.py`  
**Idea**: If a country has sparse WVS coverage, augment its persona pool by borrowing “culturally similar” voices from Hofstede-nearest neighbors (kernel-weighted).

**Run config (this run)**:
- Models: Qwen2.5-7B, Gemma-2-9B
- Countries: BRA (sparse) + JPN/DEU/USA/CHN (dense controls)
- N_THRESHOLD=200, K_neighbors=3, tau=0.15
- Persona counts observed: BRA=7 (augmented), dense=4
- Note: CHN still uses Afrikaans fallback MultiTP data (dataset bug) ⚠️

#### EXP-07a Raw Results

| Model | Country | Sparse? | N personas | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:------:|:---------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Qwen2.5-7B | BRA | ✅ | 7 | 0.3792 | 0.0855 | +0.105 | 13.64 | 1.0% |
| Qwen2.5-7B | JPN | ❌ | 4 | **0.2801** | 0.0551 | +0.406 | 9.93 | 1.0% |
| Qwen2.5-7B | DEU | ❌ | 4 | 0.3444 | 0.0559 | +0.460 | 10.55 | 2.3% |
| Qwen2.5-7B | USA | ❌ | 4 | 0.3687 | 0.0763 | +0.631 | 9.72 | 2.3% |
| Qwen2.5-7B | CHN ⚠️ | ❌ | 4 | 0.4094 | 0.0956 | +0.422 | 13.66 | 0.0% |
| Gemma-2-9B | BRA | ✅ | 7 | 0.4002 | 0.0799 | -0.136 | 14.58 | 1.9% |
| Gemma-2-9B | JPN | ❌ | 4 | 0.4616 | 0.0784 | +0.339 | 15.70 | 1.6% |
| Gemma-2-9B | DEU | ❌ | 4 | **0.3286** | 0.0673 | +0.796 | 10.19 | 0.6% |
| Gemma-2-9B | USA | ❌ | 4 | 0.6067 | 0.1112 | +0.626 | 22.45 | 0.3% |
| Gemma-2-9B | CHN ⚠️ | ❌ | 4 | 0.4517 | 0.1024 | +0.782 | 14.48 | 0.3% |

#### EXP-07a vs EXP-01 SWA-PTIS (MIS) — same 5 countries

Notation: `delta = MIS_exp01 - MIS_exp07a` so **positive delta = EXP-07a improved**.

##### Qwen2.5-7B — MIS improvement (EXP-01 → EXP-07a)

| country | exp01_mis | exp07a_mis | delta | improv % | win |
|:------:|----------:|-----------:|------:|---------:|:---:|
| BRA | 0.4025 | 0.3792 | +0.0233 | +5.79% | ✅ |
| JPN | 0.2802 | 0.2801 | +0.0001 | +0.04% | ✅ |
| DEU | 0.3424 | 0.3444 | -0.0020 | -0.60% | ❌ |
| USA | 0.3677 | 0.3687 | -0.0010 | -0.27% | ❌ |
| CHN ⚠️ | 0.4078 | 0.4094 | -0.0016 | -0.39% | ❌ |

##### Gemma-2-9B — MIS improvement (EXP-01 → EXP-07a)

| country | exp01_mis | exp07a_mis | delta | improv % | win |
|:------:|----------:|-----------:|------:|---------:|:---:|
| BRA | 0.3655 | 0.4002 | -0.0347 | -9.49% | ❌ |
| JPN | 0.4667 | 0.4616 | +0.0051 | +1.09% | ✅ |
| DEU | 0.3289 | 0.3286 | +0.0003 | +0.09% | ✅ |
| USA | 0.6038 | 0.6067 | -0.0029 | -0.48% | ❌ |
| CHN ⚠️ | 0.4536 | 0.4517 | +0.0019 | +0.42% | ✅ |

#### Key takeaways
- **BRA (Qwen)**: MIS improves **0.4025 → 0.3792** (**+5.8%**) from augmentation, but **SocialValue_High still dominates error** (≈27.5pp).
- **BRA (Gemma)**: MIS worsens **0.3655 → 0.4002** (augmentation not helpful here).
- **Net**: augmentation helps a subset of sparse-country cases but does **not** resolve the core SocialValue bias; still needs EXP-03/06/07.

---

### EXP-09 — Hierarchical IS (✅ Completed 2026-04-09)

**Script**: `exp09_hierarchical_is.py`  
**Idea**: Add a country-level prior \(\delta_{\text{country}}\) (EMA over scenarios) and anneal toward it to reduce IS drift/instability.

#### EXP-09 MIS (all 3 models × 5 countries)

| Model | Country | MIS ↓ |
|:------|:-------:|:-----:|
| Qwen2.5-7B | USA | 0.3538 |
| Qwen2.5-7B | CHN ⚠️ | 0.3526 |
| Qwen2.5-7B | JPN | 0.3392 |
| Qwen2.5-7B | DEU | 0.4262 |
| Qwen2.5-7B | BRA | 0.3546 |
| Gemma-2-9B | USA | 0.4922 |
| Gemma-2-9B | CHN ⚠️ | 0.3592 |
| Gemma-2-9B | JPN | 0.4411 |
| Gemma-2-9B | DEU | 0.3653 |
| Gemma-2-9B | BRA | 0.3438 |
| Mistral-7B | USA | 0.5266 |
| Mistral-7B | CHN ⚠️ | 0.4099 |
| Mistral-7B | JPN | 0.3273 |
| Mistral-7B | DEU | 0.4634 |
| Mistral-7B | BRA | 0.4138 |

#### EXP-09 Notable per-dimension breakdowns (examples copied from logs)

| Model | Country | Flip% | Pearson r | Worst dim | |err| (pp) |
|:------|:-------:|:-----:|----------:|:----------|----------:|
| Mistral | DEU | 17.7% | -0.884 | Species_Humans | 30.7 |
| Mistral | BRA | 16.8% | -0.710 | Age_Young | 30.0 |

> Full per-dimension tables should be exported from `results/hierarchical_is/compare/per_dim_breakdown.csv` on Kaggle runs.

#### EXP-09 Full metrics (from this Kaggle run)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | Cosine ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:--------:|:-----:|:-----:|
| Qwen2.5-7B | USA | 0.3538 | 0.0505 | +0.685 | +0.685 | 12.23 | 17.7% |
| Qwen2.5-7B | CHN ⚠️ | 0.3526 | 0.0681 | +0.349 | +0.349 | 11.35 | 18.1% |
| Qwen2.5-7B | JPN | 0.3392 | 0.0445 | +0.388 | +0.388 | 11.27 | 14.8% |
| Qwen2.5-7B | DEU | 0.4262 | 0.0526 | +0.357 | +0.357 | 14.56 | 16.5% |
| Qwen2.5-7B | BRA | 0.3546 | 0.0662 | +0.058 | +0.058 | 11.69 | 14.2% |
| Gemma-2-9B | USA | 0.4922 | 0.0556 | +0.675 | +0.675 | 18.32 | 11.9% |
| Gemma-2-9B | CHN ⚠️ | 0.3592 | 0.0494 | +0.784 | +0.784 | 12.78 | 12.9% |
| Gemma-2-9B | JPN | 0.4411 | 0.0565 | +0.346 | +0.346 | 15.56 | 13.5% |
| Gemma-2-9B | DEU | 0.3653 | 0.0461 | +0.772 | +0.772 | 13.19 | 15.8% |
| Gemma-2-9B | BRA | 0.3438 | 0.0493 | +0.336 | +0.336 | 11.49 | 17.7% |
| Mistral-7B | USA | 0.5266 | 0.1031 | -0.569 | -0.569 | 19.76 | 14.5% |
| Mistral-7B | CHN ⚠️ | 0.4099 | 0.0837 | -0.612 | -0.612 | 16.25 | 17.7% |
| Mistral-7B | JPN | 0.3273 | 0.0603 | -0.644 | -0.644 | 12.29 | 16.1% |
| Mistral-7B | DEU | 0.4634 | 0.0837 | -0.884 | -0.884 | 16.06 | 17.7% |
| Mistral-7B | BRA | 0.4138 | 0.0677 | -0.710 | -0.710 | 13.34 | 16.8% |

#### EXP-09 Per-dimension worst error (all model × country)

| Model | Country | Worst dim | |err| (pp) |
|:------|:-------:|:----------|----------:|
| Qwen2.5-7B | USA | SocialValue_High | 28.7 |
| Qwen2.5-7B | CHN ⚠️ | SocialValue_High | 26.7 |
| Qwen2.5-7B | JPN | Species_Humans | 24.2 |
| Qwen2.5-7B | DEU | Species_Humans | 28.4 |
| Qwen2.5-7B | BRA | SocialValue_High | 25.0 |
| Gemma-2-9B | USA | Age_Young | 27.3 |
| Gemma-2-9B | CHN ⚠️ | Age_Young | 21.0 |
| Gemma-2-9B | JPN | Utilitarianism_More | 27.2 |
| Gemma-2-9B | DEU | SocialValue_High | 25.4 |
| Gemma-2-9B | BRA | SocialValue_High | 20.5 |
| Mistral-7B | USA | Utilitarianism_More | 30.8 |
| Mistral-7B | CHN ⚠️ | Species_Humans | 24.6 |
| Mistral-7B | JPN | Species_Humans | 21.6 |
| Mistral-7B | DEU | Species_Humans | 30.7 |
| Mistral-7B | BRA | Age_Young | 30.0 |

## Hyperparameter Differences vs EXP-01

| Param | EXP-01 | EXP-02 | EXP-03 | EXP-04 (Mistral) | EXP-05 | EXP-06b | EXP-07 | EXP-09 |
|:------|:------:|:------:|:------:|:----------------:|:------:|:-------:|:------:|:------:|
| N personas | 4 | **8** | **5** | 4 | 4 | 4 | 5 (SV) / 4 (other) | 4 |
| λ_coop | 0.70 | **0.75** | **0.60** | 0.70 | 0.70 | 0.70 | 0.70 | 0.70 |
| σ₀ floor | 0.30 | 0.30 | 0.30 | **0.80** | 0.30 | 0.30 | **0.80** (Mistral) | 0.30 |
| K samples | 128 | **256** | 128 | **512** | 128 | 128 | **512** (Mistral) | 128 |
| T_decision | 0.50 | 0.50 | 0.50 | **0.50** | 0.50 | 0.50 | 0.50 | 0.50 |
| Anchor reg. | ✗ | ✗ | ✗ | ✗ | **✓ ESS-α** | ✗ | **✓ ESS-α** | ✗ |
| Country prior | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ EMA + annealing** |
| N_warmup | — | — | — | — | — | — | — | **50** |
| Decay tau | — | — | — | — | — | — | — | **100** |
| Beta EMA | — | — | — | — | — | — | — | **0.10** |
| Category routing | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** | **✓** | ✗ |
| Urban/rural axis | ✗ | **✓** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Global-citizen agent | ✗ | **✓** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

---

## TODO

- [x] Run EXP-02 on Kaggle H100 (8-agent expanded personas) ✅ 2026-04-09
- [ ] Fix CHN data bug in `data.py` (Afrikaans fallback) — confirmed still present in EXP-02
- [x] Run EXP-03 on Kaggle H100 (SocialValue personas) → **Qwen SocialValue improved; Gemma/Mistral still problematic**
- [x] Run EXP-04 on Kaggle H100 (Mistral cross-lingual) → **mixed** (CHN/DEU/JPN improved; USA/BRA still Pearson<0)
- [x] Run EXP-05 on Kaggle H100 (ESS-adaptive anchor regularization) ✅ 2026-04-09
- [ ] Fix EXP-05 diagnostics export (`alpha_reg`, `anchor_divergence` columns missing → NaN run means)
- [x] Run EXP-06b on Kaggle H100 (category routing ablation) ✅ 2026-04-09
- [ ] Debug EXP-06b routing effect (results currently mirror EXP-01 almost exactly)
- [ ] Run EXP-07 on Kaggle H100 (15 countries × 3 models) → **Final priority**
- [ ] Compute per-dimension MIS from EXP-02 results in analysis script (SocialValue target: err < 10)
- [ ] Update `docs/experiment_tracker.md` with final EXP-07 numbers
- [ ] Update paper §5 results table with EXP-07 as "SWA-PTIS+"
- [ ] Verify EXP-02 JPN Qwen JSD=0.0488 is publishable (best single-country JSD so far)
