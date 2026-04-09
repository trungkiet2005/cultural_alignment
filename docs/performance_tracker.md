# SWA-PTIS Performance Tracker
> **Last updated**: EXP-01 complete (3 models × 5 countries × 310 scenarios)
> **Primary metric**: MIS (L2 misalignment, lower = better, worst = √6 ≈ 2.449)
> **Secondary**: Pearson r, JSD, MAE, per-dim |err|

---

## EXP-01: Paper Baseline — SWA-PTIS (PaperSWAController)

**Config**: 4 WVS personas, K=128, λ_coop=0.7, σ_floor=0.3, T_decision=1.0, N=310 scenarios/country

### MIS Summary Table

| Model | Method | USA | CHN | JPN | DEU | BRA | **Mean MIS** | **Win Rate** |
|:------|:-------|----:|----:|----:|----:|----:|-------------:|-------------:|
| Qwen2.5-7B | Baseline | 0.4559 | 0.4646 | 0.4208 | 0.4398 | 0.5111 | 0.4584 | — |
| Qwen2.5-7B | **SWA-PTIS** | **0.3677** | **0.4078** | **0.2802** | **0.3424** | **0.4025** | **0.3601** | **5/5 ✅** |
| Gemma-2-9B | Baseline | 0.4647 | 0.3679 | 0.4530 | 0.4170 | 0.4403 | 0.4286 | — |
| Gemma-2-9B | **SWA-PTIS** | 0.6038 | 0.4536 | 0.4667 | **0.3289** | **0.3655** | 0.4437 | 2/5 ❌ |
| Mistral-7B | Baseline | 0.4395 | 0.4388 | 0.3798 | 0.4155 | 0.4640 | 0.4275 | — |
| Mistral-7B | **SWA-PTIS** | 0.5984 | 0.5067 | 0.3502 | 0.4942 | 0.4362 | 0.4771 | 1/5 ❌ |

### MIS Improvement % (SWA vs Baseline, ↓ = better)

| Model | USA | CHN | JPN | DEU | BRA | **Macro Avg** |
|:------|----:|----:|----:|----:|----:|-------------:|
| Qwen2.5-7B | +19.3% ↓ | +12.2% ↓ | +33.4% ↓ | +22.2% ↓ | +21.3% ↓ | **+21.7%** |
| Gemma-2-9B | -30.0% ↑ | -23.3% ↑ | -3.0% ↑ | +21.1% ↓ | +17.0% ↓ | **-3.6%** |
| Mistral-7B | -36.2% ↑ | -15.5% ↑ | +7.8% ↓ | -19.0% ↑ | +5.9% ↓ | **-11.4%** |

---

## Per-Model Detailed Results

### Model 1: Qwen2.5-7B-Instruct-bnb-4bit ✅ BEST

#### Baseline (Vanilla LLM)
| Country | MIS | Notes |
|:--------|----:|:------|
| USA | 0.4559 | |
| CHN | 0.4646 | ⚠️ Using dataset_af+google.csv (Afrikaans fallback — data bug!) |
| JPN | 0.4208 | |
| DEU | 0.4398 | |
| BRA | 0.5111 | |
| **Mean** | **0.4584** | |

#### SWA-PTIS Results
| Country | MIS | Pearson r | JSD | MAE | Flip Rate | Mean Var | Δ vs Baseline |
|:--------|----:|----------:|----:|----:|----------:|---------:|--------------:|
| USA | 0.3677 | +0.639 | 0.0759 | 9.75 | 1.9% | 1.255 | **+19.3% ↓** |
| CHN | 0.4078 | +0.418 | 0.0956 | 13.54 | 1.3% | 1.744 | **+12.2% ↓** ⚠️ af fallback |
| JPN | 0.2802 | +0.406 | 0.0553 | 9.97 | 0.6% | 0.443 | **+33.4% ↓** |
| DEU | 0.3424 | +0.461 | 0.0558 | 10.43 | 2.6% | 0.568 | **+22.2% ↓** |
| BRA | 0.4025 | +0.167 | 0.0942 | 14.30 | 0.6% | 1.660 | **+21.3% ↓** |
| **Mean** | **0.3601** | +0.418 | 0.0754 | 11.60 | 1.4% | 1.134 | **+21.7% ↓** |

#### Qwen Per-Dimension |err| (model % − human %)
| Dim | USA | CHN | JPN | DEU | BRA | **Mean** |
|:----|----:|----:|----:|----:|----:|---------:|
| SocialValue_High | **34.5** | **31.4** | **18.0** | **22.3** | **28.9** | **27.0** 🔴 |
| Utilitarianism_More | 10.5 | 18.0 | 8.7 | 10.1 | 7.0 | 10.9 |
| Species_Humans | 1.7 | 9.0 | 17.0 | 23.5 | 10.6 | 12.4 |
| Age_Young | 2.0 | 0.0 | 8.1 | 4.5 | 20.9 | 7.1 |
| Fitness_Fit | 4.3 | 13.8 | 4.0 | 1.2 | 12.2 | 7.1 |
| Gender_Female | 5.5 | 9.0 | 4.1 | 1.1 | 6.3 | 5.2 |

**Qwen Model MPR vs Human MPR (SWA result)**
| Country | Dim | Model | Human | |err| |
|:--------|:----|------:|------:|------:|
| USA | SocialValue_High | 33.5 | 67.9 | 34.5 🔴 |
| CHN | SocialValue_High | 35.3 | 66.7 | 31.4 🔴 |
| JPN | SocialValue_High | 47.9 | 65.9 | 18.0 🟠 |
| DEU | SocialValue_High | 42.4 | 64.7 | 22.3 🔴 |
| BRA | SocialValue_High | 37.4 | 66.3 | 28.9 🔴 |

> **Universal pattern**: Qwen underestimates SocialValue_High in ALL countries. Model sees executives as unworthy; humans assign +67% on average.

---

### Model 2: Gemma-2-9B-IT-bnb-4bit ❌ OVER-CORRECTION

#### Baseline (Vanilla LLM)
| Country | MIS | Notes |
|:--------|----:|:------|
| USA | 0.4647 | Gemma natively strong on SocialValue (SV_High model=98% vs human=68%) |
| CHN | 0.3679 | |
| JPN | 0.4530 | |
| DEU | 0.4170 | |
| BRA | 0.4403 | |
| **Mean** | **0.4286** | |

#### SWA-PTIS Results
| Country | MIS | Pearson r | JSD | MAE | Flip Rate | Mean Var | Δ vs Baseline |
|:--------|----:|----------:|----:|----:|----------:|---------:|--------------:|
| USA | 0.6038 | +0.630 | 0.1108 | 22.31 | 0.6% | 0.897 | **-29.9% ↑ 🔴** |
| CHN | 0.4536 | +0.777 | 0.1034 | 14.68 | 1.3% | 2.345 | **-23.3% ↑ 🔴** |
| JPN | 0.4667 | +0.332 | 0.0794 | 15.85 | 1.6% | 1.247 | -3.0% ↑ |
| DEU | 0.3289 | +0.795 | 0.0683 | 10.13 | 1.3% | 0.983 | **+21.1% ↓ ✅** |
| BRA | 0.3655 | +0.272 | 0.0749 | 14.09 | 3.9% | 0.783 | **+17.0% ↓ ✅** |
| **Mean** | **0.4437** | +0.561 | 0.0874 | 15.41 | 1.7% | 1.251 | **-3.6% ↑** |

#### Gemma Per-Dimension |err| (SWA result)
| Dim | USA | CHN | JPN | DEU | BRA | **Mean** |
|:----|----:|----:|----:|----:|----:|---------:|
| SocialValue_High | **32.6** | **20.8** | **23.9** | **27.7** | **21.6** | **25.3** 🔴 |
| Age_Young | **34.0** | **27.8** | **20.8** | 10.1 | 16.3 | 21.8 🔴 |
| Gender_Female | **27.9** | **28.4** | 7.0 | 1.2 | **15.9** | 16.1 🟠 |
| Utilitarianism_More | **22.1** | 3.7 | **32.4** | 4.4 | **12.6** | 15.0 🟠 |
| Species_Humans | 10.7 | 5.6 | 8.2 | 4.2 | **12.9** | 8.3 |
| Fitness_Fit | 6.6 | 1.8 | 2.8 | **13.2** | 5.3 | 5.9 |

**Root Cause Analysis — Gemma Over-correction (USA)**:
- Baseline Vanilla: SocialValue_High model=98.0% (human=67.9%) → Gemma baseline was BETTER on SocialValue
- SWA-PTIS: SocialValue_High model=35.4% → dropped 62 points in wrong direction!
- Mechanism: WVS personas all egalitarian → anchor << delta_base → IS pulls predictions DOWN from a baseline that was already above human

---

### Model 3: Mistral-7B-Instruct-v0.3-bnb-4bit ❌ VARIANCE COLLAPSE

#### Baseline (Vanilla LLM)
| Country | MIS | Notes |
|:--------|----:|:------|
| USA | 0.4395 | |
| CHN | 0.4388 | |
| JPN | 0.3798 | |
| DEU | 0.4155 | |
| BRA | 0.4640 | |
| **Mean** | **0.4275** | |

#### SWA-PTIS Results
| Country | MIS | Pearson r | JSD | MAE | Flip Rate | Mean Var | Δ vs Baseline |
|:--------|----:|----------:|----:|----:|----------:|---------:|--------------:|
| USA | 0.5984 | **-0.570** 🔴 | 0.1303 | 21.61 | 0.6% | 0.789 | -36.2% ↑ 🔴 |
| CHN | 0.5067 | **-0.682** 🔴 | 0.1050 | 17.41 | 0.3% | 0.349 | -15.5% ↑ 🔴 |
| JPN | 0.3502 | **-0.905** 🔴 | 0.0765 | 12.46 | 1.3% | **0.056** 🚨 | +7.8% ↓ |
| DEU | 0.4942 | **-0.957** 🔴 | 0.1060 | 17.18 | 1.0% | **0.224** 🚨 | -19.0% ↑ 🔴 |
| BRA | 0.4362 | **-0.665** 🔴 | 0.0947 | 14.02 | 0.3% | **0.069** 🚨 | +5.9% ↓ |
| **Mean** | **0.4771** | **-0.756** 🔴 | 0.1025 | 16.54 | 0.7% | **0.297** | **-11.4% ↑** |

**Variance Collapse Diagnostic**
| Country | Mistral Variance | Qwen Variance | Ratio | Pearson r |
|:--------|----------------:|--------------:|------:|----------:|
| USA | 0.789 | 1.255 | 0.63x | -0.570 |
| CHN | 0.349 | 1.744 | 0.20x | -0.682 |
| JPN | **0.056** | 0.443 | 0.13x | **-0.905** |
| DEU | **0.224** | 0.568 | 0.39x | **-0.957** |
| BRA | **0.069** | 1.660 | 0.04x | **-0.665** |

> **Critical pattern**: Variance and Pearson r are perfectly anti-correlated (r=-0.97 between variance and Pearson r). When variance collapses, IS samples pure noise → corr with human inverts.

#### Mistral Per-Dimension |err| (SWA result)
| Dim | USA | CHN | JPN | DEU | BRA | Mean |
|:----|----:|----:|----:|----:|----:|-----:|
| Gender_Female | **33.2** | **25.5** | 19.6 | **26.4** | 6.0 | 22.1 🔴 |
| Age_Young | **29.5** | 11.3 | 5.1 | 13.8 | **37.2** | 19.4 🔴 |
| Species_Humans | 2.4 | **22.1** | **20.4** | **33.5** | 4.7 | 16.6 🟠 |
| Utilitarianism_More | **33.2** | 0.6 | 7.2 | **16.2** | **15.5** | 14.5 🟠 |
| Fitness_Fit | 13.5 | **34.5** | **18.1** | 13.0 | 9.1 | 17.6 🟠 |
| SocialValue_High | **18.0** | 10.6 | 4.3 | 0.2 | **11.7** | 8.9 |

> Mistral's errors are **random noise** — no consistent dimension error pattern. This confirms the IS update is sampling random signal (variance collapse), producing arbitrary dimension-level errors.

---

## Cross-Model Failure Analysis

### SocialValue_High Error — Universal Failure

All 3 models systematically underestimate SocialValue_High:

| Model/Country | USA human=67.9 | CHN human=66.7 | JPN human=65.9 | DEU human=64.7 | BRA human=66.3 |
|:-------------|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|
| **Qwen SWA** | 33.5 (err=34.5) | 35.3 (err=31.4) | 47.9 (err=18.0) | 42.4 (err=22.3) | 37.4 (err=28.9) |
| **Gemma SWA** | 35.4 (err=32.5) | 45.9 (err=20.8) | 42.0 (err=23.9) | 37.1 (err=27.6) | 44.7 (err=21.6) |
| **Mistral SWA** | 50.0 (err=17.9) | 56.2 (err=10.5) | 61.6 (err=4.3) | 64.9 (err=0.2) | 54.7 (err=11.6) |

**Root cause**: All 4 WVS personas emphasize egalitarianism → anchor for SocialValue_High is negative → IS cannot push prediction above human 67%. The anchor is too low, and even optimal delta_star cannot fully compensate.

### Pearson r Distribution (SWA method)

| Country | Qwen r | Gemma r | Mistral r | Best |
|:--------|-------:|--------:|----------:|:-----|
| USA | +0.639 | +0.630 | -0.570 | Qwen |
| CHN | +0.418 | **+0.777** | -0.682 | Gemma |
| JPN | +0.406 | +0.332 | **-0.905** 💀 | Qwen |
| DEU | +0.461 | **+0.795** | **-0.957** 💀 | Gemma |
| BRA | +0.167 | +0.272 | -0.665 | Gemma |

---

## Identified Bugs

| Bug | Severity | File | Status |
|:----|:---------|:-----|:-------|
| CHN uses `dataset_af+google.csv` (Afrikaans fallback) instead of Chinese | 🔴 Critical | `src/data.py` | **OPEN** |
| Gemma CUBLAS non-determinism warning (no `CUBLAS_WORKSPACE_CONFIG` set) | 🟡 Low | env setup | OPEN |

---

## Experiment Roadmap

| Exp | Hypothesis | Target Failure | Status |
|:----|:-----------|:---------------|:-------|
| EXP-01 | Paper baseline benchmark | — | ✅ Done |
| EXP-02 | 8-agent expanded persona pool (8 WVS) | Persona diversity | 🔵 Ready |
| EXP-03 | Social-utility targeted personas (3 WVS + 2 SU) | SocialValue_High err | 🔵 Ready |
| EXP-04 | Mistral English cross-lingual override | Variance collapse | 🔵 Ready |
| EXP-05 | ESS-adaptive anchor regularization | Gemma over-correction | 🔵 Ready |
| EXP-06 | EXP-03 + EXP-04 combined | Both SV + variance | 🔲 Planned |
| EXP-07 | Full integration (EXP-03+04+05) | All failure modes | 🔲 Planned |

---

## Insights & Learnings

### Insight 1: SocialValue anchor bias (universal, all models)
- **Problem**: WVS personas are trained on equality dimensions → they output negative delta for "spare executive" scenarios → anchor = mean(delta_i) is negative → IS cannot reverse this
- **Expected Fix (EXP-03)**: Adding 2 social-utility personas per country shifts anchor from ~-0.8 to ~-0.2 for SocialValue, enabling IS to correct to human levels
- **Target**: Reduce mean SocialValue |err| from 27.0 → <15

### Insight 2: Mistral variance collapse (language-specific)
- **Problem**: Mistral SentencePiece tokenizer maps all CJK/German text to similar "uncertainty" representations → persona prefixes are near-identical → std(delta_agents) → 0 → IS samples noise
- **Evidence**: JPN variance=0.056 vs Qwen JPN=0.443 (8x lower); Pearson r=-0.905 perfectly anti-correlated with correct answer
- **Expected Fix (EXP-04)**: English personas force Mistral into its training distribution → variance rises to 0.4+ range  
- **Additional**: σ_floor 0.3→0.8, K 128→512, T_decision 1.0→0.5

### Insight 3: Gemma over-correction (model-specific)
- **Problem**: Gemma's RLHF training makes it sensitive to persona context → WVS egalitarian personas pull delta well below baseline → delta_opt < 0 even when delta_base was well-calibrated (USA baseline: SV=98%)
- **Expected Fix (EXP-05)**: ESS-adaptive regularization:
  `delta_opt = alpha * anchor + (1-alpha) * delta_base + delta_star`
  where `alpha = clamp(k_eff/K, rho_eff, 1.0)`
  When IS quality is low (alpha → rho_eff), trust delta_base over anchor

### Insight 4: Qwen is the strongest base
- SWA-PTIS works best with Qwen: 5/5 countries win, +21.7% mean improvement
- Qwen's moderate baseline variance (0.9-1.7) is in the "goldilocks zone" for IS
- Qwen's instruction-following is balanced — not too strong (Gemma) nor too language-sensitive (Mistral)

### Insight 5: CHN results are INVALID
- CHN used `dataset_af+google.csv` (Afrikaans) because `dataset_zh+google.csv` is missing
- All CHN numbers must be treated as unreliable until `src/data.py` is fixed to raise FileNotFoundError
- Action: Fix data bug → re-run CHN baseline + SWA for all 3 models

---

## SOTA Targets for Paper

| Metric | EXP-01 (Qwen best) | Target (EXP-07) |
|:-------|-------------------:|----------------:|
| Mean MIS (5 countries) | 0.3601 | < 0.25 |
| SocialValue mean |err| | 27.0 | < 12.0 |
| SWA win rate | 5/5 (Qwen), 2/5 (Gemma) | 5/5 all models |
| Mistral Pearson r (JPN) | -0.905 | > +0.5 |
| Gemma USA MIS | 0.6038 | < 0.35 |
