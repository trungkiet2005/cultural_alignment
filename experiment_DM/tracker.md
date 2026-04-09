# SWA-PTIS — experiment_DM Tracker

> **Folder purpose**: Custom experiments built on top of the paper baseline (`experiment/kaggle_experiment.py`).  
> Each script is **self-contained** and Kaggle-ready (auto-bootstrap + pip-install).  
> Read the `Motivation` docstring at the top of each file for full design rationale.
>
> **Run order on Kaggle H100**: EXP-07 → EXP-06 (routing) → EXP-03 → EXP-05 → EXP-04  
> **Primary metric**: MIS = L2 misalignment vs human AMCE ↓. Secondary: JSD ↓, Pearson r ↑.  
> **Completed runs**: EXP-01 ✅ (2026-04-09) | EXP-02 ✅ (2026-04-09)

---

## File Index

| File | EXP ID | Status | Key Innovation | Fixes |
|:-----|:------:|:------:|:--------------|:------|
| `exp02_expanded_personas.py` | EXP-02 | ✅ DONE | 8 personas (urban/rural split) | More coverage |
| `exp03_socialvalue_personas.py` | EXP-03 | 🟡 READY | Social-utility gradient personas (P4/P5) | SocialValue underestimation |
| `exp04_mistral_crosslingual.py` | EXP-04 | 🟡 READY | English persona override + σ₀=0.8 + K=512 | Mistral variance collapse |
| `exp05_anchor_regularization.py` | EXP-05 | 🟡 READY | ESS-adaptive anchor α·anchor + (1-α)·base | Gemma over-correction |
| `exp06_adaptive_sigma.py` | EXP-06a | 🟡 READY | Per-scenario adaptive σ (entropy-based) | Qwen32B logit collapse |
| `exp06_category_routing.py` | EXP-06b | 🟡 READY | Per-category expert persona pools (6 panels) | Dim-level anchor bias |
| `exp07_best_config_sweep.py` | EXP-07 | 🟡 READY | **Combined EXP-03+04+05+06, 15 countries** | ALL |
| `exp07_wvs_augmentation.py` | EXP-07a | 🟡 READY | WVS Wave 6+7 temporal augmentation | Persona staleness |
| `exp08_category_routing.py` | EXP-08 | 🟡 READY | Extended category routing (8 panels) | Precision routing |
| `exp09_hierarchical_is.py` | EXP-09 | 🟡 READY | Two-level IS (persona-level + sample-level) | IS variance |

---

## Failure Modes Being Addressed

### Insight 1 — SocialValue Underestimation (ALL models)
- **Target file**: `exp03_socialvalue_personas.py`, `exp06_category_routing.py`
- EXP-02 confirmed: mean SocialValue error = **27–30pp** across all models/countries
  - Qwen: SV err ≈ 20–30pp | Gemma: SV err ≈ 17–29pp | Mistral: SV err ≈ 1–17pp
- EXP-02 urban/rural personas did NOT fix SocialValue → 8 agents still under-assign social value
- Root cause confirmed: WVS personas are structurally egalitarian → anchor < 0 for SocialValue
- Fix needed: social-utility gradient personas (EXP-03)

### Insight 2 — Mistral Variance Collapse (CONFIRMED & WORSENED in EXP-02)
- **Target file**: `exp04_mistral_crosslingual.py`
- EXP-02 results: JPN variance = **0.065** (collapsed!), Pearson r = **-0.911** (anti-correlated!)
- DEU: Pearson r = **-0.962**, Spearman ρ = **-0.943** → severe ranking inversion
- BRA: Pearson r = **-0.696**, Age_Young err = **37.8pp**
- Urban/rural modulation had no effect on SentencePiece collapse
- Fix needed: English-only personas + σ₀=0.8 + K=512 (EXP-04)

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

### EXP-03 through EXP-09 — results pending Kaggle run

| EXP | Model | Country | Baseline MIS | EXP MIS | Δ | Status |
|:----|:-----:|:-------:|:------------:|:-------:|:-:|:------:|
| 03 | Qwen | USA | 0.3496 | — | — | ⏳ |
| 03 | Gemma | USA | 0.6073 | — | — | ⏳ |
| 04 | Mistral | JPN | 0.3508 | — | — | ⏳ |
| 05 | Gemma | USA | 0.6073 | — | — | ⏳ |
| 07 | All | All 15 | — | — | — | ⏳ |

---

## Hyperparameter Differences vs EXP-01

| Param | EXP-01 | EXP-02 | EXP-03 | EXP-04 (Mistral) | EXP-05 | EXP-06b | EXP-07 |
|:------|:------:|:------:|:------:|:----------------:|:------:|:-------:|:------:|
| N personas | 4 | **8** | **5** | 4 | 4 | 4 | 5 (SV) / 4 (other) |
| λ_coop | 0.70 | **0.75** | **0.60** | 0.70 | 0.70 | 0.70 | 0.70 |
| σ₀ floor | 0.30 | 0.30 | 0.30 | **0.80** | 0.30 | 0.30 | **0.80** (Mistral) |
| K samples | 128 | **256** | 128 | **512** | 128 | 128 | **512** (Mistral) |
| T_decision | 0.50 | 0.50 | 0.50 | **0.50** | 0.50 | 0.50 | 0.50 |
| Anchor reg. | ✗ | ✗ | ✗ | ✗ | **✓ ESS-α** | ✗ | **✓ ESS-α** |
| Category routing | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** | **✓** |
| Urban/rural axis | ✗ | **✓** | ✗ | ✗ | ✗ | ✗ | ✗ |
| Global-citizen agent | ✗ | **✓** | ✗ | ✗ | ✗ | ✗ | ✗ |

---

## TODO

- [x] Run EXP-02 on Kaggle H100 (8-agent expanded personas) ✅ 2026-04-09
- [ ] Fix CHN data bug in `data.py` (Afrikaans fallback) — confirmed still present in EXP-02
- [ ] Run EXP-03 on Kaggle H100 (SocialValue personas) → address 27pp SV error
- [ ] Run EXP-04 on Kaggle H100 (Mistral cross-lingual) → urgently needed (r = -0.962 DEU!)
- [ ] Run EXP-05 on Kaggle H100 (Gemma anchor regularization)
- [ ] Run EXP-06b on Kaggle H100 (category routing ablation)
- [ ] Run EXP-07 on Kaggle H100 (15 countries × 3 models) → **Final priority**
- [ ] Compute per-dimension MIS from EXP-02 results in analysis script (SocialValue target: err < 10)
- [ ] Update `docs/experiment_tracker.md` with final EXP-07 numbers
- [ ] Update paper §5 results table with EXP-07 as "SWA-PTIS+"
- [ ] Verify EXP-02 JPN Qwen JSD=0.0488 is publishable (best single-country JSD so far)
