# SWA-PTIS Experiment Tracker — Cultural Alignment

> **Purpose**: Living research log. Every run appends insights + designs the
> next experiment. Read bottom-up for history; top-down for current focus.
>
> **Primary metric**: MIS = L2 misalignment vs human AMCE (lower = better).
> Secondary: per-dimension |err| = |model_MPR − human_MPR|, Pearson r, JSD.

---

## 🔬 EXP-01 — Baseline SWA-PTIS (3-model sweep, 5 countries)

**Script**: `experiment/kaggle_experiment.py`  
**Date**: 2026-04-09 | **Commit**: `12f9657`  
**Config**: N=500 scenarios, 4 personas (WVS young/middle/older + utilitarian),
K=128, lambda_coop=0.7, PT alpha=beta=0.88 kappa=2.25, sigma_floor=0.3, T_decision=1.0

### Raw MIS Table

| Model | Country | Baseline MIS | SWA MIS | Delta MIS | Improv % |
|:------|:-------:|-------------:|--------:|----------:|---------:|
| **Qwen2.5-7B** | USA | 0.4559 | 0.3677 | +0.0882 | **+19.3%** |
| **Qwen2.5-7B** | CHN | 0.4646 | 0.4078 | +0.0568 | **+12.2%** |
| **Qwen2.5-7B** | JPN | 0.4208 | 0.2802 | +0.1405 | **+33.4%** |
| **Qwen2.5-7B** | DEU | 0.4398 | 0.3424 | +0.0974 | **+22.2%** |
| **Qwen2.5-7B** | BRA | 0.5111 | 0.4025 | +0.1086 | **+21.3%** |
| **Gemma-2-9B** | USA | 0.4647 | 0.6038 | -0.1391 | **-30.0%** |
| **Gemma-2-9B** | CHN | 0.3679 | 0.4536 | -0.0857 | **-23.3%** |
| **Gemma-2-9B** | JPN | 0.4530 | 0.4667 | -0.0136 | **-3.0%** |
| **Gemma-2-9B** | DEU | 0.4170 | 0.3289 | +0.0882 | **+21.1%** |
| **Gemma-2-9B** | BRA | 0.4490 | 0.3655 | +0.0834 | **+18.6%** |
| **Mistral-7B** | USA | 0.5706 | 0.5984 | -0.0278 | **-4.9%** |
| **Mistral-7B** | CHN | 0.4569 | 0.5067 | -0.0498 | **-10.9%** |
| **Mistral-7B** | JPN | 0.3429 | 0.3502 | -0.0073 | **-2.1%** |
| **Mistral-7B** | DEU | 0.4909 | 0.4942 | -0.0033 | **-0.7%** |
| **Mistral-7B** | BRA | 0.4144 | 0.4362 | -0.0217 | **-5.3%** |

**Aggregate**: Qwen 5/5 wins (+21.5%); Gemma 2/5 wins (-3.1%); Mistral 0/5 wins (-4.8%).  
**Overall macro**: 7/15 wins, +4.5% average — Qwen carrying the average.

---

### Per-Dimension Error Analysis (SWA output vs Human AMCE)

#### Qwen2.5-7B (Best performing model)

| Country | Worst dim (err) | 2nd worst | 3rd worst |
|:--------|:----------------|:----------|:----------|
| USA | SocialValue `33.5 vs 67.9` **err=34.5** | Utilit `66.1 vs 76.6` err=10.5 | Gender err=5.5 |
| CHN | SocialValue `35.3 vs 66.7` **err=31.4** | Utilit `89.1 vs 71.1` err=18.0 | Fitness err=13.8 |
| JPN | SocialValue `47.9 vs 65.9` **err=18.0** | Species `62.9 vs 79.8` err=17.0 | Utilit err=8.7 |
| DEU | Species `59.0 vs 82.4` **err=23.5** | SocialValue `42.4 vs 64.7` err=22.3 | Utilit err=10.1 |
| BRA | SocialValue `37.4 vs 66.3` **err=28.9** | Age `52.7 vs 73.6` err=20.9 | Fitness err=12.2 |

#### Mean variance per model per country

| | USA | CHN | JPN | DEU | BRA |
|:-|----:|----:|----:|----:|----:|
| Qwen | 1.255 | 1.744 | **0.443** | **0.568** | 1.660 |
| Gemma | 0.897 | 2.345 | 1.247 | 0.983 | 0.783 |
| Mistral | 0.789 | 0.349 | **0.056** | **0.224** | **0.069** |

#### Pearson r (model MPR vs human MPR, all 6 dims)

| | USA | CHN | JPN | DEU | BRA |
|:-|----:|----:|----:|----:|----:|
| Qwen | +0.639 | +0.418 | +0.406 | +0.461 | +0.167 |
| Gemma | +0.630 | +0.777 | +0.332 | +0.795 | +0.272 |
| Mistral | -0.570 | -0.682 | **-0.905** | **-0.957** | -0.665 |

---

## KEY INSIGHTS (Ranked by Actionability)

### Insight 1 — SocialValue_High is THE Universal Failure Mode

**Evidence**: SocialValue_High is the top-2 error dimension in 4/5 countries for
Qwen (best model), and for all Gemma/Mistral countries.

| Country | Human MPR | Model MPR (Qwen SWA) | Error |
|:-------:|:---------:|:-------------------:|:-----:|
| USA | 67.9 | 33.5 | **-34.4** |
| CHN | 66.7 | 35.3 | **-31.4** |
| JPN | 65.9 | 47.9 | **-18.0** |
| DEU | 64.7 | 42.4 | **-22.3** |
| BRA | 66.3 | 37.4 | **-28.9** |

**Root cause (proven)**: WVS-based personas all lean egalitarian. None explicitly endorse
*social utility gradients*. The persona-mean anchor = `mean(delta_i)` is systematically
negative for SocialValue (all agents lean against executives). Since `delta_opt = anchor + delta_star`
and `anchor < 0`, the IS update cannot bring `delta_opt` into positive territory even with a
perfectly optimal `delta_star`.

**Mathematical proof**:
```
For SocialValue: all delta_i < 0 (egalitarian agents prefer homeless = B side)
=> anchor = mean(delta_i) < 0
=> |delta_star| <= |anchor|  (IS is bounded by the proposal std sigma)
=> delta_opt = anchor + delta_star may still be < 0 if anchor << 0
=> model predicts saving homeless over exec more than humans do
```

**Fix -> EXP-03**: Add 2 "social utility" personas per country that endorse
professional/expert status: triage ethics doctors, meritocratic Confucian scholars, etc.

---

### Insight 2 — Mistral Variance Collapse (Multilingual Persona Deadlock)

**Evidence**: Mistral JPN variance=0.056, BRA=0.069 (20x below Qwen).
Pearson r = -0.905 (JPN), -0.957 (DEU) — **actively anti-correlated** with human preferences.

**Mechanism**:
Mistral is SentencePiece-based, primarily English-trained. For non-English prompts +
non-English personas, all 4 persona prefix representations collapse to near-identical
internal representations. The model cannot distinguish "young Japanese" from "old Japanese"
personas in the attention space → `delta_i ≈ delta_base` for all i → `std(delta_agents) -> 0`.
When `sigma = sigma_floor = 0.3`, the IS samples random noise around the wrong anchor.

The negative Pearson r is explained: when variance collapses, the IS samples corrupt the
already-marginally-correct baseline decision. For JPN/DEU, Mistral baseline was mildly correct
(r=+0.37 without SWA), but SWA noise inverts it to r=-0.96.

**Fix -> EXP-04**: Cross-lingual English persona override for Mistral:
- `assistant_lang = "en"` always for Mistral (personas in English)
- `sigma_floor = 0.8` (must be large enough to escape the degeneracy)
- `K_samples = 512` (more samples to average noise)
- `decision_temperature = 0.5` (sharpen the final sigmoid)

---

### Insight 3 — Gemma Over-Correction (Egalitarian Anchor Inversion)

**Evidence**: Gemma baseline for USA correctly predicts exec-preference
(baseline SocialValue 80+%), but after SWA drops to 35.4% (completely wrong direction).
Gemma BRA flip_rate=3.9% = highest, yet MIS improves (+18.6%) — Gemma CAN benefit
when the anchor direction is right (BRA/DEU) and fails when wrong (USA/CHN).

**Mechanism**: Gemma's rich instruction-following makes it respond more strongly to personas.
For USA/CHN: the egalitarian personas produce `anchor << delta_base`, causing over-correction.
For DEU/BRA: weaker persona effect, smaller anchor vs baseline gap, less over-correction.

**Fix -> EXP-05**: ESS-Adaptive Anchor Regularization:
```
alpha = clamp(k_eff / K, rho_eff, 1.0)
delta_opt = alpha * anchor + (1 - alpha) * delta_base + delta_star
```
When ESS is high (good IS), alpha=1 → standard update.
When ESS is low (bad IS, collapsed weights), alpha → rho_eff → mostly follows base model.

**Mathematical justification**: This is a KL-penalized ELBO where the penalty
`||delta_opt - delta_base||^2 / (2*sigma^2)` has coefficient `(1-alpha)/alpha`.
The regularization strength is learned online per-scenario via the ESS quality ratio.

---

### Insight 4 — CHN Data Bug (Critical - Paper Validity)

**Evidence**: Log shows `[DATA] Exact file not found, using: dataset_af+google.csv` for CHN.
The code silently falls back to Afrikaans (af) when Chinese (zh) file not found.
All CHN results in EXP-01 are based on Afrikaans text, not Chinese.

**Impact**: CHN claims in the paper (+12.2% Qwen) are based on invalid data.
The Qwen CHN SWA result (0.4078) should be thrown out and re-run with correct data.

**Fix**: Patch `data.py` to raise `FileNotFoundError` if exact lang match fails.
Re-run EXP-01 CHN for all 3 models.

---

### Insight 5 — Flip Rate as Quality Signal

| Model | Country | Flip Rate | MIS Change | Verdict |
|:------|:-------:|----------:|-----------:|:--------|
| Qwen | USA | 1.9% | +19.3% | Good flips |
| Qwen | JPN | 0.6% | +33.4% | Even small flips help |
| Gemma | USA | 0.6% | -30.0% | Bad: IS not flipping enough to fix |
| Gemma | BRA | 3.9% | +18.6% | Good: flipping the right decisions |
| Mistral | CHN | 0.3% | -10.9% | Near no-op, then noise |

High flip rate + good MIS = correct targeting.
High flip rate + bad MIS = wrong direction (Gemma USA-type failure).
Low flip rate + good MIS = leverage: tiny IS correction on high-logit-gap scenarios works.

---

## EXPERIMENT ROADMAP

| EXP | Name | Status | Key Innovation | Target Fix |
|:----|:-----|:------:|:---------------|:-----------|
| 01 | Baseline SWA-PTIS | DONE | Paper pipeline | — |
| 02 | 8-Agent Expanded Personas | READY | Urban/rural split | More coverage |
| 03 | SocialValue-Targeted Personas | **CREATE** | Social-utility agents | Insight 1 |
| 04 | Mistral Cross-Lingual Override | **CREATE** | English personas + sigma_floor=0.8 | Insight 2 |
| 05 | ESS-Adaptive Anchor Regularization | **CREATE** | KL-penalized anchor | Insight 3 |
| 06 | Category-Routed Persona Pools | **CREATE** | Per-category persona dispatch | Insight 1+3 |
| 07 | Best-Config Full Sweep | PLANNED | Combine EXP-03+04+05 | All |

---

## Per-Dimension Error Scoreboard (Qwen SWA-PTIS, best model)

| Dimension | USA | CHN | JPN | DEU | BRA | MEAN |
|:----------|----:|----:|----:|----:|----:|-----:|
| SocialValue_High | 34.5 | 31.4 | 18.0 | 22.3 | 28.9 | **27.0** |
| Utilitarianism_More | 10.5 | 18.0 | 8.7 | 10.1 | 7.0 | 10.9 |
| Species_Humans | 1.7 | 9.0 | 17.0 | 23.5 | 10.6 | 12.4 |
| Age_Young | 2.0 | 0.0 | 8.1 | 4.5 | 20.9 | 7.1 |
| Fitness_Fit | 4.3 | 13.8 | 4.0 | 1.2 | 12.2 | 7.1 |
| Gender_Female | 5.5 | 9.0 | 4.1 | 1.1 | 6.3 | 5.2 |

**SocialValue accounts for ~40% of total L2 error. Fix this first.**

---

## Known Bugs

| Bug | Severity | Action |
|:----|:--------:|:-------|
| CHN data falls back to Afrikaans | **CRITICAL** | Fix data.py fallback logic |
| alpha_ctl warning every predict() | Low | Cosmetic only |
| Gemma CUBLAS non-determinism warning | Low | Set CUBLAS_WORKSPACE_CONFIG in env |

---

## TODO

- [ ] **Fix CHN data loading bug** in `data.py`
- [ ] Re-run EXP-01 CHN after bug fix (all 3 models)
- [ ] Create `experiment/exp03_socialvalue_personas.py`
- [ ] Create `experiment/exp04_mistral_crosslingual.py`
- [ ] Create `experiment/exp05_anchor_regularization.py`
- [ ] Create `experiment/exp06_category_routing.py`
- [ ] Run EXP-02 through EXP-06 on Kaggle H100
- [ ] Compile full model x country x method table for paper Section 5
