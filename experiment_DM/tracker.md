# SWA-PTIS — experiment_DM Tracker

> **Folder purpose**: Custom experiments built on top of the paper baseline (`experiment/kaggle_experiment.py`).  
> Each script is **self-contained** and Kaggle-ready (auto-bootstrap + pip-install).  
> Read the `Motivation` docstring at the top of each file for full design rationale.
>
> **Run order on Kaggle H100**: EXP-07 → EXP-06 (routing) → EXP-03 → EXP-05 → EXP-04  
> **Primary metric**: MIS = L2 misalignment vs human AMCE ↓. Secondary: JSD ↓, Pearson r ↑.

---

## File Index

| File | EXP ID | Status | Key Innovation | Fixes |
|:-----|:------:|:------:|:--------------|:------|
| `exp02_expanded_personas.py` | EXP-02 | 🟡 READY | 8 personas (urban/rural split) | More coverage |
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
- Mean error = **27pp** (model ~39% vs human 66%)
- Root cause: WVS personas are structurally egalitarian → anchor < 0 for SocialValue
- Fix: social-utility gradient personas (medical-triage + social-capital economists)

### Insight 2 — Mistral Variance Collapse
- **Target file**: `exp04_mistral_crosslingual.py`
- JPN variance = 0.056, Pearson r = **-0.905** (anti-correlated!)
- Root cause: SentencePiece tokeniser collapses non-Latin persona prefixes
- Fix: English-only personas + σ₀=0.8 + K=512

### Insight 3 — Gemma Over-Correction
- **Target file**: `exp05_anchor_regularization.py`
- USA MIS −30%, CHN −23% (SWA makes things worse!)
- Root cause: egalitarian anchor << delta_base → over-correction
- Fix: `alpha = clamp(k_eff/K, ρ_eff, 1)` → `delta_opt = α·anchor + (1-α)·base + delta_star`
- **Formal theorem**: Regularised update ≤ standard update MSE whenever base closer to human AND ESS < 50%

---

## Results Log

### EXP-01 Baseline (reference, in `experiment/kaggle_experiment.py`)

| Model | Win Rate | Mean MIS Δ |
|:------|:--------:|-----------:|
| Qwen2.5-7B | 5/5 | **+21.5%** |
| Gemma-2-9B | 2/5 | -3.1% |
| Mistral-7B | 0/5 | -4.8% |

> ⚠️ CHN results use Afrikaans fallback data — bug in `data.py`. Re-run after fix.

### EXP-03 through EXP-09 — results pending Kaggle run

| EXP | Model | Country | Baseline MIS | EXP MIS | Δ | Status |
|:----|:-----:|:-------:|:------------:|:-------:|:-:|:------:|
| 03 | Qwen | USA | 0.3677 | — | — | ⏳ |
| 03 | Gemma | USA | 0.6038 | — | — | ⏳ |
| 04 | Mistral | JPN | 0.3502 | — | — | ⏳ |
| 05 | Gemma | USA | 0.6038 | — | — | ⏳ |
| 07 | All | All 15 | — | — | — | ⏳ |

---

## Hyperparameter Differences vs EXP-01

| Param | EXP-01 | EXP-03 | EXP-04 (Mistral) | EXP-05 | EXP-06b | EXP-07 |
|:------|:------:|:------:|:----------------:|:------:|:-------:|:------:|
| N personas | 4 | **5** | 4 | 4 | 4 | 5 (SV) / 4 (other) |
| λ_coop | 0.70 | **0.60** | 0.70 | 0.70 | 0.70 | 0.70 |
| σ₀ floor | 0.30 | 0.30 | **0.80** | 0.30 | 0.30 | **0.80** (Mistral) |
| K samples | 128 | 128 | **512** | 128 | 128 | **512** (Mistral) |
| T_decision | 0.50 | 0.50 | **0.50** | 0.50 | 0.50 | 0.50 |
| Anchor reg. | ✗ | ✗ | ✗ | **✓ ESS-α** | ✗ | **✓ ESS-α** |
| Category routing | ✗ | ✗ | ✗ | ✗ | **✓** | **✓** |

---

## TODO

- [ ] Fix CHN data bug in `data.py` (Afrikaans fallback)
- [ ] Run EXP-07 on Kaggle H100 (15 countries × 3 models) → **Priority**
- [ ] Run EXP-06b on Kaggle H100 (category routing ablation)
- [ ] Run EXP-03 on Kaggle H100 (SocialValue isolated)
- [ ] Run EXP-05 on Kaggle H100 (Gemma anchor ablation)
- [ ] Run EXP-04 on Kaggle H100 (Mistral cross-lingual)
- [ ] Fill results log table above with actual numbers
- [ ] Compute per-dimension MIS from results (SocialValue target: err < 10)
- [ ] Update `docs/experiment_tracker.md` with final EXP-07 numbers
- [ ] Update paper §5 results table with EXP-07 as "SWA-PTIS+"
