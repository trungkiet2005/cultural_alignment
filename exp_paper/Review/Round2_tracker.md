# Round 2 Review Tracker — SWA-DPBR (NeurIPS 2026)

Camera-ready action log for the **lean accept with revisions** Round 2 decision.
Each row maps one review item to: (i) code we wrote, (ii) Kaggle experiment
script to run, (iii) paper section to update, and (iv) status.

**Status legend:** `TODO` → not started, `CODE` → script written, runs locally,
`RUNNING` → script submitted to Kaggle, `DONE` → results in tracker.md and paper
tex updated.

---

## W1 — Baseline coverage (dropout calibration, T/margin scaling, DIFFPO-adapted)

| Item | File | Script | Paper section |
|---|---|---|---|
| W1a. Inference-time dropout calibration (MC-Dropout uncertainty inflation) | [src/mc_dropout_runner.py](../../src/mc_dropout_runner.py) | [exp_paper/exp_r2_baseline_dropout.py](../exp_r2_baseline_dropout.py) | §Baselines + §Results |
| W1b. Per-country temperature / margin scaling (calibration-only) | [src/calibration_baselines.py](../../src/calibration_baselines.py) | [exp_paper/exp_r2_baseline_tempmargin.py](../exp_r2_baseline_tempmargin.py) | §Baselines + §Results |
| W1c. DIFFPO-adapted binary-decision refinement (black-box persona-rewrite + prob. mixing) | [src/diffpo_binary_baseline.py](../../src/diffpo_binary_baseline.py) | [exp_paper/exp_r2_baseline_diffpo.py](../exp_r2_baseline_diffpo.py) | §Baselines + §Results |

**Status: CODE** (syntax checked; Kaggle run pending) — all three baselines implemented, Kaggle scripts ready.

---

## W2 — Ablation breadth (beyond Phi-4 / USA)

| Item | File | Script | Paper section |
|---|---|---|---|
| W2a. Ablation on 3 extra models × 3 countries (Qwen2.5-7B, Llama-3.3-70B, Phi-3.5-mini × USA, JPN, VNM) | — (reuses [exp_paper/exp_paper_ablation_phi4.py](../exp_paper_ablation_phi4.py) pattern) | [exp_paper/exp_r2_ablation_breadth.py](../exp_r2_ablation_breadth.py) | §Ablation + App. Detailed Ablation |

**Status: CODE** (syntax checked; Kaggle run pending)

---

## W3 — Hyperparameter sensitivity (s, λ_coop, σ, T_cat)

| Item | File | Script | Paper section |
|---|---|---|---|
| W3. 4-axis sweep reporting JSD/MIS/r vs hyperparam value | [exp_paper/exp_r2_hparam_sensitivity.py](../exp_r2_hparam_sensitivity.py) | same | App. Hyperparameter Validation |

**Status: CODE** (syntax checked; Kaggle run pending) — uses env-var overrides (`EXP24_VAR_SCALE`, `EXP24_LAMBDA_COOP`) + new `EXP24_SIGMA_FLOOR`, `EXP24_TCAT_SCALE`.

---

## W4 — PT-IS reliability (s selection, dual-pass disagreement regime)

| Item | File | Script | Paper section |
|---|---|---|---|
| W4a. Sensitivity of r = exp(−(δ₁*−δ₂*)²/s) to s ∈ {0.01, 0.02, 0.04, 0.08, 0.16} | [exp_paper/exp_r2_hparam_sensitivity.py](../exp_r2_hparam_sensitivity.py) (VAR_SCALE axis) | same | App. PT--IS §Why split K into two passes |
| W4b. Quantify "high-ESS + high-disagreement" regime counts | [src/dpbr_reliability_audit.py](../../src/dpbr_reliability_audit.py) | [exp_paper/exp_r2_reliability_audit.py](../exp_r2_reliability_audit.py) | App. PT--IS |

**Status: CODE** (syntax checked; Kaggle run pending)

---

## W5 — 20-country selection & preprocessing (full-pool + oversampling)

| Item | File | Script | Paper section |
|---|---|---|---|
| W5a. Publish exact scenario ids + seeds per country | [scripts/dump_scenario_seeds.py](../scripts/dump_scenario_seeds.py) | — | App. Dataset Preprocessing |
| W5b. No-oversampling variant (drop countries under minimum; no per-country capping) | [src/data.py](../../src/data.py) (new `oversample=False` flag) | [exp_paper/exp_r2_no_oversampling.py](../exp_r2_no_oversampling.py) | App. Dataset Preprocessing |

**Status: CODE** (syntax checked; Kaggle run pending)

---

## W6 — Clarity / persona definition inconsistency

| Item | File | Script | Paper section |
|---|---|---|---|
| W6a. Resolve "3 cohorts + aggregate" vs "3 cohorts + utilitarian" | [src/personas.py](../../src/personas.py) (expose `build_country_personas(..., fourth="aggregate"\|"utilitarian")`) | [exp_paper/exp_r2_persona_variant.py](../exp_r2_persona_variant.py) | §Method + fig_pipeline.tex |
| W6b. Run head-to-head on 20-country slice with both variants | — | same | §Method |

**Status: CODE** (syntax checked; Kaggle run pending) — code defaults stay unchanged (aggregate); experiment produces a head-to-head table. Paper caption is updated to match code reality (`aggregate`).

---

## W7 — Pearson r interpretation (why negative while MIS improves)

| Item | File | Script | Paper section |
|---|---|---|---|
| W7. Add rank-based per-dimension agreement + r-vs-MIS scatter | [src/amce.py](../../src/amce.py) (new `per_dim_rank_agreement()`) | [exp_paper/exp_r2_rank_agreement.py](../exp_r2_rank_agreement.py) | App. Per-Dimension Error Analysis |

**Status: CODE** (syntax checked; Kaggle run pending) — computed post-hoc from existing per-country CSVs; no new model runs needed.

---

## W8 — Related-work coverage + PT reliability caveat

| Item | What | Paper section |
|---|---|---|
| W8a. Cite & contrast PITA, DIFFPO, AISP, KTO | paragraph additions | §Related Work |
| W8b. Discuss PT fragility under linguistic uncertainty (2508.08992) | paragraph + citation | §Discussion (PT caveat) |
| W8c. Per-persona floor safeguard for minority views | short paragraph + pointer to controller | §Discussion (ethics) |

**Status: CODE** (syntax checked; Kaggle run pending) — bib + section additions drafted below; paper tex updates pending bib population.

---

## W9 — WVS-to-trolley causal linkage

| Item | File | Script | Paper section |
|---|---|---|---|
| W9. Drop one WVS dim at a time from persona → track dim-wise AMCE error | [src/personas.py](../../src/personas.py) (WVS_DIMS drop flag via env) | [exp_paper/exp_r2_wvs_dropout.py](../exp_r2_wvs_dropout.py) | App. WVS-to-Trolley Dimension Linkage |

**Status: CODE** (syntax checked; Kaggle run pending)

---

## W10 — Logit conditioning diagnostic

| Item | File | Script | Paper section |
|---|---|---|---|
| W10. Decision-gap entropy / margin statistics + scatter vs MIS-improvement | [src/logit_conditioning.py](../../src/logit_conditioning.py) | [exp_paper/exp_r2_logit_conditioning.py](../exp_r2_logit_conditioning.py) | App. Logit Conditioning Diagnostic |

**Status: CODE** (syntax checked; Kaggle run pending)

---

## Extras raised by Round 2

| Item | File | Script | Status |
|---|---|---|---|
| Multi-seed CI for Phi-4 (3 seeds on 20 countries → 95% CI around MIS) | [exp_paper/exp_r2_multiseed_phi4.py](../exp_r2_multiseed_phi4.py) | same | CODE |

---

## Kaggle run order (recommended)

1. **Sensitivity + reliability audit** (no model reload between runs): `exp_r2_hparam_sensitivity.py`, `exp_r2_reliability_audit.py`
2. **Baselines on Phi-4**: `exp_r2_baseline_dropout.py`, `exp_r2_baseline_tempmargin.py`, `exp_r2_baseline_diffpo.py`
3. **Multi-seed**: `exp_r2_multiseed_phi4.py` (3 seeds × 20 countries)
4. **Ablation breadth**: `exp_r2_ablation_breadth.py` (3 models × 3 countries)
5. **Persona + no-oversampling + WVS-dropout + logit-conditioning + rank-agreement** (all Phi-4; can share a single session): `exp_r2_persona_variant.py`, `exp_r2_no_oversampling.py`, `exp_r2_wvs_dropout.py`, `exp_r2_logit_conditioning.py`, `exp_r2_rank_agreement.py`

Each script writes CSVs under `results/exp24_round2/<study>/` for post-hoc aggregation.

---

## Code + paper deliverables (2026-04-18)

### New `src/` modules
- [src/mc_dropout_runner.py](../../src/mc_dropout_runner.py) — MC-Dropout baseline (reviewer W1a)
- [src/calibration_baselines.py](../../src/calibration_baselines.py) — per-country T/margin scaling (W1b)
- [src/diffpo_binary_baseline.py](../../src/diffpo_binary_baseline.py) — DIFFPO-binary mixing baseline (W1c)
- [src/dpbr_reliability_audit.py](../../src/dpbr_reliability_audit.py) — post-hoc gate audit (W4b)
- [src/logit_conditioning.py](../../src/logit_conditioning.py) — decision-margin diagnostic (W10)
- [src/amce.py](../../src/amce.py) — new `compute_per_dim_rank_agreement()` (W7)

### Extended `src/` modules (backward-compatible flags)
- [src/personas.py](../../src/personas.py) — `build_country_personas(..., drop_dims=, fourth=)` + env hooks
  `SWA_WVS_DROP_DIMS`, `SWA_FOURTH_PERSONA` (W6, W9)
- [src/data.py](../../src/data.py) — `load_multitp_dataset(..., cap_per_category=, dump_ids_path=)` (W5)

### New Kaggle runners (under `exp_paper/`)
- `exp_r2_baseline_dropout.py`, `exp_r2_baseline_tempmargin.py`, `exp_r2_baseline_diffpo.py` (W1)
- `exp_r2_hparam_sensitivity.py`, `exp_r2_reliability_audit.py` (W3, W4)
- `exp_r2_wvs_dropout.py`, `exp_r2_persona_variant.py` (W6, W9)
- `exp_r2_logit_conditioning.py` (W10)
- `exp_r2_ablation_breadth.py`, `exp_r2_multiseed_phi4.py` (W2, multi-seed CI)
- `exp_r2_no_oversampling.py`, `scripts/dump_scenario_seeds.py` (W5)
- `exp_r2_rank_agreement.py` (W7)
- `_r2_common.py` — shared Kaggle bootstrap + per-country loop helpers

### Paper updates (paper_revised.tex)
- §Related Work: paragraph on PITA / DIFFPO / AISP / KTO + MC-Dropout clarification (W8a)
- §Baselines: three new baselines (6, 7, 8) added to the grid, future-work language removed
- §Results: new paragraph "Why MIS can improve while Pearson r stays negative" (W7)
- §Discussion: new paragraph on PT fragility under linguistic uncertainty (W8b)
- §Broader impact: per-persona floor safeguard paragraph (W8c)
- Figure 1 pipeline: 4th persona relabelled "Aggregate" (reconciles W6 text/figure mismatch)
- 9 new appendix sections: `app:r2_baselines`, `app:r2_hparam_sensitivity`,
  `app:r2_reliability_audit`, `app:r2_wvs_dropout`, `app:r2_persona_variant`,
  `app:r2_logit_conditioning`, `app:rank_agreement`, `app:r2_ablation_breadth`,
  `app:r2_multiseed`, `app:r2_oversampling`

### Citations added to references.bib (arXiv IDs are from the reviewer; author placeholders are `Anonymous` pending camera-ready lookup — audit before final submission)
- `chen2025diffpo` (2503.04240)
- `lee2025pita` (2507.20067)
- `aisp2025` (2510.26219)
- `ethayarajh2024kto` (2402.01306 — real authors filled in)
- `gal2016dropout` (ICML 2016 — real authors filled in)
- `kumar2025pt` (2508.08992 — PT fragility caveat)

### Outstanding (pending Kaggle GPU session)
1. Run `exp_r2_baseline_*` on Phi-4 × 20 countries → fill Table macro rows.
2. Run `exp_r2_hparam_sensitivity` (3 countries × 4 axes × 5 grid) → fill Table~\ref{tab:r2_hparam_sensitivity}.
3. Run `exp_r2_reliability_audit` (post-hoc; seconds) → regime counts.
4. Run `exp_r2_wvs_dropout` (3 countries × 11 configs) → linkage table.
5. Run `exp_r2_persona_variant` (20 countries × 2 variants) → head-to-head delta.
6. Run `exp_r2_logit_conditioning` (20 countries) → conditioning per-country CSV + scatter.
7. Run `exp_r2_ablation_breadth` (3 models × 3 countries × 6 rows = 54 cells).
8. Run `exp_r2_multiseed_phi4` (3 seeds × 20 countries × vanilla + SWA = 120 cells) → CI table.
9. Run `exp_r2_no_oversampling` + `scripts/dump_scenario_seeds.py` → reproducibility artifacts.
10. Post-hoc: run `exp_r2_rank_agreement.py` (reads existing SWA CSVs → ~seconds).

### Camera-ready bib audit
Before final camera-ready, manually verify arXiv-placeholder entries
(`chen2025diffpo`, `lee2025pita`, `aisp2025`, `kumar2025pt`) against the
arXiv page metadata and replace `Anonymous` with the real first-author
name. This follows the pre-existing repo policy of no fabricated
citations (see commit 67762d4 "Audit citations").
