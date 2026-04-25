# Phase 7 — Oral-level experiments

Four targeted experiments that address the remaining reviewer gaps and push
the paper toward NeurIPS 2026 oral consideration.

## Motivation

| Script | Reviewer question | GPU? | Time (H100) |
|---|---|---|---|
| `exp_r2_per_dim_mis.py`       | Q6 — per-dimension AMCE breakdown          | No  | ~1 min  |
| `exp_r2_persona_amce_corr.py` | Q5 — do WVS personas track human AMCEs?    | No  | ~2 min  |
| `exp_r2_prism_baseline.py`    | Missing comparator: cultural prompting     | Yes | ~2–3 h  |
| `exp_r2_latency_benchmark.py` | Q7 — wall-time overhead vs vanilla         | Yes | ~30 min |
| `aggregate_phase7.py`         | Merge all four into LaTeX tables + summary | No  | ~1 min  |

---

## Script details

### `exp_r2_per_dim_mis.py` — per-dimension AMCE breakdown (NO GPU)

Decomposes the aggregate 19–24% MIS gain into six MultiTP moral dimensions:
Species · Gender · Age · Fitness · SocialValue · Utilitarianism.

For each country × dimension reports `vanilla_err`, `swa_err`, `delta`,
`pct_gain`. Produces a colour-coded LaTeX heatmap table and a summary text
with macro mean gain per dimension.

**Expected finding:** gain is broad across dimensions but not uniform;
one or two dimensions may show regression in specific countries, which is
an honest and nuanced finding that reviewers will appreciate.

**Outputs:** `results/exp24_round2/per_dim_mis/`

```bash
python exp_paper/review/round2/phase7_oral/exp_r2_per_dim_mis.py
```

Key env override: `R2_MODEL_SHORT=phi_4`, `R2_MODEL_SLUG=phi-4`

---

### `exp_r2_persona_amce_corr.py` — WVS → AMCE correlation (NO GPU)

Validates the SWA-DPBR premise: do the WVS-grounded personas point in the
right direction, independently of any LLM correction?

Two analyses:
1. **OLS regression** — predicts each of the 6 human AMCE dimensions from
   the 10 WVS country-level feature means. Reports R², adjusted R²,
   standardised coefficients per AMCE dimension.
2. **Pearson/Spearman correlation** — pairwise WVS feature vs AMCE dim
   across the 20 countries.

**Expected finding:** WVS religiosity, gender_equality, and tolerance_diversity
will be the strongest predictors of the corresponding AMCE dimensions (e.g.,
gender_equality → Gender_Female AMCE), with R² > 0.4 for several dimensions —
confirming the WVS grounding is empirically justified.

**Outputs:** `results/exp24_round2/persona_amce_corr/`

```bash
python exp_paper/review/round2/phase7_oral/exp_r2_persona_amce_corr.py
```

---

### `exp_r2_prism_baseline.py` — cultural prompting baseline (GPU)

PRISM-style comparator: prepend a short cultural-context sentence to every
scenario ("Please answer as a typical person from {country} would") and
measure AMCE/MIS with no logit correction.

This directly answers the reviewer's implicit question: "why not just prompt
the model to behave like someone from that culture?"

**Expected finding:** PRISM prompting gives 2–6% MIS reduction (modest);
SWA-DPBR gives 19–24% — confirming that logit-space IS correction is
doing real work beyond what cultural framing alone achieves.

**Env:**
```bash
R2_N_SCENARIOS=300   # 300 per country (~2h on H100)
R2_PRISM_STRENGTH=short   # or 'long' for stronger framing
```

**Outputs:** `results/exp24_round2/prism_baseline/`

```bash
python exp_paper/review/round2/phase7_oral/exp_r2_prism_baseline.py
```

---

### `exp_r2_latency_benchmark.py` — wall-time overhead (GPU)

Measures seconds-per-scenario for:
- Vanilla decoding
- SWA-DPBR at K = 32, 64, 128 (total IS samples)

Also records GPU peak memory and DPBR reliability gate flip rate per K.

**Expected finding:** SWA-DPBR at K=128 adds ~1.2–1.4× wall-time overhead
(not 128× because the IS loop is pure Python / tensor math, not extra
transformer forward passes). This is a strong practical deployability
argument.

**Env:**
```bash
R2_COUNTRIES=USA,VNM,DEU   # 3-country panel
R2_N_SCENARIOS=100          # 100 per country (~30 min)
R2_K_GRID=32,64,128
```

**Outputs:** `results/exp24_round2/latency/`

```bash
python exp_paper/review/round2/phase7_oral/exp_r2_latency_benchmark.py
```

---

### `aggregate_phase7.py` — merge into paper-ready LaTeX (NO GPU)

Reads all phase-7 CSVs and produces:
```
results/exp24_round2/phase7_oral/
  ├── per_dim_macro_table.tex    # 6-row table: macro gain per dimension
  ├── wvs_regression_r2.tex     # R² of WVS→AMCE OLS per dimension
  ├── prism_macro_compare.tex   # PRISM vs SWA-DPBR macro MIS row
  ├── latency_compact.tex       # overhead table (K=32/64/128)
  └── phase7_summary.txt        # one-liners for paper body / rebuttal
```

Run after all four experiment scripts complete.

```bash
python exp_paper/review/round2/phase7_oral/aggregate_phase7.py
```

---

## Recommended Kaggle session order

**Session A (no GPU, run locally or on any Kaggle session):**
```bash
python exp_paper/review/round2/phase7_oral/exp_r2_per_dim_mis.py
python exp_paper/review/round2/phase7_oral/exp_r2_persona_amce_corr.py
```

**Session B (H100, ~30 min):**
```bash
python exp_paper/review/round2/phase7_oral/exp_r2_latency_benchmark.py
```

**Session C (H100, ~2–3 h):**
```bash
python exp_paper/review/round2/phase7_oral/exp_r2_prism_baseline.py
```

**Session D (no GPU, after A–C):**
```bash
python exp_paper/review/round2/phase7_oral/aggregate_phase7.py
```

---

## Where findings land in the paper

| Finding | Paper location |
|---|---|
| Per-dim heatmap | New Appendix §A.per_dim or inline Table |
| WVS → AMCE R² | §Method WVS grounding paragraph + Appendix |
| PRISM comparison | §Experiments Table 2 (new row) |
| Latency table | §Experiments / Appendix §deployment |
