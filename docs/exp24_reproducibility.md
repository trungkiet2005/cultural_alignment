# EXP-24 (DPBR): reproducibility & paper checklist

This note supports **NeurIPS-style** claims: what is fixed, what is tunable, and how to ablate.

**Relation to the SWA-PTIS manuscript** (`SWA_MPPI_paper/paper_revised.tex`): that draft describes the **single-pass** method and its limitations. EXP-24 is a **different** inference-time variant (dual-pass bootstrap reliability). A **line-by-line** map of paper limitations → DPBR inheritance → suggested fixes is in [`exp24_paper_gap_analysis.md`](exp24_paper_gap_analysis.md).

## Theory (single source of truth)

- **Implementation:** `experiment_DM/exp24_dpbr_core.py`
- **Runner (per-model sweep):** `exp_model/_base_dpbr.py`
- **Runner (multi-model):** `experiment_DM/exp24_dual_pass_bootstrap.py`

Do **not** fork the DPBR math into another file; extend the core module or import it.

## Hyperparameters

| Symbol | Role | Default | Notes |
|--------|------|---------|--------|
| `K_HALF` | IS samples **per** pass | `64` | Total IS budget `2 × K_HALF` = **128** (matches EXP-09 `K_samples`). |
| `VAR_SCALE` | Soft reliability scale `r = exp(-bootstrap_var / VAR_SCALE)` | `0.04` | At `bootstrap_var = 0.04`, `r = e^{-1} ≈ 0.37`. Ablate via env (below). |
| `N_WARMUP`, `DECAY_TAU`, `BETA_EMA` | EXP-09 hierarchical prior | 50, 100, 0.1 | Unchanged vs EXP-09. |

**Paper suggestion:** report sensitivity of main metrics to `VAR_SCALE` and (optionally) `K_HALF` on a **held-out validation slice** or fixed scenario subset, not only test-country aggregates.

## Environment variables (ablations)

Set **before** Python imports `experiment_DM.exp24_dpbr_core` (e.g. in the shell or Kaggle notebook first cell):

| Variable | Effect |
|----------|--------|
| `EXP24_VAR_SCALE` | Overrides `VAR_SCALE` (e.g. `0.02`, `0.08`). |
| `EXP24_K_HALF` | Overrides half-pass sample count (must keep `2 * K_HALF` consistent with reporting). |
| `EXP24_ESS_ANCHOR_REG` | `1` (default): ESS-adaptive anchor blend `δ = α·anchor + (1-α)·δ_base + δ*` with `α = clip(min(ESS₁,ESS₂), ρ, 1)` (EXP-05 / paper). `0`: legacy `δ = anchor + δ*` only. |

Example (Linux / Kaggle):

```bash
export EXP24_VAR_SCALE=0.08
python exp_model/exp_24/exp_phi_4.py
```

**Windows PowerShell:**

```powershell
$env:EXP24_VAR_SCALE="0.08"; python exp_model/exp_24/exp_phi_4.py
```

Helper script (prints effective values after import):

```bash
python experiment_DM/exp24_ablation_env.py
```

## Seeds & determinism

- **EXP-24 runs** call `setup_seeds(42)` in `_base_dpbr.run_for_model`.
- Torch / cuDNN deterministic flags are set in `setup_seeds`; tiny numerical drift may still appear on GPU.
- For **bit-identical** reruns, document the exact **GPU type**, **CUDA**, **torch**, and **transformers/unsloth** versions per model family (`ref_*` profiles in each `exp_24/exp_*.py`).

## Model stack matrix (per `exp_24` script)

Each `exp_model/exp_24/exp_*.py` documents its own `ref_*` pip profile (e.g. `ref_qwen3`, `ref_gemma4`). **Use a fresh Kaggle session** when switching families (different `transformers` pins).

## Diagnostics logged in CSV

- `reliability_r`, `bootstrap_var`, `ess_pass1`, `ess_pass2`, `delta_star_1`, `delta_star_2`
- `p_spare_preferred_is_pass1_micro`, `p_spare_preferred_is_pass2_micro`: **micro-only** preference prob from each IS pass (no hierarchical prior); for analysis, not the primary decision.

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/test_exp24_dpbr.py -q
```

## Controller wiring (no monkey-patches)

`ImplicitSWAController` receives `country_iso` from `src/swa_runner.run_country_experiment`; `Exp24DualPassController` uses it for the per-country prior. EXP-24 runners no longer patch `__init__` at runtime.
