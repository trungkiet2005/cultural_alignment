# Phase 3 — sensitivity + diagnostics (H100 80GB, ≈4–6 h combined)

Everything in this phase runs on Phi-4 and is tunable via `R2_N_SCENARIOS`
if you're short on session time.

| Script | Reviewer | What it does |
|---|---|---|
| `exp_r2_logit_conditioning.py`  | W10  | Per-scenario decision margin / entropy / |gap| — explains why some countries benefit more than others. |
| `exp_r2_persona_variant.py`     | W6   | Head-to-head between `fourth=aggregate` (current) and `fourth=utilitarian` (original figure 1 label). |
| `exp_r2_hparam_sensitivity.py`  | W3+W4a | Sweep s / λ_coop / σ / T_cat on 3-country panel (5 points per axis, 4 axes). |
| `exp_r2_wvs_dropout.py`         | W9   | Drop each WVS dimension one at a time → track per-MultiTP-dim AMCE error. |
| `exp_r2_no_oversampling.py`     | W5b  | Rerun Phi-4 with the per-category cap disabled (+ scenario-id dump). |

## Run (can share a single session)

```python
!python exp_paper/round2/phase3_sensitivity/exp_r2_logit_conditioning.py      # ~30 min
!python exp_paper/round2/phase3_sensitivity/exp_r2_persona_variant.py         # ~2 h
!R2_N_SCENARIOS=250 python exp_paper/round2/phase3_sensitivity/exp_r2_hparam_sensitivity.py   # ~6 h at default
!python exp_paper/round2/phase3_sensitivity/exp_r2_wvs_dropout.py             # ~2 h
!python exp_paper/round2/phase3_sensitivity/exp_r2_no_oversampling.py         # ~2 h
```

## Key env overrides

- `R2_AXES` (hparam only): comma list from `{s,lambda,sigma,tcat}` (default: all 4)
- `R2_DROP_SET` (wvs_dropout only): comma list of dim names (default: all 10)
- `SWA_WVS_DROP_DIMS` / `SWA_FOURTH_PERSONA` — global overrides if you want to
  apply a drop/variant to **any** script that uses `build_country_personas`.
