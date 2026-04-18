# Phase 2 — inference-time baselines (H100 80GB, ≈2–3 h each)

Three additional baselines on the 20-country Phi-4 grid — these fill the
Round-2 reviewer-requested rows in the main results table.

| Script | Reviewer | Method | Backend |
|---|---|---|---|
| `exp_r2_baseline_dropout.py`   | W1a | MC-Dropout (T=8, p=0.10) uncertainty inflation | **`hf_native`** (vLLM has no dropout hooks) |
| `exp_r2_baseline_tempmargin.py` | W1b | Per-country T_c **or** m_c fit on 25% cal split, applied to 75% test | `vllm` (default) |
| `exp_r2_baseline_diffpo.py`    | W1c | `(1-α)·p_vanilla + α·p_target` where `p_target` comes from the public human AMCE (α fit on cal split) | `vllm` (default) |

## Run

```python
# Session A — vLLM runs (parallel-safe in same session if you want)
!python exp_paper/round2/phase2_baselines/exp_r2_baseline_tempmargin.py
!python exp_paper/round2/phase2_baselines/exp_r2_baseline_diffpo.py

# Session B — MC-Dropout (hf_native, forced automatically)
!python exp_paper/round2/phase2_baselines/exp_r2_baseline_dropout.py
```

## Env overrides (common to all three)

- `R2_MODEL` (default `microsoft/phi-4`)
- `R2_COUNTRIES` (default: 20 paper countries)
- `R2_N_SCENARIOS` (default 500 for tempmargin/diffpo; 500 for dropout)
- `R2_CAL_FRAC` (default 0.25) — calibration-split size for tempmargin/diffpo
- `R2_MC_T` / `R2_MC_P` — MC-Dropout pass count / override rate

## Output

`results/exp24_round2/mc_dropout/`, `.../tempmargin/`, `.../diffpo_binary/` —
each with per-country CSVs + a summary CSV.
