# Phase 4 — expensive sweeps (H100 80GB, ≈6–8 h each)

Run these **last** — each wants a long Kaggle session. Partial CSVs are
streamed after every cell so a timeout keeps whatever's already done.

| Script | Reviewer | Shape | ETA |
|---|---|---|---|
| `exp_r2_ablation_breadth.py`   | W2 | 3 models × 3 countries × 6 ablation rows = 54 cells | ≈6 h |
| `exp_r2_multiseed_phi4.py`     | CI | 3 seeds × 20 countries × (vanilla + SWA-DPBR) = 120 cells | ≈8 h |

## Run (separate Kaggle sessions)

```python
# Session A
!python exp_paper/round2/phase4_big_sweeps/exp_r2_ablation_breadth.py

# Session B
!python exp_paper/round2/phase4_big_sweeps/exp_r2_multiseed_phi4.py
```

## Tuning for shorter sessions

- `exp_r2_ablation_breadth.py`: set `R2_BREADTH_MODELS=microsoft/phi-4` and
  `R2_BREADTH_COUNTRIES=USA,JPN,VNM` for a 1-model × 3-country smoke test.
- `exp_r2_multiseed_phi4.py`: `R2_SEEDS=42,101` cuts to 2 seeds;
  `R2_COUNTRIES=USA,VNM,DEU` cuts the country panel.

## Don't do this on a single H100

Adding Llama-3.3-70B to `R2_BREADTH_MODELS` requires 2×H100 with
`VLLM_TENSOR_PARALLEL_SIZE=2`. The default model list (Phi-4,
Qwen2.5-7B, Phi-3.5-mini) all fit BF16 on one H100 80GB.
