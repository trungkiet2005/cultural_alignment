# Exp Model Performance Tracker

> Purpose: track method/model performance across countries to quickly see which model-method combo is best.
> Primary metric: `MIS` (lower is better). Secondary: `JSD` (lower), `Pearson r` (higher), `MAE` (lower), `Flip%`.

---

## Leaderboard by Method Run

| Rank | Run ID | Method | Model(s) | Coverage | Mean MIS ↓ | Mean JSD ↓ | Mean r ↑ | Mean MAE ↓ | Mean Flip% | Notes |
|:----:|:-------|:-------|:---------|:--------:|-----------:|-----------:|---------:|-----------:|-----------:|:------|
| 1 | EXP-24-PHI_4 | Dual-Pass Bootstrap IS Reliability (DPBR) | Phi-4 | 1 model x 5 countries | **0.2462** | 0.0619 | +0.692 | 7.93 | 16.83% | Win vs vanilla: 4/5, macro +34.35% |
| 2 | EXP-24-QWEN3_VL_8B | Dual-Pass Bootstrap IS Reliability (DPBR) | Qwen3-VL-8B | 1 model x 5 countries | **0.3813** | 0.0871 | +0.446 | 11.81 | 15.15% | Win vs vanilla: 4/5, macro +23.74% |
| 3 | EXP-24-QWEN35_08B | Dual-Pass Bootstrap IS Reliability (DPBR) | Qwen3.5-0.8B | 1 model x 5 countries | 0.4708 | 0.0485 | -0.504 | 16.93 | 6.50% | Win vs vanilla: 4/5, macro +3.22% |
| 4 | EXP-24-GEMMA_7B | Dual-Pass Bootstrap IS Reliability (DPBR) | gemma-7b-it-bnb-4bit | 1 model x 5 countries | 0.4315 | 0.0575 | -0.477 | 14.76 | 11.22% | Win vs vanilla: 3/5, macro +2.46% |
| 5 | EXP-24-LLAMA32_1B | Dual-Pass Bootstrap IS Reliability (DPBR) | Llama-3.2-1B | 1 model x 5 countries | 0.4761 | 0.0586 | -0.558 | 17.16 | 0.76% | Win vs vanilla: 1/5, macro +1.14% |
| 6 | EXP-09-LLAMA32_1B | Hierarchical IS | Llama-3.2-1B | 1 model x 5 countries | 0.4762 | 0.0586 | -0.570 | 17.17 | 0.84% | Win vs vanilla: 1/5, macro +1.11% |
| 7 | EXP-24-QWEN2_7B | Dual-Pass Bootstrap IS Reliability (DPBR) | Qwen2-7B | 1 model x 5 countries | 0.3946 | 0.0523 | +0.194 | 13.60 | 16.03% | Win vs vanilla: 2/5, macro -3.53% |
| 8 | EXP-24-QWEN3_8B | Dual-Pass Bootstrap IS Reliability (DPBR) | Qwen3-8B-unsloth-bnb-4bit | 1 model x 5 countries | 0.4957 | 0.0529 | -0.441 | 17.78 | 7.03% | Win vs vanilla: 0/5, macro -5.86% |
| 9 | EXP-24-QWEN35_4B | Dual-Pass Bootstrap IS Reliability (DPBR) | Qwen3.5-4B | 1 model x 5 countries | 0.3983 | **0.0395** | +0.527 | 14.34 | 14.65% | Win vs vanilla: 0/5, macro -24.03% |
| 10 | EXP-24-QWEN3_CODER_30B | Dual-Pass Bootstrap IS Reliability (DPBR) | Qwen3-Coder-30B-A3B | 1 model x 5 countries | **0.3900** | 0.0752 | +0.503 | 12.86 | 16.13% | Win vs vanilla: 1/5, macro -24.41% |

---

## Run: EXP-24-QWEN35_08B

- Date: 2026-04-10
- Script: `exp_model/exp_qwen35_08b.py`
- Model: `unsloth/Qwen3.5-0.8B`
- Method: Dual-Pass Bootstrap IS Reliability (DPBR)
- Output dir: `/kaggle/working/cultural_alignment/results/exp24_model_sweep/qwen35_08b/compare`

### Full Metrics (DPBR)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Qwen3.5-0.8B | BRA | 0.4201 | 0.0419 | -0.430 | 15.38 | 6.8% |
| Qwen3.5-0.8B | CHN | 0.4779 | 0.0506 | -0.353 | 17.02 | 9.7% |
| Qwen3.5-0.8B | DEU | 0.5072 | 0.0547 | -0.651 | 17.93 | 0.0% |
| Qwen3.5-0.8B | JPN | 0.4438 | 0.0473 | -0.606 | 15.86 | 11.9% |
| Qwen3.5-0.8B | USA | 0.5049 | 0.0481 | -0.481 | 18.48 | 3.9% |

### vs Vanilla (MIS)

| Model | Country | Vanilla MIS | DPBR MIS | Delta (ref-cur) | Improv% | Win? |
|:------|:-------:|------------:|---------:|----------------:|--------:|:----:|
| Qwen3.5-0.8B | BRA | 0.4415 | 0.4201 | +0.0213 | +4.83% | ✅ |
| Qwen3.5-0.8B | CHN | 0.4890 | 0.4779 | +0.0111 | +2.28% | ✅ |
| Qwen3.5-0.8B | DEU | 0.4971 | 0.5072 | -0.0101 | -2.02% | ❌ |
| Qwen3.5-0.8B | JPN | 0.4895 | 0.4438 | +0.0457 | +9.34% | ✅ |
| Qwen3.5-0.8B | USA | 0.5150 | 0.5049 | +0.0101 | +1.97% | ✅ |

- Win rate: **4/5**
- Mean vanilla MIS: **0.4864**
- Mean method MIS: **0.4708**
- Macro improvement: **+3.22%**
- Mean per-row improvement (micro): **+3.28%**

### Per-Dimension Worst Error

| Model | Country | Worst Dimension | Human | Model | \|err\| (pp) |
|:------|:-------:|:----------------|:-----:|:-----:|:------------:|
| Qwen3.5-0.8B | BRA | Utilitarianism_More | 73.7 | 49.6 | 24.1 |
| Qwen3.5-0.8B | CHN | Species_Humans | 83.0 | 49.4 | 33.6 |
| Qwen3.5-0.8B | DEU | Species_Humans | 82.4 | 49.3 | 33.1 |
| Qwen3.5-0.8B | JPN | Species_Humans | 79.8 | 49.6 | 30.3 |
| Qwen3.5-0.8B | USA | Species_Humans | 79.2 | 49.5 | 29.7 |

### Notes

- Strong overall gain vs vanilla (4/5 wins), with the biggest improvement in JPN.
- Species_Humans remains the largest residual error in 4/5 countries.

---

## Run: EXP-24-PHI_4

- Date: 2026-04-10
- Script: `exp_model/exp_phi_4.py`
- Model: `unsloth/Phi-4`
- Method: Dual-Pass Bootstrap IS Reliability (DPBR)
- Output dir: `/kaggle/working/cultural_alignment/results/exp24_model_sweep/phi_4/compare`

### Full Metrics (DPBR)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Phi-4 | BRA | 0.2971 | 0.0717 | +0.485 | 9.39 | 19.7% |
| Phi-4 | CHN | 0.2144 | 0.0568 | +0.812 | 5.98 | 13.2% |
| Phi-4 | DEU | 0.2703 | 0.0705 | +0.744 | 9.85 | 16.1% |
| Phi-4 | JPN | 0.2173 | 0.0517 | +0.721 | 7.10 | 16.5% |
| Phi-4 | USA | 0.2321 | 0.0586 | +0.701 | 7.33 | 18.7% |

### vs Vanilla (MIS)

| Model | Country | Vanilla MIS | DPBR MIS | Delta (ref-cur) | Improv% | Win? |
|:------|:-------:|------------:|---------:|----------------:|--------:|:----:|
| Phi-4 | BRA | 0.2739 | 0.2971 | -0.0233 | -8.49% | ❌ |
| Phi-4 | CHN | 0.4069 | 0.2144 | +0.1925 | +47.31% | ✅ |
| Phi-4 | DEU | 0.3023 | 0.2703 | +0.0320 | +10.57% | ✅ |
| Phi-4 | JPN | 0.3948 | 0.2173 | +0.1774 | +44.95% | ✅ |
| Phi-4 | USA | 0.4975 | 0.2321 | +0.2654 | +53.35% | ✅ |

- Win rate: **4/5**
- Mean vanilla MIS: **0.3750**
- Mean method MIS: **0.2462**
- Macro improvement: **+34.35%**
- Mean per-row improvement (micro): **+29.54%**

### Per-Dimension Worst Error

| Model | Country | Worst Dimension | Human | Model | \|err\| (pp) |
|:------|:-------:|:----------------|:-----:|:-----:|:------------:|
| Phi-4 | BRA | SocialValue_High | 66.3 | 45.8 | 20.5 |
| Phi-4 | CHN | SocialValue_High | 66.7 | 50.2 | 16.5 |
| Phi-4 | DEU | SocialValue_High | 64.7 | 47.1 | 17.6 |
| Phi-4 | JPN | Utilitarianism_More | 68.7 | 84.4 | 15.7 |
| Phi-4 | USA | SocialValue_High | 67.9 | 49.6 | 18.4 |

### Notes

- Large gains in USA/CHN/JPN; BRA is the only regression vs vanilla.
- SocialValue_High remains the dominant error dimension in 4/5 countries.

---

## Run: EXP-24-QWEN3_VL_8B

- Date: 2026-04-10
- Script: `exp_model/exp_qwen3_vl_8b.py`
- Model: `unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit`
- Method: Dual-Pass Bootstrap IS Reliability (DPBR)
- Output dir: `/kaggle/working/cultural_alignment/results/exp24_model_sweep/qwen3_vl_8b/compare`

### Full Metrics (DPBR)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Qwen3-VL-8B | BRA | 0.3580 | 0.0842 | +0.358 | 10.94 | 18.1% |
| Qwen3-VL-8B | CHN | 0.3890 | 0.0973 | +0.408 | 10.65 | 14.2% |
| Qwen3-VL-8B | DEU | 0.3215 | 0.0742 | +0.553 | 9.64 | 13.5% |
| Qwen3-VL-8B | JPN | 0.3557 | 0.0844 | +0.407 | 11.54 | 15.5% |
| Qwen3-VL-8B | USA | 0.4826 | 0.0954 | +0.504 | 16.26 | 14.5% |

### vs Vanilla (MIS)

| Model | Country | Vanilla MIS | DPBR MIS | Delta (ref-cur) | Improv% | Win? |
|:------|:-------:|------------:|---------:|----------------:|--------:|:----:|
| Qwen3-VL-8B | BRA | 0.5288 | 0.3580 | +0.1708 | +32.30% | ✅ |
| Qwen3-VL-8B | CHN | 0.5693 | 0.3890 | +0.1803 | +31.67% | ✅ |
| Qwen3-VL-8B | DEU | 0.3181 | 0.3215 | -0.0034 | -1.05% | ❌ |
| Qwen3-VL-8B | JPN | 0.4075 | 0.3557 | +0.0519 | +12.73% | ✅ |
| Qwen3-VL-8B | USA | 0.6767 | 0.4826 | +0.1941 | +28.68% | ✅ |

- Win rate: **4/5**
- Mean vanilla MIS: **0.5001**
- Mean method MIS: **0.3813**
- Macro improvement: **+23.74%**
- Mean per-row improvement (micro): **+20.86%**

### Per-Dimension Worst Error

| Model | Country | Worst Dimension | Human | Model | \|err\| (pp) |
|:------|:-------:|:----------------|:-----:|:-----:|:------------:|
| Qwen3-VL-8B | BRA | SocialValue_High | 66.3 | 35.3 | 31.1 |
| Qwen3-VL-8B | CHN | SocialValue_High | 66.7 | 30.2 | 36.5 |
| Qwen3-VL-8B | DEU | SocialValue_High | 64.7 | 36.9 | 27.8 |
| Qwen3-VL-8B | JPN | SocialValue_High | 65.9 | 36.9 | 29.0 |
| Qwen3-VL-8B | USA | SocialValue_High | 67.9 | 30.2 | 37.7 |

### Notes

- Strong gains in 4/5 countries vs vanilla; only DEU is a slight regression.
- SocialValue_High is consistently the largest error dimension across all countries.

---

## Run: EXP-24-QWEN35_4B

- Date: 2026-04-10
- Script: `exp_model/exp_qwen35_4b.py`
- Model: `unsloth/Qwen3.5-4B`
- Method: Dual-Pass Bootstrap IS Reliability (DPBR)
- Output dir: `/kaggle/working/cultural_alignment/results/exp24_model_sweep/qwen35_4b/compare`

### Full Metrics (DPBR)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Qwen3.5-4B | BRA | 0.3574 | 0.0335 | +0.563 | 13.25 | 17.4% |
| Qwen3.5-4B | CHN | 0.3689 | 0.0428 | +0.490 | 12.69 | 19.0% |
| Qwen3.5-4B | DEU | 0.4386 | 0.0495 | +0.314 | 15.16 | 16.1% |
| Qwen3.5-4B | JPN | 0.3905 | 0.0400 | +0.492 | 14.03 | 12.3% |
| Qwen3.5-4B | USA | 0.4362 | 0.0316 | +0.775 | 16.59 | 8.4% |

### vs Vanilla (MIS)

| Model | Country | Vanilla MIS | DPBR MIS | Delta (ref-cur) | Improv% | Win? |
|:------|:-------:|------------:|---------:|----------------:|--------:|:----:|
| Qwen3.5-4B | BRA | 0.3225 | 0.3574 | -0.0350 | -10.85% | ❌ |
| Qwen3.5-4B | CHN | 0.2345 | 0.3689 | -0.1344 | -57.34% | ❌ |
| Qwen3.5-4B | DEU | 0.3965 | 0.4386 | -0.0421 | -10.62% | ❌ |
| Qwen3.5-4B | JPN | 0.3300 | 0.3905 | -0.0604 | -18.31% | ❌ |
| Qwen3.5-4B | USA | 0.3222 | 0.4362 | -0.1140 | -35.37% | ❌ |

- Win rate: **0/5**
- Mean vanilla MIS: **0.3212**
- Mean method MIS: **0.3983**
- Macro improvement: **-24.03%**
- Mean per-row improvement (micro): **-26.50%**

### Per-Dimension Worst Error

| Model | Country | Worst Dimension | Human | Model | \|err\| (pp) |
|:------|:-------:|:----------------|:-----:|:-----:|:------------:|
| Qwen3.5-4B | BRA | Age_Young | 73.6 | 50.9 | 22.7 |
| Qwen3.5-4B | CHN | Species_Humans | 83.0 | 55.9 | 27.0 |
| Qwen3.5-4B | DEU | Species_Humans | 82.4 | 52.6 | 29.9 |
| Qwen3.5-4B | JPN | Species_Humans | 79.8 | 53.6 | 26.2 |
| Qwen3.5-4B | USA | Age_Young | 74.5 | 50.2 | 24.4 |

### Notes

- DPBR underperforms vanilla for this model in all 5 countries (0/5 wins).
- Largest errors concentrate on Species_Humans and Age_Young.

---

## Run: EXP-24-QWEN3_CODER_30B

- Date: 2026-04-10
- Script: `exp_model/exp_qwen3_coder_30b.py`
- Model: `unsloth/Qwen3-Coder-30B-A3B-Instruct`
- Method: Dual-Pass Bootstrap IS Reliability (DPBR)
- Output dir: `/kaggle/working/cultural_alignment/results/exp24_model_sweep/qwen3_coder_30b/compare`

### Full Metrics (DPBR)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Qwen3-Coder-30B-A3B | BRA | 0.3595 | 0.0720 | +0.070 | 11.29 | 17.7% |
| Qwen3-Coder-30B-A3B | CHN | 0.4269 | 0.0855 | +0.579 | 14.21 | 12.9% |
| Qwen3-Coder-30B-A3B | DEU | 0.3282 | 0.0559 | +0.647 | 9.76 | 16.1% |
| Qwen3-Coder-30B-A3B | JPN | 0.3771 | 0.0701 | +0.513 | 13.03 | 20.6% |
| Qwen3-Coder-30B-A3B | USA | 0.4583 | 0.0923 | +0.705 | 15.99 | 13.2% |

### vs Vanilla (MIS)

| Model | Country | Vanilla MIS | DPBR MIS | Delta (ref-cur) | Improv% | Win? |
|:------|:-------:|------------:|---------:|----------------:|--------:|:----:|
| Qwen3-Coder-30B-A3B | BRA | 0.3309 | 0.3595 | -0.0286 | -8.65% | ❌ |
| Qwen3-Coder-30B-A3B | CHN | 0.2900 | 0.4269 | -0.1368 | -47.18% | ❌ |
| Qwen3-Coder-30B-A3B | DEU | 0.3297 | 0.3282 | +0.0015 | +0.45% | ✅ |
| Qwen3-Coder-30B-A3B | JPN | 0.2611 | 0.3771 | -0.1160 | -44.42% | ❌ |
| Qwen3-Coder-30B-A3B | USA | 0.3557 | 0.4583 | -0.1026 | -28.85% | ❌ |

- Win rate: **1/5**
- Mean vanilla MIS: **0.3135**
- Mean method MIS: **0.3900**
- Macro improvement: **-24.41%**
- Mean per-row improvement (micro): **-25.73%**

### Per-Dimension Worst Error

| Model | Country | Worst Dimension | Human | Model | \|err\| (pp) |
|:------|:-------:|:----------------|:-----:|:-----:|:------------:|
| Qwen3-Coder-30B-A3B | BRA | SocialValue_High | 66.3 | 41.6 | 24.8 |
| Qwen3-Coder-30B-A3B | CHN | SocialValue_High | 66.7 | 35.5 | 31.2 |
| Qwen3-Coder-30B-A3B | DEU | Age_Young | 73.9 | 49.2 | 24.8 |
| Qwen3-Coder-30B-A3B | JPN | SocialValue_High | 65.9 | 38.0 | 27.9 |
| Qwen3-Coder-30B-A3B | USA | SocialValue_High | 67.9 | 38.4 | 29.5 |

### Notes

- Only DEU improves slightly; strong regressions in USA/CHN/JPN vs vanilla.
- SocialValue_High is the main failure mode in 4/5 countries.

---

## Run: EXP-24-QWEN2_7B

- Date: 2026-04-10
- Script: `exp_model/exp_qwen2_7b.py`
- Model: `unsloth/Qwen2-7B-Instruct-bnb-4bit`
- Method: Dual-Pass Bootstrap IS Reliability (DPBR)
- Output dir: `/kaggle/working/cultural_alignment/results/exp24_model_sweep/qwen2_7b/compare`

### Full Metrics (DPBR)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Qwen2-7B | BRA | 0.3462 | 0.0453 | +0.176 | 11.69 | 12.3% |
| Qwen2-7B | CHN | 0.3449 | 0.0608 | +0.426 | 10.85 | 15.8% |
| Qwen2-7B | DEU | 0.4598 | 0.0537 | +0.035 | 15.82 | 15.5% |
| Qwen2-7B | JPN | 0.3400 | 0.0523 | +0.077 | 12.04 | 15.5% |
| Qwen2-7B | USA | 0.4823 | 0.0496 | +0.255 | 17.61 | 21.0% |

### vs Vanilla (MIS)

| Model | Country | Vanilla MIS | DPBR MIS | Delta (ref-cur) | Improv% | Win? |
|:------|:-------:|------------:|---------:|----------------:|--------:|:----:|
| Qwen2-7B | BRA | 0.4195 | 0.3462 | +0.0733 | +17.48% | ✅ |
| Qwen2-7B | CHN | 0.3064 | 0.3449 | -0.0385 | -12.58% | ❌ |
| Qwen2-7B | DEU | 0.3956 | 0.4598 | -0.0642 | -16.22% | ❌ |
| Qwen2-7B | JPN | 0.3411 | 0.3400 | +0.0011 | +0.33% | ✅ |
| Qwen2-7B | USA | 0.4432 | 0.4823 | -0.0391 | -8.82% | ❌ |

- Win rate: **2/5**
- Mean vanilla MIS: **0.3812**
- Mean method MIS: **0.3946**
- Macro improvement: **-3.53%**
- Mean per-row improvement (micro): **-3.96%**

### Per-Dimension Worst Error

| Model | Country | Worst Dimension | Human | Model | \|err\| (pp) |
|:------|:-------:|:----------------|:-----:|:-----:|:------------:|
| Qwen2-7B | BRA | Utilitarianism_More | 73.7 | 53.1 | 20.6 |
| Qwen2-7B | CHN | SocialValue_High | 66.7 | 40.5 | 26.2 |
| Qwen2-7B | DEU | Species_Humans | 82.4 | 55.2 | 27.3 |
| Qwen2-7B | JPN | Species_Humans | 79.8 | 61.4 | 18.4 |
| Qwen2-7B | USA | SocialValue_High | 67.9 | 41.8 | 26.2 |

### Notes

- Mixed outcomes: wins only in BRA and JPN; regressions in USA/CHN/DEU.
- SocialValue_High and Species_Humans remain dominant weak dimensions.

---

## Run: EXP-24-GEMMA_7B

- Date: 2026-04-10
- Script: `exp_model/exp_gemma_7b.py`
- Model: `unsloth/gemma-7b-it-bnb-4bit`
- Method: Dual-Pass Bootstrap IS Reliability (DPBR)
- Output dir: `/kaggle/working/cultural_alignment/results/exp24_model_sweep/gemma_7b/compare`

### Full Metrics (DPBR)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| gemma-7b-it-bnb-4bit | BRA | 0.3534 | 0.0526 | -0.372 | 12.49 | 12.9% |
| gemma-7b-it-bnb-4bit | CHN | 0.4492 | 0.0514 | -0.136 | 15.59 | 1.6% |
| gemma-7b-it-bnb-4bit | DEU | 0.4432 | 0.0722 | -0.612 | 14.54 | 7.7% |
| gemma-7b-it-bnb-4bit | JPN | 0.4066 | 0.0596 | -0.452 | 12.96 | 23.5% |
| gemma-7b-it-bnb-4bit | USA | 0.5050 | 0.0515 | -0.812 | 18.24 | 10.3% |

### vs Vanilla (MIS)

| Model | Country | Vanilla MIS | DPBR MIS | Delta (ref-cur) | Improv% | Win? |
|:------|:-------:|------------:|---------:|----------------:|--------:|:----:|
| gemma-7b-it-bnb-4bit | BRA | 0.4162 | 0.3534 | +0.0628 | +15.08% | ✅ |
| gemma-7b-it-bnb-4bit | CHN | 0.4209 | 0.4492 | -0.0283 | -6.73% | ❌ |
| gemma-7b-it-bnb-4bit | DEU | 0.4256 | 0.4432 | -0.0176 | -4.14% | ❌ |
| gemma-7b-it-bnb-4bit | JPN | 0.4417 | 0.4066 | +0.0351 | +7.95% | ✅ |
| gemma-7b-it-bnb-4bit | USA | 0.5073 | 0.5050 | +0.0024 | +0.47% | ✅ |

- Win rate: **3/5**
- Mean vanilla MIS: **0.4423**
- Mean method MIS: **0.4315**
- Macro improvement: **+2.46%**
- Mean per-row improvement (micro): **+2.53%**

### Per-Dimension Worst Error

| Model | Country | Worst Dimension | Human | Model | \|err\| (pp) |
|:------|:-------:|:----------------|:-----:|:-----:|:------------:|
| gemma-7b-it-bnb-4bit | BRA | Age_Young | 73.6 | 48.8 | 24.8 |
| gemma-7b-it-bnb-4bit | CHN | Species_Humans | 83.0 | 49.2 | 33.8 |
| gemma-7b-it-bnb-4bit | DEU | Species_Humans | 82.4 | 50.0 | 32.5 |
| gemma-7b-it-bnb-4bit | JPN | Species_Humans | 79.8 | 50.3 | 29.5 |
| gemma-7b-it-bnb-4bit | USA | Species_Humans | 79.2 | 50.3 | 28.9 |

### Notes

- DPBR gives net positive mean improvement for this model (+2.46%, 3/5 wins).
- Species_Humans remains the biggest residual error in most countries.

---

## Run: EXP-24-QWEN3_8B

- Date: 2026-04-10
- Script: `exp_model/exp_qwen3_8b.py`
- Model: `unsloth/Qwen3-8B-unsloth-bnb-4bit`
- Method: Dual-Pass Bootstrap IS Reliability (DPBR)
- Output dir: `/kaggle/working/cultural_alignment/results/exp24_model_sweep/qwen3_8b/compare`

### Full Metrics (DPBR)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Qwen3-8B-unsloth-bnb-4bit | BRA | 0.4447 | 0.0443 | -0.168 | 16.37 | 2.3% |
| Qwen3-8B-unsloth-bnb-4bit | CHN | 0.4912 | 0.0527 | -0.580 | 17.48 | 12.6% |
| Qwen3-8B-unsloth-bnb-4bit | DEU | 0.5406 | 0.0630 | -0.449 | 18.89 | 5.8% |
| Qwen3-8B-unsloth-bnb-4bit | JPN | 0.4529 | 0.0496 | -0.410 | 16.12 | 5.5% |
| Qwen3-8B-unsloth-bnb-4bit | USA | 0.5490 | 0.0550 | -0.595 | 20.05 | 9.0% |

### vs Vanilla (MIS)

| Model | Country | Vanilla MIS | DPBR MIS | Delta (ref-cur) | Improv% | Win? |
|:------|:-------:|------------:|---------:|----------------:|--------:|:----:|
| Qwen3-8B-unsloth-bnb-4bit | BRA | 0.4165 | 0.4447 | -0.0282 | -6.78% | ❌ |
| Qwen3-8B-unsloth-bnb-4bit | CHN | 0.4687 | 0.4912 | -0.0224 | -4.78% | ❌ |
| Qwen3-8B-unsloth-bnb-4bit | DEU | 0.5047 | 0.5406 | -0.0360 | -7.13% | ❌ |
| Qwen3-8B-unsloth-bnb-4bit | JPN | 0.4419 | 0.4529 | -0.0110 | -2.49% | ❌ |
| Qwen3-8B-unsloth-bnb-4bit | USA | 0.5094 | 0.5490 | -0.0395 | -7.76% | ❌ |

- Win rate: **0/5**
- Mean vanilla MIS: **0.4682**
- Mean method MIS: **0.4957**
- Macro improvement: **-5.86%**
- Mean per-row improvement (micro): **-5.79%**

### Per-Dimension Worst Error

| Model | Country | Worst Dimension | Human | Model | \|err\| (pp) |
|:------|:-------:|:----------------|:-----:|:-----:|:------------:|
| Qwen3-8B-unsloth-bnb-4bit | BRA | Utilitarianism_More | 73.7 | 48.2 | 25.5 |
| Qwen3-8B-unsloth-bnb-4bit | CHN | Species_Humans | 83.0 | 48.2 | 34.8 |
| Qwen3-8B-unsloth-bnb-4bit | DEU | Species_Humans | 82.4 | 45.9 | 36.5 |
| Qwen3-8B-unsloth-bnb-4bit | JPN | Species_Humans | 79.8 | 48.1 | 31.7 |
| Qwen3-8B-unsloth-bnb-4bit | USA | Species_Humans | 79.2 | 46.5 | 32.7 |

### Notes

- DPBR underperforms vanilla for all countries (0/5 wins).
- Dominant failure mode is Species_Humans in 4/5 countries.

---

## Run: EXP-24-LLAMA32_1B

- Date: 2026-04-10
- Script: `exp_model/exp_llama32_1b.py`
- Model: `unsloth/Llama-3.2-1B-Instruct`
- Method: Dual-Pass Bootstrap IS Reliability (DPBR)
- Output dir: `/kaggle/working/cultural_alignment/results/exp24_model_sweep/llama32_1b/compare`

### Full Metrics (DPBR)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Llama-3.2-1B | BRA | 0.4183 | 0.0492 | -0.304 | 15.37 | 0.6% |
| Llama-3.2-1B | CHN | 0.4911 | 0.0664 | -0.943 | 17.27 | 1.0% |
| Llama-3.2-1B | DEU | 0.5040 | 0.0631 | -0.629 | 17.96 | 1.9% |
| Llama-3.2-1B | JPN | 0.4547 | 0.0570 | -0.366 | 16.37 | 0.0% |
| Llama-3.2-1B | USA | 0.5122 | 0.0576 | -0.546 | 18.83 | 0.3% |

### vs Vanilla (MIS)

| Model | Country | Vanilla MIS | DPBR MIS | Delta (ref-cur) | Improv% | Win? |
|:------|:-------:|------------:|---------:|----------------:|--------:|:----:|
| Llama-3.2-1B | BRA | 0.4158 | 0.4183 | -0.0025 | -0.61% | ❌ |
| Llama-3.2-1B | CHN | 0.5363 | 0.4911 | +0.0452 | +8.43% | ✅ |
| Llama-3.2-1B | DEU | 0.4980 | 0.5040 | -0.0060 | -1.21% | ❌ |
| Llama-3.2-1B | JPN | 0.4544 | 0.4547 | -0.0003 | -0.06% | ❌ |
| Llama-3.2-1B | USA | 0.5034 | 0.5122 | -0.0088 | -1.76% | ❌ |

- Win rate: **1/5**
- Mean vanilla MIS: **0.4816**
- Mean method MIS: **0.4761**
- Macro improvement: **+1.14%**
- Mean per-row improvement (micro): **+0.96%**

### Per-Dimension Worst Error

| Model | Country | Worst Dimension | Human | Model | \|err\| (pp) |
|:------|:-------:|:----------------|:-----:|:-----:|:------------:|
| Llama-3.2-1B | BRA | Age_Young | 73.6 | 49.9 | 23.7 |
| Llama-3.2-1B | CHN | Species_Humans | 83.0 | 48.1 | 34.9 |
| Llama-3.2-1B | DEU | Species_Humans | 82.4 | 50.0 | 32.5 |
| Llama-3.2-1B | JPN | Species_Humans | 79.8 | 48.9 | 30.9 |
| Llama-3.2-1B | USA | Species_Humans | 79.2 | 50.4 | 28.8 |

### Notes

- Similar behavior to EXP-09 on this model; only CHN is a clear win vs vanilla.
- Species_Humans remains the dominant failure mode in 4/5 countries.

---

## Run: EXP-09-LLAMA32_1B

- Date: 2026-04-10
- Script: `exp_model/exp9/exp_llama32_1b.py`
- Model: `unsloth/Llama-3.2-1B-Instruct`
- Method: Hierarchical IS with Country-Level Prior
- Output dir: `/kaggle/working/cultural_alignment/results/exp09_model_sweep/llama32_1b/compare`

### Full Metrics (Hierarchical IS)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Llama-3.2-1B | BRA | 0.4183 | 0.0491 | -0.325 | 15.37 | 0.0% |
| Llama-3.2-1B | CHN | 0.4905 | 0.0662 | -0.941 | 17.25 | 1.3% |
| Llama-3.2-1B | DEU | 0.5037 | 0.0631 | -0.656 | 17.95 | 2.9% |
| Llama-3.2-1B | JPN | 0.4557 | 0.0571 | -0.395 | 16.40 | 0.0% |
| Llama-3.2-1B | USA | 0.5128 | 0.0577 | -0.534 | 18.86 | 0.0% |

### vs Vanilla (MIS)

| Model | Country | Vanilla MIS | Hierarchical IS MIS | Delta (ref-cur) | Improv% | Win? |
|:------|:-------:|------------:|--------------------:|----------------:|--------:|:----:|
| Llama-3.2-1B | BRA | 0.4158 | 0.4183 | -0.0024 | -0.59% | ❌ |
| Llama-3.2-1B | CHN | 0.5363 | 0.4905 | +0.0458 | +8.54% | ✅ |
| Llama-3.2-1B | DEU | 0.4980 | 0.5037 | -0.0058 | -1.16% | ❌ |
| Llama-3.2-1B | JPN | 0.4544 | 0.4557 | -0.0013 | -0.28% | ❌ |
| Llama-3.2-1B | USA | 0.5034 | 0.5128 | -0.0095 | -1.88% | ❌ |

- Win rate: **1/5**
- Mean vanilla MIS: **0.4816**
- Mean method MIS: **0.4762**
- Macro improvement: **+1.11%**
- Mean per-row improvement (micro): **+0.93%**

### Per-Dimension Worst Error

| Model | Country | Worst Dimension | Human | Model | \|err\| (pp) |
|:------|:-------:|:----------------|:-----:|:-----:|:------------:|
| Llama-3.2-1B | BRA | Age_Young | 73.6 | 49.8 | 23.8 |
| Llama-3.2-1B | CHN | Species_Humans | 83.0 | 48.1 | 34.8 |
| Llama-3.2-1B | DEU | Species_Humans | 82.4 | 49.9 | 32.5 |
| Llama-3.2-1B | JPN | Species_Humans | 79.8 | 48.7 | 31.1 |
| Llama-3.2-1B | USA | Species_Humans | 79.2 | 50.4 | 28.8 |

### Notes

- Strongest gain in CHN; remaining countries are near-tie or slight regression vs vanilla.
- Species_Humans is the dominant failure mode in 4/5 countries.

---

## Template for New Runs

Copy this block and append at the top when a new run completes.

```md
## Run: <RUN_ID>

- Date: YYYY-MM-DD
- Script: `exp_model/<...>.py`
- Model: `<model_name>`
- Method: <method_name>
- Output dir: `<kaggle_or_local_path>`

### Full Metrics (<method_name>)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| ... | ... | ... | ... | ... | ... | ... |

### vs Vanilla (MIS)

| Model | Country | Vanilla MIS | <method_name> MIS | Delta (ref-cur) | Improv% | Win? |
|:------|:-------:|------------:|------------------:|----------------:|--------:|:----:|
| ... | ... | ... | ... | ... | ... | ... |

### Per-Dimension Worst Error

| Model | Country | Worst Dimension | Human | Model | \|err\| (pp) |
|:------|:-------:|:----------------|:-----:|:-----:|:------------:|
| ... | ... | ... | ... | ... | ... |
```
