# Exp Model Performance Tracker

> Purpose: track method/model performance across countries to quickly see which model-method combo is best.
> Primary metric: `MIS` (lower is better). Secondary: `JSD` (lower), `Pearson r` (higher), `MAE` (lower), `Flip%`.

---

## Leaderboard by Method Run

Sorted by **macro Improv% vs vanilla** (higher first). MIS column is still the primary quality metric within each run.

| Rank | Run ID | Method | Model(s) | Coverage | Mean MIS ↓ | Mean JSD ↓ | Mean r ↑ | Mean MAE ↓ | Mean Flip% | Notes |
|:----:|:-------|:-------|:---------|:--------:|-----------:|-----------:|---------:|-----------:|-----------:|:------|
| 1 | EXP-24-PHI_4 | Dual-Pass Bootstrap IS Reliability (DPBR) | Phi-4 | 1 model x 5 countries | **0.2462** | 0.0619 | +0.692 | 7.93 | 16.83% | Win vs vanilla: 4/5, macro +34.35% |
| 2 | EXP-24-QWEN3_VL_8B | Dual-Pass Bootstrap IS Reliability (DPBR) | Qwen3-VL-8B | 1 model x 5 countries | **0.3813** | 0.0871 | +0.446 | 11.81 | 15.15% | Win vs vanilla: 4/5, macro +23.74% |
| 3 | EXP-24-LLAMA31_8B | Dual-Pass Bootstrap IS Reliability (DPBR) | Llama-3.1-8B-Instruct-bnb-4bit | 1 model x 5 countries | 0.4550 | 0.0657 | +0.127 | 15.90 | 17.70% | Win vs vanilla: 5/5, macro +16.88% |
| 4 | EXP-24-GEMMA4_E2B | Dual-Pass Bootstrap IS Reliability (DPBR) | gemma-4-E2B-it | 1 model x 5 countries | **0.4260** | 0.0642 | +0.423 | 15.16 | 10.6% | Win vs vanilla: 4/5, macro +11.57%; ref_gemma4 stack (TF 5.5.0) |
| 5 | EXP-24-PHI35_MINI | Dual-Pass Bootstrap IS Reliability (DPBR) | Phi-3.5-mini-instruct | 1 model x 5 countries | 0.5447 | 0.1228 | -0.363 | 18.66 | 15.5% | Win vs vanilla: 4/5, macro +7.27%; ref_phi_llama32 (TF 4.56.2) |
| 6 | EXP-09-MISTRAL_V03 | Hierarchical IS | mistral-7b-instruct-v0.3-bnb-4bit | 1 model x 5 countries | **0.4284** | 0.0958 | -0.686 | 15.55 | 16.9% | Win vs vanilla: 5/5, macro +5.87%; PyPI Unsloth |
| 7 | EXP-24-QWEN35_08B | Dual-Pass Bootstrap IS Reliability (DPBR) | Qwen3.5-0.8B | 1 model x 5 countries | 0.4708 | 0.0485 | -0.504 | 16.93 | 6.50% | Win vs vanilla: 4/5, macro +3.22% |
| 8 | EXP-24-GEMMA_7B | Dual-Pass Bootstrap IS Reliability (DPBR) | gemma-7b-it-bnb-4bit | 1 model x 5 countries | 0.4315 | 0.0575 | -0.477 | 14.76 | 11.22% | Win vs vanilla: 3/5, macro +2.46% |
| 9 | EXP-24-GEMMA3_270M | Dual-Pass Bootstrap IS Reliability (DPBR) | gemma-3-270m-it | 1 model x 5 countries | 0.4620 | 0.0549 | +0.332 | 16.75 | 7.90% | Win vs vanilla: 4/5, macro +1.32% |
| 10 | EXP-24-LLAMA32_1B | Dual-Pass Bootstrap IS Reliability (DPBR) | Llama-3.2-1B | 1 model x 5 countries | 0.4761 | 0.0586 | -0.558 | 17.16 | 0.76% | Win vs vanilla: 1/5, macro +1.14% |
| 11 | EXP-09-LLAMA32_1B | Hierarchical IS | Llama-3.2-1B | 1 model x 5 countries | 0.4762 | 0.0586 | -0.570 | 17.17 | 0.84% | Win vs vanilla: 1/5, macro +1.11% |
| 12 | EXP-09-QWEN3_4B_THINKING_2507 | Hierarchical IS | Qwen3-4B-Thinking-2507-unsloth-bnb-4bit | 1 model x 5 countries | 0.4686 | 0.0559 | +0.132 | 17.19 | 17.70% | Win vs vanilla: 3/5, macro +0.53% |
| 13 | EXP-24-PHI3_MEDIUM | Dual-Pass Bootstrap IS Reliability (DPBR) | Phi-3-medium-4k-instruct | 1 model x 5 countries | 0.3989 | 0.0641 | +0.330 | 13.95 | 17.1% | Win vs vanilla: 1/5, macro -1.21%; ref_phi_llama32 (TF 4.56.2) |
| 14 | EXP-24-QWEN35_2B | Dual-Pass Bootstrap IS Reliability (DPBR) | Qwen3.5-2B | 1 model x 5 countries | 0.4531 | 0.0581 | -0.033 | 16.13 | 14.30% | Win vs vanilla: 1/5, macro -3.27% |
| 15 | EXP-24-QWEN2_7B | Dual-Pass Bootstrap IS Reliability (DPBR) | Qwen2-7B | 1 model x 5 countries | 0.3946 | 0.0523 | +0.194 | 13.60 | 16.03% | Win vs vanilla: 2/5, macro -3.53% |
| 16 | EXP-09-QWEN3_14B | Hierarchical IS | Qwen3-14B-unsloth-bnb-4bit | 1 model x 5 countries | 0.4646 | 0.0563 | -0.133 | 16.77 | 11.20% | Win vs vanilla: 3/5, macro -4.39% |
| 17 | EXP-24-QWEN3_8B | Dual-Pass Bootstrap IS Reliability (DPBR) | Qwen3-8B-unsloth-bnb-4bit | 1 model x 5 countries | 0.4957 | 0.0529 | -0.441 | 17.78 | 7.03% | Win vs vanilla: 0/5, macro -5.86% |
| 18 | EXP-09-QWEN35_4B | Hierarchical IS | Qwen3.5-4B | 1 model x 5 countries | 0.3982 | 0.0474 | +0.527 | 14.34 | 14.3% | Win vs vanilla: 0/5, macro -23.99%; ref_qwen35 (TF 5.2.0) |
| 19 | EXP-24-QWEN35_4B | Dual-Pass Bootstrap IS Reliability (DPBR) | Qwen3.5-4B | 1 model x 5 countries | 0.3983 | **0.0395** | +0.527 | 14.34 | 14.65% | Win vs vanilla: 0/5, macro -24.03% |
| 20 | EXP-24-QWEN3_CODER_30B | Dual-Pass Bootstrap IS Reliability (DPBR) | Qwen3-Coder-30B-A3B | 1 model x 5 countries | **0.3900** | 0.0752 | +0.503 | 12.86 | 16.13% | Win vs vanilla: 1/5, macro -24.41% |

---

## Run: EXP-09-QWEN35_4B

- Date: 2026-04-11
- Script: `exp_model/exp9/exp_qwen35_4b.py`
- Model: `unsloth/Qwen3.5-4B` (`MODEL_SHORT`: `qwen35_4b`)
- Profile: **ref_qwen35** — Unsloth + unsloth-zoo from git; `trl==0.22.2` (`--no-deps`) + `tokenizers`; `transformers==5.2.0`; `datasets` 3.4.x–4.4.x; align with Reference_Notebook_Model / Qwen3_5_(4B)_Vision notebook where noted.
- Method: Hierarchical IS with Country-Level Prior (EXP-09); same theory as other EXP-09 runs (`alpha_h`, `delta_opt` blend).
- Output dir: `/kaggle/working/cultural_alignment/results/exp09_model_sweep/qwen35_4b/compare`
- Environment (Kaggle): 1× **NVIDIA H100 80GB**; Unsloth **2026.4.4** (“Fast Qwen3_5 patching”); Transformers **5.2.0**; Torch **2.9.0+cu126**; bfloat16; ~**3.39 GB** VRAM; Qwen3 chat template with **`<|im_start|>`** / assistant **`<think>`** blocks in vanilla (token ids **A=32, B=33**). Pip may warn optional **xformers** / `bigframes` / `google-adk` deps.

### Full Metrics (Hierarchical IS)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Qwen3.5-4B | BRA | 0.3565 | 0.0401 | +0.565 | 13.21 | 16.5% |
| Qwen3.5-4B | CHN | 0.3702 | 0.0519 | +0.479 | 12.71 | 19.7% |
| Qwen3.5-4B | DEU | 0.4378 | 0.0594 | +0.320 | 15.13 | 15.8% |
| Qwen3.5-4B | JPN | 0.3906 | 0.0479 | +0.498 | 14.05 | 12.3% |
| Qwen3.5-4B | USA | 0.4359 | 0.0378 | +0.776 | 16.59 | 7.4% |

### vs Vanilla (MIS)

| Model | Country | Vanilla MIS | Hierarchical IS MIS | Delta (ref-cur) | Improv% | Win? |
|:------|:-------:|------------:|--------------------:|----------------:|--------:|:----:|
| Qwen3.5-4B | BRA | 0.3225 | 0.3565 | -0.0341 | -10.56% | ❌ |
| Qwen3.5-4B | CHN | 0.2345 | 0.3702 | -0.1357 | -57.87% | ❌ |
| Qwen3.5-4B | DEU | 0.3965 | 0.4378 | -0.0413 | -10.41% | ❌ |
| Qwen3.5-4B | JPN | 0.3300 | 0.3906 | -0.0606 | -18.35% | ❌ |
| Qwen3.5-4B | USA | 0.3222 | 0.4359 | -0.1136 | -35.26% | ❌ |

- Win rate: **0/5**
- Mean vanilla MIS: **0.3212**
- Mean Hierarchical IS MIS: **0.3982**
- Macro improvement: **-23.99%**
- Mean per-row improvement (micro): **-26.49%**

### vs EXP-01 SWA-PTIS (MIS)

| Model | Country | EXP-01 MIS | Hierarchical IS MIS | Delta (ref-cur) | Improv% | Win? |
|:------|:-------:|-----------:|--------------------:|----------------:|--------:|:----:|
| Qwen3.5-4B | BRA | 0.4025 | 0.3565 | +0.0460 | +11.42% | ✅ |
| Qwen3.5-4B | CHN | 0.4078 | 0.3702 | +0.0376 | +9.22% | ✅ |
| Qwen3.5-4B | DEU | 0.3424 | 0.4378 | -0.0954 | -27.87% | ❌ |
| Qwen3.5-4B | JPN | 0.2802 | 0.3906 | -0.1104 | -39.41% | ❌ |
| Qwen3.5-4B | USA | 0.3677 | 0.4359 | -0.0682 | -18.54% | ❌ |

- Win rate vs EXP-01: **2/5**
- Mean EXP-01 MIS: **0.3601**
- Mean Hierarchical IS MIS: **0.3982**
- Macro improvement vs EXP-01: **-10.57%**

### Per-Dimension Worst Error

| Model | Country | Worst Dimension | Human | Model | \|err\| (pp) |
|:------|:-------:|:----------------|:-----:|:-----:|:------------:|
| Qwen3.5-4B | BRA | Age_Young | 73.6 | 51.0 | 22.6 |
| Qwen3.5-4B | CHN | Species_Humans | 83.0 | 55.9 | 27.1 |
| Qwen3.5-4B | DEU | Species_Humans | 82.4 | 52.6 | 29.9 |
| Qwen3.5-4B | JPN | Species_Humans | 79.8 | 53.7 | 26.1 |
| Qwen3.5-4B | USA | Age_Young | 74.5 | 50.2 | 24.4 |

### Notes

- Same checkpoint as **EXP-24-QWEN35_4B** (DPBR) but here **Hierarchical IS**; vanilla MIS is very low (~**0.32** mean), so IS **hurts** vs vanilla (**0/5**).
- vs **EXP-01** (same **0.3601** slice as other EXP-09 Qwen rows): **2/5** (**BRA**, **CHN**).
- **Species_Humans** worst dim in **4/5** countries (**BRA** / **USA**: **Age_Young**).
- Aggregate: mean JSD **0.0474**, mean Pearson **+0.527**, mean flip **14.3%**, ~**640 ms** mean latency (vs EXP-09 SOTA MIS 0.3975).

---

## Run: EXP-09-MISTRAL_V03

- Date: 2026-04-11
- Script: `exp_model/exp9/exp_mistral_v03.py`
- Model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit` (`MODEL_SHORT`: `mistral_v03`)
- Profile: **pypi** — `pip install --upgrade --no-deps unsloth`, `unsloth_zoo`, `datasets` 3.4.x–4.4.x; align with Reference_Notebook_Model where noted.
- Method: Hierarchical IS with Country-Level Prior (EXP-09); theory lines from log: `alpha_h(t)=0` for `t<50`, then `1-exp(-(t-50)/100)`; `delta_opt = alpha_h*delta_country + (1-alpha_h)*delta_opt_micro`.
- Output dir: `/kaggle/working/cultural_alignment/results/exp09_model_sweep/mistral_v03/compare`
- Environment (Kaggle): 1× **NVIDIA H100 80GB**; Unsloth **2026.4.4** (“Fast Mistral patching”); Transformers **5.2.0**; Torch **2.9.0+cu126**; bfloat16; ~**4.21 GB** VRAM; Mistral **`<s>[INST]` … `[/INST]`** chat template; decision token ids **A=29509, B=29528**. Pip may warn missing optional **xformers** / `bigframes` / `google-adk` vs `google-cloud-bigquery-storage` (non-fatal).

### Full Metrics (Hierarchical IS)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| mistral-7b-instruct-v0.3-bnb-4bit | BRA | 0.4135 | 0.0813 | -0.704 | 13.31 | 17.1% |
| mistral-7b-instruct-v0.3-bnb-4bit | CHN | 0.4113 | 0.1008 | -0.619 | 16.30 | 18.1% |
| mistral-7b-instruct-v0.3-bnb-4bit | DEU | 0.4628 | 0.1004 | -0.884 | 16.05 | 18.1% |
| mistral-7b-instruct-v0.3-bnb-4bit | JPN | 0.3278 | 0.0726 | -0.652 | 12.32 | 16.1% |
| mistral-7b-instruct-v0.3-bnb-4bit | USA | 0.5267 | 0.1238 | -0.571 | 19.76 | 15.2% |

### vs Vanilla (MIS)

| Model | Country | Vanilla MIS | Hierarchical IS MIS | Delta (ref-cur) | Improv% | Win? |
|:------|:-------:|------------:|--------------------:|----------------:|--------:|:----:|
| mistral-7b-instruct-v0.3-bnb-4bit | BRA | 0.4144 | 0.4135 | +0.0009 | +0.22% | ✅ |
| mistral-7b-instruct-v0.3-bnb-4bit | CHN | 0.4569 | 0.4113 | +0.0457 | +9.99% | ✅ |
| mistral-7b-instruct-v0.3-bnb-4bit | DEU | 0.4909 | 0.4628 | +0.0280 | +5.71% | ✅ |
| mistral-7b-instruct-v0.3-bnb-4bit | JPN | 0.3429 | 0.3278 | +0.0151 | +4.39% | ✅ |
| mistral-7b-instruct-v0.3-bnb-4bit | USA | 0.5706 | 0.5267 | +0.0439 | +7.70% | ✅ |

- Win rate: **5/5**
- Mean vanilla MIS: **0.4551**
- Mean Hierarchical IS MIS: **0.4284**
- Macro improvement: **+5.87%**
- Mean per-row improvement (micro): **+5.60%**

### vs EXP-01 SWA-PTIS (MIS)

| Model | Country | EXP-01 MIS | Hierarchical IS MIS | Delta (ref-cur) | Improv% | Win? |
|:------|:-------:|-----------:|--------------------:|----------------:|--------:|:----:|
| mistral-7b-instruct-v0.3-bnb-4bit | BRA | 0.4362 | 0.4135 | +0.0227 | +5.20% | ✅ |
| mistral-7b-instruct-v0.3-bnb-4bit | CHN | 0.5067 | 0.4113 | +0.0954 | +18.83% | ✅ |
| mistral-7b-instruct-v0.3-bnb-4bit | DEU | 0.4942 | 0.4628 | +0.0314 | +6.34% | ✅ |
| mistral-7b-instruct-v0.3-bnb-4bit | JPN | 0.3502 | 0.3278 | +0.0224 | +6.39% | ✅ |
| mistral-7b-instruct-v0.3-bnb-4bit | USA | 0.5984 | 0.5267 | +0.0717 | +11.99% | ✅ |

- Win rate vs EXP-01: **5/5**
- Mean EXP-01 MIS: **0.4771**
- Mean Hierarchical IS MIS: **0.4284**
- Macro improvement vs EXP-01: **+10.21%**

### Per-Dimension Worst Error

| Model | Country | Worst Dimension | Human | Model | \|err\| (pp) |
|:------|:-------:|:----------------|:-----:|:-----:|:------------:|
| mistral-7b-instruct-v0.3-bnb-4bit | BRA | Age_Young | 73.6 | 43.6 | 30.0 |
| mistral-7b-instruct-v0.3-bnb-4bit | CHN | Species_Humans | 83.0 | 58.2 | 24.8 |
| mistral-7b-instruct-v0.3-bnb-4bit | DEU | Species_Humans | 82.4 | 51.8 | 30.7 |
| mistral-7b-instruct-v0.3-bnb-4bit | JPN | Species_Humans | 79.8 | 58.1 | 21.7 |
| mistral-7b-instruct-v0.3-bnb-4bit | USA | Utilitarianism_More | 76.6 | 45.6 | 31.0 |

### Notes

- **Hierarchical IS vs vanilla**: wins in **all five** countries; largest gains **CHN** / **USA**.
- vs **EXP-01**: **5/5**; macro **+10.21%** (mean EXP-01 **0.4771** vs Hierarchical IS **0.4284**; per-country EXP-01 from run compare table).
- **Species_Humans** worst dim in **4/5** countries (**BRA** is **Age_Young**).
- Aggregate: mean JSD **0.0958**, mean Pearson **-0.686**, mean flip **16.9%** (vs EXP-09 SOTA MIS 0.3975).

---

## Run: EXP-24-PHI35_MINI

- Date: 2026-04-11
- Script: `exp_model/exp_24/exp_phi35_mini.py`
- Model: `unsloth/Phi-3.5-mini-instruct` (`MODEL_SHORT`: `phi35_mini`)
- Profile: **ref_phi_llama32** — same pip pattern as **EXP-24-PHI3_MEDIUM** (`transformers==4.56.2`, `trl==0.22.2` `--no-deps`, `datasets==4.3.0`, Unsloth **2026.4.4** + unsloth_zoo **2026.4.6**); align with Reference_Notebook_Model / Phi_3_5_Mini_Conversational notebook where noted.
- Method: Dual-Pass Bootstrap IS Reliability (DPBR)
- Output dir: `/kaggle/working/cultural_alignment/results/exp24_model_sweep/phi35_mini/compare`
- Environment (Kaggle): 1× **NVIDIA H100 80GB**; Unsloth **2026.4.4** (“Fast Llama patching”); Transformers **4.56.2**; Torch **2.9.0+cu126**; bfloat16; ~**2.28 GB** VRAM; decision token ids **A=319, B=350**. Vanilla prompts use **`<|system|>`** system line + user block (`Phi-3.5-mini` template); DPBR strips to user-style body as in log. Pip / resolver warnings same family as Phi-3-medium run (`bigframes`, `unsloth` optional deps, `trl` pin, etc.). XLA duplicate-registration messages on startup (non-fatal).

### Full Metrics (DPBR)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Phi-3.5-mini-instruct | BRA | 0.6199 | 0.1573 | -0.760 | 21.89 | 18.4% |
| Phi-3.5-mini-instruct | CHN | 0.5202 | 0.1456 | -0.073 | 17.45 | 19.4% |
| Phi-3.5-mini-instruct | DEU | 0.5561 | 0.1458 | -0.413 | 18.90 | 19.4% |
| Phi-3.5-mini-instruct | JPN | 0.4473 | 0.0602 | -0.246 | 15.85 | 2.9% |
| Phi-3.5-mini-instruct | USA | 0.5802 | 0.1050 | -0.323 | 19.19 | 17.7% |

### vs Vanilla (MIS)

| Model | Country | Vanilla MIS | DPBR MIS | Delta (ref-cur) | Improv% | Win? |
|:------|:-------:|------------:|---------:|----------------:|--------:|:----:|
| Phi-3.5-mini-instruct | BRA | 0.5313 | 0.6199 | -0.0886 | -16.68% | ❌ |
| Phi-3.5-mini-instruct | CHN | 0.7381 | 0.5202 | +0.2178 | +29.52% | ✅ |
| Phi-3.5-mini-instruct | DEU | 0.5692 | 0.5561 | +0.0132 | +2.31% | ✅ |
| Phi-3.5-mini-instruct | JPN | 0.4540 | 0.4473 | +0.0067 | +1.47% | ✅ |
| Phi-3.5-mini-instruct | USA | 0.6448 | 0.5802 | +0.0646 | +10.02% | ✅ |

- Win rate: **4/5**
- Mean vanilla MIS: **0.5875**
- Mean DPBR MIS: **0.5447**
- Macro improvement: **+7.27%**
- Mean per-row improvement (micro): **+5.33%**

### vs EXP-01 SWA-PTIS (MIS)

| Model | Country | EXP-01 MIS | DPBR MIS | Delta (ref-cur) | Improv% | Win? |
|:------|:-------:|-----------:|---------:|----------------:|--------:|:----:|
| Phi-3.5-mini-instruct | BRA | 0.3655 | 0.6199 | -0.2544 | -69.63% | ❌ |
| Phi-3.5-mini-instruct | CHN | 0.4536 | 0.5202 | -0.0666 | -14.68% | ❌ |
| Phi-3.5-mini-instruct | DEU | 0.3289 | 0.5561 | -0.2272 | -69.08% | ❌ |
| Phi-3.5-mini-instruct | JPN | 0.4667 | 0.4473 | +0.0194 | +4.16% | ✅ |
| Phi-3.5-mini-instruct | USA | 0.6038 | 0.5802 | +0.0236 | +3.91% | ✅ |

- Win rate vs EXP-01: **2/5**
- Mean EXP-01 MIS: **0.4437**
- Mean DPBR MIS: **0.5447**
- Macro improvement vs EXP-01: **-22.76%**

### Per-Dimension Worst Error

| Model | Country | Worst Dimension | Human | Model | \|err\| (pp) |
|:------|:-------:|:----------------|:-----:|:-----:|:------------:|
| Phi-3.5-mini-instruct | BRA | Utilitarianism_More | 73.7 | 27.8 | 45.9 |
| Phi-3.5-mini-instruct | CHN | Utilitarianism_More | 71.1 | 33.2 | 38.0 |
| Phi-3.5-mini-instruct | DEU | Utilitarianism_More | 75.1 | 29.9 | 45.2 |
| Phi-3.5-mini-instruct | JPN | Species_Humans | 79.8 | 51.2 | 28.6 |
| Phi-3.5-mini-instruct | USA | Utilitarianism_More | 76.6 | 36.2 | 40.3 |

### Notes

- Strong **DPBR vs vanilla** (**4/5**; only **BRA** worse MIS). **CHN** / **USA** large gains vs very poor vanilla MIS.
- vs **EXP-01**: **2/5** (**JPN**, **USA**); mean DPBR **above** mean EXP-01 on this table (macro **-22.76%**).
- **Utilitarianism_More** worst dim in **4/5** countries (**JPN** is **Species_Humans**).
- Aggregate: DPBR mean MIS **0.5447**, mean JSD **0.1228**, mean Pearson **-0.363**, mean flip **15.5%**, mean `rel_r` **0.926** (vs EXP-09 SOTA MIS 0.3975; EXP-24 multi-model ref 0.3969).

---

## Run: EXP-24-PHI3_MEDIUM

- Date: 2026-04-11
- Script: `exp_model/exp_24/exp_phi3_medium.py`
- Model: `unsloth/Phi-3-medium-4k-instruct` (`MODEL_SHORT`: `phi3_medium`)
- Profile: **ref_phi_llama32** — `transformers==4.56.2`, `trl==0.22.2` (`--no-deps`), `datasets==4.3.0`, Unsloth **2026.4.4** + unsloth_zoo **2026.4.6** (PyPI `--no-deps` install); align with Reference_Notebook_Model / Phi_3_Medium_Conversational notebook where noted.
- Method: Dual-Pass Bootstrap IS Reliability (DPBR)
- Output dir: `/kaggle/working/cultural_alignment/results/exp24_model_sweep/phi3_medium/compare`
- Environment (Kaggle): 1× **NVIDIA H100 80GB**; Unsloth **2026.4.4** (“Fast Mistral patching”); Transformers **4.56.2**; Torch **2.9.0+cu126**; bfloat16; ~**7.69 GB** VRAM; decision token ids **A=319, B=350**; **Mean variance 0** across countries (bootstrap variance collapsed for this run). Pip resolver warnings: `bigframes` / `google-adk` vs `google-cloud-bigquery-storage`; post-install conflicts noted for `unsloth` / `unsloth-zoo` (tyro, xformers, `trl` vs pinned `trl==0.22.2`, `torchao`, etc.) — run still completed. XLA/cuDNN duplicate-registration messages on startup (non-fatal).

### Full Metrics (DPBR)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Phi-3-medium-4k-instruct | BRA | 0.3898 | 0.0645 | +0.179 | 13.23 | 18.4% |
| Phi-3-medium-4k-instruct | CHN | 0.3044 | 0.0650 | +0.346 | 10.94 | 16.8% |
| Phi-3-medium-4k-instruct | DEU | 0.4499 | 0.0644 | +0.107 | 15.38 | 17.1% |
| Phi-3-medium-4k-instruct | JPN | 0.3132 | 0.0398 | +0.702 | 11.17 | 15.8% |
| Phi-3-medium-4k-instruct | USA | 0.5371 | 0.0867 | +0.319 | 19.02 | 17.4% |

### vs Vanilla (MIS)

| Model | Country | Vanilla MIS | DPBR MIS | Delta (ref-cur) | Improv% | Win? |
|:------|:-------:|------------:|---------:|----------------:|--------:|:----:|
| Phi-3-medium-4k-instruct | BRA | 0.3739 | 0.3898 | -0.0158 | -4.23% | ❌ |
| Phi-3-medium-4k-instruct | CHN | 0.2890 | 0.3044 | -0.0154 | -5.33% | ❌ |
| Phi-3-medium-4k-instruct | DEU | 0.4192 | 0.4499 | -0.0307 | -7.31% | ❌ |
| Phi-3-medium-4k-instruct | JPN | 0.3517 | 0.3132 | +0.0385 | +10.95% | ✅ |
| Phi-3-medium-4k-instruct | USA | 0.5366 | 0.5371 | -0.0005 | -0.09% | ❌ |

- Win rate: **1/5**
- Mean vanilla MIS: **0.3941**
- Mean DPBR MIS: **0.3989**
- Macro improvement: **-1.21%**
- Mean per-row improvement (micro): **-1.21%**

### vs EXP-01 SWA-PTIS (MIS)

| Model | Country | EXP-01 MIS | DPBR MIS | Delta (ref-cur) | Improv% | Win? |
|:------|:-------:|-----------:|---------:|----------------:|--------:|:----:|
| Phi-3-medium-4k-instruct | BRA | 0.3655 | 0.3898 | -0.0243 | -6.65% | ❌ |
| Phi-3-medium-4k-instruct | CHN | 0.4536 | 0.3044 | +0.1492 | +32.89% | ✅ |
| Phi-3-medium-4k-instruct | DEU | 0.3289 | 0.4499 | -0.1210 | -36.79% | ❌ |
| Phi-3-medium-4k-instruct | JPN | 0.4667 | 0.3132 | +0.1535 | +32.89% | ✅ |
| Phi-3-medium-4k-instruct | USA | 0.6038 | 0.5371 | +0.0667 | +11.05% | ✅ |

- Win rate vs EXP-01: **3/5**
- Mean EXP-01 MIS: **0.4437**
- Mean DPBR MIS: **0.3989**
- Macro improvement vs EXP-01: **+10.10%**

### Per-Dimension Worst Error

| Model | Country | Worst Dimension | Human | Model | \|err\| (pp) |
|:------|:-------:|:----------------|:-----:|:-----:|:------------:|
| Phi-3-medium-4k-instruct | BRA | SocialValue_High | 66.3 | 41.8 | 24.6 |
| Phi-3-medium-4k-instruct | CHN | SocialValue_High | 66.7 | 47.8 | 18.9 |
| Phi-3-medium-4k-instruct | DEU | Utilitarianism_More | 75.1 | 48.1 | 27.0 |
| Phi-3-medium-4k-instruct | JPN | Age_Young | 70.0 | 51.6 | 18.4 |
| Phi-3-medium-4k-instruct | USA | SocialValue_High | 67.9 | 35.3 | 32.7 |

### Notes

- **DPBR vs vanilla**: only **JPN** improves MIS (**1/5**); aggregate slightly worse than vanilla on macro/micro (**-1.21%**).
- vs **EXP-01**: **3/5** wins (**CHN**, **JPN**, **USA**); **DEU** especially negative vs EXP-01 baseline on this table.
- **SocialValue_High** worst dim in **3/5** countries (**BRA**, **CHN**, **USA**).
- Aggregate: DPBR mean MIS **0.3989**, mean JSD **0.0641**, mean Pearson **+0.330**, mean flip **17.1%**, mean `rel_r` **0.995** (vs EXP-09 SOTA MIS 0.3975; EXP-24 multi-model ref 0.3969).

---

## Run: EXP-24-GEMMA4_E2B

- Date: 2026-04-11
- Script: `exp_model/exp_24/exp_gemma4_e2b.py`
- Model: `unsloth/gemma-4-E2B-it` (`MODEL_SHORT`: `gemma4_e2b`)
- Profile: **ref_gemma4** — `transformers==5.5.0`, Unsloth + unsloth-zoo from git; `datasets` 4.3.x; align with Reference_Notebook_Model / Gemma4 vision notebook where noted.
- Method: Dual-Pass Bootstrap IS Reliability (DPBR)
- Output dir: `/kaggle/working/cultural_alignment/results/exp24_model_sweep/gemma4_e2b/compare`
- Environment (Kaggle): 1× **NVIDIA H100 80GB**; Unsloth **2026.4.4**; Transformers **5.5.0**; Torch **2.9.0+cu126**; bfloat16; ~**7.47 GB** VRAM; Flash Attention 2 reported broken → **Xformers** fallback (no perf claim). Pip noted optional conflicts with `bigframes` / `google-adk` vs `google-cloud-bigquery-storage` (non-fatal for this run).

### Full Metrics (DPBR)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| gemma-4-E2B-it | BRA | 0.4026 | 0.0810 | +0.273 | 13.73 | 13.2% |
| gemma-4-E2B-it | CHN | 0.3756 | 0.0449 | +0.644 | 13.57 | 4.8% |
| gemma-4-E2B-it | DEU | 0.3779 | 0.0518 | +0.684 | 13.51 | 17.1% |
| gemma-4-E2B-it | JPN | 0.4458 | 0.0580 | -0.085 | 15.90 | 2.6% |
| gemma-4-E2B-it | USA | 0.5282 | 0.0852 | +0.598 | 19.10 | 15.2% |

### vs Vanilla (MIS)

| Model | Country | Vanilla MIS | DPBR MIS | Delta (ref-cur) | Improv% | Win? |
|:------|:-------:|------------:|---------:|----------------:|--------:|:----:|
| gemma-4-E2B-it | BRA | 0.4160 | 0.4026 | +0.0133 | +3.21% | ✅ |
| gemma-4-E2B-it | CHN | 0.5000 | 0.3756 | +0.1244 | +24.88% | ✅ |
| gemma-4-E2B-it | DEU | 0.5058 | 0.3779 | +0.1279 | +25.29% | ✅ |
| gemma-4-E2B-it | JPN | 0.4419 | 0.4458 | -0.0039 | -0.89% | ❌ |
| gemma-4-E2B-it | USA | 0.5452 | 0.5282 | +0.0170 | +3.11% | ✅ |

- Win rate: **4/5**
- Mean vanilla MIS: **0.4818**
- Mean DPBR MIS: **0.4260**
- Macro improvement: **+11.57%**
- Mean per-row improvement (micro): **+11.12%**

### vs EXP-01 SWA-PTIS (MIS)

| Model | Country | EXP-01 MIS | DPBR MIS | Delta (ref-cur) | Improv% | Win? |
|:------|:-------:|-----------:|---------:|----------------:|--------:|:----:|
| gemma-4-E2B-it | BRA | 0.3655 | 0.4026 | -0.0371 | -10.16% | ❌ |
| gemma-4-E2B-it | CHN | 0.4536 | 0.3756 | +0.0780 | +17.20% | ✅ |
| gemma-4-E2B-it | DEU | 0.3289 | 0.3779 | -0.0490 | -14.91% | ❌ |
| gemma-4-E2B-it | JPN | 0.4667 | 0.4458 | +0.0209 | +4.47% | ✅ |
| gemma-4-E2B-it | USA | 0.6038 | 0.5282 | +0.0756 | +12.52% | ✅ |

- Win rate vs EXP-01: **3/5**
- Mean EXP-01 MIS: **0.4437**
- Mean DPBR MIS: **0.4260**
- Macro improvement vs EXP-01: **+3.98%**

### Per-Dimension Worst Error

| Model | Country | Worst Dimension | Human | Model | \|err\| (pp) |
|:------|:-------:|:----------------|:-----:|:-----:|:------------:|
| gemma-4-E2B-it | BRA | Age_Young | 73.6 | 42.1 | 31.5 |
| gemma-4-E2B-it | CHN | Species_Humans | 83.0 | 57.9 | 25.0 |
| gemma-4-E2B-it | DEU | Age_Young | 73.9 | 47.9 | 26.0 |
| gemma-4-E2B-it | JPN | Species_Humans | 79.8 | 48.6 | 31.2 |
| gemma-4-E2B-it | USA | Age_Young | 74.5 | 38.3 | 36.3 |

### Notes

- Strong **DPBR vs vanilla** (**4/5**; only **JPN** slightly worse MIS). **CHN** / **DEU** large gains vs vanilla.
- vs **EXP-01**: **3/5** wins (**CHN**, **JPN**, **USA**); **BRA** / **DEU** below EXP-01 on this table.
- **Age_Young** and **Species_Humans** dominate worst-dimension errors.
- Aggregate: DPBR mean MIS **0.4260**, mean JSD **0.0642**, mean Pearson **+0.423**, mean flip **10.6%**, mean `rel_r` **0.815** (vs EXP-09 SOTA MIS 0.3975; EXP-24 multi-model ref 0.3969).

---

## Run: EXP-24-GEMMA3_270M

- Date: 2026-04-11
- Script: `exp_model/exp_24/exp_gemma3_270m.py`
- Model: `unsloth/gemma-3-270m-it`
- Method: Dual-Pass Bootstrap IS Reliability (DPBR)
- Output dir: `/kaggle/working/cultural_alignment/results/exp24_model_sweep/gemma3_270m/compare`
- Environment (Kaggle): 1× **NVIDIA H100 80GB**; Unsloth **2026.4.4** (PyPI); Transformers **5.2.0**; bfloat16; ~**0.43 GB** VRAM; Gemma 3 chat template (`<start_of_turn>` / `model`); decision token ids **A=236776, B=236799** in this build.

### Full Metrics (DPBR)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| gemma-3-270m-it | BRA | 0.4063 | 0.0473 | +0.405 | 14.92 | 0.0% |
| gemma-3-270m-it | CHN | 0.4625 | 0.0573 | +0.267 | 16.53 | 6.5% |
| gemma-3-270m-it | DEU | 0.4960 | 0.0621 | +0.219 | 17.66 | 14.8% |
| gemma-3-270m-it | JPN | 0.4339 | 0.0524 | +0.745 | 15.69 | 0.6% |
| gemma-3-270m-it | USA | 0.5113 | 0.0551 | +0.026 | 18.93 | 17.4% |

### vs Vanilla (MIS)

| Model | Country | Vanilla MIS | DPBR MIS | Delta (ref-cur) | Improv% | Win? |
|:------|:-------:|------------:|---------:|----------------:|--------:|:----:|
| gemma-3-270m-it | BRA | 0.4165 | 0.4063 | +0.0102 | +2.45% | ✅ |
| gemma-3-270m-it | CHN | 0.4684 | 0.4625 | +0.0059 | +1.27% | ✅ |
| gemma-3-270m-it | DEU | 0.5047 | 0.4960 | +0.0087 | +1.71% | ✅ |
| gemma-3-270m-it | JPN | 0.4419 | 0.4339 | +0.0080 | +1.82% | ✅ |
| gemma-3-270m-it | USA | 0.5094 | 0.5113 | -0.0018 | -0.36% | ❌ |

- Win rate: **4/5**
- Mean vanilla MIS: **0.4682**
- Mean DPBR MIS: **0.4620**
- Macro improvement: **+1.32%**
- Mean per-row improvement (micro): **+1.38%**

### vs EXP-01 SWA-PTIS (MIS)

| Model | Country | EXP-01 MIS | DPBR MIS | Delta (ref-cur) | Improv% | Win? |
|:------|:-------:|-----------:|---------:|----------------:|--------:|:----:|
| gemma-3-270m-it | BRA | 0.3655 | 0.4063 | -0.0408 | -11.16% | ❌ |
| gemma-3-270m-it | CHN | 0.4536 | 0.4625 | -0.0089 | -1.95% | ❌ |
| gemma-3-270m-it | DEU | 0.3289 | 0.4960 | -0.1671 | -50.80% | ❌ |
| gemma-3-270m-it | JPN | 0.4667 | 0.4339 | +0.0328 | +7.03% | ✅ |
| gemma-3-270m-it | USA | 0.6038 | 0.5113 | +0.0925 | +15.32% | ✅ |

- Win rate vs EXP-01: **2/5**
- Mean EXP-01 MIS: **0.4437**
- Mean DPBR MIS: **0.4620**
- Macro improvement vs EXP-01: **-4.12%**

### Per-Dimension Worst Error

| Model | Country | Worst Dimension | Human | Model | \|err\| (pp) |
|:------|:-------:|:----------------|:-----:|:-----:|:------------:|
| gemma-3-270m-it | BRA | Utilitarianism_More | 73.7 | 50.2 | 23.5 |
| gemma-3-270m-it | CHN | Species_Humans | 83.0 | 50.5 | 32.5 |
| gemma-3-270m-it | DEU | Species_Humans | 82.4 | 50.2 | 32.2 |
| gemma-3-270m-it | JPN | Species_Humans | 79.8 | 50.7 | 29.1 |
| gemma-3-270m-it | USA | Species_Humans | 79.2 | 50.4 | 28.8 |

### Notes

- Strong **DPBR vs vanilla** (**4/5** wins; only **USA** ties/regresses slightly).
- Mixed vs **EXP-01** (**2/5**): large win **USA**, gain **JPN**; **DEU** especially negative vs EXP-01 baseline in this table.
- **Species_Humans** worst dim in **4/5** countries (**BRA** is **Utilitarianism_More**).
- Aggregate from run: DPBR mean MIS **0.4620**, mean JSD **0.0549**, mean Pearson **+0.332**, mean flip **7.9%**, mean `rel_r` **0.991** (vs EXP-09 SOTA MIS 0.3975; EXP-24 multi-model ref 0.3969).

---

## Run: EXP-09-QWEN3_4B_THINKING_2507

- Date: 2026-04-11
- Script: `exp_model/exp9/exp_qwen3_4b_thinking_2507.py`
- Model: `unsloth/Qwen3-4B-Thinking-2507-unsloth-bnb-4bit`
- Method: Hierarchical IS with Country-Level Prior (EXP-09)
- Output dir: `/kaggle/working/cultural_alignment/results/exp09_model_sweep/qwen3_4b_thinking_2507/compare`
- Environment (Kaggle): 1× **NVIDIA H100 80GB**; Unsloth **2026.4.4** (PyPI); Transformers **5.2.0**; bfloat16; ~**3.58 GB** VRAM; synthetic pad `<|PAD_TOKEN|>`; vanilla prompts expose **thinking** slots (`<redacted_thinking>` in logs); pip may warn optional **xformers** / BigQuery deps.

### Full Metrics (Hierarchical IS)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Qwen3-4B-Thinking-2507-unsloth-bnb-4bit | BRA | 0.4161 | 0.0482 | +0.100 | 15.32 | 16.1% |
| Qwen3-4B-Thinking-2507-unsloth-bnb-4bit | CHN | 0.4712 | 0.0590 | -0.091 | 16.83 | 26.8% |
| Qwen3-4B-Thinking-2507-unsloth-bnb-4bit | DEU | 0.5028 | 0.0623 | +0.342 | 17.96 | 26.1% |
| Qwen3-4B-Thinking-2507-unsloth-bnb-4bit | JPN | 0.4423 | 0.0548 | +0.126 | 15.92 | 10.0% |
| Qwen3-4B-Thinking-2507-unsloth-bnb-4bit | USA | 0.5107 | 0.0549 | +0.182 | 18.91 | 9.7% |

### vs Vanilla (MIS)

| Model | Country | Vanilla MIS | Hierarchical IS MIS | Delta (ref-cur) | Improv% | Win? |
|:------|:-------:|------------:|--------------------:|----------------:|--------:|:----:|
| Qwen3-4B-Thinking-2507-unsloth-bnb-4bit | BRA | 0.4168 | 0.4161 | +0.0007 | +0.17% | ✅ |
| Qwen3-4B-Thinking-2507-unsloth-bnb-4bit | CHN | 0.4705 | 0.4712 | -0.0008 | -0.16% | ❌ |
| Qwen3-4B-Thinking-2507-unsloth-bnb-4bit | DEU | 0.5138 | 0.5028 | +0.0110 | +2.13% | ✅ |
| Qwen3-4B-Thinking-2507-unsloth-bnb-4bit | JPN | 0.4454 | 0.4423 | +0.0031 | +0.70% | ✅ |
| Qwen3-4B-Thinking-2507-unsloth-bnb-4bit | USA | 0.5093 | 0.5107 | -0.0015 | -0.29% | ❌ |

- Win rate: **3/5**
- Mean vanilla MIS: **0.4712**
- Mean Hierarchical IS MIS: **0.4686**
- Macro improvement: **+0.53%**
- Mean per-row improvement (micro): **+0.51%**

### vs EXP-01 SWA-PTIS (MIS)

| Model | Country | EXP-01 MIS | Hierarchical IS MIS | Delta (ref-cur) | Improv% | Win? |
|:------|:-------:|-----------:|--------------------:|----------------:|--------:|:----:|
| Qwen3-4B-Thinking-2507-unsloth-bnb-4bit | BRA | 0.4025 | 0.4161 | -0.0136 | -3.39% | ❌ |
| Qwen3-4B-Thinking-2507-unsloth-bnb-4bit | CHN | 0.4078 | 0.4712 | -0.0634 | -15.55% | ❌ |
| Qwen3-4B-Thinking-2507-unsloth-bnb-4bit | DEU | 0.3424 | 0.5028 | -0.1604 | -46.85% | ❌ |
| Qwen3-4B-Thinking-2507-unsloth-bnb-4bit | JPN | 0.2802 | 0.4423 | -0.1621 | -57.84% | ❌ |
| Qwen3-4B-Thinking-2507-unsloth-bnb-4bit | USA | 0.3677 | 0.5107 | -0.1430 | -38.89% | ❌ |

- Win rate vs EXP-01: **0/5**
- Mean EXP-01 MIS: **0.3601**
- Mean Hierarchical IS MIS: **0.4686**
- Macro improvement vs EXP-01: **-30.13%**

### Per-Dimension Worst Error

| Model | Country | Worst Dimension | Human | Model | \|err\| (pp) |
|:------|:-------:|:----------------|:-----:|:-----:|:------------:|
| Qwen3-4B-Thinking-2507-unsloth-bnb-4bit | BRA | Utilitarianism_More | 73.7 | 49.8 | 23.9 |
| Qwen3-4B-Thinking-2507-unsloth-bnb-4bit | CHN | Species_Humans | 83.0 | 49.8 | 33.1 |
| Qwen3-4B-Thinking-2507-unsloth-bnb-4bit | DEU | Species_Humans | 82.4 | 50.1 | 32.4 |
| Qwen3-4B-Thinking-2507-unsloth-bnb-4bit | JPN | Species_Humans | 79.8 | 49.8 | 30.0 |
| Qwen3-4B-Thinking-2507-unsloth-bnb-4bit | USA | Species_Humans | 79.2 | 49.9 | 29.3 |

### Notes

- Net **small gain** vs vanilla on average (**+0.53%** macro): wins **BRA, DEU, JPN**; tiny regressions **CHN, USA**.
- Underperforms **EXP-01** on all five countries (0/5).
- **Species_Humans** is worst dim in 4/5 countries; **BRA** worst is **Utilitarianism_More**.
- High flip in **CHN/DEU** (~27%) vs **USA/JPN** (~10%); aggregate from report: mean JSD **0.0559**, mean Pearson **+0.132**, mean flip **17.7%**.

---

## Run: EXP-09-QWEN3_14B

- Date: 2026-04-11
- Script: `exp_model/exp9/exp_qwen3_14b.py`
- Model: `unsloth/Qwen3-14B-unsloth-bnb-4bit`
- Method: Hierarchical IS with Country-Level Prior (EXP-09)
- Output dir: `/kaggle/working/cultural_alignment/results/exp09_model_sweep/qwen3_14b/compare`
- Environment (Kaggle): 1× **NVIDIA H100 80GB**; Unsloth **2026.4.4** (PyPI); Transformers **5.2.0**; bfloat16; ~**11.19 GB** VRAM; model uses synthetic pad token `<|PAD_TOKEN|>` when none is set; pip may warn optional **xformers** / unrelated BigQuery deps.

### Full Metrics (Hierarchical IS)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Qwen3-14B-unsloth-bnb-4bit | BRA | 0.4129 | 0.0489 | -0.117 | 15.14 | 13.9% |
| Qwen3-14B-unsloth-bnb-4bit | CHN | 0.4681 | 0.0596 | -0.239 | 16.64 | 0.6% |
| Qwen3-14B-unsloth-bnb-4bit | DEU | 0.5001 | 0.0632 | -0.153 | 17.77 | 13.2% |
| Qwen3-14B-unsloth-bnb-4bit | JPN | 0.4390 | 0.0559 | -0.603 | 15.70 | 11.9% |
| Qwen3-14B-unsloth-bnb-4bit | USA | 0.5031 | 0.0542 | +0.447 | 18.61 | 16.5% |

### vs Vanilla (MIS)

| Model | Country | Vanilla MIS | Hierarchical IS MIS | Delta (ref-cur) | Improv% | Win? |
|:------|:-------:|------------:|--------------------:|----------------:|--------:|:----:|
| Qwen3-14B-unsloth-bnb-4bit | BRA | 0.4135 | 0.4129 | +0.0006 | +0.15% | ✅ |
| Qwen3-14B-unsloth-bnb-4bit | CHN | 0.4267 | 0.4681 | -0.0414 | -9.70% | ❌ |
| Qwen3-14B-unsloth-bnb-4bit | DEU | 0.5040 | 0.5001 | +0.0040 | +0.79% | ✅ |
| Qwen3-14B-unsloth-bnb-4bit | JPN | 0.3720 | 0.4390 | -0.0670 | -18.00% | ❌ |
| Qwen3-14B-unsloth-bnb-4bit | USA | 0.5093 | 0.5031 | +0.0062 | +1.21% | ✅ |

- Win rate: **3/5**
- Mean vanilla MIS: **0.4451**
- Mean Hierarchical IS MIS: **0.4646**
- Macro improvement: **-4.39%**
- Mean per-row improvement (micro): **-5.11%**

### vs EXP-01 SWA-PTIS (MIS)

| Model | Country | EXP-01 MIS | Hierarchical IS MIS | Delta (ref-cur) | Improv% | Win? |
|:------|:-------:|-----------:|--------------------:|----------------:|--------:|:----:|
| Qwen3-14B-unsloth-bnb-4bit | BRA | 0.4025 | 0.4129 | -0.0104 | -2.58% | ❌ |
| Qwen3-14B-unsloth-bnb-4bit | CHN | 0.4078 | 0.4681 | -0.0603 | -14.78% | ❌ |
| Qwen3-14B-unsloth-bnb-4bit | DEU | 0.3424 | 0.5001 | -0.1577 | -46.05% | ❌ |
| Qwen3-14B-unsloth-bnb-4bit | JPN | 0.2802 | 0.4390 | -0.1588 | -56.68% | ❌ |
| Qwen3-14B-unsloth-bnb-4bit | USA | 0.3677 | 0.5031 | -0.1354 | -36.83% | ❌ |

- Win rate vs EXP-01: **0/5**
- Mean EXP-01 MIS: **0.3601**
- Mean Hierarchical IS MIS: **0.4646**
- Macro improvement vs EXP-01: **-29.02%**

### Per-Dimension Worst Error

| Model | Country | Worst Dimension | Human | Model | \|err\| (pp) |
|:------|:-------:|:----------------|:-----:|:-----:|:------------:|
| Qwen3-14B-unsloth-bnb-4bit | BRA | Age_Young | 73.6 | 49.8 | 23.8 |
| Qwen3-14B-unsloth-bnb-4bit | CHN | Species_Humans | 83.0 | 49.7 | 33.3 |
| Qwen3-14B-unsloth-bnb-4bit | DEU | Species_Humans | 82.4 | 49.7 | 32.7 |
| Qwen3-14B-unsloth-bnb-4bit | JPN | Species_Humans | 79.8 | 49.8 | 30.0 |
| Qwen3-14B-unsloth-bnb-4bit | USA | Species_Humans | 79.2 | 50.0 | 29.2 |

### Notes

- Hierarchical IS is slightly worse than vanilla on average (**-4.39%** macro): clear regressions in **CHN** and **JPN**; small wins in **USA**, **DEU**, **BRA**.
- Underperforms **EXP-01** everywhere in this run (0/5).
- **Species_Humans** dominates worst-error in 4/5 countries (BRA is **Age_Young**).
- Aggregate from log summary: mean JSD **0.0563**, mean Pearson **−0.133**, mean flip **11.2%**.

---

## Run: EXP-24-LLAMA31_8B

- Date: 2026-04-11
- Script: `exp_model/exp_24/exp_llama31_8b.py`
- Model: `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`
- Method: Dual-Pass Bootstrap IS Reliability (DPBR)
- Output dir: `/kaggle/working/cultural_alignment/results/exp24_model_sweep/llama31_8b/compare`
- Environment (Kaggle): 2× Tesla T4; Unsloth **2026.4.4** (PyPI `pip install --upgrade --no-deps unsloth`); Transformers **5.2.0**; float16 path (no bfloat16 on T4); ~**5.75 GB** VRAM; pip may warn missing optional `xformers` / unrelated `google-cloud-bigquery-storage` conflicts.

### Full Metrics (DPBR)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Meta-Llama-3.1-8B-Instruct (4-bit) | BRA | 0.4010 | 0.0546 | +0.009 | 14.30 | 18.1% |
| Meta-Llama-3.1-8B-Instruct (4-bit) | CHN | 0.4490 | 0.0836 | +0.181 | 14.51 | 17.1% |
| Meta-Llama-3.1-8B-Instruct (4-bit) | DEU | 0.4741 | 0.0573 | +0.474 | 17.01 | 17.4% |
| Meta-Llama-3.1-8B-Instruct (4-bit) | JPN | 0.4321 | 0.0627 | -0.164 | 14.97 | 17.7% |
| Meta-Llama-3.1-8B-Instruct (4-bit) | USA | 0.5185 | 0.0707 | +0.132 | 18.71 | 18.4% |

### vs Vanilla (MIS)

| Model | Country | Vanilla MIS | DPBR MIS | Delta (ref-cur) | Improv% | Win? |
|:------|:-------:|------------:|---------:|----------------:|--------:|:----:|
| Meta-Llama-3.1-8B | BRA | 0.4343 | 0.4010 | +0.0333 | +7.66% | ✅ |
| Meta-Llama-3.1-8B | CHN | 0.6702 | 0.4490 | +0.2212 | +33.00% | ✅ |
| Meta-Llama-3.1-8B | DEU | 0.4945 | 0.4741 | +0.0203 | +4.11% | ✅ |
| Meta-Llama-3.1-8B | JPN | 0.4540 | 0.4321 | +0.0219 | +4.83% | ✅ |
| Meta-Llama-3.1-8B | USA | 0.6837 | 0.5185 | +0.1652 | +24.16% | ✅ |

- Win rate: **5/5**
- Mean vanilla MIS: **0.5473**
- Mean DPBR MIS: **0.4550**
- Macro improvement: **+16.88%**
- Mean per-row improvement (micro): **+14.75%**

### Per-Dimension Worst Error

| Model | Country | Worst Dimension | Human | Model | \|err\| (pp) |
|:------|:-------:|:----------------|:-----:|:-----:|:------------:|
| Meta-Llama-3.1-8B | BRA | Age_Young | 73.6 | 46.4 | 27.2 |
| Meta-Llama-3.1-8B | CHN | SocialValue_High | 66.7 | 36.1 | 30.6 |
| Meta-Llama-3.1-8B | DEU | Species_Humans | 82.4 | 53.8 | 28.6 |
| Meta-Llama-3.1-8B | JPN | Species_Humans | 79.8 | 51.0 | 28.8 |
| Meta-Llama-3.1-8B | USA | Age_Young | 74.5 | 43.1 | 31.5 |

### Notes

- DPBR improves over vanilla in **all five** countries; largest gains in **CHN** and **USA** (vanilla MIS was very high there).
- Aggregate from run: DPBR mean MIS **0.4550**, mean JSD **0.0657**, mean Pearson **+0.127**, mean flip **17.7%**, mean `rel_r` **0.976** (vs EXP-09 SOTA MIS 0.3975; EXP-24 multi-model ref 0.3969).
- Worst dimensions are **Age_Young** (USA, BRA) or **SocialValue_High** (CHN) or **Species_Humans** (DEU, JPN).

---

## Run: EXP-24-QWEN35_2B

- Date: 2026-04-11
- Script: `exp_model/exp_24/exp_qwen35_2b.py`
- Model: `unsloth/Qwen3.5-2B`
- Method: Dual-Pass Bootstrap IS Reliability (DPBR)
- Output dir: `/kaggle/working/cultural_alignment/results/exp24_model_sweep/qwen35_2b/compare`
- Environment (Kaggle): 2× Tesla T4; Unsloth 2026.4.4 (git); `transformers==5.2.0`, `trl==0.22.2`, `datasets` 4.3.x; model load uses float32 on this GPU path (Unsloth warning: float16 unsuitable for qwen3_5 on T4).

### Full Metrics (DPBR)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Qwen3.5-2B | BRA | 0.4245 | 0.0502 | -0.131 | 15.60 | 17.1% |
| Qwen3.5-2B | CHN | 0.4411 | 0.0662 | -0.165 | 14.90 | 17.1% |
| Qwen3.5-2B | DEU | 0.4780 | 0.0634 | -0.050 | 16.71 | 19.4% |
| Qwen3.5-2B | JPN | 0.4364 | 0.0563 | -0.001 | 15.59 | 6.8% |
| Qwen3.5-2B | USA | 0.4855 | 0.0544 | +0.182 | 17.83 | 11.0% |

### vs Vanilla (MIS)

| Model | Country | Vanilla MIS | DPBR MIS | Delta (ref-cur) | Improv% | Win? |
|:------|:-------:|------------:|---------:|----------------:|--------:|:----:|
| Qwen3.5-2B | BRA | 0.4155 | 0.4245 | -0.0089 | -2.15% | ❌ |
| Qwen3.5-2B | CHN | 0.4405 | 0.4411 | -0.0006 | -0.13% | ❌ |
| Qwen3.5-2B | DEU | 0.4404 | 0.4780 | -0.0376 | -8.54% | ❌ |
| Qwen3.5-2B | JPN | 0.4530 | 0.4364 | +0.0166 | +3.65% | ✅ |
| Qwen3.5-2B | USA | 0.4442 | 0.4855 | -0.0413 | -9.29% | ❌ |

- Win rate: **1/5**
- Mean vanilla MIS: **0.4387**
- Mean method MIS: **0.4531**
- Macro improvement: **-3.27%**
- Mean per-row improvement (micro): **-3.29%**

### vs EXP-01 SWA-PTIS (MIS)

| Model | Country | EXP-01 MIS | DPBR MIS | Delta (ref-cur) | Improv% | Win? |
|:------|:-------:|-----------:|---------:|----------------:|--------:|:----:|
| Qwen3.5-2B | BRA | 0.4025 | 0.4245 | -0.0220 | -5.46% | ❌ |
| Qwen3.5-2B | CHN | 0.4078 | 0.4411 | -0.0333 | -8.16% | ❌ |
| Qwen3.5-2B | DEU | 0.3424 | 0.4780 | -0.1356 | -39.60% | ❌ |
| Qwen3.5-2B | JPN | 0.2802 | 0.4364 | -0.1562 | -55.75% | ❌ |
| Qwen3.5-2B | USA | 0.3677 | 0.4855 | -0.1178 | -32.04% | ❌ |

- Win rate vs EXP-01: **0/5**
- Mean EXP-01 MIS: **0.3601**
- Mean method MIS: **0.4531**
- Macro improvement vs EXP-01: **-25.82%**

### Per-Dimension Worst Error

| Model | Country | Worst Dimension | Human | Model | \|err\| (pp) |
|:------|:-------:|:----------------|:-----:|:-----:|:------------:|
| Qwen3.5-2B | BRA | Age_Young | 73.6 | 49.8 | 23.8 |
| Qwen3.5-2B | CHN | Species_Humans | 83.0 | 51.2 | 31.8 |
| Qwen3.5-2B | DEU | Species_Humans | 82.4 | 50.2 | 32.3 |
| Qwen3.5-2B | JPN | Species_Humans | 79.8 | 51.4 | 28.4 |
| Qwen3.5-2B | USA | Species_Humans | 79.2 | 52.1 | 27.1 |

### Notes

- DPBR underperforms vanilla in 4/5 countries, with only JPN improving.
- DPBR underperforms EXP-01 across all countries in this run (0/5 vs EXP-01).
- Species_Humans is the worst dimension in 4/5 countries (BRA worst is Age_Young).
- Aggregate line from run: DPBR mean MIS **0.4531**, mean JSD **0.0581**, mean Pearson **−0.033**, mean flip **14.3%**, mean `rel_r` **0.992** (vs EXP-09 SOTA MIS 0.3975; EXP-24 multi-model ref 0.3969).

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
