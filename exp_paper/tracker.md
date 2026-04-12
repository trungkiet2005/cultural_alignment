# EXP paper — results tracker

---

#### 2026-04-13 — EXP-24 Ablation (Phi-4 14B, USA) — `exp_paper/exp_paper_ablation_phi4.py`

**Setup:** microsoft/phi-4 · USA · 310 scenarios (post quality-filter from 342) · K=64×2=128 · VAR_SCALE=0.04 · seed=42

| # | Configuration | JSD ↓ | Δ JSD | r ↑ | Δ r | MAE ↓ | MIS ↓ | flip% |
|:--|:-------------|:-----:|:------:|:---:|:---:|:-----:|:-----:|:-----:|
| — | **Full SWA-DPBR** | **0.0543** | — | **+0.378** | — | **18.93** | **0.5104** | 50.3% |
| 1 | No-IS (consensus only) | 0.0548 | +.001 | +0.247 | −.131 | 18.94 | 0.5112 | 17.4% |
| 2 | Always-on PT-IS | 0.0545 | +.000 | +0.325 | −.053 | 18.96 | 0.5114 | 49.4% |
| 3 | No debiasing | 0.0556 | +.001 | −0.526 | −.904 | 18.90 | 0.5114 | 0.0% |
| 4 | Without persona | 0.0541 | −.000 | +0.389 | +.011 | 18.89 | 0.5093 | 9.7% |
| 5 | No country prior | 0.0547 | +.000 | +0.140 | −.238 | 19.15 | 0.5157 | 43.2% |

**Internal diagnostics (Full):** pos_b μ=−4.076 · delta_consensus μ=0.003 · ESS=0.351 · rel_r=0.995 · α_h=0.928

**Key findings vs Qwen2.5-72B paper ablation:**
- Row 3 (No debiasing): Phi-4 Δr=**−0.904** vs Qwen Δr=+0.019 — Phi-4 has extreme raw A-position bias (~4 logit units); debiasing is the single most load-bearing component
- Row 4 (Without persona): Phi-4 Δr=**+0.011** vs Qwen Δr=−0.283 — personas are neutral/harmful for Phi-4 because debiased logits are near-flat (delta_consensus≈0.003), making WVS disagreement pure noise
- Row 5 (No prior): Phi-4 Δr=**−0.238** vs Qwen Δr=−0.010 — prior unexpectedly load-bearing for Phi-4; EMA is the only accumulator of signal when per-scenario IS corrections are degenerate
- All model MPR ≈ 50% (vs human 56–79%) — IS has no gradient to exploit after positional bias is removed; confirms "collapsed logit entropy" failure mode (§Discussion)

**Artifacts:** `/kaggle/working/cultural_alignment/results/exp24_ablation_phi4/`

> **Post-hoc note (2026-04-13) — Backend mismatch explains discrepancy vs main EXP-24-PHI_4**
>
> The ablation was run with **Unsloth 4-bit** (the `MORAL_MODEL_BACKEND` default), but
> the main EXP-24-PHI_4 sweep was run with `MORAL_MODEL_BACKEND=vllm` set in the Kaggle
> session (confirmed by the `(vLLM)` annotation in `exp_paper_phi_4.py` and by the fact
> that `src/model.py::load_model` routes to vLLM when that env var is set).  Main-run
> metrics: MIS=0.2433, r=+0.723, flip%=18.7%.  Ablation metrics: MIS=0.5104, r=+0.378, flip%=50.3%.
>
> Root cause: Unsloth INT4 quantisation degrades Phi-4's moral-reasoning logit resolution
> (`delta_consensus` μ≈0.003 after A↔B debiasing vs the vLLM regime where the model
> retains real content signal).  With near-zero delta_consensus the IS correction is also
> near-zero and flip direction is random → flip%≈50%, all MPR≈50%, MIS≈0.51.
>
> **Fixes applied**:
> - `exp_paper_ablation_phi4.py`: `os.environ.setdefault("MORAL_MODEL_BACKEND", "vllm")` before `configure_paper_env()`
> - `paper_runtime.py` vllm install sequence: `pip install --upgrade 'huggingface_hub>=0.24.0'` added first,
>   plus `sys.modules` cache flush, to resolve `ImportError: cannot import name 'reset_sessions'` on Kaggle's base image.
>
> **Re-run on Kaggle** to obtain ablation numbers comparable to the main experiment.

---

Artifacts:

- `[EXP-24-LLAMA32_1B] DONE` — `/kaggle/working/cultural_alignment/results/exp24_paper_20c/llama32_1b/compare`
- `[EXP-24-LLAMA31_8B] DONE` — `/kaggle/working/cultural_alignment/results/exp24_paper_20c/llama31_8b/compare`
- `[EXP-24-LLAMA33_70B] DONE` — `/kaggle/working/cultural_alignment/results/exp24_paper_20c/llama33_70b/compare`
- `[EXP-24-PHI_4] DONE` — `/kaggle/working/cultural_alignment/results/exp24_paper_20c/phi_4/compare`
- `[EXP-24-PHI35_MINI] DONE` — `/kaggle/working/cultural_alignment/results/exp24_paper_20c/phi35_mini/compare`
- `[EXP-24-GEMMA3_270M] DONE` — `/kaggle/working/cultural_alignment/results/exp24_paper_20c/gemma3_270m/compare`
- `[EXP-24-GEMMA4_E2B] DONE` — `/kaggle/working/cultural_alignment/results/exp24_paper_20c/gemma4_e2b/compare`
- `[EXP-24-GEMMA_7B] DONE` — `/kaggle/working/cultural_alignment/results/exp24_paper_20c/gemma_7b/compare`
- `[EXP-24-QWEN3_VL_8B] DONE` — `/kaggle/working/cultural_alignment/results/exp24_paper_20c/qwen3_vl_8b/compare`
- `[EXP-24-QWEN3_4B_THINKING_2507] DONE` — `/kaggle/working/cultural_alignment/results/exp24_paper_20c/qwen3_4b_thinking_2507/compare`
- `[EXP-24-HF_QWEN25_7B_BF16] DONE` — `/kaggle/working/cultural_alignment/results/exp24_paper_20c/hf_qwen25_7b_bf16/compare`
- `[EXP-24-QWEN25_7B] DONE` — `/kaggle/working/cultural_alignment/results/exp24_paper_20c/qwen25_7b/compare`
- `[EXP-24-QWEN35_08B] DONE` — `/kaggle/working/cultural_alignment/results/exp24_paper_20c/qwen35_08b/compare`
- `[EXP-24-MISTRAL_V03] DONE` — `/kaggle/working/cultural_alignment/results/exp24_paper_20c/mistral_v03/compare`
- `[EXP-24-GPT_OSS_20B] DONE` — `/kaggle/working/cultural_alignment/results/exp24_paper_20c/gpt_oss_20b/compare`
- `[EXP-24-MAGISTRAL_SMALL_2509] DONE` — `/kaggle/working/cultural_alignment/results/exp24_paper_20c/magistral_small_2509/compare`

---

#### EXP-24-LLAMA32_1B Full Metrics

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Llama-3.2-1B | ARG | 0.4558 | 0.0468 | -0.153 | 17.11 | 41.0% |
| Llama-3.2-1B | BGD | 0.4833 | 0.0568 | -0.456 | 17.64 | 22.9% |
| Llama-3.2-1B | BRA | 0.4167 | 0.0488 | -0.187 | 15.31 | 3.2% |
| Llama-3.2-1B | CHN | 0.4866 | 0.0659 | -0.883 | 17.08 | 2.9% |
| Llama-3.2-1B | COL | 0.4791 | 0.0464 | -0.170 | 18.12 | 44.5% |
| Llama-3.2-1B | DEU | 0.5036 | 0.0627 | +0.201 | 17.97 | 7.4% |
| Llama-3.2-1B | ETH | 0.6114 | 0.0674 | +0.185 | 22.03 | 10.0% |
| Llama-3.2-1B | GBR | 0.5249 | 0.0592 | -0.534 | 19.26 | 1.3% |
| Llama-3.2-1B | IDN | 0.4639 | 0.0520 | -0.477 | 17.14 | 0.0% |
| Llama-3.2-1B | IRN | 0.5146 | 0.0685 | -0.169 | 18.00 | 44.5% |
| Llama-3.2-1B | JPN | 0.4510 | 0.0565 | -0.312 | 16.22 | 0.0% |
| Llama-3.2-1B | KGZ | 0.4864 | 0.0579 | -0.276 | 17.69 | 2.6% |
| Llama-3.2-1B | MEX | 0.4781 | 0.0472 | +0.150 | 18.04 | 41.9% |
| Llama-3.2-1B | MMR | 0.4717 | 0.0666 | -0.437 | 16.32 | 2.6% |
| Llama-3.2-1B | MYS | 0.4464 | 0.0511 | -0.474 | 16.45 | 45.5% |
| Llama-3.2-1B | ROU | 0.5069 | 0.0581 | -0.553 | 18.55 | 71.9% |
| Llama-3.2-1B | SRB | 0.5051 | 0.0562 | -0.197 | 18.61 | 0.0% |
| Llama-3.2-1B | THA | 0.4385 | 0.0568 | -0.060 | 15.62 | 4.2% |
| Llama-3.2-1B | USA | 0.5139 | 0.0562 | -0.299 | 18.98 | 0.0% |
| Llama-3.2-1B | VNM | 0.4840 | 0.0497 | +0.178 | 18.13 | 7.7% |

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-LLAMA32_1B vs Vanilla (MIS)

| Model | Country | Vanilla MIS | EXP-24-LLAMA32_1B MIS | Δ | Improv% | Win? |
|:------|:-------:|:-----------:|:---------------------:|:-:|:-------:|:----:|
| Llama-3.2-1B | ARG | 0.4625 | 0.4558 | +0.0066 | **+1.43%** | ✅ |
| Llama-3.2-1B | BGD | 0.4879 | 0.4833 | +0.0046 | **+0.95%** | ✅ |
| Llama-3.2-1B | BRA | 0.4157 | 0.4167 | -0.0009 | **-0.23%** | ❌ |
| Llama-3.2-1B | CHN | 0.5321 | 0.4866 | +0.0456 | **+8.57%** | ✅ |
| Llama-3.2-1B | COL | 0.4858 | 0.4791 | +0.0067 | **+1.39%** | ✅ |
| Llama-3.2-1B | DEU | 0.5004 | 0.5036 | -0.0032 | **-0.65%** | ❌ |
| Llama-3.2-1B | ETH | 0.6265 | 0.6114 | +0.0150 | **+2.40%** | ✅ |
| Llama-3.2-1B | GBR | 0.5190 | 0.5249 | -0.0060 | **-1.15%** | ❌ |
| Llama-3.2-1B | IDN | 0.4589 | 0.4639 | -0.0049 | **-1.07%** | ❌ |
| Llama-3.2-1B | IRN | 0.5363 | 0.5146 | +0.0217 | **+4.04%** | ✅ |
| Llama-3.2-1B | JPN | 0.4551 | 0.4510 | +0.0041 | **+0.90%** | ✅ |
| Llama-3.2-1B | KGZ | 0.4930 | 0.4864 | +0.0065 | **+1.33%** | ✅ |
| Llama-3.2-1B | MEX | 0.4841 | 0.4781 | +0.0060 | **+1.24%** | ✅ |
| Llama-3.2-1B | MMR | 0.4661 | 0.4717 | -0.0056 | **-1.20%** | ❌ |
| Llama-3.2-1B | MYS | 0.4540 | 0.4464 | +0.0076 | **+1.67%** | ✅ |
| Llama-3.2-1B | ROU | 0.5084 | 0.5069 | +0.0015 | **+0.30%** | ✅ |
| Llama-3.2-1B | SRB | 0.5048 | 0.5051 | -0.0003 | **-0.07%** | ❌ |
| Llama-3.2-1B | THA | 0.4496 | 0.4385 | +0.0111 | **+2.47%** | ✅ |
| Llama-3.2-1B | USA | 0.5113 | 0.5139 | -0.0026 | **-0.52%** | ❌ |
| Llama-3.2-1B | VNM | 0.4921 | 0.4840 | +0.0081 | **+1.65%** | ✅ |

- **Llama-3.2-1B** Win Rate: **13/20** | Vanilla=0.4922 → EXP-24-LLAMA32_1B=0.4861 | Macro Δ: **+1.24%**

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-LLAMA32_1B vs EXP-01 SWA-PTIS (MIS)

| Model | Country | EXP-01 MIS | EXP-24-LLAMA32_1B MIS | Δ | Improv% | Win? |
|:------|:-------:|:----------:|:---------------------:|:-:|:-------:|:----:|

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-LLAMA32_1B Leaderboard Entry

| Rank | Method | Coverage | Mean MIS ↓ | Notes |
|:---:|:-------|:--------:|-----------:|:------|
| ? | **EXP-24-LLAMA32_1B** | 1 models × 20 countries | **0.4861** | MIS↓ JSD=0.0565 r=-0.246 Flip=17.7% |

**DPBR summary:** Mean MIS=0.4861, r=-0.246, Flip=17.7%, rel_r=0.993 · *(EXP-09 SOTA: 0.3975 | EXP-24 multi-model ref: 0.3969)*

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-LLAMA32_1B Per-Dimension Worst Errors

| Model | Country | Worst Dim | Human | Model | Abs err (pp) |
|:------|:-------:|:----------|:-----:|:-----:|:------------:|
| Llama-3.2-1B | ARG | Age_Young | 75.1 | 49.9 | **25.2** |
| Llama-3.2-1B | BGD | Species_Humans | 78.5 | 49.9 | **28.6** |
| Llama-3.2-1B | BRA | Age_Young | 73.6 | 49.8 | **23.8** |
| Llama-3.2-1B | CHN | Species_Humans | 83.0 | 48.4 | **34.6** |
| Llama-3.2-1B | COL | Age_Young | 76.3 | 50.0 | **26.3** |
| Llama-3.2-1B | DEU | Species_Humans | 82.4 | 50.1 | **32.4** |
| Llama-3.2-1B | ETH | Species_Humans | 93.9 | 50.1 | **43.8** |
| Llama-3.2-1B | GBR | Species_Humans | 79.9 | 50.1 | **29.8** |
| Llama-3.2-1B | IDN | Species_Humans | 77.4 | 49.7 | **27.7** |
| Llama-3.2-1B | IRN | Species_Humans | 84.0 | 50.1 | **33.9** |
| Llama-3.2-1B | JPN | Species_Humans | 79.8 | 49.2 | **30.7** |
| Llama-3.2-1B | KGZ | Species_Humans | 78.7 | 50.1 | **28.6** |
| Llama-3.2-1B | MEX | Age_Young | 75.2 | 50.0 | **25.2** |
| Llama-3.2-1B | MMR | Utilitarianism_More | 78.7 | 49.9 | **28.8** |
| Llama-3.2-1B | MYS | Species_Humans | 76.6 | 49.9 | **26.8** |
| Llama-3.2-1B | ROU | Species_Humans | 80.1 | 49.8 | **30.3** |
| Llama-3.2-1B | SRB | Species_Humans | 77.7 | 50.1 | **27.6** |
| Llama-3.2-1B | THA | Species_Humans | 79.7 | 50.1 | **29.6** |
| Llama-3.2-1B | USA | Species_Humans | 79.2 | 50.4 | **28.8** |
| Llama-3.2-1B | VNM | Species_Humans | 77.7 | 50.1 | **27.6** |

---

#### EXP-24-LLAMA31_8B Full Metrics

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Meta-Llama-3.1-8B | ARG | 0.4563 | 0.0465 | +0.114 | 17.15 | 42.3% |
| Meta-Llama-3.1-8B | BGD | 0.4790 | 0.0570 | -0.830 | 17.43 | 28.1% |
| Meta-Llama-3.1-8B | BRA | 0.4152 | 0.0487 | -0.255 | 15.25 | 16.1% |
| Meta-Llama-3.1-8B | CHN | 0.4689 | 0.0595 | -0.466 | 16.68 | 7.7% |
| Meta-Llama-3.1-8B | COL | 0.4798 | 0.0461 | +0.000 | 18.16 | 38.4% |
| Meta-Llama-3.1-8B | DEU | 0.5033 | 0.0627 | +0.032 | 17.95 | 17.4% |
| Meta-Llama-3.1-8B | ETH | 0.6101 | 0.0685 | -0.472 | 21.89 | 53.2% |
| Meta-Llama-3.1-8B | GBR | 0.5206 | 0.0589 | -0.867 | 19.08 | 13.5% |
| Meta-Llama-3.1-8B | IDN | 0.4616 | 0.0512 | -0.178 | 17.08 | 3.9% |
| Meta-Llama-3.1-8B | IRN | 0.5145 | 0.0683 | -0.287 | 18.01 | 11.6% |
| Meta-Llama-3.1-8B | JPN | 0.4423 | 0.0554 | -0.265 | 15.89 | 47.4% |
| Meta-Llama-3.1-8B | KGZ | 0.4847 | 0.0588 | -0.855 | 17.54 | 15.8% |
| Meta-Llama-3.1-8B | MEX | 0.4790 | 0.0473 | +0.200 | 18.07 | 36.5% |
| Meta-Llama-3.1-8B | MMR | 0.4689 | 0.0671 | -0.910 | 16.15 | 29.4% |
| Meta-Llama-3.1-8B | MYS | 0.4430 | 0.0515 | -0.860 | 16.27 | 13.9% |
| Meta-Llama-3.1-8B | ROU | 0.5031 | 0.0587 | -0.857 | 18.34 | 17.1% |
| Meta-Llama-3.1-8B | SRB | 0.5025 | 0.0572 | -0.841 | 18.44 | 31.9% |
| Meta-Llama-3.1-8B | THA | 0.4373 | 0.0581 | -0.889 | 15.47 | 10.0% |
| Meta-Llama-3.1-8B | USA | 0.5110 | 0.0565 | -0.867 | 18.83 | 17.7% |
| Meta-Llama-3.1-8B | VNM | 0.4789 | 0.0504 | -0.332 | 17.87 | 2.6% |

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-LLAMA31_8B vs Vanilla (MIS)

| Model | Country | Vanilla MIS | EXP-24-LLAMA31_8B MIS | Δ | Improv% | Win? |
|:------|:-------:|:-----------:|:---------------------:|:-:|:-------:|:----:|
| Meta-Llama-3.1-8B | ARG | 0.4771 | 0.4563 | +0.0208 | **+4.36%** | ✅ |
| Meta-Llama-3.1-8B | BGD | 0.4778 | 0.4790 | -0.0013 | **-0.26%** | ❌ |
| Meta-Llama-3.1-8B | BRA | 0.4179 | 0.4152 | +0.0027 | **+0.64%** | ✅ |
| Meta-Llama-3.1-8B | CHN | 0.4690 | 0.4689 | +0.0001 | **+0.02%** | ✅ |
| Meta-Llama-3.1-8B | COL | 0.4988 | 0.4798 | +0.0190 | **+3.80%** | ✅ |
| Meta-Llama-3.1-8B | DEU | 0.5006 | 0.5033 | -0.0027 | **-0.54%** | ❌ |
| Meta-Llama-3.1-8B | ETH | 0.6148 | 0.6101 | +0.0047 | **+0.76%** | ✅ |
| Meta-Llama-3.1-8B | GBR | 0.5173 | 0.5206 | -0.0033 | **-0.64%** | ❌ |
| Meta-Llama-3.1-8B | IDN | 0.4526 | 0.4616 | -0.0090 | **-1.99%** | ❌ |
| Meta-Llama-3.1-8B | IRN | 0.5242 | 0.5145 | +0.0097 | **+1.85%** | ✅ |
| Meta-Llama-3.1-8B | JPN | 0.4509 | 0.4423 | +0.0086 | **+1.91%** | ✅ |
| Meta-Llama-3.1-8B | KGZ | 0.4838 | 0.4847 | -0.0009 | **-0.19%** | ❌ |
| Meta-Llama-3.1-8B | MEX | 0.5013 | 0.4790 | +0.0224 | **+4.46%** | ✅ |
| Meta-Llama-3.1-8B | MMR | 0.4613 | 0.4689 | -0.0076 | **-1.64%** | ❌ |
| Meta-Llama-3.1-8B | MYS | 0.4427 | 0.4430 | -0.0003 | **-0.06%** | ❌ |
| Meta-Llama-3.1-8B | ROU | 0.5020 | 0.5031 | -0.0011 | **-0.21%** | ❌ |
| Meta-Llama-3.1-8B | SRB | 0.4991 | 0.5025 | -0.0034 | **-0.68%** | ❌ |
| Meta-Llama-3.1-8B | THA | 0.4403 | 0.4373 | +0.0030 | **+0.69%** | ✅ |
| Meta-Llama-3.1-8B | USA | 0.5086 | 0.5110 | -0.0024 | **-0.48%** | ❌ |
| Meta-Llama-3.1-8B | VNM | 0.4845 | 0.4789 | +0.0056 | **+1.15%** | ✅ |

- **Meta-Llama-3.1-8B** Win Rate: **10/20** | Vanilla=0.4862 → EXP-24-LLAMA31_8B=0.4830 | Macro Δ: **+0.66%**

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-LLAMA31_8B vs EXP-01 SWA-PTIS (MIS)

| Model | Country | EXP-01 MIS | EXP-24-LLAMA31_8B MIS | Δ | Improv% | Win? |
|:------|:-------:|:----------:|:---------------------:|:-:|:-------:|:----:|

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-LLAMA31_8B Leaderboard Entry

| Rank | Method | Coverage | Mean MIS ↓ | Notes |
|:---:|:-------|:--------:|-----------:|:------|
| ? | **EXP-24-LLAMA31_8B** | 1 models × 20 countries | **0.4830** | MIS↓ JSD=0.0564 r=-0.484 Flip=22.7% |

**DPBR summary:** Mean MIS=0.4830, r=-0.484, Flip=22.7%, rel_r=0.995 · *(EXP-09 SOTA: 0.3975 | EXP-24 multi-model ref: 0.3969)*

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-LLAMA31_8B Per-Dimension Worst Errors

| Model | Country | Worst Dim | Human | Model | Abs err (pp) |
|:------|:-------:|:----------|:-----:|:-----:|:------------:|
| Meta-Llama-3.1-8B | ARG | Utilitarianism_More | 75.0 | 50.0 | **25.0** |
| Meta-Llama-3.1-8B | BGD | Species_Humans | 78.5 | 49.8 | **28.7** |
| Meta-Llama-3.1-8B | BRA | Utilitarianism_More | 73.7 | 50.2 | **23.5** |
| Meta-Llama-3.1-8B | CHN | Species_Humans | 83.0 | 49.9 | **33.0** |
| Meta-Llama-3.1-8B | COL | Age_Young | 76.3 | 50.2 | **26.1** |
| Meta-Llama-3.1-8B | DEU | Species_Humans | 82.4 | 50.1 | **32.3** |
| Meta-Llama-3.1-8B | ETH | Species_Humans | 93.9 | 49.9 | **44.0** |
| Meta-Llama-3.1-8B | GBR | Species_Humans | 79.9 | 49.7 | **30.2** |
| Meta-Llama-3.1-8B | IDN | Species_Humans | 77.4 | 49.8 | **27.6** |
| Meta-Llama-3.1-8B | IRN | Species_Humans | 84.0 | 50.0 | **33.9** |
| Meta-Llama-3.1-8B | JPN | Species_Humans | 79.8 | 49.9 | **29.9** |
| Meta-Llama-3.1-8B | KGZ | Species_Humans | 78.7 | 49.9 | **28.8** |
| Meta-Llama-3.1-8B | MEX | Age_Young | 75.2 | 50.2 | **25.0** |
| Meta-Llama-3.1-8B | MMR | Utilitarianism_More | 78.7 | 49.9 | **28.8** |
| Meta-Llama-3.1-8B | MYS | Species_Humans | 76.6 | 49.8 | **26.8** |
| Meta-Llama-3.1-8B | ROU | Species_Humans | 80.1 | 49.8 | **30.3** |
| Meta-Llama-3.1-8B | SRB | Species_Humans | 77.7 | 49.8 | **27.9** |
| Meta-Llama-3.1-8B | THA | Species_Humans | 79.7 | 49.8 | **29.9** |
| Meta-Llama-3.1-8B | USA | Species_Humans | 79.2 | 49.9 | **29.3** |
| Meta-Llama-3.1-8B | VNM | Species_Humans | 77.7 | 49.9 | **27.8** |

---

#### EXP-24-LLAMA33_70B Full Metrics

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Llama-3.3-70B | ARG | 0.8909 | 0.1785 | -0.456 | 32.57 | 21.3% |
| Llama-3.3-70B | BGD | 0.6281 | 0.1624 | +0.029 | 18.69 | 19.7% |
| Llama-3.3-70B | BRA | 0.4580 | 0.1191 | +0.222 | 13.84 | 19.0% |
| Llama-3.3-70B | CHN | 0.7484 | 0.1831 | -0.025 | 24.43 | 18.7% |
| Llama-3.3-70B | COL | 0.9313 | 0.1892 | -0.594 | 33.90 | 21.0% |
| Llama-3.3-70B | DEU | 0.6434 | 0.1893 | +0.101 | 17.38 | 17.4% |
| Llama-3.3-70B | ETH | 0.7473 | 0.1722 | -0.077 | 26.27 | 21.0% |
| Llama-3.3-70B | GBR | 0.5983 | 0.1408 | +0.125 | 18.71 | 18.4% |
| Llama-3.3-70B | IDN | 0.5571 | 0.0924 | +0.071 | 19.90 | 21.3% |
| Llama-3.3-70B | IRN | 0.8524 | 0.1588 | -0.294 | 30.40 | 16.5% |
| Llama-3.3-70B | JPN | 0.4709 | 0.1070 | +0.249 | 15.39 | 22.9% |
| Llama-3.3-70B | KGZ | 0.5848 | 0.1486 | +0.176 | 16.89 | 20.6% |
| Llama-3.3-70B | MEX | 0.9093 | 0.1789 | -0.385 | 33.42 | 21.6% |
| Llama-3.3-70B | MMR | 0.6521 | 0.1847 | -0.031 | 16.38 | 21.3% |
| Llama-3.3-70B | MYS | 0.5890 | 0.1552 | +0.003 | 18.04 | 20.6% |
| Llama-3.3-70B | ROU | 0.5932 | 0.1470 | +0.197 | 17.80 | 19.7% |
| Llama-3.3-70B | SRB | 0.6048 | 0.1500 | +0.157 | 17.72 | 20.0% |
| Llama-3.3-70B | THA | 0.5766 | 0.1517 | +0.125 | 17.63 | 19.7% |
| Llama-3.3-70B | USA | 0.5780 | 0.1213 | +0.266 | 18.59 | 15.8% |
| Llama-3.3-70B | VNM | 0.7404 | 0.1417 | -0.045 | 26.53 | 21.9% |

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-LLAMA33_70B vs Vanilla (MIS)

| Model | Country | Vanilla MIS | EXP-24-LLAMA33_70B MIS | Δ | Improv% | Win? |
|:------|:-------:|:-----------:|:----------------------:|:-:|:-------:|:----:|
| Llama-3.3-70B | ARG | 1.0147 | 0.8909 | +0.1238 | **+12.20%** | ✅ |
| Llama-3.3-70B | BGD | 0.8640 | 0.6281 | +0.2359 | **+27.30%** | ✅ |
| Llama-3.3-70B | BRA | 0.5213 | 0.4580 | +0.0633 | **+12.15%** | ✅ |
| Llama-3.3-70B | CHN | 0.9010 | 0.7484 | +0.1526 | **+16.94%** | ✅ |
| Llama-3.3-70B | COL | 1.0409 | 0.9313 | +0.1095 | **+10.52%** | ✅ |
| Llama-3.3-70B | DEU | 0.7155 | 0.6434 | +0.0720 | **+10.07%** | ✅ |
| Llama-3.3-70B | ETH | 0.9673 | 0.7473 | +0.2200 | **+22.74%** | ✅ |
| Llama-3.3-70B | GBR | 0.8726 | 0.5983 | +0.2743 | **+31.43%** | ✅ |
| Llama-3.3-70B | IDN | 0.6876 | 0.5571 | +0.1306 | **+18.99%** | ✅ |
| Llama-3.3-70B | IRN | 1.0389 | 0.8524 | +0.1865 | **+17.95%** | ✅ |
| Llama-3.3-70B | JPN | 0.5206 | 0.4709 | +0.0497 | **+9.54%** | ✅ |
| Llama-3.3-70B | KGZ | 0.8492 | 0.5848 | +0.2644 | **+31.13%** | ✅ |
| Llama-3.3-70B | MEX | 1.0367 | 0.9093 | +0.1274 | **+12.29%** | ✅ |
| Llama-3.3-70B | MMR | 0.8717 | 0.6521 | +0.2196 | **+25.19%** | ✅ |
| Llama-3.3-70B | MYS | 0.8405 | 0.5890 | +0.2514 | **+29.92%** | ✅ |
| Llama-3.3-70B | ROU | 0.8511 | 0.5932 | +0.2578 | **+30.30%** | ✅ |
| Llama-3.3-70B | SRB | 0.8624 | 0.6048 | +0.2576 | **+29.87%** | ✅ |
| Llama-3.3-70B | THA | 0.8145 | 0.5766 | +0.2378 | **+29.20%** | ✅ |
| Llama-3.3-70B | USA | 0.8730 | 0.5780 | +0.2950 | **+33.79%** | ✅ |
| Llama-3.3-70B | VNM | 0.8450 | 0.7404 | +0.1046 | **+12.38%** | ✅ |

- **Llama-3.3-70B** Win Rate: **20/20** | Vanilla=0.8494 → EXP-24-LLAMA33_70B=0.6677 | Macro Δ: **+21.39%**

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-LLAMA33_70B vs EXP-01 SWA-PTIS (MIS)

| Model | Country | EXP-01 MIS | EXP-24-LLAMA33_70B MIS | Δ | Improv% | Win? |
|:------|:-------:|:----------:|:----------------------:|:-:|:-------:|:----:|

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-LLAMA33_70B Leaderboard Entry

| Rank | Method | Coverage | Mean MIS ↓ | Notes |
|:---:|:-------|:--------:|-----------:|:------|
| ? | **EXP-24-LLAMA33_70B** | 1 models × 20 countries | **0.6677** | MIS↓ JSD=0.1536 r=-0.009 Flip=19.9% |

**DPBR summary:** Mean MIS=0.6677, r=-0.009, Flip=19.9%, rel_r=0.831 · *(EXP-09 SOTA: 0.3975 | EXP-24 multi-model ref: 0.3969)*

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-LLAMA33_70B Per-Dimension Worst Errors

| Model | Country | Worst Dim | Human | Model | Abs err (pp) |
|:------|:-------:|:----------|:-----:|:-----:|:------------:|
| Llama-3.3-70B | ARG | Utilitarianism_More | 75.0 | 18.8 | **56.2** |
| Llama-3.3-70B | BGD | Utilitarianism_More | 75.0 | 26.8 | **48.2** |
| Llama-3.3-70B | BRA | SocialValue_High | 66.3 | 28.3 | **38.1** |
| Llama-3.3-70B | CHN | SocialValue_High | 66.7 | 16.5 | **50.2** |
| Llama-3.3-70B | COL | Utilitarianism_More | 75.8 | 18.1 | **57.7** |
| Llama-3.3-70B | DEU | Utilitarianism_More | 75.1 | 19.2 | **55.9** |
| Llama-3.3-70B | ETH | Utilitarianism_More | 80.0 | 27.3 | **52.7** |
| Llama-3.3-70B | GBR | Utilitarianism_More | 77.0 | 32.5 | **44.4** |
| Llama-3.3-70B | IDN | SocialValue_High | 69.0 | 30.2 | **38.8** |
| Llama-3.3-70B | IRN | SocialValue_High | 67.4 | 18.4 | **49.0** |
| Llama-3.3-70B | JPN | SocialValue_High | 65.9 | 32.7 | **33.2** |
| Llama-3.3-70B | KGZ | Utilitarianism_More | 73.0 | 29.2 | **43.7** |
| Llama-3.3-70B | MEX | Utilitarianism_More | 74.8 | 18.5 | **56.3** |
| Llama-3.3-70B | MMR | Utilitarianism_More | 78.7 | 22.0 | **56.7** |
| Llama-3.3-70B | MYS | Utilitarianism_More | 72.0 | 27.8 | **44.3** |
| Llama-3.3-70B | ROU | Utilitarianism_More | 74.0 | 29.2 | **44.8** |
| Llama-3.3-70B | SRB | Utilitarianism_More | 75.6 | 28.7 | **46.8** |
| Llama-3.3-70B | THA | Utilitarianism_More | 70.3 | 26.9 | **43.4** |
| Llama-3.3-70B | USA | Utilitarianism_More | 76.6 | 33.4 | **43.2** |
| Llama-3.3-70B | VNM | SocialValue_High | 69.0 | 21.3 | **47.7** |

---

#### EXP-24-PHI_4 Full Metrics

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Phi-4 | ARG | 0.3885 | 0.0459 | +0.412 | 14.30 | 22.9% |
| Phi-4 | BGD | 0.3684 | 0.0477 | +0.607 | 13.33 | 13.5% |
| Phi-4 | BRA | 0.2993 | 0.0607 | +0.471 | 9.26 | 25.8% |
| Phi-4 | CHN | 0.2141 | 0.0541 | +0.781 | 6.96 | 17.1% |
| Phi-4 | COL | 0.4377 | 0.0570 | +0.093 | 15.81 | 21.9% |
| Phi-4 | DEU | 0.2550 | 0.0595 | +0.779 | 8.41 | 17.7% |
| Phi-4 | ETH | 0.4848 | 0.0410 | +0.808 | 18.15 | 8.4% |
| Phi-4 | GBR | 0.3189 | 0.0823 | +0.594 | 8.76 | 20.3% |
| Phi-4 | IDN | 0.2742 | 0.0591 | +0.595 | 8.43 | 18.1% |
| Phi-4 | IRN | 0.5296 | 0.0865 | +0.055 | 17.72 | 18.4% |
| Phi-4 | JPN | 0.1951 | 0.0507 | +0.671 | 6.12 | 17.4% |
| Phi-4 | KGZ | 0.3934 | 0.0584 | +0.476 | 13.73 | 13.5% |
| Phi-4 | MEX | 0.4204 | 0.0441 | +0.512 | 15.88 | 24.5% |
| Phi-4 | MMR | 0.3568 | 0.0568 | +0.570 | 11.91 | 10.0% |
| Phi-4 | MYS | 0.3280 | 0.0530 | +0.614 | 11.37 | 17.4% |
| Phi-4 | ROU | 0.3853 | 0.0577 | +0.570 | 13.51 | 15.2% |
| Phi-4 | SRB | 0.3854 | 0.0570 | +0.564 | 13.55 | 16.8% |
| Phi-4 | THA | 0.3127 | 0.0467 | +0.685 | 10.99 | 13.2% |
| Phi-4 | USA | 0.2433 | 0.0581 | +0.723 | 7.05 | 18.7% |
| Phi-4 | VNM | 0.3363 | 0.0640 | +0.596 | 10.67 | 17.4% |

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-PHI_4 vs Vanilla (MIS)

| Model | Country | Vanilla MIS | EXP-24-PHI_4 MIS | Δ | Improv% | Win? |
|:------|:-------:|:-----------:|:----------------:|:-:|:-------:|:----:|
| Phi-4 | ARG | 0.4629 | 0.3885 | +0.0744 | **+16.06%** | ✅ |
| Phi-4 | BGD | 0.4894 | 0.3684 | +0.1210 | **+24.73%** | ✅ |
| Phi-4 | BRA | 0.2739 | 0.2993 | -0.0254 | **-9.29%** | ❌ |
| Phi-4 | CHN | 0.4069 | 0.2141 | +0.1927 | **+47.36%** | ✅ |
| Phi-4 | COL | 0.4868 | 0.4377 | +0.0491 | **+10.08%** | ✅ |
| Phi-4 | DEU | 0.3023 | 0.2550 | +0.0473 | **+15.64%** | ✅ |
| Phi-4 | ETH | 0.6150 | 0.4848 | +0.1302 | **+21.17%** | ✅ |
| Phi-4 | GBR | 0.5029 | 0.3189 | +0.1839 | **+36.58%** | ✅ |
| Phi-4 | IDN | 0.4591 | 0.2742 | +0.1849 | **+40.28%** | ✅ |
| Phi-4 | IRN | 0.3967 | 0.5296 | -0.1329 | **-33.51%** | ❌ |
| Phi-4 | JPN | 0.3948 | 0.1951 | +0.1996 | **+50.57%** | ✅ |
| Phi-4 | KGZ | 0.4987 | 0.3934 | +0.1054 | **+21.13%** | ✅ |
| Phi-4 | MEX | 0.4853 | 0.4204 | +0.0649 | **+13.38%** | ✅ |
| Phi-4 | MMR | 0.4818 | 0.3568 | +0.1250 | **+25.95%** | ✅ |
| Phi-4 | MYS | 0.4588 | 0.3280 | +0.1308 | **+28.51%** | ✅ |
| Phi-4 | ROU | 0.5039 | 0.3853 | +0.1186 | **+23.53%** | ✅ |
| Phi-4 | SRB | 0.4999 | 0.3854 | +0.1146 | **+22.92%** | ✅ |
| Phi-4 | THA | 0.4559 | 0.3127 | +0.1431 | **+31.40%** | ✅ |
| Phi-4 | USA | 0.4975 | 0.2433 | +0.2542 | **+51.09%** | ✅ |
| Phi-4 | VNM | 0.4065 | 0.3363 | +0.0702 | **+17.28%** | ✅ |

- **Phi-4** Win Rate: **18/20** | Vanilla=0.4539 → EXP-24-PHI_4=0.3464 | Macro Δ: **+23.70%**

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-PHI_4 vs EXP-01 SWA-PTIS (MIS)

| Model | Country | EXP-01 MIS | EXP-24-PHI_4 MIS | Δ | Improv% | Win? |
|:------|:-------:|:----------:|:----------------:|:-:|:-------:|:----:|

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-PHI_4 Leaderboard Entry

| Rank | Method | Coverage | Mean MIS ↓ | Notes |
|:---:|:-------|:--------:|-----------:|:------|
| ? | **EXP-24-PHI_4** | 1 models × 20 countries | **0.3464** | MIS↓ JSD=0.0570 r=+0.559 Flip=17.4% |

**DPBR summary:** Mean MIS=0.3464, r=+0.559, Flip=17.4%, rel_r=0.934 · *(EXP-09 SOTA: 0.3975 | EXP-24 multi-model ref: 0.3969)*

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-PHI_4 Per-Dimension Worst Errors

| Model | Country | Worst Dim | Human | Model | Abs err (pp) |
|:------|:-------:|:----------|:-----:|:-----:|:------------:|
| Phi-4 | ARG | Utilitarianism_More | 75.0 | 53.8 | **21.2** |
| Phi-4 | BGD | SocialValue_High | 68.1 | 46.6 | **21.5** |
| Phi-4 | BRA | Age_Young | 73.6 | 52.1 | **21.5** |
| Phi-4 | CHN | SocialValue_High | 66.7 | 50.5 | **16.2** |
| Phi-4 | COL | SocialValue_High | 72.7 | 45.7 | **27.0** |
| Phi-4 | DEU | SocialValue_High | 64.7 | 48.3 | **16.5** |
| Phi-4 | ETH | Species_Humans | 93.9 | 60.4 | **33.4** |
| Phi-4 | GBR | SocialValue_High | 67.7 | 38.9 | **28.8** |
| Phi-4 | IDN | SocialValue_High | 69.0 | 45.9 | **23.1** |
| Phi-4 | IRN | Species_Humans | 84.0 | 49.5 | **34.4** |
| Phi-4 | JPN | Utilitarianism_More | 68.7 | 82.1 | **13.4** |
| Phi-4 | KGZ | Age_Young | 74.0 | 48.8 | **25.1** |
| Phi-4 | MEX | SocialValue_High | 69.6 | 46.0 | **23.6** |
| Phi-4 | MMR | Age_Young | 75.9 | 49.4 | **26.5** |
| Phi-4 | MYS | SocialValue_High | 67.4 | 44.3 | **23.2** |
| Phi-4 | ROU | Age_Young | 74.9 | 50.0 | **24.9** |
| Phi-4 | SRB | Age_Young | 76.0 | 50.8 | **25.2** |
| Phi-4 | THA | Age_Young | 68.1 | 48.9 | **19.2** |
| Phi-4 | USA | SocialValue_High | 67.9 | 48.2 | **19.7** |
| Phi-4 | VNM | Age_Young | 72.7 | 43.6 | **29.1** |

---

#### EXP-24-PHI35_MINI Full Metrics

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Phi-3.5-mini | ARG | 0.6454 | 0.1587 | -0.498 | 20.50 | 16.8% |
| Phi-3.5-mini | BGD | 0.5645 | 0.1153 | -0.426 | 18.82 | 17.1% |
| Phi-3.5-mini | BRA | 0.6487 | 0.1734 | -0.743 | 22.41 | 20.3% |
| Phi-3.5-mini | CHN | 0.5586 | 0.1620 | -0.041 | 18.09 | 19.4% |
| Phi-3.5-mini | COL | 0.6781 | 0.1653 | -0.602 | 21.06 | 16.5% |
| Phi-3.5-mini | DEU | 0.5547 | 0.1462 | -0.403 | 18.71 | 21.0% |
| Phi-3.5-mini | ETH | 0.6609 | 0.1138 | -0.027 | 22.00 | 21.9% |
| Phi-3.5-mini | GBR | 0.5881 | 0.1135 | -0.392 | 19.50 | 20.0% |
| Phi-3.5-mini | IDN | 0.4838 | 0.1199 | -0.769 | 17.66 | 15.2% |
| Phi-3.5-mini | IRN | 0.4454 | 0.0563 | +0.766 | 15.65 | 1.3% |
| Phi-3.5-mini | JPN | 0.4552 | 0.0628 | -0.233 | 16.08 | 3.2% |
| Phi-3.5-mini | KGZ | 0.5720 | 0.1191 | -0.405 | 19.57 | 17.7% |
| Phi-3.5-mini | MEX | 0.6622 | 0.1589 | -0.494 | 21.44 | 14.8% |
| Phi-3.5-mini | MMR | 0.5901 | 0.1352 | -0.623 | 19.68 | 19.4% |
| Phi-3.5-mini | MYS | 0.5352 | 0.1180 | -0.426 | 18.16 | 19.4% |
| Phi-3.5-mini | ROU | 0.5769 | 0.1173 | -0.352 | 19.51 | 19.0% |
| Phi-3.5-mini | SRB | 0.6015 | 0.1241 | -0.463 | 20.12 | 19.0% |
| Phi-3.5-mini | THA | 0.5200 | 0.1123 | -0.283 | 18.00 | 19.4% |
| Phi-3.5-mini | USA | 0.5788 | 0.1101 | -0.380 | 18.84 | 19.0% |
| Phi-3.5-mini | VNM | 0.5069 | 0.0668 | -0.143 | 18.38 | 12.9% |

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-PHI35_MINI vs Vanilla (MIS)

| Model | Country | Vanilla MIS | EXP-24-PHI35_MINI MIS | Δ | Improv% | Win? |
|:------|:-------:|:-----------:|:---------------------:|:-:|:-------:|:----:|
| Phi-3.5-mini | ARG | 0.7697 | 0.6454 | +0.1243 | **+16.15%** | ✅ |
| Phi-3.5-mini | BGD | 0.6663 | 0.5645 | +0.1018 | **+15.27%** | ✅ |
| Phi-3.5-mini | BRA | 0.5313 | 0.6487 | -0.1174 | **-22.10%** | ❌ |
| Phi-3.5-mini | CHN | 0.7381 | 0.5586 | +0.1795 | **+24.32%** | ✅ |
| Phi-3.5-mini | COL | 0.7885 | 0.6781 | +0.1104 | **+14.00%** | ✅ |
| Phi-3.5-mini | DEU | 0.5692 | 0.5547 | +0.0145 | **+2.54%** | ✅ |
| Phi-3.5-mini | ETH | 0.7342 | 0.6609 | +0.0733 | **+9.98%** | ✅ |
| Phi-3.5-mini | GBR | 0.6541 | 0.5881 | +0.0660 | **+10.10%** | ✅ |
| Phi-3.5-mini | IDN | 0.4918 | 0.4838 | +0.0080 | **+1.62%** | ✅ |
| Phi-3.5-mini | IRN | 0.5113 | 0.4454 | +0.0659 | **+12.88%** | ✅ |
| Phi-3.5-mini | JPN | 0.4540 | 0.4552 | -0.0012 | **-0.26%** | ❌ |
| Phi-3.5-mini | KGZ | 0.6599 | 0.5720 | +0.0879 | **+13.31%** | ✅ |
| Phi-3.5-mini | MEX | 0.7824 | 0.6622 | +0.1202 | **+15.37%** | ✅ |
| Phi-3.5-mini | MMR | 0.6846 | 0.5901 | +0.0945 | **+13.81%** | ✅ |
| Phi-3.5-mini | MYS | 0.6306 | 0.5352 | +0.0954 | **+15.13%** | ✅ |
| Phi-3.5-mini | ROU | 0.6707 | 0.5769 | +0.0938 | **+13.98%** | ✅ |
| Phi-3.5-mini | SRB | 0.6767 | 0.6015 | +0.0752 | **+11.11%** | ✅ |
| Phi-3.5-mini | THA | 0.6194 | 0.5200 | +0.0994 | **+16.05%** | ✅ |
| Phi-3.5-mini | USA | 0.6448 | 0.5788 | +0.0660 | **+10.24%** | ✅ |
| Phi-3.5-mini | VNM | 0.4810 | 0.5069 | -0.0259 | **-5.38%** | ❌ |

- **Phi-3.5-mini** Win Rate: **17/20** | Vanilla=0.6379 → EXP-24-PHI35_MINI=0.5713 | Macro Δ: **+10.44%**

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-PHI35_MINI vs EXP-01 SWA-PTIS (MIS)

| Model | Country | EXP-01 MIS | EXP-24-PHI35_MINI MIS | Δ | Improv% | Win? |
|:------|:-------:|:----------:|:---------------------:|:-:|:-------:|:----:|

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-PHI35_MINI Leaderboard Entry

| Rank | Method | Coverage | Mean MIS ↓ | Notes |
|:---:|:-------|:--------:|-----------:|:------|
| ? | **EXP-24-PHI35_MINI** | 1 models × 20 countries | **0.5713** | MIS↓ JSD=0.1225 r=-0.347 Flip=16.7% |

**DPBR summary:** Mean MIS=0.5713, r=-0.347, Flip=16.7%, rel_r=0.953 · *(EXP-09 SOTA: 0.3975 | EXP-24 multi-model ref: 0.3969)*

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-PHI35_MINI Per-Dimension Worst Errors

| Model | Country | Worst Dim | Human | Model | Abs err (pp) |
|:------|:-------:|:----------|:-----:|:-----:|:------------:|
| Phi-3.5-mini | ARG | Utilitarianism_More | 75.0 | 28.8 | **46.2** |
| Phi-3.5-mini | BGD | Utilitarianism_More | 75.0 | 35.9 | **39.1** |
| Phi-3.5-mini | BRA | Utilitarianism_More | 73.7 | 23.2 | **50.5** |
| Phi-3.5-mini | CHN | Utilitarianism_More | 71.1 | 28.2 | **43.0** |
| Phi-3.5-mini | COL | Utilitarianism_More | 75.8 | 28.4 | **47.4** |
| Phi-3.5-mini | DEU | Utilitarianism_More | 75.1 | 30.0 | **45.2** |
| Phi-3.5-mini | ETH | Utilitarianism_More | 80.0 | 34.1 | **45.9** |
| Phi-3.5-mini | GBR | Utilitarianism_More | 77.0 | 35.6 | **41.4** |
| Phi-3.5-mini | IDN | Utilitarianism_More | 72.8 | 45.3 | **27.5** |
| Phi-3.5-mini | IRN | Species_Humans | 84.0 | 55.7 | **28.3** |
| Phi-3.5-mini | JPN | Species_Humans | 79.8 | 51.5 | **28.3** |
| Phi-3.5-mini | KGZ | Utilitarianism_More | 73.0 | 35.2 | **37.8** |
| Phi-3.5-mini | MEX | Utilitarianism_More | 74.8 | 29.3 | **45.5** |
| Phi-3.5-mini | MMR | Utilitarianism_More | 78.7 | 34.7 | **44.0** |
| Phi-3.5-mini | MYS | Utilitarianism_More | 72.0 | 34.4 | **37.7** |
| Phi-3.5-mini | ROU | Utilitarianism_More | 74.0 | 35.2 | **38.9** |
| Phi-3.5-mini | SRB | Utilitarianism_More | 75.6 | 34.1 | **41.5** |
| Phi-3.5-mini | THA | Utilitarianism_More | 70.3 | 35.0 | **35.3** |
| Phi-3.5-mini | USA | Utilitarianism_More | 76.6 | 35.9 | **40.7** |
| Phi-3.5-mini | VNM | Utilitarianism_More | 73.6 | 41.4 | **32.3** |

---

#### EXP-24-GEMMA3_270M Full Metrics

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| gemma-3-270m-it | ARG | 0.4525 | 0.0460 | +0.292 | 17.01 | 13.5% |
| gemma-3-270m-it | BGD | 0.4752 | 0.0548 | +0.222 | 17.40 | 0.0% |
| gemma-3-270m-it | BRA | 0.4037 | 0.0462 | +0.457 | 14.87 | 0.0% |
| gemma-3-270m-it | CHN | 0.4603 | 0.0566 | +0.383 | 16.48 | 8.1% |
| gemma-3-270m-it | COL | 0.4743 | 0.0455 | +0.219 | 17.95 | 16.5% |
| gemma-3-270m-it | DEU | 0.4978 | 0.0610 | +0.644 | 17.81 | 18.7% |
| gemma-3-270m-it | ETH | 0.6060 | 0.0677 | +0.067 | 21.76 | 21.3% |
| gemma-3-270m-it | GBR | 0.5176 | 0.0567 | +0.174 | 19.08 | 22.6% |
| gemma-3-270m-it | IDN | 0.4432 | 0.0477 | +0.580 | 16.46 | 0.0% |
| gemma-3-270m-it | IRN | 0.5162 | 0.0690 | -0.445 | 18.04 | 3.9% |
| gemma-3-270m-it | JPN | 0.4348 | 0.0536 | +0.566 | 15.65 | 1.0% |
| gemma-3-270m-it | KGZ | 0.4850 | 0.0575 | -0.008 | 17.64 | 51.0% |
| gemma-3-270m-it | MEX | 0.4796 | 0.0484 | -0.168 | 18.05 | 12.6% |
| gemma-3-270m-it | MMR | 0.4702 | 0.0668 | -0.182 | 16.23 | 28.7% |
| gemma-3-270m-it | MYS | 0.4364 | 0.0478 | +0.558 | 16.18 | 2.9% |
| gemma-3-270m-it | ROU | 0.4972 | 0.0553 | +0.477 | 18.27 | 9.0% |
| gemma-3-270m-it | SRB | 0.4957 | 0.0550 | +0.260 | 18.26 | 2.3% |
| gemma-3-270m-it | THA | 0.4318 | 0.0557 | +0.225 | 15.39 | 8.1% |
| gemma-3-270m-it | USA | 0.5066 | 0.0545 | +0.168 | 18.75 | 26.8% |
| gemma-3-270m-it | VNM | 0.4769 | 0.0481 | +0.548 | 17.90 | 15.2% |

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-GEMMA3_270M vs Vanilla (MIS)

| Model | Country | Vanilla MIS | EXP-24-GEMMA3_270M MIS | Δ | Improv% | Win? |
|:------|:-------:|:-----------:|:----------------------:|:-:|:-------:|:----:|
| gemma-3-270m-it | ARG | 0.4578 | 0.4525 | +0.0053 | **+1.16%** | ✅ |
| gemma-3-270m-it | BGD | 0.4782 | 0.4752 | +0.0030 | **+0.63%** | ✅ |
| gemma-3-270m-it | BRA | 0.4165 | 0.4037 | +0.0128 | **+3.06%** | ✅ |
| gemma-3-270m-it | CHN | 0.4684 | 0.4603 | +0.0081 | **+1.73%** | ✅ |
| gemma-3-270m-it | COL | 0.4809 | 0.4743 | +0.0066 | **+1.37%** | ✅ |
| gemma-3-270m-it | DEU | 0.5047 | 0.4978 | +0.0068 | **+1.35%** | ✅ |
| gemma-3-270m-it | ETH | 0.6099 | 0.6060 | +0.0039 | **+0.65%** | ✅ |
| gemma-3-270m-it | GBR | 0.5181 | 0.5176 | +0.0006 | **+0.11%** | ✅ |
| gemma-3-270m-it | IDN | 0.3936 | 0.4432 | -0.0496 | **-12.60%** | ❌ |
| gemma-3-270m-it | IRN | 0.5139 | 0.5162 | -0.0023 | **-0.45%** | ❌ |
| gemma-3-270m-it | JPN | 0.4419 | 0.4348 | +0.0071 | **+1.61%** | ✅ |
| gemma-3-270m-it | KGZ | 0.4836 | 0.4850 | -0.0014 | **-0.28%** | ❌ |
| gemma-3-270m-it | MEX | 0.4811 | 0.4796 | +0.0015 | **+0.31%** | ✅ |
| gemma-3-270m-it | MMR | 0.4681 | 0.4702 | -0.0021 | **-0.45%** | ❌ |
| gemma-3-270m-it | MYS | 0.4424 | 0.4364 | +0.0060 | **+1.36%** | ✅ |
| gemma-3-270m-it | ROU | 0.5025 | 0.4972 | +0.0052 | **+1.04%** | ✅ |
| gemma-3-270m-it | SRB | 0.5022 | 0.4957 | +0.0065 | **+1.30%** | ✅ |
| gemma-3-270m-it | THA | 0.4359 | 0.4318 | +0.0041 | **+0.95%** | ✅ |
| gemma-3-270m-it | USA | 0.5094 | 0.5066 | +0.0029 | **+0.56%** | ✅ |
| gemma-3-270m-it | VNM | 0.4810 | 0.4769 | +0.0041 | **+0.85%** | ✅ |

- **gemma-3-270m-it** Win Rate: **16/20** | Vanilla=0.4795 → EXP-24-GEMMA3_270M=0.4781 | Macro Δ: **+0.30%**

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-GEMMA3_270M vs EXP-01 SWA-PTIS (MIS)

| Model | Country | EXP-01 MIS | EXP-24-GEMMA3_270M MIS | Δ | Improv% | Win? |
|:------|:-------:|:----------:|:----------------------:|:-:|:-------:|:----:|
| gemma-3-270m-it | BRA | 0.3655 | 0.4037 | -0.0382 | **-10.46%** | ❌ |
| gemma-3-270m-it | CHN | 0.4536 | 0.4603 | -0.0067 | **-1.47%** | ❌ |
| gemma-3-270m-it | DEU | 0.3289 | 0.4978 | -0.1689 | **-51.36%** | ❌ |
| gemma-3-270m-it | JPN | 0.4667 | 0.4348 | +0.0319 | **+6.84%** | ✅ |
| gemma-3-270m-it | USA | 0.6038 | 0.5066 | +0.0972 | **+16.10%** | ✅ |

- **gemma-3-270m-it** Win Rate: **2/5** | EXP-01=0.4437 → EXP-24-GEMMA3_270M=0.4606 | Macro Δ: **-3.82%**

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-GEMMA3_270M Leaderboard Entry

| Rank | Method | Coverage | Mean MIS ↓ | Notes |
|:---:|:-------|:--------:|-----------:|:------|
| ? | **EXP-24-GEMMA3_270M** | 1 models × 20 countries | **0.4781** | MIS↓ JSD=0.0547 r=+0.252 Flip=13.1% |

**DPBR summary:** Mean MIS=0.4781, r=+0.252, Flip=13.1%, rel_r=0.991 · *(EXP-09 SOTA: 0.3975 | EXP-24 multi-model ref: 0.3969)*

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-GEMMA3_270M Per-Dimension Worst Errors

| Model | Country | Worst Dim | Human | Model | Abs err (pp) |
|:------|:-------:|:----------|:-----:|:-----:|:------------:|
| gemma-3-270m-it | ARG | Utilitarianism_More | 75.0 | 50.3 | **24.7** |
| gemma-3-270m-it | BGD | Species_Humans | 78.5 | 51.2 | **27.3** |
| gemma-3-270m-it | BRA | Utilitarianism_More | 73.7 | 50.2 | **23.5** |
| gemma-3-270m-it | CHN | Species_Humans | 83.0 | 50.9 | **32.1** |
| gemma-3-270m-it | COL | Age_Young | 76.3 | 50.9 | **25.4** |
| gemma-3-270m-it | DEU | Species_Humans | 82.4 | 50.5 | **31.9** |
| gemma-3-270m-it | ETH | Species_Humans | 93.9 | 50.7 | **43.2** |
| gemma-3-270m-it | GBR | Species_Humans | 79.9 | 51.2 | **28.7** |
| gemma-3-270m-it | IDN | Species_Humans | 77.4 | 51.8 | **25.6** |
| gemma-3-270m-it | IRN | Species_Humans | 84.0 | 49.4 | **34.5** |
| gemma-3-270m-it | JPN | Species_Humans | 79.8 | 50.4 | **29.4** |
| gemma-3-270m-it | KGZ | Species_Humans | 78.7 | 50.2 | **28.5** |
| gemma-3-270m-it | MEX | Utilitarianism_More | 74.8 | 49.4 | **25.4** |
| gemma-3-270m-it | MMR | Utilitarianism_More | 78.7 | 48.9 | **29.8** |
| gemma-3-270m-it | MYS | Species_Humans | 76.6 | 51.0 | **25.7** |
| gemma-3-270m-it | ROU | Species_Humans | 80.1 | 50.9 | **29.2** |
| gemma-3-270m-it | SRB | Species_Humans | 77.7 | 51.0 | **26.7** |
| gemma-3-270m-it | THA | Species_Humans | 79.7 | 50.9 | **28.8** |
| gemma-3-270m-it | USA | Species_Humans | 79.2 | 50.9 | **28.3** |
| gemma-3-270m-it | VNM | Species_Humans | 77.7 | 50.6 | **27.1** |

---

#### EXP-24-GEMMA4_E2B Full Metrics

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| gemma-4-E2B-it | ARG | 0.4337 | 0.0504 | +0.213 | 16.03 | 9.0% |
| gemma-4-E2B-it | BGD | 0.4960 | 0.0516 | +0.435 | 18.67 | 8.4% |
| gemma-4-E2B-it | BRA | 0.4040 | 0.0741 | +0.211 | 13.82 | 14.2% |
| gemma-4-E2B-it | CHN | 0.3925 | 0.0393 | +0.767 | 14.60 | 5.5% |
| gemma-4-E2B-it | COL | 0.4598 | 0.0510 | -0.011 | 17.05 | 7.1% |
| gemma-4-E2B-it | DEU | 0.4156 | 0.0534 | +0.617 | 15.00 | 18.1% |
| gemma-4-E2B-it | ETH | 0.6123 | 0.0567 | +0.628 | 22.80 | 5.8% |
| gemma-4-E2B-it | GBR | 0.5020 | 0.0623 | +0.342 | 18.37 | 21.3% |
| gemma-4-E2B-it | IDN | 0.4431 | 0.0435 | +0.572 | 16.88 | 11.6% |
| gemma-4-E2B-it | IRN | 0.6300 | 0.0880 | +0.408 | 23.34 | 25.5% |
| gemma-4-E2B-it | JPN | 0.4595 | 0.0621 | -0.357 | 16.28 | 2.9% |
| gemma-4-E2B-it | KGZ | 0.4728 | 0.0489 | +0.600 | 17.63 | 10.3% |
| gemma-4-E2B-it | MEX | 0.4555 | 0.0409 | +0.503 | 17.43 | 5.5% |
| gemma-4-E2B-it | MMR | 0.4815 | 0.0611 | +0.372 | 17.22 | 2.3% |
| gemma-4-E2B-it | MYS | 0.4295 | 0.0386 | +0.676 | 16.42 | 6.1% |
| gemma-4-E2B-it | ROU | 0.4824 | 0.0455 | +0.644 | 18.25 | 4.8% |
| gemma-4-E2B-it | SRB | 0.5022 | 0.0470 | +0.573 | 19.06 | 5.5% |
| gemma-4-E2B-it | THA | 0.4363 | 0.0405 | +0.750 | 16.55 | 3.5% |
| gemma-4-E2B-it | USA | 0.5165 | 0.0598 | +0.618 | 19.65 | 18.1% |
| gemma-4-E2B-it | VNM | 0.5436 | 0.0481 | +0.566 | 21.17 | 30.0% |

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-GEMMA4_E2B vs Vanilla (MIS)

| Model | Country | Vanilla MIS | EXP-24-GEMMA4_E2B MIS | Δ | Improv% | Win? |
|:------|:-------:|:-----------:|:---------------------:|:-:|:-------:|:----:|
| gemma-4-E2B-it | ARG | 0.4234 | 0.4337 | -0.0103 | **-2.44%** | ❌ |
| gemma-4-E2B-it | BGD | 0.5187 | 0.4960 | +0.0227 | **+4.37%** | ✅ |
| gemma-4-E2B-it | BRA | 0.4160 | 0.4040 | +0.0120 | **+2.88%** | ✅ |
| gemma-4-E2B-it | CHN | 0.5000 | 0.3925 | +0.1074 | **+21.49%** | ✅ |
| gemma-4-E2B-it | COL | 0.4640 | 0.4598 | +0.0042 | **+0.91%** | ✅ |
| gemma-4-E2B-it | DEU | 0.5058 | 0.4156 | +0.0903 | **+17.85%** | ✅ |
| gemma-4-E2B-it | ETH | 0.6447 | 0.6123 | +0.0324 | **+5.02%** | ✅ |
| gemma-4-E2B-it | GBR | 0.5564 | 0.5020 | +0.0545 | **+9.79%** | ✅ |
| gemma-4-E2B-it | IDN | 0.4172 | 0.4431 | -0.0260 | **-6.22%** | ❌ |
| gemma-4-E2B-it | IRN | 0.5139 | 0.6300 | -0.1161 | **-22.59%** | ❌ |
| gemma-4-E2B-it | JPN | 0.4419 | 0.4595 | -0.0176 | **-3.98%** | ❌ |
| gemma-4-E2B-it | KGZ | 0.5181 | 0.4728 | +0.0453 | **+8.74%** | ✅ |
| gemma-4-E2B-it | MEX | 0.4301 | 0.4555 | -0.0254 | **-5.90%** | ❌ |
| gemma-4-E2B-it | MMR | 0.4883 | 0.4815 | +0.0068 | **+1.39%** | ✅ |
| gemma-4-E2B-it | MYS | 0.4838 | 0.4295 | +0.0543 | **+11.22%** | ✅ |
| gemma-4-E2B-it | ROU | 0.5468 | 0.4824 | +0.0643 | **+11.77%** | ✅ |
| gemma-4-E2B-it | SRB | 0.5481 | 0.5022 | +0.0459 | **+8.37%** | ✅ |
| gemma-4-E2B-it | THA | 0.4703 | 0.4363 | +0.0340 | **+7.23%** | ✅ |
| gemma-4-E2B-it | USA | 0.5452 | 0.5165 | +0.0287 | **+5.26%** | ✅ |
| gemma-4-E2B-it | VNM | 0.4684 | 0.5436 | -0.0752 | **-16.06%** | ❌ |

- **gemma-4-E2B-it** Win Rate: **14/20** | Vanilla=0.4951 → EXP-24-GEMMA4_E2B=0.4785 | Macro Δ: **+3.35%**

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-GEMMA4_E2B vs EXP-01 SWA-PTIS (MIS)

| Model | Country | EXP-01 MIS | EXP-24-GEMMA4_E2B MIS | Δ | Improv% | Win? |
|:------|:-------:|:----------:|:---------------------:|:-:|:-------:|:----:|
| gemma-4-E2B-it | BRA | 0.3655 | 0.4040 | -0.0385 | **-10.54%** | ❌ |
| gemma-4-E2B-it | CHN | 0.4536 | 0.3925 | +0.0611 | **+13.46%** | ✅ |
| gemma-4-E2B-it | DEU | 0.3289 | 0.4156 | -0.0867 | **-26.35%** | ❌ |
| gemma-4-E2B-it | JPN | 0.4667 | 0.4595 | +0.0072 | **+1.54%** | ✅ |
| gemma-4-E2B-it | USA | 0.6038 | 0.5165 | +0.0873 | **+14.45%** | ✅ |

- **gemma-4-E2B-it** Win Rate: **3/5** | EXP-01=0.4437 → EXP-24-GEMMA4_E2B=0.4376 | Macro Δ: **+1.37%**

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-GEMMA4_E2B Leaderboard Entry

| Rank | Method | Coverage | Mean MIS ↓ | Notes |
|:---:|:-------|:--------:|-----------:|:------|
| ? | **EXP-24-GEMMA4_E2B** | 1 models × 20 countries | **0.4785** | MIS↓ JSD=0.0531 r=+0.456 Flip=10.8% |

**DPBR summary:** Mean MIS=0.4785, r=+0.456, Flip=10.8%, rel_r=0.869 · *(EXP-09 SOTA: 0.3975 | EXP-24 multi-model ref: 0.3969)*

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-GEMMA4_E2B Per-Dimension Worst Errors

| Model | Country | Worst Dim | Human | Model | Abs err (pp) |
|:------|:-------:|:----------|:-----:|:-----:|:------------:|
| gemma-4-E2B-it | ARG | Age_Young | 75.1 | 48.0 | **27.2** |
| gemma-4-E2B-it | BGD | Utilitarianism_More | 75.0 | 47.5 | **27.4** |
| gemma-4-E2B-it | BRA | Age_Young | 73.6 | 43.7 | **29.9** |
| gemma-4-E2B-it | CHN | Species_Humans | 83.0 | 57.7 | **25.3** |
| gemma-4-E2B-it | COL | Age_Young | 76.3 | 48.8 | **27.5** |
| gemma-4-E2B-it | DEU | Age_Young | 73.9 | 46.9 | **27.1** |
| gemma-4-E2B-it | ETH | Species_Humans | 93.9 | 54.5 | **39.4** |
| gemma-4-E2B-it | GBR | Age_Young | 75.1 | 44.7 | **30.5** |
| gemma-4-E2B-it | IDN | Age_Young | 70.1 | 46.7 | **23.4** |
| gemma-4-E2B-it | IRN | SocialValue_High | 67.4 | 30.3 | **37.0** |
| gemma-4-E2B-it | JPN | Species_Humans | 79.8 | 47.2 | **32.6** |
| gemma-4-E2B-it | KGZ | Utilitarianism_More | 73.0 | 48.4 | **24.5** |
| gemma-4-E2B-it | MEX | Age_Young | 75.2 | 48.0 | **27.2** |
| gemma-4-E2B-it | MMR | Utilitarianism_More | 78.7 | 49.8 | **28.9** |
| gemma-4-E2B-it | MYS | Utilitarianism_More | 72.0 | 49.7 | **22.3** |
| gemma-4-E2B-it | ROU | Age_Young | 74.9 | 49.4 | **25.6** |
| gemma-4-E2B-it | SRB | Age_Young | 76.0 | 50.0 | **25.9** |
| gemma-4-E2B-it | THA | Species_Humans | 79.7 | 55.5 | **24.2** |
| gemma-4-E2B-it | USA | Age_Young | 74.5 | 42.5 | **32.1** |
| gemma-4-E2B-it | VNM | SocialValue_High | 69.0 | 40.4 | **28.6** |

---

#### EXP-24-QWEN3_VL_8B Full Metrics

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Qwen3-VL-8B | ARG | 0.3385 | 0.0832 | +0.508 | 10.14 | 22.3% |
| Qwen3-VL-8B | BGD | 0.5870 | 0.1334 | -0.116 | 20.56 | 19.0% |
| Qwen3-VL-8B | BRA | 0.3751 | 0.1013 | +0.212 | 11.79 | 19.0% |
| Qwen3-VL-8B | CHN | 0.4155 | 0.1207 | +0.269 | 12.41 | 17.7% |
| Qwen3-VL-8B | COL | 0.3904 | 0.0928 | +0.161 | 12.21 | 19.7% |
| Qwen3-VL-8B | DEU | 0.3064 | 0.0813 | +0.568 | 9.04 | 17.4% |
| Qwen3-VL-8B | ETH | 0.6842 | 0.1410 | -0.095 | 23.00 | 19.4% |
| Qwen3-VL-8B | GBR | 0.5238 | 0.1277 | +0.230 | 18.40 | 18.1% |
| Qwen3-VL-8B | IDN | 0.4020 | 0.0913 | +0.318 | 13.11 | 18.7% |
| Qwen3-VL-8B | IRN | 0.5064 | 0.0900 | +0.057 | 17.19 | 21.3% |
| Qwen3-VL-8B | JPN | 0.3164 | 0.0902 | +0.453 | 10.00 | 15.5% |
| Qwen3-VL-8B | KGZ | 0.5743 | 0.1329 | -0.017 | 19.22 | 18.4% |
| Qwen3-VL-8B | MEX | 0.3501 | 0.0807 | +0.529 | 9.63 | 21.6% |
| Qwen3-VL-8B | MMR | 0.5632 | 0.1389 | -0.066 | 18.13 | 20.0% |
| Qwen3-VL-8B | MYS | 0.5045 | 0.1275 | +0.097 | 16.42 | 20.3% |
| Qwen3-VL-8B | ROU | 0.5349 | 0.1235 | +0.162 | 19.05 | 21.3% |
| Qwen3-VL-8B | SRB | 0.5666 | 0.1268 | +0.075 | 19.99 | 18.1% |
| Qwen3-VL-8B | THA | 0.4796 | 0.1122 | +0.231 | 16.64 | 17.7% |
| Qwen3-VL-8B | USA | 0.4874 | 0.1097 | +0.422 | 16.91 | 17.4% |
| Qwen3-VL-8B | VNM | 0.4199 | 0.0873 | +0.339 | 12.78 | 21.6% |

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-QWEN3_VL_8B vs Vanilla (MIS)

| Model | Country | Vanilla MIS | EXP-24-QWEN3_VL_8B MIS | Δ | Improv% | Win? |
|:------|:-------:|:-----------:|:----------------------:|:-:|:-------:|:----:|
| Qwen3-VL-8B | ARG | 0.5204 | 0.3385 | +0.1819 | **+34.96%** | ✅ |
| Qwen3-VL-8B | BGD | 0.6738 | 0.5870 | +0.0868 | **+12.88%** | ✅ |
| Qwen3-VL-8B | BRA | 0.5288 | 0.3751 | +0.1537 | **+29.07%** | ✅ |
| Qwen3-VL-8B | CHN | 0.5693 | 0.4155 | +0.1538 | **+27.01%** | ✅ |
| Qwen3-VL-8B | COL | 0.5589 | 0.3904 | +0.1685 | **+30.15%** | ✅ |
| Qwen3-VL-8B | DEU | 0.3181 | 0.3064 | +0.0117 | **+3.69%** | ✅ |
| Qwen3-VL-8B | ETH | 0.7235 | 0.6842 | +0.0393 | **+5.43%** | ✅ |
| Qwen3-VL-8B | GBR | 0.6767 | 0.5238 | +0.1529 | **+22.59%** | ✅ |
| Qwen3-VL-8B | IDN | 0.4081 | 0.4020 | +0.0061 | **+1.49%** | ✅ |
| Qwen3-VL-8B | IRN | 0.5072 | 0.5064 | +0.0009 | **+0.17%** | ✅ |
| Qwen3-VL-8B | JPN | 0.4075 | 0.3164 | +0.0912 | **+22.37%** | ✅ |
| Qwen3-VL-8B | KGZ | 0.6610 | 0.5743 | +0.0867 | **+13.12%** | ✅ |
| Qwen3-VL-8B | MEX | 0.5171 | 0.3501 | +0.1670 | **+32.29%** | ✅ |
| Qwen3-VL-8B | MMR | 0.6948 | 0.5632 | +0.1315 | **+18.93%** | ✅ |
| Qwen3-VL-8B | MYS | 0.6560 | 0.5045 | +0.1516 | **+23.10%** | ✅ |
| Qwen3-VL-8B | ROU | 0.6591 | 0.5349 | +0.1242 | **+18.84%** | ✅ |
| Qwen3-VL-8B | SRB | 0.6759 | 0.5666 | +0.1093 | **+16.17%** | ✅ |
| Qwen3-VL-8B | THA | 0.6191 | 0.4796 | +0.1395 | **+22.54%** | ✅ |
| Qwen3-VL-8B | USA | 0.6767 | 0.4874 | +0.1892 | **+27.97%** | ✅ |
| Qwen3-VL-8B | VNM | 0.4445 | 0.4199 | +0.0246 | **+5.53%** | ✅ |

- **Qwen3-VL-8B** Win Rate: **20/20** | Vanilla=0.5748 → EXP-24-QWEN3_VL_8B=0.4663 | Macro Δ: **+18.88%**

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-QWEN3_VL_8B vs EXP-01 SWA-PTIS (MIS)

| Model | Country | EXP-01 MIS | EXP-24-QWEN3_VL_8B MIS | Δ | Improv% | Win? |
|:------|:-------:|:----------:|:----------------------:|:-:|:-------:|:----:|
| Qwen3-VL-8B | BRA | 0.4025 | 0.3751 | +0.0274 | **+6.82%** | ✅ |
| Qwen3-VL-8B | CHN | 0.4078 | 0.4155 | -0.0077 | **-1.90%** | ❌ |
| Qwen3-VL-8B | DEU | 0.3424 | 0.3064 | +0.0360 | **+10.52%** | ✅ |
| Qwen3-VL-8B | JPN | 0.2802 | 0.3164 | -0.0362 | **-12.91%** | ❌ |
| Qwen3-VL-8B | USA | 0.3677 | 0.4874 | -0.1197 | **-32.56%** | ❌ |

- **Qwen3-VL-8B** Win Rate: **2/5** | EXP-01=0.3601 → EXP-24-QWEN3_VL_8B=0.3802 | Macro Δ: **-5.56%**

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-QWEN3_VL_8B Leaderboard Entry

| Rank | Method | Coverage | Mean MIS ↓ | Notes |
|:---:|:-------|:--------:|-----------:|:------|
| ? | **EXP-24-QWEN3_VL_8B** | 1 models × 20 countries | **0.4663** | MIS↓ JSD=0.1096 r=+0.217 Flip=19.2% |

**DPBR summary:** Mean MIS=0.4663, r=+0.217, Flip=19.2%, rel_r=0.895 · *(EXP-09 SOTA: 0.3975 | EXP-24 multi-model ref: 0.3969)*

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-QWEN3_VL_8B Per-Dimension Worst Errors

| Model | Country | Worst Dim | Human | Model | Abs err (pp) |
|:------|:-------:|:----------|:-----:|:-----:|:------------:|
| Qwen3-VL-8B | ARG | SocialValue_High | 68.5 | 38.5 | **30.1** |
| Qwen3-VL-8B | BGD | SocialValue_High | 68.1 | 29.2 | **38.9** |
| Qwen3-VL-8B | BRA | SocialValue_High | 66.3 | 33.8 | **32.6** |
| Qwen3-VL-8B | CHN | SocialValue_High | 66.7 | 29.8 | **36.9** |
| Qwen3-VL-8B | COL | SocialValue_High | 72.7 | 39.1 | **33.6** |
| Qwen3-VL-8B | DEU | SocialValue_High | 64.7 | 38.0 | **26.8** |
| Qwen3-VL-8B | ETH | Utilitarianism_More | 80.0 | 38.8 | **41.2** |
| Qwen3-VL-8B | GBR | SocialValue_High | 67.7 | 28.8 | **38.9** |
| Qwen3-VL-8B | IDN | SocialValue_High | 69.0 | 34.6 | **34.4** |
| Qwen3-VL-8B | IRN | SocialValue_High | 67.4 | 38.9 | **28.5** |
| Qwen3-VL-8B | JPN | SocialValue_High | 65.9 | 39.3 | **26.6** |
| Qwen3-VL-8B | KGZ | SocialValue_High | 68.7 | 28.6 | **40.1** |
| Qwen3-VL-8B | MEX | SocialValue_High | 69.6 | 38.7 | **30.8** |
| Qwen3-VL-8B | MMR | Utilitarianism_More | 78.7 | 38.4 | **40.3** |
| Qwen3-VL-8B | MYS | SocialValue_High | 67.4 | 27.8 | **39.7** |
| Qwen3-VL-8B | ROU | SocialValue_High | 67.6 | 28.9 | **38.7** |
| Qwen3-VL-8B | SRB | SocialValue_High | 67.3 | 29.4 | **37.9** |
| Qwen3-VL-8B | THA | SocialValue_High | 65.1 | 30.5 | **34.6** |
| Qwen3-VL-8B | USA | SocialValue_High | 67.9 | 30.2 | **37.7** |
| Qwen3-VL-8B | VNM | SocialValue_High | 69.0 | 38.2 | **30.8** |

---

#### EXP-24-QWEN3_4B_THINKING_2507 Full Metrics

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Qwen3-4B-Thinking-2507 | ARG | 0.4565 | 0.0452 | +0.568 | 17.22 | 63.5% |
| Qwen3-4B-Thinking-2507 | BGD | 0.4719 | 0.0572 | -0.267 | 17.09 | 13.5% |
| Qwen3-4B-Thinking-2507 | BRA | 0.4193 | 0.0468 | +0.518 | 15.54 | 37.4% |
| Qwen3-4B-Thinking-2507 | CHN | 0.4840 | 0.0601 | -0.341 | 17.35 | 13.9% |
| Qwen3-4B-Thinking-2507 | COL | 0.4788 | 0.0440 | +0.813 | 18.22 | 59.7% |
| Qwen3-4B-Thinking-2507 | DEU | 0.4977 | 0.0586 | +0.569 | 17.96 | 16.8% |
| Qwen3-4B-Thinking-2507 | ETH | 0.6114 | 0.0709 | -0.641 | 21.80 | 12.9% |
| Qwen3-4B-Thinking-2507 | GBR | 0.5193 | 0.0596 | -0.372 | 18.98 | 30.6% |
| Qwen3-4B-Thinking-2507 | IDN | 0.4652 | 0.0510 | +0.051 | 17.26 | 17.1% |
| Qwen3-4B-Thinking-2507 | IRN | 0.5166 | 0.0681 | -0.004 | 18.12 | 14.2% |
| Qwen3-4B-Thinking-2507 | JPN | 0.4492 | 0.0556 | -0.174 | 16.20 | 71.0% |
| Qwen3-4B-Thinking-2507 | KGZ | 0.4781 | 0.0577 | -0.053 | 17.31 | 6.1% |
| Qwen3-4B-Thinking-2507 | MEX | 0.4788 | 0.0458 | +0.633 | 18.14 | 51.6% |
| Qwen3-4B-Thinking-2507 | MMR | 0.4662 | 0.0667 | -0.167 | 16.05 | 21.0% |
| Qwen3-4B-Thinking-2507 | MYS | 0.4362 | 0.0515 | -0.229 | 15.96 | 21.6% |
| Qwen3-4B-Thinking-2507 | ROU | 0.4978 | 0.0591 | -0.305 | 18.07 | 14.5% |
| Qwen3-4B-Thinking-2507 | SRB | 0.4957 | 0.0557 | +0.070 | 18.22 | 16.5% |
| Qwen3-4B-Thinking-2507 | THA | 0.4352 | 0.0590 | -0.525 | 15.31 | 33.9% |
| Qwen3-4B-Thinking-2507 | USA | 0.5085 | 0.0567 | -0.286 | 18.71 | 21.6% |
| Qwen3-4B-Thinking-2507 | VNM | 0.4936 | 0.0541 | -0.418 | 18.34 | 7.7% |

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-QWEN3_4B_THINKING_2507 vs Vanilla (MIS)

| Model | Country | Vanilla MIS | EXP-24-QWEN3_4B_THINKING_2507 MIS | Δ | Improv% | Win? |
|:------|:-------:|:-----------:|:--------------------------------:|:-:|:-------:|:----:|
| Qwen3-4B-Thinking-2507 | ARG | 0.4377 | 0.4565 | -0.0189 | **-4.31%** | ❌ |
| Qwen3-4B-Thinking-2507 | BGD | 0.4974 | 0.4719 | +0.0255 | **+5.13%** | ✅ |
| Qwen3-4B-Thinking-2507 | BRA | 0.3851 | 0.4193 | -0.0342 | **-8.88%** | ❌ |
| Qwen3-4B-Thinking-2507 | CHN | 0.4685 | 0.4840 | -0.0155 | **-3.31%** | ❌ |
| Qwen3-4B-Thinking-2507 | COL | 0.4600 | 0.4788 | -0.0188 | **-4.09%** | ❌ |
| Qwen3-4B-Thinking-2507 | DEU | 0.5099 | 0.4977 | +0.0123 | **+2.40%** | ✅ |
| Qwen3-4B-Thinking-2507 | ETH | 0.6309 | 0.6114 | +0.0195 | **+3.09%** | ✅ |
| Qwen3-4B-Thinking-2507 | GBR | 0.5360 | 0.5193 | +0.0167 | **+3.11%** | ✅ |
| Qwen3-4B-Thinking-2507 | IDN | 0.4493 | 0.4652 | -0.0159 | **-3.54%** | ❌ |
| Qwen3-4B-Thinking-2507 | IRN | 0.5143 | 0.5166 | -0.0023 | **-0.46%** | ❌ |
| Qwen3-4B-Thinking-2507 | JPN | 0.4418 | 0.4492 | -0.0074 | **-1.68%** | ❌ |
| Qwen3-4B-Thinking-2507 | KGZ | 0.5028 | 0.4781 | +0.0247 | **+4.92%** | ✅ |
| Qwen3-4B-Thinking-2507 | MEX | 0.4633 | 0.4788 | -0.0155 | **-3.35%** | ❌ |
| Qwen3-4B-Thinking-2507 | MMR | 0.4847 | 0.4662 | +0.0186 | **+3.83%** | ✅ |
| Qwen3-4B-Thinking-2507 | MYS | 0.4617 | 0.4362 | +0.0255 | **+5.53%** | ✅ |
| Qwen3-4B-Thinking-2507 | ROU | 0.5216 | 0.4978 | +0.0239 | **+4.58%** | ✅ |
| Qwen3-4B-Thinking-2507 | SRB | 0.5203 | 0.4957 | +0.0247 | **+4.74%** | ✅ |
| Qwen3-4B-Thinking-2507 | THA | 0.4571 | 0.4352 | +0.0218 | **+4.78%** | ✅ |
| Qwen3-4B-Thinking-2507 | USA | 0.5272 | 0.5085 | +0.0188 | **+3.56%** | ✅ |
| Qwen3-4B-Thinking-2507 | VNM | 0.4825 | 0.4936 | -0.0112 | **-2.31%** | ❌ |

- **Qwen3-4B-Thinking-2507** Win Rate: **11/20** | Vanilla=0.4876 → EXP-24-QWEN3_4B_THINKING_2507=0.4830 | Macro Δ: **+0.95%**

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-QWEN3_4B_THINKING_2507 vs EXP-01 SWA-PTIS (MIS)

| Model | Country | EXP-01 MIS | EXP-24-QWEN3_4B_THINKING_2507 MIS | Δ | Improv% | Win? |
|:------|:-------:|:----------:|:--------------------------------:|:-:|:-------:|:----:|
| Qwen3-4B-Thinking-2507 | BRA | 0.4025 | 0.4193 | -0.0168 | **-4.18%** | ❌ |
| Qwen3-4B-Thinking-2507 | CHN | 0.4078 | 0.4840 | -0.0762 | **-18.69%** | ❌ |
| Qwen3-4B-Thinking-2507 | DEU | 0.3424 | 0.4977 | -0.1553 | **-45.35%** | ❌ |
| Qwen3-4B-Thinking-2507 | JPN | 0.2802 | 0.4492 | -0.1690 | **-60.33%** | ❌ |
| Qwen3-4B-Thinking-2507 | USA | 0.3677 | 0.5085 | -0.1408 | **-38.28%** | ❌ |

- **Qwen3-4B-Thinking-2507** Win Rate: **0/5** | EXP-01=0.3601 → EXP-24-QWEN3_4B_THINKING_2507=0.4717 | Macro Δ: **-31.00%**

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-QWEN3_4B_THINKING_2507 Leaderboard Entry

| Rank | Method | Coverage | Mean MIS ↓ | Notes |
|:---:|:-------|:--------:|-----------:|:------|
| ? | **EXP-24-QWEN3_4B_THINKING_2507** | 1 models × 20 countries | **0.4830** | MIS↓ JSD=0.0562 r=-0.028 Flip=27.3% |

**DPBR summary:** Mean MIS=0.4830, r=-0.028, Flip=27.3%, rel_r=0.992 · *(EXP-09 SOTA: 0.3975 | EXP-24 multi-model ref: 0.3969)*

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-QWEN3_4B_THINKING_2507 Per-Dimension Worst Errors

| Model | Country | Worst Dim | Human | Model | Abs err (pp) |
|:------|:-------:|:----------|:-----:|:-----:|:------------:|
| Qwen3-4B-Thinking-2507 | ARG | Age_Young | 75.1 | 49.8 | **25.3** |
| Qwen3-4B-Thinking-2507 | BGD | Species_Humans | 78.5 | 49.2 | **29.3** |
| Qwen3-4B-Thinking-2507 | BRA | Utilitarianism_More | 73.7 | 49.9 | **23.8** |
| Qwen3-4B-Thinking-2507 | CHN | Species_Humans | 83.0 | 49.3 | **33.7** |
| Qwen3-4B-Thinking-2507 | COL | Age_Young | 76.3 | 50.1 | **26.2** |
| Qwen3-4B-Thinking-2507 | DEU | Species_Humans | 82.4 | 50.4 | **32.1** |
| Qwen3-4B-Thinking-2507 | ETH | Species_Humans | 93.9 | 49.2 | **44.7** |
| Qwen3-4B-Thinking-2507 | GBR | Species_Humans | 79.9 | 48.6 | **31.2** |
| Qwen3-4B-Thinking-2507 | IDN | Species_Humans | 77.4 | 49.7 | **27.7** |
| Qwen3-4B-Thinking-2507 | IRN | Species_Humans | 84.0 | 49.5 | **34.5** |
| Qwen3-4B-Thinking-2507 | JPN | Species_Humans | 79.8 | 49.8 | **30.0** |
| Qwen3-4B-Thinking-2507 | KGZ | Species_Humans | 78.7 | 49.3 | **29.4** |
| Qwen3-4B-Thinking-2507 | MEX | Age_Young | 75.2 | 50.0 | **25.2** |
| Qwen3-4B-Thinking-2507 | MMR | Utilitarianism_More | 78.7 | 49.8 | **28.9** |
| Qwen3-4B-Thinking-2507 | MYS | Species_Humans | 76.6 | 49.3 | **27.3** |
| Qwen3-4B-Thinking-2507 | ROU | Species_Humans | 80.1 | 49.2 | **31.0** |
| Qwen3-4B-Thinking-2507 | SRB | Species_Humans | 77.7 | 49.3 | **28.5** |
| Qwen3-4B-Thinking-2507 | THA | Species_Humans | 79.7 | 49.2 | **30.4** |
| Qwen3-4B-Thinking-2507 | USA | Species_Humans | 79.2 | 48.9 | **30.3** |
| Qwen3-4B-Thinking-2507 | VNM | Species_Humans | 77.7 | 48.7 | **29.0** |

---

#### EXP-24-HF_QWEN25_7B_BF16 Full Metrics

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Qwen2.5-7B | ARG | 0.3767 | 0.0635 | +0.196 | 12.65 | 14.8% |
| Qwen2.5-7B | BGD | 0.2842 | 0.0548 | +0.695 | 8.67 | 21.3% |
| Qwen2.5-7B | BRA | 0.4107 | 0.0814 | -0.205 | 13.93 | 11.9% |
| Qwen2.5-7B | CHN | 0.3557 | 0.0840 | +0.405 | 11.50 | 21.6% |
| Qwen2.5-7B | COL | 0.4174 | 0.0704 | -0.096 | 13.76 | 15.5% |
| Qwen2.5-7B | DEU | 0.4147 | 0.0687 | +0.144 | 14.31 | 18.1% |
| Qwen2.5-7B | ETH | 0.4376 | 0.0587 | +0.676 | 15.83 | 14.8% |
| Qwen2.5-7B | GBR | 0.3985 | 0.0819 | +0.620 | 12.66 | 22.9% |
| Qwen2.5-7B | IDN | 0.4023 | 0.0584 | +0.251 | 14.16 | 15.2% |
| Qwen2.5-7B | IRN | 0.4665 | 0.0674 | +0.252 | 15.91 | 8.4% |
| Qwen2.5-7B | JPN | 0.3560 | 0.0474 | +0.535 | 12.60 | 17.1% |
| Qwen2.5-7B | KGZ | 0.2492 | 0.0394 | +0.829 | 8.52 | 21.9% |
| Qwen2.5-7B | MEX | 0.3875 | 0.0617 | +0.252 | 13.23 | 14.8% |
| Qwen2.5-7B | MMR | 0.3024 | 0.0624 | +0.590 | 9.90 | 18.4% |
| Qwen2.5-7B | MYS | 0.2739 | 0.0496 | +0.763 | 9.18 | 18.4% |
| Qwen2.5-7B | ROU | 0.3303 | 0.0507 | +0.785 | 11.76 | 19.0% |
| Qwen2.5-7B | SRB | 0.3031 | 0.0506 | +0.773 | 10.26 | 18.1% |
| Qwen2.5-7B | THA | 0.2214 | 0.0459 | +0.841 | 6.57 | 20.6% |
| Qwen2.5-7B | USA | 0.3828 | 0.0749 | +0.709 | 12.56 | 21.9% |
| Qwen2.5-7B | VNM | 0.4678 | 0.0603 | -0.018 | 16.91 | 10.0% |

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-HF_QWEN25_7B_BF16 vs Vanilla (MIS)

| Model | Country | Vanilla MIS | EXP-24-HF_QWEN25_7B_BF16 MIS | Δ | Improv% | Win? |
|:------|:-------:|:-----------:|:----------------------------:|:-:|:-------:|:----:|
| Qwen2.5-7B | ARG | 0.4693 | 0.3767 | +0.0926 | **+19.74%** | ✅ |
| Qwen2.5-7B | BGD | 0.4607 | 0.2842 | +0.1765 | **+38.30%** | ✅ |
| Qwen2.5-7B | BRA | 0.5795 | 0.4107 | +0.1687 | **+29.12%** | ✅ |
| Qwen2.5-7B | CHN | 0.4370 | 0.3557 | +0.0814 | **+18.62%** | ✅ |
| Qwen2.5-7B | COL | 0.5079 | 0.4174 | +0.0905 | **+17.81%** | ✅ |
| Qwen2.5-7B | DEU | 0.3081 | 0.4147 | -0.1065 | **-34.57%** | ❌ |
| Qwen2.5-7B | ETH | 0.5581 | 0.4376 | +0.1205 | **+21.59%** | ✅ |
| Qwen2.5-7B | GBR | 0.4827 | 0.3985 | +0.0842 | **+17.44%** | ✅ |
| Qwen2.5-7B | IDN | 0.4798 | 0.4023 | +0.0775 | **+16.14%** | ✅ |
| Qwen2.5-7B | IRN | 0.3730 | 0.4665 | -0.0934 | **-25.05%** | ❌ |
| Qwen2.5-7B | JPN | 0.2964 | 0.3560 | -0.0596 | **-20.12%** | ❌ |
| Qwen2.5-7B | KGZ | 0.4430 | 0.2492 | +0.1938 | **+43.75%** | ✅ |
| Qwen2.5-7B | MEX | 0.4668 | 0.3875 | +0.0793 | **+16.98%** | ✅ |
| Qwen2.5-7B | MMR | 0.4605 | 0.3024 | +0.1581 | **+34.33%** | ✅ |
| Qwen2.5-7B | MYS | 0.4451 | 0.2739 | +0.1712 | **+38.46%** | ✅ |
| Qwen2.5-7B | ROU | 0.4416 | 0.3303 | +0.1113 | **+25.20%** | ✅ |
| Qwen2.5-7B | SRB | 0.4534 | 0.3031 | +0.1502 | **+33.13%** | ✅ |
| Qwen2.5-7B | THA | 0.4142 | 0.2214 | +0.1928 | **+46.56%** | ✅ |
| Qwen2.5-7B | USA | 0.4833 | 0.3828 | +0.1006 | **+20.81%** | ✅ |
| Qwen2.5-7B | VNM | 0.5008 | 0.4678 | +0.0329 | **+6.58%** | ✅ |

- **Qwen2.5-7B** Win Rate: **17/20** | Vanilla=0.4531 → EXP-24-HF_QWEN25_7B_BF16=0.3619 | Macro Δ: **+20.11%**

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-HF_QWEN25_7B_BF16 vs EXP-01 SWA-PTIS (MIS)

| Model | Country | EXP-01 MIS | EXP-24-HF_QWEN25_7B_BF16 MIS | Δ | Improv% | Win? |
|:------|:-------:|:----------:|:----------------------------:|:-:|:-------:|:----:|
| Qwen2.5-7B | BRA | 0.4025 | 0.4107 | -0.0082 | **-2.05%** | ❌ |
| Qwen2.5-7B | CHN | 0.4078 | 0.3557 | +0.0521 | **+12.78%** | ✅ |
| Qwen2.5-7B | DEU | 0.3424 | 0.4147 | -0.0723 | **-21.10%** | ❌ |
| Qwen2.5-7B | JPN | 0.2802 | 0.3560 | -0.0758 | **-27.07%** | ❌ |
| Qwen2.5-7B | USA | 0.3677 | 0.3828 | -0.0151 | **-4.09%** | ❌ |

- **Qwen2.5-7B** Win Rate: **1/5** | EXP-01=0.3601 → EXP-24-HF_QWEN25_7B_BF16=0.3840 | Macro Δ: **-6.62%**

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-HF_QWEN25_7B_BF16 Leaderboard Entry

| Rank | Method | Coverage | Mean MIS ↓ | Notes |
|:---:|:-------|:--------:|-----------:|:------|
| ? | **EXP-24-HF_QWEN25_7B_BF16** | 1 models × 20 countries | **0.3619** | MIS↓ JSD=0.0616 r=+0.450 Flip=17.2% |

**DPBR summary:** Mean MIS=0.3619, r=+0.450, Flip=17.2%, rel_r=0.892 · *(EXP-09 SOTA: 0.3975 | EXP-24 multi-model ref: 0.3969)*

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-HF_QWEN25_7B_BF16 Per-Dimension Worst Errors

| Model | Country | Worst Dim | Human | Model | Abs err (pp) |
|:------|:-------:|:----------|:-----:|:-----:|:------------:|
| Qwen2.5-7B | ARG | SocialValue_High | 68.5 | 43.6 | **24.9** |
| Qwen2.5-7B | BGD | SocialValue_High | 68.1 | 47.1 | **21.0** |
| Qwen2.5-7B | BRA | Age_Young | 73.6 | 49.7 | **23.9** |
| Qwen2.5-7B | CHN | SocialValue_High | 66.7 | 37.9 | **28.9** |
| Qwen2.5-7B | COL | SocialValue_High | 72.7 | 44.1 | **28.7** |
| Qwen2.5-7B | DEU | Species_Humans | 82.4 | 56.5 | **25.9** |
| Qwen2.5-7B | ETH | Utilitarianism_More | 80.0 | 56.4 | **23.6** |
| Qwen2.5-7B | GBR | SocialValue_High | 67.7 | 35.2 | **32.5** |
| Qwen2.5-7B | IDN | SocialValue_High | 69.0 | 43.9 | **25.1** |
| Qwen2.5-7B | IRN | Species_Humans | 84.0 | 53.9 | **30.1** |
| Qwen2.5-7B | JPN | Species_Humans | 79.8 | 58.8 | **21.0** |
| Qwen2.5-7B | KGZ | SocialValue_High | 68.7 | 49.8 | **18.9** |
| Qwen2.5-7B | MEX | SocialValue_High | 69.6 | 43.9 | **25.6** |
| Qwen2.5-7B | MMR | Utilitarianism_More | 78.7 | 56.2 | **22.5** |
| Qwen2.5-7B | MYS | SocialValue_High | 67.4 | 46.7 | **20.7** |
| Qwen2.5-7B | ROU | SocialValue_High | 67.6 | 43.7 | **23.8** |
| Qwen2.5-7B | SRB | SocialValue_High | 67.3 | 46.0 | **21.3** |
| Qwen2.5-7B | THA | SocialValue_High | 65.1 | 47.2 | **17.9** |
| Qwen2.5-7B | USA | SocialValue_High | 67.9 | 36.9 | **31.0** |
| Qwen2.5-7B | VNM | SocialValue_High | 69.0 | 44.6 | **24.4** |

---

#### EXP-24-QWEN25_7B Full Metrics

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Qwen2.5-7B | ARG | 0.4556 | 0.0468 | -0.194 | 17.11 | 11.9% |
| Qwen2.5-7B | BGD | 0.4750 | 0.0555 | +0.126 | 17.34 | 3.9% |
| Qwen2.5-7B | BRA | 0.4153 | 0.0485 | -0.137 | 15.27 | 43.2% |
| Qwen2.5-7B | CHN | 0.4691 | 0.0590 | -0.147 | 16.73 | 20.6% |
| Qwen2.5-7B | COL | 0.4789 | 0.0462 | -0.135 | 18.12 | 7.4% |
| Qwen2.5-7B | DEU | 0.5041 | 0.0626 | +0.249 | 18.00 | 11.3% |
| Qwen2.5-7B | ETH | 0.6062 | 0.0673 | +0.312 | 21.79 | 2.3% |
| Qwen2.5-7B | GBR | 0.5147 | 0.0568 | +0.755 | 18.93 | 25.8% |
| Qwen2.5-7B | IDN | 0.4602 | 0.0510 | -0.021 | 17.03 | 56.1% |
| Qwen2.5-7B | IRN | 0.5138 | 0.0677 | +0.333 | 18.02 | 59.0% |
| Qwen2.5-7B | JPN | 0.4408 | 0.0548 | +0.203 | 15.85 | 5.5% |
| Qwen2.5-7B | KGZ | 0.4694 | 0.0556 | +0.749 | 17.04 | 4.5% |
| Qwen2.5-7B | MEX | 0.4786 | 0.0476 | -0.201 | 18.04 | 5.5% |
| Qwen2.5-7B | MMR | 0.4643 | 0.0656 | +0.148 | 16.04 | 4.2% |
| Qwen2.5-7B | MYS | 0.4389 | 0.0499 | +0.304 | 16.18 | 4.5% |
| Qwen2.5-7B | ROU | 0.4979 | 0.0561 | +0.541 | 18.25 | 1.9% |
| Qwen2.5-7B | SRB | 0.4982 | 0.0553 | +0.423 | 18.36 | 6.1% |
| Qwen2.5-7B | THA | 0.4326 | 0.0562 | +0.355 | 15.38 | 5.5% |
| Qwen2.5-7B | USA | 0.5050 | 0.0542 | +0.726 | 18.69 | 18.1% |
| Qwen2.5-7B | VNM | 0.4802 | 0.0503 | -0.347 | 17.93 | 22.6% |

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-QWEN25_7B vs Vanilla (MIS)

| Model | Country | Vanilla MIS | EXP-24-QWEN25_7B MIS | Δ | Improv% | Win? |
|:------|:-------:|:-----------:|:--------------------:|:-:|:-------:|:----:|
| Qwen2.5-7B | ARG | 0.4517 | 0.4556 | -0.0040 | **-0.88%** | ❌ |
| Qwen2.5-7B | BGD | 0.4794 | 0.4750 | +0.0044 | **+0.92%** | ✅ |
| Qwen2.5-7B | BRA | 0.4171 | 0.4153 | +0.0018 | **+0.43%** | ✅ |
| Qwen2.5-7B | CHN | 0.4674 | 0.4691 | -0.0017 | **-0.36%** | ❌ |
| Qwen2.5-7B | COL | 0.4745 | 0.4789 | -0.0044 | **-0.93%** | ❌ |
| Qwen2.5-7B | DEU | 0.5045 | 0.5041 | +0.0004 | **+0.08%** | ✅ |
| Qwen2.5-7B | ETH | 0.6118 | 0.6062 | +0.0055 | **+0.91%** | ✅ |
| Qwen2.5-7B | GBR | 0.5177 | 0.5147 | +0.0030 | **+0.58%** | ✅ |
| Qwen2.5-7B | IDN | 0.4604 | 0.4602 | +0.0002 | **+0.04%** | ✅ |
| Qwen2.5-7B | IRN | 0.5144 | 0.5138 | +0.0006 | **+0.12%** | ✅ |
| Qwen2.5-7B | JPN | 0.4417 | 0.4408 | +0.0009 | **+0.21%** | ✅ |
| Qwen2.5-7B | KGZ | 0.4846 | 0.4694 | +0.0152 | **+3.15%** | ✅ |
| Qwen2.5-7B | MEX | 0.4752 | 0.4786 | -0.0035 | **-0.73%** | ❌ |
| Qwen2.5-7B | MMR | 0.4688 | 0.4643 | +0.0044 | **+0.95%** | ✅ |
| Qwen2.5-7B | MYS | 0.4436 | 0.4389 | +0.0047 | **+1.07%** | ✅ |
| Qwen2.5-7B | ROU | 0.5035 | 0.4979 | +0.0056 | **+1.11%** | ✅ |
| Qwen2.5-7B | SRB | 0.5032 | 0.4982 | +0.0049 | **+0.98%** | ✅ |
| Qwen2.5-7B | THA | 0.4372 | 0.4326 | +0.0046 | **+1.05%** | ✅ |
| Qwen2.5-7B | USA | 0.5090 | 0.5050 | +0.0041 | **+0.80%** | ✅ |
| Qwen2.5-7B | VNM | 0.4806 | 0.4802 | +0.0003 | **+0.07%** | ✅ |

- **Qwen2.5-7B** Win Rate: **16/20** | Vanilla=0.4823 → EXP-24-QWEN25_7B=0.4800 | Macro Δ: **+0.49%**

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-QWEN25_7B vs EXP-01 SWA-PTIS (MIS)

| Model | Country | EXP-01 MIS | EXP-24-QWEN25_7B MIS | Δ | Improv% | Win? |
|:------|:-------:|:----------:|:--------------------:|:-:|:-------:|:----:|
| Qwen2.5-7B | BRA | 0.4025 | 0.4153 | -0.0128 | **-3.19%** | ❌ |
| Qwen2.5-7B | CHN | 0.4078 | 0.4691 | -0.0613 | **-15.03%** | ❌ |
| Qwen2.5-7B | DEU | 0.3424 | 0.5041 | -0.1617 | **-47.22%** | ❌ |
| Qwen2.5-7B | JPN | 0.2802 | 0.4408 | -0.1606 | **-57.32%** | ❌ |
| Qwen2.5-7B | USA | 0.3677 | 0.5050 | -0.1373 | **-37.34%** | ❌ |

- **Qwen2.5-7B** Win Rate: **0/5** | EXP-01=0.3601 → EXP-24-QWEN25_7B=0.4669 | Macro Δ: **-29.64%**

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-QWEN25_7B Leaderboard Entry

| Rank | Method | Coverage | Mean MIS ↓ | Notes |
|:---:|:-------|:--------:|-----------:|:------|
| ? | **EXP-24-QWEN25_7B** | 1 models × 20 countries | **0.4800** | MIS↓ JSD=0.0554 r=+0.202 Flip=16.0% |

**DPBR summary:** Mean MIS=0.4800, r=+0.202, Flip=16.0%, rel_r=0.994 · *(EXP-09 SOTA: 0.3975 | EXP-24 multi-model ref: 0.3969)*

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-QWEN25_7B Per-Dimension Worst Errors

| Model | Country | Worst Dim | Human | Model | Abs err (pp) |
|:------|:-------:|:----------|:-----:|:-----:|:------------:|
| Qwen2.5-7B | ARG | Age_Young | 75.1 | 50.2 | **25.0** |
| Qwen2.5-7B | BGD | Species_Humans | 78.5 | 50.4 | **28.1** |
| Qwen2.5-7B | BRA | Age_Young | 73.6 | 49.9 | **23.7** |
| Qwen2.5-7B | CHN | Species_Humans | 83.0 | 50.0 | **32.9** |
| Qwen2.5-7B | COL | Age_Young | 76.3 | 50.2 | **26.1** |
| Qwen2.5-7B | DEU | Species_Humans | 82.4 | 50.1 | **32.3** |
| Qwen2.5-7B | ETH | Species_Humans | 93.9 | 50.4 | **43.4** |
| Qwen2.5-7B | GBR | Species_Humans | 79.9 | 50.3 | **29.6** |
| Qwen2.5-7B | IDN | Species_Humans | 77.4 | 50.1 | **27.3** |
| Qwen2.5-7B | IRN | Species_Humans | 84.0 | 50.1 | **33.8** |
| Qwen2.5-7B | JPN | Species_Humans | 79.8 | 50.0 | **29.8** |
| Qwen2.5-7B | KGZ | Species_Humans | 78.7 | 51.0 | **27.8** |
| Qwen2.5-7B | MEX | Age_Young | 75.2 | 50.2 | **25.0** |
| Qwen2.5-7B | MMR | Utilitarianism_More | 78.7 | 49.9 | **28.8** |
| Qwen2.5-7B | MYS | Species_Humans | 76.6 | 50.4 | **26.3** |
| Qwen2.5-7B | ROU | Species_Humans | 80.1 | 50.6 | **29.6** |
| Qwen2.5-7B | SRB | Species_Humans | 77.7 | 50.4 | **27.3** |
| Qwen2.5-7B | THA | Species_Humans | 79.7 | 50.4 | **29.3** |
| Qwen2.5-7B | USA | Species_Humans | 79.2 | 50.4 | **28.8** |
| Qwen2.5-7B | VNM | Species_Humans | 77.7 | 49.9 | **27.8** |

---

#### EXP-24-QWEN35_08B Full Metrics

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Qwen3.5-0.8B | ARG | 0.4564 | 0.0459 | +0.225 | 17.19 | 24.8% |
| Qwen3.5-0.8B | BGD | 0.4725 | 0.0547 | +0.992 | 17.27 | 47.4% |
| Qwen3.5-0.8B | BRA | 0.4183 | 0.0480 | +0.246 | 15.43 | 56.5% |
| Qwen3.5-0.8B | CHN | 0.4725 | 0.0584 | +0.152 | 16.93 | 1.6% |
| Qwen3.5-0.8B | COL | 0.4796 | 0.0448 | +0.365 | 18.22 | 23.2% |
| Qwen3.5-0.8B | DEU | 0.5053 | 0.0631 | -0.209 | 18.02 | 0.0% |
| Qwen3.5-0.8B | ETH | 0.6041 | 0.0669 | +0.810 | 21.72 | 49.7% |
| Qwen3.5-0.8B | GBR | 0.5137 | 0.0564 | +0.645 | 18.91 | 49.0% |
| Qwen3.5-0.8B | IDN | 0.4600 | 0.0510 | +0.029 | 17.02 | 15.8% |
| Qwen3.5-0.8B | IRN | 0.5108 | 0.0678 | +0.088 | 17.86 | 67.4% |
| Qwen3.5-0.8B | JPN | 0.4424 | 0.0551 | -0.039 | 15.91 | 0.3% |
| Qwen3.5-0.8B | KGZ | 0.4785 | 0.0565 | +0.883 | 17.40 | 41.9% |
| Qwen3.5-0.8B | MEX | 0.4818 | 0.0477 | -0.023 | 18.18 | 22.6% |
| Qwen3.5-0.8B | MMR | 0.4625 | 0.0648 | +0.795 | 16.01 | 22.6% |
| Qwen3.5-0.8B | MYS | 0.4383 | 0.0494 | +0.881 | 16.17 | 49.4% |
| Qwen3.5-0.8B | ROU | 0.4976 | 0.0565 | +0.730 | 18.21 | 58.4% |
| Qwen3.5-0.8B | SRB | 0.4972 | 0.0550 | +0.847 | 18.32 | 36.5% |
| Qwen3.5-0.8B | THA | 0.4313 | 0.0559 | +0.860 | 15.33 | 55.2% |
| Qwen3.5-0.8B | USA | 0.5039 | 0.0543 | +0.699 | 18.64 | 48.7% |
| Qwen3.5-0.8B | VNM | 0.4862 | 0.0509 | -0.568 | 18.16 | 11.6% |

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-QWEN35_08B vs Vanilla (MIS)

| Model | Country | Vanilla MIS | EXP-24-QWEN35_08B MIS | Δ | Improv% | Win? |
|:------|:-------:|:-----------:|:--------------------:|:-:|:-------:|:----:|
| Qwen3.5-0.8B | ARG | 0.4571 | 0.4564 | +0.0007 | **+0.16%** | ✅ |
| Qwen3.5-0.8B | BGD | 0.4764 | 0.4725 | +0.0040 | **+0.83%** | ✅ |
| Qwen3.5-0.8B | BRA | 0.4159 | 0.4183 | -0.0024 | **-0.58%** | ❌ |
| Qwen3.5-0.8B | CHN | 0.4686 | 0.4725 | -0.0039 | **-0.82%** | ❌ |
| Qwen3.5-0.8B | COL | 0.4802 | 0.4796 | +0.0006 | **+0.12%** | ✅ |
| Qwen3.5-0.8B | DEU | 0.5040 | 0.5053 | -0.0013 | **-0.26%** | ❌ |
| Qwen3.5-0.8B | ETH | 0.6080 | 0.6041 | +0.0039 | **+0.64%** | ✅ |
| Qwen3.5-0.8B | GBR | 0.5164 | 0.5137 | +0.0027 | **+0.52%** | ✅ |
| Qwen3.5-0.8B | IDN | 0.4450 | 0.4600 | -0.0150 | **-3.36%** | ❌ |
| Qwen3.5-0.8B | IRN | 0.5138 | 0.5108 | +0.0031 | **+0.60%** | ✅ |
| Qwen3.5-0.8B | JPN | 0.4418 | 0.4424 | -0.0005 | **-0.12%** | ❌ |
| Qwen3.5-0.8B | KGZ | 0.4818 | 0.4785 | +0.0033 | **+0.68%** | ✅ |
| Qwen3.5-0.8B | MEX | 0.4804 | 0.4818 | -0.0014 | **-0.29%** | ❌ |
| Qwen3.5-0.8B | MMR | 0.4667 | 0.4625 | +0.0042 | **+0.89%** | ✅ |
| Qwen3.5-0.8B | MYS | 0.4407 | 0.4383 | +0.0024 | **+0.54%** | ✅ |
| Qwen3.5-0.8B | ROU | 0.5006 | 0.4976 | +0.0030 | **+0.61%** | ✅ |
| Qwen3.5-0.8B | SRB | 0.5006 | 0.4972 | +0.0033 | **+0.66%** | ✅ |
| Qwen3.5-0.8B | THA | 0.4339 | 0.4313 | +0.0027 | **+0.62%** | ✅ |
| Qwen3.5-0.8B | USA | 0.5077 | 0.5039 | +0.0038 | **+0.75%** | ✅ |
| Qwen3.5-0.8B | VNM | 0.4809 | 0.4862 | -0.0053 | **-1.10%** | ❌ |

- **Qwen3.5-0.8B** Win Rate: **13/20** | Vanilla=0.4810 → EXP-24-QWEN35_08B=0.4806 | Macro Δ: **+0.08%**

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-QWEN35_08B vs EXP-01 SWA-PTIS (MIS)

| Model | Country | EXP-01 MIS | EXP-24-QWEN35_08B MIS | Δ | Improv% | Win? |
|:------|:-------:|:----------:|:--------------------:|:-:|:-------:|:----:|
| Qwen3.5-0.8B | BRA | 0.4025 | 0.4183 | -0.0158 | **-3.92%** | ❌ |
| Qwen3.5-0.8B | CHN | 0.4078 | 0.4725 | -0.0647 | **-15.86%** | ❌ |
| Qwen3.5-0.8B | DEU | 0.3424 | 0.5053 | -0.1629 | **-47.57%** | ❌ |
| Qwen3.5-0.8B | JPN | 0.2802 | 0.4424 | -0.1622 | **-57.88%** | ❌ |
| Qwen3.5-0.8B | USA | 0.3677 | 0.5039 | -0.1362 | **-37.05%** | ❌ |

- **Qwen3.5-0.8B** Win Rate: **0/5** | EXP-01=0.3601 → EXP-24-QWEN35_08B=0.4685 | Macro Δ: **-30.09%**

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-QWEN35_08B Leaderboard Entry

| Rank | Method | Coverage | Mean MIS ↓ | Notes |
|:---:|:-------|:--------:|-----------:|:------|
| ? | **EXP-24-QWEN35_08B** | 1 models × 20 countries | **0.4806** | MIS↓ JSD=0.0552 r=+0.420 Flip=34.1% |

**DPBR summary:** Mean MIS=0.4806, r=+0.420, Flip=34.1%, rel_r=0.994 · *(EXP-09 SOTA: 0.3975 | EXP-24 multi-model ref: 0.3969)*

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-QWEN35_08B Per-Dimension Worst Errors

| Model | Country | Worst Dim | Human | Model | Abs err (pp) |
|:------|:-------:|:----------|:-----:|:-----:|:------------:|
| Qwen3.5-0.8B | ARG | Age_Young | 75.1 | 50.0 | **25.2** |
| Qwen3.5-0.8B | BGD | Species_Humans | 78.5 | 50.4 | **28.1** |
| Qwen3.5-0.8B | BRA | Age_Young | 73.6 | 50.0 | **23.6** |
| Qwen3.5-0.8B | CHN | Species_Humans | 83.0 | 50.0 | **33.0** |
| Qwen3.5-0.8B | COL | Age_Young | 76.3 | 50.0 | **26.3** |
| Qwen3.5-0.8B | DEU | Species_Humans | 82.4 | 49.7 | **32.7** |
| Qwen3.5-0.8B | ETH | Species_Humans | 93.9 | 50.3 | **43.5** |
| Qwen3.5-0.8B | GBR | Species_Humans | 79.9 | 50.3 | **29.6** |
| Qwen3.5-0.8B | IDN | Species_Humans | 77.4 | 49.9 | **27.5** |
| Qwen3.5-0.8B | IRN | Species_Humans | 84.0 | 50.0 | **34.0** |
| Qwen3.5-0.8B | JPN | Species_Humans | 79.8 | 50.1 | **29.7** |
| Qwen3.5-0.8B | KGZ | Species_Humans | 78.7 | 50.4 | **28.4** |
| Qwen3.5-0.8B | MEX | Age_Young | 75.2 | 50.0 | **25.2** |
| Qwen3.5-0.8B | MMR | Utilitarianism_More | 78.7 | 50.4 | **28.3** |
| Qwen3.5-0.8B | MYS | Species_Humans | 76.6 | 50.2 | **26.4** |
| Qwen3.5-0.8B | ROU | Species_Humans | 80.1 | 50.3 | **29.9** |
| Qwen3.5-0.8B | SRB | Species_Humans | 77.7 | 50.4 | **27.4** |
| Qwen3.5-0.8B | THA | Species_Humans | 79.7 | 50.2 | **29.4** |
| Qwen3.5-0.8B | USA | Species_Humans | 79.2 | 50.3 | **28.9** |
| Qwen3.5-0.8B | VNM | Species_Humans | 77.7 | 49.5 | **28.2** |

---

#### EXP-24-MISTRAL_V03 Full Metrics

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Mistral-7B | ARG | 0.3787 | 0.0626 | -0.560 | 12.37 | 15.5% |
| Mistral-7B | BGD | 0.5294 | 0.1083 | -0.840 | 18.92 | 18.4% |
| Mistral-7B | BRA | 0.3937 | 0.0741 | -0.591 | 13.41 | 25.2% |
| Mistral-7B | CHN | 0.4104 | 0.0910 | -0.462 | 15.05 | 14.8% |
| Mistral-7B | COL | 0.4050 | 0.0656 | -0.619 | 13.35 | 13.9% |
| Mistral-7B | DEU | 0.4467 | 0.0823 | -0.971 | 14.59 | 14.2% |
| Mistral-7B | ETH | 0.6243 | 0.1115 | -0.421 | 19.68 | 21.0% |
| Mistral-7B | GBR | 0.5652 | 0.1285 | -0.817 | 21.26 | 18.1% |
| Mistral-7B | IDN | 0.4480 | 0.0610 | -0.778 | 15.84 | 13.5% |
| Mistral-7B | IRN | 0.4320 | 0.0640 | +0.474 | 14.33 | 23.5% |
| Mistral-7B | JPN | 0.3422 | 0.0656 | -0.519 | 11.34 | 20.0% |
| Mistral-7B | KGZ | 0.5341 | 0.1134 | -0.919 | 19.43 | 18.7% |
| Mistral-7B | MEX | 0.4000 | 0.0635 | -0.453 | 13.32 | 15.5% |
| Mistral-7B | MMR | 0.5309 | 0.1214 | -0.962 | 19.29 | 22.3% |
| Mistral-7B | MYS | 0.5165 | 0.1196 | -0.864 | 19.59 | 22.3% |
| Mistral-7B | ROU | 0.5613 | 0.1205 | -0.879 | 20.62 | 23.2% |
| Mistral-7B | SRB | 0.5644 | 0.1257 | -0.889 | 20.96 | 20.0% |
| Mistral-7B | THA | 0.5019 | 0.1161 | -0.727 | 18.52 | 19.4% |
| Mistral-7B | USA | 0.5232 | 0.1097 | -0.774 | 19.06 | 22.6% |
| Mistral-7B | VNM | 0.3728 | 0.0541 | +0.044 | 12.75 | 21.9% |

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-MISTRAL_V03 vs Vanilla (MIS)

| Model | Country | Vanilla MIS | EXP-24-MISTRAL_V03 MIS | Δ | Improv% | Win? |
|:------|:-------:|:-----------:|:----------------------:|:-:|:-------:|:----:|
| Mistral-7B | ARG | 0.4060 | 0.3787 | +0.0273 | **+6.72%** | ✅ |
| Mistral-7B | BGD | 0.4852 | 0.5294 | -0.0442 | **-9.10%** | ❌ |
| Mistral-7B | BRA | 0.3759 | 0.3937 | -0.0177 | **-4.71%** | ❌ |
| Mistral-7B | CHN | 0.4672 | 0.4104 | +0.0569 | **+12.17%** | ✅ |
| Mistral-7B | COL | 0.4284 | 0.4050 | +0.0233 | **+5.45%** | ✅ |
| Mistral-7B | DEU | 0.4892 | 0.4467 | +0.0425 | **+8.69%** | ✅ |
| Mistral-7B | ETH | 0.6159 | 0.6243 | -0.0083 | **-1.35%** | ❌ |
| Mistral-7B | GBR | 0.5330 | 0.5652 | -0.0322 | **-6.03%** | ❌ |
| Mistral-7B | IDN | 0.4611 | 0.4480 | +0.0131 | **+2.83%** | ✅ |
| Mistral-7B | IRN | 0.5122 | 0.4320 | +0.0802 | **+15.65%** | ✅ |
| Mistral-7B | JPN | 0.4410 | 0.3422 | +0.0989 | **+22.42%** | ✅ |
| Mistral-7B | KGZ | 0.4907 | 0.5341 | -0.0433 | **-8.83%** | ❌ |
| Mistral-7B | MEX | 0.4015 | 0.4000 | +0.0015 | **+0.37%** | ✅ |
| Mistral-7B | MMR | 0.4740 | 0.5309 | -0.0569 | **-12.01%** | ❌ |
| Mistral-7B | MYS | 0.4498 | 0.5165 | -0.0667 | **-14.84%** | ❌ |
| Mistral-7B | ROU | 0.5089 | 0.5613 | -0.0524 | **-10.29%** | ❌ |
| Mistral-7B | SRB | 0.5087 | 0.5644 | -0.0558 | **-10.96%** | ❌ |
| Mistral-7B | THA | 0.4424 | 0.5019 | -0.0595 | **-13.44%** | ❌ |
| Mistral-7B | USA | 0.5248 | 0.5232 | +0.0016 | **+0.31%** | ✅ |
| Mistral-7B | VNM | 0.4544 | 0.3728 | +0.0816 | **+17.96%** | ✅ |

- **Mistral-7B** Win Rate: **10/20** | Vanilla=0.4735 → EXP-24-MISTRAL_V03=0.4740 | Macro Δ: **-0.11%**

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-MISTRAL_V03 vs EXP-01 SWA-PTIS (MIS)

| Model | Country | EXP-01 MIS | EXP-24-MISTRAL_V03 MIS | Δ | Improv% | Win? |
|:------|:-------:|:----------:|:----------------------:|:-:|:-------:|:----:|
| Mistral-7B | BRA | 0.4362 | 0.3937 | +0.0425 | **+9.75%** | ✅ |
| Mistral-7B | CHN | 0.5067 | 0.4104 | +0.0963 | **+19.02%** | ✅ |
| Mistral-7B | DEU | 0.4942 | 0.4467 | +0.0475 | **+9.60%** | ✅ |
| Mistral-7B | JPN | 0.3502 | 0.3422 | +0.0080 | **+2.30%** | ✅ |
| Mistral-7B | USA | 0.5984 | 0.5232 | +0.0752 | **+12.57%** | ✅ |

- **Mistral-7B** Win Rate: **5/5** | EXP-01=0.4771 → EXP-24-MISTRAL_V03=0.4232 | Macro Δ: **+11.30%**

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-MISTRAL_V03 Leaderboard Entry

| Rank | Method | Coverage | Mean MIS ↓ | Notes |
|:---:|:-------|:--------:|-----------:|:------|
| ? | **EXP-24-MISTRAL_V03** | 1 models × 20 countries | **0.4740** | MIS↓ JSD=0.0929 r=-0.626 Flip=19.2% |

**DPBR summary:** Mean MIS=0.4740, r=-0.626, Flip=19.2%, rel_r=0.974 · *(EXP-09 SOTA: 0.3975 | EXP-24 multi-model ref: 0.3969)*

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-MISTRAL_V03 Per-Dimension Worst Errors

| Model | Country | Worst Dim | Human | Model | Abs err (pp) |
|:------|:-------:|:----------|:-----:|:-----:|:------------:|
| Mistral-7B | ARG | Age_Young | 75.1 | 49.8 | **25.3** |
| Mistral-7B | BGD | Utilitarianism_More | 75.0 | 45.5 | **29.5** |
| Mistral-7B | BRA | Age_Young | 73.6 | 44.3 | **29.3** |
| Mistral-7B | CHN | Species_Humans | 83.0 | 56.0 | **27.0** |
| Mistral-7B | COL | Age_Young | 76.3 | 48.6 | **27.7** |
| Mistral-7B | DEU | Species_Humans | 82.4 | 51.9 | **30.6** |
| Mistral-7B | ETH | Species_Humans | 93.9 | 48.0 | **45.9** |
| Mistral-7B | GBR | Utilitarianism_More | 77.0 | 45.6 | **31.3** |
| Mistral-7B | IDN | Species_Humans | 77.4 | 49.0 | **28.4** |
| Mistral-7B | IRN | Species_Humans | 84.0 | 53.4 | **30.6** |
| Mistral-7B | JPN | Species_Humans | 79.8 | 55.1 | **24.8** |
| Mistral-7B | KGZ | Species_Humans | 78.7 | 48.6 | **30.2** |
| Mistral-7B | MEX | Age_Young | 75.2 | 48.9 | **26.3** |
| Mistral-7B | MMR | Age_Young | 75.9 | 44.4 | **31.5** |
| Mistral-7B | MYS | Species_Humans | 76.6 | 49.2 | **27.4** |
| Mistral-7B | ROU | Age_Young | 74.9 | 44.0 | **30.9** |
| Mistral-7B | SRB | Age_Young | 76.0 | 42.7 | **33.3** |
| Mistral-7B | THA | Species_Humans | 79.7 | 50.1 | **29.6** |
| Mistral-7B | USA | Age_Young | 74.5 | 43.1 | **31.4** |
| Mistral-7B | VNM | Age_Young | 72.7 | 50.3 | **22.4** |

---

#### EXP-24-GPT_OSS_20B Full Metrics

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| gpt-oss-20b-unsloth-bnb-4bit | ARG | 0.4531 | 0.0450 | +0.658 | 17.08 | 41.9% |
| gpt-oss-20b-unsloth-bnb-4bit | BGD | 0.4772 | 0.0565 | -0.263 | 17.38 | 31.9% |
| gpt-oss-20b-unsloth-bnb-4bit | BRA | 0.4177 | 0.0500 | -0.422 | 15.29 | 35.8% |
| gpt-oss-20b-unsloth-bnb-4bit | CHN | 0.4553 | 0.0547 | +0.684 | 16.37 | 23.2% |
| gpt-oss-20b-unsloth-bnb-4bit | COL | 0.4868 | 0.0467 | -0.250 | 18.44 | 31.6% |
| gpt-oss-20b-unsloth-bnb-4bit | DEU | 0.5051 | 0.0632 | -0.110 | 18.01 | 36.8% |
| gpt-oss-20b-unsloth-bnb-4bit | ETH | 0.6054 | 0.0668 | +0.340 | 21.78 | 39.0% |
| gpt-oss-20b-unsloth-bnb-4bit | GBR | 0.5215 | 0.0573 | +0.055 | 19.21 | 32.9% |
| gpt-oss-20b-unsloth-bnb-4bit | IDN | 0.4656 | 0.0515 | -0.525 | 17.24 | 40.3% |
| gpt-oss-20b-unsloth-bnb-4bit | IRN | 0.5141 | 0.0694 | -0.573 | 17.92 | 27.7% |
| gpt-oss-20b-unsloth-bnb-4bit | JPN | 0.4409 | 0.0538 | +0.669 | 15.92 | 43.5% |
| gpt-oss-20b-unsloth-bnb-4bit | KGZ | 0.4863 | 0.0580 | -0.277 | 17.67 | 42.3% |
| gpt-oss-20b-unsloth-bnb-4bit | MEX | 0.4798 | 0.0476 | -0.101 | 18.10 | 34.8% |
| gpt-oss-20b-unsloth-bnb-4bit | MMR | 0.4717 | 0.0675 | -0.700 | 16.25 | 44.2% |
| gpt-oss-20b-unsloth-bnb-4bit | MYS | 0.4394 | 0.0496 | +0.212 | 16.21 | 35.8% |
| gpt-oss-20b-unsloth-bnb-4bit | ROU | 0.5078 | 0.0584 | -0.457 | 18.57 | 29.7% |
| gpt-oss-20b-unsloth-bnb-4bit | SRB | 0.5091 | 0.0568 | -0.764 | 18.76 | 43.5% |
| gpt-oss-20b-unsloth-bnb-4bit | THA | 0.4408 | 0.0588 | -0.419 | 15.59 | 47.7% |
| gpt-oss-20b-unsloth-bnb-4bit | USA | 0.5075 | 0.0553 | -0.123 | 18.74 | 49.7% |
| gpt-oss-20b-unsloth-bnb-4bit | VNM | 0.4796 | 0.0493 | +0.266 | 17.95 | 35.5% |

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-GPT_OSS_20B vs Vanilla (MIS)

| Model | Country | Vanilla MIS | EXP-24-GPT_OSS_20B MIS | Δ | Improv% | Win? |
|:------|:-------:|:-----------:|:----------------------:|:-:|:-------:|:----:|
| gpt-oss-20b-unsloth-bnb-4bit | ARG | 0.4477 | 0.4531 | -0.0054 | **-1.22%** | ❌ |
| gpt-oss-20b-unsloth-bnb-4bit | BGD | 0.5027 | 0.4772 | +0.0256 | **+5.09%** | ✅ |
| gpt-oss-20b-unsloth-bnb-4bit | BRA | 0.4439 | 0.4177 | +0.0262 | **+5.90%** | ✅ |
| gpt-oss-20b-unsloth-bnb-4bit | CHN | 0.4304 | 0.4553 | -0.0249 | **-5.79%** | ❌ |
| gpt-oss-20b-unsloth-bnb-4bit | COL | 0.4716 | 0.4868 | -0.0152 | **-3.21%** | ❌ |
| gpt-oss-20b-unsloth-bnb-4bit | DEU | 0.5116 | 0.5051 | +0.0065 | **+1.28%** | ✅ |
| gpt-oss-20b-unsloth-bnb-4bit | ETH | 0.6257 | 0.6054 | +0.0203 | **+3.25%** | ✅ |
| gpt-oss-20b-unsloth-bnb-4bit | GBR | 0.5019 | 0.5215 | -0.0196 | **-3.90%** | ❌ |
| gpt-oss-20b-unsloth-bnb-4bit | IDN | 0.4205 | 0.4656 | -0.0451 | **-10.71%** | ❌ |
| gpt-oss-20b-unsloth-bnb-4bit | IRN | 0.5452 | 0.5141 | +0.0311 | **+5.70%** | ✅ |
| gpt-oss-20b-unsloth-bnb-4bit | JPN | 0.4195 | 0.4409 | -0.0214 | **-5.09%** | ❌ |
| gpt-oss-20b-unsloth-bnb-4bit | KGZ | 0.5064 | 0.4863 | +0.0202 | **+3.98%** | ✅ |
| gpt-oss-20b-unsloth-bnb-4bit | MEX | 0.4718 | 0.4798 | -0.0079 | **-1.68%** | ❌ |
| gpt-oss-20b-unsloth-bnb-4bit | MMR | 0.5010 | 0.4717 | +0.0293 | **+5.85%** | ✅ |
| gpt-oss-20b-unsloth-bnb-4bit | MYS | 0.4664 | 0.4394 | +0.0270 | **+5.80%** | ✅ |
| gpt-oss-20b-unsloth-bnb-4bit | ROU | 0.5245 | 0.5078 | +0.0167 | **+3.19%** | ✅ |
| gpt-oss-20b-unsloth-bnb-4bit | SRB | 0.5277 | 0.5091 | +0.0186 | **+3.52%** | ✅ |
| gpt-oss-20b-unsloth-bnb-4bit | THA | 0.4544 | 0.4408 | +0.0136 | **+2.99%** | ✅ |
| gpt-oss-20b-unsloth-bnb-4bit | USA | 0.4937 | 0.5075 | -0.0138 | **-2.80%** | ❌ |
| gpt-oss-20b-unsloth-bnb-4bit | VNM | 0.6097 | 0.4796 | +0.1301 | **+21.34%** | ✅ |

- **gpt-oss-20b-unsloth-bnb-4bit** Win Rate: **12/20** | Vanilla=0.4938 → EXP-24-GPT_OSS_20B=0.4832 | Macro Δ: **+2.15%**

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-GPT_OSS_20B vs EXP-01 SWA-PTIS (MIS)

| Model | Country | EXP-01 MIS | EXP-24-GPT_OSS_20B MIS | Δ | Improv% | Win? |
|:------|:-------:|:----------:|:----------------------:|:-:|:-------:|:----:|

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-GPT_OSS_20B Leaderboard Entry

| Rank | Method | Coverage | Mean MIS ↓ | Notes |
|:---:|:-------|:--------:|-----------:|:------|
| ? | **EXP-24-GPT_OSS_20B** | 1 models × 20 countries | **0.4832** | MIS↓ JSD=0.0558 r=-0.105 Flip=37.4% |

**DPBR summary:** Mean MIS=0.4832, r=-0.105, Flip=37.4%, rel_r=0.991 · *(EXP-09 SOTA: 0.3975 | EXP-24 multi-model ref: 0.3969)*

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-GPT_OSS_20B Per-Dimension Worst Errors

| Model | Country | Worst Dim | Human | Model | Abs err (pp) |
|:------|:-------:|:----------|:-----:|:-----:|:------------:|
| gpt-oss-20b-unsloth-bnb-4bit | ARG | Age_Young | 75.1 | 49.9 | **25.2** |
| gpt-oss-20b-unsloth-bnb-4bit | BGD | Species_Humans | 78.5 | 50.0 | **28.5** |
| gpt-oss-20b-unsloth-bnb-4bit | BRA | Age_Young | 73.6 | 49.8 | **23.8** |
| gpt-oss-20b-unsloth-bnb-4bit | CHN | Species_Humans | 83.0 | 51.1 | **31.9** |
| gpt-oss-20b-unsloth-bnb-4bit | COL | Utilitarianism_More | 75.8 | 49.2 | **26.6** |
| gpt-oss-20b-unsloth-bnb-4bit | DEU | Species_Humans | 82.4 | 50.4 | **32.0** |
| gpt-oss-20b-unsloth-bnb-4bit | ETH | Species_Humans | 93.9 | 50.0 | **43.9** |
| gpt-oss-20b-unsloth-bnb-4bit | GBR | Species_Humans | 79.9 | 49.4 | **30.5** |
| gpt-oss-20b-unsloth-bnb-4bit | IDN | Species_Humans | 77.4 | 49.8 | **27.6** |
| gpt-oss-20b-unsloth-bnb-4bit | IRN | Species_Humans | 84.0 | 50.0 | **33.9** |
| gpt-oss-20b-unsloth-bnb-4bit | JPN | Species_Humans | 79.8 | 50.1 | **29.7** |
| gpt-oss-20b-unsloth-bnb-4bit | KGZ | Species_Humans | 78.7 | 49.5 | **29.3** |
| gpt-oss-20b-unsloth-bnb-4bit | MEX | Age_Young | 75.2 | 50.3 | **24.9** |
| gpt-oss-20b-unsloth-bnb-4bit | MMR | Utilitarianism_More | 78.7 | 49.7 | **29.0** |
| gpt-oss-20b-unsloth-bnb-4bit | MYS | Species_Humans | 76.6 | 49.8 | **26.9** |
| gpt-oss-20b-unsloth-bnb-4bit | ROU | Species_Humans | 80.1 | 49.4 | **30.8** |
| gpt-oss-20b-unsloth-bnb-4bit | SRB | Species_Humans | 77.7 | 49.6 | **28.2** |
| gpt-oss-20b-unsloth-bnb-4bit | THA | Species_Humans | 79.7 | 49.8 | **29.9** |
| gpt-oss-20b-unsloth-bnb-4bit | USA | Species_Humans | 79.2 | 49.7 | **29.5** |
| gpt-oss-20b-unsloth-bnb-4bit | VNM | Species_Humans | 77.7 | 49.8 | **27.9** |

---

#### EXP-24-MAGISTRAL_SMALL_2509 Full Metrics

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Magistral-Small-2509 | ARG | 0.3422 | 0.0464 | +0.601 | 12.41 | 21.3% |
| Magistral-Small-2509 | BGD | 0.3614 | 0.0470 | +0.625 | 13.06 | 16.8% |
| Magistral-Small-2509 | BRA | 0.3188 | 0.0510 | +0.333 | 11.14 | 19.7% |
| Magistral-Small-2509 | CHN | 0.2932 | 0.0499 | +0.711 | 10.16 | 25.2% |
| Magistral-Small-2509 | COL | 0.3686 | 0.0546 | +0.386 | 12.85 | 21.6% |
| Magistral-Small-2509 | DEU | 0.3445 | 0.0451 | +0.724 | 12.01 | 19.7% |
| Magistral-Small-2509 | ETH | 0.4717 | 0.0538 | +0.655 | 17.05 | 15.8% |
| Magistral-Small-2509 | GBR | 0.3837 | 0.0484 | +0.674 | 14.05 | 17.1% |
| Magistral-Small-2509 | IDN | 0.3443 | 0.0588 | +0.511 | 11.52 | 21.9% |
| Magistral-Small-2509 | IRN | 0.4012 | 0.0547 | +0.637 | 14.10 | 27.1% |
| Magistral-Small-2509 | JPN | 0.2646 | 0.0542 | +0.709 | 8.63 | 20.3% |
| Magistral-Small-2509 | KGZ | 0.3534 | 0.0480 | +0.633 | 12.62 | 17.1% |
| Magistral-Small-2509 | MEX | 0.3327 | 0.0403 | +0.707 | 12.37 | 20.6% |
| Magistral-Small-2509 | MMR | 0.3647 | 0.0621 | +0.457 | 12.53 | 19.4% |
| Magistral-Small-2509 | MYS | 0.3127 | 0.0428 | +0.683 | 11.31 | 17.1% |
| Magistral-Small-2509 | ROU | 0.3628 | 0.0429 | +0.708 | 13.35 | 16.1% |
| Magistral-Small-2509 | SRB | 0.3715 | 0.0446 | +0.670 | 13.65 | 18.7% |
| Magistral-Small-2509 | THA | 0.2949 | 0.0410 | +0.738 | 10.49 | 17.4% |
| Magistral-Small-2509 | USA | 0.3741 | 0.0422 | +0.731 | 13.99 | 20.3% |
| Magistral-Small-2509 | VNM | 0.3473 | 0.0500 | +0.589 | 12.32 | 25.5% |

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-MAGISTRAL_SMALL_2509 vs Vanilla (MIS)

| Model | Country | Vanilla MIS | EXP-24-MAGISTRAL_SMALL_2509 MIS | Δ | Improv% | Win? |
|:------|:-------:|:-----------:|:-------------------------------:|:-:|:-------:|:----:|
| Magistral-Small-2509 | ARG | 0.3642 | 0.3422 | +0.0219 | **+6.03%** | ✅ |
| Magistral-Small-2509 | BGD | 0.4496 | 0.3614 | +0.0881 | **+19.61%** | ✅ |
| Magistral-Small-2509 | BRA | 0.4115 | 0.3188 | +0.0928 | **+22.54%** | ✅ |
| Magistral-Small-2509 | CHN | 0.3673 | 0.2932 | +0.0742 | **+20.19%** | ✅ |
| Magistral-Small-2509 | COL | 0.4127 | 0.3686 | +0.0440 | **+10.67%** | ✅ |
| Magistral-Small-2509 | DEU | 0.2645 | 0.3445 | -0.0800 | **-30.26%** | ❌ |
| Magistral-Small-2509 | ETH | 0.4624 | 0.4717 | -0.0092 | **-1.99%** | ❌ |
| Magistral-Small-2509 | GBR | 0.4422 | 0.3837 | +0.0585 | **+13.23%** | ✅ |
| Magistral-Small-2509 | IDN | 0.2834 | 0.3443 | -0.0609 | **-21.48%** | ❌ |
| Magistral-Small-2509 | IRN | 0.4271 | 0.4012 | +0.0259 | **+6.06%** | ✅ |
| Magistral-Small-2509 | JPN | 0.3406 | 0.2646 | +0.0761 | **+22.33%** | ✅ |
| Magistral-Small-2509 | KGZ | 0.4603 | 0.3534 | +0.1069 | **+23.23%** | ✅ |
| Magistral-Small-2509 | MEX | 0.3488 | 0.3327 | +0.0161 | **+4.61%** | ✅ |
| Magistral-Small-2509 | MMR | 0.4962 | 0.3647 | +0.1314 | **+26.49%** | ✅ |
| Magistral-Small-2509 | MYS | 0.4430 | 0.3127 | +0.1303 | **+29.42%** | ✅ |
| Magistral-Small-2509 | ROU | 0.4463 | 0.3628 | +0.0835 | **+18.71%** | ✅ |
| Magistral-Small-2509 | SRB | 0.4634 | 0.3715 | +0.0918 | **+19.82%** | ✅ |
| Magistral-Small-2509 | THA | 0.4058 | 0.2949 | +0.1110 | **+27.34%** | ✅ |
| Magistral-Small-2509 | USA | 0.4480 | 0.3741 | +0.0739 | **+16.50%** | ✅ |
| Magistral-Small-2509 | VNM | 0.3239 | 0.3473 | -0.0234 | **-7.24%** | ❌ |

- **Magistral-Small-2509** Win Rate: **16/20** | Vanilla=0.4031 → EXP-24-MAGISTRAL_SMALL_2509=0.3504 | Macro Δ: **+13.06%**

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-MAGISTRAL_SMALL_2509 vs EXP-01 SWA-PTIS (MIS)

| Model | Country | EXP-01 MIS | EXP-24-MAGISTRAL_SMALL_2509 MIS | Δ | Improv% | Win? |
|:------|:-------:|:----------:|:-------------------------------:|:-:|:-------:|:----:|
| Magistral-Small-2509 | BRA | 0.4362 | 0.3188 | +0.1174 | **+26.92%** | ✅ |
| Magistral-Small-2509 | CHN | 0.5067 | 0.2932 | +0.2135 | **+42.14%** | ✅ |
| Magistral-Small-2509 | DEU | 0.4942 | 0.3445 | +0.1497 | **+30.29%** | ✅ |
| Magistral-Small-2509 | JPN | 0.3502 | 0.2646 | +0.0856 | **+24.45%** | ✅ |
| Magistral-Small-2509 | USA | 0.5984 | 0.3741 | +0.2243 | **+37.48%** | ✅ |

- **Magistral-Small-2509** Win Rate: **5/5** | EXP-01=0.4771 → EXP-24-MAGISTRAL_SMALL_2509=0.3190 | Macro Δ: **+33.14%**

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-MAGISTRAL_SMALL_2509 Leaderboard Entry

| Rank | Method | Coverage | Mean MIS ↓ | Notes |
|:---:|:-------|:--------:|-----------:|:------|
| ? | **EXP-24-MAGISTRAL_SMALL_2509** | 1 models × 20 countries | **0.3504** | MIS↓ JSD=0.0489 r=+0.624 Flip=19.9% |

**DPBR summary:** Mean MIS=0.3504, r=+0.624, Flip=19.9%, rel_r=0.973 · *(EXP-09 SOTA: 0.3975 | EXP-24 multi-model ref: 0.3969)*

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-MAGISTRAL_SMALL_2509 Per-Dimension Worst Errors

| Model | Country | Worst Dim | Human | Model | Abs err (pp) |
|:------|:-------:|:----------|:-----:|:-----:|:------------:|
| Magistral-Small-2509 | ARG | SocialValue_High | 68.5 | 45.8 | **22.8** |
| Magistral-Small-2509 | BGD | SocialValue_High | 68.1 | 45.6 | **22.5** |
| Magistral-Small-2509 | BRA | Age_Young | 73.6 | 53.8 | **19.7** |
| Magistral-Small-2509 | CHN | SocialValue_High | 66.7 | 45.2 | **21.5** |
| Magistral-Small-2509 | COL | SocialValue_High | 72.7 | 46.4 | **26.3** |
| Magistral-Small-2509 | DEU | Species_Humans | 82.4 | 61.6 | **20.9** |
| Magistral-Small-2509 | ETH | Species_Humans | 93.9 | 66.5 | **27.3** |
| Magistral-Small-2509 | GBR | SocialValue_High | 67.7 | 43.6 | **24.0** |
| Magistral-Small-2509 | IDN | SocialValue_High | 69.0 | 43.4 | **25.6** |
| Magistral-Small-2509 | IRN | Age_Young | 73.7 | 51.6 | **22.1** |
| Magistral-Small-2509 | JPN | SocialValue_High | 65.9 | 44.5 | **21.4** |
| Magistral-Small-2509 | KGZ | SocialValue_High | 68.7 | 45.9 | **22.8** |
| Magistral-Small-2509 | MEX | SocialValue_High | 69.6 | 47.1 | **22.5** |
| Magistral-Small-2509 | MMR | Utilitarianism_More | 78.7 | 54.9 | **23.8** |
| Magistral-Small-2509 | MYS | SocialValue_High | 67.4 | 45.8 | **21.7** |
| Magistral-Small-2509 | ROU | SocialValue_High | 67.6 | 46.0 | **21.5** |
| Magistral-Small-2509 | SRB | SocialValue_High | 67.3 | 45.9 | **21.4** |
| Magistral-Small-2509 | THA | SocialValue_High | 65.1 | 46.6 | **18.5** |
| Magistral-Small-2509 | USA | SocialValue_High | 67.9 | 46.3 | **21.6** |
| Magistral-Small-2509 | VNM | Age_Young | 72.7 | 47.2 | **25.5** |

---

#### EXP-24-GEMMA_7B Full Metrics

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| gemma-7b-it-bnb-4bit | ARG | 0.3985 | 0.0613 | -0.333 | 13.86 | 5.5% |
| gemma-7b-it-bnb-4bit | BGD | 0.5033 | 0.0779 | -0.760 | 17.16 | 23.5% |
| gemma-7b-it-bnb-4bit | BRA | 0.3689 | 0.0658 | -0.549 | 13.09 | 14.5% |
| gemma-7b-it-bnb-4bit | CHN | 0.4293 | 0.0635 | -0.107 | 14.51 | 3.5% |
| gemma-7b-it-bnb-4bit | COL | 0.4097 | 0.0584 | -0.179 | 14.28 | 5.2% |
| gemma-7b-it-bnb-4bit | DEU | 0.4323 | 0.0860 | -0.538 | 14.39 | 7.7% |
| gemma-7b-it-bnb-4bit | ETH | 0.6121 | 0.0793 | -0.428 | 21.29 | 25.8% |
| gemma-7b-it-bnb-4bit | GBR | 0.5251 | 0.0729 | -0.767 | 18.38 | 11.0% |
| gemma-7b-it-bnb-4bit | IDN | 0.3951 | 0.0552 | +0.009 | 13.79 | 4.8% |
| gemma-7b-it-bnb-4bit | IRN | 0.4925 | 0.0717 | -0.150 | 16.68 | 2.3% |
| gemma-7b-it-bnb-4bit | JPN | 0.3803 | 0.0642 | -0.122 | 12.22 | 37.7% |
| gemma-7b-it-bnb-4bit | KGZ | 0.5073 | 0.0777 | -0.799 | 17.32 | 24.2% |
| gemma-7b-it-bnb-4bit | MEX | 0.4160 | 0.0602 | -0.264 | 14.38 | 5.5% |
| gemma-7b-it-bnb-4bit | MMR | 0.4853 | 0.0858 | -0.948 | 16.13 | 21.3% |
| gemma-7b-it-bnb-4bit | MYS | 0.4545 | 0.0627 | -0.628 | 16.13 | 28.7% |
| gemma-7b-it-bnb-4bit | ROU | 0.5020 | 0.0720 | -0.698 | 17.41 | 17.1% |
| gemma-7b-it-bnb-4bit | SRB | 0.5107 | 0.0723 | -0.761 | 17.85 | 19.0% |
| gemma-7b-it-bnb-4bit | THA | 0.4470 | 0.0744 | -0.636 | 14.71 | 24.8% |
| gemma-7b-it-bnb-4bit | USA | 0.5081 | 0.0657 | -0.816 | 18.13 | 14.5% |
| gemma-7b-it-bnb-4bit | VNM | 0.4012 | 0.0667 | -0.181 | 13.65 | 6.5% |

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-GEMMA_7B vs Vanilla (MIS)

| Model | Country | Vanilla MIS | EXP-24-GEMMA_7B MIS | Δ | Improv% | Win? |
|:------|:-------:|:-----------:|:-------------------:|:-:|:-------:|:----:|
| gemma-7b-it-bnb-4bit | ARG | 0.4040 | 0.3985 | +0.0055 | **+1.37%** | ✅ |
| gemma-7b-it-bnb-4bit | BGD | 0.4754 | 0.5033 | -0.0279 | **-5.86%** | ❌ |
| gemma-7b-it-bnb-4bit | BRA | 0.4162 | 0.3689 | +0.0473 | **+11.36%** | ✅ |
| gemma-7b-it-bnb-4bit | CHN | 0.4209 | 0.4293 | -0.0085 | **-2.01%** | ❌ |
| gemma-7b-it-bnb-4bit | COL | 0.4161 | 0.4097 | +0.0064 | **+1.53%** | ✅ |
| gemma-7b-it-bnb-4bit | DEU | 0.4256 | 0.4323 | -0.0068 | **-1.59%** | ❌ |
| gemma-7b-it-bnb-4bit | ETH | 0.6065 | 0.6121 | -0.0057 | **-0.93%** | ❌ |
| gemma-7b-it-bnb-4bit | GBR | 0.5161 | 0.5251 | -0.0091 | **-1.76%** | ❌ |
| gemma-7b-it-bnb-4bit | IDN | 0.4330 | 0.3951 | +0.0379 | **+8.75%** | ✅ |
| gemma-7b-it-bnb-4bit | IRN | 0.5139 | 0.4925 | +0.0215 | **+4.18%** | ✅ |
| gemma-7b-it-bnb-4bit | JPN | 0.4417 | 0.3803 | +0.0614 | **+13.90%** | ✅ |
| gemma-7b-it-bnb-4bit | KGZ | 0.4810 | 0.5073 | -0.0263 | **-5.47%** | ❌ |
| gemma-7b-it-bnb-4bit | MEX | 0.4237 | 0.4160 | +0.0077 | **+1.83%** | ✅ |
| gemma-7b-it-bnb-4bit | MMR | 0.4664 | 0.4853 | -0.0189 | **-4.04%** | ❌ |
| gemma-7b-it-bnb-4bit | MYS | 0.4396 | 0.4545 | -0.0149 | **-3.39%** | ❌ |
| gemma-7b-it-bnb-4bit | ROU | 0.4998 | 0.5020 | -0.0022 | **-0.44%** | ❌ |
| gemma-7b-it-bnb-4bit | SRB | 0.4998 | 0.5107 | -0.0110 | **-2.19%** | ❌ |
| gemma-7b-it-bnb-4bit | THA | 0.4331 | 0.4470 | -0.0139 | **-3.21%** | ❌ |
| gemma-7b-it-bnb-4bit | USA | 0.5073 | 0.5081 | -0.0007 | **-0.14%** | ❌ |
| gemma-7b-it-bnb-4bit | VNM | 0.3281 | 0.4012 | -0.0731 | **-22.28%** | ❌ |

- **gemma-7b-it-bnb-4bit** Win Rate: **7/20** | Vanilla=0.4574 → EXP-24-GEMMA_7B=0.4590 | Macro Δ: **-0.34%**

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-GEMMA_7B vs EXP-01 SWA-PTIS (MIS)

| Model | Country | EXP-01 MIS | EXP-24-GEMMA_7B MIS | Δ | Improv% | Win? |
|:------|:-------:|:----------:|:-------------------:|:-:|:-------:|:----:|
| gemma-7b-it-bnb-4bit | BRA | 0.3655 | 0.3689 | -0.0034 | **-0.93%** | ❌ |
| gemma-7b-it-bnb-4bit | CHN | 0.4536 | 0.4293 | +0.0243 | **+5.36%** | ✅ |
| gemma-7b-it-bnb-4bit | DEU | 0.3289 | 0.4323 | -0.1034 | **-31.45%** | ❌ |
| gemma-7b-it-bnb-4bit | JPN | 0.4667 | 0.3803 | +0.0864 | **+18.51%** | ✅ |
| gemma-7b-it-bnb-4bit | USA | 0.6038 | 0.5081 | +0.0957 | **+15.85%** | ✅ |

- **gemma-7b-it-bnb-4bit** Win Rate: **3/5** | EXP-01=0.4437 → EXP-24-GEMMA_7B=0.4238 | Macro Δ: **+4.49%**

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-GEMMA_7B Leaderboard Entry

| Rank | Method | Coverage | Mean MIS ↓ | Notes |
|:---:|:-------|:--------:|-----------:|:------|
| ? | **EXP-24-GEMMA_7B** | 1 models × 20 countries | **0.4590** | MIS↓ JSD=0.0697 r=-0.483 Flip=15.2% |

**DPBR summary:** Mean MIS=0.4590, r=-0.483, Flip=15.2%, rel_r=0.974 · *(EXP-09 SOTA: 0.3975 | EXP-24 multi-model ref: 0.3969)*

────────────────────────────────────────────────────────────────────────────────

#### EXP-24-GEMMA_7B Per-Dimension Worst Errors

| Model | Country | Worst Dim | Human | Model | Abs err (pp) |
|:------|:-------:|:----------|:-----:|:-----:|:------------:|
| gemma-7b-it-bnb-4bit | ARG | Age_Young | 75.1 | 50.5 | **24.6** |
| gemma-7b-it-bnb-4bit | BGD | Utilitarianism_More | 75.0 | 45.0 | **30.0** |
| gemma-7b-it-bnb-4bit | BRA | Age_Young | 73.6 | 48.3 | **25.3** |
| gemma-7b-it-bnb-4bit | CHN | Species_Humans | 83.0 | 51.2 | **31.8** |
| gemma-7b-it-bnb-4bit | COL | Age_Young | 76.3 | 50.4 | **25.9** |
| gemma-7b-it-bnb-4bit | DEU | Species_Humans | 82.4 | 52.8 | **29.6** |
| gemma-7b-it-bnb-4bit | ETH | Species_Humans | 93.9 | 49.4 | **44.4** |
| gemma-7b-it-bnb-4bit | GBR | Species_Humans | 79.9 | 49.1 | **30.8** |
| gemma-7b-it-bnb-4bit | IDN | Species_Humans | 77.4 | 50.7 | **26.7** |
| gemma-7b-it-bnb-4bit | IRN | Species_Humans | 84.0 | 48.7 | **35.2** |
| gemma-7b-it-bnb-4bit | JPN | Species_Humans | 79.8 | 53.0 | **26.8** |
| gemma-7b-it-bnb-4bit | KGZ | Species_Humans | 78.7 | 49.4 | **29.3** |
| gemma-7b-it-bnb-4bit | MEX | Age_Young | 75.2 | 50.6 | **24.6** |
| gemma-7b-it-bnb-4bit | MMR | Utilitarianism_More | 78.7 | 47.7 | **31.0** |
| gemma-7b-it-bnb-4bit | MYS | Species_Humans | 76.6 | 50.5 | **26.1** |
| gemma-7b-it-bnb-4bit | ROU | Species_Humans | 80.1 | 51.0 | **29.2** |
| gemma-7b-it-bnb-4bit | SRB | Utilitarianism_More | 75.6 | 46.9 | **28.7** |
| gemma-7b-it-bnb-4bit | THA | Species_Humans | 79.7 | 49.7 | **30.0** |
| gemma-7b-it-bnb-4bit | USA | Species_Humans | 79.2 | 49.8 | **29.4** |
| gemma-7b-it-bnb-4bit | VNM | Age_Young | 72.7 | 47.3 | **25.4** |
