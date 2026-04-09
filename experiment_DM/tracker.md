# SWA-PTIS — experiment_DM Tracker

> **Folder purpose**: Custom experiments built on top of the paper baseline (`experiment/kaggle_experiment.py`).  
> Each script is **self-contained** and Kaggle-ready (auto-bootstrap + pip-install).  
> Read the `Motivation` docstring at the top of each file for full design rationale.
>
> **Run order on Kaggle H100** (EXP-09-based priority): **EXP-19** → **EXP-18** → **EXP-17** → EXP-16 → EXP-15 → EXP-14 → EXP-10  
> **Primary metric**: MIS = L2 misalignment vs human AMCE ↓. Secondary: JSD ↓, Pearson r ↑.  
> **Completed runs**: EXP-01 ✅ (2026-04-09) | EXP-01-SHIS ✅ (2026-04-10) | EXP-02 ✅ (2026-04-09) | EXP-03 ✅ (2026-04-09) | EXP-04 ✅ (2026-04-09) | EXP-05 ✅ (2026-04-09) | EXP-06 ✅ (2026-04-10) | EXP-06b ✅ (2026-04-09) | EXP-07 ✅ (2026-04-10) | EXP-07a ✅ (2026-04-09) | EXP-08 ✅ (2026-04-10) | EXP-09 ✅ (2026-04-09) | EXP-10 ✅ (2026-04-10) | EXP-12 ✅ (2026-04-10) | EXP-13 ✅ (2026-04-10)

---

## File Index

| File | EXP ID | Status | Key Innovation | Fixes |
|:-----|:------:|:------:|:--------------|:------|
| `exp02_expanded_personas.py` | EXP-02 | ✅ DONE | 8 personas (urban/rural split) | More coverage |
| `exp03_socialvalue_personas.py` | EXP-03 | ✅ DONE | Social-utility gradient personas (P4/P5) | SocialValue underestimation |
| `exp04_mistral_crosslingual.py` | EXP-04 | ✅ DONE | English persona override + σ₀=0.8 + K=512 + T=0.5 | Mistral variance collapse |
| `exp05_anchor_regularization.py` | EXP-05 | ✅ DONE | ESS-adaptive anchor α·anchor + (1-α)·base | Gemma over-correction |
| `exp01_stratified_hier_is.py` | EXP-01-SHIS | ✅ DONE | Stratified hierarchical IS + confidence gating + anchor regularization | EXP-09 prior contamination + high flip% |
| `exp06_adaptive_sigma.py` | EXP-06a | ✅ DONE | Per-scenario adaptive σ (entropy-based) | Qwen32B logit collapse |
| `exp06_category_routing.py` | EXP-06b | ✅ DONE | Per-category expert persona pools (6 panels) | Dim-level anchor bias |
| `exp07_best_config_sweep.py` | EXP-07 | ✅ DONE | **Combined EXP-03+04+05+06, 15 countries** | ALL |
| `exp07_wvs_augmentation.py` | EXP-07a | ✅ DONE | Hofstede-neighbor persona augmentation (sparse WVS) | Sparse WVS coverage |
| `exp08_category_routing.py` | EXP-08 | ✅ DONE | Extended category routing (8 panels) | Precision routing |
| `exp09_hierarchical_is.py` | EXP-09 | ✅ DONE | Hierarchical IS with country prior (EMA + annealing) | IS variance / stability |
| `exp10_grand_fusion.py` | EXP-10 | ✅ DONE | **Grand Fusion: EXP-03 + EXP-05 + EXP-09 combined** | ALL (orthogonal fix stack) |
| `exp11_dimension_adaptive_pt.py` | EXP-11 | 🟡 READY | **Dimension-specific κ/σ (heterogeneous PT)** | SocialValue κ↓, Species κ↑ |
| `exp12_contrastive_persona.py` | EXP-12 | ✅ DONE | **Contrastive Persona Decoding (world-avg subtraction)** | Egalitarian anchor bias |
| `exp13_model_adaptive_meta.py` | EXP-13 | ✅ DONE | **Model-adaptive meta-controller (auto-config per family)** | Model-specific failures |
| `exp14_adaptive_kappa.py` | EXP-14 | 🟡 READY | **Direction-Conditioned Adaptive κ (DCAL) + sign consistency** | Mistral anti-corr + flip% |
| `exp15_persona_credibility.py` | EXP-15 | 🟡 READY | **Online Persona Credibility Reweighting (OPCR, per-category)** | SocialValue gap + Mistral bias |
| `exp16_nesterov_is.py` | EXP-16 | 🟡 READY | **Nesterov IS + Progressive PT Sharpening (NMIS)** | IS convergence + flip% damping |
| `exp17_dual_momentum.py` | EXP-17 | 🟡 READY | **Dual-Momentum Prior (fast β=0.20 + slow β=0.03, EXP-09 base)** | Flip% + Mistral early noise |
| `exp18_ess_gated.py` | EXP-18 | 🟡 READY | **ESS-Quality-Gated Prior β_eff=β·ρ + anchor reg (EXP-09 base)** | Mistral collapse + anchor bias |
| `exp19_per_dim_prior.py` | EXP-19 | 🟡 READY | **Per-Dimension Hierarchical Prior — 6 independent EMAs (EXP-09 base)** | SocialValue/Species dim errors |
| `exp20_variance_adaptive_alpha.py` | EXP-20 | 🟡 READY | **Variance-Adaptive Alpha: α_h_eff = α_h_base·exp(-σ_roll/scale) (EXP-09 base)** | BRA + Mistral noisy prior |
| `exp21_directional_noise.py` | EXP-21 | 🟡 READY | **Directional IS Noise: ε_k~N(μ_shift,σ²) + IS correction (EXP-09 base)** | IS sample efficiency + ESS |
| `exp22_adaptive_sigma.py` | EXP-22 | 🟡 READY | **Adaptive IS Sigma from history: σ_eff=f(σ_agents, σ_hist) (EXP-09 base)** | ESS collapse + BRA/Mistral |
| `exp23_category_coherence.py` | EXP-23 | 🟡 READY | **Category-Coherence Reg: blend IS output toward per-cat mean (EXP-09 base)** | Flip% + JSD intra-cat variance |
| `exp24_dual_pass_bootstrap.py` | EXP-24 | 🟡 READY | **Dual-Pass Bootstrap IS: soft r=exp(-bootstrap_var/scale) (EXP-09 base, =K)** | Binary ESS guard + Mistral |
| `exp25_sign_constrained.py` | EXP-25 | 🟡 READY | **Sign-Constrained EMA + Dampening: anti-aligned β·ANTI, damp=0.4 (EXP-09 base)** | Mistral anti-corr + Flip% |

---

## Key Tables (paper-ready)

These are the **most important** tables for fast decision-making and for paper insertion.

### Leaderboard (Mean MIS ↓)

| Rank | Method | Coverage | Mean MIS ↓ | Notes |
|:---:|:-------|:--------:|-----------:|:------|
| 1 | **EXP-09 Hierarchical IS** | 3 models × 5 countries | **0.3975** | Strong overall; Mistral still shows negative correlation risk in some countries |
| 2 | **EXP-10 Grand fusion (03+05+09)** | 3 models × 5 countries | **0.3982** | Near-best MIS; strongest broad improvement vs EXP-01 on Gemma/Mistral, but very high Flip% and diagnostics NaN in anchor-reg fields |
| 3 | **EXP-01-SHIS Stratified hierarchical IS** | 3 models × 5 countries | **0.4156** | Strong on Qwen (5/5 wins vs Vanilla), near-flat on Gemma/Mistral; still behind EXP-09 |
| 4 | **EXP-05 Anchor regularization** | 3 models × 5 countries | **0.4174** | Big gain on Qwen (esp. JPN/DEU); Gemma improves vs EXP-01 on USA/CHN but still not “fixed” |
| 5 | **EXP-13 Model-adaptive meta-controller** | 3 models × 5 countries | **0.4203** | Best known among fully model-aware methods; strong Gemma gains, Qwen mixed, Mistral still anti-correlated |
| 6 | **EXP-06 Adaptive sigma (entropy-calibrated)** | 3 models × 5 countries | **0.4267** | Near-identical to EXP-01 overall; Qwen/Gemma small gains in parts, Mistral remains anti-correlated |
| 7 | EXP-01 SWA-PTIS (4-agent) | 3 models × 5 countries | 0.4269 | Strong for Qwen; harms Gemma/Mistral in some countries |
| 8 | EXP-06b Category routing | 3 models × 5 countries | 0.4269 | Practically identical to EXP-01 (no measurable gain) |
| 9 | EXP-08 Category routing (8 panels) | 3 models × 5 countries | 0.4270 | Nearly identical to EXP-01 aggregate (no measurable gain from routing extension) |
| 10 | EXP-02 Expanded personas (8-agent) | 3 models × 5 countries | 0.4304 | Improves Qwen; regresses Gemma/Mistral |
| 11 | EXP-03 SocialValue personas | 3 models × 5 countries | 0.4413 | Strong gains on Qwen SocialValue (USA/CHN/JPN), but Gemma/Mistral remain problematic and anti-correlation persists in some countries |
| 12 | EXP-04 Mistral cross-lingual | 1 model × 5 countries | 0.4463* | *Only Mistral was run (Qwen/Gemma unchanged vs EXP-01); mixed outcome (CHN/DEU/JPN improved, USA/BRA regressed) |
| 13 | EXP-12 Contrastive persona decoding | 3 models × 5 countries | 0.4517 | Underperforms EXP-01 overall; helps Qwen selectively but regresses Gemma/Mistral |
| 14 | EXP-07a WVS augmentation | 2 models × 5 countries | 0.4031* | *Not directly comparable (missing Mistral) |
| 15 | EXP-07 Unified best-config (15-country) | 3 models × 15 countries | 0.4522* | *Full-sweep setting; despite stronger coverage, headline MIS is not better than EXP-09/05 on 5-country benchmark |
| — | **EXP-14 Adaptive κ (DCAL)** | 3 models × 5 countries | **TBD** | Direction-conditioned κ: κ_low=1.5 (aligned), κ_high=3.5 (anti-aligned) |
| — | **EXP-15 Persona Credibility (OPCR)** | 3 models × 5 countries | **TBD** | Online credibility EMA per (persona,category); anchor = Σ w_i·δ_i |
| — | **EXP-16 Nesterov IS (NMIS)** | 3 models × 5 countries | **TBD** | NAG lookahead anchor + progressive κ annealing 1.80→2.80 |
| — | **EXP-17 Dual-Momentum (DMHP)** | 3 models × 5 countries | **TBD** | Fast EMA (β=0.20) + slow EMA (β=0.03) blended prior on EXP-09 base |
| — | **EXP-18 ESS-Gated Prior (EGPU)** | 3 models × 5 countries | **TBD** | β_eff=β·(k_eff/K) + EXP-05 anchor reg on EXP-09 base |
| — | **EXP-19 Per-Dim Prior (PDHP)** | 3 models × 5 countries | **TBD** | 6 dim-specific EMAs; SV/Species priors independent on EXP-09 base |
| — | **EXP-20 Variance-Adaptive Alpha (VAAA)** | 3 models × 5 countries | **TBD** | α_h_eff = α_h_base·confidence(roll_σ); fixes BRA noisy prior lock-in |
| — | **EXP-21 Directional IS Noise (DISP)** | 3 models × 5 countries | **TBD** | ε_k~N(μ_shift,σ²) biased toward country prior; IS-corrected unbiased |
| — | **EXP-22 Adaptive IS Sigma (AISH)** | 3 models × 5 countries | **TBD** | σ_eff=max(σ_agents, σ_hist·scale+σ₀); annealing explore→exploit |
| — | **EXP-23 Category-Coherence (CCR)** | 3 models × 5 countries | **TBD** | Per-cat running mean blend; target Flip% < 10%, JSD ↓ |
| — | **EXP-24 Dual-Pass Bootstrap (DPBR)** | 3 models × 5 countries | **TBD** | Soft bootstrap reliability r=exp(-Δ²/scale); same K=128 total compute |
| — | **EXP-25 Sign-Constrained EMA (SCED)** | 3 models × 5 countries | **TBD** | Anti-aligned: β·BETA_ANTI + output damp=0.4; target Mistral r > 0 |

> Mean MIS computed as the simple average over the reported (model,country) rows for that method.

### Big Table — MIS vs Vanilla (all methods)

Reference point: **EXP-01 Vanilla** (raw LLM, no personas).  
Notation: `delta = MIS_vanilla - MIS_method` so **positive delta = method improved**.

| Model | Country | Vanilla MIS | EXP-01 SWA | Δ | Improv% | EXP-02 (8-agent) | Δ | Improv% | EXP-03 (SV personas) | Δ | Improv% | EXP-04 (Mistral xling) | Δ | Improv% | EXP-05 (anchor reg) | Δ | Improv% | EXP-06 (adaptive σ) | Δ | Improv% | EXP-07 (best config) | Δ | Improv% | EXP-07a (WVS aug) | Δ | Improv% | EXP-08 (cat routing) | Δ | Improv% | EXP-09 (hier IS) | Δ | Improv% | EXP-10 (grand fusion) | Δ | Improv% | EXP-12 (contrastive) | Δ | Improv% | EXP-13 (meta) | Δ | Improv% |
|:------|:-------:|------------:|-----------:|--:|--------:|-----------------:|--:|--------:|---------------------:|--:|--------:|--------------------:|--:|--------:|------------------:|--:|--------:|------------------:|--:|--------:|--------------------:|--:|--------:|-----------------:|--:|--------:|--------------------:|--:|--------:|-----------------:|--:|--------:|----------------------:|--:|--------:|---------------------:|--:|--------:|----------------:|--:|--------:|
| Qwen2.5-7B | USA | 0.4559 | 0.3677 | +0.0882 | +19.34% | 0.3496 | +0.1063 | +23.31% | **0.2491** | **+0.2069** | **+45.38%** | — | — | — | 0.3628 | +0.0931 | +20.43% | 0.3675 | +0.0884 | +19.39% | 0.3628 | +0.0931 | +20.43% | 0.3687 | +0.0872 | +19.13% | 0.3677 | +0.0882 | +19.34% | 0.3538 | +0.1021 | +22.40% | 0.3165 | +0.1394 | +30.58% | 0.3765 | +0.0794 | +17.41% | 0.3315 | +0.1244 | +27.30% |
| Qwen2.5-7B | CHN ⚠️ | 0.4646 | 0.4078 | +0.0568 | +12.22% | 0.3680 | +0.0966 | +20.79% | 0.2930 | +0.1717 | +36.95% | — | — | — | 0.3791 | +0.0855 | +18.40% | 0.4093 | +0.0553 | +11.90% | 0.3791 | +0.0855 | +18.40% | 0.4094 | +0.0552 | +11.89% | 0.4078 | +0.0568 | +12.22% | 0.3526 | +0.1121 | +24.12% | 0.3529 | +0.1117 | +24.05% | 0.4129 | +0.0517 | +11.13% | 0.3518 | +0.1128 | +24.27% |
| Qwen2.5-7B | JPN | 0.4208 | 0.2802 | +0.1405 | +33.40% | 0.2808 | +0.1400 | +33.26% | 0.2925 | +0.1283 | +30.49% | — | — | — | **0.2493** | **+0.1714** | **+40.72%** | 0.2801 | +0.1407 | +33.43% | **0.2493** | **+0.1714** | **+40.72%** | **0.2801** | +0.1407 | +33.44% | 0.2802 | +0.1405 | +33.40% | 0.3392 | +0.0816 | +19.39% | 0.3435 | +0.0773 | +18.38% | 0.2735 | +0.1473 | +35.00% | 0.3401 | +0.0807 | +19.18% |
| Qwen2.5-7B | DEU | 0.4398 | 0.3424 | +0.0974 | +22.15% | 0.3895 | +0.0503 | +11.43% | 0.3827 | +0.0571 | +12.99% | — | — | — | **0.3140** | **+0.1259** | **+28.61%** | 0.3426 | +0.0972 | +22.10% | **0.3140** | **+0.1259** | **+28.61%** | 0.3444 | +0.0954 | +21.69% | 0.3424 | +0.0974 | +22.15% | 0.4262 | +0.0136 | +3.09% | 0.3988 | +0.0410 | +9.32% | 0.3352 | +0.1046 | +23.79% | 0.3901 | +0.0497 | +11.31% |
| Qwen2.5-7B | BRA | 0.5111 | 0.4025 | +0.1086 | +21.26% | 0.3904 | +0.1207 | +23.62% | 0.4041 | +0.1070 | +20.94% | — | — | — | 0.4493 | +0.0618 | +12.09% | 0.4029 | +0.1082 | +21.17% | 0.4493 | +0.0618 | +12.09% | 0.3792 | +0.1319 | +25.81% | 0.4025 | +0.1086 | +21.26% | 0.3546 | +0.1565 | +30.62% | 0.4394 | +0.0717 | +14.03% | 0.4178 | +0.0933 | +18.26% | 0.4588 | +0.0523 | +10.23% |
| Gemma-2-9B | USA | 0.4647 | 0.6038 | -0.1391 | -29.95% | 0.6073 | -0.1426 | -30.68% | 0.5497 | -0.0850 | -18.30% | — | — | — | 0.5599 | -0.0952 | -20.48% | 0.6045 | -0.1398 | -30.09% | 0.5545 | -0.0898 | -19.32% | 0.6067 | -0.1420 | -30.55% | 0.6038 | -0.1391 | -29.95% | 0.4922 | -0.0275 | -5.91% | 0.4246 | +0.0401 | +8.63% | 0.6683 | -0.2036 | -43.82% | 0.4965 | -0.0318 | -6.85% |
| Gemma-2-9B | CHN ⚠️ | 0.3679 | 0.4536 | -0.0857 | -23.28% | 0.4095 | -0.0416 | -11.30% | 0.6321 | -0.2642 | -71.81% | — | — | — | 0.4002 | -0.0323 | -8.78% | 0.4527 | -0.0848 | -23.05% | 0.4006 | -0.0327 | -8.89% | 0.4517 | -0.0838 | -22.78% | 0.4536 | -0.0857 | -23.28% | 0.3592 | +0.0087 | +2.36% | 0.3466 | +0.0213 | +5.79% | 0.5274 | -0.1595 | -43.35% | 0.3620 | +0.0059 | +1.60% |
| Gemma-2-9B | JPN | 0.4530 | 0.4667 | -0.0136 | -3.01% | 0.5730 | -0.1200 | -26.50% | 0.5265 | -0.0735 | -16.24% | — | — | — | 0.5012 | -0.0482 | -10.63% | 0.4612 | -0.0082 | -1.82% | 0.4774 | -0.0244 | -5.39% | 0.4616 | -0.0086 | -1.90% | 0.4667 | -0.0136 | -3.01% | 0.4411 | +0.0119 | +2.63% | 0.4580 | -0.0050 | -1.10% | 0.4654 | -0.0124 | -2.73% | 0.4660 | -0.0130 | -2.86% |
| Gemma-2-9B | DEU | 0.4170 | **0.3289** | +0.0882 | +21.14% | 0.3418 | +0.0752 | +18.03% | 0.3948 | +0.0222 | +5.33% | — | — | — | 0.3420 | +0.0750 | +17.98% | 0.3285 | +0.0885 | +21.22% | 0.3383 | +0.0787 | +18.87% | **0.3286** | +0.0884 | +21.20% | 0.3289 | +0.0882 | +21.14% | 0.3653 | +0.0517 | +12.39% | 0.3851 | +0.0319 | +7.65% | 0.3533 | +0.0637 | +15.29% | 0.3792 | +0.0378 | +9.07% |
| Gemma-2-9B | BRA | 0.4490 | 0.3655 | +0.0834 | +18.58% | 0.3873 | +0.0617 | +13.74% | 0.4406 | +0.0084 | +1.87% | — | — | — | **0.3446** | **+0.1045** | **+23.27%** | 0.3652 | +0.0838 | +18.66% | 0.3475 | +0.1015 | +22.61% | 0.4002 | +0.0488 | +10.86% | 0.3655 | +0.0834 | +18.58% | 0.3438 | +0.1052 | +23.43% | 0.3320 | +0.1170 | +26.05% | 0.4083 | +0.0407 | +9.07% | 0.3452 | +0.1038 | +23.11% |
| Mistral-7B | USA | 0.5706 | 0.5984 | -0.0278 | -4.87% | 0.6368 | -0.0661 | -11.59% | 0.6666 | -0.0960 | -16.83% | 0.6303 | -0.0597 | -10.46% | — | — | — | 0.5984 | -0.0278 | -4.87% | 0.6235 | -0.0529 | -9.27% | 0.5266 | +0.0440 | +7.72% | 0.5984 | -0.0278 | -4.87% | 0.5266 | +0.0440 | +7.72% | 0.5414 | +0.0292 | +5.11% | 0.6188 | -0.0482 | -8.46% | 0.5970 | -0.0264 | -4.63% |
| Mistral-7B | CHN ⚠️ | 0.4569 | 0.5067 | -0.0498 | -10.90% | 0.5053 | -0.0484 | -10.59% | **0.3091** | **+0.1478** | **+32.35%** | 0.4764 | -0.0195 | -4.26% | — | — | — | 0.5070 | -0.0501 | -10.97% | 0.3573 | +0.0996 | +21.80% | 0.4099 | +0.0470 | +10.29% | 0.5067 | -0.0498 | -10.90% | 0.4099 | +0.0470 | +10.29% | 0.4153 | +0.0416 | +9.10% | 0.5415 | -0.0846 | -18.52% | 0.5069 | -0.0500 | -10.94% |
| Mistral-7B | JPN | 0.3429 | 0.3502 | -0.0073 | -2.12% | 0.3508 | -0.0079 | -2.30% | 0.3221 | +0.0208 | +6.06% | **0.3442** | -0.0013 | -0.37% | — | — | — | 0.3499 | -0.0070 | -2.03% | 0.3776 | -0.0347 | -10.12% | 0.3273 | +0.0156 | +4.55% | 0.3502 | -0.0073 | -2.12% | 0.3273 | +0.0156 | +4.55% | 0.3429 | -0.0000 | -0.01% | 0.3862 | -0.0433 | -12.63% | 0.3495 | -0.0066 | -1.93% |
| Mistral-7B | DEU | 0.4909 | 0.4942 | -0.0033 | -0.67% | 0.5106 | -0.0197 | -4.01% | 0.4091 | +0.0818 | +16.66% | 0.4889 | +0.0020 | +0.41% | — | — | — | 0.4942 | -0.0033 | -0.67% | 0.4762 | +0.0147 | +2.99% | 0.4634 | +0.0275 | +5.60% | 0.4942 | -0.0033 | -0.67% | 0.4634 | +0.0275 | +5.60% | 0.4658 | +0.0251 | +5.12% | 0.5261 | -0.0352 | -7.16% | 0.4947 | -0.0038 | -0.78% |
| Mistral-7B | BRA | 0.4144 | 0.4362 | -0.0217 | -5.25% | 0.4447 | -0.0303 | -7.32% | 0.5246 | -0.1102 | -26.59% | 0.4195 | -0.0051 | -1.22% | — | — | — | 0.4361 | -0.0217 | -5.24% | 0.4872 | -0.0728 | -17.57% | 0.4138 | +0.0006 | +0.13% | 0.4362 | -0.0217 | -5.25% | 0.4138 | +0.0006 | +0.13% | 0.4102 | +0.0042 | +1.00% | 0.4645 | -0.0501 | -12.09% | 0.4353 | -0.0209 | -5.04% |

### Big Table — Full metrics (most complete we have)

This is the “copy-paste into paper” table format with **max available metrics**.  
For Vanilla, only MIS is tracked here (we did not log vanilla JSD/Pearson/MAE in EXP-01).

#### EXP-01 SWA-PTIS (full metrics)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Qwen2.5-7B | USA | 0.3677 | 0.0759 | +0.639 | 9.75 | 1.9% |
| Qwen2.5-7B | CHN ⚠️ | 0.4078 | 0.0956 | +0.418 | 13.54 | 1.3% |
| Qwen2.5-7B | JPN | 0.2802 | 0.0553 | +0.406 | 9.97 | 0.6% |
| Qwen2.5-7B | DEU | 0.3424 | 0.0558 | +0.461 | 10.43 | 2.6% |
| Qwen2.5-7B | BRA | 0.4025 | 0.0942 | +0.167 | 14.30 | 0.6% |
| Gemma-2-9B | USA | 0.6038 | 0.1108 | +0.630 | 22.31 | 0.6% |
| Gemma-2-9B | CHN ⚠️ | 0.4536 | 0.1034 | +0.777 | 14.68 | 1.3% |
| Gemma-2-9B | JPN | 0.4667 | 0.0794 | +0.332 | 15.85 | 1.6% |
| Gemma-2-9B | DEU | 0.3289 | 0.0683 | +0.795 | 10.13 | 1.3% |
| Gemma-2-9B | BRA | 0.3655 | 0.0749 | +0.272 | 14.09 | 3.9% |
| Mistral-7B | USA | 0.5984 | 0.1303 | -0.570 | 21.61 | 0.6% |
| Mistral-7B | CHN ⚠️ | 0.5067 | 0.1050 | -0.682 | 17.41 | 0.3% |
| Mistral-7B | JPN | 0.3502 | 0.0765 | -0.905 | 12.46 | 1.3% |
| Mistral-7B | DEU | 0.4942 | 0.1060 | -0.957 | 17.18 | 1.0% |
| Mistral-7B | BRA | 0.4362 | 0.0947 | -0.665 | 14.02 | 0.3% |

#### EXP-02 Expanded Personas (8-agent, full metrics)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Qwen2.5-7B | USA | 0.3496 | 0.0636 | +0.625 | 10.30 | 0.3% |
| Qwen2.5-7B | CHN | 0.3680 | 0.0842 | +0.343 | 12.56 | 0.6% |
| Qwen2.5-7B | JPN | 0.2808 | 0.0488 | +0.366 | 9.29 | 0.3% |
| Qwen2.5-7B | DEU | 0.3895 | 0.0592 | +0.256 | 12.53 | 1.3% |
| Qwen2.5-7B | BRA | 0.3904 | 0.0880 | -0.010 | 13.20 | 0.3% |
| Gemma-2-9B | USA | 0.6073 | 0.1098 | +0.557 | 22.12 | 0.6% |
| Gemma-2-9B | CHN | 0.4095 | 0.0860 | +0.709 | 14.25 | 1.0% |
| Gemma-2-9B | JPN | 0.5730 | 0.1122 | +0.117 | 18.99 | 0.6% |
| Gemma-2-9B | DEU | 0.3418 | 0.0595 | +0.684 | 11.79 | 1.3% |
| Gemma-2-9B | BRA | 0.3873 | 0.0753 | +0.074 | 14.93 | 2.9% |
| Mistral-7B | USA | 0.6368 | 0.1398 | -0.619 | 23.10 | 1.0% |
| Mistral-7B | CHN | 0.5053 | 0.1049 | -0.665 | 17.60 | 0.3% |
| Mistral-7B | JPN | 0.3508 | 0.0767 | -0.911 | 12.74 | 0.0% |
| Mistral-7B | DEU | 0.5106 | 0.1094 | -0.962 | 18.08 | 0.3% |
| Mistral-7B | BRA | 0.4447 | 0.0973 | -0.696 | 14.39 | 0.3% |

#### EXP-03 SocialValue-Targeted Personas (full metrics)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Qwen2.5-7B | USA | **0.2491** | 0.0388 | +0.704 | **7.56** | 0.0% |
| Qwen2.5-7B | CHN ⚠️ | 0.2930 | 0.0614 | +0.271 | 11.24 | 3.2% |
| Qwen2.5-7B | JPN | 0.2925 | 0.0388 | +0.527 | 10.37 | 1.9% |
| Qwen2.5-7B | DEU | 0.3827 | 0.0569 | +0.004 | 12.77 | 1.3% |
| Qwen2.5-7B | BRA | 0.4041 | 0.0870 | -0.327 | 15.55 | 1.9% |
| Gemma-2-9B | USA | 0.5497 | 0.1199 | +0.397 | 17.63 | 1.6% |
| Gemma-2-9B | CHN ⚠️ | 0.6321 | 0.1622 | +0.445 | 23.05 | 0.6% |
| Gemma-2-9B | JPN | 0.5265 | 0.1050 | +0.171 | 16.39 | 1.0% |
| Gemma-2-9B | DEU | 0.3948 | 0.0760 | +0.420 | 10.86 | 2.6% |
| Gemma-2-9B | BRA | 0.4406 | 0.0993 | -0.235 | 16.38 | 3.5% |
| Mistral-7B | USA | 0.6500 | 0.1476 | -0.565 | 22.31 | 0.6% |
| Mistral-7B | CHN ⚠️ | 0.4925 | 0.1040 | -0.553 | 17.61 | 0.3% |
| Mistral-7B | JPN | 0.3341 | 0.0733 | -0.871 | 12.23 | 0.3% |
| Mistral-7B | DEU | 0.4947 | 0.1060 | -0.942 | 18.85 | 0.0% |
| Mistral-7B | BRA | 0.4434 | 0.0966 | -0.687 | 13.39 | 0.6% |

#### EXP-07a WVS Augmentation (full metrics; Qwen+Gemma only)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Qwen2.5-7B | BRA | 0.3792 | 0.0855 | +0.105 | 13.64 | 1.0% |
| Qwen2.5-7B | JPN | 0.2801 | 0.0551 | +0.406 | 9.93 | 1.0% |
| Qwen2.5-7B | DEU | 0.3444 | 0.0559 | +0.460 | 10.55 | 2.3% |
| Qwen2.5-7B | USA | 0.3687 | 0.0763 | +0.631 | 9.72 | 2.3% |
| Qwen2.5-7B | CHN ⚠️ | 0.4094 | 0.0956 | +0.422 | 13.66 | 0.0% |
| Gemma-2-9B | BRA | 0.4002 | 0.0799 | -0.136 | 14.58 | 1.9% |
| Gemma-2-9B | JPN | 0.4616 | 0.0784 | +0.339 | 15.70 | 1.6% |
| Gemma-2-9B | DEU | 0.3286 | 0.0673 | +0.796 | 10.19 | 0.6% |
| Gemma-2-9B | USA | 0.6067 | 0.1112 | +0.626 | 22.45 | 0.3% |
| Gemma-2-9B | CHN ⚠️ | 0.4517 | 0.1024 | +0.782 | 14.48 | 0.3% |

#### EXP-07 Unified Best-Config Sweep (full metrics; 15 countries × 3 models)

**Overall mean MIS (45 rows): `0.4522`**  
Per-model means from this run:
- Qwen: `0.3742`
- Gemma: `0.5001`
- Mistral: `0.4818`

Five-country slice (for direct comparability with EXP-01/02/03/04/05/09):

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Qwen2.5-7B | USA | 0.3628 | 0.0674 | +0.610 | 10.20 | 6.1% |
| Qwen2.5-7B | CHN ⚠️ | 0.3791 | 0.0885 | +0.372 | 12.35 | 5.2% |
| Qwen2.5-7B | JPN | **0.2493** | **0.0442** | +0.543 | **8.70** | 8.7% |
| Qwen2.5-7B | DEU | **0.3140** | 0.0520 | +0.445 | 10.38 | 17.1% |
| Qwen2.5-7B | BRA | 0.4493 | 0.0965 | -0.218 | 16.67 | 8.1% |
| Gemma-2-9B | USA | 0.5545 | 0.0911 | +0.552 | 19.74 | 4.8% |
| Gemma-2-9B | CHN ⚠️ | 0.4006 | 0.0738 | +0.770 | 12.77 | 13.5% |
| Gemma-2-9B | JPN | 0.4774 | 0.0757 | +0.207 | 16.67 | 5.8% |
| Gemma-2-9B | DEU | 0.3383 | 0.0526 | +0.725 | 11.36 | 10.6% |
| Gemma-2-9B | BRA | 0.3475 | 0.0648 | +0.071 | 12.92 | 17.1% |
| Mistral-7B | USA | 0.6235 | 0.1361 | -0.685 | 23.33 | 5.2% |
| Mistral-7B | CHN ⚠️ | 0.3573 | 0.0782 | -0.051 | 13.08 | 3.5% |
| Mistral-7B | JPN | 0.3776 | 0.0828 | -0.616 | 13.37 | 5.8% |
| Mistral-7B | DEU | 0.4762 | 0.0974 | -0.756 | 17.27 | 4.5% |
| Mistral-7B | BRA | 0.4872 | 0.1124 | -0.695 | 18.61 | 6.8% |

> Full 45-row table (all 15 countries × 3 models) is available in `results/best_config_full_sweep/compare/comparison.csv`.

#### EXP-04 Mistral Cross-Lingual (English personas + σ₀=0.8 + K=512 + T=0.5; Mistral-only)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Mistral-7B | USA | 0.6666 | 0.1538 | -0.339 | 23.33 | 0.3% |
| Mistral-7B | CHN ⚠️ | **0.3091** | 0.0676 | **+0.573** | 10.45 | 0.0% |
| Mistral-7B | JPN | 0.3221 | 0.0745 | +0.461 | 10.65 | 0.3% |
| Mistral-7B | DEU | 0.4091 | **0.0574** | +0.227 | 13.09 | 0.3% |
| Mistral-7B | BRA | 0.5246 | 0.1153 | -0.339 | 20.34 | 1.0% |

#### EXP-09 Hierarchical IS (full metrics)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Qwen2.5-7B | USA | 0.3538 | 0.0505 | +0.685 | 12.23 | 17.7% |
| Qwen2.5-7B | CHN ⚠️ | 0.3526 | 0.0681 | +0.349 | 11.35 | 18.1% |
| Qwen2.5-7B | JPN | 0.3392 | 0.0445 | +0.388 | 11.27 | 14.8% |
| Qwen2.5-7B | DEU | 0.4262 | 0.0526 | +0.357 | 14.56 | 16.5% |
| Qwen2.5-7B | BRA | 0.3546 | 0.0662 | +0.058 | 11.69 | 14.2% |
| Gemma-2-9B | USA | 0.4922 | 0.0556 | +0.675 | 18.32 | 11.9% |
| Gemma-2-9B | CHN ⚠️ | 0.3592 | 0.0494 | +0.784 | 12.78 | 12.9% |
| Gemma-2-9B | JPN | 0.4411 | 0.0565 | +0.346 | 15.56 | 13.5% |
| Gemma-2-9B | DEU | 0.3653 | 0.0461 | +0.772 | 13.19 | 15.8% |
| Gemma-2-9B | BRA | 0.3438 | 0.0493 | +0.336 | 11.49 | 17.7% |
| Mistral-7B | USA | 0.5266 | 0.1031 | -0.569 | 19.76 | 14.5% |
| Mistral-7B | CHN ⚠️ | 0.4099 | 0.0837 | -0.612 | 16.25 | 17.7% |
| Mistral-7B | JPN | 0.3273 | 0.0603 | -0.644 | 12.29 | 16.1% |
| Mistral-7B | DEU | 0.4634 | 0.0837 | -0.884 | 16.06 | 17.7% |
| Mistral-7B | BRA | 0.4138 | 0.0677 | -0.710 | 13.34 | 16.8% |

> EXP-09 has the best mean MIS overall, but it comes with much higher flip% than EXP-01/02 and persistent SocialValue/Species errors.

#### EXP-05 ESS-Adaptive Anchor Regularization (full metrics)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Qwen2.5-7B | USA | 0.3628 | 0.0674 | +0.610 | 10.20 | 6.1% |
| Qwen2.5-7B | CHN ⚠️ | 0.3791 | 0.0885 | +0.372 | 12.35 | 5.2% |
| Qwen2.5-7B | JPN | **0.2493** | **0.0442** | +0.543 | **8.70** | 8.7% |
| Qwen2.5-7B | DEU | **0.3140** | 0.0520 | +0.445 | 10.38 | 17.1% |
| Qwen2.5-7B | BRA | 0.4493 | 0.0965 | -0.218 | 16.67 | 8.1% |
| Gemma-2-9B | USA | 0.5599 | 0.0921 | +0.550 | 19.96 | 4.8% |
| Gemma-2-9B | CHN ⚠️ | 0.4002 | 0.0744 | +0.770 | 12.73 | 13.2% |
| Gemma-2-9B | JPN | 0.5012 | 0.0832 | +0.167 | 17.27 | 7.1% |
| Gemma-2-9B | DEU | 0.3420 | 0.0532 | +0.717 | 11.52 | 11.3% |
| Gemma-2-9B | BRA | **0.3446** | 0.0641 | +0.091 | 12.83 | 16.8% |
| Mistral-7B | USA | 0.6303 | 0.1385 | -0.653 | 23.25 | 5.5% |
| Mistral-7B | CHN ⚠️ | 0.4764 | 0.1003 | -0.657 | 17.12 | 1.6% |
| Mistral-7B | JPN | 0.3442 | 0.0755 | -0.944 | 12.31 | 3.2% |
| Mistral-7B | DEU | 0.4889 | 0.1042 | -0.984 | 17.39 | 4.5% |
| Mistral-7B | BRA | 0.4195 | 0.0822 | -0.689 | 13.51 | 5.8% |

> Note: the run printed `mean_alpha_reg` / `mean_anchor_div` as NaN in `comparison.csv`, indicating those columns were not present in `results_df` (diagnostic logging bug in EXP-05 script).

### Leaderboard (per-dimension “worst gaps” snapshot)

This table is meant for the paper’s diagnostic narrative (“where alignment fails”).

| Method | Model | Country | Worst dim | |err| (pp) | Notes |
|:------|:------|:-------:|:----------|----------:|:------|
| EXP-02 | Qwen | USA | SocialValue_High | 30.4 | SocialValue is consistently top error |
| EXP-07a | Qwen | BRA | SocialValue_High | 27.5 | augmentation helps MIS but not SocialValue |
| EXP-09 | Qwen | USA | SocialValue_High | 28.7 | high flip% (17.7%) despite better MIS |
| EXP-09 | Gemma | USA | Age_Young | 27.3 | age + social-value underestimation |
| EXP-09 | Gemma | DEU | SocialValue_High | 25.4 | SocialValue remains dominant |
| EXP-09 | Mistral | USA | Utilitarianism_More | 30.8 | strongest anti-correlation risk cluster |
| EXP-09 | Mistral | DEU | Species_Humans | 30.7 | high flip% (17.7%) + negative r |
| EXP-09 | Mistral | BRA | Age_Young | 30.0 | high flip% (16.8%) + age underestimation |

> Full per-dimension tables are stored as `CMP_ROOT/per_dim_breakdown.csv` for each run.

---

## Failure Modes Being Addressed

### Insight 1 — SocialValue Underestimation (ALL models)
- **Target file**: `exp03_socialvalue_personas.py`, `exp06_category_routing.py`
- EXP-02 confirmed: mean SocialValue error = **27–30pp** across all models/countries
  - Qwen: SV err ≈ 20–30pp | Gemma: SV err ≈ 17–29pp | Mistral: SV err ≈ 1–17pp
- EXP-02 urban/rural personas did NOT fix SocialValue → 8 agents still under-assign social value
- Root cause confirmed: WVS personas are structurally egalitarian → anchor < 0 for SocialValue.
- EXP-03 ran a **social-utility persona augmentation** (5 personas: 3 WVS + 2 targeted P4/P5):
  - **Qwen**: SocialValue_High error drops meaningfully (e.g., USA 27.0pp → 14.1pp; CHN 20–25pp → 5.5pp).
  - **Gemma/Mistral**: SocialValue improves in some countries but large Age/Utilitarianism errors remain; anti-correlation persists.

### Insight 2 — Mistral Variance Collapse (CONFIRMED & WORSENED in EXP-02)
- **Target file**: `exp04_mistral_crosslingual.py`
- EXP-02 results: JPN variance = **0.065** (collapsed!), Pearson r = **-0.911** (anti-correlated!)
- DEU: Pearson r = **-0.962**, Spearman ρ = **-0.943** → severe ranking inversion
- BRA: Pearson r = **-0.696**, Age_Young err = **37.8pp**
- Urban/rural modulation had no effect on SentencePiece collapse
- EXP-04 ran the planned mitigation (English-only personas + σ₀=0.8 + K=512 + T=0.5) with **mixed** results:
  - **CHN/DEU/JPN**: Pearson r became **positive** and MIS improved substantially vs EXP-01/02.
  - **USA/BRA**: Pearson r remained **negative** and MIS regressed (still unstable / collapse warnings observed).

### Insight 3 — Gemma Over-Correction (Partially improved vs EXP-01)
- **Target file**: `exp05_anchor_regularization.py`
- EXP-02: Gemma DEU MIS = 0.342 (best!), BRA MIS = 0.387 — modest improvement
- But USA MIS = 0.607, JPN MIS = 0.573 remain high
- Specific failure: JPN Utilitarianism_More error = **44.7pp** (model=24.0, human=68.7)
- Root cause: egalitarian anchor << delta_base → over-correction persists with 8 agents
- Fix needed: ESS-adaptive anchor regularization (EXP-05)

### Insight 4 — EXP-02 Net Impact vs EXP-01 (Urban/Rural Axis)
- See the **paper-style MIS comparison blocks** in the EXP-02 section below (format matched to `experiment/kaggle_experiment.py`).
- **Conclusion**: EXP-02 yields a small net gain for Qwen on average, but regresses for Gemma and Mistral. Urban/rural axis alone does **not** reliably reduce MIS; model-specific failure modes remain the dominant drivers (EXP-04, EXP-05).
- **Correction note**: Earlier Insight 4 accidentally compared EXP-02 Qwen JPN to **Mistral** EXP-01 JPN (0.3502) instead of **Qwen** EXP-01 JPN (0.2802). Fixed in the tables below.

---

## Results Log

### EXP-01 — Paper Baseline SWA-PTIS 4-Agent (✅ Completed 2026-04-09)

**Config**: N=4 personas (WVS young/middle/older + utilitarian), λ_coop=0.70, K=128, σ₀=0.30, seed=42, n_scenarios=310 (after quality filter). Both vanilla baseline AND SWA-PTIS run for all 3 models.

#### EXP-01 Vanilla Baseline MIS (no persona, raw LLM)

| Model | Country | Vanilla MIS |
|:------|:-------:|:-----------:|
| Qwen2.5-7B | USA | 0.4559 |
| Qwen2.5-7B | CHN ⚠️ | 0.4646 |
| Qwen2.5-7B | JPN | 0.4208 |
| Qwen2.5-7B | DEU | 0.4398 |
| Qwen2.5-7B | BRA | 0.5111 |
| Gemma-2-9B | USA | 0.4647 |
| Gemma-2-9B | CHN ⚠️ | 0.3679 |
| Gemma-2-9B | JPN | 0.4530 |
| Gemma-2-9B | DEU | 0.4170 |
| Gemma-2-9B | BRA | **0.4490** |
| Mistral-7B | USA | **0.5706** |
| Mistral-7B | CHN ⚠️ | **0.4569** |
| Mistral-7B | JPN | **0.3429** |
| Mistral-7B | DEU | **0.4909** |
| Mistral-7B | BRA | **0.4144** |

> ⚠️ CHN uses Afrikaans fallback data — bug in `data.py`.

#### EXP-01 SWA-PTIS MIS (4-agent, paper pipeline)

| Model | Country | SWA MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:---------:|:-----:|:-----------:|:-----:|:-----:|
| Qwen2.5-7B | USA | **0.3677** | 0.0759 | +0.639 | 9.75 | 1.9% |
| Qwen2.5-7B | CHN ⚠️ | 0.4078 | 0.0956 | +0.418 | 13.54 | 1.3% |
| Qwen2.5-7B | JPN | **0.2802** | 0.0553 | +0.406 | 9.97 | 0.6% |
| Qwen2.5-7B | DEU | 0.3424 | 0.0558 | +0.461 | 10.43 | 2.6% |
| Qwen2.5-7B | BRA | 0.4025 | 0.0942 | +0.167 | 14.30 | 0.6% |
| Gemma-2-9B | USA | 0.6038 | 0.1108 | +0.630 | 22.31 | 0.6% |
| Gemma-2-9B | CHN ⚠️ | 0.4536 | 0.1034 | +0.777 | 14.68 | 1.3% |
| Gemma-2-9B | JPN | 0.4667 | 0.0794 | +0.332 | 15.85 | 1.6% |
| Gemma-2-9B | DEU | **0.3289** | 0.0683 | +0.795 | 10.13 | 1.3% |
| Gemma-2-9B | BRA | 0.3655 | 0.0749 | +0.272 | 14.09 | 3.9% |
| Mistral-7B | USA | 0.5984 | 0.1303 | -0.570 | 21.61 | 0.6% |
| Mistral-7B | CHN ⚠️ | 0.5067 | 0.1050 | -0.682 | 17.41 | 0.3% |
| Mistral-7B | JPN | 0.3502 | 0.0765 | **-0.905** | 12.46 | 1.3% |
| Mistral-7B | DEU | 0.4942 | 0.1060 | **-0.957** | 17.18 | 1.0% |
| Mistral-7B | BRA | 0.4362 | 0.0947 | -0.665 | 14.02 | 0.3% |

#### EXP-01 MIS Comparison: Vanilla → SWA-PTIS

| Model | Country | Vanilla MIS | SWA MIS | Δ | Improv% | Win? |
|:------|:-------:|:-----------:|:-------:|:-:|:-------:|:----:|
| Qwen | USA | 0.4559 | 0.3677 | +0.0882 | **+19.34%** | ✅ |
| Qwen | CHN ⚠️ | 0.4646 | 0.4078 | +0.0568 | **+12.22%** | ✅ |
| Qwen | JPN | 0.4208 | 0.2802 | +0.1405 | **+33.40%** | ✅ |
| Qwen | DEU | 0.4398 | 0.3424 | +0.0974 | **+22.15%** | ✅ |
| Qwen | BRA | 0.5111 | 0.4025 | +0.1086 | **+21.26%** | ✅ |
| Gemma | USA | 0.4647 | 0.6038 | -0.1391 | -29.95% | ❌ |
| Gemma | CHN ⚠️ | 0.3679 | 0.4536 | -0.0857 | -23.28% | ❌ |
| Gemma | JPN | 0.4530 | 0.4667 | -0.0136 | -3.01% | ❌ |
| Gemma | DEU | 0.4170 | 0.3289 | +0.0882 | **+21.14%** | ✅ |
| Gemma | BRA | 0.4490 | 0.3655 | +0.0834 | **+18.58%** | ✅ |
| Mistral | USA | 0.5706 | 0.5984 | -0.0278 | -4.87% | ❌ |
| Mistral | CHN ⚠️ | 0.4569 | 0.5067 | -0.0498 | -10.90% | ❌ |
| Mistral | JPN | 0.3429 | 0.3502 | -0.0073 | -2.12% | ❌ |
| Mistral | DEU | 0.4909 | 0.4942 | -0.0033 | -0.67% | ❌ |
| Mistral | BRA | 0.4144 | 0.4362 | -0.0217 | -5.25% | ❌ |

#### EXP-01 Summary (SWA vs Vanilla)

| Model | Win Rate | Baseline MIS | SWA MIS | Macro Δ% | Micro Δ% |
|:------|:--------:|:------------:|:-------:|:--------:|:--------:|
| Qwen2.5-7B | **5/5** | 0.4584 | 0.3601 | **+21.45%** | **+21.68%** |
| Gemma-2-9B | 2/5 | 0.4303 | 0.4437 | **-3.11%** | **-3.30%** |
| Mistral-7B | 0/5 | 0.4551 | 0.4771 | **-4.83%** | **-4.76%** |

#### EXP-01 Notable Per-Dimension Errors (Qwen SWA)

| Country | Worst Dim | Human | Model | Error |
|:-------:|:---------:|:-----:|:-----:|:-----:|
| USA | SocialValue_High | 67.9 | 33.5 | **34.5pp** |
| CHN ⚠️ | SocialValue_High | 66.7 | 35.3 | **31.4pp** |
| JPN | SocialValue_High | 65.9 | 47.9 | **18.0pp** |
| DEU | SocialValue_High | 64.7 | 42.4 | **22.3pp** |
| BRA | SocialValue_High | 66.3 | 37.4 | **28.9pp** |

> SocialValue underestimation present in ALL countries even for best-performing Qwen.

---

### EXP-02 — Expanded Personas 8-Agent (✅ Completed 2026-04-09)

**Config**: N=8 personas (3 WVS base + urban-young + rural-young + urban-older + utilitarian + global-citizen), λ_coop=0.75, K=256, seed=42, n_scenarios=310 (after quality filter from 500 target).

#### EXP-02 Raw Results

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | Cosine ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:--------:|:-----:|:-----:|
| Qwen2.5-7B | USA | 0.3496 | 0.0636 | +0.625 | +0.625 | 10.30 | 0.3% |
| Qwen2.5-7B | CHN | 0.3680 | 0.0842 | +0.343 | +0.343 | 12.56 | 0.6% |
| Qwen2.5-7B | JPN | **0.2808** | **0.0488** | +0.366 | +0.366 | 9.29 | 0.3% |
| Qwen2.5-7B | DEU | 0.3895 | 0.0592 | +0.256 | +0.256 | 12.53 | 1.3% |
| Qwen2.5-7B | BRA | 0.3904 | 0.0880 | -0.010 | -0.010 | 13.20 | 0.3% |
| Gemma-2-9B | USA | 0.6073 | 0.1098 | +0.557 | +0.557 | 22.12 | 0.6% |
| Gemma-2-9B | CHN | 0.4095 | 0.0860 | +0.709 | +0.709 | 14.25 | 1.0% |
| Gemma-2-9B | JPN | 0.5730 | 0.1122 | +0.117 | +0.117 | 18.99 | 0.6% |
| Gemma-2-9B | DEU | **0.3418** | 0.0595 | +0.684 | +0.684 | 11.79 | 1.3% |
| Gemma-2-9B | BRA | 0.3873 | 0.0753 | +0.074 | +0.074 | 14.93 | 2.9% |
| Mistral-7B | USA | 0.6368 | 0.1398 | -0.619 | -0.619 | 23.10 | 1.0% |
| Mistral-7B | CHN | 0.5053 | 0.1049 | -0.665 | -0.665 | 17.60 | 0.3% |
| Mistral-7B | JPN | 0.3508 | 0.0767 | **-0.911** | **-0.911** | 12.74 | 0.0% |
| Mistral-7B | DEU | 0.5106 | 0.1094 | **-0.962** | **-0.962** | 18.08 | 0.3% |
| Mistral-7B | BRA | 0.4447 | 0.0973 | -0.696 | -0.696 | 14.39 | 0.3% |

#### EXP-02 Per-Dimension Worst Errors (notable findings)

| Model | Country | Worst Dim | Human | Model | Error |
|:------|:-------:|:---------:|:-----:|:-----:|:-----:|
| All | All | SocialValue_High | ~66% | ~40% | **~27pp** |
| Gemma | JPN | Utilitarianism_More | 68.7 | 24.0 | **44.7pp** |
| Mistral | DEU | Species_Humans | 82.4 | 49.4 | **33.0pp** |
| Mistral | USA | Utilitarianism_More | 76.6 | 39.9 | **36.7pp** |
| Qwen | BRA | Age_Young | 73.6 | 49.9 | **23.7pp** |

#### EXP-02 vs EXP-01 Comparison (Qwen, MIS) — corrected

Below is the **paper-style MIS improvement printout** (same layout as `experiment/kaggle_experiment.py`), comparing EXP-01 SWA (4-agent) → EXP-02 SWA (8-agent).  
Notation: `delta = MIS_exp01 - MIS_exp02` so **positive delta = EXP-02 improved**.

##### Qwen2.5-7B — MIS improvement (EXP-01 → EXP-02)

| country | exp01_mis | exp02_mis | delta | improv % | win |
|:------:|----------:|----------:|------:|---------:|:---:|
| USA | 0.3677 | 0.3496 | +0.0181 | +4.92% | ✅ |
| CHN ⚠️ | 0.4078 | 0.3680 | +0.0398 | +9.76% | ✅ |
| JPN | 0.2802 | 0.2808 | -0.0006 | -0.21% | ❌ |
| DEU | 0.3424 | 0.3895 | -0.0471 | -13.76% | ❌ |
| BRA | 0.4025 | 0.3904 | +0.0121 | +3.01% | ✅ |

- Mean exp01 MIS: 0.3601  
- Mean exp02 MIS: 0.3557  
- Absolute reduction: **+0.0045**  
- Improvement on the means (macro-average): **+1.24%**  
- Mean per-country improvement (micro-average): **+0.74%**  
- Wins (MIS lower): **3/5**

##### Gemma-2-9B — MIS improvement (EXP-01 → EXP-02)

| country | exp01_mis | exp02_mis | delta | improv % | win |
|:------:|----------:|----------:|------:|---------:|:---:|
| USA | 0.6038 | 0.6073 | -0.0035 | -0.58% | ❌ |
| CHN ⚠️ | 0.4536 | 0.4095 | +0.0441 | +9.72% | ✅ |
| JPN | 0.4667 | 0.5730 | -0.1063 | -22.78% | ❌ |
| DEU | 0.3289 | 0.3418 | -0.0129 | -3.92% | ❌ |
| BRA | 0.3655 | 0.3873 | -0.0218 | -5.96% | ❌ |

- Mean exp01 MIS: 0.4437  
- Mean exp02 MIS: 0.4638  
- Absolute reduction: **-0.0201**  
- Improvement on the means (macro-average): **-4.53%**  
- Mean per-country improvement (micro-average): **-4.70%**  
- Wins (MIS lower): **1/5**

##### Mistral-7B — MIS improvement (EXP-01 → EXP-02)

| country | exp01_mis | exp02_mis | delta | improv % | win |
|:------:|----------:|----------:|------:|---------:|:---:|
| USA | 0.5984 | 0.6368 | -0.0384 | -6.41% | ❌ |
| CHN ⚠️ | 0.5067 | 0.5053 | +0.0014 | +0.28% | ✅ |
| JPN | 0.3502 | 0.3508 | -0.0006 | -0.17% | ❌ |
| DEU | 0.4942 | 0.5106 | -0.0164 | -3.32% | ❌ |
| BRA | 0.4362 | 0.4447 | -0.0085 | -1.95% | ❌ |

- Mean exp01 MIS: 0.4771  
- Mean exp02 MIS: 0.4896  
- Absolute reduction: **-0.0125**  
- Improvement on the means (macro-average): **-2.62%**  
- Mean per-country improvement (micro-average): **-2.31%**  
- Wins (MIS lower): **1/5**

> **Hypothesis check (Confucian cluster)**: For Qwen, JPN MIS is essentially unchanged (0.2802 → 0.2808), so the “urban–rural expanded pool improves Confucian countries” hypothesis is **not supported** by MIS here.

---

### EXP-06b — Category-Routed Persona Pools (✅ Completed 2026-04-09)

**Script**: `exp06_category_routing.py`  
**Idea**: route 6 category-specific persona pools (Species/Gender/Age/Fitness/SocialValue/Utilitarianism) while keeping EXP-01 hyperparameters unchanged.

#### EXP-06b Raw Results (all 3 models × 5 countries)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | Cosine ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:--------:|:-----:|:-----:|
| Qwen2.5-7B | USA | 0.3677 | 0.0759 | +0.639 | +0.639 | 9.75 | 1.9% |
| Qwen2.5-7B | CHN ⚠️ | 0.4078 | 0.0956 | +0.418 | +0.418 | 13.54 | 1.3% |
| Qwen2.5-7B | JPN | 0.2802 | 0.0553 | +0.406 | +0.406 | 9.97 | 0.6% |
| Qwen2.5-7B | DEU | 0.3424 | 0.0558 | +0.461 | +0.461 | 10.43 | 2.6% |
| Qwen2.5-7B | BRA | 0.4025 | 0.0942 | +0.167 | +0.167 | 14.30 | 0.6% |
| Gemma-2-9B | USA | 0.6038 | 0.1108 | +0.630 | +0.630 | 22.31 | 0.6% |
| Gemma-2-9B | CHN ⚠️ | 0.4536 | 0.1034 | +0.777 | +0.777 | 14.68 | 1.3% |
| Gemma-2-9B | JPN | 0.4667 | 0.0794 | +0.332 | +0.332 | 15.85 | 1.6% |
| Gemma-2-9B | DEU | 0.3289 | 0.0683 | +0.795 | +0.795 | 10.13 | 1.3% |
| Gemma-2-9B | BRA | 0.3655 | 0.0749 | +0.272 | +0.272 | 14.09 | 3.9% |
| Mistral-7B | USA | 0.5984 | 0.1303 | -0.570 | -0.570 | 21.61 | 0.6% |
| Mistral-7B | CHN ⚠️ | 0.5067 | 0.1050 | -0.682 | -0.682 | 17.41 | 0.3% |
| Mistral-7B | JPN | 0.3502 | 0.0765 | -0.905 | -0.905 | 12.46 | 1.3% |
| Mistral-7B | DEU | 0.4942 | 0.1060 | -0.957 | -0.957 | 17.18 | 1.0% |
| Mistral-7B | BRA | 0.4362 | 0.0947 | -0.665 | -0.665 | 14.02 | 0.3% |

#### EXP-06b vs EXP-01 SWA (MIS)

| Model | Mean MIS (EXP-01) | Mean MIS (EXP-06b) | Delta |
|:------|-------------------:|-------------------:|------:|
| Qwen2.5-7B | 0.3601 | 0.3601 | +0.0000 |
| Gemma-2-9B | 0.4437 | 0.4437 | +0.0000 |
| Mistral-7B | 0.4771 | 0.4771 | +0.0000 |
| **Overall (15 rows)** | **0.4269** | **0.4269** | **+0.0000** |

#### EXP-06b key takeaways

- Category-routing run completed successfully, but **aggregate metrics are effectively identical to EXP-01**.
- The expected SocialValue gain (**27pp -> <10pp**) did **not** materialize in this run.
- Most likely action item: verify that routed personas are actually used inside the logits path (prefix build / persona dispatch coupling), not just logged.

---

### EXP-05 — ESS-Adaptive Anchor Regularization (✅ Completed 2026-04-09)

**Script**: `exp05_anchor_regularization.py`  
**Idea**: regularize toward the base model when ESS is low:
\[
\delta_{\text{opt}}=\alpha\cdot \text{anchor}+(1-\alpha)\cdot \delta_{\text{base}}+\delta_\star,\quad
\alpha=\mathrm{clip}(\mathrm{ESS}/K,\rho_{\text{eff}},1)
\]

#### EXP-05 Raw Results (all 3 models × 5 countries)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | Cosine ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:--------:|:-----:|:-----:|
| Qwen2.5-7B | USA | 0.3628 | 0.0674 | +0.610 | +0.610 | 10.20 | 6.1% |
| Qwen2.5-7B | CHN ⚠️ | 0.3791 | 0.0885 | +0.372 | +0.372 | 12.35 | 5.2% |
| Qwen2.5-7B | JPN | **0.2493** | **0.0442** | +0.543 | +0.543 | **8.70** | 8.7% |
| Qwen2.5-7B | DEU | **0.3140** | 0.0520 | +0.445 | +0.445 | 10.38 | 17.1% |
| Qwen2.5-7B | BRA | 0.4493 | 0.0965 | -0.218 | -0.218 | 16.67 | 8.1% |
| Gemma-2-9B | USA | 0.5599 | 0.0921 | +0.550 | +0.550 | 19.96 | 4.8% |
| Gemma-2-9B | CHN ⚠️ | 0.4002 | 0.0744 | +0.770 | +0.770 | 12.73 | 13.2% |
| Gemma-2-9B | JPN | 0.5012 | 0.0832 | +0.167 | +0.167 | 17.27 | 7.1% |
| Gemma-2-9B | DEU | 0.3420 | 0.0532 | +0.717 | +0.717 | 11.52 | 11.3% |
| Gemma-2-9B | BRA | **0.3446** | 0.0641 | +0.091 | +0.091 | 12.83 | 16.8% |
| Mistral-7B | USA | 0.6303 | 0.1385 | -0.653 | -0.653 | 23.25 | 5.5% |
| Mistral-7B | CHN ⚠️ | 0.4764 | 0.1003 | -0.657 | -0.657 | 17.12 | 1.6% |
| Mistral-7B | JPN | 0.3442 | 0.0755 | -0.944 | -0.944 | 12.31 | 3.2% |
| Mistral-7B | DEU | 0.4889 | 0.1042 | -0.984 | -0.984 | 17.39 | 4.5% |
| Mistral-7B | BRA | 0.4195 | 0.0822 | -0.689 | -0.689 | 13.51 | 5.8% |

#### EXP-05 key takeaways

- **Mean MIS (overall 15 rows)**: **0.4174** (2nd best after EXP-09).
- **Qwen**: big wins on JPN/DEU; BRA regresses vs EXP-01.
- **Gemma**: reduces the worst failures (USA/CHN) but not fully fixed.
- **Mistral**: still anti-correlated in some countries; EXP-04 helps CHN/DEU/JPN but does **not** fully fix USA/BRA.
- **Diagnostics missing**: `mean_alpha_reg` / `mean_anchor_div` are **NaN** in `comparison.csv` → need to ensure `results_df` contains `alpha_reg` / `anchor_divergence` columns so run-level means can be computed.

---

### EXP-04 — Mistral Cross-Lingual English Override + Variance Floor Fix (✅ Completed 2026-04-09)

**Script**: `exp04_mistral_crosslingual.py`  
**Idea**: Mitigate Mistral’s multilingual variance collapse by forcing **English personas** (for all countries) and increasing exploration/robustness (σ₀=0.8, K=512, decision T=0.5).

**Run config (this run)**:
- Model: Mistral-7B (Unsloth 4-bit)
- Countries: USA, CHN ⚠️, JPN, DEU, BRA
- n_scenarios: 310 (after quality filter)
- Personas: forced English for all 5 countries (`persona_lang=en`)

#### EXP-04 Raw Results (Mistral-only, 5 countries)

| Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| USA | 0.6666 | 0.1538 | -0.339 | 23.33 | 0.3% |
| CHN ⚠️ | **0.3091** | 0.0676 | **+0.573** | **10.45** | 0.0% |
| JPN | 0.3221 | 0.0745 | +0.461 | 10.65 | 0.3% |
| DEU | 0.4091 | **0.0574** | +0.227 | 13.09 | 0.3% |
| BRA | 0.5246 | 0.1153 | -0.339 | 20.34 | 1.0% |

#### EXP-04 vs EXP-01 SWA-PTIS (Mistral MIS)

`delta = MIS_EXP-01 - MIS_EXP-04` (positive = EXP-04 improved)

| Country | EXP-01 MIS | EXP-04 MIS | Δ | Outcome |
|:-------:|-----------:|-----------:|--:|:--------|
| USA | 0.5984 | 0.6666 | -0.0682 | worse |
| CHN ⚠️ | 0.5067 | **0.3091** | **+0.1976** | win |
| JPN | 0.3502 | 0.3221 | +0.0281 | win |
| DEU | 0.4942 | 0.4091 | +0.0851 | win |
| BRA | 0.4362 | 0.5246 | -0.0884 | worse |

**Key takeaways**:
- **Correlation**: Sign flipped to **positive** for CHN/JPN/DEU, but **still negative** for USA/BRA.
- **Not fully solved**: Many per-scenario logs still emitted “variance collapsed” warnings → English personas + σ₀ floor reduces the worst collapses in some countries but doesn’t reliably stabilize all.

---

### EXP-07 — Unified Best-Config Full Sweep (✅ Completed 2026-04-10)

**Script**: `exp07_best_config_sweep.py`  
**Scope**: 15 countries × 3 models = **45 rows**.

#### EXP-07 headline numbers

- **Overall mean MIS (45 rows)**: **0.4522**
- **Model means**: Qwen **0.3742**, Gemma **0.5001**, Mistral **0.4818**
- **Top 3 country-model rows (lowest MIS)**:
  - Mistral-VNM: **0.2233**
  - Qwen-JPN: **0.2493**
  - Qwen-RUS: **0.2612**

#### EXP-07 vs Vanilla (5-country comparable slice)

`delta = MIS_vanilla - MIS_exp07` (positive = improved vs vanilla)

| Model | USA | CHN ⚠️ | JPN | DEU | BRA |
|:------|----:|-------:|----:|----:|----:|
| Qwen2.5-7B | +20.43% | +18.40% | +40.72% | +28.61% | +12.09% |
| Gemma-2-9B | -19.32% | -8.89% | -5.39% | +18.87% | +22.61% |
| Mistral-7B | -9.27% | +21.80% | -10.12% | +2.99% | -17.57% |

#### EXP-07 key takeaways

- **Qwen** remains strongest and stable across many countries, but SocialValue underestimation is still often the top error.
- **Gemma** improves in CHN/DEU/BRA relative to EXP-01 but still regresses on several countries (notably USA/JPN and multiple 15-country additions).
- **Mistral** remains unstable with many negative Pearson-r rows despite the combined stack; EXP-04 gains do not transfer consistently to the full 15-country setting.
- **Conclusion**: EXP-07 is useful for broad coverage diagnostics, but not the best paper headline on 5-country benchmark (EXP-09/EXP-05 remain stronger there).

---

### EXP-01-SHIS — Stratified Hierarchical IS + Confidence Gating (✅ Completed 2026-04-10)

**Script**: `exp01_stratified_hier_is.py`  
**Scope**: 3 models × 5 countries = **15 rows**.  
**Design**: EXP-09-style hierarchical prior upgraded with category-stratified priors + confidence-gated prior application + ESS-adaptive anchor regularization.

#### EXP-01-SHIS headline numbers

- **Overall mean MIS (15 rows)**: **0.4156**
- **Model means**: Qwen **0.3636**, Gemma **0.4282**, Mistral **0.4551**
- **Reference comparison**: better than EXP-01/06/06b/08 and EXP-12, but still worse than EXP-09 (0.3975), EXP-10 (0.3982), EXP-05 (0.4174 is close), and EXP-13 (0.4203)
- **Global vs Vanilla**: MIS **0.4480 → 0.4156** (**+7.21%**)

#### EXP-01-SHIS full metrics (5-country benchmark)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Qwen2.5-7B | USA | 0.3754 | 0.0723 | +0.599 | 10.86 | 3.5% |
| Qwen2.5-7B | CHN ⚠️ | 0.3991 | 0.0922 | +0.389 | 13.90 | 4.5% |
| Qwen2.5-7B | JPN | **0.2933** | 0.0518 | +0.402 | 9.70 | 6.5% |
| Qwen2.5-7B | DEU | 0.3669 | 0.0551 | +0.444 | 11.49 | 6.1% |
| Qwen2.5-7B | BRA | 0.3832 | 0.0878 | +0.139 | 12.76 | 2.9% |
| Gemma-2-9B | USA | 0.5780 | 0.0982 | +0.635 | 21.15 | 0.6% |
| Gemma-2-9B | CHN ⚠️ | 0.4157 | 0.0882 | +0.780 | 12.69 | 2.9% |
| Gemma-2-9B | JPN | 0.4537 | 0.0700 | +0.323 | 15.71 | 6.5% |
| Gemma-2-9B | DEU | 0.3332 | 0.0643 | +0.782 | 10.21 | 3.2% |
| Gemma-2-9B | BRA | 0.3607 | 0.0710 | +0.282 | 13.66 | 6.1% |
| Mistral-7B | USA | 0.5630 | 0.1215 | -0.557 | 20.70 | 4.5% |
| Mistral-7B | CHN ⚠️ | 0.4787 | 0.1018 | -0.685 | 17.27 | 1.9% |
| Mistral-7B | JPN | 0.3443 | 0.0755 | -0.911 | 12.60 | 1.6% |
| Mistral-7B | DEU | 0.4753 | 0.1004 | -0.968 | 16.63 | 2.3% |
| Mistral-7B | BRA | 0.4143 | 0.0860 | -0.683 | 13.56 | 2.3% |

#### EXP-01-SHIS vs Vanilla (MIS improvement %, positive is better)

| Model | USA | CHN ⚠️ | JPN | DEU | BRA |
|:------|----:|-------:|----:|----:|----:|
| Qwen2.5-7B | +17.66% | +14.10% | +30.30% | +16.57% | +25.03% |
| Gemma-2-9B | -24.40% | -12.98% | -0.15% | +20.11% | +19.67% |
| Mistral-7B | +1.33% | -4.77% | -0.40% | +3.16% | +0.02% |

#### EXP-01-SHIS key takeaways

- **Qwen is the main winner**: 5/5 country wins vs Vanilla and strong macro gain (+20.69%).
- **Gemma becomes near-neutral overall**: large wins in DEU/BRA but still large regressions in USA/CHN.
- **Mistral remains bottlenecked**: mean MIS essentially unchanged vs Vanilla and Pearson-r is still negative in all five countries.
- **Compared to EXP-09**, SHIS-CG does not beat aggregate MIS; confidence gating lowers aggressive prior effects but does not resolve core anti-correlation for Mistral.
- **Usefulness**: valuable targeted-prior ablation and candidate component for next controllers, but not a new SOTA headline.

---

### EXP-06 — Adaptive Sigma via Entropy Calibration (✅ Completed 2026-04-10)

**Script**: `exp06_adaptive_sigma.py`  
**Scope**: 3 models × 5 countries = **15 rows**.  
**Design**: Replace fixed/floored IS proposal sigma with entropy-aware scaling `sigma = clamp(max(sigma_entropy, sigma_agents), floor, ceil)`.

#### EXP-06 headline numbers

- **Overall mean MIS (15 rows)**: **0.4267**
- **Model means**: Qwen **0.3605**, Gemma **0.4424**, Mistral **0.4771**
- **Reference comparison**: essentially tied with EXP-01/EXP-06b (both ~0.4269), worse than EXP-09/10/13, better than EXP-12
- **Behavioral note**: very low flip rate (mostly 0.3–3.5%), but this comes with persistent Mistral anti-correlation and large per-dimension bias

#### EXP-06 full metrics (5-country benchmark)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Qwen2.5-7B | USA | 0.3675 | 0.0759 | +0.638 | 9.72 | 1.6% |
| Qwen2.5-7B | CHN ⚠️ | 0.4093 | 0.0959 | +0.417 | 13.59 | 1.3% |
| Qwen2.5-7B | JPN | **0.2801** | 0.0553 | +0.405 | 9.96 | 1.0% |
| Qwen2.5-7B | DEU | 0.3426 | 0.0558 | +0.460 | 10.44 | 2.3% |
| Qwen2.5-7B | BRA | 0.4029 | 0.0943 | +0.168 | 14.32 | 0.6% |
| Gemma-2-9B | USA | 0.6045 | 0.1109 | +0.629 | 22.34 | 0.6% |
| Gemma-2-9B | CHN ⚠️ | 0.4527 | 0.1031 | +0.778 | 14.66 | 1.0% |
| Gemma-2-9B | JPN | 0.4612 | 0.0776 | +0.341 | 15.73 | 1.0% |
| Gemma-2-9B | DEU | 0.3285 | 0.0682 | +0.796 | 10.09 | 0.6% |
| Gemma-2-9B | BRA | 0.3652 | 0.0749 | +0.273 | 14.07 | 3.5% |
| Mistral-7B | USA | 0.5984 | 0.1303 | -0.570 | 21.61 | 0.6% |
| Mistral-7B | CHN ⚠️ | 0.5070 | 0.1051 | -0.683 | 17.42 | 0.3% |
| Mistral-7B | JPN | 0.3499 | 0.0764 | -0.906 | 12.45 | 1.3% |
| Mistral-7B | DEU | 0.4942 | 0.1060 | -0.957 | 17.18 | 1.0% |
| Mistral-7B | BRA | 0.4361 | 0.0947 | -0.663 | 14.00 | 1.0% |

#### EXP-06 vs Vanilla (MIS improvement %, positive is better)

| Model | USA | CHN ⚠️ | JPN | DEU | BRA |
|:------|----:|-------:|----:|----:|----:|
| Qwen2.5-7B | +19.39% | +11.90% | +33.43% | +22.10% | +21.17% |
| Gemma-2-9B | -30.09% | -23.05% | -1.82% | +21.22% | +18.66% |
| Mistral-7B | -4.87% | -10.97% | -2.03% | -0.67% | -5.24% |

#### EXP-06 vs EXP-01 (MIS improvement %, positive is better)

| Model | USA | CHN ⚠️ | JPN | DEU | BRA |
|:------|----:|-------:|----:|----:|----:|
| Qwen2.5-7B | +0.06% | -0.36% | +0.04% | -0.06% | -0.11% |
| Gemma-2-9B | -0.12% | +0.20% | +1.17% | +0.12% | +0.08% |
| Mistral-7B | +0.00% | -0.06% | +0.08% | +0.00% | +0.02% |

#### EXP-06 key takeaways

- **Main finding**: entropy-calibrated sigma does not materially shift aggregate quality on this 5-country benchmark (0.4267 vs 0.4269 baseline).
- **Qwen/Gemma** are mostly near-parity with small local wins/losses; this looks like a calibration tweak, not a breakthrough method.
- **Mistral remains unstable in direction** (negative Pearson r across all 5 countries), so adaptive sigma alone does not solve anti-correlation.
- **Flip% is very low**, confirming more conservative updates, but low flip does not automatically imply better alignment.
- EXP-06 should be treated as a useful ablation reference and a component candidate for later composite methods, not as a standalone SOTA path.

---

### EXP-08 — Category-Routed Persona Dispatch (✅ Completed 2026-04-10)

**Script**: `exp08_category_routing.py`  
**Scope**: 3 models × 5 countries = **15 rows**.

#### EXP-08 headline numbers

- **Overall mean MIS (15 rows)**: **0.4270**
- **Model means**: Qwen **0.3601**, Gemma **0.4437**, Mistral **0.4771**
- **Best row**: Qwen-JPN (**MIS 0.2802**)
- **Worst row**: Gemma-USA (**MIS 0.6038**)

#### EXP-08 full metrics (5-country benchmark)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Qwen2.5-7B | USA | 0.3677 | 0.0759 | +0.639 | 9.75 | 1.9% |
| Qwen2.5-7B | CHN ⚠️ | 0.4078 | 0.0956 | +0.418 | 13.54 | 1.3% |
| Qwen2.5-7B | JPN | **0.2802** | **0.0553** | +0.406 | 9.97 | 0.6% |
| Qwen2.5-7B | DEU | 0.3424 | 0.0558 | +0.461 | 10.43 | 2.6% |
| Qwen2.5-7B | BRA | 0.4025 | 0.0942 | +0.167 | 14.30 | 0.6% |
| Gemma-2-9B | USA | 0.6038 | 0.1108 | +0.630 | 22.31 | 0.6% |
| Gemma-2-9B | CHN ⚠️ | 0.4536 | 0.1034 | +0.777 | 14.68 | 1.3% |
| Gemma-2-9B | JPN | 0.4667 | 0.0794 | +0.332 | 15.85 | 1.6% |
| Gemma-2-9B | DEU | 0.3289 | 0.0683 | +0.795 | 10.13 | 1.3% |
| Gemma-2-9B | BRA | 0.3655 | 0.0749 | +0.272 | 14.09 | 3.9% |
| Mistral-7B | USA | 0.5984 | 0.1303 | -0.570 | 21.61 | 0.6% |
| Mistral-7B | CHN ⚠️ | 0.5067 | 0.1050 | -0.682 | 17.41 | 0.3% |
| Mistral-7B | JPN | 0.3502 | 0.0765 | -0.905 | 12.46 | 1.3% |
| Mistral-7B | DEU | 0.4942 | 0.1060 | -0.957 | 17.18 | 1.0% |
| Mistral-7B | BRA | 0.4362 | 0.0947 | -0.665 | 14.02 | 0.3% |

#### EXP-08 vs Vanilla (MIS improvement %, positive is better)

| Model | USA | CHN ⚠️ | JPN | DEU | BRA |
|:------|----:|-------:|----:|----:|----:|
| Qwen2.5-7B | +19.34% | +12.22% | +33.40% | +22.15% | +21.26% |
| Gemma-2-9B | -29.95% | -23.28% | -3.01% | +21.14% | +18.58% |
| Mistral-7B | -4.87% | -10.90% | -2.12% | -0.67% | -5.25% |

#### EXP-08 key takeaways

- **No measurable aggregate gain over EXP-01**: overall mean MIS is effectively identical (0.4270 vs 0.4269 baseline SWA table).
- **Qwen stays strong**, but pattern remains the same as EXP-01 (largest residual errors still often on SocialValue).
- **Gemma mixed**, with strong DEU/BRA but major regressions on USA/CHN/JPN.
- **Mistral remains unstable**, with consistently negative Pearson-r across all 5 countries.
- **Conclusion**: the current keyword-routed expert-pool design does not yet outperform simpler baselines; should be treated as an ablation and improved via fusion/meta routing (EXP-10/13).

---

### EXP-07a — WVS Augmentation for Sparse Countries (✅ Completed 2026-04-09)

**Script**: `exp07_wvs_augmentation.py`  
**Idea**: If a country has sparse WVS coverage, augment its persona pool by borrowing “culturally similar” voices from Hofstede-nearest neighbors (kernel-weighted).

**Run config (this run)**:
- Models: Qwen2.5-7B, Gemma-2-9B
- Countries: BRA (sparse) + JPN/DEU/USA/CHN (dense controls)
- N_THRESHOLD=200, K_neighbors=3, tau=0.15
- Persona counts observed: BRA=7 (augmented), dense=4
- Note: CHN still uses Afrikaans fallback MultiTP data (dataset bug) ⚠️

#### EXP-07a Raw Results

| Model | Country | Sparse? | N personas | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:------:|:---------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Qwen2.5-7B | BRA | ✅ | 7 | 0.3792 | 0.0855 | +0.105 | 13.64 | 1.0% |
| Qwen2.5-7B | JPN | ❌ | 4 | **0.2801** | 0.0551 | +0.406 | 9.93 | 1.0% |
| Qwen2.5-7B | DEU | ❌ | 4 | 0.3444 | 0.0559 | +0.460 | 10.55 | 2.3% |
| Qwen2.5-7B | USA | ❌ | 4 | 0.3687 | 0.0763 | +0.631 | 9.72 | 2.3% |
| Qwen2.5-7B | CHN ⚠️ | ❌ | 4 | 0.4094 | 0.0956 | +0.422 | 13.66 | 0.0% |
| Gemma-2-9B | BRA | ✅ | 7 | 0.4002 | 0.0799 | -0.136 | 14.58 | 1.9% |
| Gemma-2-9B | JPN | ❌ | 4 | 0.4616 | 0.0784 | +0.339 | 15.70 | 1.6% |
| Gemma-2-9B | DEU | ❌ | 4 | **0.3286** | 0.0673 | +0.796 | 10.19 | 0.6% |
| Gemma-2-9B | USA | ❌ | 4 | 0.6067 | 0.1112 | +0.626 | 22.45 | 0.3% |
| Gemma-2-9B | CHN ⚠️ | ❌ | 4 | 0.4517 | 0.1024 | +0.782 | 14.48 | 0.3% |

#### EXP-07a vs EXP-01 SWA-PTIS (MIS) — same 5 countries

Notation: `delta = MIS_exp01 - MIS_exp07a` so **positive delta = EXP-07a improved**.

##### Qwen2.5-7B — MIS improvement (EXP-01 → EXP-07a)

| country | exp01_mis | exp07a_mis | delta | improv % | win |
|:------:|----------:|-----------:|------:|---------:|:---:|
| BRA | 0.4025 | 0.3792 | +0.0233 | +5.79% | ✅ |
| JPN | 0.2802 | 0.2801 | +0.0001 | +0.04% | ✅ |
| DEU | 0.3424 | 0.3444 | -0.0020 | -0.60% | ❌ |
| USA | 0.3677 | 0.3687 | -0.0010 | -0.27% | ❌ |
| CHN ⚠️ | 0.4078 | 0.4094 | -0.0016 | -0.39% | ❌ |

##### Gemma-2-9B — MIS improvement (EXP-01 → EXP-07a)

| country | exp01_mis | exp07a_mis | delta | improv % | win |
|:------:|----------:|-----------:|------:|---------:|:---:|
| BRA | 0.3655 | 0.4002 | -0.0347 | -9.49% | ❌ |
| JPN | 0.4667 | 0.4616 | +0.0051 | +1.09% | ✅ |
| DEU | 0.3289 | 0.3286 | +0.0003 | +0.09% | ✅ |
| USA | 0.6038 | 0.6067 | -0.0029 | -0.48% | ❌ |
| CHN ⚠️ | 0.4536 | 0.4517 | +0.0019 | +0.42% | ✅ |

#### Key takeaways
- **BRA (Qwen)**: MIS improves **0.4025 → 0.3792** (**+5.8%**) from augmentation, but **SocialValue_High still dominates error** (≈27.5pp).
- **BRA (Gemma)**: MIS worsens **0.3655 → 0.4002** (augmentation not helpful here).
- **Net**: augmentation helps a subset of sparse-country cases but does **not** resolve the core SocialValue bias; still needs stronger routing/fusion variants (EXP-08/10/13).

---

### EXP-09 — Hierarchical IS (✅ Completed 2026-04-09)

**Script**: `exp09_hierarchical_is.py`  
**Idea**: Add a country-level prior \(\delta_{\text{country}}\) (EMA over scenarios) and anneal toward it to reduce IS drift/instability.

#### EXP-09 MIS (all 3 models × 5 countries)

| Model | Country | MIS ↓ |
|:------|:-------:|:-----:|
| Qwen2.5-7B | USA | 0.3538 |
| Qwen2.5-7B | CHN ⚠️ | 0.3526 |
| Qwen2.5-7B | JPN | 0.3392 |
| Qwen2.5-7B | DEU | 0.4262 |
| Qwen2.5-7B | BRA | 0.3546 |
| Gemma-2-9B | USA | 0.4922 |
| Gemma-2-9B | CHN ⚠️ | 0.3592 |
| Gemma-2-9B | JPN | 0.4411 |
| Gemma-2-9B | DEU | 0.3653 |
| Gemma-2-9B | BRA | 0.3438 |
| Mistral-7B | USA | 0.5266 |
| Mistral-7B | CHN ⚠️ | 0.4099 |
| Mistral-7B | JPN | 0.3273 |
| Mistral-7B | DEU | 0.4634 |
| Mistral-7B | BRA | 0.4138 |

#### EXP-09 Notable per-dimension breakdowns (examples copied from logs)

| Model | Country | Flip% | Pearson r | Worst dim | |err| (pp) |
|:------|:-------:|:-----:|----------:|:----------|----------:|
| Mistral | DEU | 17.7% | -0.884 | Species_Humans | 30.7 |
| Mistral | BRA | 16.8% | -0.710 | Age_Young | 30.0 |

> Full per-dimension tables should be exported from `results/hierarchical_is/compare/per_dim_breakdown.csv` on Kaggle runs.

#### EXP-09 Full metrics (from this Kaggle run)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | Cosine ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:--------:|:-----:|:-----:|
| Qwen2.5-7B | USA | 0.3538 | 0.0505 | +0.685 | +0.685 | 12.23 | 17.7% |
| Qwen2.5-7B | CHN ⚠️ | 0.3526 | 0.0681 | +0.349 | +0.349 | 11.35 | 18.1% |
| Qwen2.5-7B | JPN | 0.3392 | 0.0445 | +0.388 | +0.388 | 11.27 | 14.8% |
| Qwen2.5-7B | DEU | 0.4262 | 0.0526 | +0.357 | +0.357 | 14.56 | 16.5% |
| Qwen2.5-7B | BRA | 0.3546 | 0.0662 | +0.058 | +0.058 | 11.69 | 14.2% |
| Gemma-2-9B | USA | 0.4922 | 0.0556 | +0.675 | +0.675 | 18.32 | 11.9% |
| Gemma-2-9B | CHN ⚠️ | 0.3592 | 0.0494 | +0.784 | +0.784 | 12.78 | 12.9% |
| Gemma-2-9B | JPN | 0.4411 | 0.0565 | +0.346 | +0.346 | 15.56 | 13.5% |
| Gemma-2-9B | DEU | 0.3653 | 0.0461 | +0.772 | +0.772 | 13.19 | 15.8% |
| Gemma-2-9B | BRA | 0.3438 | 0.0493 | +0.336 | +0.336 | 11.49 | 17.7% |
| Mistral-7B | USA | 0.5266 | 0.1031 | -0.569 | -0.569 | 19.76 | 14.5% |
| Mistral-7B | CHN ⚠️ | 0.4099 | 0.0837 | -0.612 | -0.612 | 16.25 | 17.7% |
| Mistral-7B | JPN | 0.3273 | 0.0603 | -0.644 | -0.644 | 12.29 | 16.1% |
| Mistral-7B | DEU | 0.4634 | 0.0837 | -0.884 | -0.884 | 16.06 | 17.7% |
| Mistral-7B | BRA | 0.4138 | 0.0677 | -0.710 | -0.710 | 13.34 | 16.8% |

#### EXP-09 Per-dimension worst error (all model × country)

| Model | Country | Worst dim | |err| (pp) |
|:------|:-------:|:----------|----------:|
| Qwen2.5-7B | USA | SocialValue_High | 28.7 |
| Qwen2.5-7B | CHN ⚠️ | SocialValue_High | 26.7 |
| Qwen2.5-7B | JPN | Species_Humans | 24.2 |
| Qwen2.5-7B | DEU | Species_Humans | 28.4 |
| Qwen2.5-7B | BRA | SocialValue_High | 25.0 |
| Gemma-2-9B | USA | Age_Young | 27.3 |
| Gemma-2-9B | CHN ⚠️ | Age_Young | 21.0 |
| Gemma-2-9B | JPN | Utilitarianism_More | 27.2 |
| Gemma-2-9B | DEU | SocialValue_High | 25.4 |
| Gemma-2-9B | BRA | SocialValue_High | 20.5 |
| Mistral-7B | USA | Utilitarianism_More | 30.8 |
| Mistral-7B | CHN ⚠️ | Species_Humans | 24.6 |
| Mistral-7B | JPN | Species_Humans | 21.6 |
| Mistral-7B | DEU | Species_Humans | 30.7 |
| Mistral-7B | BRA | Age_Young | 30.0 |

---

### EXP-10 — Grand Fusion (✅ Completed 2026-04-10)

**Script**: `exp10_grand_fusion.py`  
**Scope**: 3 models × 5 countries = **15 rows**.  
**Design**: EXP-03 persona pool (3 WVS + 2 social-utility) + EXP-05 ESS-adaptive anchor regularization + EXP-09 hierarchical country prior.

#### EXP-10 headline numbers

- **Overall mean MIS (15 rows)**: **0.3982**
- **Model means**: Qwen **0.3702**, Gemma **0.3893**, Mistral **0.4351**
- **Reference comparison**: slightly worse than EXP-09 (0.3975), better than EXP-01 (0.4269), EXP-08 (0.4270), EXP-12 (0.4517), EXP-13 (0.4203)
- **Diagnostics note**: `alpha_reg` / `anchor_divergence` means are currently `NaN` in exported rows (same diagnostics-export bug family as EXP-05 TODO)

#### EXP-10 full metrics (5-country benchmark)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Qwen2.5-7B | USA | 0.3165 | 0.0366 | +0.718 | 11.24 | 21.3% |
| Qwen2.5-7B | CHN ⚠️ | 0.3529 | 0.0558 | +0.281 | 12.62 | 20.6% |
| Qwen2.5-7B | JPN | 0.3435 | 0.0354 | +0.694 | 12.25 | 21.3% |
| Qwen2.5-7B | DEU | 0.3988 | 0.0506 | +0.276 | 13.63 | 27.4% |
| Qwen2.5-7B | BRA | 0.4394 | 0.0816 | -0.360 | 14.74 | 20.3% |
| Gemma-2-9B | USA | 0.4246 | 0.0505 | +0.502 | 15.03 | 19.7% |
| Gemma-2-9B | CHN ⚠️ | 0.3466 | 0.0460 | +0.575 | 12.76 | 26.1% |
| Gemma-2-9B | JPN | 0.4580 | 0.0607 | +0.053 | 15.79 | 19.0% |
| Gemma-2-9B | DEU | 0.3851 | 0.0449 | +0.587 | 13.45 | 21.6% |
| Gemma-2-9B | BRA | 0.3320 | 0.0536 | -0.063 | 10.60 | 22.9% |
| Mistral-7B | USA | 0.5414 | 0.1030 | -0.605 | 19.63 | 19.4% |
| Mistral-7B | CHN ⚠️ | 0.4153 | 0.0820 | -0.483 | 16.23 | 13.2% |
| Mistral-7B | JPN | 0.3429 | 0.0597 | -0.670 | 12.28 | 14.8% |
| Mistral-7B | DEU | 0.4658 | 0.0822 | -0.848 | 15.77 | 15.8% |
| Mistral-7B | BRA | 0.4102 | 0.0618 | -0.714 | 13.54 | 19.0% |

#### EXP-10 key takeaways

- **Best broad trade-off so far**: EXP-10 reaches **0.3982 MIS**, effectively tied with EXP-09 and clearly above most non-hierarchical methods.
- **Model behavior differs**: Gemma and Mistral improve strongly vs EXP-01; Qwen has mixed transfer (wins on USA/CHN, regressions on JPN/DEU/BRA vs EXP-01).
- **Flip rate remains high**: unlike EXP-01/12/13, EXP-10 runs at ~13–27% Flip, indicating the fusion stack is more dynamic but less stable per scenario.
- **Largest residual errors** are still concentrated on **Species_Humans**, **Age_Young**, and **Utilitarianism_More** for Gemma/Mistral.
- **Diagnostics export still incomplete**: `alpha_reg`/`anchor_divergence` row means are NaN, so diagnostic claims are provisional until export is fixed.

---

### EXP-12 — Contrastive Persona Decoding (✅ Completed 2026-04-10)

**Script**: `exp12_contrastive_persona.py`  
**Scope**: 3 models × 5 countries = **15 rows**.

#### EXP-12 headline numbers

- **Overall mean MIS (15 rows)**: **0.4517**
- **Model means**: Qwen **0.3632**, Gemma **0.4845**, Mistral **0.5074**
- **Reference comparison**: worse than EXP-01 (0.4269), EXP-09 (0.3975), EXP-13 (0.4203)

#### EXP-12 full metrics (5-country benchmark)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Qwen2.5-7B | USA | 0.3765 | 0.0804 | +0.642 | 9.82 | 1.9% |
| Qwen2.5-7B | CHN ⚠️ | 0.4129 | 0.0962 | +0.477 | 13.74 | 1.0% |
| Qwen2.5-7B | JPN | **0.2735** | 0.0572 | +0.423 | 10.05 | 1.6% |
| Qwen2.5-7B | DEU | 0.3352 | 0.0592 | +0.434 | 9.77 | 4.5% |
| Qwen2.5-7B | BRA | 0.4178 | 0.0977 | +0.179 | 15.42 | 1.0% |
| Gemma-2-9B | USA | 0.6683 | 0.1377 | +0.648 | 24.90 | 1.0% |
| Gemma-2-9B | CHN ⚠️ | 0.5274 | 0.1318 | +0.791 | 17.63 | 0.0% |
| Gemma-2-9B | JPN | 0.4654 | 0.0849 | +0.417 | 15.14 | 0.3% |
| Gemma-2-9B | DEU | 0.3533 | 0.0806 | +0.799 | 10.82 | 2.3% |
| Gemma-2-9B | BRA | 0.4083 | 0.0903 | +0.276 | 15.75 | 5.2% |
| Mistral-7B | USA | 0.6188 | 0.1344 | -0.527 | 22.05 | 0.3% |
| Mistral-7B | CHN ⚠️ | 0.5415 | 0.1086 | -0.656 | 18.87 | 0.3% |
| Mistral-7B | JPN | 0.3862 | 0.0821 | -0.828 | 13.33 | 1.9% |
| Mistral-7B | DEU | 0.5261 | 0.1128 | -0.928 | 18.14 | 1.9% |
| Mistral-7B | BRA | 0.4645 | 0.1077 | -0.635 | 14.49 | 1.9% |

#### EXP-12 vs Vanilla (MIS improvement %, positive is better)

| Model | USA | CHN ⚠️ | JPN | DEU | BRA |
|:------|----:|-------:|----:|----:|----:|
| Qwen2.5-7B | +17.41% | +11.13% | +35.00% | +23.79% | +18.26% |
| Gemma-2-9B | -43.82% | -43.35% | -2.73% | +15.29% | +9.07% |
| Mistral-7B | -8.46% | -18.52% | -12.63% | -7.16% | -12.09% |

#### EXP-12 vs EXP-01 (MIS improvement %, positive is better)

| Model | USA | CHN ⚠️ | JPN | DEU | BRA |
|:------|----:|-------:|----:|----:|----:|
| Qwen2.5-7B | -2.40% | -1.25% | +2.38% | +2.11% | -3.80% |
| Gemma-2-9B | -10.69% | -16.27% | +0.28% | -7.41% | -11.70% |
| Mistral-7B | -3.42% | -6.87% | -10.28% | -6.45% | -6.49% |

#### EXP-12 key takeaways

- **Overall underperforms**: mean MIS 0.4517 is below EXP-01/09/13.
- **Qwen is partially robust**: near-parity vs EXP-01 with small wins on JPN/DEU, losses on USA/CHN/BRA.
- **Gemma regresses strongly** in USA/CHN under current contrastive setup.
- **Mistral regresses across all 5 countries** and remains negative Pearson-r.
- **Implementation warning**: `cultural_signal`/`anchor_shift` are `NaN` in logs, so the contrastive diagnostics path likely needs debugging before any CPD claim is publishable.

---

### EXP-13 — Model-Adaptive Meta-Controller (✅ Completed 2026-04-10)

**Script**: `exp13_model_adaptive_meta.py`  
**Scope**: 3 models × 5 countries = **15 rows**.

#### EXP-13 headline numbers

- **Overall mean MIS (15 rows)**: **0.4203**
- **Model means**: Qwen **0.3745**, Gemma **0.4098**, Mistral **0.4767**
- **Reference comparison**: better than EXP-01 (0.4269) and EXP-08 (0.4270), worse than EXP-09 (0.3975)

#### EXP-13 full metrics (5-country benchmark)

| Model | Country | MIS ↓ | JSD ↓ | Pearson r ↑ | MAE ↓ | Flip% |
|:------|:-------:|:-----:|:-----:|:-----------:|:-----:|:-----:|
| Qwen2.5-7B | USA | 0.3315 | 0.0440 | +0.608 | 11.18 | 5.2% |
| Qwen2.5-7B | CHN ⚠️ | 0.3518 | 0.0630 | +0.153 | 12.79 | 8.4% |
| Qwen2.5-7B | JPN | 0.3401 | **0.0390** | +0.547 | 11.87 | 6.8% |
| Qwen2.5-7B | DEU | 0.3901 | 0.0578 | -0.056 | 13.36 | 13.5% |
| Qwen2.5-7B | BRA | 0.4588 | 0.0868 | -0.517 | 16.05 | 6.8% |
| Gemma-2-9B | USA | 0.4965 | 0.0553 | +0.560 | 18.30 | 4.8% |
| Gemma-2-9B | CHN ⚠️ | 0.3620 | 0.0439 | +0.755 | 13.14 | 13.2% |
| Gemma-2-9B | JPN | 0.4660 | 0.0618 | +0.091 | 16.28 | 7.1% |
| Gemma-2-9B | DEU | 0.3792 | **0.0388** | +0.719 | 13.84 | 11.0% |
| Gemma-2-9B | BRA | 0.3452 | 0.0484 | +0.122 | 11.45 | 14.8% |
| Mistral-7B | USA | 0.5970 | 0.1300 | -0.568 | 21.56 | 0.3% |
| Mistral-7B | CHN ⚠️ | 0.5069 | 0.1052 | -0.685 | 17.44 | 0.0% |
| Mistral-7B | JPN | 0.3495 | 0.0763 | -0.901 | 12.44 | 0.6% |
| Mistral-7B | DEU | 0.4947 | 0.1062 | -0.957 | 17.19 | 0.6% |
| Mistral-7B | BRA | 0.4353 | 0.0946 | -0.661 | 13.97 | 1.0% |

#### EXP-13 vs Vanilla (MIS improvement %, positive is better)

| Model | USA | CHN ⚠️ | JPN | DEU | BRA |
|:------|----:|-------:|----:|----:|----:|
| Qwen2.5-7B | +27.30% | +24.27% | +19.18% | +11.31% | +10.23% |
| Gemma-2-9B | -6.85% | +1.60% | -2.86% | +9.07% | +23.11% |
| Mistral-7B | -4.63% | -10.94% | -1.93% | -0.78% | -5.04% |

#### EXP-13 vs EXP-01 (MIS improvement %, positive is better)

| Model | USA | CHN ⚠️ | JPN | DEU | BRA |
|:------|----:|-------:|----:|----:|----:|
| Qwen2.5-7B | +9.86% | +13.73% | -21.37% | -13.92% | -13.99% |
| Gemma-2-9B | +17.77% | +20.19% | +0.16% | -15.29% | +5.55% |
| Mistral-7B | +0.23% | -0.03% | +0.19% | -0.11% | +0.21% |

#### EXP-13 key takeaways

- **Overall**: EXP-13 improves aggregate MIS over EXP-01/08 but does not beat EXP-09.
- **Qwen**: strong vs Vanilla (5/5 wins), but mixed vs EXP-01 with regressions on JPN/DEU/BRA.
- **Gemma**: main winner of EXP-13 design (4/5 wins vs EXP-01; large gains on USA/CHN).
- **Mistral**: near-identical MIS to EXP-01 and still negative Pearson-r in all 5 countries.
- **Conclusion**: model-adaptive tuning is useful, but current profile is still bottlenecked by Mistral-family instability.

## Hyperparameter Differences vs EXP-01

| Param | EXP-01 | EXP-02 | EXP-03 | EXP-04 (Mistral) | EXP-05 | EXP-06a | EXP-06b | EXP-07 | EXP-09 | EXP-12 | EXP-13 |
|:------|:------:|:------:|:------:|:----------------:|:------:|:-------:|:-------:|:------:|:------:|:------:|:------:|
| N personas | 4 | **8** | **5** | 4 | 4 | 4 | 4 | 5 (SV) / 4 (other) | 4 | 4 country + 4 world reference | family-adaptive (Qwen=5, Gemma/Mistral=4) |
| λ_coop | 0.70 | **0.75** | **0.60** | 0.70 | 0.70 | 0.70 | 0.70 | 0.70 | 0.70 | 0.70 | family-adaptive (Qwen=0.60, others=0.70) |
| σ policy | fixed floor 0.30 | fixed floor 0.30 | fixed floor 0.30 | fixed floor 0.80 | fixed floor 0.30 | **entropy-adaptive** `max(σ_entropy, σ_agents)` | fixed floor 0.30 | mixed (Mistral floor 0.80) | fixed floor 0.30 | fixed floor 0.30 | family-adaptive (Q=0.30, G=0.25, M=0.80) |
| K samples | 128 | **256** | 128 | **512** | 128 | 128 | 128 | **512** (Mistral) | 128 | 128 | family-adaptive (Q/G=128, M=512) |
| T_decision | 0.50 | 0.50 | 0.50 | **0.50** | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | family-adaptive (Q/G=1.0, M=0.5) |
| Anchor reg. | ✗ | ✗ | ✗ | ✗ | **✓ ESS-α** | ✗ | ✗ | **✓ ESS-α** | ✗ | ✗ | family-adaptive (Q/G ✓, M ✗) |
| Contrastive term | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ (λ=0.5)** | ✗ |
| Country prior | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ EMA + annealing** | ✗ | ✗ |
| N_warmup | — | — | — | — | — | — | — | — | **50** | — | — |
| Decay tau | — | — | — | — | — | — | — | — | **100** | — | — |
| Beta EMA | — | — | — | — | — | — | — | — | **0.10** | — | — |
| Category routing | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** | **✓** | ✗ | ✗ | ✗ |
| Urban/rural axis | ✗ | **✓** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Global-citizen agent | ✗ | **✓** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ (world ref set)** | ✗ |

---

## EXP-14 — Direction-Conditioned Adaptive Loss Aversion (DCAL)

**Script**: `exp14_adaptive_kappa.py`  
**Status**: 🟡 READY  

### Design Rationale

The paper (§3.2) establishes PT with **fixed κ=2.25** as the load-bearing ingredient. EXP-14 makes κ
**direction-conditioned**:
- After `N_warmup=50` scenarios, the country EMA prior `δ_country` reveals the consistent IS direction.
- When a candidate's direction **aligns** with `δ_country`: use `κ_low=1.50` (less loss-averse) → bigger push.
- When a candidate's direction **opposes** `δ_country`: use `κ_high=3.50` (strong brake) → prevent doubling down on bad direction.

Additionally, a **soft sign consistency penalty** (`λ_sign=0.30`) rewards candidates aligned with country prior direction.

**Annealing**: κ adaptation strength grows with step via `adapt_weight = 1 - exp(-(step-N_warmup)/τ_κ)` so warmup = EXP-01 behaviour.

**Target**: Mean MIS < 0.3800 | Mistral Pearson r > 0 | Flip% < 10%  
**Fixes**: Mistral anti-correlation (brake on wrong direction) + EXP-09 flip% (κ_high penalises sign flips)

---

## EXP-15 — Online Persona Credibility Reweighting (OPCR)

**Script**: `exp15_persona_credibility.py`  
**Status**: 🟡 READY  

### Design Rationale

Paper uses **equal-weight mean** `anchor = (1/N) Σ δ_i`. EXP-15 replaces this with a
**credibility-weighted mean** `anchor = Σ w_i(cat)·δ_i` where weights are updated online.

**Agreement function**: after each scenario with outcome `δ_opt`:  
    `agree_i = exp(-|δ_i - δ_opt| / σ_agree)` (Gaussian agreement, `σ_agree=0.30`)

**EMA update**: `c_i[cat] ← (1-α)·c_i + α·agree_i` where `α=0.15`

**Category-conditioned**: separate credibility vectors per MultiTP category (Species/Gender/Age/Fitness/SocialValue/Utilitarianism). This specifically addresses **SocialValue gap**: after processing SV scenarios, personas that consistently predict toward high-SV human preference gain `c_i[SocialValue]↑`.

**Fixes Mistral anti-correlation**: wrong personas eventually get `c_i → 0`; the dominant (less-wrong) persona gains weight → anchor corrects.

**Target**: Mean MIS < 0.3800 | SocialValue error < 20pp | Mistral Pearson r > 0

---

## EXP-16 — Nesterov Momentum IS + Progressive PT Sharpening (NMIS)

**Script**: `exp16_nesterov_is.py`  
**Status**: 🟡 READY  

### Design Rationale

**Nesterov IS**: borrow Nesterov Accelerated Gradient mechanics for the IS correction loop.
- Maintain momentum `m_t = β_m·m_{t-1} + (1-β_m)·δ*_t` (EMA on IS corrections, `β_m=0.90`)
- IS proposal not centred at `δ̄` but at **Nesterov lookahead**: `anchor_NAG = δ̄ + γ·m_t` (`γ=0.70`)
- When IS consistently corrects in one direction, momentum nudges the next proposal toward that direction **before** sampling → faster convergence + reduced effective IS variance

**Progressive PT Sharpening**:  
    `κ_t = 1.80 + (2.80 - 1.80)·(1 - exp(-t / 150))`  
Early scenarios: `κ=1.80` (explorative). Late scenarios: `κ→2.80` (commitment). Natural curriculum.

**Country prior**: same as EXP-09 (hierarchical EMA annealing), applied post-IS.

**Theory**: This is IS in the spirit of NAG where "gradient" = `δ*`. When `m_t → 0`
(IS consistently returns ~0 correction): NAG reduces to standard IS.
When `m_t → c` (systematic correction): NAG accelerates by factor `1/(1-β_m) = 10`.

**Target**: Mean MIS < 0.3700 (*most ambitious*) | Flip% < 12% | δ*_std < EXP-09

## TODO

- [x] Run EXP-02 on Kaggle H100 (8-agent expanded personas) ✅ 2026-04-09
- [ ] Fix CHN data bug in `data.py` (Afrikaans fallback) — confirmed still present in EXP-02
- [x] Run EXP-03 on Kaggle H100 (SocialValue personas) → **Qwen SocialValue improved; Gemma/Mistral still problematic**
- [x] Run EXP-04 on Kaggle H100 (Mistral cross-lingual) → **mixed** (CHN/DEU/JPN improved; USA/BRA still Pearson<0)
- [x] Run EXP-05 on Kaggle H100 (ESS-adaptive anchor regularization) ✅ 2026-04-09
- [ ] Fix EXP-05 diagnostics export (`alpha_reg`, `anchor_divergence` columns missing → NaN run means)
- [x] Run EXP-01-SHIS on Kaggle H100 (stratified hierarchical IS + confidence gating) → completed (mean MIS=0.4156; strong Qwen gains, Gemma/Mistral mixed)
- [x] Run EXP-06a on Kaggle H100 (adaptive sigma via entropy calibration) → completed (mean MIS=0.4267; near-parity with EXP-01/06b)
- [x] Run EXP-06b on Kaggle H100 (category routing ablation) ✅ 2026-04-09
- [ ] Debug EXP-06b routing effect (results currently mirror EXP-01 almost exactly)
- [x] Run EXP-07 on Kaggle H100 (15 countries × 3 models) → completed (mean MIS=0.4522; mixed by model)
- [x] Run EXP-08 on Kaggle H100 (category-routed expert pools) → completed (mean MIS=0.4270; effectively no gain vs EXP-01)
- [x] Run EXP-10 (Grand Fusion) on Kaggle H100 → completed (mean MIS=0.3982; #2 leaderboard, near EXP-09; diagnostics `alpha_reg` currently NaN)
- [x] Run EXP-12 on Kaggle H100 (contrastive persona decoding) → completed (mean MIS=0.4517; underperforms EXP-01, diagnostics `cultural_signal` currently NaN)
- [x] Run EXP-13 on Kaggle H100 (model-adaptive meta-controller) → completed (mean MIS=0.4203; better than EXP-01/08, below EXP-09)
- [ ] Compute per-dimension MIS from EXP-02 results in analysis script (SocialValue target: err < 10)
- [ ] Update `docs/experiment_tracker.md` with final EXP-07 numbers
- [ ] Update paper §5 results table with EXP-07 as "SWA-PTIS+"
- [ ] Verify EXP-02 JPN Qwen JSD=0.0488 is publishable (best single-country JSD so far)
- [ ] Run EXP-14 (DCAL) — target MIS < 0.3800, Mistral Pearson r > 0
- [ ] Run EXP-15 (OPCR) — target MIS < 0.3800, SocialValue error < 20pp
- [ ] Run EXP-16 (NMIS) — target MIS < 0.3700 (most ambitious)
- [ ] **Run EXP-19 (PDHP) — #1 PRIORITY** — per-dim priors, target MIS < 0.3700, SV err < 20pp
- [ ] Run EXP-18 (EGPU) — #2 PRIORITY — ESS-gated prior + anchor reg, target MIS < 0.3800
- [ ] Run EXP-17 (DMHP) — #3 PRIORITY — dual-momentum prior, target Flip% < 10%
- [ ] Run EXP-20 (VAAA) — variance-modulated alpha, target BRA MIS improved
- [ ] Run EXP-21 (DISP) — directional IS noise, target ESS ↑ vs EXP-09
- [ ] Run EXP-22 (AISH) — adaptive sigma from IS history, target Mistral ESS collapse fixed
- [ ] Run EXP-23 (CCR) — category coherence regularization, target Flip% < 10% and JSD ↓
- [ ] Run EXP-24 (DPBR) — dual-pass bootstrap, target Mistral r > 0 and soft reliability > 0.6
- [ ] **Run EXP-25 (SCED) — sign-constrained EMA + dampening, HIGHEST PRIORITY for Mistral anti-corr fix**
- [ ] After all EXP-17…25 complete: grand ablation table vs EXP-09 (pick best 3 for NeurIPS table)

---

## EXP-17 — Dual-Momentum Hierarchical Prior (DMHP)

**Script**: `exp17_dual_momentum.py`  
**Base**: EXP-09 (Hierarchical IS, SOTA MIS=0.3975)  
**Status**: 🟡 READY  

### Design Rationale

EXP-09 uses a single EMA with β=0.10. The bias-variance tradeoff of single-EMA:
- Large β → noisy (captures instantaneous signal but volatile)
- Small β → smooth (stable prior but slow to correct systematic errors)

**EXP-17** uses **two EMA arms** (Adam-style):
- `m_fast` (β=0.20): captures recent IS correction trends — 2× faster than EXP-09
- `m_slow` (β=0.03): captures long-term drift — 3× slower than EXP-09
- `delta_country = (1-λ_t)·m_fast + λ_t·m_slow`
- `λ_t` anneals from 0 → 0.50 over `τ_blend=80` steps (after N_warmup=50)

Early: weighted toward fast arm (responsive); Late: blend with slow arm (stable).

**Single change vs EXP-09**: `CountryPriorState` → `DualMomentumPriorState` (2 EMAs)

**Target**: MIS < 0.3700 | Flip% < 10% | Mistral r > 0

---

## EXP-18 — ESS-Quality-Gated Prior Update (EGPU)

**Script**: `exp18_ess_gated.py`  
**Base**: EXP-09 (Hierarchical IS, SOTA MIS=0.3975)  
**Status**: 🟡 READY  

### Design Rationale

EXP-09 updates `delta_country` with **fixed β=0.10** regardless of IS quality:
    delta_country ← (1-β)·delta_country + β·delta_opt_micro  [β always 0.10]

When Mistral's IS collapses (k_eff/K ≈ 0.10), delta_opt_micro is unreliable but still
gets full β weight → corrupts the country prior for all subsequent scenarios.

**EXP-18** makes the update quality-gated:
    β_eff = β_base · max(k_eff/K, ρ_floor)  → β_eff ∈ [0.010, 0.10]
High ESS: β_eff ≈ 0.10 (= EXP-09); Low ESS: β_eff ≈ 0.010 (barely moves prior).

**Also integrates EXP-05 anchor reg** before the prior step:
    delta_opt_micro = α_reg·anchor + (1-α_reg)·base + δ*  (EXP-05)
    alpha_reg = clamp(k_eff/K, ρ_eff, 1.0)

**Changes vs EXP-09**: (1) anchor reg before prior; (2) β_eff = β·ρ

**Target**: MIS < 0.3800 | Mistral Flip% < 12% | mean_beta_eff diagnostic

---

## EXP-19 — Per-Dimension Hierarchical Prior (PDHP)

**Script**: `exp19_per_dim_prior.py`  
**Base**: EXP-09 (Hierarchical IS, SOTA MIS=0.3975)  
**Status**: 🟡 READY  

### Design Rationale

EXP-09's single scalar `delta_country` mixes corrections from all six dimensions:
- SocialValue scenarios push `delta_country` up (model under-assigns SV)
- Species scenarios push `delta_country` down (model over-assigns Species)
- Result: two opposite signals CANCEL in the same scalar → neither fixed

**EXP-19** maintains **SIX independent country priors**, one per dimension:
    delta_country[dim] ← (1-β)·delta_country[dim] + β·delta_opt_micro  (dim-specific)
    alpha_h[dim] = 1 - exp(-(n_step[dim] - N_warmup_dim) / τ)  (per-dim warmup)
    delta_opt_final = alpha_h[dim]·delta_country[dim] + (1-alpha_h[dim])·delta_opt_micro

**N_warmup_dim=25** (half of EXP-09's 50): with ~50-80 scenarios per dim in a 310-run,
this gives a similar effective annealing horizon per dimension.

Each dimension independently learns its country-level correction:
- SocialValue prior: large positive (always needs bigger push)
- Species prior: near zero or negative (model already over-assigns)
- Age prior: varies by country (JPN: young; BRA: mixed)

**Single change vs EXP-09**: `CountryPriorState` (1 scalar) → `PerDimPriorState` (6 scalars)

**Outputs bonus diagnostic**: per-dimension final prior table (novel paper contribution)

**Target**: MIS < 0.3700 | SocialValue err < 20pp | Mistral r > 0
