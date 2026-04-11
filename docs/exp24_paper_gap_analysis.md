# EXP-24 (DPBR) vs `SWA_MPPI_paper/paper_revised.tex` — gap analysis

The LaTeX draft describes **SWA-PTIS** (single-pass PT-IS + binary ESS guard + optional extensions in discussion). **EXP-24** replaces the binary ESS gate with **dual-pass bootstrap reliability** \(r=\exp(-(\delta^\star_1-\delta^\star_2)^2/\texttt{VAR\_SCALE})\) on the **same** debiased logit gaps and PT utility inside each IS pass (`experiment_DM/exp24_dpbr_core.py`).

Below: limitations and theory caveats **stated in the paper**, whether **DPBR inherits them**, and **concrete improvements** toward a stronger (oral-grade) method story.

---

## A. Theory & estimator (§method + appendix IS discussion)

| Paper point | Meaning | Applies to EXP-24? | Notes / improvement |
|-------------|---------|-------------------|---------------------|
| Self-normalised IS is **biased at finite \(K\)**; bias \(O(1/K)\); **no explicit finite-\(K\) bound** | Headline claims are empirical, not concentration guarantees | **Yes** — two passes use **same** \(K/2\) per pass; total \(K=128\) unchanged | Report **variance across seeds** or bootstrap over IS noise; optional **larger \(K\)** ablation (`EXP24_K_HALF`). |
| PT **kink at 0** → **non-smooth** \(U\); weights can **jump** when \(g\) crosses 0 | Extra variance not analysed closed-form | **Yes** | Same PT kernel; dual-pass \(r\) **downweights** inconsistent \(\delta^\star\) but does **not** remove kink pathology. |
| **Debias before PT-IS**; averaging **after** nonlinear step would **not** cancel bias | Implementation order matters | **Same pipeline** if controller debias matches paper | Keep **one** debiased \((\delta_{\text{base}},\delta_i)\) then IS (already true in `Exp24DualPassController.predict`). |
| Additive positional bias = **first-order approximation** | Real transformers may deviate | **Yes** | Same as SWA-PTIS; report **positional_bias** diagnostics if exposed per run. |

---

## B. Limitations (main text §Discussion)

| Paper limitation | Applies to DPBR? | Improvement direction |
|------------------|------------------|------------------------|
| **Hyperparameter validation** (\(\lambda_{\text{coop}}, T_{\text{dec}}, T_{\text{cat}}\); \(\sigma_0\); \(\rho_{\text{eff}}\)) | **Yes** — EXP-24 uses same SWAConfig knobs + **`VAR_SCALE`** | **Sensitivity table** for `VAR_SCALE` and (if used) `EXP24_K_HALF`; document selection protocol (holdout / grid). |
| **SocialValue egalitarian-anchor bias** (WVS personas don’t encode social-utility gradients) | **Yes** — same personas/AMCE target | **Category-routed expert personas** (`experiment_DM/exp08_category_routing.py`); **Social-Utility** persona extension described in paper. |
| **ESS-adaptive anchor regularization** (theorem when base closer to human than anchor, ESS low) | **Not in core DPBR** — `exp24_dpbr_core` uses standard \(\delta_{\text{opt}}=\bar\delta+\delta^\star\) then hierarchical prior | **Compose** DPBR with **EXP-05 / grand-fusion** regularised update, or add optional flag after careful validation (see `experiment_DM/exp05_anchor_regularization.py`). |
| **Instruction-tuned over-correction** (anchor vs base direction) | **Yes** | Same **ESS-REG** idea; or **reduce** steering when \(r\) small (DPBR already shrinks \(\delta^\star\)). |
| **Cross-lingual persona collapse** (e.g. Mistral + JP) | **Yes** — model/tokenizer issue | **Entropy-aware \(\sigma\)**, English personas, higher \(K\) (paper’s fix); not solved by DPBR alone. |
| **JSD vs MIS**; JSD not magnitude-fair on simplex | **Yes** — same metrics in pipeline | Keep **MIS primary** in writing; report **per-dimension MAE** (already in code paths). |
| **Utilitarianism MPR confound** (intercept vs slope) | **Yes** | Report **`compute_utilitarianism_slope`** alongside JSD for Util rows. |
| **Consensus vs utilitarian** weighting not fully dissected | **Yes** | Ablate \(\lambda_{\text{coop}}\); consider **category-routed** pools. |
| **Ground truth = AMCE**, not human “felt” legitimacy; **single seed** | **Yes** | Multi-seed / CI for **algorithmic** noise; honest **user study** sentence in discussion. |

---

## C. Appendix “Extended Limitations”

| Item | DPBR |
|------|------|
| Single seed; IS stochasticity not fully characterised | **Worse visibility** — **two** random passes per scenario → document **seed sweep** or report **var(reliability_r)** across reruns. |
| Pearson \(r\) on 6-D vector noisy | Same |
| **WVS→trolley** indirect mapping | Same personas |
| **Inter-persona reward** \(r_i \approx \delta_i-\delta_{\text{base}}\) may misalign with human target | Same — DPBR does not fix wrong personas |
| 15/100+ countries, preprocessing dependence | Same evaluation scope |
| **Binary** logits only | Same |
| Language–culture conflation | Same |
| **Quantisation** logit artefacts | Same — preflight does not fix |
| **Qwen2.5-32B** concentrated logits / off-manifold IS | Same failure mode — DPBR **does not** replace need for **adaptive \(\sigma\)** (`experiment_DM/exp22_adaptive_sigma.py` etc.) |

---

## D. What EXP-24 already improves vs vanilla SWA-PTIS (paper Alg. 1)

- **Soft reliability** instead of **binary** \(K_{\text{eff}}/K < \rho_{\text{eff}}\) **only** for the *combined* \(\delta^\star\): each pass still uses the **per-pass ESS guard**; disagreement **continuously** downweights the blended update.
- **Documented** ablation hooks: `EXP24_VAR_SCALE`, `EXP24_K_HALF`, tests in `tests/test_exp24_dpbr.py`, `docs/exp24_reproducibility.md`.

---

## E. Prioritised roadmap (oral-level method + evaluation)

1. **Theory/write-up:** State clearly that DPBR is a **heuristic variance-based reliability** on **two IS replicates**, not a **unbiased** IS estimator; connect to **bootstrap** intuition (already in `exp24_dual_pass_bootstrap.py` header).
2. **Experiments:** `VAR_SCALE` sensitivity; **multi-seed** (e.g. 5 seeds) on **one** model × 5 countries for **CIs** on MIS.
3. **Failure modes:** Reuse paper’s **ESS-REG** when \(\delta_{\text{base}}\) is empirically closer to human than \(\bar\delta\) on a **validation** slice — implement as **optional** branch or **separate** `exp_model` entry that imports `exp24_dpbr_core` + reg controller.
4. **Personas:** Pilot **EXP-08**-style routing for **SocialValue** + **Species** where paper shows largest gaps.
5. **Metrics:** Always pair Utilitarianism JSD with **slope diagnostic** in tables where claims touch “utilitarianism”.

This file is the **audit trail** linking the SWA-PTIS manuscript’s stated limits to the **current DPBR** codebase and to **existing** experiments (`exp05`, `exp08`, `exp22`, …) that address specific gaps.
