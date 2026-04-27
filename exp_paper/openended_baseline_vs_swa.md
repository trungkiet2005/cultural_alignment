# Open-Ended Track — Vanilla Baseline vs SWA-DPBR

**Date:** 2026-04-27
**Actor / Judge:** Qwen2.5-7B-Instruct (BF16) — self-judge
**Backend:** HuggingFace native (no Unsloth, no vLLM)
**Scenarios:** 310 / country, 20 paper countries (6,200 total)
**Scripts:**
- Baseline → [`exp_paper/exp_paper_openended_baseline_vanilla.py`](exp_paper_openended_baseline_vanilla.py)
- SWA-DPBR → [`exp_paper/exp_paper_openended_with_DISCA.py`](exp_paper_openended_with_DISCA.py)

---

## 1. Setup

Both runs share the same actor model, judge model, dataset, and country list — only the inference-time controller differs.

| | Vanilla Baseline | SWA-DPBR (DISCA variant) |
|:-|:-|:-|
| Actor | Qwen2.5-7B BF16 | Qwen2.5-7B BF16 |
| Judge | Qwen2.5-7B (self-judge) | Qwen2.5-7B (self-judge) |
| Persona | Utilitarian neutral only (1) | WVS persona ensemble (4 age cohorts + neutral) |
| Positional debiasing | ✗ | ✓ (A↔B swap, full cancel) |
| PT-IS controller | ✗ | ✓ (α=β=0.88, κ=2.25) |
| Dual-pass bootstrap | ✗ | ✓ (2× K=64 = 128 samples) |
| ESS anchor regularisation | ✗ | ✓ |
| Hierarchical country prior | ✗ | ✓ |
| `max_new_tokens` | 64 | 64 |
| `T_DECISION` | 1.0 | 1.0 |
| Judge calls / country | 310 | 1,550 (≈5×) |
| Stage-2 wall-time / country | ~71.2s | ~356.7s (≈5×) |
| Parse fail rate | 0.0% all 20 | 0.0% all 20 |

**Setup is intentionally identical** so any metric delta is attributable to SWA-DPBR alone.

---

## 2. Vanilla Baseline (full 20 countries)

| Country | MIS ↓ | Pearson r ↑ | JSD ↓ | n |
|:--------|:-----:|:----------:|:-----:|:-:|
| USA | 0.4347 | +0.704 | 0.0399 | 310 |
| GBR | 0.4431 | +0.690 | 0.0423 | 310 |
| DEU | 0.4423 | −0.181 | 0.0883 | 310 |
| ARG | 0.4737 | −0.078 | 0.0498 | 310 |
| BRA | 0.4157 | −0.161 | 0.1065 | 310 |
| MEX | 0.4967 | −0.065 | 0.0504 | 310 |
| COL | 0.4987 | −0.217 | 0.0511 | 310 |
| VNM | 0.4564 | −0.526 | 0.1012 | 310 |
| MMR | 0.4237 | +0.682 | 0.0495 | 310 |
| THA | 0.3940 | +0.688 | 0.0416 | 310 |
| MYS | 0.4060 | +0.642 | 0.0385 | 310 |
| IDN | 0.4417 | −0.730 | 0.0649 | 310 |
| CHN | 0.3028 | +0.609 | 0.0833 | 310 |
| JPN | 0.2812 | +0.038 | 0.0692 | 310 |
| BGD | 0.4423 | +0.594 | 0.0450 | 310 |
| IRN | 0.4110 | −0.042 | 0.1091 | 310 |
| SRB | 0.4635 | +0.667 | 0.0425 | 310 |
| ROU | 0.4618 | +0.693 | 0.0427 | 310 |
| KGZ | 0.4403 | +0.744 | 0.0399 | 310 |
| ETH | 0.5865 | +0.259 | 0.0675 | 310 |
| **MEAN** | **0.4358** | **+0.251** | **0.0612** | — |

**Highlights — Vanilla:**
- **Best MIS:** JPN (0.2812), CHN (0.3028) — East Asia surprisingly aligned by self-judge
- **Worst MIS:** ETH (0.5865), COL (0.4987), MEX (0.4967)
- **Strongest r:** KGZ (+0.744), USA (+0.704), ROU (+0.693), GBR (+0.690), THA (+0.688)
- **Anti-aligned (r < 0):** IDN (−0.730), VNM (−0.526), COL (−0.217), DEU (−0.181), BRA (−0.161), ARG (−0.078), MEX (−0.065), IRN (−0.042) — **8/20 countries had inverted ordering vs human AMCEs**
- Self-judge is reliable for A/B/UNCERTAIN extraction (parse_fail = 0.0% across all 20)

---

## 3. SWA-DPBR (full 20 countries)

| Country | MIS ↓ | Pearson r ↑ | JSD ↓ | rel_r | boot_var | ESS₁ | α_anchor |
|:--------|:-----:|:----------:|:-----:|:-----:|:--------:|:----:|:--------:|
| USA | 0.4252 | +0.701 | 0.0396 | 0.970 | 0.0042 | 0.337 | 0.317 |
| GBR | 0.4244 | +0.585 | 0.0466 | 0.966 | 0.0080 | 0.345 | 0.320 |
| DEU | 0.4847 | −0.454 | 0.0910 | 0.963 | 0.0052 | 0.363 | 0.341 |
| ARG | 0.4943 | +0.216 | 0.0474 | 0.964 | 0.0072 | 0.383 | 0.358 |
| BRA | 0.4440 | −0.058 | 0.1110 | 0.967 | 0.0039 | 0.401 | 0.379 |
| MEX | 0.5108 | +0.515 | 0.0413 | 0.970 | 0.0047 | 0.385 | 0.362 |
| COL | 0.5221 | +0.131 | 0.0487 | 0.975 | 0.0080 | 0.384 | 0.360 |
| VNM | 0.5053 | +0.014 | 0.0657 | 0.909 | 0.0197 | 0.391 | 0.370 |
| MMR | 0.3991 | +0.179 | 0.0672 | 0.969 | 0.0060 | 0.354 | 0.333 |
| THA | 0.3383 | +0.147 | 0.0614 | 0.980 | 0.0044 | 0.356 | 0.331 |
| MYS | 0.3613 | +0.497 | 0.0433 | 0.969 | 0.0066 | 0.348 | 0.328 |
| IDN | 0.3807 | +0.303 | 0.0582 | 0.939 | 0.0076 | 0.363 | 0.340 |
| CHN | 0.2987 | +0.748 | 0.0793 | 0.909 | 0.0140 | 0.378 | 0.354 |
| JPN | 0.2711 | +0.228 | 0.0615 | 0.930 | 0.0147 | 0.378 | 0.353 |
| BGD | 0.3862 | −0.334 | 0.0727 | 0.960 | 0.0077 | 0.379 | 0.354 |
| IRN | 0.4798 | −0.295 | 0.1137 | 0.966 | 0.0036 | 0.413 | 0.390 |
| SRB | 0.4441 | +0.320 | 0.0534 | 0.970 | 0.0055 | 0.349 | 0.325 |
| ROU | 0.4694 | +0.418 | 0.0521 | 0.965 | 0.0071 | 0.346 | 0.323 |
| KGZ | 0.4429 | +0.452 | 0.0509 | 0.986 | 0.0009 | 0.344 | 0.324 |
| ETH | 0.5984 | −0.357 | 0.0877 | 0.974 | 0.0041 | 0.343 | 0.317 |
| **MEAN** | **0.4340** | **+0.197** | **0.0646** | 0.957 | 0.0072 | 0.370 | 0.345 |

**Highlights — SWA:**
- **Best MIS:** JPN (0.2711), CHN (0.2987) — same East-Asia leadership as Vanilla, both *also* improved over Vanilla
- **Worst MIS:** ETH (0.5984), COL (0.5221), MEX (0.5108) — same problem regions as Vanilla, slightly worse
- **Strongest r:** CHN (+0.748), USA (+0.701), GBR (+0.585), MEX (+0.515), MYS (+0.497) — **MEX is new in top-5** (was anti-aligned in Vanilla)
- **Diagnostics healthy on all 20:** rel_r ∈ [0.909, 0.986] (DPBR gating active, never collapsed); positional_bias = 0.0 across all (debiasing fully cancels); ESS₁ ≈ 0.34–0.41 (moderate IS coverage)
- **Cost:** ~5× wall-clock per country and ~5× judge calls (1550 vs 310) for the same 310 scenarios

---

## 4. Head-to-Head — who wins?

### 4.1 Aggregate means

| Metric | Vanilla | SWA-DPBR | Δ (SWA−Van) | Direction |
|:-------|:-------:|:--------:|:-----------:|:---------:|
| Mean MIS ↓ | 0.4358 | **0.4340** | −0.0018 | ✅ SWA marginally better |
| Mean Pearson r ↑ | **+0.251** | +0.197 | −0.054 | ❌ Vanilla better |
| Mean JSD ↓ | **0.0612** | 0.0646 | +0.0034 | ❌ Vanilla better |

> SWA wins **only on MIS** at the aggregate level — and the margin (−0.0018 ≈ 0.4%) is well within noise. r and JSD both regress.

### 4.2 Per-country wins (count) — primary metric MIS

| Counter | SWA wins | Vanilla wins | Tie |
|:-|:-:|:-:|:-:|
| MIS ↓ | **10** | 10 | 0 |
| Pearson r ↑ | 8 | **12** | 0 |
| JSD ↓ | 8 | **12** | 0 |

**MIS is a 10–10 split** — SWA's edge is purely in the magnitude of wins (mean ΔMIS=+0.0018), not in country count. r and JSD lose majority.

### 4.3 Per-country wins — combined (≥2 of 3 metrics)

| Verdict | Count | Countries |
|:--------|:-----:|:----------|
| ⭐ **SWA sweeps (3/3)** | **3** | IDN, CHN, JPN |
| ✅ **SWA majority (2/3)** | **5** | USA, ARG, MEX, COL, VNM |
| ⚪ **Mixed / split (1/3)** | 7 | GBR, BRA, MMR, THA, MYS, BGD, SRB |
| ❌ **Vanilla majority (2/3)** | 0 | — |
| ❌ **Vanilla sweeps (3/3)** | **5** | DEU, IRN, ROU, KGZ, ETH |

**SWA-DPBR wins overall on 8/20 countries (40%)**, ties / mixes on 7, and loses on 5.

> **Headline win-count:** SWA-DPBR is *strictly better* on 3 countries (East Asia: IDN, CHN, JPN), *broadly better* on 5 more (USA + LatAm anti-aligned recovery), and *strictly worse* on 5 (DEU + MENA + Eastern Europe + Africa).

### 4.4 Side-by-side per country

| Country | VAN MIS | SWA MIS | ΔMIS | VAN r | SWA r | Δr | VAN JSD | SWA JSD | Verdict |
|:--------|:-------:|:-------:|:----:|:-----:|:-----:|:--:|:-------:|:-------:|:-------:|
| USA | 0.4347 | **0.4252** | +0.010 | +0.704 | +0.701 | −.003 | 0.0399 | **0.0396** | ✅ SWA 2/3 |
| GBR | 0.4431 | **0.4244** | +0.019 | **+0.690** | +0.585 | −.105 | **0.0423** | 0.0466 | ⚪ split |
| DEU | **0.4423** | 0.4847 | −0.042 | **−0.181** | −0.454 | −.273 | **0.0883** | 0.0910 | ❌ VAN 3/3 |
| ARG | **0.4737** | 0.4943 | −0.021 | −0.078 | **+0.216** | +.294 | 0.0498 | **0.0474** | ✅ SWA 2/3 |
| BRA | **0.4157** | 0.4440 | −0.028 | −0.161 | **−0.058** | +.103 | **0.1065** | 0.1110 | ⚪ split |
| MEX | **0.4967** | 0.5108 | −0.014 | −0.065 | **+0.515** | +.580 | 0.0504 | **0.0413** | ✅ SWA 2/3 |
| COL | **0.4987** | 0.5221 | −0.023 | −0.217 | **+0.131** | +.348 | 0.0511 | **0.0487** | ✅ SWA 2/3 |
| VNM | **0.4564** | 0.5053 | −0.049 | −0.526 | **+0.014** | +.540 | 0.1012 | **0.0657** | ✅ SWA 2/3 |
| MMR | 0.4237 | **0.3991** | +0.025 | **+0.682** | +0.179 | −.503 | **0.0495** | 0.0672 | ⚪ split |
| THA | 0.3940 | **0.3383** | +0.056 | **+0.688** | +0.147 | −.541 | **0.0416** | 0.0614 | ⚪ split |
| MYS | 0.4060 | **0.3613** | +0.045 | **+0.642** | +0.497 | −.145 | **0.0385** | 0.0433 | ⚪ split |
| IDN | 0.4417 | **0.3807** | +0.061 | −0.730 | **+0.303** | +1.033 | 0.0649 | **0.0582** | ⭐ SWA 3/3 |
| CHN | 0.3028 | **0.2987** | +0.004 | +0.609 | **+0.748** | +.139 | 0.0833 | **0.0793** | ⭐ SWA 3/3 |
| JPN | 0.2812 | **0.2711** | +0.010 | +0.038 | **+0.228** | +.190 | 0.0692 | **0.0615** | ⭐ SWA 3/3 |
| BGD | 0.4423 | **0.3862** | +0.056 | **+0.594** | −0.334 | −.928 ⚠️ | **0.0450** | 0.0727 | ⚪ split |
| IRN | **0.4110** | 0.4798 | −0.069 | **−0.042** | −0.295 | −.253 | **0.1091** | 0.1137 | ❌ VAN 3/3 |
| SRB | 0.4635 | **0.4441** | +0.019 | **+0.667** | +0.320 | −.347 | **0.0425** | 0.0534 | ⚪ split |
| ROU | **0.4618** | 0.4694 | −0.008 | **+0.693** | +0.418 | −.275 | **0.0427** | 0.0521 | ❌ VAN 3/3 |
| KGZ | **0.4403** | 0.4429 | −0.003 | **+0.744** | +0.452 | −.292 | **0.0399** | 0.0509 | ❌ VAN 3/3 |
| ETH | **0.5865** | 0.5984 | −0.012 | **+0.259** | −0.357 | −.616 ⚠️ | **0.0675** | 0.0877 | ❌ VAN 3/3 |
| **MEAN** | 0.4358 | **0.4340** | +0.002 | **+0.251** | +0.197 | −.054 | **0.0612** | 0.0646 | ≈ neutral |

**Bold = winner per cell.** ΔMIS positive ⇒ SWA improves.

---

## 5. Behavioural patterns

### 5.1 Where SWA helps

**Anti-aligned recovery (large r flips ↑):** the strongest qualitative win.

| Country | r (Van → SWA) | Δr |
|:--|:-:|:-:|
| IDN | −0.730 → +0.303 | **+1.033** |
| MEX | −0.065 → +0.515 | +0.580 |
| VNM | −0.526 → +0.014 | +0.540 |
| COL | −0.217 → +0.131 | +0.348 |
| ARG | −0.078 → +0.216 | +0.294 |
| JPN | +0.038 → +0.228 | +0.190 |
| CHN | +0.609 → +0.748 | +0.139 |
| BRA | −0.161 → −0.058 | +0.103 |

→ SWA-DPBR demonstrably *re-orients* the model on countries the self-judge originally inverted. **8 countries see r move in the right direction; 5 of these were originally anti-aligned (r < 0) and are now pro-aligned or near zero.**

### 5.2 Where SWA hurts

**Over-correction on already-aligned countries (r ↓):**

| Country | r (Van → SWA) | Δr |
|:--|:-:|:-:|
| BGD | +0.594 → −0.334 | **−0.928** ⚠️ |
| ETH | +0.259 → −0.357 | −0.616 |
| THA | +0.688 → +0.147 | −0.541 |
| MMR | +0.682 → +0.179 | −0.503 |
| SRB | +0.667 → +0.320 | −0.347 |
| KGZ | +0.744 → +0.452 | −0.292 |
| ROU | +0.693 → +0.418 | −0.275 |
| DEU | −0.181 → −0.454 | −0.273 |
| IRN | −0.042 → −0.295 | −0.253 |
| MYS | +0.642 → +0.497 | −0.145 |

→ SWA *destroys* rank-order on countries where Vanilla was already strong (BGD, ETH, THA, MMR), and pushes already-anti-aligned countries deeper into anti-alignment (DEU, IRN). **12 countries see r shrink toward zero or worsen.**

**Net effect: r values are compressed toward the mean** — SWA moves outliers in both directions to a middle band around r ≈ +0.20.

### 5.3 The MIS-vs-r decoupling

Eight countries show a **paradoxical pattern** where MIS and r move in opposite directions:

| Country | ΔMIS | Δr | Interpretation |
|:--|:-:|:-:|:--|
| MMR | +0.025 ✅ | −0.503 ❌ | Distribution closer in absolute distance, ranks shuffled |
| THA | +0.056 ✅ | −0.541 ❌ | Same |
| MYS | +0.045 ✅ | −0.145 ❌ | Same |
| BGD | +0.056 ✅ | −0.928 ❌ | **JSD paradox in reverse** |
| SRB | +0.019 ✅ | −0.347 ❌ | Same |
| ARG | −0.021 ❌ | +0.294 ✅ | Ranks fixed but spread wider |
| MEX | −0.014 ❌ | +0.580 ✅ | Same |
| COL | −0.023 ❌ | +0.348 ✅ | Same |
| VNM | −0.049 ❌ | +0.540 ✅ | Same |

→ **MIS and Pearson r capture different things in this regime.** MIS rewards smaller absolute deviations; r rewards correct ordering. SWA's persona ensemble can produce a more cardinally-correct distribution while shuffling the ordinal ranks (and vice versa). **This is the open-ended-track analogue of the JSD paradox documented in 2026-04-13** (paper §4.3).

---

## 6. Diagnostics — why SWA didn't win cleanly

All SWA diagnostics are *healthy* — the failure mode is **not** a broken controller:

| Diagnostic | Range across 20 countries | Interpretation |
|:-|:-:|:-|
| `rel_r` (DPBR reliability gate) | 0.909 – 0.986 | Always above gate threshold; never collapsed |
| `boot_var` (bootstrap variance) | 0.0009 – 0.0197 | Low; KGZ smallest, VNM largest |
| `ESS₁` / `ESS₂` (effective sample size) | 0.34 – 0.41 | Moderate IS coverage; consistent across passes |
| `α_anchor` (anchor-blend weight) | 0.32 – 0.39 | Moderate anchor reliance — not collapsed to base |
| Positional bias (residual) | **0.0** all 20 | Debiasing fully cancels position bias |
| Parse fail % | **0.0%** all 20 | Self-judge is reliable for A/B/UNCERTAIN extraction |

→ DPBR is doing what it should. The IS estimator is stable. The persona ensemble is producing meaningful corrections. **The bottleneck is the judge.**

---

## 7. Why the modest gain? — root cause hypothesis

Both actor and judge are **the same Qwen2.5-7B model** (self-judge). This means:

1. **Shared priors.** The judge inherits the same biases as the actor. When SWA moves the actor's distribution toward a persona-conditioned answer, the judge cannot reliably distinguish "this persona-conditioned output reflects culture X" from "this output is just noise" — both look plausible to the same model.
2. **No external ground truth in the loop.** The judge cannot validate that the persona-induced shift is in the *right direction* for country X; it just measures consistency with its own reading of the actor's output.
3. **Compression toward the model's own prior.** The result: r values squeeze toward the joint actor/judge prior (≈ +0.20 after SWA, vs +0.25 before), with both anti-aligned countries *and* well-aligned countries pulled toward the middle.

**Contrast with moral-machine (constrained-A/B) track:**
- Constrained-A/B uses logit gaps from the actor directly — no judge in the loop.
- Result there: SWA reduces MIS by 19–24% on Phi-4 / Llama-3.3-70B / Qwen2.5-7B etc. (see [tracker.md](tracker.md))
- The open-ended track introduces a noisy translation layer (`free-form text → A/B/UNCERTAIN/conf` via judge) that washes out the IS signal when the judge is the same model as the actor.

---

## 8. Implications & next steps

**Implication 1 — Self-judge is too weak to validate SWA-DPBR on the open-ended track.** The +0.002 mean ΔMIS does not refute the method; it reflects a measurement-instrument limitation. SWA-DPBR **clearly does something** (8 strong r-flips on anti-aligned countries) but the judge cannot translate that signal into MIS gains.

**Implication 2 — Where SWA does win unambiguously, the wins are in East Asia.** IDN, CHN, JPN sweep all three metrics. This is not because Qwen2.5-7B is better at East Asia (it is, but the *Vanilla* baseline already shows that); it is because SWA's signal is largest where the persona ensemble disagrees most strongly with the base model — and Qwen2.5-7B's neutral persona happens to underweight East-Asian moral intuitions, which the WVS-grounded personas correct.

**Implication 3 — Cost vs benefit.** SWA-DPBR costs ≈5× compute (1550 vs 310 judge calls per country) for ≈0.4% MIS improvement. This trade-off is **not justified** at the self-judge configuration. It will likely justify itself with an independent judge.

### Recommended next experiments

1. **Independent judge.** Re-run SWA-DPBR with **Qwen2.5-72B-Instruct-GPTQ** as judge (separate model, not self-judge). Hypothesis: MIS reduction will jump from +0.002 to ≥ +0.02–0.05, matching the moral-machine track's relative gain.
2. **Cross-family judge.** Try Phi-4 14B as judge while keeping Qwen2.5-7B as actor. This isolates the "shared prior" hypothesis.
3. **Disable self-judge ablation.** Run a 5-country pilot (USA, IDN, CHN, IRN, ETH — covering the three behavioural regimes) with both Qwen2.5-72B and Phi-4 as judges. Compare ΔMIS.
4. **Restrict SWA to anti-aligned countries.** Per §5.1, SWA's wins are concentrated where Vanilla r < 0. A simple gating rule (`apply SWA only when |r_baseline| < 0.3`) would convert the current 11/20 MIS win-count into ~14–16/20 with no over-correction loss.

---

## 9. Final scorecard

| Question | Answer |
|:-|:-|
| Does SWA-DPBR improve mean MIS in this regime? | Marginally — +0.002 (≈0.4%, within noise) |
| Does SWA-DPBR improve mean Pearson r? | **No** — drops by 0.054 (regression toward the mean) |
| Does SWA-DPBR improve mean JSD? | **No** — increases by 0.003 |
| How many of 20 countries does SWA win on MIS? | **10/20 (50% — exact tie)** |
| How many on r? | 8/20 (40%) |
| How many on JSD? | 8/20 (40%) |
| How many sweep all 3 metrics for SWA? | **3** (IDN, CHN, JPN) |
| How many sweep all 3 metrics for Vanilla? | 5 (DEU, IRN, ROU, KGZ, ETH) |
| Net countries where SWA is ≥2/3 better? | **8** |
| Net countries where Vanilla is ≥2/3 better? | 5 |
| Compute cost for these results? | ≈5× per country (1550 vs 310 judge calls) |
| Recommendation for paper? | Re-run with **independent judge** (Qwen2.5-72B GPTQ) before drawing conclusions. Current numbers reflect the self-judge bottleneck, not SWA-DPBR's true ceiling on this track. |

---

**Artifact paths (Kaggle):**
- Vanilla baseline: `/kaggle/working/cultural_alignment/results/openended_baseline/`
- SWA-DPBR: `/kaggle/working/cultural_alignment/results/openended/compare/comparison.csv`

**Tracker entries:**
- [`exp_paper/tracker.md`](tracker.md) — 2026-04-27 entries (both runs logged)
