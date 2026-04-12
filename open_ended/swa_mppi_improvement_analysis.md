# SWA-MPPI MC: Improvement Analysis

## How Baseline vs SWA-MPPI Work

| Aspect | [baseline_mc.py](file:///d:/AI_RESEARCH/moral_machine/open_ended/baseline_mc.py) | [swa_mppi_mc.py](file:///d:/AI_RESEARCH/moral_machine/open_ended/swa_mppi_mc.py) |
|---|---|---|
| **Method** | Greedy generation → parse JSON | Logit-level probability over A/B/C/D |
| **Personas** | None (vanilla model) | 4 cultural personas + 1 base |
| **Answer selection** | String match from generated text | `argmax(p_debiased)` over 4 rotations |
| **Cost per question** | 1 forward pass (generation) | ~20 forward passes (5 agents × 4 rotations) |

## Root Cause: Why SWA-MPPI Can Hurt Accuracy

> [!IMPORTANT]
> The MPPI optimizer **averages persona opinions** and then optimizes a socially-weighted utility. If persona agents are noisy or biased, MPPI can **pull the answer away** from the correct one — especially when the base model alone would have gotten it right.

The key risk: **MPPI trades base model confidence for multi-agent consensus**. Good when agents are well-calibrated. Bad when they add noise.

---

## Proposed Improvements (Ranked by Impact)

### 1. 🔑 Weight Base Model Higher in Consensus (HIGH IMPACT)

Currently [line 932](file:///d:/AI_RESEARCH/moral_machine/open_ended/swa_mppi_mc.py#L932):
```python
p_consensus = p_agents.mean(dim=0)  # Only persona agents, ignores base
```

**Problem**: The base model (which baseline_mc uses!) is excluded from consensus. Personas can override a confident correct answer.

**Fix**: Include base model with higher weight:
```python
# Weighted consensus: base gets 2x weight
w_base = 2.0
p_all = torch.cat([p_base.expand(1, -1) * w_base, p_agents], dim=0)
p_consensus = p_all.sum(dim=0) / (w_base + N)
```

---

### 2. 🔑 Reduce Noise / Use Fewer MPPI Samples (HIGH IMPACT)

Current config:
```python
K_samples: int = 128    # too many perturbations
noise_std: float = 0.3  # too much noise
```

**Problem**: 128 samples with σ=0.3 in log-space creates massive perturbation spread on a 4-choice simplex. Many perturbations land far from the consensus and dilute the signal.

**Fix**: 
```python
K_samples: int = 32     # sufficient for 4-dim simplex
noise_std: float = 0.1  # much tighter around consensus
```

---

### 3. 🔑 Increase KL Penalty (MEDIUM-HIGH IMPACT)

Current: `alpha_kl: float = 0.05` — too low.

**Problem**: MPPI can drift far from consensus without penalty. On a 4-choice problem, you want the optimizer to stay close to the consensus distribution.

**Fix**:
```python
alpha_kl: float = 0.5  # 10x stronger → MPPI stays near consensus
```

---

### 4. Reduce `lambda_coop` (MEDIUM IMPACT)

Current: `lambda_coop: float = 0.7` — social component dominates.

**Problem**: For **factual knowledge** questions (BLEnD MC), the "correct" answer isn't a social negotiation — it's a fact. High social weighting lets one wrong persona drag the answer.

**Fix**:
```python
lambda_coop: float = 0.3  # private utility dominates for factual QA
```

---

### 5. Smarter Persona Construction (MEDIUM IMPACT)

Current personas are culturally-flavored but generic. They don't specifically help with **everyday knowledge** questions.

**Improvement ideas**:
- Add a "knowledgeable local" persona (e.g., "You are a trivia enthusiast from {country}")
- Add a "daily life" persona focused on food, sports, holidays
- Remove cultural value traits (religion, gender equality) since they're irrelevant to factual MC questions

---

### 6. Confidence-Gated MPPI (MEDIUM IMPACT)

Instead of always-on MPPI, only activate when the base model is **uncertain**:

```python
p_base_max = p_base.max().item()
if p_base_max > 0.7:  # Base is confident → trust it
    p_star = p_base.squeeze(0)
else:                  # Base is uncertain → use MPPI consensus
    p_star = self._mppi_solve_mc(p_consensus, r_agents)
```

This preserves baseline accuracy on easy questions while using MPPI only where it can help.

---

### 7. Temperature-Scaled Logits (LOW-MEDIUM IMPACT)

Current: `logit_temperature: float = 1.0` (no scaling).

Lower temperature sharpens distributions, making the argmax more decisive:
```python
logit_temperature: float = 0.5  # Sharper distributions
```

---

## Recommended Quick-Win Config

These changes can be applied immediately by editing [CulturalMCConfig](file:///d:/AI_RESEARCH/moral_machine/open_ended/swa_mppi_mc.py#92-154):

```python
@dataclass
class CulturalMCConfig:
    lambda_coop: float = 0.3        # ← was 0.7
    alpha_kl: float = 0.5           # ← was 0.05  
    K_samples: int = 32             # ← was 128
    noise_std: float = 0.1          # ← was 0.3
    logit_temperature: float = 0.5  # ← was 1.0
```

Plus the code change to include the base model in consensus (Improvement #1).

## Summary

| # | Change | Expected Effect | Effort |
|---|--------|----------------|--------|
| 1 | Base model in consensus | +2-5% accuracy | 3 lines |
| 2 | Lower K & noise | +1-3% accuracy, 4x faster | Config only |
| 3 | Higher KL penalty | +1-2% accuracy | Config only |
| 4 | Lower lambda_coop | +1-2% accuracy | Config only |
| 5 | Better personas | +1-3% accuracy | Persona text |
| 6 | Confidence gating | +2-4% accuracy | ~10 lines |
| 7 | Lower logit temp | +0-1% accuracy | Config only |

> [!TIP]
> Start with improvements **1 + 2 + 3 + 4** (config + 3 lines of code). These are the lowest-risk, highest-impact changes. Run on 1 country to A/B test against baseline.
