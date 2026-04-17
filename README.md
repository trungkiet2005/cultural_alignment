# SWA-DPBR: Socially-Weighted Alignment with Dual-Pass Bootstrap Reliability

Official research codebase for **"SWA-DPBR: Train-Free Country-Aware Moral Alignment for Frozen LLMs via Persona-Disagreement Importance Sampling"** — NeurIPS 2026 submission.

SWA-DPBR is an **inference-time**, **train-free** method that steers frozen LLMs' moral choices toward country-specific human preferences — no finetuning, no per-country reward models. Within-country disagreement across WVS-derived personas is converted into logit-level corrections via Prospect Theory importance sampling.

> **TL;DR** — 19–24% MIS reduction across strong open-source LLMs on the MultiTP / Moral Machine benchmark (20 countries, 6 moral dimensions, 25+ languages). Calibration competes with scale: Phi-4 14B beats Llama-3.3-70B in absolute MIS.

📄 Paper: [exp_paper/Paper_New/SWA_DPBR/paper_revised.pdf](exp_paper/Paper_New/SWA_DPBR/paper_revised.pdf)
📘 Full context for AI/humans: [docs/CLAUDE.md](docs/CLAUDE.md)

---

## Repository layout

```
.
├── src/                   # Core library (importable as `src.*`)
├── experiment_DM/         # Active SWA-DPBR core (exp24_dpbr_core, exp_reporting)
├── exp_model/             # Active per-model runner base (_base_dpbr, _base_exp09)
├── exp_paper/             # 📄 NeurIPS 2026 track (moral machine, 22 models × 20 countries)
│   ├── Paper_New/SWA_DPBR/  # LaTeX source + compiled paper
│   ├── exp_paper_*.py       # One script per model
│   ├── paper_countries.py   # 20-country target list
│   └── tracker.md           # Full per-model per-country results log
├── open_ended/            # 📝 BLEnD benchmark track (open-ended QA, Phi-4 14B)
├── docs/                  # Documentation (CLAUDE.md, gap analysis, repro notes)
├── tests/                 # pytest suite
├── legacy/                # Archived iterations (kept for audit trail)
└── WVS_data/              # World Values Survey Wave 7 (gitignored, ~190MB)
```

### Entry points

| Script | Purpose |
|---|---|
| [`run_baseline.py`](run_baseline.py) | Vanilla LLM baseline (no steering) |
| [`run_swa_mppi.py`](run_swa_mppi.py) | SWA-DPBR single-model run |
| [`exp_paper/exp_paper_{model}.py`](exp_paper/) | Per-model paper sweep (20 countries) |
| [`exp_paper/exp_paper_ablation_phi4.py`](exp_paper/exp_paper_ablation_phi4.py) | Ablation study |
| [`open_ended/main.py`](open_ended/main.py) | BLEnD SWA run on Phi-4 |

---

## Install

```bash
pip install -r requirements.txt
# Dev extras (pytest, ruff, etc.)
pip install -r requirements-dev.txt
```

**GPU:** ≥24GB VRAM for 7B models. ≥80GB for 70B via Unsloth 4-bit quantization or vLLM tensor parallelism.

---

## Data

| Dataset | Description | Source |
|---|---|---|
| MultiTP | Multilingual moral dilemmas (40M responses, 233 countries) | [Jin et al., ICLR 2025](https://arxiv.org/abs/2407.02273) |
| Human AMCE | Country-specific human preferences | MultiTP |
| WVS Wave 7 | Cultural value profiles (persona grounding) | [worldvaluessurvey.org](https://www.worldvaluessurvey.org/) |

---

## Usage

### Paper sweep (recommended — Kaggle)

```bash
# One model × 20 countries × full SWA-DPBR
python exp_paper/exp_paper_phi_4.py

# Ablation study
python exp_paper/exp_paper_ablation_phi4.py
```

### Local / single-model

```bash
# Baseline
python run_baseline.py --model unsloth/Phi-4 --countries USA VNM DEU

# SWA-DPBR
python run_swa_mppi.py --model unsloth/Phi-4 --countries USA VNM DEU
```

### BLEnD open-ended QA

```bash
python open_ended/baseline_open_ended.py
python open_ended/main.py
```

### Compile paper

```bash
cd exp_paper/Paper_New/SWA_DPBR
latexmk -pdf paper_revised.tex
```

---

## Key metrics

- **MIS** (Misalignment Score) — L2 distance between model AMCE and human AMCE ↓
- **JSD** (Jensen-Shannon Divergence) — distributional alignment ↓
- **Pearson r** — correlation between model and human AMCEs ↑
- **SEM-B / SEM-W** — Soft Exact Match for open-ended QA (BLEnD)

⚠️ **JSD paradox:** removing debiasing/personas can *improve* JSD while catastrophically degrading rank-order (r, MIS). Report MIS as primary.

---

## Models evaluated

**Primary:** Phi-4 14B, Llama-3.3-70B, Qwen2.5-7B, Qwen3-VL-8B, Magistral-Small-2509, Phi-3.5-mini.
**Extended:** 16 additional models spanning 0.27B → 72B, including Gemma, Mistral, Qwen3, GPT-OSS.
**Backends:** Unsloth (4-bit for ≥70B), vLLM (BF16 for <70B).

---

## Countries (20)

USA, GBR, DEU, ARG, BRA, MEX, COL, VNM, MMR, THA, MYS, IDN, CHN, JPN, BGD, IRN, SRB, ROU, KGZ, ETH

Each country evaluated using **native-language prompts** (25+ languages supported).

---

## Review status

NeurIPS 2026 Round 1: **lean accept**. Round 2 work in progress — confidence intervals, per-dimension MIS deltas, attention-dropout calibration baseline, hierarchical prior spec. See [exp_paper/Review/Round1.md](exp_paper/Review/Round1.md).

---

## Citation

```bibtex
@inproceedings{jin2025multitp,
  title     = {Language Model Alignment in Multilingual Trolley Problems},
  author    = {Jin, Zhijing and others},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
