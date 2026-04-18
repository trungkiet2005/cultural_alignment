# CLAUDE.md — Full Repo Context (SWA-DPBR)

> **Read this file first.** It is the single source of truth for what this repo is, how it's organized, and how to work in it. Every other file in the repo should be understandable once you've read this.

---

## 1. What this project is

A research codebase for **SWA-DPBR** — *Socially-Weighted Alignment with Dual-Pass Bootstrap Reliability*.

**Goal:** Steer frozen LLMs toward country-specific moral preferences at inference time — **no finetuning, no per-country reward models, no extra training data**.

**Submitted to:** NeurIPS 2026 (Round 1: *lean accept*).

**Core idea:** Use **within-country disagreement** among WVS-derived cultural personas as a signal. Convert that disagreement into logit-level corrections via **Prospect-Theory importance sampling** with a **dual-pass bootstrap reliability** gate that suppresses corrections when the IS estimator is unstable.

**Why it matters:**
- Train-free → cheap, works on any frozen model
- Calibration competes with scale: Phi-4 (14B) beats Llama-3.3-70B in absolute MIS
- Gains are geographically broad — not concentrated in Western countries

---

## 2. Where everything lives

```
.
├── src/                       # 🧠 Core library — all active algorithms
│   ├── config.py              # BaselineConfig, SWAConfig dataclasses + argparse
│   ├── constants.py           # Characters, 6 moral categories, 45-country mappings
│   ├── data.py                # MultiTP dataset loading & balancing
│   ├── model.py               # Unsloth / HF native / vLLM loader + chat template
│   ├── scenarios.py           # Multilingual moral dilemma prompt construction
│   ├── personas.py            # WVS-7 persona generation (4 age cohorts + utilitarian)
│   ├── persona_i18n.py        # Persona translations (25+ languages)
│   ├── i18n.py                # Scenario frame translations
│   ├── controller.py          # ⭐ ImplicitSWAController — the SWA-PTIS engine
│   ├── amce.py                # AMCE computation + MIS/JSD/Pearson r metrics
│   ├── baseline_runner.py     # Vanilla LLM baseline
│   ├── swa_runner.py          # SWA-DPBR experiment runner
│   ├── hf_env.py              # HF token / cache setup
│   ├── vllm_causal.py         # vLLM causal LM wrapper
│   ├── vllm_env.py            # vLLM environment setup
│   ├── vllm_logit_model.py    # vLLM logit extraction
│   └── viz/                   # Radar, bar, heatmap, dendrogram, trigger plots
│
├── experiment_DM/             # 🔬 Active DPBR core (pruned — legacy moved out)
│   ├── exp24_dpbr_core.py     # ⭐ Exp24DualPassController, BootstrapPriorState
│   └── exp_reporting.py       # Tracker-ready report formatter
│
├── exp_model/                 # 🏃 Per-model runner base (pruned)
│   ├── _base_dpbr.py          # run_for_model() — used by every exp_paper_*.py
│   └── _base_exp09.py         # EXP-09 hierarchical IS variant
│
├── exp_paper/                 # 📄 NeurIPS 2026 track — moral machine
│   ├── Paper_New/SWA_DPBR/    # LaTeX source + compiled PDF
│   │   ├── paper_revised.tex  # Main paper
│   │   ├── paper_revised.pdf  # Compiled output
│   │   ├── references.bib     # Bibliography
│   │   └── checklist.tex      # NeurIPS submission checklist
│   ├── Review/Round1.md       # Reviewer feedback from Round 1
│   ├── exp_paper_{model}.py   # 22 per-model scripts (Phi-4, Llama, Qwen, Gemma, …)
│   ├── exp_paper_ablation_phi4.py      # Ablation study on Phi-4
│   ├── exp_persona_debiasing_sweep.py  # Persona debiasing hyperparameter sweep
│   ├── paper_countries.py     # 20-country target list
│   ├── paper_runtime.py       # Kaggle env bootstrap (vLLM / Unsloth)
│   ├── scripts/               # Helper scripts (e.g. per-country table builder)
│   └── tracker.md             # ⭐ Full results log — every model × country
│
├── open_ended/                # 📝 BLEnD benchmark track — open-ended QA
│   ├── main.py                # SWA for QA on Phi-4 14B
│   ├── baseline_open_ended.py # Vanilla greedy baseline
│   ├── swa_open_ended.py      # SWA-PTIS for QA
│   ├── tune_swa_mppi.py       # 3-phase hyperparameter tuning
│   ├── model_inference.py     # Shared inference utils
│   ├── utils.py               # 20+ model paths, country-language maps
│   ├── data/                  # BLEnD dataset
│   │   ├── annotations/       # Human annotations (per country)
│   │   ├── prompts/           # Per-language prompts
│   │   └── questions/         # Question files
│   ├── evaluation/            # SEM-B/SEM-W scoring + multiple-choice pipeline
│   └── tracker.md             # BLEnD results log
│
├── docs/                      # 📘 Documentation
│   ├── CLAUDE.md              # ← you are here
│   ├── exp24_paper_gap_analysis.md
│   └── exp24_reproducibility.md
│
├── tests/                     # pytest suite (test_exp24_dpbr.py)
├── WVS_data/                  # World Values Survey Wave 7 CSV (gitignored, ~190MB)
├── legacy/                    # 🗄️ Archived iterations — audit trail only
│   ├── experiment/            # Early exp01–04
│   ├── experiment_DM_archive/ # Old exp02–25 iterations
│   ├── experiment_open_ended/ # Single old open-ended file
│   ├── exp_model_archive/     # Old per-model scripts (exp9 + exp_24 variants)
│   ├── Reference_Notebook_Model/  # 16 Kaggle reference notebooks
│   ├── SWA_MPPI_paper_old/    # Pre-revision paper (superseded)
│   └── *.py, *.txt            # One-off scripts, persona sample dumps
│
├── run_baseline.py            # Entry point: vanilla baseline
├── run_swa_mppi.py            # Entry point: SWA-DPBR
├── setup_kaggle.py            # Kaggle environment bootstrap
└── requirements.txt / requirements-dev.txt
```

**Key rule:** `legacy/` is read-only audit trail. Don't import from it. Don't delete — reviewers may ask about earlier iterations (exp01 → exp25).

---

## 3. Core concepts (glossary)

| Term | Meaning |
|---|---|
| **MultiTP** | Multilingual trolley-problem dataset; 40M+ responses, 233 countries. Our input data. |
| **Moral Machine** | Original MIT trolley-problem experiment; MultiTP is its multilingual extension. |
| **AMCE** | *Average Marginal Component Effect* — per-dimension preference (species, gender, age, fitness, social value, utilitarianism). |
| **MIS** | *Misalignment Score* — L2 distance between model AMCE and human AMCE. **Primary metric. Lower = better.** |
| **JSD** | Jensen-Shannon divergence over choice distributions. ⚠️ Can mislead alone — see JSD paradox below. |
| **Pearson r** | Correlation between model and human AMCEs. |
| **SEM-B / SEM-W** | *Soft Exact Match* (Best / Worst) — BLEnD open-ended QA metrics. |
| **WVS-7** | World Values Survey Wave 7 — 10-variable cultural profile per country. Source for personas. |
| **Persona** | Synthetic cultural agent derived from WVS-7 descriptors (e.g. "highly religious / low social trust / young"). 4 age cohorts + utilitarian neutral. |
| **SWA-PTIS** | Socially-Weighted Alignment with Prospect-Theory Importance Sampling (the controller itself). |
| **DPBR** | *Dual-Pass Bootstrap Reliability* — runs IS twice with independent noise, gates corrections by bootstrap variance. |
| **Debiasing** | A↔B position swap to cancel positional bias in logit gaps. **Load-bearing** (ablation: −0.282 r drop without it). |
| **ESS anchor reg** | When effective sample size is low, blend anchor toward `delta_base` before adding `delta_star`. |
| **Prospect Theory (PT)** | Loss aversion + diminishing sensitivity. Parameters: α=β=0.88, κ=2.25. |

### JSD paradox ⚠️

Removing debiasing or personas can *improve* JSD (distribution gets more uniform) while **catastrophically degrading** rank-order metrics (r, MIS). **Always report MIS as primary.** Document at [exp_paper/tracker.md](../exp_paper/tracker.md) 2026-04-13 entry.

---

## 4. SWA-DPBR math (short)

From [src/controller.py](../src/controller.py) `_is_solve_decision` and [experiment_DM/exp24_dpbr_core.py](../experiment_DM/exp24_dpbr_core.py):

```
Per-agent gain:       g_{i,k} = |δ_base - δ_i| - |δ̃_k - δ_i|
Consensus gain:       g_cons_k = |δ_base - δ̄| - |δ̃_k - δ̄|
Collective utility:   U(ε_k) = (1-λ) · mean_i v(g_{i,k}/σ) + λ · v(g_cons_k/σ)
Softmax weights:      w_k = softmax(U(ε_k)/η)
IS update:            δ* = Σ_k w_k · ε_k
```

**Dual-pass:** run twice with independent noise, combine via bootstrap reliability weight `r = exp(-var/VAR_SCALE)` (default `VAR_SCALE=0.04`). Tunable via env var `EXP24_VAR_SCALE`.

Hyperparameters: `K_HALF=64` samples per pass → 128 total. Override via `EXP24_K_HALF`.

---

## 5. Active models

**Primary 6** (paper headline):
- Phi-4 (14B) — **best MIS** despite smaller size
- Llama-3.3-70B
- Qwen2.5-7B, Qwen3-VL-8B
- Magistral-Small-2509
- Phi-3.5-mini

**Extended 16:** Gemma 7B / 2-9B / 3-270M / 4-E2B, Mistral 7B v0.2/v0.3, Llama 3.1 8B/70B, Llama 3.2 1B, Qwen 2.5 72B / 3 4B thinking / 3.5 0.8B, GPT-OSS 20B, etc.

**Backends:**
- `unsloth/*-bnb-4bit` for ≥70B (4-bit quantization)
- vLLM BF16 for <70B (more faithful logits — quantization artifacts bit us on Unsloth Phi-4 ablation, now superseded)

See [exp_paper/exp_paper_*.py](../exp_paper/) for one-to-one mapping.

---

## 6. Countries & languages

**20 paper countries** (ISO 3166-1 alpha-3), covering 5 continents and 6 language families:

```
Western:        USA GBR DEU
Latin America:  ARG BRA MEX COL
SE Asia:        VNM MMR THA MYS IDN
East Asia:      CHN JPN
South Asia:     BGD
MENA:           IRN
Eastern Europe: SRB ROU KGZ
Africa:         ETH
```

**25+ languages** supported in [src/i18n.py](../src/i18n.py) (scenario frames) and [src/persona_i18n.py](../src/persona_i18n.py) (personas). Each country is prompted in its native language.

---

## 7. Key findings

1. **19–24% MIS reduction** across strongest models — train-free.
2. **Positional debiasing is load-bearing** — ablation shows −0.282 Pearson r drop without it.
3. **Personas are load-bearing** — ablation shows −0.183 r drop without them.
4. **Calibration competes with scale** — Phi-4 (14B) > Llama-3.3-70B in MIS.
5. **Geographically broad** — gains aren't concentrated in Western countries.
6. **JSD paradox** — removing components can improve JSD while destroying rank-order; MIS is primary.
7. **Backend matters** — Unsloth 4-bit collapsed IS signal on Phi-4; vLLM BF16 restored it.

Full per-model per-country numbers: [exp_paper/tracker.md](../exp_paper/tracker.md).

---

## 8. Review status (NeurIPS 2026)

**Round 1: lean accept.** Round 2 contingencies:
- [ ] Confidence intervals / multi-seed reporting
- [ ] Per-dimension MIS deltas
- [ ] Comparison to attention-dropout calibration baseline
- [ ] Exact hierarchical prior specification

Full reviewer comments: [exp_paper/Review/Round1.md](../exp_paper/Review/Round1.md).

---

## 9. Common commands

### Paper track (moral machine)

```bash
# Single model × 20 countries × full SWA-DPBR (on Kaggle)
python exp_paper/exp_paper_phi_4.py

# Ablation study
python exp_paper/exp_paper_ablation_phi4.py

# Persona debiasing sweep
python exp_paper/exp_persona_debiasing_sweep.py

# Compile paper
cd exp_paper/Paper_New/SWA_DPBR && latexmk -pdf paper_revised.tex
```

### Local / single-model

```bash
python run_baseline.py --model unsloth/Phi-4 --countries USA VNM DEU
python run_swa_mppi.py  --model unsloth/Phi-4 --countries USA VNM DEU
```

### BLEnD track (open-ended QA)

```bash
python open_ended/baseline_open_ended.py
python open_ended/main.py
python open_ended/tune_swa_mppi.py    # 3-phase hyperparameter tuning
```

### Tests

```bash
pytest tests/ -v
```

### Ablation env vars

```bash
EXP24_VAR_SCALE=0.04   # Bootstrap reliability scale
EXP24_K_HALF=64        # Samples per pass (total = 2 × K_HALF)
EXP24_ESS_ANCHOR_REG=1 # Enable ESS-adaptive anchor blend (set 0 to disable)
```

---

## 10. Conventions

- **Country codes:** ISO 3166-1 alpha-3 (USA, VNM, DEU, GBR, …). Always 3 letters.
- **Experiment scripts:** `exp_paper/exp_paper_{model_slug}.py` → calls `exp_model._base_dpbr.run_for_model()`.
- **Results directory:** `results/exp24_paper_20c/{model}/{country}/` (Kaggle: prefixed with `/kaggle/working/cultural_alignment/`).
- **Tracker updates:** when user says *"update the tracker"*, prepend a new dated entry to [exp_paper/tracker.md](../exp_paper/tracker.md) (not `docs/performance_tracker.md` — that doesn't exist yet).
- **Model slug:** `model_name.split("/")[-1].lower().replace("_", "-")` (see [src/config.py](../src/config.py) `model_slug()`).
- **LaTeX:** `latexmk -pdf paper_revised.tex` from the paper dir; never edit `.aux`/`.bbl`/`.log` by hand.

---

## 11. Environment

- **Python 3.10+**, PyTorch, Transformers
- **Unsloth** (4-bit for ≥70B) or **vLLM** (BF16 for <70B, preferred)
- **Kaggle** is the primary compute. Scripts auto-detect `/kaggle/working` and clone themselves.
- **WVS data**: `WVS_data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv` (gitignored; ~190MB).
- **HF token**: set via `.env` (see `.env.example`) → loaded by [src/hf_env.py](../src/hf_env.py).

---

## 12. Dependencies between active modules

```
exp_paper/exp_paper_{model}.py
  → exp_model/_base_dpbr.py
    → experiment_DM/exp24_dpbr_core.py   ← DPBR controller
    → experiment_DM/exp_reporting.py     ← tracker format
    → src/controller.py                  ← SWA-PTIS engine
    → src/swa_runner.py / baseline_runner.py
    → src/personas.py, scenarios.py, data.py, model.py, amce.py
    → src/viz/*                          ← plots

exp_paper/exp_paper_ablation_phi4.py
  → experiment_DM/exp24_dpbr_core.py (direct, for ablation toggles)

tests/test_exp24_dpbr.py
  → experiment_DM/exp24_dpbr_core.py
```

⚠️ **Do not import from `legacy/`.** Anything there is frozen; pulling from it resurrects dead code paths.

---

## 13. When things go wrong

- **Import errors on `experiment_DM` or `exp_model`:** you're probably running from the wrong cwd. All scripts assume repo root in `sys.path`. On Kaggle, `paper_runtime.configure_paper_env()` handles this.
- **OOM on 70B:** switch to Unsloth 4-bit (`unsloth/Meta-Llama-3.3-70B-Instruct-bnb-4bit`) or enable vLLM tensor parallelism.
- **IS collapses (flip% ≈ 50%):** likely Unsloth quantization artifact — switch to vLLM BF16. See [exp_paper/tracker.md](../exp_paper/tracker.md) 2026-04-13.
- **LaTeX errors on `neurips_2026.sty`:** verify you're compiling from `exp_paper/Paper_New/SWA_DPBR/`, not from root or `legacy/SWA_MPPI_paper_old/`.

---

## 14. One-paragraph project pitch (for external audiences)

> Large language models are increasingly asked to make morally loaded judgments across cultures, yet we have no efficient way to make a frozen model answer *like people from country X would answer*. Finetuning per country is impossibly expensive at the scale of MultiTP (233 countries, 40M responses). We propose **SWA-DPBR**, a train-free inference-time controller that reshapes a frozen LLM's token logits by importance-sampling over culturally grounded personas derived from the World Values Survey, gated by a bootstrap reliability test. On the Moral Machine / MultiTP benchmark across 20 countries and 6 moral dimensions, SWA-DPBR reduces misalignment by 19–24% on strong open-source LLMs, with gains distributed broadly across cultures rather than concentrated in Western ones. Notably, a well-calibrated 14B model (Phi-4) outperforms a 70B model (Llama-3.3-70B) in absolute misalignment — suggesting **calibration competes with scale**.
