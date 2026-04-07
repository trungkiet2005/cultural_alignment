# SWA-MPPI: Socially-Weighted Alignment via Model Predictive Path Integral

Official implementation for **"SWA-MPPI: Dynamic Social Consensus for Cross-Cultural Value Negotiation via Implicit Pre-Logit Control"**.

This framework evaluates and improves LLM moral alignment across **15 countries** and **6 moral dimensions** using the Moral Machine / MultiTP dataset. It includes:

- **Vanilla LLM Baseline**: Token-logit extraction for direct moral preference measurement
- **SWA-MPPI**: Multi-persona MPPI optimization with Prospect Theory and WVS-based cultural agents

---

## Repository Structure

```
swa-mppi/
├── run_baseline.py          # Entry point: Vanilla LLM baseline
├── run_swa_mppi.py          # Entry point: SWA-MPPI experiment
├── setup_kaggle.py          # Kaggle environment bootstrap
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT License
│
└── src/                     # Core library
    ├── config.py            # Configuration dataclasses + argparse
    ├── constants.py         # Characters, categories, country maps
    ├── i18n.py              # 12-language translation dictionaries
    ├── scenarios.py         # Moral dilemma scenario generation
    ├── data.py              # MultiTP dataset loading & balancing
    ├── model.py             # Model loading & chat template helper
    ├── amce.py              # AMCE computation & alignment metrics
    ├── personas.py          # WVS-based cultural persona generation
    ├── baseline_runner.py   # Vanilla baseline inference
    ├── controller.py        # ImplicitSWAController (MPPI engine)
    ├── swa_runner.py        # SWA-MPPI experiment runner
    └── viz/                 # Visualization modules
        ├── style.py         # Matplotlib configuration
        ├── radar.py         # Radar charts (model vs human)
        ├── bar_charts.py    # AMCE comparison bar charts
        ├── tables.py        # Results summary tables
        ├── clustering.py    # Cultural clustering dendrograms
        ├── heatmap.py       # Alignment heatmaps (SWA)
        └── trigger.py       # MPPI trigger analysis (SWA)
```

---

## Installation

```bash
pip install -r requirements.txt
```

**GPU Required**: Experiments use Llama 3.1 70B (4-bit quantized via [Unsloth](https://github.com/unslothai/unsloth)). A GPU with >= 24GB VRAM is recommended.

---

## Data

This project uses three datasets:

| Dataset | Description | Source |
|---------|-------------|--------|
| **MultiTP** | Multilingual moral dilemma scenarios | [Jin et al., ICLR 2025](https://arxiv.org/abs/2407.02273) |
| **Human AMCE** | Country-specific human moral preferences | MultiTP dataset |
| **WVS Wave 7** | World Values Survey cultural profiles | [worldvaluessurvey.org](https://www.worldvaluessurvey.org/) |

---

## Usage

### Vanilla LLM Baseline

```bash
# With real MultiTP data
python run_baseline.py \
    --multitp-data-path ./data/multitp \
    --human-amce-path ./data/country_specific_ACME.csv \
    --output-dir results/baseline

# With synthetic data (no external dataset needed)
python run_baseline.py \
    --use-synthetic-data \
    --output-dir results/baseline_synth

# Quick test: 3 countries, 50 scenarios each
python run_baseline.py \
    --use-synthetic-data \
    --countries USA CHN JPN \
    --n-scenarios 50
```

### SWA-MPPI

```bash
# Full experiment
python run_swa_mppi.py \
    --multitp-data-path ./data/multitp \
    --human-amce-path ./data/country_specific_ACME.csv \
    --wvs-data-path ./data/WVS_Wave7.csv \
    --output-dir results/swa_mppi

# Custom hyperparameters
python run_swa_mppi.py \
    --use-synthetic-data \
    --lambda-coop 0.8 \
    --noise-std 0.2 \
    --K-samples 256 \
    --tau-target-trigger-rate 0.4
```

### Custom Model

```bash
python run_swa_mppi.py \
    --model-name "unsloth/Qwen2.5-72B-Instruct-bnb-4bit" \
    --use-synthetic-data
```

### Run on Kaggle

```python
# In a Kaggle notebook cell:
!python setup_kaggle.py
!python run_swa_mppi.py \
    --multitp-data-path /kaggle/input/datasets/trungkiet/mutltitp-data/data/data \
    --human-amce-path /kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv \
    --wvs-data-path /kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv
```

---

## Key Arguments

### Shared (both methods)

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-name` | `unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit` | HuggingFace model ID |
| `--n-scenarios` | `500` | Scenarios per country |
| `--countries` | 15 countries | ISO3 codes (e.g., `USA CHN JPN`) |
| `--decision-temperature` | `0.5` | Logit sharpening temperature |
| `--seed` | `42` | Random seed |
| `--use-synthetic-data` | off | Use generated scenarios instead of MultiTP |
| `--output-dir` | varies | Output directory for results and figures |

### SWA-MPPI Specific

| Argument | Default | Description |
|----------|---------|-------------|
| `--lambda-coop` | `0.7` | Cooperation weight (social vs. private utility) |
| `--alpha-kl` | `0.05` | KL divergence penalty in MPPI |
| `--K-samples` | `128` | Number of MPPI noise samples |
| `--noise-std` | `0.3` | Gaussian noise std for perturbations |
| `--pt-alpha` | `0.88` | Prospect Theory gain curvature |
| `--pt-beta` | `0.88` | Prospect Theory loss curvature |
| `--pt-kappa` | `2.25` | Loss aversion coefficient |
| `--logit-temperature` | `3.0` | Global logit temperature |
| `--tau-target-trigger-rate` | `0.35` | Target MPPI trigger rate |

---

## Moral Dimensions

| Dimension | Preferred Group | Description |
|-----------|----------------|-------------|
| Species | Humans | Humans vs. animals |
| Gender | Female | Male vs. female |
| Age | Young | Young vs. elderly |
| Fitness | Fit | Athletic vs. large body type |
| Social Value | High | High-status vs. homeless |
| Utilitarianism | More | More lives vs. fewer lives |

---

## Countries Evaluated

USA, DEU, CHN, JPN, BRA, SAU, VNM, FRA, IND, KOR, GBR, RUS, MEX, NGA, AUS

Each country is evaluated using **native-language prompts** (12 languages supported).

---

## Outputs

Both experiments produce:

- **Per-country CSVs**: Scenario-level predictions with probabilities
- **AMCE summary CSV**: Model vs. human AMCE scores across all countries
- **Radar charts**: 6-axis comparison (model vs. human preferences)
- **Results table**: Alignment metrics per country (PDF + LaTeX)
- **Cultural clustering**: Dendrogram of moral preference profiles

SWA-MPPI additionally produces:
- **Alignment heatmap**: Cross-country JSD matrix
- **Trigger analysis**: MPPI activation patterns vs. alignment quality
- **Decision gap analysis**: Distribution of agent disagreements

---

## Reference

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

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
