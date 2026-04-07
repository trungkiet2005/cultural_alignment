"""Configuration dataclasses and argument parsing for SWA-MPPI experiments."""

import os
import argparse
from dataclasses import dataclass, field
from typing import List, Dict

_ON_KAGGLE = os.path.exists("/kaggle/working")


def model_slug(model_name: str) -> str:
    """Convert a HuggingFace model ID into a filesystem-safe slug.

    Examples:
        "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit" -> "meta-llama-3.1-70b-instruct-bnb-4bit"
        "Qwen/Qwen2.5-72B-Instruct" -> "qwen2.5-72b-instruct"
    """
    name = model_name.split("/")[-1]
    return name.lower().replace("_", "-")


def resolve_output_dir(output_dir: str, model_name: str) -> str:
    """Append model slug to output_dir so multi-model runs do not overwrite each other."""
    return os.path.join(output_dir, model_slug(model_name))


# ============================================================================
# Base configuration (shared fields)
# ============================================================================
@dataclass
class BaseConfig:
    """Shared hyperparameters for all experiment variants."""

    model_name: str = "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"
    max_seq_length: int = 2048
    load_in_4bit: bool = True

    # Decision sharpening (< 1 amplifies final output, undoes RLHF compression)
    decision_temperature: float = 0.5

    # Inference (0 = auto-detect from free VRAM)
    batch_size: int = 0

    # Experiment
    n_scenarios: int = 500
    seed: int = 42
    target_countries: List[str] = field(default_factory=lambda: [
        # Original 15
        "USA", "DEU", "CHN", "JPN", "BRA", "SAU", "VNM", "FRA", "IND", "KOR",
        "GBR", "RUS", "MEX", "NGA", "AUS",
        # Tier 1 (+6)
        "IDN", "TUR", "POL", "ARG", "EGY", "ZAF",
        # Tier 2 (+4)
        "SWE", "PAK", "COL", "UKR",
    ])

    # Paths
    dataset_path: str = "data/scenarios.csv"
    output_dir: str = "results"

    # MultiTP real dataset loading
    multitp_data_path: str = ""
    multitp_lang: str = "en"
    multitp_translator: str = "google"
    multitp_suffix: str = ""
    use_real_data: bool = True

    # WVS data path (World Values Survey Wave 7)
    wvs_data_path: str = ""

    # Human AMCE data from MultiTP (long format: Estimates, se, Label, Country)
    human_amce_path: str = ""

    MULTITP_CATEGORIES: List[str] = field(default_factory=lambda: [
        "Species", "Gender", "Age", "Fitness", "SocialValue", "Utilitarianism",
    ])
    MULTITP_GROUPS: Dict[str, List[str]] = field(default_factory=lambda: {
        "Species":        ["Animals", "Humans"],
        "SocialValue":    ["Low",     "High"],
        "Gender":         ["Male",    "Female"],
        "Age":            ["Old",     "Young"],
        "Fitness":        ["Unfit",   "Fit"],
        "Utilitarianism": ["Less",    "More"],
    })


# ============================================================================
# Baseline configuration
# ============================================================================
@dataclass
class BaselineConfig(BaseConfig):
    """Hyperparameters for the Vanilla LLM Baseline experiment."""

    output_dir: str = "results_baseline"


# ============================================================================
# SWA-MPPI configuration
# ============================================================================
@dataclass
class SWAConfig(BaseConfig):
    """Hyperparameters for the SWA-MPPI experiment."""

    output_dir: str = "results_swa"

    # SWA-MPPI Core
    lambda_coop: float = 0.7
    alpha_kl: float = 0.05

    # Prospect Theory value function (Kahneman & Tversky, 1979)
    pt_alpha: float = 0.88            # gain curvature (diminishing sensitivity)
    pt_beta: float = 0.88             # loss curvature
    pt_kappa: float = 2.25            # loss aversion coefficient (lambda in K&T notation)

    K_samples: int = 128
    noise_std: float = 0.3
    temperature: float = 0.5
    tau_conflict: float = 0.001       # Auto-calibrated per country
    logit_temperature: float = 3.0    # Global default; overridden per-category

    category_logit_temperatures: Dict[str, float] = field(default_factory=lambda: {
        "Species":        4.0,
        "Gender":         3.5,
        "Age":            1.5,
        "Fitness":        1.5,
        "SocialValue":    1.5,
        "Utilitarianism": 1.5,
    })

    # Adaptive tau target trigger rate
    tau_target_trigger_rate: float = 0.35
    tau_calibration_n: int = 50


# ============================================================================
# Argument parsing helpers
# ============================================================================
def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add shared CLI flags that apply to all experiment variants."""
    parser.add_argument("--model-name", type=str,
                        default="unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit",
                        help="HuggingFace model identifier")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                        help="Maximum sequence length for the model")
    parser.add_argument("--no-4bit", action="store_true",
                        help="Disable 4-bit quantisation (default: enabled)")
    parser.add_argument("--decision-temperature", type=float, default=0.5,
                        help="Decision sharpening temperature (< 1 amplifies)")
    parser.add_argument("--n-scenarios", type=int, default=500,
                        help="Number of scenarios per country")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Inference batch size (0 = auto-detect from free VRAM)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--countries", nargs="+", default=None,
                        help="Target country ISO-3 codes (default: 15 countries)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for experiment outputs")
    parser.add_argument("--multitp-data-path", type=str, default="",
                        help="Path to MultiTP real dataset directory")
    parser.add_argument("--wvs-data-path", type=str, default="",
                        help="Path to WVS Wave 7 CSV file")
    parser.add_argument("--human-amce-path", type=str, default="",
                        help="Path to human AMCE CSV (country_specific_ACME.csv)")
    parser.add_argument("--use-synthetic-data", action="store_true",
                        help="Use synthetic scenarios instead of real MultiTP data")

    # Kaggle environment defaults
    if _ON_KAGGLE:
        parser.set_defaults(
            multitp_data_path="/kaggle/input/datasets/trungkiet/mutltitp-data/data/data",
            wvs_data_path="/kaggle/input/datasets/trungkiet/mutltitp-data/"
                          "WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv",
            human_amce_path="/kaggle/input/datasets/trungkiet/mutltitp-data/"
                            "data/data/country_specific_ACME.csv",
        )


def add_swa_args(parser: argparse.ArgumentParser) -> None:
    """Add SWA-MPPI-specific CLI flags."""
    parser.add_argument("--lambda-coop", type=float, default=0.7,
                        help="Cooperation weight lambda")
    parser.add_argument("--alpha-kl", type=float, default=0.05,
                        help="KL divergence penalty weight")
    parser.add_argument("--pt-alpha", type=float, default=0.88,
                        help="Prospect Theory gain curvature")
    parser.add_argument("--pt-beta", type=float, default=0.88,
                        help="Prospect Theory loss curvature")
    parser.add_argument("--pt-kappa", type=float, default=2.25,
                        help="Prospect Theory loss aversion coefficient")
    parser.add_argument("--K-samples", type=int, default=128,
                        help="Number of MPPI perturbation samples")
    parser.add_argument("--noise-std", type=float, default=0.3,
                        help="Standard deviation of MPPI noise")
    parser.add_argument("--mppi-temperature", type=float, default=0.5,
                        help="MPPI softmax temperature")
    parser.add_argument("--logit-temperature", type=float, default=3.0,
                        help="Global logit temperature (overridden per-category)")
    parser.add_argument("--tau-target-trigger-rate", type=float, default=0.35,
                        help="Target trigger rate for adaptive tau calibration")
    parser.add_argument("--tau-calibration-n", type=int, default=50,
                        help="Number of scenarios for tau calibration")


def config_from_args(args: argparse.Namespace, config_cls: type) -> BaseConfig:
    """Instantiate a config dataclass from parsed CLI arguments.

    Maps CLI flag names (hyphens) to dataclass field names (underscores) and
    handles special-case flags like ``--no-4bit`` and ``--use-synthetic-data``.
    """
    kwargs: Dict[str, object] = {}

    # --- shared fields ---
    if args.model_name is not None:
        kwargs["model_name"] = args.model_name
    if args.max_seq_length is not None:
        kwargs["max_seq_length"] = args.max_seq_length
    if args.no_4bit:
        kwargs["load_in_4bit"] = False
    if args.decision_temperature is not None:
        kwargs["decision_temperature"] = args.decision_temperature
    if args.n_scenarios is not None:
        kwargs["n_scenarios"] = args.n_scenarios
    if getattr(args, "batch_size", None) is not None:
        kwargs["batch_size"] = args.batch_size
    if args.seed is not None:
        kwargs["seed"] = args.seed
    if args.countries is not None:
        kwargs["target_countries"] = args.countries
    if args.output_dir is not None:
        kwargs["output_dir"] = args.output_dir
    if args.multitp_data_path:
        kwargs["multitp_data_path"] = args.multitp_data_path
    if args.wvs_data_path:
        kwargs["wvs_data_path"] = args.wvs_data_path
    if args.human_amce_path:
        kwargs["human_amce_path"] = args.human_amce_path
    if args.use_synthetic_data:
        kwargs["use_real_data"] = False

    # --- SWA-MPPI specific fields ---
    if config_cls is SWAConfig:
        for attr in (
            "lambda_coop", "alpha_kl", "pt_alpha", "pt_beta", "pt_kappa",
            "K_samples", "noise_std", "logit_temperature",
            "tau_target_trigger_rate", "tau_calibration_n",
        ):
            val = getattr(args, attr, None)
            if val is not None:
                kwargs[attr] = val
        # --mppi-temperature -> temperature
        mppi_temp = getattr(args, "mppi_temperature", None)
        if mppi_temp is not None:
            kwargs["temperature"] = mppi_temp

    return config_cls(**kwargs)
