"""Twenty-country target list for EXP-24 (DPBR) paper sweeps in ``exp_paper``."""

# Geographic coverage used for the paper’s 20-country evaluation (ISO 3166-1 alpha-3).
PAPER_20_COUNTRIES = [
    "USA",
    "GBR",
    "DEU",
    "ARG",
    "BRA",
    "MEX",
    "COL",
    "VNM",
    "MMR",
    "THA",
    "MYS",
    "IDN",
    "CHN",
    "JPN",
    "BGD",
    "IRN",
    "SRB",
    "ROU",
    "KGZ",
    "ETH",
]

_KAGGLE_ROOT = "/kaggle/working/cultural_alignment"

# All ``exp_paper`` scripts call ``exp_model._base_dpbr.run_for_model`` (EXP-24 / DPBR).
RESULTS_BASE_EXP24_20C = f"{_KAGGLE_ROOT}/results/exp24_paper_20c"

# Alias for older scripts / clarity (DPBR = EXP-24 in this repo).
RESULTS_BASE_DPBR = RESULTS_BASE_EXP24_20C
