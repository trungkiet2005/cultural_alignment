"""
EXP-09 Hierarchical IS — shared base
====================================
Core EXP-09 algorithm + runner reused by every per-model EXP-09 file.
NOT self-contained: env bootstrap must have run before this is imported.
"""

import gc
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    torch._dynamo.config.disable = True
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

from experiment_DM.exp_reporting import (
    CompareSpec,
    append_rows_csv,
    flatten_per_dim_alignment,
    print_alignment_table,
    print_metric_comparison,
    print_tracker_ready_report,
    try_load_reference_comparison,
)
from src.baseline_runner import run_baseline_vanilla
from src.config import SWAConfig, resolve_output_dir
from src.constants import COUNTRY_LANG
from src.controller import ImplicitSWAController
from src.data import load_multitp_dataset
from src.model import load_model, setup_seeds
from src.personas import SUPPORTED_COUNTRIES, build_country_personas
from src.scenarios import generate_multitp_scenarios
from src.swa_runner import run_country_experiment

# ─── EXP-09 shared hyperparameters ───────────────────────────────────────────
EXP_BASE = "EXP-09"
TARGET_COUNTRIES: List[str] = ["USA", "CHN", "JPN", "DEU", "BRA"]
N_SCENARIOS = 500
BATCH_SIZE = 1
SEED = 42
LAMBDA_COOP = 0.70

N_WARMUP = 50
DECAY_TAU = 100
BETA_EMA = 0.10

RESULTS_BASE = "/kaggle/working/cultural_alignment/results/exp09_model_sweep"
MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
HUMAN_AMCE_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"
PREFLIGHT_TIMEOUT_MINUTES = int(os.environ.get("EXP09_PREFLIGHT_TIMEOUT_MINUTES", "15"))


class CountryPriorState:
    """EXP-09 country prior with warmup + annealed blend."""

    def __init__(self) -> None:
        self.delta_country = 0.0
        self.step = 0
        self._history: List[float] = []

    def alpha_h(self) -> float:
        if self.step < N_WARMUP:
            return 0.0
        t = self.step - N_WARMUP
        return 1.0 - np.exp(-t / DECAY_TAU)

    def update(self, delta_opt_micro: float) -> None:
        self.delta_country = (1.0 - BETA_EMA) * self.delta_country + BETA_EMA * delta_opt_micro
        self._history.append(delta_opt_micro)
        self.step += 1

    def apply_prior(self, delta_opt_micro: float) -> float:
        a = self.alpha_h()
        return a * self.delta_country + (1.0 - a) * delta_opt_micro

    @property
    def stats(self) -> Dict:
        return {
            "step": self.step,
            "delta_country": self.delta_country,
            "alpha_h": self.alpha_h(),
            "history_std": float(np.std(self._history)) if len(self._history) > 1 else 0.0,
        }


PRIOR_STATE: Dict[str, CountryPriorState] = {}


class Exp09HierarchicalController(ImplicitSWAController):
    """Two-level hierarchical IS: country prior + per-scenario micro update."""

    def __init__(self, *args, country: str = "UNKNOWN", **kwargs):
        super().__init__(*args, **kwargs)
        self.country = country

    def _get_prior_state(self) -> CountryPriorState:
        if self.country not in PRIOR_STATE:
            PRIOR_STATE[self.country] = CountryPriorState()
        return PRIOR_STATE[self.country]

    def _pt_value(self, x: torch.Tensor) -> torch.Tensor:
        a, b, k = self.pt_alpha, self.pt_beta, self.pt_kappa
        return torch.where(x >= 0, x.abs().pow(a), -k * x.abs().pow(b))

    @torch.no_grad()
    def predict(self, user_query, preferred_on_right=True, phenomenon_category="default", lang="en"):
        db1, da1, logit_temp = self._extract_logit_gaps(user_query, phenomenon_category, lang)
        swapped_query, swap_changed = self._swap_positional_labels(user_query, lang)
        if swap_changed:
            db2, da2, _ = self._extract_logit_gaps(swapped_query, phenomenon_category, lang)
        else:
            db2, da2 = db1, da1

        delta_base = (db1 - db2) / 2.0 if swap_changed else db1
        delta_agents = (da1 - da2) / 2.0 if swap_changed else da1
        sigma = max(
            float(delta_agents.std(unbiased=True).item()) if delta_agents.numel() >= 2 else 0.0,
            self.noise_std,
        )
        anchor = delta_agents.mean()

        eps = torch.randn(self.K, device=self.device) * sigma
        delta_tilde = anchor + eps
        dist_base_to_i = (delta_base - delta_agents).abs()
        dist_cand_to_i = (delta_tilde.unsqueeze(1) - delta_agents.unsqueeze(0)).abs()
        g_per_agent = (dist_base_to_i.unsqueeze(0) - dist_cand_to_i) / sigma
        mean_v = self._pt_value(g_per_agent).mean(dim=1)
        g_cons = ((delta_base - anchor).abs() - (delta_tilde - anchor).abs()) / sigma
        U = (1.0 - self.lambda_coop) * mean_v + self.lambda_coop * self._pt_value(g_cons)
        w = F.softmax(U / self.beta, dim=0)

        k_eff = 1.0 / torch.sum(w * w).clamp_min(1e-12)
        delta_star = torch.sum(w * eps) if float(k_eff.item()) / self.K >= self.rho_eff else torch.zeros((), device=self.device)

        delta_opt_micro = float((anchor + delta_star).item())
        prior_state = self._get_prior_state()
        delta_opt_final = prior_state.apply_prior(delta_opt_micro)
        prior_state.update(delta_opt_micro)
        ps = prior_state.stats

        p_right = torch.sigmoid(torch.tensor(delta_opt_final / self.decision_temperature)).item()
        p_pref = p_right if preferred_on_right else 1.0 - p_right
        variance = float(delta_agents.var(unbiased=True).item()) if delta_agents.numel() > 1 else 0.0
        return {
            "p_right": p_right,
            "p_left": 1.0 - p_right,
            "p_spare_preferred": p_pref,
            "variance": variance,
            "sigma_used": sigma,
            "mppi_flipped": (float(anchor.item()) > 0) != (delta_opt_final > 0),
            "delta_z_norm": abs(delta_opt_final - float(anchor.item())),
            "delta_consensus": float(anchor.item()),
            "delta_opt": delta_opt_final,
            "delta_opt_micro": delta_opt_micro,
            "delta_country": ps["delta_country"],
            "alpha_h": ps["alpha_h"],
            "prior_step": ps["step"],
            "logit_temp_used": logit_temp,
            "agent_decision_gaps": delta_agents.tolist(),
            "agent_rewards": (delta_agents - delta_base).tolist(),
            "p_spare_preferred_pass1": p_pref,
            "p_spare_preferred_pass2": p_pref,
            "positional_bias": 0.0,
        }


def _patch_swa_runner_controller() -> None:
    import src.swa_runner as _swa_runner_mod
    _swa_runner_mod.ImplicitSWAController = Exp09HierarchicalController


def _free_model_cache(model_name: str) -> None:
    safe = "models--" + model_name.replace("/", "--")
    for root in [os.environ.get("HF_HUB_CACHE"), os.environ.get("HF_HOME"), os.path.expanduser("~/.cache/huggingface"), "/root/.cache/huggingface"]:
        if not root:
            continue
        hub = root if os.path.basename(root.rstrip("/")) == "hub" else os.path.join(root, "hub")
        target = os.path.join(hub, safe)
        if os.path.isdir(target):
            try:
                shutil.rmtree(target)
                print(f"[CLEANUP] removed {target}")
            except Exception as exc:
                print(f"[CLEANUP] error: {exc}")


def _build_cfg(model_name: str, swa_root: str, *, target_countries: List[str]) -> SWAConfig:
    return SWAConfig(
        model_name=model_name, n_scenarios=N_SCENARIOS, batch_size=BATCH_SIZE,
        target_countries=list(target_countries), load_in_4bit=True, use_real_data=True,
        multitp_data_path=MULTITP_DATA_PATH, wvs_data_path=WVS_DATA_PATH,
        human_amce_path=HUMAN_AMCE_PATH, output_dir=swa_root,
        lambda_coop=LAMBDA_COOP, K_samples=128,
    )


def _load_scen(cfg: SWAConfig, country: str) -> pd.DataFrame:
    lang = COUNTRY_LANG.get(country, "en")
    if cfg.use_real_data:
        df = load_multitp_dataset(
            data_base_path=cfg.multitp_data_path, lang=lang,
            translator=cfg.multitp_translator, suffix=cfg.multitp_suffix,
            n_scenarios=cfg.n_scenarios,
        )
    else:
        df = generate_multitp_scenarios(cfg.n_scenarios, lang=lang)
    df = df.copy()
    df["lang"] = lang
    return df


def _on_kaggle() -> bool:
    return os.path.isdir("/kaggle/working")


def _write_preflight_error_flag(flag_path: Path, message: str) -> None:
    flag_path.parent.mkdir(parents=True, exist_ok=True)
    flag_path.write_text(message + "\n", encoding="utf-8")
    print(f"[PREFLIGHT] wrote error flag: {flag_path}")


def _preflight_model_load(model_name: str, timeout_minutes: int) -> tuple[bool, str]:
    timeout_seconds = max(1, timeout_minutes) * 60
    code = f"""
import gc
import os
import torch
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("UNSLOTH_DISABLE_AUTO_COMPILE", "1")
os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
from src.model import load_model
model, tokenizer = load_model("{model_name}", max_seq_length=2048, load_in_4bit=True)
del model, tokenizer
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("PREFLIGHT_OK")
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"Model pre-flight timeout after {timeout_minutes} minute(s)."

    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip()[-1500:]
        return False, f"Model pre-flight failed (rc={result.returncode}). Details:\n{tail}"

    if "PREFLIGHT_OK" not in (result.stdout or ""):
        return False, "Model pre-flight did not confirm successful load."
    return True, "ok"


def _abort_kaggle_run(reason: str) -> None:
    print(f"[PREFLIGHT][ERROR] {reason}")
    if _on_kaggle():
        raise SystemExit("[PREFLIGHT] Stopping Kaggle run to avoid wasting GPU.")
    raise RuntimeError(reason)


def run_for_model(
    model_name: str,
    model_short: str,
    *,
    target_countries: Optional[List[str]] = None,
    results_base: Optional[str] = None,
) -> None:
    """Full EXP-09 run for a single model."""
    setup_seeds(SEED)
    countries: List[str] = list(target_countries) if target_countries is not None else list(TARGET_COUNTRIES)
    rb = results_base if results_base is not None else RESULTS_BASE
    exp_id = f"{EXP_BASE}-{model_short.upper()}"
    swa_root = f"{rb}/{model_short}/swa"
    cmp_root = f"{rb}/{model_short}/compare"
    for d in (swa_root, cmp_root):
        Path(d).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  {exp_id}: Hierarchical IS [{model_name}]")
    print(f"{'='*70}")
    print(f"[THEORY] alpha_h(t)=0 for t<{N_WARMUP}, then 1-exp(-(t-{N_WARMUP})/{DECAY_TAU})")
    print(f"[THEORY] delta_opt = alpha_h*delta_country + (1-alpha_h)*delta_opt_micro")

    cfg = _build_cfg(model_name, swa_root, target_countries=countries)
    out_dir = Path(swa_root) / resolve_output_dir("", model_name).strip("/\\")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(out_dir)
    preflight_flag = out_dir / "preflight_error.flag"

    print(f"[PREFLIGHT] checking model load with timeout={PREFLIGHT_TIMEOUT_MINUTES} minute(s)")
    ok, message = _preflight_model_load(model_name, PREFLIGHT_TIMEOUT_MINUTES)
    if not ok:
        _write_preflight_error_flag(preflight_flag, message)
        _abort_kaggle_run(message)
    if preflight_flag.exists():
        preflight_flag.unlink(missing_ok=True)
    print("[PREFLIGHT] PASS")

    model, tokenizer = load_model(model_name, max_seq_length=2048, load_in_4bit=True)
    rows: List[dict] = []
    h_method = f"{exp_id}_hierarchical_is"
    try:
        for ci, country in enumerate(countries):
            if country not in SUPPORTED_COUNTRIES:
                continue
            print(f"\n[{ci+1}/{len(countries)}] {exp_id} | {country}")
            scen = _load_scen(cfg, country)
            personas = build_country_personas(country, wvs_path=WVS_DATA_PATH)

            bl = run_baseline_vanilla(model, tokenizer, scen, country, cfg)
            bl["results_df"].to_csv(out_dir / f"vanilla_results_{country}.csv", index=False)
            rows.append({
                "model": model_name, "method": "baseline_vanilla", "country": country,
                **{f"align_{k}": v for k, v in bl["alignment"].items()},
                "flip_rate": float("nan"), "n_scenarios": len(bl["results_df"]),
            })

            PRIOR_STATE.clear()
            PRIOR_STATE[country] = CountryPriorState()
            orig_init = Exp09HierarchicalController.__init__
            def patched_init(self, *a, country=country, **kw):
                orig_init(self, *a, country=country, **kw)
            Exp09HierarchicalController.__init__ = patched_init
            _patch_swa_runner_controller()
            results_df, summary = run_country_experiment(model, tokenizer, country, personas, scen, cfg)
            Exp09HierarchicalController.__init__ = orig_init

            results_df.to_csv(out_dir / f"swa_results_{country}.csv", index=False)
            append_rows_csv(
                str(Path(cmp_root) / "per_dim_breakdown.csv"),
                flatten_per_dim_alignment(summary.get("per_dimension_alignment", {}), model=model_name, method=h_method, country=country),
            )
            ps = PRIOR_STATE.get(country, CountryPriorState()).stats
            rows.append({
                "model": model_name, "method": h_method, "country": country,
                **{f"align_{k}": v for k, v in summary["alignment"].items()},
                "flip_rate": summary["flip_rate"], "n_scenarios": summary["n_scenarios"],
                "final_delta_country": ps["delta_country"], "final_alpha_h": ps["alpha_h"], "history_std": ps["history_std"],
            })

            torch.cuda.empty_cache()
            gc.collect()
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _free_model_cache(model_name)

    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(Path(cmp_root) / "comparison.csv", index=False)
    vanilla_df = cmp_df[cmp_df["method"] == "baseline_vanilla"].copy()
    hier_df = cmp_df[cmp_df["method"] == h_method].copy()
    if not vanilla_df.empty:
        print_alignment_table(vanilla_df, title=f"{exp_id} VANILLA")
    print_alignment_table(hier_df, title=f"{exp_id} HIERARCHICAL IS")
    if not vanilla_df.empty and not hier_df.empty:
        print_metric_comparison(
            vanilla_df, hier_df, title=f"{exp_id} vs Vanilla (MIS)",
            spec=CompareSpec(metric_col="align_mis", ref_method="baseline_vanilla", cur_method=h_method),
        )
    ref = try_load_reference_comparison()
    if ref is not None and not hier_df.empty:
        print_metric_comparison(
            ref, hier_df, title=f"{exp_id} vs EXP-01 SWA-PTIS (MIS)",
            spec=CompareSpec(metric_col="align_mis", ref_method="swa_ptis", cur_method=h_method),
        )
    print_tracker_ready_report(cmp_df, exp_id=exp_id, cur_method=h_method, per_dim_csv_path=str(Path(cmp_root) / "per_dim_breakdown.csv"))
    print(f"\n[{exp_id}] DONE — {cmp_root}")
