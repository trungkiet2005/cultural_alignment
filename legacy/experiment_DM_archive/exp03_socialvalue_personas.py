#!/usr/bin/env python3
"""
EXP-03: SocialValue-Targeted Social-Utility Personas
=====================================================

**Motivation** (from EXP-01 analysis, docs/experiment_tracker.md Insight 1):

SocialValue_High is the #1 error dimension across ALL models and ALL countries.
Mean error = 27.0 points (model predicts ~35% for exec vs human 67%).

Root cause: All 4 WVS-based personas are politically/morally egalitarian — they
emphasize dignity, equality, and non-discrimination. The persona-mean anchor
`mean(delta_i)` is NEGATIVE for SocialValue scenarios (all agents prefer
not-saving-executives). Since `delta_opt = anchor + delta_star`, the IS update
cannot bring delta_opt positive even with optimal sampling.

**Fix**: Replace the standard utilitarian anchor (P4) with TWO culturally-specific
"social utility gradient" personas per country:
  - P4: A professional ethics / triage-based social utility voice
  - P5: A Confucian/meritocratic social hierarchy voice (culturally appropriate)

This shifts `mean(delta_i)` toward 0 or mildly positive for SocialValue, enabling
the IS update to correctly weight scenarios where social expertise matters.

**Key changes** vs EXP-01:
  - 5 personas total (3 WVS age cohorts + 2 social-utility)
  - Social-utility personas authored in the country's native language
  - All other hyperparameters identical to EXP-01 for clean comparison
  - `lambda_coop=0.6` (slightly less consensus weight; social-utility agents
    should have pull, not be averaged away)

**Expected**: SocialValue_High |err| drops from 27.0 -> <15; MIS improvement
on Qwen improves from +21.5% -> +28%+ averaged across countries.

Usage on Kaggle
---------------
    !python experiment/exp03_socialvalue_personas.py
"""

# ============================================================================
# Step 0: env vars MUST be set before any torch import
# ============================================================================
import os, sys, subprocess

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("UNSLOTH_DISABLE_AUTO_COMPILE", "1")

# ============================================================================
# Step 1: bootstrap
# ============================================================================
REPO_URL = "https://github.com/trungkiet2005/cultural_alignment.git"
REPO_DIR_KAGGLE = "/kaggle/working/cultural_alignment"


def _on_kaggle() -> bool:
    return os.path.isdir("/kaggle/working")


def _ensure_repo() -> str:
    here = os.getcwd()
    if os.path.isfile(os.path.join(here, "src", "controller.py")):
        return here
    if not _on_kaggle():
        raise RuntimeError("Not on Kaggle and not inside the repo.")
    if not os.path.isdir(REPO_DIR_KAGGLE):
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR_KAGGLE], check=True)
    os.chdir(REPO_DIR_KAGGLE)
    if REPO_DIR_KAGGLE not in sys.path:
        sys.path.insert(0, REPO_DIR_KAGGLE)
    return REPO_DIR_KAGGLE


def _install_deps() -> None:
    if not _on_kaggle():
        return
    for c in [
        "pip install -q bitsandbytes scipy tqdm matplotlib seaborn",
        "pip install --upgrade --no-deps unsloth",
        "pip install -q unsloth_zoo",
        "pip install --quiet --no-deps --force-reinstall pyarrow",
        "pip install --quiet 'datasets>=3.4.1,<4.4.0'",
    ]:
        subprocess.run(c, shell=True, check=False)


_REPO_DIR = _ensure_repo()
_install_deps()

# ============================================================================
# Step 2: imports
# ============================================================================
import gc, shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
try:
    torch._dynamo.config.disable = True
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass
import torch.nn.functional as F
import pandas as pd

from experiment_DM.exp_reporting import (
    CompareSpec,
    append_rows_csv,
    flatten_per_dim_alignment,
    print_alignment_table,
    print_metric_comparison,
    try_load_reference_comparison,
)

from src.config import SWAConfig, BaselineConfig, resolve_output_dir
from src.constants import COUNTRY_LANG
from src.model import setup_seeds, load_model
from src.data import load_multitp_dataset
from src.scenarios import generate_multitp_scenarios
from src.personas import build_country_personas, SUPPORTED_COUNTRIES
from src.controller import ImplicitSWAController
import src.swa_runner as _swa_runner_mod
from src.swa_runner import run_country_experiment
from src.baseline_runner import run_baseline_vanilla

# ============================================================================
# Step 3: experiment configuration
# ============================================================================
EXP_ID   = "EXP-03"
EXP_NAME = "socialvalue_targeted_personas"

MODELS: List[str] = [
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "unsloth/gemma-2-9b-it-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
]
TARGET_COUNTRIES: List[str] = ["USA", "CHN", "JPN", "DEU", "BRA"]
N_SCENARIOS: int = 500
BATCH_SIZE: int = 1
SEED: int = 42
SKIP_BASELINE: bool = True  # baseline from EXP-01 reused

SWA_ROOT: str = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/swa"
CMP_ROOT: str = f"/kaggle/working/cultural_alignment/results/{EXP_NAME}/compare"

MULTITP_DATA_PATH = "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data"
WVS_DATA_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/"
    "WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
)
HUMAN_AMCE_PATH = (
    "/kaggle/input/datasets/trungkiet/mutltitp-data/data/data/country_specific_ACME.csv"
)

# ============================================================================
# Step 4: Social-Utility Persona Library
#
# Design principles:
#   1. Each persona must articulate WHY professional/skilled/high-status people
#      have HIGHER marginal social value (not just that they deserve to live).
#   2. The reasoning must be culturally grounded (not generic utilitarian).
#   3. Written in the country's native language to match scenario prompts.
#   4. ~150 tokens to match WVS persona length (balanced prefix budget).
# ============================================================================
SOCIAL_UTILITY_PERSONAS: Dict[str, List[str]] = {
    "USA": [
        # P4: Medical triage ethics
        "You are a senior emergency-room physician with 20 years of experience in mass-casualty triage. "
        "Your ethical framework is based on the principle of 'save who you can save, maximize life-years saved'. "
        "You believe doctors, teachers, and skilled professionals carry a higher marginal social value "
        "because their work directly multiplies the well-being of others. In an unavoidable dilemma, "
        "you prioritize those whose skills will prevent future harm to many more people.",

        # P5: Meritocratic social capital
        "You are an economist who studies human capital and social productivity. "
        "You believe that individuals who have invested heavily in acquiring rare, society-critical skills "
        "(executives managing large teams, doctors, engineers) represent embodied social capital that "
        "took decades and enormous resources to build. In a tragic, unavoidable tradeoff, you reason that "
        "preserving high-skill, high-social-contribution individuals maximizes aggregate social welfare. "
        "This is not about personal worth — every life has dignity — but about minimizing societal loss."
    ],

    "CHN": [
        # P4: Confucian meritocratic hierarchy (社会贡献的儒家观点)
        "你是一位深信儒家精英主义的中国学者。你认为，通过多年苦学和奋斗取得社会地位的人，"
        "代表了整个社会的精华与进步动力。医生、工程师和企业高管承载着更大的社会责任，"
        "他们的技能和经验是不可替代的社会资产。在无法避免的悲剧中，保全对社会贡献最大的人，"
        "是最符合集体利益的理性选择。这不是歧视弱势群体，而是最大化整体社会价值的务实判断。",

        # P5: Social investment ethics
        "你是一位研究人力资本的中国经济学家。你的研究表明：高度专业化的人才（高管、医生）"
        "是国家经济和社会体系运转的支柱。社会为培养每一位高学历、高技能人才投入了大量资源。"
        "在残酷的道德困境中，你倾向于以保存社会平均效益最大化的原则进行判断——这意味着"
        "优先保全那些技能最稀缺、替代成本最高的个体，以维护整个社会体系的可持续运转。"
    ],

    "JPN": [
        # P4: Japanese professional duty ethics (職業的責任の倫理)
        "あなたは日本の救急医療のトリアージ専門家です。長年の経験から、困難な状況での意思決定は"
        "「社会的影響の最大化」という原則に基づくべきだと確信しています。医師、経営者、熟練技術者は、"
        "その専門知識と経験を通じて、社会全体に多大な貢献をもたらします。"
        "不可避のトレードオフでは、社会的波及効果が最も大きい人を優先することが、"
        "集合的利益の観点から合理的な判断です。これは人間の価値の差別ではなく、社会損失の最小化です。",

        # P5: Confucian-Japanese social hierarchy
        "あなたは日本の伝統的な職業倫理と現代的な功利主義の融合を研究する社会哲学者です。"
        "日本の和の精神は、個人ではなく共同体全体の繁栄を重視します。この観点から、"
        "専門的スキルと社会的役割を持つ人々（医師・管理職・熟練職人）は、"
        "社会秩序と集団的機能の維持に不可欠な存在です。"
        "極限の選択では、社会全体への貢献ポテンシャルを基準に判断することが、和の実現につながります。"
    ],

    "DEU": [
        # P4: Professional ethics + Kantian value of expertise
        "Du bist ein deutscher Notfallmediziner und Ethiker, der auf Katastrophenmedizin spezialisiert ist. "
        "Dein ethisches Fundament verbindet kantische Pflichtethik mit utilitaristischen Triage-Grundsätzen. "
        "Du glaubst, dass Fachleute mit seltenen, lebenswichtigen Fähigkeiten (Ärzte, Ingenieure, Führungskräfte) "
        "einen höheren sozialen Grenznutzen haben, weil ihre Expertise viele andere schützt. "
        "In einem unvermeidbaren Dilemma priorisierst du die Erhaltung sozialen Humankapitals, "
        "das nicht kurzfristig ersetzt werden kann.",

        # P5: German social market economy ethics
        "Du bist ein Ökonom, der am Deutschen Institut für Wirtschaftsforschung arbeitet. "
        "Du anerkennst, dass die Soziale Marktwirtschaft auf Fachkräften und qualifizierten Führungskräften beruht. "
        "Der Verlust eines erfahrenen Chirurgen oder einer Führungskraft, die 200 Mitarbeiter leitet, "
        "hat einen gesellschaftlichen Schaden weit über den individuellen Tod hinaus. "
        "In tragischen Dilemma-Situationen, wo alle Beteiligten einen intrinsischen Wert haben, "
        "minimierst du den Gesamtschaden für die Gesellschaft als rationale Entscheidungsgrundlage."
    ],

    "BRA": [
        # P4: Brazilian medical ethics + triage
        "Você é um médico brasileiro especializado em medicina de emergência e bioética. "
        "Seu treinamento em triagem de múltiplas vítimas ensina que, quando recursos são escassos, "
        "o objetivo é maximizar anos de vida salvos E preservar a capacidade social de cuidar de mais pessoas. "
        "Você acredita que médicos, engenheiros e líderes empresariais carregam um capital humano raro "
        "que demora décadas para ser formado. Num dilema trágico inevitável, você pondera tanto o número "
        "de vidas quanto o impacto multiplicador futuro de cada grupo para a sociedade.",

        # P5: Social capital and development ethics
        "Você é um pesquisador brasileiro de desenvolvimento humano e capital social. "
        "Suas pesquisas mostram que profissionais altamente qualificados (executivos, médicos) "
        "têm efeitos multiplicadores sobre o bem-estar coletivo — cada um deles impacta dezenas ou centenas "
        "de vidas por meio de seu trabalho. Num país em desenvolvimento como o Brasil, o capital humano "
        "qualificado é especialmente escasso e valioso. Em escolhas trágicas inevitáveis, você raciocina que "
        "preservar quem tem maior potencial de impacto social futuro minimiza o dano coletivo total."
    ],
}


def build_socialvalue_targeted_personas(country_iso: str, wvs_path: str = "") -> List[str]:
    """
    Return 5 personas:
      P1-P3: Standard WVS age cohort personas (young/middle/older)
      P4-P5: Social-utility gradient personas (culturally grounded)

    If no WVS data, uses BASE_PERSONAS[:3] + social utility personas.
    Falls back to English social utility if country not in library.
    """
    # Get WVS base personas (P1=young, P2=middle, P3=older, P4=utilitarian_standard)
    wvs_personas = build_country_personas(country_iso, wvs_path=wvs_path)
    p_young, p_middle, p_older = wvs_personas[0], wvs_personas[1], wvs_personas[2]

    # Get social utility personas
    su_personas = SOCIAL_UTILITY_PERSONAS.get(country_iso, SOCIAL_UTILITY_PERSONAS["USA"])

    personas_5 = [
        p_young,         # P1: WVS young (egalitarian leaning)
        p_middle,        # P2: WVS middle (egalitarian leaning)
        p_older,         # P3: WVS older (egalitarian leaning)
        su_personas[0],  # P4: Social utility - professional ethics
        su_personas[1],  # P5: Social utility - capital/meritocracy
    ]
    print(f"[EXP-03] Built {len(personas_5)} personas for {country_iso} "
          f"(3 WVS age + 2 social-utility targeted)")
    return personas_5


# ============================================================================
# Step 5: EXP-03 Controller — identical math, modified lambda_coop
# ============================================================================
class Exp03SWAController(ImplicitSWAController):
    """
    SWA-PTIS with social-utility-augmented persona pool.

    Math is identical to the paper (EXP-01 PaperSWAController).
    The only changes are:
      - 5 personas (not 4): 3 WVS + 2 social-utility
      - lambda_coop = 0.60 (lower than EXP-01's 0.70)
        Justification: The social-utility agents are "opinionated" on SocialValue
        but agree with WVS agents on other dimensions. We want individual agent
        influence (mean_v term) to dominate over consensus (v_cons term), because
        the consensus now correctly captures SocialValue via the new agents.
    """

    def _pt_value(self, x: torch.Tensor) -> torch.Tensor:
        a, b, k = self.pt_alpha, self.pt_beta, self.pt_kappa
        return torch.where(x >= 0, x.abs().pow(a), -k * x.abs().pow(b))

    @torch.no_grad()
    def predict(
        self,
        user_query: str,
        preferred_on_right: bool = True,
        phenomenon_category: str = "default",
        lang: str = "en",
    ) -> Dict:
        # Two-pass positional debiasing (identical to paper)
        db1, da1, logit_temp = self._extract_logit_gaps(user_query, phenomenon_category, lang)
        swapped_query, swap_changed = self._swap_positional_labels(user_query, lang)
        if swap_changed:
            db2, da2, _ = self._extract_logit_gaps(swapped_query, phenomenon_category, lang)
        else:
            db2, da2 = db1, da1

        delta_base   = (db1 - db2) / 2.0 if swap_changed else db1
        delta_agents = (da1 - da2) / 2.0 if swap_changed else da1

        # Adaptive sigma (floored at noise_std)
        sigma = max(
            float(delta_agents.std(unbiased=True).item()) if delta_agents.numel() >= 2 else self.noise_std,
            self.noise_std
        )
        anchor = delta_agents.mean()
        K, device = self.K, self.device

        eps         = torch.randn(K, device=device) * sigma
        delta_tilde = anchor + eps

        dist_base_to_i = (delta_base - delta_agents).abs()
        dist_cand_to_i = (delta_tilde.unsqueeze(1) - delta_agents.unsqueeze(0)).abs()
        g_per_agent    = (dist_base_to_i.unsqueeze(0) - dist_cand_to_i) / sigma

        v_per_agent = self._pt_value(g_per_agent)
        mean_v      = v_per_agent.mean(dim=1)

        g_cons = ((delta_base - anchor).abs() - (delta_tilde - anchor).abs()) / sigma
        v_cons = self._pt_value(g_cons)

        U = (1.0 - self.lambda_coop) * mean_v + self.lambda_coop * v_cons
        w = F.softmax(U / self.beta, dim=0)

        k_eff      = 1.0 / torch.sum(w * w).clamp_min(1e-12)
        delta_star = torch.sum(w * eps) if float(k_eff.item()) / K >= self.rho_eff else torch.zeros((), device=device)

        delta_opt = anchor + delta_star
        p_right   = torch.sigmoid(delta_opt / self.decision_temperature).item()
        p_pref    = p_right if preferred_on_right else 1.0 - p_right
        variance  = float(delta_agents.var(unbiased=True).item()) if delta_agents.numel() > 1 else 0.0

        return {
            "p_right": p_right,
            "p_left": 1.0 - p_right,
            "p_spare_preferred": p_pref,
            "variance": variance,
            "sigma_used": float(sigma),
            "mppi_flipped": (float(anchor.item()) > 0) != (float(delta_opt.item()) > 0),
            "delta_z_norm": abs(float(delta_star.item())),
            "delta_consensus": float(anchor.item()),
            "delta_opt": float(delta_opt.item()),
            "logit_temp_used": logit_temp,
            "n_personas": len(self.personas),
            "agent_decision_gaps": delta_agents.tolist(),
            "agent_rewards": (delta_agents - delta_base).tolist(),
            "p_spare_preferred_pass1": p_pref,
            "p_spare_preferred_pass2": p_pref,
            "positional_bias": 0.0,
        }


_swa_runner_mod.ImplicitSWAController = Exp03SWAController


# ============================================================================
# Step 6: runner helpers
# ============================================================================
def _dir_size_gb(path: str) -> float:
    total = sum(
        os.path.getsize(os.path.join(d, f))
        for d, _, files in os.walk(path) for f in files
        if not os.path.islink(os.path.join(d, f))
    )
    return total / 1e9


def _free_model_cache(model_name: str) -> None:
    safe = "models--" + model_name.replace("/", "--")
    for root in [os.environ.get("HF_HUB_CACHE"), os.environ.get("HF_HOME"),
                 os.path.expanduser("~/.cache/huggingface"), "/root/.cache/huggingface"]:
        if not root:
            continue
        hub_dir = root if os.path.basename(root.rstrip("/")) == "hub" else os.path.join(root, "hub")
        target = os.path.join(hub_dir, safe)
        if os.path.isdir(target):
            try:
                shutil.rmtree(target)
                print(f"[CLEANUP] removed {target}")
            except Exception as e:
                print(f"[CLEANUP] error: {e}")


def _build_swa_config(model_name: str) -> SWAConfig:
    return SWAConfig(
        model_name=model_name,
        n_scenarios=N_SCENARIOS,
        batch_size=BATCH_SIZE,
        target_countries=list(TARGET_COUNTRIES),
        load_in_4bit=True,
        use_real_data=True,
        multitp_data_path=MULTITP_DATA_PATH,
        wvs_data_path=WVS_DATA_PATH,
        human_amce_path=HUMAN_AMCE_PATH,
        output_dir=SWA_ROOT,
        lambda_coop=0.60,   # reduced to let social-utility agents influence individual scores
        K_samples=128,       # same as EXP-01 for clean comparison
    )


def _load_country_scenarios(cfg, country: str) -> pd.DataFrame:
    lang = COUNTRY_LANG.get(country, "en")
    if cfg.use_real_data:
        df = load_multitp_dataset(
            data_base_path=cfg.multitp_data_path,
            lang=lang,
            translator=cfg.multitp_translator,
            suffix=cfg.multitp_suffix,
            n_scenarios=cfg.n_scenarios,
        )
    else:
        df = generate_multitp_scenarios(cfg.n_scenarios, lang=lang)
    df = df.copy()
    df["lang"] = lang
    return df


def _run_swa_for_model(model, tokenizer, model_name: str) -> List[dict]:
    cfg = _build_swa_config(model_name)
    model_slug_dir = resolve_output_dir("", model_name).strip("/\\")
    out_dir = Path(SWA_ROOT) / model_slug_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(out_dir)
    print(f"\n{'#'*70}\n# {EXP_ID} [{model_name}] -> {out_dir}\n{'#'*70}")

    rows = []
    for ci, country in enumerate(cfg.target_countries):
        if country not in SUPPORTED_COUNTRIES:
            continue
        print(f"\n[{ci+1}/{len(cfg.target_countries)}] {EXP_ID} {model_name} | {country}")
        scen     = _load_country_scenarios(cfg, country)
        personas = build_socialvalue_targeted_personas(country, wvs_path=WVS_DATA_PATH)
        results_df, summary = run_country_experiment(model, tokenizer, country, personas, scen, cfg)
        results_df.to_csv(out_dir / f"swa_results_{country}.csv", index=False)
        append_rows_csv(
            str(Path(CMP_ROOT) / "per_dim_breakdown.csv"),
            flatten_per_dim_alignment(
                summary.get("per_dimension_alignment", {}),
                model=model_name,
                method=f"{EXP_ID}_socialvalue_targeted",
                country=country,
            ),
        )
        rows.append({
            "model":   model_name,
            "method":  f"{EXP_ID}_socialvalue_targeted",
            "country": country,
            **{f"align_{k}": v for k, v in summary["alignment"].items()},
            "flip_rate":       summary["flip_rate"],
            "mean_latency_ms": summary["mean_latency_ms"],
            "n_scenarios":     summary["n_scenarios"],
            "n_personas":      5,
            "lambda_coop":     0.60,
        })
        torch.cuda.empty_cache()
        gc.collect()
    return rows


# ============================================================================
# Step 7: main
# ============================================================================
def main() -> None:
    setup_seeds(SEED)
    for d in (SWA_ROOT, CMP_ROOT):
        Path(d).mkdir(parents=True, exist_ok=True)

    print(f"\n[{EXP_ID}] {EXP_NAME}")
    print(f"[CONFIG] Targeting SocialValue_High error (EXP-01 mean err=27.0)")
    print(f"[CONFIG] 5 personas: 3 WVS + 2 social-utility (per country)")
    print(f"[CONFIG] lambda_coop=0.60 (vs 0.70 in EXP-01)")

    all_rows: List[dict] = []
    for mi, model_name in enumerate(MODELS):
        print(f"\n{'='*70}\n  MODEL {mi+1}/{len(MODELS)}: {model_name}\n{'='*70}")
        model, tokenizer = load_model(model_name, max_seq_length=2048, load_in_4bit=True)
        try:
            all_rows.extend(_run_swa_for_model(model, tokenizer, model_name))
        finally:
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _free_model_cache(model_name)

        pd.DataFrame(all_rows).to_csv(Path(CMP_ROOT) / "comparison.csv", index=False)
        print(f"[SAVE] partial -> {CMP_ROOT}/comparison.csv  ({len(all_rows)} rows)")

    cmp_df = pd.DataFrame(all_rows)
    cmp_df.to_csv(Path(CMP_ROOT) / "comparison.csv", index=False)
    print_alignment_table(cmp_df, title=f"{EXP_ID} RESULTS — {EXP_NAME}")

    ref = try_load_reference_comparison()
    if ref is not None:
        print_metric_comparison(
            ref,
            cmp_df,
            title=f"{EXP_ID} vs EXP-01 (reference) — MIS",
            spec=CompareSpec(
                metric_col="align_mis",
                ref_method="swa_ptis",
                cur_method=f"{EXP_ID}_socialvalue_targeted",
            ),
        )
        print_metric_comparison(
            ref,
            cmp_df,
            title=f"{EXP_ID} vs EXP-01 (reference) — JSD",
            spec=CompareSpec(
                metric_col="align_jsd",
                ref_method="swa_ptis",
                cur_method=f"{EXP_ID}_socialvalue_targeted",
            ),
        )

    print(f"\n[{EXP_ID}] DONE — results under {CMP_ROOT}")


if __name__ == "__main__":
    main()
