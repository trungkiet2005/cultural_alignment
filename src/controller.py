"""ImplicitSWAController: SWA-PTIS engine for cultural value negotiation.

SWA-PTIS = Socially-Weighted Alignment with Prospect-Theory Importance Sampling.

The algorithm applies a single-step, scalar, KL-regularised importance-sampling
update with a cooperative Prospect-Theory utility aggregated over N culturally
grounded persona agents. See `src/controller.py::_is_solve_decision` for the
math; it corresponds exactly to Eqs. (5)-(10) of the paper.

Math summary:
  Per-agent gain (logit-gap units, bounded, no unit mismatch, no explosion):
      g_{i,k} = |delta_base - delta_i| - |delta_tilde_k - delta_i|
  Consensus-target gain:
      g_cons_k = |delta_base - delta_bar| - |delta_tilde_k - delta_bar|
  Collective utility (mean-of-v, NOT v-of-mean -- preserves loss aversion):
      U(eps_k) = (1 - lambda_coop) * mean_i v(g_{i,k})
               +       lambda_coop  * v(g_cons_k)
               -       alpha_ctl    * eps_k^2 / (2 * sigma^2)
  Softmax weights / IS update:
      w_k = softmax(U(eps_k) / eta)
      delta_star = sum_k w_k * eps_k
"""

import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from src.i18n import (
    BASE_ASSISTANT_I18N,
    PROMPT_FRAME_I18N,
    SCENARIO_FRAME_I18N,
)
from src.model import ChatTemplateHelper


# ============================================================================
# CORE SWA-PTIS ENGINE (v4 — corrected math, all MPPI framing removed)
# ============================================================================
class ImplicitSWAController:
    """
    SWA-PTIS: Socially-Weighted Alignment with Prospect-Theory Importance Sampling.

    Applies a single-step KL-regularised importance-sampling update in logit
    space with a cooperative Prospect-Theory utility. This is a standard
    self-normalised importance sampler with a Boltzmann policy target; it is
    NOT path-integral MPPI and does not inherit MPPI's multi-step derivation.

    Features:
      - Per-category logit temperature (T_cat) applied per predict() call
      - Importance-sampling update runs unconditionally; self-attenuates at
        consensus (delta_star -> 0 as all delta_i -> delta_base)
      - Prospect-Theory value function applied PER-PERSONA before averaging,
        so loss aversion is preserved (no Jensen violation)
      - Per-agent gain g_i = |delta_base - delta_i| - |delta_tilde - delta_i|
        lives in logit-gap units (same as delta) — no sigma^2 amplification,
        no logit-gap-squared unit mismatch
      - Logit-level positional debiasing (two-pass A↔B swap)
    """

    def __init__(
        self,
        model,
        tokenizer,
        personas: List[str],
        lambda_coop: float = 0.7,
        alpha_ctl: float = 0.05,  # quadratic control cost strength (was alpha_kl)
        K_samples: int = 128,
        noise_std: float = 0.3,
        temperature: float = 0.5,
        logit_temperature: float = 3.0,
        category_logit_temperatures: Optional[Dict[str, float]] = None,
        pt_alpha: float = 0.88,
        pt_beta: float = 0.88,
        pt_kappa: float = 2.25,
        decision_temperature: float = 1.0,
        assistant_lang: str = "en",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.personas = personas
        self.N = len(personas)
        self.assistant_lang = assistant_lang
        self.lambda_coop = lambda_coop
        self.alpha_ctl = alpha_ctl
        self.pt_alpha = pt_alpha
        self.pt_beta = pt_beta
        self.pt_kappa = pt_kappa
        self.K = K_samples
        self.noise_std = noise_std
        self.beta = temperature
        self.logit_temperature = logit_temperature
        self.category_logit_temperatures = category_logit_temperatures or {}
        self.decision_temperature = decision_temperature
        self.device = next(model.parameters()).device
        self.chat_helper = ChatTemplateHelper(tokenizer)

        # Per-language cache of (a_token_id, b_token_id) for the answer position.
        # MUST be resolved per-language because the answer token differs based on
        # the trailing characters of the prompt template (e.g. ' A' with leading
        # space vs ':A' merged vs 'A' bare). See _resolve_decision_tokens_for_lang.
        self._answer_token_cache: Dict[str, Tuple[int, int]] = {}

        self.pad_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )

        self._build_persona_prefixes()

    def _resolve_decision_tokens_for_lang(self, lang: str) -> Tuple[int, int]:
        """Find the token ids that the model would emit for "A"/"B" answers
        in this specific language's prompt.

        This MUST be resolved using the actual chat-templated, formatted prompt
        because BPE tokenizers (Llama 3, Qwen 2.5, GPT-OSS, Mistral-NeMo) merge
        the answer letter with the preceding character. For example, in Qwen2.5
        with the English template ending in "Choice: " (trailing space), the
        emitted answer token is " A" (id 362), NOT bare "A" (id 32). Reading
        the logit at id 32 gives a token that has nothing to do with the model's
        decision — essentially noise.

        SentencePiece tokenizers (Llama 2, Mistral 7B v0.1) happen to be robust
        to the naive `encode("A")[0]` because their metaspace prefix produces
        the same `▁A` token regardless of context, but BPE tokenizers are not.
        We always use the per-language formatted-prompt resolution for safety.
        """
        if lang in self._answer_token_cache:
            return self._answer_token_cache[lang]

        frame = PROMPT_FRAME_I18N.get(lang, PROMPT_FRAME_I18N["en"])
        # Dummy scenario — the answer suffix is what matters, not the content.
        user_content = frame.format(scenario="DUMMY_SCENARIO_FOR_TOKEN_RESOLUTION")
        formatted = self.chat_helper.format_query_with_suffix(user_content)

        base_ids = self.tokenizer.encode(formatted, add_special_tokens=False)
        a_full = self.tokenizer.encode(formatted + "A", add_special_tokens=False)
        b_full = self.tokenizer.encode(formatted + "B", add_special_tokens=False)

        def _first_diff(base: list, full: list) -> int:
            n = min(len(base), len(full))
            for i in range(n):
                if base[i] != full[i]:
                    return i
            return n

        a_pos = _first_diff(base_ids, a_full)
        b_pos = _first_diff(base_ids, b_full)
        if a_pos >= len(a_full) or b_pos >= len(b_full):
            raise RuntimeError(
                f"[SWA] Failed to resolve A/B answer tokens for lang={lang}: "
                f"prompt+'A'/'B' produced no new token. base_len={len(base_ids)}, "
                f"a_len={len(a_full)}, b_len={len(b_full)}"
            )
        a_id = a_full[a_pos]
        b_id = b_full[b_pos]
        self._answer_token_cache[lang] = (a_id, b_id)
        a_str = self.tokenizer.decode([a_id])
        b_str = self.tokenizer.decode([b_id])
        print(f"[SWA] Decision tokens for lang={lang}: "
              f"A={a_id}({a_str!r})  B={b_id}({b_str!r})")
        return a_id, b_id

    @torch.no_grad()
    def _build_persona_prefixes(self):
        print(
            f"[SWA] Building persona prefixes for {self.N} agents + 1 base "
            f"(assistant_lang={self.assistant_lang!r})..."
        )
        t0 = time.time()

        self.persona_prefix_ids = []
        for persona_text in self.personas:
            ids = self.chat_helper.build_prefix_ids(persona_text, self.device)
            self.persona_prefix_ids.append(ids)

        base_text = BASE_ASSISTANT_I18N.get(
            self.assistant_lang, BASE_ASSISTANT_I18N["en"]
        )
        self.base_prefix_ids = self.chat_helper.build_prefix_ids(
            base_text, self.device
        )

        elapsed = time.time() - t0
        print(f"[SWA] Prefix tokenisation: {elapsed:.2f}s")

    # ------------------------------------------------------------------
    # Core forward: batched evaluation of base + N persona agents
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _evaluate_all_agents(
        self,
        query_ids: torch.Tensor,
        lang: str,
        logit_temp: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if logit_temp is None:
            logit_temp = self.logit_temperature

        a_id, b_id = self._resolve_decision_tokens_for_lang(lang)

        all_prefixes = [self.base_prefix_ids] + self.persona_prefix_ids
        seqs = [torch.cat([p, query_ids], dim=1) for p in all_prefixes]
        lengths = torch.tensor(
            [s.shape[1] for s in seqs], dtype=torch.long, device=self.device,
        )
        max_len = int(lengths.max().item())

        # Right-pad (not left-pad) and DO NOT pass attention_mask. Unsloth's
        # patched attention for Gemma2 / GPT-OSS mishandles 2D masks under
        # Transformers >=5.2 (Gemma2: broadcast bug in slow_attention_softcapping;
        # GPT-OSS: IndexError in inplace_eager_attention_forward). With causal
        # attention + right-pad, logits at `lengths[i] - 1` depend only on real
        # tokens 0..len-1, so no mask is required.
        batch_ids = torch.full(
            (len(seqs), max_len), self.pad_id,
            dtype=seqs[0].dtype, device=self.device,
        )
        for i, s in enumerate(seqs):
            batch_ids[i, : s.shape[1]] = s[0]

        out = self.model(input_ids=batch_ids, use_cache=False)
        batch_idx = torch.arange(len(seqs), device=self.device)
        logits = out.logits[batch_idx, lengths - 1, :]

        # Index 0 -> "A" token, index 1 -> "B" token (per-language correct ids).
        z_decision = logits[:, [a_id, b_id]] / logit_temp
        z_base = z_decision[0:1]
        z_agents = z_decision[1:]
        return z_base, z_agents

    @torch.no_grad()
    def _compute_decision_rewards(
        self, z_base: torch.Tensor, z_agents: torch.Tensor
    ) -> Tuple[torch.Tensor, float, torch.Tensor]:
        delta_base = z_base[:, 1] - z_base[:, 0]
        delta_agents = z_agents[:, 1] - z_agents[:, 0]
        r_agents = delta_agents - delta_base.squeeze()
        delta_consensus = delta_agents.mean()
        variance = torch.var(delta_agents).item()   # diagnostic (χ²(N-1) scaled)
        return r_agents, variance, delta_consensus

    def _adaptive_noise_std(self, z_agents: torch.Tensor) -> float:
        """Compute per-scenario adaptive noise std from the empirical spread of
        agent logit gaps (T_cat-scaled).  Using the inter-agent std as σ ensures
        the perturbation magnitude is commensurate with the actual disagreement
        in logit space rather than being fixed across all models and categories.

        Falls back to self.noise_std when fewer than 2 agents are available or
        when the empirical std is zero (all agents agree exactly).
        """
        delta_agents = z_agents[:, 1] - z_agents[:, 0]
        if delta_agents.numel() < 2:
            return self.noise_std
        std = float(delta_agents.std(unbiased=True).item())
        return std if std > 1e-6 else self.noise_std

    def _prospect_value(self, x: torch.Tensor) -> torch.Tensor:
        """Prospect Theory value function (Kahneman & Tversky, 1979).

        v(x) =  x^α           if x ≥ 0   (diminishing sensitivity to gains)
        v(x) = -κ · |x|^β     if x < 0   (loss aversion + diminishing sensitivity)
        """
        return torch.where(
            x >= 0,
            x.abs().pow(self.pt_alpha),
            -self.pt_kappa * x.abs().pow(self.pt_beta),
        )

    @torch.no_grad()
    def _is_solve_decision(
        self,
        delta_consensus: torch.Tensor,
        delta_agents: torch.Tensor,
        z_base: torch.Tensor,
        sigma: Optional[float] = None,
    ) -> torch.Tensor:
        """Prospect-Theory weighted importance-sampling update (paper Eqs. 5-10).

        Args:
            delta_consensus: mean T_cat-scaled decision gap across personas (scalar).
            delta_agents: per-persona decision gaps delta_i (shape [N]).
            z_base: base-model logits for [A, B] tokens (shape 1×2, T_cat-scaled).
            sigma: noise std to use; defaults to the adaptive inter-agent spread
                   computed upstream in _adaptive_noise_std.

        Math:
            g_{i,k} = |delta_base - delta_i| - |delta_tilde_k - delta_i|     (per-agent gain)
            g_cons_k = |delta_base - delta_bar| - |delta_tilde_k - delta_bar|  (consensus gain)
            U(eps_k) = (1 - lambda_coop) * mean_i v(g_{i,k})
                     +       lambda_coop  * v(g_cons_k)
                     -       alpha_ctl    * eps_k^2 / (2 sigma^2)
            w_k       = softmax(U / eta)
            delta_star= sum_k w_k * eps_k

        Key corrections vs. the v3 formulation:
            * Gain lives in logit-gap units (not logit-gap^2); no sigma^2 division
              and no explosion when personas agree.
            * v() is applied per-persona BEFORE averaging -> loss aversion is
              preserved (fixes the previous Jensen violation).
            * The alpha_ctl * eps^2/(2 sigma^2) term is identified honestly as a
              quadratic control cost (= -log p(eps)), NOT a per-sample log
              importance ratio. That claim was algebraically wrong.
        """
        sigma = sigma if sigma is not None else self.noise_std
        delta_base_scalar = (z_base[0, 1] - z_base[0, 0])  # scalar (T_cat-scaled)

        # Sample K perturbations from N(0, sigma^2)
        epsilon = torch.randn(self.K, device=self.device) * sigma
        delta_tilde = delta_consensus + epsilon                         # shape (K,)

        # Distance from base to each persona's target (scalars).
        # Shape: (N,)
        dist_base_to_i = (delta_base_scalar - delta_agents).abs()

        # Distance from every candidate delta_tilde_k to every persona i.
        # Shape: (K, N).  delta_tilde: (K,) -> (K,1); delta_agents: (N,) -> (1,N).
        dist_cand_to_i = (delta_tilde.unsqueeze(1) - delta_agents.unsqueeze(0)).abs()

        # Per-agent gain g_{i,k} = |base - delta_i| - |cand_k - delta_i|   shape (K, N)
        g_per_agent = dist_base_to_i.unsqueeze(0) - dist_cand_to_i

        # Apply v() PER AGENT then mean over personas -> preserves loss aversion.
        v_per_agent = self._prospect_value(g_per_agent)                 # (K, N)
        mean_v_individual = v_per_agent.mean(dim=1)                     # (K,)

        # Consensus-target gain:
        #   delta_bar = mean_i delta_i  (same as delta_consensus)
        #   g_cons_k = |base - delta_bar| - |cand_k - delta_bar|
        delta_bar = delta_consensus
        g_cons = (delta_base_scalar - delta_bar).abs() - (delta_tilde - delta_bar).abs()
        v_consensus = self._prospect_value(g_cons)                      # (K,)

        # Quadratic control cost = -log p(eps_k) up to constant.
        # Honest interpretation: trust-region penalty on |eps|. NOT log(p/q).
        control_cost = 0.5 * (epsilon ** 2) / (sigma ** 2 + 1e-8)        # (K,)

        # Collective utility, paper Eq. (7).
        U_total = (
            (1.0 - self.lambda_coop) * mean_v_individual
            + self.lambda_coop * v_consensus
            - self.alpha_ctl * control_cost  # paper Eq. (7): quadratic control cost
        )

        # Softmax-weighted importance sampling estimate of the optimal shift.
        weights = F.softmax(U_total / self.beta, dim=0)
        delta_star = torch.sum(weights * epsilon)
        return delta_star


    @torch.no_grad()
    def _swap_positional_labels(self, user_query: str, lang: str) -> Tuple[str, bool]:
        """Swap Option A/Option B (and Group A/B) labels; return (swapped_query, changed_flag)."""
        sf = SCENARIO_FRAME_I18N.get(lang, SCENARIO_FRAME_I18N["en"])
        left_label = sf["left_lane"]
        right_label = sf["right_lane"]
        ga = sf.get("group_a", "Group A")
        gb = sf.get("group_b", "Group B")

        changed = False
        _PH = "\x00SWAP_PLACEHOLDER\x00"

        q = user_query
        q2 = q.replace(left_label, _PH)
        q2 = q2.replace(right_label, left_label)
        q2 = q2.replace(_PH, right_label)
        if q2 != q:
            changed = True
        q = q2

        if ga != gb:
            q2 = q.replace(ga, _PH)
            q2 = q2.replace(gb, ga)
            q2 = q2.replace(_PH, gb)
            if q2 != q:
                changed = True
            q = q2

        # Fallback: swap "Option A"/"Option B" literally if i18n lane labels
        # were not present (e.g. caller passed a raw English prompt while lang
        # is set to a different locale).
        if not changed:
            q2 = q.replace("Option A", _PH)
            q2 = q2.replace("Option B", "Option A")
            q2 = q2.replace(_PH, "Option B")
            if q2 != q:
                changed = True
                q = q2

        return q, changed

    @torch.no_grad()
    def _predict_single_pass(
        self,
        user_query: str,
        preferred_on_right: bool,
        phenomenon_category: str,
        lang: str,
    ) -> Dict:
        """Single forward pass (no debiasing). Returns raw prediction dict."""
        logit_temp = self.category_logit_temperatures.get(
            phenomenon_category, self.logit_temperature
        )

        frame = PROMPT_FRAME_I18N.get(lang, PROMPT_FRAME_I18N["en"])
        user_content = frame.format(scenario=user_query)
        formatted = self.chat_helper.format_query_with_suffix(user_content)
        query_ids = self.tokenizer(formatted, return_tensors="pt",
                                   add_special_tokens=False).input_ids.to(self.device)

        z_base, z_agents = self._evaluate_all_agents(query_ids, lang, logit_temp=logit_temp)
        r_agents, variance, delta_consensus = self._compute_decision_rewards(z_base, z_agents)
        delta_agents = z_agents[:, 1] - z_agents[:, 0]  # per-persona logit gaps

        # Adaptive σ: scale noise to the empirical inter-agent logit spread so
        # the perturbation magnitude is commensurate with actual disagreement,
        # rather than fixed across all models and categories.
        sigma = self._adaptive_noise_std(z_agents)

        # IS update runs unconditionally; self-attenuates at consensus because
        # every g_{i,k} becomes a loss when all personas collapse to delta_base.
        delta_star = self._is_solve_decision(delta_consensus, delta_agents, z_base, sigma=sigma)
        delta_opt = delta_consensus + delta_star

        consensus_sign = (delta_consensus > 0).item()
        opt_sign = (delta_opt > 0).item() if hasattr(delta_opt, 'item') else (delta_opt > 0)
        mppi_flipped = consensus_sign != opt_sign  # diagnostic name kept for logs

        p_right = torch.sigmoid(delta_opt / self.decision_temperature).item()

        if preferred_on_right:
            p_spare_preferred = p_right
        else:
            p_spare_preferred = 1.0 - p_right

        return {
            "p_right": p_right,
            "p_left": 1.0 - p_right,
            "p_spare_preferred": p_spare_preferred,
            "variance": variance,
            "sigma_used": sigma,            # adaptive σ for this scenario
            "mppi_flipped": mppi_flipped,
            "delta_z_norm": abs(delta_star.item()),
            "delta_consensus": delta_consensus.item(),
            "delta_opt": delta_opt.item() if hasattr(delta_opt, 'item') else float(delta_opt),
            "logit_temp_used": logit_temp,
            "agent_decision_gaps": (z_agents[:, 1] - z_agents[:, 0]).tolist(),
            "agent_rewards": r_agents.tolist(),
            "z_base_a": z_base[0, 0].item(),
            "z_base_b": z_base[0, 1].item(),
        }

    @torch.no_grad()
    def predict(
        self,
        user_query: str,
        preferred_on_right: bool = True,
        phenomenon_category: str = "default",
        lang: str = "en",
    ) -> Dict:
        """
        Run SWA-PTIS prediction with positional debiasing.

        Runs TWO passes — original and A/B-swapped — to cancel out
        the model's intrinsic token bias toward option A or option B.
        """
        # Pass 1: original ordering
        r1 = self._predict_single_pass(
            user_query, preferred_on_right, phenomenon_category, lang
        )

        # Pass 2: swap Option A↔Option B in scenario text (robustly, with fallback).
        swapped_query, swap_changed = self._swap_positional_labels(user_query, lang)
        if not swap_changed:
            # If no swap happened, skip debias averaging to avoid false correction.
            result = r1.copy()
            result["positional_bias_warning"] = "swap_not_applied"
            result["p_spare_preferred_pass1"] = r1["p_spare_preferred"]
            result["p_spare_preferred_pass2"] = r1["p_spare_preferred"]
            result["positional_bias"] = 0.0
            return result

        r2 = self._predict_single_pass(
            swapped_query, not preferred_on_right, phenomenon_category, lang
        )

        # Debiasing at logit level (then apply sigmoid once).
        # Under additive positional bias b:
        #   delta_opt_1 = delta_true + b   (original ordering)
        #   delta_opt_2 = -delta_true + b  (A↔B swapped: signal negated, bias preserved)
        # => debiased_delta = (delta_opt_1 - delta_opt_2) / 2 = delta_true  (bias cancels)
        # Averaging probabilities after sigmoid gives a different result due to
        # sigmoid nonlinearity — logit-level averaging is the principled choice.
        debiased_delta = (r1["delta_opt"] - r2["delta_opt"]) / 2.0
        p_right_avg = torch.sigmoid(
            torch.tensor(debiased_delta / self.decision_temperature)
        ).item()

        if preferred_on_right:
            p_pref_avg = p_right_avg
        else:
            p_pref_avg = 1.0 - p_right_avg

        # Use pass-1 diagnostics but override with logit-debiased result.
        result = r1.copy()
        result["p_spare_preferred"] = p_pref_avg
        result["p_spare_preferred_pass1"] = r1["p_spare_preferred"]
        result["p_spare_preferred_pass2"] = r2["p_spare_preferred"]
        result["positional_bias"] = abs(r1["p_spare_preferred"] - r2["p_spare_preferred"])
        result["delta_opt_debiased"] = debiased_delta
        result["p_right"] = p_right_avg
        result["p_left"] = 1.0 - p_right_avg

        return result
