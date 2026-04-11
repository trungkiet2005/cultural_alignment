"""ImplicitSWAController: SWA-PTIS engine for cultural value negotiation.

SWA-PTIS = Socially-Weighted Alignment with Prospect-Theory Importance Sampling.

The algorithm applies a single-step, scalar, KL-regularised importance-sampling
update with a cooperative Prospect-Theory utility aggregated over N culturally
grounded persona agents. See `src/controller.py::_is_solve_decision` for the
math; it corresponds exactly to Eqs. (5)-(10) of the paper.

Math summary (matches ``_is_solve_decision``; gains are sigma-normalised inside PT):
  Per-agent gain (logit-gap units before / sigma in PT):
      g_{i,k} = |delta_base - delta_i| - |delta_tilde_k - delta_i|
  Consensus-target gain:
      g_cons_k = |delta_base - delta_bar| - |delta_tilde_k - delta_bar|
  Collective utility (mean-of-v, NOT v-of-mean — preserves loss aversion):
      U(eps_k) = (1 - lambda_coop) * mean_i v(g_{i,k} / sigma)
               +       lambda_coop  * v(g_cons_k / sigma)
  (No separate quadratic control cost: Gaussian proposal supplies KL-like breadth.)
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
from src.model import ChatTemplateHelper, encode_text_to_tensor, text_tokenizer


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
        alpha_ctl: Optional[float] = None,  # deprecated; kept for config compat
        rho_eff: float = 0.1,              # K_eff/K threshold for IS collapse guard
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
        country_iso: str = "UNKNOWN",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.personas = personas
        self.N = len(personas)
        self.assistant_lang = assistant_lang
        self.country_iso = country_iso
        self.lambda_coop = lambda_coop
        # alpha_ctl is retained for backward-compatible config loading but is
        # no longer used by the utility (see _is_solve_decision docstring).
        self.alpha_ctl = alpha_ctl
        if alpha_ctl is not None and alpha_ctl != 0.0:
            print("[SWA-PTIS] note: alpha_ctl is deprecated and has no effect; "
                  "the Gaussian proposal already regularises |epsilon|.")
        self.rho_eff = rho_eff
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

        tt = text_tokenizer(self.tokenizer)
        base_ids = tt.encode(formatted, add_special_tokens=False)
        a_full = tt.encode(formatted + "A", add_special_tokens=False)
        b_full = tt.encode(formatted + "B", add_special_tokens=False)

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
        a_str = tt.decode([a_id])
        b_str = tt.decode([b_id])
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
        setattr(self.tokenizer, "_moral_vllm_ab", (a_id, b_id))

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

    def _adaptive_noise_std(self, delta_agents: torch.Tensor) -> float:
        """Compute per-scenario proposal std from the empirical spread of persona
        logit gaps, with a conservative floor to guard against the small-sample
        noise of std(N=4).

        The sample-std of N=4 iid observations has relative sampling uncertainty
        ~ 1/sqrt(2(N-1)) ≈ 0.41, so a point estimate of sigma can drift
        substantially below the true inter-persona spread purely by chance.
        We therefore floor the proposal std at `self.noise_std` (the calibrated
        fallback) so the IS proposal always explores at least this much of the
        logit-gap space. This does NOT "fix" the estimator; it only bounds the
        worst-case volatility from below.
        """
        if delta_agents.numel() < 2:
            return self.noise_std
        std = float(delta_agents.std(unbiased=True).item())
        return max(std, self.noise_std)

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
        delta_base_scalar: torch.Tensor,
        delta_agents: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        """Prospect-Theory weighted importance-sampling update.

        Args:
            delta_base_scalar: scalar base-model decision gap (T_cat-scaled).
            delta_agents: per-persona decision gaps delta_i (shape [N]).
            sigma: proposal standard deviation (already floored upstream).

        Math (paper Eqs. 5-8):
            delta_bar = mean_i delta_i
            delta_tilde_k = delta_bar + eps_k,    eps_k ~ N(0, sigma^2)
            g_{i,k}   = |delta_base - delta_i| - |delta_tilde_k - delta_i|
            g_cons_k  = |delta_base - delta_bar| - |delta_tilde_k - delta_bar|
            # sigma-normalised gain (dimensionless; cancels T_cat scale):
            tilde{g}_{i,k}  = g_{i,k}  / sigma
            tilde{g}_cons_k = g_cons_k / sigma
            U(eps_k)  = (1 - lambda_coop) * mean_i v(tilde{g}_{i,k})
                      +       lambda_coop  * v(tilde{g}_cons_k)
            w_k       = softmax(U / eta)
            delta_star= sum_k w_k * eps_k

        Design invariants:
            * g_i is in logit-gap units; dividing by sigma (already floored
              at sigma_0 = self.noise_std) makes the PT input dimensionless
              AND cross-dimensionally comparable. Because sigma captures
              the local logit-temperature scale via the inter-persona
              spread, dividing by sigma cancels the T_cat scaling that
              would otherwise give low-T_cat dimensions more PT utility
              magnitude in the aggregate. Bounded: |tilde{g}_{i,k}| <=
              |eps_k|/sigma, which with eps_k ~ N(0, sigma^2) is O(1) and
              does NOT explode (sigma is floored at sigma_0 > 0).
            * v() is applied per persona BEFORE averaging, so Jensen's
              inequality preserves the kappa=2.25 loss-aversion asymmetry
              at the social level.
            * Self-attenuation at consensus: if all delta_i -> delta_base
              then g_{i,k} -> -|eps_k| and tilde{g}_{i,k} -> -|eps_k|/sigma
              is O(1) and <= 0 for every i; every candidate incurs a loss
              and the softmax concentrates on |eps_k| ~ 0.
            * NO quadratic control cost term (see controller module
              docstring / paper Appendix C for the derivation).
        """
        # Sample K perturbations from N(0, sigma^2)
        epsilon = torch.randn(self.K, device=self.device) * sigma
        delta_bar = delta_agents.mean()
        delta_tilde = delta_bar + epsilon                                # (K,)

        # Per-agent gain in raw logit-gap units.
        dist_base_to_i = (delta_base_scalar - delta_agents).abs()        # (N,)
        dist_cand_to_i = (delta_tilde.unsqueeze(1)
                          - delta_agents.unsqueeze(0)).abs()             # (K, N)
        g_per_agent = dist_base_to_i.unsqueeze(0) - dist_cand_to_i       # (K, N)

        # Dimensionality fix: divide the PT input by sigma (already floored
        # at sigma_0) so the utility is dimensionless and cross-dimensionally
        # comparable. sigma_0 > 0 prevents division-by-zero.
        g_per_agent_tilde = g_per_agent / sigma

        # Apply v() PER AGENT on the sigma-normalised gain, then mean.
        v_per_agent = self._prospect_value(g_per_agent_tilde)            # (K, N)
        mean_v_individual = v_per_agent.mean(dim=1)                      # (K,)

        # Consensus-target PT utility, same sigma normalisation.
        g_cons = (delta_base_scalar - delta_bar).abs() \
                 - (delta_tilde - delta_bar).abs()                       # (K,)
        v_consensus = self._prospect_value(g_cons / sigma)               # (K,)

        # Collective utility (paper Eq. 7; no control-cost term).
        U_total = ((1.0 - self.lambda_coop) * mean_v_individual
                   + self.lambda_coop * v_consensus)

        weights = F.softmax(U_total / self.beta, dim=0)

        # Effective sample size guard: when the importance weights concentrate
        # on very few samples, the IS estimator's own variance explodes and its
        # point estimate cannot be trusted. In that case we return delta_star=0
        # (fall back to the plain consensus) -- a conservative no-op, not a
        # learned correction.
        k_eff = 1.0 / torch.sum(weights * weights).clamp_min(1e-12)
        if float(k_eff.item()) / float(self.K) < self.rho_eff:
            return torch.zeros((), device=self.device)

        return torch.sum(weights * epsilon)


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
    def _extract_logit_gaps(
        self,
        user_query: str,
        phenomenon_category: str,
        lang: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Run a single batched forward pass and return the per-persona logit
        GAPS (scalar deltas) along with the base decision gap and logit temp.

        Returns:
            delta_base_scalar: scalar torch.Tensor (T_cat-scaled).
            delta_agents:      shape (N,) torch.Tensor (T_cat-scaled).
            logit_temp:        float, the T_cat used for scaling.
        """
        logit_temp = self.category_logit_temperatures.get(
            phenomenon_category, self.logit_temperature
        )
        frame = PROMPT_FRAME_I18N.get(lang, PROMPT_FRAME_I18N["en"])
        user_content = frame.format(scenario=user_query)
        formatted = self.chat_helper.format_query_with_suffix(user_content)
        query_ids = encode_text_to_tensor(
            self.tokenizer, formatted, self.device, add_special_tokens=False
        )

        z_base, z_agents = self._evaluate_all_agents(
            query_ids, lang, logit_temp=logit_temp)
        delta_base_scalar = z_base[0, 1] - z_base[0, 0]          # scalar
        delta_agents = z_agents[:, 1] - z_agents[:, 0]           # (N,)
        return delta_base_scalar, delta_agents, logit_temp

    @torch.no_grad()
    def predict(
        self,
        user_query: str,
        preferred_on_right: bool = True,
        phenomenon_category: str = "default",
        lang: str = "en",
    ) -> Dict:
        """Run SWA-PTIS prediction with positional debiasing.

        Order of operations (this is the whole point of the rewrite):
            1. Extract logit gaps (delta_base, delta_i) from the ORIGINAL prompt.
            2. Extract logit gaps (delta_base', delta_i') from the A<->B SWAPPED prompt.
            3. Combine linearly to get bias-debiased logit gaps:
                   delta_base_deb = (delta_base - delta_base') / 2
                   delta_i_deb    = (delta_i - delta_i') / 2
               Under an additive positional bias b on the logit gap, the two
               passes give (delta_true + b) and (-delta_true + b), so the
               linear combination above yields delta_true exactly.
            4. Run the nonlinear PT-IS update ONCE on the debiased gaps.

        This ordering matters. The previous version ran the full PT-IS pipeline
        independently on each pass and then averaged the two delta_opt values;
        because PT-IS is nonlinear (v() has a kink at zero with kappa=2.25 loss
        aversion), the bias b can map to very different regions of v() on the
        two passes, so averaging the two PT-IS outputs does NOT cancel b.
        Debiasing at the logit level (BEFORE any nonlinearity) preserves the
        linear-cancellation guarantee that the additive-bias model provides.
        """
        # -- Pass 1: original ordering --
        delta_base_1, delta_agents_1, logit_temp = self._extract_logit_gaps(
            user_query, phenomenon_category, lang)

        # -- Pass 2: A<->B swapped ordering --
        swapped_query, swap_changed = self._swap_positional_labels(
            user_query, lang)

        if not swap_changed:
            # No swap was actually applied (e.g. literal label absent in the
            # native-language frame). Fall back to single-pass inference and
            # warn; no debiasing is possible.
            sigma = self._adaptive_noise_std(delta_agents_1)
            delta_consensus = delta_agents_1.mean()
            delta_star = self._is_solve_decision(
                delta_base_1, delta_agents_1, sigma=sigma)
            delta_opt = delta_consensus + delta_star
            p_right = torch.sigmoid(delta_opt / self.decision_temperature).item()
            p_pref = p_right if preferred_on_right else 1.0 - p_right
            return {
                "p_right": p_right,
                "p_left": 1.0 - p_right,
                "p_spare_preferred": p_pref,
                "variance": float(delta_agents_1.var(unbiased=True).item()),
                "sigma_used": sigma,
                "mppi_flipped": (delta_consensus > 0).item() != (delta_opt > 0).item(),
                "delta_z_norm": abs(delta_star.item()),
                "delta_consensus": delta_consensus.item(),
                "delta_opt": float(delta_opt.item()),
                "logit_temp_used": logit_temp,
                "agent_decision_gaps": delta_agents_1.tolist(),
                "agent_rewards": (delta_agents_1 - delta_base_1).tolist(),
                "positional_bias_warning": "swap_not_applied",
                "p_spare_preferred_pass1": p_pref,
                "p_spare_preferred_pass2": p_pref,
                "positional_bias": 0.0,
            }

        delta_base_2, delta_agents_2, _ = self._extract_logit_gaps(
            swapped_query, phenomenon_category, lang)

        # -- Linear debiasing at the logit-gap level --
        # Under additive bias b on the logit gap:
        #     pass 1 gap   =  true_gap + b
        #     pass 2 gap   = -true_gap + b        (A and B swapped: signal flips)
        #  => (pass1 - pass2) / 2 = true_gap  exactly (b cancels linearly).
        delta_base_deb = (delta_base_1 - delta_base_2) / 2.0               # scalar
        delta_agents_deb = (delta_agents_1 - delta_agents_2) / 2.0         # (N,)

        # Single PT-IS update on the debiased gaps.
        sigma = self._adaptive_noise_std(delta_agents_deb)
        delta_consensus = delta_agents_deb.mean()
        delta_star = self._is_solve_decision(
            delta_base_deb, delta_agents_deb, sigma=sigma)
        delta_opt = delta_consensus + delta_star

        p_right = torch.sigmoid(delta_opt / self.decision_temperature).item()
        p_pref = p_right if preferred_on_right else 1.0 - p_right

        # Diagnostics: also compute what each single-pass gap would have said,
        # purely for logging / comparison with the old pipeline.
        p_right_1 = torch.sigmoid(
            delta_agents_1.mean() / self.decision_temperature).item()
        p_right_2 = torch.sigmoid(
            delta_agents_2.mean() / self.decision_temperature).item()
        p_pref_1 = p_right_1 if preferred_on_right else 1.0 - p_right_1
        p_pref_2 = (1.0 - p_right_2) if preferred_on_right else p_right_2

        consensus_sign = (delta_consensus > 0).item()
        opt_sign = (delta_opt > 0).item()

        return {
            "p_right": p_right,
            "p_left": 1.0 - p_right,
            "p_spare_preferred": p_pref,
            "variance": float(delta_agents_deb.var(unbiased=True).item()),
            "sigma_used": sigma,
            "mppi_flipped": consensus_sign != opt_sign,  # diagnostic name
            "delta_z_norm": abs(delta_star.item()),
            "delta_consensus": delta_consensus.item(),
            "delta_opt": float(delta_opt.item()),
            "delta_opt_debiased": float(delta_opt.item()),
            "logit_temp_used": logit_temp,
            "agent_decision_gaps": delta_agents_deb.tolist(),
            "agent_decision_gaps_pass1": delta_agents_1.tolist(),
            "agent_decision_gaps_pass2": delta_agents_2.tolist(),
            "agent_rewards": (delta_agents_deb - delta_base_deb).tolist(),
            "p_spare_preferred_pass1": p_pref_1,
            "p_spare_preferred_pass2": p_pref_2,
            "positional_bias": abs(p_pref_1 - p_pref_2),
        }
