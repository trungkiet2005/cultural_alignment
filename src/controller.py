"""ImplicitSWAController: SWA-MPPI engine for cultural value negotiation."""

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
# CORE SWA-MPPI ENGINE (v3 — All Fixes Integrated)
# ============================================================================
class ImplicitSWAController:
    """
    Socially-Weighted Alignment (SWA) via Model Predictive Path Integral (MPPI)
    on the DECISION-FOCUSED logit space.

    Features:
      - Per-category logit temperature applied per predict() call
      - MPPI runs unconditionally on every scenario (no adaptive gating)
    """

    def __init__(
        self,
        model,
        tokenizer,
        personas: List[str],
        lambda_coop: float = 0.7,
        alpha_kl: float = 0.05,
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
        self.alpha_kl = alpha_kl
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
        max_len = max(s.shape[1] for s in seqs)

        batch_ids, batch_mask = [], []
        for s in seqs:
            pad_len = max_len - s.shape[1]
            batch_ids.append(F.pad(s, (pad_len, 0), value=self.pad_id))
            batch_mask.append(F.pad(
                torch.ones(1, s.shape[1], dtype=torch.long, device=self.device),
                (pad_len, 0), value=0,
            ))

        batch_ids = torch.cat(batch_ids, dim=0)
        batch_mask = torch.cat(batch_mask, dim=0)

        out = self.model(input_ids=batch_ids, attention_mask=batch_mask, use_cache=False)
        logits = out.logits[:, -1, :]

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
        variance = torch.var(delta_agents).item()
        return r_agents, variance, delta_consensus

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
    def _mppi_solve_decision(
        self,
        delta_consensus: torch.Tensor,
        r_agents: torch.Tensor,
        z_base: torch.Tensor,
    ) -> torch.Tensor:
        epsilon = torch.randn(self.K, device=self.device) * self.noise_std
        delta_pert = delta_consensus + epsilon
        kl_penalty = 0.5 * (epsilon ** 2) / (self.noise_std ** 2 + 1e-8)

        U_total = torch.zeros(self.K, device=self.device)
        for i in range(self.N):
            r_i = r_agents[i].item()
            r_others = (r_agents.sum() - r_agents[i]) / max(1, self.N - 1)
            u_private = self._prospect_value(r_i * delta_pert)
            u_social = self._prospect_value(r_others.item() * delta_pert)
            u_i = (1 - self.lambda_coop) * u_private + self.lambda_coop * u_social
            U_total += u_i
        U_total /= self.N
        U_total -= self.alpha_kl * kl_penalty

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

        # MPPI runs unconditionally (adaptive conflict gating removed).
        delta_star = self._mppi_solve_decision(delta_consensus, r_agents, z_base)
        delta_opt = delta_consensus + delta_star

        consensus_sign = (delta_consensus > 0).item()
        opt_sign = (delta_opt > 0).item() if hasattr(delta_opt, 'item') else (delta_opt > 0)
        mppi_flipped = consensus_sign != opt_sign

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
        Run SWA-MPPI prediction with positional debiasing.

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

        # Average the debiased p_spare_preferred
        p_pref_avg = (r1["p_spare_preferred"] + r2["p_spare_preferred"]) / 2.0

        # Use pass-1 diagnostics but override the debiased result
        result = r1.copy()
        result["p_spare_preferred"] = p_pref_avg
        result["p_spare_preferred_pass1"] = r1["p_spare_preferred"]
        result["p_spare_preferred_pass2"] = r2["p_spare_preferred"]
        result["positional_bias"] = abs(r1["p_spare_preferred"] - r2["p_spare_preferred"])
        # Recompute p_right/p_left from debiased p_spare_preferred
        if preferred_on_right:
            result["p_right"] = p_pref_avg
            result["p_left"] = 1.0 - p_pref_avg
        else:
            result["p_right"] = 1.0 - p_pref_avg
            result["p_left"] = p_pref_avg

        return result
