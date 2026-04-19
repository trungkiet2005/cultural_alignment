"""Activation-steering baseline (Arditi et al., NeurIPS 2024 style).

Computes a per-country *cultural* steering vector as the difference of mean
residual-stream activations on culturally-contrastive prompt pairs, then
applies it at inference via a forward hook:

        h_l(x)  ←  h_l(x) + α · v_steer(country)            [layer l fixed]

This is intentionally a *minimal*, paper-faithful implementation:
contrastive pairs are derived from the same WVS profile descriptors that
SWA-DPBR uses for its persona panel, so the activation baseline gets the
same cultural information SWA-DPBR does — just delivered as a
constant residual-stream offset rather than a per-scenario logit
correction.

Hard requirements:
  • HuggingFace Transformers backend (``MORAL_MODEL_BACKEND=hf_native``).
    vLLM and Unsloth do not expose Python forward hooks on residual
    streams in a stable way, so we error early if the backend is wrong.
  • The model must expose ``model.model.layers[l]`` (Llama / Phi / Qwen /
    Gemma family). Falls back to ``model.transformer.h[l]`` for GPT-OSS
    / Mistral-style attribute names.

Outputs are A/B logit-gap style ``p_spare_preferred`` records identical
in shape to the vanilla baseline, so existing AMCE / MIS aggregators
slot in unchanged.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch

from src.amce import (
    compute_alignment_metrics,
    compute_amce_from_preferences,
    load_human_amce,
)
from src.baseline_runner import (
    BASE_ASSISTANT_I18N,
    PROMPT_FRAME_I18N,
    logit_fallback_p_spare,
    resolve_decision_tokens_for_lang,
)
from src.constants import COUNTRY_FULL_NAMES, COUNTRY_LANG
from src.model import ChatTemplateHelper
from src.prompt_baselines import _quartile_descriptor, load_wvs_vector

# Default contrastive-pair templates.  We construct N pairs per country:
# (positive, negative) where positive matches the country's WVS pole and
# negative is its opposite, on the same dimension.
_PAIR_TEMPLATES = [
    ("religiosity",
     "I am a person from {country} who is {pole}.",
     {"high": "deeply religious", "low": "highly secular"}),
    ("gender_equality",
     "I am a person from {country} who is {pole}.",
     {"high": "strongly egalitarian on gender",
      "low":  "strongly traditional on gender"}),
    ("tolerance_diversity",
     "I am a person from {country} who is {pole}.",
     {"high": "very tolerant of social diversity",
      "low":  "intolerant of social diversity"}),
    ("social_trust",
     "I am a person from {country} who has {pole}.",
     {"high": "very high social trust", "low": "very low social trust"}),
    ("national_pride",
     "I am a person from {country} who is {pole}.",
     {"high": "very proud of my nation", "low": "not proud of my nation"}),
]


def _resolve_layer(model, layer_idx: int):
    """Return the residual-stream module at ``layer_idx``.

    Tries the common layouts: ``model.model.layers[i]`` (Llama / Phi /
    Qwen / Gemma) then ``model.transformer.h[i]`` (Mistral / GPT-style).
    """
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return inner.layers[layer_idx]
    tr = getattr(model, "transformer", None)
    if tr is not None and hasattr(tr, "h"):
        return tr.h[layer_idx]
    raise RuntimeError(
        "activation_steering: cannot locate transformer layers "
        "(tried model.model.layers and model.transformer.h)."
    )


def _build_contrastive_pairs(country: str,
                             wvs_vec: Optional[Dict[str, float]] = None
                             ) -> List[Tuple[str, str]]:
    """Build (positive, negative) prompt pairs using the country's WVS pole."""
    name = COUNTRY_FULL_NAMES.get(country, country)
    pairs: List[Tuple[str, str]] = []
    for feat, tpl, poles in _PAIR_TEMPLATES:
        if wvs_vec and feat in wvs_vec and wvs_vec[feat] == wvs_vec[feat]:
            high_first = wvs_vec[feat] >= 0.5
        else:
            high_first = True  # neutral default
        pos_pole = poles["high"] if high_first else poles["low"]
        neg_pole = poles["low"]  if high_first else poles["high"]
        pairs.append(
            (tpl.format(country=name, pole=pos_pole),
             tpl.format(country=name, pole=neg_pole))
        )
    return pairs


@torch.no_grad()
def _mean_residual(model, tokenizer, chat_helper, prompts: List[str],
                   layer_idx: int) -> torch.Tensor:
    """Capture the mean last-token residual at ``layer_idx`` over ``prompts``."""
    device = next(model.parameters()).device
    layer = _resolve_layer(model, layer_idx)
    captured: List[torch.Tensor] = []

    def _hook(_module, _inputs, output):
        # Layer modules return either a tensor or a tuple (tensor, ...). We
        # take the first element either way.
        h = output[0] if isinstance(output, tuple) else output
        captured.append(h[:, -1, :].detach().float().cpu())

    handle = layer.register_forward_hook(_hook)
    try:
        for p in prompts:
            ids = chat_helper.encode_query_suffix(p, device)
            model(input_ids=ids, use_cache=False)
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("activation_steering: no residuals captured.")
    return torch.cat(captured, dim=0).mean(dim=0)


def compute_steering_vector(model, tokenizer, country: str,
                            layer_idx: int = 32,
                            wvs_csv_path: Optional[str] = None
                            ) -> torch.Tensor:
    """v_steer = mean(positive residuals) − mean(negative residuals)."""
    chat_helper = ChatTemplateHelper(tokenizer)
    wvs_vec = load_wvs_vector(wvs_csv_path, country) if wvs_csv_path else None
    pairs = _build_contrastive_pairs(country, wvs_vec=wvs_vec)
    pos_prompts = [p for p, _ in pairs]
    neg_prompts = [n for _, n in pairs]
    pos_mean = _mean_residual(model, tokenizer, chat_helper, pos_prompts, layer_idx)
    neg_mean = _mean_residual(model, tokenizer, chat_helper, neg_prompts, layer_idx)
    v = pos_mean - neg_mean
    # Normalise so α has comparable magnitude across models / countries.
    v = v / (v.norm() + 1e-8)
    return v


class _SteeringHook:
    """Adds α · v_steer to the residual at ``layer_idx`` on every forward pass.

    Use as a context manager; remove() is called automatically on exit.
    """

    def __init__(self, model, layer_idx: int, v_steer: torch.Tensor, alpha: float):
        self.layer = _resolve_layer(model, layer_idx)
        self.alpha = float(alpha)
        # Place v on the model device + match dtype lazily inside the hook so
        # we don't lock the dtype at construction time.
        self._v_cpu = v_steer.detach().float().cpu()
        self._v_dev: Optional[torch.Tensor] = None
        self._handle = None

    def __enter__(self):
        def _hook(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if self._v_dev is None or self._v_dev.device != h.device:
                self._v_dev = self._v_cpu.to(h.device, dtype=h.dtype)
            h2 = h + self.alpha * self._v_dev[None, None, :]
            return (h2, *output[1:]) if isinstance(output, tuple) else h2

        self._handle = self.layer.register_forward_hook(_hook)
        return self

    def __exit__(self, *_):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def run_activation_steering_country(
    model, tokenizer, scenario_df: pd.DataFrame, country: str, cfg,
    layer_idx: int = 32, alpha: float = 1.5,
    wvs_csv_path: Optional[str] = None,
    human_amce_path: Optional[str] = None,
) -> Dict:
    """Run the activation-steering baseline for one country.

    Hard requirement: the model must support residual-stream forward hooks
    (HuggingFace Transformers backend; not vLLM / Unsloth-quantised).
    """
    import os
    if os.environ.get("MORAL_MODEL_BACKEND", "").strip().lower() not in ("", "hf_native"):
        raise RuntimeError(
            "activation_steering requires MORAL_MODEL_BACKEND=hf_native "
            "(vLLM / Unsloth do not expose Python forward hooks)."
        )

    device = next(model.parameters()).device
    lang = COUNTRY_LANG.get(country, "en")
    chat_helper = ChatTemplateHelper(tokenizer)
    base_text = BASE_ASSISTANT_I18N.get(lang, BASE_ASSISTANT_I18N["en"])
    base_ids = chat_helper.build_prefix_ids(base_text, device)
    a_id, b_id = resolve_decision_tokens_for_lang(tokenizer, chat_helper, lang)
    if hasattr(model, "set_decision_tokens"):
        model.set_decision_tokens(int(a_id), int(b_id))
    setattr(tokenizer, "_moral_vllm_ab", (int(a_id), int(b_id)))

    print(f"  [steer] {country}: building steering vector at layer {layer_idx} …")
    v_steer = compute_steering_vector(model, tokenizer, country,
                                      layer_idx=layer_idx, wvs_csv_path=wvs_csv_path)

    decision_temp = float(getattr(cfg, "decision_temperature", 0.5))
    frame = PROMPT_FRAME_I18N.get(lang, PROMPT_FRAME_I18N["en"])

    records = []
    t0 = time.time()
    with _SteeringHook(model, layer_idx, v_steer, alpha):
        for _, row in scenario_df.iterrows():
            prompt = row.get("Prompt", row.get("prompt", ""))
            if not prompt:
                continue
            pref_right = int(row.get("preferred_on_right", 1))
            user_content = frame.format(scenario=prompt)
            try:
                query_ids = chat_helper.encode_query_suffix(user_content, device)
                full_ids = torch.cat([base_ids, query_ids], dim=1)
                # logit_fallback_p_spare uses positional pref_right (bool).
                p_spare = float(logit_fallback_p_spare(
                    model, full_ids, a_id, b_id, bool(pref_right),
                    temperature=decision_temp,
                ))
            except Exception as exc:
                print(f"  [warn] {country} row {row.name}: {exc}")
                p_spare = 0.5
            rec = {
                "country":             country,
                "method":              "activation_steering",
                "layer":               layer_idx,
                "alpha":               alpha,
                "phenomenon_category": row.get("phenomenon_category", ""),
                "preferred_on_right":  pref_right,
                "p_spare_preferred":   p_spare,
            }
            for col in ("n_left", "n_right"):
                if col in row.index:
                    rec[col] = row[col]
            records.append(rec)
    elapsed = time.time() - t0
    res_df = pd.DataFrame(records)

    summary = {
        "method":      "activation_steering",
        "country":     country,
        "layer":       layer_idx,
        "alpha":       alpha,
        "n_scenarios": len(records),
        "elapsed_sec": elapsed,
    }
    if human_amce_path:
        try:
            model_amce = compute_amce_from_preferences(res_df)
            human_amce = load_human_amce(human_amce_path, country)
            alignment = compute_alignment_metrics(model_amce, human_amce)
            summary["mis"]       = alignment.get("mis", float("nan"))
            summary["jsd"]       = alignment.get("jsd", float("nan"))
            summary["pearson_r"] = alignment.get("pearson_r", float("nan"))
        except Exception as exc:
            print(f"  [warn] alignment metrics failed for {country}: {exc}")
    return {"summary": summary, "results_df": res_df}
