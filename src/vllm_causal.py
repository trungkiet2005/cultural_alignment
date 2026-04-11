"""
vLLM-backed causal LM wrapper for EXP-24 token-logit scoring.

The moral pipeline needs next-token scores at the last prompt position for the
**A/B answer token ids** (per language). vLLM caps ``logprobs`` (often max 20);
``logprobs=-1`` is treated as “full vocab” and fails validation on recent engines.

We therefore use ``SamplingParams(allowed_token_ids=[a_id, b_id], logprobs=2)`` and
scatter returned log-probs into a dense **[B, vocab_size]** row (rest ``-1e9``).
Call ``set_decision_tokens(a_id, b_id)`` before ``forward`` (see ``baseline_runner`` /
``ImplicitSWAController``).
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import torch
import torch.nn as nn

from src.model import text_tokenizer

if TYPE_CHECKING:
    pass


def _logprob_map_to_row(
    logprob_map: Any,
    vocab_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Scatter vLLM logprob dict into a dense 1D logits row."""
    row = torch.full((vocab_size,), -1e9, device=device, dtype=dtype)
    if not logprob_map:
        return row
    for tid, lp_obj in logprob_map.items():
        if isinstance(lp_obj, (int, float)):
            v = float(lp_obj)
        else:
            v = getattr(lp_obj, "logprob", None)
            if v is None:
                v = float(lp_obj)
        row[int(tid)] = float(v)
    return row


class VllmCausalWrapper(nn.Module):
    """Wraps ``vllm.LLM`` so ``forward(input_ids)`` returns last-position logits [B, V]."""

    def __init__(self, llm: Any, *, pad_token_id: int, vocab_size: int):
        super().__init__()
        self.llm = llm
        self.pad_token_id = int(pad_token_id)
        self._vocab_size = int(vocab_size)
        self._decision_ab: Optional[Tuple[int, int]] = None
        _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.register_parameter("_device_probe", nn.Parameter(torch.zeros(1, device=_dev)))

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def set_decision_tokens(self, a_id: int, b_id: int) -> None:
        """Restrict next-token scoring to the language-specific A/B ids (required for vLLM)."""
        self._decision_ab = (int(a_id), int(b_id))

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False) -> SimpleNamespace:
        del use_cache
        if input_ids.dim() != 2:
            raise ValueError(f"vLLM wrapper expects input_ids [B, L], got {input_ids.shape}")

        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        if self._decision_ab is None:
            raise RuntimeError(
                "VllmCausalWrapper: call model.set_decision_tokens(a_id, b_id) before forward. "
                "Baseline/controller set this from per-language A/B token resolution."
            )
        a_id, b_id = self._decision_ab
        allowed = sorted({a_id, b_id})
        n_lp = len(allowed)

        device = input_ids.device
        dtype = torch.float32
        pad = self.pad_token_id
        B, _L = input_ids.shape

        prompts: List = []
        for i in range(B):
            row = input_ids[i]
            nz = (row != pad).nonzero(as_tuple=True)[0]
            if nz.numel() == 0:
                seq: List[int] = []
            else:
                last = int(nz[-1].item()) + 1
                seq = row[:last].tolist()
            prompts.append(TokensPrompt(prompt_token_ids=seq))

        try:
            sp = SamplingParams(
                max_tokens=1,
                temperature=0.0,
                logprobs=n_lp,
                allowed_token_ids=allowed,
            )
        except TypeError as e:
            raise RuntimeError(
                "vLLM SamplingParams rejected arguments; need a vLLM build with "
                "`allowed_token_ids` for A/B logit scoring (or upgrade vLLM)."
            ) from e
        outputs = self.llm.generate(prompts, sampling_params=sp, use_tqdm=False)

        rows: List[torch.Tensor] = []
        for i in range(B):
            ro = outputs[i]
            if not ro.outputs:
                rows.append(
                    torch.full((self._vocab_size,), -1e9, device=device, dtype=dtype)
                )
                continue
            comp = ro.outputs[0]
            lp_seq = comp.logprobs
            if not lp_seq or lp_seq[0] is None:
                rows.append(
                    torch.full((self._vocab_size,), -1e9, device=device, dtype=dtype)
                )
                continue
            rows.append(_logprob_map_to_row(lp_seq[0], self._vocab_size, device, dtype))

        logits = torch.stack(rows, dim=0)
        return SimpleNamespace(logits=logits)


def load_model_vllm(
    model_name: str,
    max_seq_length: int = 2048,
    load_in_4bit: bool = False,
):
    """
    Load a model with vLLM ``LLM`` and return *(wrapper, tokenizer)*.

    ``load_in_4bit``: models whose id contains a **≥70B** tag get vLLM
    ``quantization`` (default ``bitsandbytes`` via ``MORAL_VLLM_LARGE_QUANT``);
    smaller models still use ``bitsandbytes`` when this flag is True (legacy).
    Set ``VLLM_QUANTIZATION`` to override; ``MORAL_VLLM_AUTO_QUANT_70B=0`` skips
    the large-model auto path only.
    """
    from src.vllm_env import apply_vllm_quantization_kw, apply_vllm_runtime_defaults

    apply_vllm_runtime_defaults()

    try:
        from vllm import LLM
    except ImportError as e:
        raise RuntimeError(
            "vLLM is not installed. Install a CUDA build matching your PyTorch/CUDA, "
            "e.g. `pip install vllm` — see https://docs.vllm.ai/en/latest/getting_started/installation.html"
        ) from e

    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

    _eager_raw = os.environ.get("MORAL_VLLM_ENFORCE_EAGER") or os.environ.get(
        "VLLM_ENFORCE_EAGER", "1"
    )
    eager = _eager_raw.strip().lower() not in ("0", "false", "no")
    _gpu = (
        os.environ.get("MORAL_VLLM_GPU_MEM")
        or os.environ.get("VLLM_GPU_MEMORY_UTILIZATION")
        or "0.90"
    )
    kw: dict = {
        "model": model_name,
        "trust_remote_code": True,
        "max_model_len": int(max_seq_length),
        "dtype": "bfloat16",
        "gpu_memory_utilization": float(_gpu),
        "enforce_eager": eager,
        "tensor_parallel_size": int(os.environ.get("VLLM_TP", "1")),
    }
    if os.path.isdir("/kaggle/working"):
        if os.environ.get("VLLM_DISABLE_CUSTOM_ALL_REDUCE", "1").strip().lower() not in (
            "0",
            "false",
            "no",
        ):
            kw["disable_custom_all_reduce"] = True
    apply_vllm_quantization_kw(
        kw,
        hf_id=model_name,
        model_key=model_name,
        load_in_4bit=load_in_4bit,
        legacy_causal_always_bnb=True,
    )

    mi = os.environ.get("VLLM_MAX_NUM_SEQS")
    if mi:
        kw["max_num_seqs"] = int(mi)

    print(f"[MODEL] Loading {model_name} via vLLM LLM(...)...")
    print(
        "[MODEL] vLLM engine + weights (often 2–15+ min first run) — timing from LLM() start…"
    )
    _t_llm = time.perf_counter()
    llm = LLM(**kw)
    print(f"[MODEL] vLLM LLM() done in {time.perf_counter() - _t_llm:.1f}s")

    _t_tok = time.perf_counter()
    tok = llm.get_tokenizer()
    _tt = text_tokenizer(tok)
    if getattr(_tt, "pad_token", None) is None and getattr(_tt, "eos_token", None) is not None:
        _tt.pad_token = _tt.eos_token
        _tt.pad_token_id = _tt.eos_token_id
    try:
        tok.padding_side = "left"
    except Exception:
        pass
    try:
        _tt.padding_side = "left"
    except Exception:
        pass

    setattr(tok, "_moral_chat_content_mode", "string")
    print(f"[MODEL] vLLM tokenizer bridge ready in {time.perf_counter() - _t_tok:.1f}s")

    pad_id = int(getattr(_tt, "pad_token_id", 0) or 0)
    vocab_size = int(llm.llm_engine.model_config.get_vocab_size())

    model = VllmCausalWrapper(llm, pad_token_id=pad_id, vocab_size=vocab_size)

    print("[MODEL] vLLM wrapper ready (set_decision_tokens + generate w/ A/B logprobs).")
    return model, tok
