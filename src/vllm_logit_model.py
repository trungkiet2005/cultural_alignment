"""vLLM shim: same call pattern as HF causal LM for token-logit A/B scoring.

Before each ``__call__``, set ``tokenizer._moral_vllm_ab = (a_token_id, b_token_id)``
(controller / baseline_runner). vLLM then returns restricted next-token logprobs for
those two ids; we embed them as pseudo-logits so existing code paths unchanged.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch


class _ModelForwardOutput:
    __slots__ = ("logits",)

    def __init__(self, logits: torch.Tensor):
        self.logits = logits


def _token_logprob(logprob_dict: Any, tid: int) -> float:
    tid = int(tid)
    if logprob_dict is None or tid not in logprob_dict:
        return float("-inf")
    v = logprob_dict[tid]
    if hasattr(v, "logprob"):
        return float(v.logprob)
    if isinstance(v, (int, float)):
        return float(v)
    return float("-inf")


def _row_lengths(input_ids: torch.Tensor, pad_id: Optional[int]) -> Tuple[torch.Tensor, List[int]]:
    b, ell = input_ids.shape
    if pad_id is None:
        lens = [ell] * b
        return torch.tensor(lens, dtype=torch.long, device=input_ids.device), lens
    row = input_ids.cpu()
    lens: List[int] = []
    for i in range(b):
        r = row[i]
        m = r != pad_id
        if bool(m.any()):
            idx = int(m.nonzero()[-1].item()) + 1
        else:
            idx = int(ell)
        lens.append(max(idx, 1))
    return torch.tensor(lens, dtype=torch.long, device=input_ids.device), lens


class VllmCausalLogitModel:
    """Minimal interface: ``parameters()``, ``eval()``, ``__call__(input_ids, use_cache=...)``."""

    def __init__(
        self,
        llm: Any,
        *,
        pad_token_id: Optional[int],
        vocab_size: int,
    ):
        self._llm = llm
        self.pad_token_id = pad_token_id
        self.vocab_size = int(vocab_size)
        self._tokenizer_ref: Any = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def parameters(self):
        class _P:
            def __init__(self, d: torch.device):
                self.device = d

        yield _P(self._device)

    def eval(self):
        return self

    def train(self, mode: bool = True):
        return self

    def __call__(self, input_ids: torch.Tensor, use_cache: bool = False):
        del use_cache
        tok = self._tokenizer_ref
        if tok is None:
            raise RuntimeError("VllmCausalLogitModel: tokenizer reference not wired")
        pair = getattr(tok, "_moral_vllm_ab", None)
        if pair is None or len(pair) != 2:
            raise RuntimeError(
                "vLLM backend: set tokenizer._moral_vllm_ab = (a_id, b_id) before model forward."
            )
        a_id, b_id = int(pair[0]), int(pair[1])

        from vllm import SamplingParams

        lens_t, lens_list = _row_lengths(input_ids, self.pad_token_id)
        bsz, max_len = input_ids.shape
        dev = input_ids.device
        rows_cpu = input_ids.detach().cpu()
        prompts = []
        for i in range(bsz):
            li = lens_list[i]
            prompts.append({"prompt_token_ids": rows_cpu[i, :li].tolist()})

        sp_kw = dict(
            max_tokens=1,
            temperature=1.0,
            logprobs=2,
            seed=0,
        )
        try:
            sp = SamplingParams(**sp_kw, allowed_token_ids=[a_id, b_id])
        except TypeError:
            sp = SamplingParams(**sp_kw)

        try:
            outs = self._llm.generate(prompts, sampling_params=sp, use_tqdm=False)
        except TypeError:
            outs = self._llm.generate(prompts, sampling_params=sp)

        logits_full = torch.zeros(
            bsz, max_len, self.vocab_size, device=dev, dtype=torch.float32
        )
        pos = lens_t - 1

        for i in range(bsz):
            o = outs[i]
            if not o.outputs:
                continue
            lo = o.outputs[0].logprobs
            if not lo or lo[0] is None:
                continue
            d0 = lo[0]
            la = _token_logprob(d0, a_id)
            lb = _token_logprob(d0, b_id)
            if la == float("-inf") and lb == float("-inf"):
                continue
            gap = lb - la
            pi = int(pos[i].item())
            logits_full[i, pi, a_id] = 0.0
            logits_full[i, pi, b_id] = float(gap)

        return _ModelForwardOutput(logits_full)
