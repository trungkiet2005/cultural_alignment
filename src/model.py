"""Model loading, tokenization helpers, and seed setup."""

import random as _rng

import numpy as np
import torch

try:
    import unsloth  # noqa: F401  — must be imported before transformers
except Exception:
    pass


# ── seed ────────────────────────────────────────────────────────────────────


def setup_seeds(seed: int = 42) -> None:
    _rng.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── model loading ───────────────────────────────────────────────────────────


def load_model(
    model_name: str,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
):
    """Load a model via Unsloth and return *(model, tokenizer)*."""
    from unsloth import FastLanguageModel
    import transformers
    transformers.logging.set_verbosity_error()

    print(f"[MODEL] Loading {model_name} via Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    print(f"[MODEL] Loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model, tokenizer


# ── chat-template helper ────────────────────────────────────────────────────


class ChatTemplateHelper:
    """
    Builds tokenised chat prefixes and query suffixes using the tokenizer's
    built-in chat_template, so the same code works for Llama, Qwen, Gemma,
    Mistral, Command-R, and any other HuggingFace model.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Some chat templates (e.g. Gemma) don't accept a "system" role.
        # Detect once and fold the system prompt into the first user turn.
        self.supports_system = self._probe_system_role()

    def _probe_system_role(self) -> bool:
        try:
            self.tokenizer.apply_chat_template(
                [{"role": "system", "content": "x"},
                 {"role": "user", "content": "y"}],
                tokenize=False, add_generation_prompt=False,
            )
            return True
        except Exception:
            return False

    def _messages(self, system_prompt: str, user_content: str):
        if self.supports_system:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ]
        merged = f"{system_prompt}\n\n{user_content}" if user_content else system_prompt
        return [{"role": "user", "content": merged}]

    def build_prefix_ids(self, system_prompt: str, device) -> torch.Tensor:
        """
        Tokenise [system + empty user turn] so we can later concatenate
        the actual user query.  Returns shape (1, seq_len).
        """
        messages = self._messages(system_prompt, "")
        full = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        # Find where the empty user content sits and keep everything before it
        # (including the user-role header). We strip the trailing empty content.
        # Different templates render "" differently, so we tokenise the full
        # thing and also a version with a sentinel to locate the split point.
        sentinel = "___SPLIT___"
        messages_s = self._messages(system_prompt, sentinel)
        full_s = self.tokenizer.apply_chat_template(
            messages_s, tokenize=False, add_generation_prompt=False,
        )
        idx = full_s.find(sentinel)
        if idx == -1:
            # Fallback: just use system-only
            prefix_text = full
        else:
            prefix_text = full_s[:idx]

        ids = self.tokenizer(prefix_text, return_tensors="pt",
                             add_special_tokens=False).input_ids.to(device)
        return ids

    def format_query_with_suffix(self, user_content: str) -> str:
        """
        Render [user message + generation prompt] as text that can be
        concatenated after a prefix.  Returns only the user-content part
        plus the assistant header (generation prompt).
        """
        sentinel = "___SPLIT___"
        messages_before = self._messages("S", sentinel)
        full_before = self.tokenizer.apply_chat_template(
            messages_before, tokenize=False, add_generation_prompt=True,
        )
        # Everything from the sentinel onward (including gen prompt) is the
        # "query suffix template".  We replace sentinel with actual content.
        idx = full_before.find(sentinel)
        if idx == -1:
            # Fallback: simple concatenation
            return user_content
        suffix_template = full_before[idx:]            # "___SPLIT___...<assistant header>"
        return suffix_template.replace(sentinel, user_content)
