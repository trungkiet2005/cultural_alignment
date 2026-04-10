"""Model loading, tokenization helpers, and seed setup."""

import os

# Unsloth calls `get_statistics()` → HF snapshot_download with a 120s cap; on
# Kaggle this often times out (slow/blocked hub), falsely raising "HF is down".
os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")

import random as _rng

import numpy as np
import torch

try:
    import unsloth  # noqa: F401  — must be imported before transformers
except Exception:
    pass


# ── seed ────────────────────────────────────────────────────────────────────


def setup_seeds(seed: int = 42) -> None:
    """Seed all RNG sources used by the experiment.

    Sets `random`, `numpy`, torch CPU/CUDA, and forces cuDNN into deterministic
    mode. Without the cudnn flags, GPU convolutions/attention can pick different
    algorithms across runs, breaking reproducibility even with the same seed.
    """
    import os as _os
    _rng.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Required for `use_deterministic_algorithms(True)` to allow cuBLAS ops.
    _os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        # Older torch or kernels without deterministic implementations.
        pass


# ── model loading ───────────────────────────────────────────────────────────


def _transformers_version_tuple(ver: str) -> tuple:
    out = []
    for part in ver.split("+", 1)[0].split("."):
        if part.isdigit():
            out.append(int(part))
        else:
            break
    return tuple((out + [0, 0, 0])[:3])


def _clear_unsloth_compiled_cache() -> None:
    """Stale patches under transformers 5.2 can KeyError (e.g. ROPE_INIT_FUNCTIONS['default'])."""
    import shutil
    from pathlib import Path

    for base in (Path.cwd(), Path("/kaggle/working/cultural_alignment")):
        d = base / "unsloth_compiled_cache"
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
            print(f"[MODEL] Cleared Unsloth compile cache: {d}")


def _needs_tf55_git_unsloth_moe_fix(model_name: str) -> bool:
    """Qwen3-MoE Coder / Llama-4-Scout: need transformers>=5.5 + fresh compile (see ref_git_tf55)."""
    ln = model_name.lower()
    return ("qwen3-coder" in ln) or ("llama-4-scout" in ln)


def load_model(
    model_name: str,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
):
    """Load a model via Unsloth and return *(model, tokenizer)*."""
    import transformers
    transformers.logging.set_verbosity_error()

    print(f"[MODEL] Loading {model_name} via Unsloth...")
    # Gemma-4 is loaded via FastModel in Unsloth notebooks; FastLanguageModel
    # may raise NotImplementedError on older wheels or use the wrong path.
    if "gemma-4" in model_name.lower():
        _tv = transformers.__version__
        if _transformers_version_tuple(_tv) < (5, 5, 0):
            raise RuntimeError(
                f"Gemma-4 needs transformers>=5.5.0 (CONFIG_MAPPING['gemma4']); "
                f"this env has transformers=={_tv}. "
                "On Kaggle: run the full `ref_gemma4` pip block in exp_gemma4_31b.py "
                "— especially `pip install --upgrade --no-cache-dir transformers==5.5.0` "
                "*after* git-installing Unsloth. Then restart the kernel or pull the latest repo."
            )
        from unsloth import FastModel

        model, tokenizer = FastModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
        FastModel.for_inference(model)
        # Reference_Notebook_Model/gemma4-31b-unsloth.ipynb — template + message shape.
        try:
            from unsloth.chat_templates import get_chat_template

            tokenizer = get_chat_template(
                tokenizer,
                chat_template="gemma-4-thinking",
            )
        except Exception as exc:
            print(f"[MODEL] warn: get_chat_template(gemma-4-thinking) failed: {exc}")
        setattr(tokenizer, "_moral_chat_content_mode", "gemma4")
    else:
        if _needs_tf55_git_unsloth_moe_fix(model_name):
            _tv = transformers.__version__
            if _transformers_version_tuple(_tv) < (5, 5, 0):
                raise RuntimeError(
                    f"Qwen3-Coder / Llama-4-Scout MoE stacks need transformers>=5.5.0 with git Unsloth "
                    f"(exp profile ref_git_tf55); this env has transformers=={_tv}. "
                    "Pull latest repo and run e.g. exp_qwen3_coder_30b.py so _install_deps runs, "
                    "or restart kernel after `pip install --upgrade --no-cache-dir transformers==5.5.0` "
                    "following the ref_git_tf55 block."
                )
            _clear_unsloth_compiled_cache()
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=torch.bfloat16,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(model)
        setattr(tokenizer, "_moral_chat_content_mode", "string")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    print(f"[MODEL] Loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model, tokenizer


def text_tokenizer(tok):
    """Return the HF tokenizer used for plain text.

    Vision-Language `Processor` objects wrap the text tokenizer at `.tokenizer`;
    calling `processor(string)` routes through multimodal code and may treat text
    as image URLs. Always use this for `.encode` / `.decode` of chat strings.
    """
    inner = getattr(tok, "tokenizer", None)
    if inner is not None and callable(getattr(inner, "encode", None)):
        return inner
    return tok


def encode_text_to_tensor(
    tok, text: str, device, add_special_tokens: bool = False
) -> torch.Tensor:
    """Tokenize text only (never `Processor.__call__`, which is multimodal)."""
    tt = text_tokenizer(tok)
    ids = tt.encode(text, add_special_tokens=add_special_tokens)
    return torch.tensor([ids], dtype=torch.long, device=device)


# ── chat-template helper ────────────────────────────────────────────────────


class ChatTemplateHelper:
    """
    Builds tokenised chat prefixes and query suffixes using the tokenizer's
    built-in chat_template, so the same code works for Llama, Qwen, Gemma,
    Mistral, Command-R, and any other HuggingFace model.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Gemma-4 (Unsloth): messages use content = [{"type":"text","text":...}] — see
        # Reference_Notebook_Model/gemma4-31b-unsloth.ipynb
        self.content_mode = getattr(tokenizer, "_moral_chat_content_mode", "string")
        # Some chat templates (e.g. Gemma) don't accept a "system" role.
        # Detect once and fold the system prompt into the first user turn.
        self.supports_system = self._probe_system_role()

    def _as_turn_content(self, text: str):
        """Plain string (most models) or Gemma-4 multimodal text block."""
        if self.content_mode == "gemma4":
            return [{"type": "text", "text": text}]
        return text

    def _probe_system_role(self) -> bool:
        if self.content_mode == "gemma4":
            try:
                self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": self._as_turn_content("x")},
                        {"role": "user", "content": self._as_turn_content("y")},
                    ],
                    tokenize=False, add_generation_prompt=False,
                )
                return True
            except Exception:
                return False
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
                {"role": "system", "content": self._as_turn_content(system_prompt)},
                {"role": "user", "content": self._as_turn_content(user_content)},
            ]
        merged = f"{system_prompt}\n\n{user_content}" if user_content else system_prompt
        return [{"role": "user", "content": self._as_turn_content(merged)}]

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

        return encode_text_to_tensor(
            self.tokenizer, prefix_text, device, add_special_tokens=False
        )

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
