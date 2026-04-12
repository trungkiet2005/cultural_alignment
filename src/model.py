"""Model loading, tokenization helpers, and seed setup."""

import os
import sys

# Unsloth calls `get_statistics()` → HF snapshot_download with a 120s cap; on
# Kaggle this often times out (slow/blocked hub), falsely raising "HF is down".
os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")

import inspect
import random as _rng
import time

import numpy as np
import torch

# Unsloth is imported inside `load_model()` only so `load_model_hf_native` can run
# without installing or importing Unsloth.


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


def _has_transformers_config_key(key: str) -> bool:
    """Best-effort check for CONFIG_MAPPING support without importing model code."""
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        return key in CONFIG_MAPPING
    except Exception:
        return False


def _model_backend() -> str:
    return os.environ.get("MORAL_MODEL_BACKEND", "unsloth").strip().lower()


def _hf_from_pretrained_token_kw() -> dict:
    """Optional HF token for ``from_pretrained`` (Gemma/Llama gated repos).

    Reads ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN``. If unset, do **not** pass
    ``token=True``: that flag means \"must resolve a token\" and raises on Kaggle
    with no login (e.g. ``LocalTokenNotFoundError``). Omitting ``token`` still allows
    the hub client to use env/cache when present, and anonymous download for public
    repos (e.g. tokenizer-only Hub pulls for local Magistral weights).
    """
    t = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    return {"token": t} if t else {}


def _resolve_vllm_hf_model_id(model_name: str) -> str:
    """Map legacy Unsloth ids to upstream HF weights for vLLM (or pass through)."""
    ovr = os.environ.get("MORAL_VLLM_HF_MODEL", "").strip()
    if ovr:
        return ovr
    key = model_name.strip()
    table = {
        "unsloth/mistral-7b-instruct-v0.3-bnb-4bit": "mistralai/Mistral-7B-Instruct-v0.3",
        "unsloth/Magistral-Small-2509-unsloth-bnb-4bit": "mistralai/Magistral-Small-2509",
        "unsloth/Phi-3.5-mini-instruct": "microsoft/Phi-3.5-mini-instruct",
        "unsloth/Phi-4": "microsoft/phi-4",
        "unsloth/Llama-3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
        "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "unsloth/Llama-3.3-70B-Instruct-bnb-4bit": "meta-llama/Llama-3.3-70B-Instruct",
        "unsloth/gemma-7b-it-bnb-4bit": "google/gemma-7b-it",
        "unsloth/gemma-3-270m-it": "google/gemma-3-270m-it",
        "unsloth/Qwen3-4B-Thinking-2507-unsloth-bnb-4bit": "Qwen/Qwen3-4B-Thinking-2507",
        "unsloth/Qwen3.5-0.8B": "Qwen/Qwen3.5-0.8B",
        "unsloth/Qwen2.5-7B-Instruct-bnb-4bit": "Qwen/Qwen2.5-7B-Instruct",
        "unsloth/Qwen2.5-72B-Instruct-bnb-4bit": "Qwen/Qwen2.5-72B-Instruct",
        "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit": "meta-llama/Llama-3.1-70B-Instruct",
        "unsloth/mistral-7b-instruct-v0.2-bnb-4bit": "mistralai/Mistral-7B-Instruct-v0.2",
        "unsloth/gpt-oss-20b-unsloth-bnb-4bit": "openai/gpt-oss-20b",
    }
    if key in table:
        return table[key]
    if "/" in key and not key.lower().startswith("unsloth/"):
        return key
    raise ValueError(
        f"vLLM: unknown or Unsloth-only id {model_name!r}. Set MORAL_VLLM_HF_MODEL to the "
        "HuggingFace repo id (e.g. org/model), or add a mapping in src/model.py::_resolve_vllm_hf_model_id."
    )


def _load_model_vllm(
    model_name: str,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
):
    from transformers import AutoTokenizer

    from src.vllm_logit_model import VllmCausalLogitModel

    hf_id = _resolve_vllm_hf_model_id(model_name)
    print(f"[MODEL] Loading {hf_id} via vLLM (config key was {model_name!r})...")

    _t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        hf_id, trust_remote_code=True, **_hf_from_pretrained_token_kw()
    )
    print(f"[MODEL] Tokenizer ready in {time.perf_counter() - _t0:.1f}s")
    _tt = text_tokenizer(tokenizer)
    if getattr(_tt, "pad_token", None) is None and getattr(_tt, "eos_token", None) is not None:
        _tt.pad_token = _tt.eos_token
        _tt.pad_token_id = _tt.eos_token_id
    try:
        tokenizer.padding_side = "left"
    except Exception:
        pass
    _tt.padding_side = "left"
    setattr(tokenizer, "_moral_chat_content_mode", "string")

    from src.vllm_env import apply_vllm_quantization_kw, apply_vllm_runtime_defaults

    apply_vllm_runtime_defaults()

    try:
        from vllm import LLM
    except ImportError as exc:
        raise ImportError(
            "MORAL_MODEL_BACKEND=vllm requires the vllm package (pip install vllm)."
        ) from exc

    gpu_mem = float(
        os.environ.get("MORAL_VLLM_GPU_MEM")
        or os.environ.get("VLLM_GPU_MEMORY_UTILIZATION")
        or "0.92"
    )
    _eager_raw = os.environ.get("MORAL_VLLM_ENFORCE_EAGER") or os.environ.get(
        "VLLM_ENFORCE_EAGER", "1"
    )
    eager = _eager_raw.strip().lower() not in ("0", "false", "no")
    llm_kw: dict = {
        "model": hf_id,
        "max_model_len": max_seq_length,
        "trust_remote_code": True,
        "gpu_memory_utilization": gpu_mem,
        "enforce_eager": eager,
    }
    if os.path.isdir("/kaggle/working"):
        if os.environ.get("VLLM_DISABLE_CUSTOM_ALL_REDUCE", "1").strip().lower() not in (
            "0",
            "false",
            "no",
        ):
            llm_kw["disable_custom_all_reduce"] = True
    dtype = os.environ.get("VLLM_DTYPE", "").strip()
    if dtype:
        llm_kw["dtype"] = dtype
    apply_vllm_quantization_kw(
        llm_kw,
        hf_id=hf_id,
        model_key=model_name,
        load_in_4bit=load_in_4bit,
        legacy_causal_always_bnb=False,
    )

    tp = os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "").strip()
    if tp:
        llm_kw["tensor_parallel_size"] = max(1, int(tp))

    print(
        "[MODEL] Building vLLM engine + loading weights (often 2–15+ min first run; "
        "watch FlashAttention / GPU load lines above)…"
    )
    _t_llm = time.perf_counter()
    llm = LLM(**llm_kw)
    print(f"[MODEL] vLLM LLM() done in {time.perf_counter() - _t_llm:.1f}s")
    vocab = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
    wrapper = VllmCausalLogitModel(
        llm, pad_token_id=tokenizer.pad_token_id, vocab_size=int(vocab)
    )
    wrapper._tokenizer_ref = tokenizer
    print(f"[MODEL] vLLM engine ready (vocab_size={vocab}).")
    return wrapper, tokenizer


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


def _transformers_major_minor() -> tuple[int, int]:
    import importlib.metadata as md

    ver = md.version("transformers").split("+", 1)[0]
    parts = ver.split(".")
    return int(parts[0]), int(parts[1])


def _loaded_transformers_major_minor() -> tuple[int, int] | None:
    mod = sys.modules.get("transformers")
    if mod is None:
        return None
    ver = getattr(mod, "__version__", "0.0").split("+", 1)[0]
    parts = ver.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return None


def _require_transformers_55_for_mistral3_tekken(*, context: str) -> None:
    """Magistral uses ``tekken.json`` + Mistral3; needs transformers>=5.5 (tokenizer mapping).

    After ``pip install`` in the same notebook, an **already-imported** transformers
    (e.g. 4.56) stays stale — user must restart the kernel.
    """
    dm, dn = _transformers_major_minor()
    if dm < 5 or (dm == 5 and dn < 5):
        raise RuntimeError(
            f"{context}: need transformers>=5.5.0 for Mistral3/tekken tokenizer; "
            f"site-packages reports {dm}.{dn}.\n"
            "Install: pip install -U --force-reinstall 'transformers>=5.5.0,<6.0'\n"
            "On Kaggle: enable Internet, run that in a cell, then **Session → Restart** and re-run."
        )
    loaded = _loaded_transformers_major_minor()
    if loaded is not None:
        lm, ln = loaded
        if lm < 5 or (lm == 5 and ln < 5):
            raise RuntimeError(
                f"{context}: this Python process already imported transformers {lm}.{ln}, "
                f"but site-packages has {dm}.{dn}.\n"
                "**Restart the Jupyter kernel** (Session → Restart), then re-run — "
                "`pip install` alone cannot refresh an imported transformers."
            )


def _tokenizer_hub_for_local_weights(local_dir: str) -> str:
    """Upstream HF repo id to load tokenizer when a local snapshot omits tiktoken vocab files."""
    import json
    from pathlib import Path

    p = Path(local_dir) / "config.json"
    if not p.is_file():
        return ""
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return ""
    mt = (cfg.get("model_type") or "").lower()
    arch = " ".join(str(a) for a in (cfg.get("architectures") or [])).lower()
    if "mistral3" in mt or "mistral3" in arch:
        return (
            os.environ.get("MORAL_TOKENIZER_HUB_ID", "").strip()
            or "mistralai/Magistral-Small-2509"
        )
    return ""


def load_model_hf_native(
    model_name: str,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
):
    """Load *(model, tokenizer)* with Hugging Face ``AutoModel`` only (no Unsloth)."""
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    transformers.logging.set_verbosity_error()
    del max_seq_length  # reserved for parity with Unsloth API; HF uses tokenizer/model config limits

    print(f"[MODEL] Loading {model_name} via Hugging Face transformers (native, no Unsloth)...")

    _tok_kw = _hf_from_pretrained_token_kw()
    hub_override = os.environ.get("MORAL_TOKENIZER_HUB_ID", "").strip()
    _mn_low = model_name.lower()
    _need_mistral3_tekken = (
        (os.path.isdir(model_name) and bool(_tokenizer_hub_for_local_weights(model_name)))
        or (hub_override and "mistral" in hub_override.lower())
        or "magistral" in _mn_low
        or "mistral-small-2509" in _mn_low
    )
    if _need_mistral3_tekken:
        _require_transformers_55_for_mistral3_tekken(
            context="load_model_hf_native (Magistral / Mistral3 tekken tokenizer)",
        )

    def _load_tokenizer(path_or_id: str):
        return AutoTokenizer.from_pretrained(
            path_or_id, trust_remote_code=True, **_tok_kw
        )

    if hub_override:
        print(f"[MODEL] Tokenizer from MORAL_TOKENIZER_HUB_ID={hub_override!r} (weights from {model_name!r})")
        tokenizer = _load_tokenizer(hub_override)
    else:
        try:
            tokenizer = _load_tokenizer(model_name)
        except Exception as first_exc:
            hub = ""
            if os.path.isdir(model_name):
                hub = _tokenizer_hub_for_local_weights(model_name)
            if hub:
                print(
                    f"[MODEL] Local tokenizer failed ({type(first_exc).__name__}); "
                    f"loading tokenizer from Hub {hub!r} (weights stay local)…"
                )
                tokenizer = _load_tokenizer(hub)
            else:
                raise first_exc

    attn_impl = os.environ.get("HF_ATTN_IMPLEMENTATION", "").strip() or None
    common_kw: dict = {
        "trust_remote_code": True,
        "device_map": "auto",
        **_tok_kw,
    }
    if attn_impl:
        common_kw["attn_implementation"] = attn_impl

    if load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant,
            **common_kw,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            **common_kw,
        )

    model.eval()
    setattr(tokenizer, "_moral_chat_content_mode", "string")

    _tt = text_tokenizer(tokenizer)
    if getattr(_tt, "pad_token", None) is None and getattr(_tt, "eos_token", None) is not None:
        _tt.pad_token = _tt.eos_token
        _tt.pad_token_id = _tt.eos_token_id
    try:
        tokenizer.padding_side = "left"
    except Exception:
        pass
    _tt.padding_side = "left"
    print(f"[MODEL] Loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model, tokenizer


def gather_last_logits(
    out,
    batch_idx: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Last-position logits [B, V] from a model ``forward`` output.

    Hugging Face / Unsloth return ``logits`` shaped [B, L, V]. The vLLM wrapper
    (``VllmCausalWrapper``) returns only the last position as [B, V].
    """
    logits = out.logits
    if logits.dim() == 2:
        return logits
    return logits[batch_idx, lengths - 1, :]


def gather_last_logits_one_row(out) -> torch.Tensor:
    """Single-row forward (batch 1): return 1D logits [V]."""
    logits = out.logits
    if logits.dim() == 2:
        return logits[0]
    return logits[0, -1, :]


def load_model(
    model_name: str,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
):
    """Load a model and return *(model, tokenizer)*.

    Backend is ``unsloth`` (default) or ``vLLM`` when ``MORAL_MODEL_BACKEND=vllm``.
    """
    try:
        import unsloth  # noqa: F401  — must be imported before transformers in Unsloth path
    except Exception:
        pass

    import transformers
    transformers.logging.set_verbosity_error()

    if _model_backend() == "vllm":
        return _load_model_vllm(model_name, max_seq_length, load_in_4bit)

    print(f"[MODEL] Loading {model_name} via Unsloth...")
    # Gemma-4 is loaded via FastModel in Unsloth notebooks; FastLanguageModel
    # may raise NotImplementedError on older wheels or use the wrong path.
    if "gemma-4" in model_name.lower():
        _tv = transformers.__version__
        # Prefer feature detection over hard version-gating: Kaggle images sometimes ship
        # a newer wheel than the pinned ref_* profile, and pip installs can fail silently.
        if _transformers_version_tuple(_tv) < (5, 5, 0) and not _has_transformers_config_key("gemma4"):
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
    elif "gemma-3n" in model_name.lower():
        # Reference_Notebook_Model/Gemma3N_(4B)_Audio.ipynb — Gemma 3N uses FastModel + Processor,
        # not FastLanguageModel (which fails or mis-loads this family).
        from unsloth import FastModel

        model, tokenizer = FastModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
        FastModel.for_inference(model)
        setattr(tokenizer, "_moral_chat_content_mode", "string")
    elif "llama-4-scout" in model_name.lower():
        # HF: Llama4ForConditionalGeneration (image-text-to-text), not LlamaForCausalLM.
        # FastLanguageModel targets causal LMs only → Unsloth preflight hits "No config file found".
        _tv = transformers.__version__
        if _transformers_version_tuple(_tv) < (5, 5, 0) and not _has_transformers_config_key(
            "llama4"
        ):
            raise RuntimeError(
                f"Llama-4-Scout needs transformers>=5.5.0 (CONFIG_MAPPING['llama4']); "
                f"this env has transformers=={_tv}. "
                "Use the ref_git_tf55 pip block in exp_llama4_scout.py (fresh Kaggle session)."
            )
        if _needs_tf55_git_unsloth_moe_fix(model_name):
            _clear_unsloth_compiled_cache()
        from unsloth import FastModel

        model, tokenizer = FastModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        FastModel.for_inference(model)
        setattr(tokenizer, "_moral_chat_content_mode", "string")
    else:
        if _needs_tf55_git_unsloth_moe_fix(model_name):
            _tv = transformers.__version__
            if _transformers_version_tuple(_tv) < (5, 5, 0):
                # Don't block purely on version: if the environment already contains the needed
                # model code paths, Unsloth can still load successfully. We'll clear Unsloth's
                # compiled cache and proceed; if transformers truly lacks support, the next call
                # will raise a more concrete error.
                print(
                    f"[MODEL] warn: {model_name} was validated on transformers>=5.5.0 + git Unsloth "
                    f"(ref_git_tf55), but this env has transformers=={_tv}. "
                    "Proceeding with feature-detection; if load fails, follow ref_git_tf55 install block."
                )
            _clear_unsloth_compiled_cache()
        from unsloth import FastLanguageModel

        # MoE / custom HF modeling: without trust_remote_code, Unsloth's AutoConfig/PeftConfig
        # both fail; the loader then calls get_transformers_model_type(None) and surfaces a
        # misleading "No config file found" (see unsloth/models/loader.py before the combined error).
        ln = model_name.lower()
        _moe_remote = "nemotron" in ln
        fm_kw: dict = {
            "model_name": model_name,
            "max_seq_length": max_seq_length,
            "dtype": torch.bfloat16,
            "load_in_4bit": load_in_4bit,
            "trust_remote_code": _moe_remote,
        }
        if _moe_remote:
            fm_kw["attn_implementation"] = "eager"

        model, tokenizer = FastLanguageModel.from_pretrained(**fm_kw)
        # CodeGemma: HF tokenizer ships without `chat_template`; Unsloth notebook applies ChatML
        # via get_chat_template before inference (Reference_Notebook_Model/CodeGemma_(7B)_Conversational.ipynb).
        if "codegemma" in model_name.lower():
            try:
                from unsloth.chat_templates import get_chat_template

                tokenizer = get_chat_template(
                    tokenizer,
                    chat_template="chatml",
                    mapping={
                        "role": "from",
                        "content": "value",
                        "user": "human",
                        "assistant": "gpt",
                    },
                    map_eos_token=True,
                )
            except Exception as exc:
                print(f"[MODEL] warn: get_chat_template(chatml) for CodeGemma failed: {exc}")
        FastLanguageModel.for_inference(model)
        setattr(tokenizer, "_moral_chat_content_mode", "string")
    # Processor (Gemma 3N / VL) keeps pad/eos on the inner text tokenizer.
    _tt = text_tokenizer(tokenizer)
    if getattr(_tt, "pad_token", None) is None and getattr(_tt, "eos_token", None) is not None:
        _tt.pad_token = _tt.eos_token
        _tt.pad_token_id = _tt.eos_token_id
    try:
        tokenizer.padding_side = "left"
    except Exception:
        pass
    _tt.padding_side = "left"
    print(f"[MODEL] Loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model, tokenizer


def encode_text_to_tensor(
    tok, text: str, device, add_special_tokens: bool = False
) -> torch.Tensor:
    """Tokenize text only (never `Processor.__call__`, which is multimodal)."""
    tt = text_tokenizer(tok)
    ids = tt.encode(text, add_special_tokens=add_special_tokens)
    return torch.tensor([ids], dtype=torch.long, device=device)


def _coerce_chat_template_ids(raw) -> list:
    """Convert apply_chat_template(tokenize=…) output to a Python list of int ids."""
    if isinstance(raw, torch.Tensor):
        raw = raw.detach().cpu().flatten().tolist()
    if isinstance(raw, np.ndarray):
        raw = raw.astype(np.int64).flatten().tolist()
    if isinstance(raw, (list, tuple)):
        return [int(x) for x in raw]
    raise TypeError(f"expected token id sequence, got {type(raw)!r}")


def _is_chat_template_id_sequence(raw) -> bool:
    """True when chat-template output is already token ids (some Unsloth builds)."""
    if isinstance(raw, torch.Tensor) and raw.numel() > 0:
        return not raw.dtype.is_floating_point
    if isinstance(raw, np.ndarray) and raw.size > 0:
        return bool(np.issubdtype(raw.dtype, np.integer))
    if isinstance(raw, (list, tuple)) and len(raw) > 0:
        try:
            int(raw[0])
        except (TypeError, ValueError):
            return False
        return True
    return False


def _lcp_token_ids(a: list, b: list) -> list:
    """Longest common prefix of two equal-length token lists (used for chat splits)."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return a[:i]


def _find_subseq(haystack: list, needle: list) -> int:
    if not needle:
        return 0
    last = len(haystack) - len(needle) + 1
    for i in range(max(last, 0)):
        if haystack[i : i + len(needle)] == needle:
            return i
    return -1


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
        # Unsloth / certain tokenizers return token-id lists even when tokenize=False.
        self.chat_template_returns_ids = self._probe_chat_template_returns_ids()

    def _probe_chat_template_returns_ids(self) -> bool:
        ct = self._chat_template_target()
        try:
            msgs = self._messages("probe_sys", "probe_user")
            r = self._apply_chat_template(
                ct, msgs, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            return False
        return _is_chat_template_id_sequence(r)

    def _chat_template_target(self):
        """Prefer Processor/outer tokenizer; fall back to inner text tokenizer (Gemma 3N)."""
        tok = self.tokenizer
        if callable(getattr(tok, "apply_chat_template", None)):
            return tok
        inner = text_tokenizer(tok)
        if callable(getattr(inner, "apply_chat_template", None)):
            return inner
        return tok

    def _as_turn_content(self, text: str):
        """Plain string (most models) or Gemma-4 multimodal text block."""
        if self.content_mode == "gemma4":
            return [{"type": "text", "text": text}]
        return text

    def _apply_chat_template(self, ct, messages, *, tokenize: bool, add_generation_prompt: bool):
        """Call ``apply_chat_template`` with only kwargs the backend accepts.

        ``MistralCommonTokenizer`` (Kaggle / some transformers builds) rejects any
        unknown keyword, including ``add_generation_prompt=False``.
        """
        try:
            names = set(inspect.signature(ct.apply_chat_template).parameters)
        except (TypeError, ValueError):
            names = set()

        kw = {}
        if "tokenize" in names:
            kw["tokenize"] = tokenize
        if "add_generation_prompt" in names:
            kw["add_generation_prompt"] = add_generation_prompt

        try:
            return ct.apply_chat_template(messages, **kw)
        except (TypeError, ValueError) as e:
            es = str(e)
            if (
                "not supported" not in es
                and "unexpected keyword" not in es.lower()
                and "got an unexpected keyword" not in es.lower()
            ):
                raise

        kw2 = {k: v for k, v in kw.items() if k == "tokenize"}
        try:
            base = ct.apply_chat_template(messages, **kw2)
        except TypeError:
            base = ct.apply_chat_template(messages, tokenize=tokenize)

        if not add_generation_prompt:
            return base

        if "continue_final_message" in names:
            try:
                return ct.apply_chat_template(
                    messages, tokenize=tokenize, continue_final_message=True
                )
            except Exception:
                pass

        probe = "\u2060MORALGEN\u2060"
        ext = list(messages) + [{"role": "assistant", "content": probe}]
        try:
            w = ct.apply_chat_template(ext, **kw2)
        except TypeError:
            w = ct.apply_chat_template(ext, tokenize=tokenize)
        except Exception:
            return base

        if isinstance(base, str) and isinstance(w, str) and w.startswith(base) and probe in w:
            return base + w[len(base) :].split(probe, 1)[0]
        if isinstance(w, str) and probe in w:
            return w.split(probe, 1)[0]
        return base

    def _probe_system_role(self) -> bool:
        ct = self._chat_template_target()
        if self.content_mode == "gemma4":
            try:
                self._apply_chat_template(
                    ct,
                    [
                        {"role": "system", "content": self._as_turn_content("x")},
                        {"role": "user", "content": self._as_turn_content("y")},
                    ],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                return True
            except Exception:
                return False
        try:
            self._apply_chat_template(
                ct,
                [
                    {"role": "system", "content": "x"},
                    {"role": "user", "content": "y"},
                ],
                tokenize=False,
                add_generation_prompt=False,
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
        ct = self._chat_template_target()
        messages = self._messages(system_prompt, "")
        full = self._apply_chat_template(
            ct, messages, tokenize=False, add_generation_prompt=False
        )
        sentinel = "___SPLIT___"
        messages_s = self._messages(system_prompt, sentinel)
        full_s = self._apply_chat_template(
            ct, messages_s, tokenize=False, add_generation_prompt=False
        )

        if _is_chat_template_id_sequence(full) and _is_chat_template_id_sequence(full_s):
            prefix_ids = _lcp_token_ids(
                _coerce_chat_template_ids(full),
                _coerce_chat_template_ids(full_s),
            )
            return torch.tensor([prefix_ids], dtype=torch.long, device=device)

        if not isinstance(full, str) or not isinstance(full_s, str):
            tt = text_tokenizer(ct)
            full = (
                full
                if isinstance(full, str)
                else tt.decode(_coerce_chat_template_ids(full), skip_special_tokens=False)
            )
            full_s = (
                full_s
                if isinstance(full_s, str)
                else tt.decode(
                    _coerce_chat_template_ids(full_s), skip_special_tokens=False
                )
            )

        idx = full_s.find(sentinel)
        if idx == -1:
            prefix_text = full
        else:
            prefix_text = full[:idx]

        return encode_text_to_tensor(
            self.tokenizer, prefix_text, device, add_special_tokens=False
        )

    def query_suffix_ids(self, user_content: str, *, add_generation_prompt: bool) -> list:
        """User-turn + optional generation prompt as token ids (ids-mode templates)."""
        ct = self._chat_template_target()
        tt = text_tokenizer(self.tokenizer)
        sentinel = "___SPLIT___"
        messages_before = self._messages("S", sentinel)
        templated = self._apply_chat_template(
            ct,
            messages_before,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
        if isinstance(templated, str):
            idx = templated.find(sentinel)
            if idx < 0:
                return tt.encode(user_content, add_special_tokens=False)
            suffix = templated[idx:].replace(sentinel, user_content)
            return tt.encode(suffix, add_special_tokens=False)
        ids = _coerce_chat_template_ids(templated)
        sent_ids = tt.encode(sentinel, add_special_tokens=False)
        i = _find_subseq(ids, sent_ids)
        if i < 0:
            return tt.encode(user_content, add_special_tokens=False)
        user_ids = tt.encode(user_content, add_special_tokens=False)
        # Match string path: suffix from sentinel onward → user_content + tail after sentinel.
        tail = ids[i + len(sent_ids) :]
        return user_ids + tail

    def query_suffix_to_tensor(self, user_content: str, device) -> torch.Tensor:
        ids = self.query_suffix_ids(user_content, add_generation_prompt=True)
        return torch.tensor([ids], dtype=torch.long, device=device)

    def decode_query_suffix_str_for_ab_probe(self, user_content: str) -> str:
        """Decoded suffix string for appending 'A'/'B' (BPE boundary logic)."""
        tt = text_tokenizer(self.tokenizer)
        if self.chat_template_returns_ids:
            ids = self.query_suffix_ids(user_content, add_generation_prompt=True)
            return tt.decode(ids, skip_special_tokens=False)
        return self.format_query_with_suffix(user_content)

    def encode_query_suffix(self, user_content: str, device) -> torch.Tensor:
        if self.chat_template_returns_ids:
            return self.query_suffix_to_tensor(user_content, device)
        return encode_text_to_tensor(
            self.tokenizer,
            self.format_query_with_suffix(user_content),
            device,
            add_special_tokens=False,
        )

    def format_query_with_suffix(self, user_content: str) -> str:
        """
        Render [user message + generation prompt] as text that can be
        concatenated after a prefix.  Returns only the user-content part
        plus the assistant header (generation prompt).
        """
        sentinel = "___SPLIT___"
        if self.chat_template_returns_ids:
            tt = text_tokenizer(self.tokenizer)
            return tt.decode(
                self.query_suffix_ids(user_content, add_generation_prompt=True),
                skip_special_tokens=False,
            )
        messages_before = self._messages("S", sentinel)
        full_before = self._apply_chat_template(
            self._chat_template_target(),
            messages_before,
            tokenize=False,
            add_generation_prompt=True,
        )
        if _is_chat_template_id_sequence(full_before):
            tt = text_tokenizer(self.tokenizer)
            full_before = tt.decode(
                _coerce_chat_template_ids(full_before), skip_special_tokens=False
            )
        # Everything from the sentinel onward (including gen prompt) is the
        # "query suffix template".  We replace sentinel with actual content.
        idx = full_before.find(sentinel)
        if idx == -1:
            # Fallback: simple concatenation
            return user_content
        suffix_template = full_before[idx:]            # "___SPLIT___...<assistant header>"
        return suffix_template.replace(sentinel, user_content)
