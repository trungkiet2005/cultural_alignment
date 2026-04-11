"""vLLM defaults for Kaggle / Jupyter (v1 engine subprocess workers)."""

from __future__ import annotations

import os
import re
from pathlib import Path


def _largest_billion_param_tag(*parts: str):
    """Largest ``NNB`` / ``NNb`` token in strings (e.g. 70B, 72B); None if none."""
    text = " ".join(p for p in parts if p)
    best = None
    for m in re.finditer(r"(?<![.\d])(\d+)\s*[Bb]\b", text):
        n = int(m.group(1))
        if best is None or n > best:
            best = n
    return best


def apply_vllm_quantization_kw(
    llm_kw: dict,
    *,
    hf_id: str,
    model_key: str,
    load_in_4bit: bool,
    legacy_causal_always_bnb: bool = False,
) -> None:
    """Set ``llm_kw['quantization']`` for vLLM ``LLM()`` when appropriate.

    * If ``VLLM_QUANTIZATION`` is set, it always wins.
    * If the id strings suggest **≥70B** parameters and ``load_in_4bit`` and auto is on,
      use ``MORAL_VLLM_LARGE_QUANT`` (default ``bitsandbytes``).
    * If ``legacy_causal_always_bnb`` (``load_model_vllm`` path): when ``load_in_4bit``,
      still apply ``bitsandbytes`` for smaller models (historical behaviour).
    * Else if ``load_in_4bit``, print the usual note that vLLM ignores Unsloth's 4-bit flag.

    Disable auto 70B+ with ``MORAL_VLLM_AUTO_QUANT_70B=0``.
    """
    quant = os.environ.get("VLLM_QUANTIZATION", "").strip()
    if quant:
        llm_kw["quantization"] = quant
        return

    auto_raw = os.environ.get("MORAL_VLLM_AUTO_QUANT_70B", "1").strip().lower()
    auto_on = auto_raw not in ("0", "false", "no", "off")

    if load_in_4bit and auto_on:
        n = _largest_billion_param_tag(hf_id, model_key)
        if n is not None and n >= 70:
            large_q = os.environ.get("MORAL_VLLM_LARGE_QUANT", "bitsandbytes").strip() or "bitsandbytes"
            llm_kw["quantization"] = large_q
            print(
                f"[MODEL] Id suggests ~{n}B: vLLM quantization={large_q!r} "
                "(override: VLLM_QUANTIZATION=…; disable: MORAL_VLLM_AUTO_QUANT_70B=0)."
            )
            return

    if legacy_causal_always_bnb and load_in_4bit:
        bnb = os.environ.get("MORAL_VLLM_LEGACY_QUANT", "bitsandbytes").strip() or "bitsandbytes"
        llm_kw["quantization"] = bnb
        return

    if load_in_4bit:
        print(
            "[MODEL] note: vLLM ignores load_in_4bit; use full weights or set VLLM_QUANTIZATION "
            "(e.g. awq, gptq, bitsandbytes) if you need compressed weights."
        )


def _prepend_path_list(env_key: str, *dirs: str) -> None:
    """Prepend existing directories to a colon-separated search path (no duplicates)."""
    parts = [x for x in (os.environ.get(env_key) or "").split(":") if x]
    for d in reversed(dirs):
        if d and os.path.isdir(d) and d not in parts:
            parts.insert(0, d)
    if parts:
        os.environ[env_key] = ":".join(parts)


def _ensure_libcuda_dir_for_linker() -> str | None:
    """Return a directory where ``ld -lcuda`` can resolve ``libcuda.so``.

    FlashInfer JIT (e.g. Qwen3.5 / qwen3_next GDN on H100) links with ``-lcuda``.
    Many images only ship ``libcuda.so.1`` under ``/usr/lib/x86_64-linux-gnu``; the
    linker needs ``libcuda.so``. We symlink into a writable cache dir when needed.

    Override with ``MORAL_VLLM_LIBCUDA_DIR`` (folder containing ``libcuda.so`` or
    ``libcuda.so.1``). Set ``MORAL_VLLM_SKIP_LIBCUDA_PATH_FIX=1`` to disable.
    """
    if os.environ.get("MORAL_VLLM_SKIP_LIBCUDA_PATH_FIX", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return None

    custom = os.environ.get("MORAL_VLLM_LIBCUDA_DIR", "").strip()
    candidates = (
        [custom]
        if custom
        else [
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib64",
            "/usr/local/cuda/lib64/stubs",
            "/usr/local/cuda/targets/x86_64-linux/lib/stubs",
            "/usr/local/cuda/lib64",
            "/usr/lib/wsl/lib",
        ]
    )

    for p in candidates:
        if not p or not os.path.isdir(p):
            continue
        so = os.path.join(p, "libcuda.so")
        so1 = os.path.join(p, "libcuda.so.1")
        if os.path.isfile(so):
            return p
        if os.path.isfile(so1):
            home = os.environ.get("HOME", "/tmp")
            cache = Path(home) / ".cache" / "moral_vllm_libcuda_link"
            try:
                cache.mkdir(parents=True, exist_ok=True)
            except OSError:
                continue
            link = cache / "libcuda.so"
            if not link.is_file() and not link.is_symlink():
                try:
                    link.symlink_to(os.path.abspath(so1))
                except OSError:
                    continue
            return str(cache)
    return None


def apply_vllm_runtime_defaults() -> None:
    """Call before ``from vllm import LLM`` / first CUDA use in the process."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if os.path.isdir("/kaggle/working"):
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        # Avoid protobuf C API mismatch (MessageFactory.GetPrototype) with TF/XLA on Kaggle images.
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    libcuda_dir = _ensure_libcuda_dir_for_linker()
    if libcuda_dir:
        _prepend_path_list("LIBRARY_PATH", libcuda_dir)
        _prepend_path_list("LD_LIBRARY_PATH", libcuda_dir)
        print(
            f"[vLLM env] FlashInfer link fix: prepended LIBRARY_PATH/LD_LIBRARY_PATH with {libcuda_dir!r} "
            "(see MORAL_VLLM_LIBCUDA_DIR / MORAL_VLLM_SKIP_LIBCUDA_PATH_FIX)."
        )


def vllm_preflight_os_environ_lines() -> str:
    """Snippet for ``python -c`` preflight: must run before ``import torch``."""
    return """
import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("UNSLOTH_DISABLE_AUTO_COMPILE", "1")
os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
""".strip()
