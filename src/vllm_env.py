"""vLLM defaults for Kaggle / Jupyter (v1 engine subprocess workers)."""

from __future__ import annotations

import os


def apply_vllm_runtime_defaults() -> None:
    """Call before ``from vllm import LLM`` / first CUDA use in the process."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if os.path.isdir("/kaggle/working"):
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        # Avoid protobuf C API mismatch (MessageFactory.GetPrototype) with TF/XLA on Kaggle images.
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


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
