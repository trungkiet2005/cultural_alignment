"""Shared env + Kaggle pip bootstrap for ``exp_paper/*`` entry scripts.

Default backend is **Unsloth** (same as ``src.model.load_model`` when
``MORAL_MODEL_BACKEND`` is unset). To use vLLM instead:

    export MORAL_MODEL_BACKEND=vllm   # or set in notebook before running the cell
"""

from __future__ import annotations

import os
import subprocess


def on_kaggle() -> bool:
    return os.path.isdir("/kaggle/working")


def paper_backend() -> str:
    return os.environ.get("MORAL_MODEL_BACKEND", "unsloth").strip().lower()


def configure_paper_env(*, vllm_gpu_mem_default: str = "0.95") -> None:
    """Torch + Unsloth stats flags; optional vLLM VRAM fraction on Kaggle."""
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
    os.environ.setdefault("UNSLOTH_DISABLE_AUTO_COMPILE", "1")
    if paper_backend() == "vllm" and on_kaggle():
        os.environ.setdefault("MORAL_VLLM_GPU_MEM", vllm_gpu_mem_default)


def install_paper_kaggle_deps() -> None:
    """Install either vLLM stack or Unsloth + bitsandbytes (Kaggle only)."""
    if not on_kaggle():
        return
    if paper_backend() == "vllm":
        cmds = [
            'pip install -q "numpy<2.3"',
            "pip uninstall -y -q tensorflow tensorflow-cpu tf_keras 2>/dev/null || true",
            'pip install -q --upgrade "protobuf>=5.29.6,<6" "grpcio>=1.68" "googleapis-common-protos>=1.66"',
            "pip install -q scipy tqdm sentencepiece",
            "pip install -q vllm",
            'pip install --quiet "datasets>=3.4.1,<4.4.0"',
        ]
    else:
        cmds = [
            "pip install -q bitsandbytes scipy tqdm matplotlib seaborn",
            "pip install --upgrade --no-deps unsloth",
            "pip install -q unsloth_zoo",
            "pip install --quiet --no-deps --force-reinstall pyarrow",
            "pip install --quiet 'datasets>=3.4.1,<4.4.0'",
        ]
    for c in cmds:
        subprocess.run(c, shell=True, check=False)
