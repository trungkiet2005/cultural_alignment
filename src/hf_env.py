"""Hugging Face credentials: repo-root ``.env`` + optional Kaggle ``HF_TOKEN`` secret."""

from __future__ import annotations

import os
from pathlib import Path


def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    if "=" not in s:
        return None
    k, _, v = s.partition("=")
    k, v = k.strip(), v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        v = v[1:-1]
    if not k:
        return None
    return k, v


def load_dotenv_repo(*, cwd: str | None = None) -> None:
    """Load ``<cwd>/.env`` into ``os.environ``. Does not overwrite existing keys."""
    root = Path(cwd or os.getcwd()).resolve()
    path = root / ".env"
    if not path.is_file():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        pair = _parse_dotenv_line(raw)
        if pair is None:
            continue
        key, val = pair
        if key not in os.environ:
            os.environ[key] = val


def apply_kaggle_hf_secret_if_missing() -> None:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return
    if not os.path.isdir("/kaggle/working"):
        return
    try:
        from kaggle_secrets import UserSecretsClient

        t = UserSecretsClient().get_secret("HF_TOKEN")
        if t:
            os.environ["HF_TOKEN"] = t
            os.environ["HUGGING_FACE_HUB_TOKEN"] = t
    except Exception:
        pass


def mirror_hf_token_aliases() -> None:
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    elif os.environ.get("HUGGING_FACE_HUB_TOKEN") and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"]


def apply_hf_credentials() -> None:
    """Call after ``chdir`` to repo root (e.g. right after ``_ensure_repo()``)."""
    if os.path.isdir("/kaggle/working"):
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    load_dotenv_repo()
    apply_kaggle_hf_secret_if_missing()
    mirror_hf_token_aliases()
    try:
        from src.vllm_env import apply_vllm_runtime_defaults

        apply_vllm_runtime_defaults()
    except Exception:
        pass
