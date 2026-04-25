"""Kaggle offline bootstrap + output zipper for playbook experiments.

Every exp_r2 / exp_r3 / exp_r4 / exp_r5 / disca-playbook script in this
directory uses the two helpers exposed here.

Why offline mode? Kaggle competition kernels run with internet disabled, so
``git clone`` is not available. Instead, we expect the repo to be attached
as a Kaggle input dataset (or a Kaggle notebook output). The candidates
listed in :data:`PROJECT_DATASET_CANDIDATES` cover the common upload paths.

Why the zip? Kaggle's working dir contents are not directly browsable as a
folder tree on the dashboard — but a single .zip file is a one-click
download. After each experiment we zip ``RESULTS_BASE`` to
``/kaggle/working/<name>.zip``.

Usage from a playbook script:
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from _kaggle_setup import bootstrap_offline, zip_outputs
    bootstrap_offline()
    ...
    if __name__ == "__main__":
        main()
        zip_outputs(RESULTS_BASE)
"""

from __future__ import annotations

import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Optional, Union

PROJECT_DATASET_CANDIDATES = [
    "/kaggle/input/notebooks/foundnotkiet/git-moral/cultural-alignment",
    "/kaggle/input/notebooks/foundnotkiet/git-moral/cultural_alignment",
    "/kaggle/input/cultural-alignment",
    "/kaggle/input/cultural_alignment",
]
WORK_DIR = "/kaggle/working/cultural_alignment"


def on_kaggle() -> bool:
    return os.path.isdir("/kaggle/working")


def bootstrap_offline() -> str:
    """Set offline env + ensure repo is at :data:`WORK_DIR`. Returns repo path.

    Order of resolution:
      1. cwd already contains src/controller.py -> use cwd (local dev).
      2. /kaggle/working/cultural_alignment already populated -> chdir there.
      3. Copy from the first existing path in PROJECT_DATASET_CANDIDATES.
      4. Raise with a clear message naming all candidates we looked at.
    """
    # --- offline-mode env (no internet on Kaggle competition kernels) ---
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
    os.environ.setdefault("UNSLOTH_DISABLE_AUTO_COMPILE", "1")
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    here = os.getcwd()
    if os.path.isfile(os.path.join(here, "src", "controller.py")):
        if here not in sys.path:
            sys.path.insert(0, here)
        return here

    if not on_kaggle():
        raise RuntimeError(
            "Not on Kaggle and not inside the repo root. Run from a directory "
            "containing src/controller.py, or attach the repo as a Kaggle input."
        )

    if os.path.isfile(os.path.join(WORK_DIR, "src", "controller.py")):
        os.chdir(WORK_DIR)
        if WORK_DIR not in sys.path:
            sys.path.insert(0, WORK_DIR)
        print(f"[SETUP] Working dir exists: {WORK_DIR}")
        return WORK_DIR

    src = next((c for c in PROJECT_DATASET_CANDIDATES if os.path.isdir(c)), None)
    if src is None:
        raise RuntimeError(
            "Project dataset not found. Attach the cultural_alignment "
            "notebook or dataset as a Kaggle input. Searched:\n  - "
            + "\n  - ".join(PROJECT_DATASET_CANDIDATES)
        )
    print(f"[SETUP] Copying project {src} -> {WORK_DIR} ...")
    shutil.copytree(src, WORK_DIR, dirs_exist_ok=True)
    os.chdir(WORK_DIR)
    sys.path.insert(0, WORK_DIR)
    return WORK_DIR


def zip_outputs(
    out_dir: Union[str, os.PathLike],
    archive_name: Optional[str] = None,
) -> Optional[str]:
    """Zip ``out_dir`` to ``/kaggle/working/<archive_name>.zip`` (Kaggle) or
    next to ``out_dir`` (local). Skips ``__pycache__`` and any ``_logit_cache``.
    Returns the absolute archive path, or None if nothing was zipped.
    """
    out = Path(out_dir)
    if not out.is_dir():
        print(f"[ZIP] skipped (not a directory): {out}")
        return None
    name = archive_name or out.name or "outputs"
    archive_dir = Path("/kaggle/working") if on_kaggle() else out.parent
    archive = archive_dir / f"{name}.zip"

    skip_dirs = {"__pycache__", "_logit_cache"}
    files = []
    for p in out.rglob("*"):
        if p.is_file() and not (set(p.parts) & skip_dirs):
            files.append((p, str(p.relative_to(out))))
    if not files:
        print(f"[ZIP] skipped (empty): {out}")
        return None

    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for src_path, arcname in files:
            zf.write(src_path, arcname)

    size_mb = archive.stat().st_size / (1024 * 1024)
    print(f"[ZIP] {len(files)} files -> {archive} ({size_mb:.1f} MB)")
    return str(archive)


__all__ = ["bootstrap_offline", "zip_outputs", "on_kaggle", "WORK_DIR"]
