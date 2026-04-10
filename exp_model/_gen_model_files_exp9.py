"""Helper: generate one Kaggle-ready EXP-09 file per model in exp_model/exp9."""

import os
import textwrap

from _gen_model_files import INSTALL_PROFILES, MODELS

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp9")
os.makedirs(BASE_DIR, exist_ok=True)


def _pip_install_list_literal(cmds: list) -> str:
    return "[\n" + "".join(f"                {repr(c)},\n" for c in cmds) + "            ]"


def _make_file(suffix: str, short: str, model_name: str, comment: str, profile: str) -> None:
    if profile not in INSTALL_PROFILES:
        raise KeyError(f"Unknown install profile {profile!r}")
    sep = "=" * (len("EXP-09 Hierarchical IS — ") + len(comment))
    install_list_py = _pip_install_list_literal(INSTALL_PROFILES[profile])
    content = textwrap.dedent(f'''\
        #!/usr/bin/env python3
        """
        EXP-09 Hierarchical IS — {comment}
        {sep}

        Model  : {model_name}
        Profile: {profile}  (pip stack aligned with Reference_Notebook_Model where noted)
        Method : Hierarchical IS with Country-Level Prior — identical to EXP-09

        Usage on Kaggle
        ---------------
            !python exp_model/exp9/exp_{suffix}.py

        Note: ref_* profiles pin transformers; use a fresh Kaggle session when switching families
        (e.g. Phi/Llama 4.56.x vs Qwen3.5 5.2.x vs ref_git_tf55/ref_gemma4 5.5.x).
        """

        # ============================================================
        # Step 0: env bootstrap  (same pattern as experiment_DM/*.py)
        # ============================================================
        import os, sys, subprocess

        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
        os.environ.setdefault("UNSLOTH_DISABLE_AUTO_COMPILE", "1")
        os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")

        REPO_URL        = "https://github.com/trungkiet2005/cultural_alignment.git"
        REPO_DIR_KAGGLE = "/kaggle/working/cultural_alignment"


        def _on_kaggle() -> bool:
            return os.path.isdir("/kaggle/working")


        def _ensure_repo() -> str:
            here = os.getcwd()
            if os.path.isfile(os.path.join(here, "src", "controller.py")):
                return here
            if not _on_kaggle():
                raise RuntimeError("Not on Kaggle and not inside the repo root.")
            if not os.path.isdir(REPO_DIR_KAGGLE):
                subprocess.run(
                    ["git", "clone", "--depth", "1", REPO_URL, REPO_DIR_KAGGLE], check=True
                )
            os.chdir(REPO_DIR_KAGGLE)
            sys.path.insert(0, REPO_DIR_KAGGLE)
            return REPO_DIR_KAGGLE


        def _install_deps() -> None:
            if not _on_kaggle():
                return
            for cmd in {install_list_py}:
                subprocess.run(cmd, shell=True, check=False)


        _ensure_repo()
        _install_deps()

        # ============================================================
        # Step 1: model config
        # ============================================================
        MODEL_NAME  = "{model_name}"
        MODEL_SHORT = "{short}"

        # ============================================================
        # Step 2: run (all EXP-09 logic lives in the shared base)
        # ============================================================
        from exp_model._base_exp09 import run_for_model  # noqa: E402

        if __name__ == "__main__":
            run_for_model(MODEL_NAME, MODEL_SHORT)
    ''')
    path = os.path.join(BASE_DIR, f"exp_{suffix}.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  created: exp9/exp_{suffix}.py  <- {model_name}  [{profile}]")


if __name__ == "__main__":
    for row in MODELS:
        _make_file(*row)
    print(f"\nDone — {len(MODELS)} EXP-09 files written to {BASE_DIR}")
