"""Helper: generate one Kaggle-ready exp file per model in exp_model/."""
import os, textwrap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = [
    # (file_suffix,    model_short,     hf_model_name,                                       comment)
    ("qwen25_7b",    "qwen25_7b",    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",          "Qwen2.5-7B-Instruct (4-bit)"),
    ("qwen2_7b",     "qwen2_7b",     "unsloth/Qwen2-7B-Instruct-bnb-4bit",            "Qwen2-7B-Instruct (4-bit)"),
    ("gemma2_9b",    "gemma2_9b",    "unsloth/gemma-2-9b-it-bnb-4bit",                "Gemma-2-9B-IT (4-bit)"),
    ("gemma_7b",     "gemma_7b",     "unsloth/gemma-7b-it-bnb-4bit",                  "Gemma-7B-IT (4-bit)"),
    ("mistral_v03",  "mistral_v03",  "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",     "Mistral-7B-Instruct-v0.3 (4-bit)"),
    ("mistral_v02",  "mistral_v02",  "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",     "Mistral-7B-Instruct-v0.2 (4-bit)"),
    ("llama31_8b",   "llama31_8b",   "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",  "Llama-3.1-8B-Instruct (4-bit)"),
    ("llama3_8b",    "llama3_8b",    "unsloth/llama-3-8b-Instruct-bnb-4bit",          "Llama-3-8B-Instruct (4-bit)"),
    ("internlm25_7b","internlm25_7b","unsloth/internlm2_5-7b-chat-bnb-4bit",          "InternLM2.5-7B-Chat (4-bit)"),
    ("yi15_9b",      "yi15_9b",      "unsloth/Yi-1.5-9B-Chat-bnb-4bit",              "Yi-1.5-9B-Chat (4-bit)"),
]


def _make_file(suffix, short, model_name, comment):
    sep = "=" * (len("EXP-24 Dual-Pass Bootstrap IS — ") + len(comment))
    content = textwrap.dedent(f"""\
        #!/usr/bin/env python3
        \"\"\"
        EXP-24 Dual-Pass Bootstrap IS — {comment}
        {sep}

        Model  : {model_name}
        Method : Dual-Pass Bootstrap IS Reliability (DPBR) — identical to EXP-24
        Base   : EXP-09 Hierarchical IS  (SOTA MIS=0.3975)

        Usage on Kaggle
        ---------------
            !python exp_model/exp_{suffix}.py
        \"\"\"

        # ============================================================
        # Step 0: env bootstrap  (same pattern as experiment_DM/*.py)
        # ============================================================
        import os, sys, subprocess

        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
        os.environ.setdefault("UNSLOTH_DISABLE_AUTO_COMPILE", "1")

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
            for cmd in [
                "pip install -q bitsandbytes scipy tqdm",
                "pip install --upgrade --no-deps unsloth",
                "pip install -q unsloth_zoo",
                "pip install --quiet 'datasets>=3.4.1,<4.4.0'",
            ]:
                subprocess.run(cmd, shell=True, check=False)


        _ensure_repo()
        _install_deps()

        # ============================================================
        # Step 1: model config
        # ============================================================
        MODEL_NAME  = "{model_name}"
        MODEL_SHORT = "{short}"

        # ============================================================
        # Step 2: run (all EXP-24 logic lives in the shared base)
        # ============================================================
        from exp_model._base_dpbr import run_for_model  # noqa: E402

        if __name__ == "__main__":
            run_for_model(MODEL_NAME, MODEL_SHORT)
    """)
    path = os.path.join(BASE_DIR, f"exp_{suffix}.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  created: exp_{suffix}.py  <- {model_name}")


if __name__ == "__main__":
    for args in MODELS:
        _make_file(*args)
    print(f"\nDone — {len(MODELS)} files written to {BASE_DIR}")
