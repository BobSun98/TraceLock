from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download


DREAM_REPO_ID = "Dream-org/Dream-v0-Instruct-7B"
JUDGE_REPO_ID = "Qwen/Qwen2.5-7B-Instruct"
KODCODE_REPO_ID = "BOB12311/kodcode-humaneval-like"
AE_REPO_ID = "BOB12311/tracelock-dream-ae"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download TraceLock public assets into a workspace.")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--dream-repo-id", type=str, default=DREAM_REPO_ID)
    parser.add_argument("--judge-repo-id", type=str, default=JUDGE_REPO_ID)
    parser.add_argument("--kodcode-repo-id", type=str, default=KODCODE_REPO_ID)
    parser.add_argument("--ae-repo-id", type=str, default=os.environ.get("TRACELOCK_AE_REPO_ID", AE_REPO_ID))
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    hf_cache = workspace / "hf_cache"
    models_dir = workspace / "models"
    checkpoints_dir = workspace / "checkpoints"

    os.environ.setdefault("HF_HOME", str(hf_cache))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_cache / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_cache / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache / "hub"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    dream_dir = models_dir / "dream-v0-instruct-7b"
    snapshot_download(
        repo_id=args.dream_repo_id,
        local_dir=dream_dir,
        local_dir_use_symlinks=False,
        local_files_only=args.local_files_only,
    )

    if args.judge_repo_id:
        snapshot_download(
            repo_id=args.judge_repo_id,
            local_files_only=args.local_files_only,
        )

    load_dataset("openai/gsm8k", "main", cache_dir=str(hf_cache / "datasets"))
    load_dataset("yahma/alpaca-cleaned", cache_dir=str(hf_cache / "datasets"))
    load_dataset("openai/openai_humaneval", cache_dir=str(hf_cache / "datasets"))
    load_dataset(args.kodcode_repo_id, cache_dir=str(hf_cache / "datasets"))

    if args.ae_repo_id:
        snapshot_download(
            repo_id=args.ae_repo_id,
            local_dir=checkpoints_dir / "dream-ae-v1",
            local_dir_use_symlinks=False,
            local_files_only=args.local_files_only,
        )
    else:
        print(
            "TRACELOCK_AE_REPO_ID is not set; skipped projection autoencoder download. "
            "Place best_val_loss.pt under $TRACELOCK_HOME/checkpoints/dream-ae-v1 manually."
        )


if __name__ == "__main__":
    main()
