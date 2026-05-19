from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from tracelock.common.io_utils import load_jsonl
from tracelock.common.project import PROJECT_ROOT


KODCODE_HUMANEVAL_ROOT = PROJECT_ROOT / "datasets" / "KodCodeHumanEvalLike"
KODCODE_HUMANEVAL_DATA_ROOT = KODCODE_HUMANEVAL_ROOT / "data"
KODCODE_HUMANEVAL_SPLITS = ("train", "use_with_caution")
DEFAULT_KODCODE_HF_REPO_ID = "BOB12311/kodcode-humaneval-like"


def get_kodcode_available_splits() -> list[str]:
    return [split for split in KODCODE_HUMANEVAL_SPLITS if (KODCODE_HUMANEVAL_DATA_ROOT / f"{split}.jsonl").exists()]


def get_kodcode_split_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in get_kodcode_available_splits():
        counts[split] = len(load_jsonl(KODCODE_HUMANEVAL_DATA_ROOT / f"{split}.jsonl"))
    return counts


def load_kodcode_split(split: str) -> Dataset:
    source = os.environ.get("KODCODE_HUMANEVAL_SOURCE", "local").strip().lower()
    if source == "hf":
        repo_id = os.environ.get("KODCODE_HUMANEVAL_REPO_ID", DEFAULT_KODCODE_HF_REPO_ID)
        return load_dataset(repo_id, split=split)

    path = KODCODE_HUMANEVAL_DATA_ROOT / f"{split}.jsonl"
    if not path.exists():
        repo_id = os.environ.get("KODCODE_HUMANEVAL_REPO_ID", DEFAULT_KODCODE_HF_REPO_ID)
        try:
            return load_dataset(repo_id, split=split)
        except Exception as exc:
            raise FileNotFoundError(
                f"KodCodeHumanEvalLike split not found locally at {path}; "
                f"also failed to load Hugging Face dataset {repo_id!r} split {split!r}."
            ) from exc
    return Dataset.from_list(load_jsonl(path))


def build_code_generation_prompt(
    *,
    prompt: str,
    test: str,
    entry_point: str | None,
) -> str:
    prompt_text = str(prompt).rstrip()
    test_text = str(test).rstrip()
    entry_point_text = str(entry_point or "").strip()

    requirements = [
        f"- Implement the function `{entry_point_text}` exactly as specified." if entry_point_text else None,
        "- Return only Python code.",
        "- Put the final code inside one ```python``` block.",
        "- Do not include explanations or extra text.",
    ]
    requirement_lines = "\n".join(line for line in requirements if line)

    return (
        "Write Python code that passes the given tests.\n\n"
        "Problem:\n"
        f"{prompt_text}\n\n"
        "Unit tests:\n"
        f"{test_text}\n\n"
        "Requirements:\n"
        f"{requirement_lines}\n"
    )


def build_kodcode_prompt(item: dict[str, Any]) -> str:
    return build_code_generation_prompt(
        prompt=str(item.get("prompt", "")),
        test=str(item.get("test", "")),
        entry_point=item.get("entry_point"),
    )
