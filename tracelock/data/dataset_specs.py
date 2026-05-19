from __future__ import annotations

from dataclasses import dataclass
from typing import Any


GSM8K_COT_SUFFIX = (
    "\n\n"
    "Solve this math word problem step by step.\n"
    "Show your reasoning clearly.\n"
    "At the end, output the final answer on a new line in the format:\n"
    "#### <answer>"
)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    dataset_name: str
    subset: str | None
    preferred_splits: tuple[str, ...]


DATASET_SPECS: dict[str, DatasetSpec] = {
    "gsm8k": DatasetSpec(
        key="gsm8k",
        dataset_name="openai/gsm8k",
        subset="main",
        preferred_splits=("train",),
    ),
    "alpaca_cleaned": DatasetSpec(
        key="alpaca_cleaned",
        dataset_name="yahma/alpaca-cleaned",
        subset=None,
        preferred_splits=("train",),
    ),
    "kodcode_humaneval_like": DatasetSpec(
        key="kodcode_humaneval_like",
        dataset_name="BOB12311/kodcode-humaneval-like",
        subset=None,
        preferred_splits=("train",),
    ),
}


def normalize_gsm8k_question(item: dict[str, Any]) -> str:
    question = str(item.get("question", "")).strip()
    return f"{question}{GSM8K_COT_SUFFIX}"


def normalize_alpaca_instruction(item: dict[str, Any]) -> str:
    instruction = str(item.get("instruction", "")).strip()
    input_text = str(item.get("input", "")).strip()
    if input_text:
        return f"{instruction}\n\nInput:\n{input_text}".strip()
    return instruction


def extract_prompt_text(dataset_key: str, item: dict[str, Any]) -> str:
    if dataset_key == "gsm8k":
        return normalize_gsm8k_question(item)
    if dataset_key == "alpaca_cleaned":
        return normalize_alpaca_instruction(item)
    if dataset_key == "kodcode_humaneval_like":
        from tracelock.common.kodcode_humaneval_like import build_kodcode_prompt

        return build_kodcode_prompt(item)
    raise KeyError(f"Unsupported dataset key: {dataset_key}")

