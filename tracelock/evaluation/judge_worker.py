from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tracelock.evaluation.judge import (
    JudgeTask,
    TASK_TYPE_ACC,
    TASK_TYPE_SCORE,
    extract_json_object,
    validate_acc_extract_output,
    validate_acc_output,
    validate_qa_critic_output,
    validate_rank_output,
    validate_score_output,
)
from tracelock.evaluation.prompts import (
    build_acc_extract_final_answer_messages,
    build_acc_final_answer_compare_messages,
    build_acc_messages,
    build_acc_reasoning_review_messages,
    build_qa_lenient_critic_messages,
    build_rank_messages,
    build_score_messages,
)


def _resolve_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    if not hasattr(torch, name):
        raise ValueError(f"Unsupported torch dtype: {name}")
    dtype = getattr(torch, name)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported torch dtype: {name}")
    return dtype


def _build_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    parts: list[str] = []
    for message in messages:
        role = str(message["role"]).upper()
        content = str(message["content"])
        parts.append(f"{role}:\n{content}")
    parts.append("ASSISTANT:\n")
    return "\n\n".join(parts)


class JudgeWorkerRunner:
    def __init__(
        self,
        *,
        model_name: str,
        torch_dtype_name: str,
        temperature: float,
        max_input_length: int,
        max_new_tokens: int,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens

        dtype = _resolve_dtype(torch_dtype_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

    def _messages_for_task(self, task: JudgeTask) -> list[dict[str, str]]:
        if task.task_type == TASK_TYPE_ACC:
            if task.acc_stage == "extract_final_answer":
                return build_acc_extract_final_answer_messages(
                    question=task.prompt_text,
                    candidate_answer=task.candidate_answer or "",
                )
            if task.acc_stage == "compare_final_answer":
                return build_acc_final_answer_compare_messages(
                    question=task.prompt_text,
                    reference_answer=task.reference_answer or "",
                    candidate_final_answer=task.candidate_final_answer or "",
                )
            if task.acc_stage == "review_reasoning":
                return build_acc_reasoning_review_messages(
                    question=task.prompt_text,
                    reference_answer=task.reference_answer or "",
                    candidate_answer=task.candidate_answer or "",
                )
            if task.acc_stage == "qa_lenient_critic":
                return build_qa_lenient_critic_messages(
                    question=task.prompt_text,
                    reference_answer=task.reference_answer or "",
                    candidate_answer=task.candidate_answer or "",
                )
            return build_acc_messages(
                question=task.prompt_text,
                reference_answer=task.reference_answer or "",
                reference_final_answer=task.reference_final_answer,
                candidate_answer=task.candidate_answer or "",
            )
        if task.task_type == TASK_TYPE_SCORE:
            return build_score_messages(
                question=task.prompt_text,
                candidate_map=task.candidates or {},
            )
        return build_rank_messages(
            question=task.prompt_text,
            candidate_map=task.candidates or {},
        )

    @torch.no_grad()
    def run_batch(self, tasks: list[JudgeTask]) -> list[dict[str, Any]]:
        batch_started = time.perf_counter()
        prompts = [_build_prompt(self.tokenizer, self._messages_for_task(task)) for task in tasks]
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
        )
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0.0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature > 0.0:
            generate_kwargs["temperature"] = self.temperature

        outputs = self.model.generate(
            **encoded,
            **generate_kwargs,
        )

        decoded_outputs: list[str] = []
        for idx, sequence in enumerate(outputs):
            generated_tokens = sequence[int(prompt_lengths[idx]) :]
            decoded_outputs.append(self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip())

        batch_elapsed_sec = time.perf_counter() - batch_started
        results: list[dict[str, Any]] = []
        for task, raw_output in zip(tasks, decoded_outputs, strict=True):
            try:
                parsed = extract_json_object(raw_output)
                if task.task_type == TASK_TYPE_ACC:
                    if task.acc_stage == "extract_final_answer":
                        validated = validate_acc_extract_output(parsed)
                    elif task.acc_stage == "qa_lenient_critic":
                        validated = validate_qa_critic_output(parsed)
                    else:
                        validated = validate_acc_output(parsed)
                elif task.task_type == TASK_TYPE_SCORE:
                    validated = validate_score_output(parsed, set((task.candidates or {}).keys()))
                else:
                    validated = validate_rank_output(parsed, set((task.candidates or {}).keys()))
                results.append(
                    {
                        "status": "ok",
                        "task": task.to_payload(),
                        "parsed": validated,
                        "raw_output": raw_output,
                        "error": None,
                        "elapsed_sec": batch_elapsed_sec / max(len(tasks), 1),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                results.append(
                    {
                        "status": "error",
                        "task": task.to_payload(),
                        "parsed": None,
                        "raw_output": raw_output,
                        "error": f"{type(exc).__name__}: {exc}",
                        "elapsed_sec": batch_elapsed_sec / max(len(tasks), 1),
                    }
                )
        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM judge worker.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--torch-dtype", type=str, default="bfloat16")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-input-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--serve", action="store_true")
    return parser.parse_args()


def _build_error_results(items: list[dict[str, Any]], error: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in items:
        results.append(
            {
                "status": "error",
                "task": dict(item),
                "parsed": None,
                "raw_output": "",
                "error": error,
                "elapsed_sec": None,
            }
        )
    return results


def serve_loop(runner: JudgeWorkerRunner) -> None:
    for line in sys.stdin:
        payload_text = line.strip()
        if not payload_text:
            continue
        payload = json.loads(payload_text)
        if payload.get("cmd") == "shutdown":
            return

        items = payload.get("items", [])
        try:
            tasks = [JudgeTask.from_payload(item) for item in items]
            results = runner.run_batch(tasks)
            response = {"results": results}
        except Exception as exc:  # noqa: BLE001
            response = {"results": _build_error_results(items, f"{type(exc).__name__}: {exc}")}
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def main() -> None:
    args = parse_args()
    runner = JudgeWorkerRunner(
        model_name=args.model_name,
        torch_dtype_name=args.torch_dtype,
        temperature=args.temperature,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
    )
    if args.serve:
        serve_loop(runner)
        return

    if args.input is None or args.output is None:
        raise ValueError("--input and --output are required unless --serve is used.")
    payload = json.loads(args.input.read_text())
    tasks = [JudgeTask.from_payload(item) for item in payload["items"]]
    results = runner.run_batch(tasks)
    args.output.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
