from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import queue
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracelock.common.code_eval import verify_code_answer  # noqa: E402
from tracelock.common.io_utils import ensure_dir, load_json, write_json  # noqa: E402
from tracelock.common.kodcode_humaneval_like import build_code_generation_prompt  # noqa: E402
from tracelock.dream.generation import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    GenerationTask,
    GenerationWorkerRunner,
    SetConfig,
    _parse_set_config,
    iter_all_sets,
)


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "eval_humaneval.json"
SUPPORTED_DATASETS = {"openai/openai_humaneval"}


@dataclass(frozen=True)
class CodeExperimentConfig:
    config_path: Path
    experiment_name: str | None
    output_root: Path
    gpus: tuple[str, ...]
    dataset: str
    split: str
    seed: int
    num_samples: int
    dlm_model_name: str
    torch_dtype_name: str
    overwrite: bool
    local_files_only: bool
    python_executable: str
    timeout_seconds: float
    sets: tuple[SetConfig, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Dream config-driven code evaluation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_code_experiment_config(path: Path) -> CodeExperimentConfig:
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text())
    base_dir = config_path.parent

    gpus = tuple(str(item) for item in raw["gpus"])
    if not gpus:
        raise ValueError("config must define at least one GPU.")
    for gpu in gpus:
        if not gpu.startswith("cuda:"):
            raise ValueError(f"GPU entries must look like 'cuda:N', got {gpu!r}")

    dataset = str(raw["dataset"])
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")

    output_root = Path(raw.get("output_root", DEFAULT_OUTPUT_ROOT))
    if not output_root.is_absolute():
        output_root = (base_dir / output_root).resolve()

    sets = tuple(_parse_set_config(item, base_dir=base_dir) for item in raw["sets"])
    all_sets = iter_all_sets(sets)
    names = [item.name for item in all_sets]
    if len(names) != len(set(names)):
        raise ValueError("All set names, including subsets, must be globally unique.")

    return CodeExperimentConfig(
        config_path=config_path,
        experiment_name=str(raw.get("experiment_name")) if raw.get("experiment_name") is not None else None,
        output_root=output_root,
        gpus=gpus,
        dataset=dataset,
        split=str(raw.get("split", "test")),
        seed=int(raw.get("seed", 42)),
        num_samples=int(raw["num_samples"]),
        dlm_model_name=str(raw.get("dlm_model_name", "Dream-org/Dream-v0-Instruct-7B")),
        torch_dtype_name=str(raw.get("torch_dtype", "bfloat16")),
        overwrite=bool(raw.get("overwrite", False)),
        local_files_only=bool(raw.get("local_files_only", False)),
        python_executable=str(raw.get("python_executable", sys.executable)),
        timeout_seconds=float(raw.get("timeout_seconds", 30.0)),
        sets=sets,
    )


def resolve_run_name(experiment: CodeExperimentConfig, cli_run_name: str | None) -> str:
    if cli_run_name:
        return cli_run_name
    if experiment.experiment_name:
        return experiment.experiment_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{experiment.config_path.stem}_{timestamp}"


def load_dataset_split(experiment: CodeExperimentConfig):
    return load_dataset(experiment.dataset, split=experiment.split)


def build_code_eval_item(dataset_name: str, sample_idx: int, record: dict[str, Any]) -> dict[str, Any]:
    prompt = build_code_generation_prompt(
        prompt=str(record.get("prompt", "")),
        test=str(record.get("test", "")),
        entry_point=record.get("entry_point"),
    )
    return {
        "sample_id": f"openai_humaneval_{sample_idx:08d}",
        "dataset": dataset_name,
        "split": None,
        "dataset_index": int(sample_idx),
        "task_id": str(record.get("task_id", sample_idx)),
        "prompt_text": prompt,
        "source_record": {
            key: value
            for key, value in record.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        },
        "test": str(record.get("test", "")),
        "entry_point": str(record.get("entry_point", "")),
    }


def load_eval_items(experiment: CodeExperimentConfig) -> list[dict[str, Any]]:
    dataset = load_dataset_split(experiment)
    indices = list(range(len(dataset)))
    rng = random.Random(experiment.seed)
    rng.shuffle(indices)
    selected_indices = indices[: experiment.num_samples]
    return [build_code_eval_item(experiment.dataset, int(sample_idx), dataset[int(sample_idx)]) for sample_idx in selected_indices]


def make_task(set_config: SetConfig, item: dict[str, Any], derived_from_steps: int | None = None) -> GenerationTask:
    return GenerationTask(
        set_name=set_config.name,
        parent_set_name=set_config.parent_name,
        question_index=int(item["dataset_index"]),
        sample_id=str(item["sample_id"]),
        dataset=str(item["dataset"]),
        split=str(item.get("split") or ""),
        prompt_text=str(item["prompt_text"]),
        source_record=dict(item["source_record"]),
        gen_method=set_config.gen_method,
        gen_length=int(set_config.gen_length),
        candidate_num=set_config.candidate_num,
        temperature=float(set_config.temperature),
        cfg_scale=float(set_config.cfg_scale),
        gen_steps=set_config.gen_steps,
        block_len=set_config.block_len,
        threshold=set_config.threshold,
        pointer_window_size=set_config.pointer_window_size,
        tracelock_schedule=set_config.tracelock_schedule,
        dynamic_threshold=bool(set_config.dynamic_threshold),
        tracelock_threshold=set_config.tracelock_threshold,
        tracelock_checkpoint=str(set_config.tracelock_checkpoint) if set_config.tracelock_checkpoint else None,
        tracelock_config=str(set_config.tracelock_config) if set_config.tracelock_config else None,
        projection_checkpoint=str(set_config.projection_checkpoint) if set_config.projection_checkpoint else None,
        transfer_score_source=set_config.transfer_score_source,
        derived_from_steps=derived_from_steps,
    )


def worker_main(config_path: str, gpu: str, task_queue: mp.Queue, result_queue: mp.Queue) -> None:
    experiment = load_code_experiment_config(Path(config_path))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu.split(":", 1)[1]
    runner = GenerationWorkerRunner(experiment, "cuda")
    while True:
        payload = task_queue.get()
        if payload is None:
            return
        task = GenerationTask.from_payload(payload)
        try:
            generation_result = runner.run_task(task)
            result_queue.put(
                {
                    "status": "ok",
                    "task": task.to_payload(),
                    "answer": generation_result["answer"],
                    "elapsed_sec": generation_result["elapsed_sec"],
                    "executed_steps": generation_result["executed_steps"],
                    "final_remaining_masks": generation_result["final_remaining_masks"],
                    "error": None,
                }
            )
        except Exception as exc:  # noqa: BLE001
            result_queue.put(
                {
                    "status": "error",
                    "task": task.to_payload(),
                    "answer": "",
                    "elapsed_sec": None,
                    "executed_steps": None,
                    "final_remaining_masks": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )


def build_result_record(
    *,
    experiment: CodeExperimentConfig,
    item_by_sample_id: dict[str, dict[str, Any]],
    result_payload: dict[str, Any],
) -> dict[str, Any]:
    task = GenerationTask.from_payload(result_payload["task"])
    item = item_by_sample_id[task.sample_id]
    generation_error = result_payload["error"]
    verification = None
    if generation_error is None:
        verification = verify_code_answer(
            answer=result_payload["answer"],
            test=item["test"],
            entry_point=item["entry_point"],
            timeout_seconds=experiment.timeout_seconds,
            python_executable=experiment.python_executable,
        )
    return {
        "set_name": task.set_name,
        "question_index": task.question_index,
        "sample_id": task.sample_id,
        "dataset": task.dataset,
        "task_id": item["task_id"],
        "prompt_text": task.prompt_text,
        "test": item["test"],
        "entry_point": item["entry_point"],
        "gen_method": task.gen_method,
        "gen_length": task.gen_length,
        "requested_gen_steps": task.gen_steps,
        "block_len": task.block_len,
        "threshold": task.threshold,
        "pointer_window_size": task.pointer_window_size,
        "tracelock_schedule": task.tracelock_schedule,
        "dynamic_threshold": task.dynamic_threshold,
        "tracelock_threshold": task.tracelock_threshold,
        "answer": result_payload["answer"],
        "elapsed_sec": result_payload["elapsed_sec"],
        "executed_steps": result_payload["executed_steps"],
        "final_remaining_masks": result_payload["final_remaining_masks"],
        "generation_error": generation_error,
        "verification": verification,
        "passed": bool(verification and verification["passed"]),
        "error_type": (
            "generation" if generation_error is not None else str(verification["error_type"])
        ) if (generation_error is not None or verification is not None) else "unknown",
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    passed = sum(1 for record in records if record.get("passed"))
    generation_errors = sum(1 for record in records if record.get("error_type") == "generation")
    syntax_failures = sum(1 for record in records if record.get("error_type") == "syntax")
    runtime_failures = sum(1 for record in records if record.get("error_type") == "runtime")
    timeout_failures = sum(1 for record in records if record.get("error_type") == "timeout")
    no_code_failures = sum(1 for record in records if record.get("error_type") == "no_code")
    executed_steps = [record["executed_steps"] for record in records if record.get("executed_steps") is not None]
    final_remaining_masks = [record["final_remaining_masks"] for record in records if record.get("final_remaining_masks") is not None]
    return {
        "num_samples": total,
        "num_passed": passed,
        "pass_at_1": (passed / total) if total else 0.0,
        "generation_errors": generation_errors,
        "syntax_failures": syntax_failures,
        "runtime_failures": runtime_failures,
        "timeout_failures": timeout_failures,
        "no_code_failures": no_code_failures,
        "mean_executed_steps": (sum(executed_steps) / len(executed_steps)) if executed_steps else None,
        "mean_final_remaining_masks": (sum(final_remaining_masks) / len(final_remaining_masks)) if final_remaining_masks else None,
    }


def main() -> None:
    args = parse_args()
    experiment = load_code_experiment_config(args.config)
    run_name = resolve_run_name(experiment, args.run_name)
    run_dir = ensure_dir(experiment.output_root / run_name)
    all_sets = iter_all_sets(experiment.sets)

    if args.dry_run:
        print(
            f"[dry-run] run_dir={run_dir} dataset={experiment.dataset} split={experiment.split} "
            f"num_samples={experiment.num_samples} sets={[set_config.name for set_config in all_sets]}",
            flush=True,
        )
        return

    items = load_eval_items(experiment)
    item_by_sample_id = {item["sample_id"]: item for item in items}

    tasks: list[GenerationTask] = []
    for set_config in all_sets:
        set_dir = ensure_dir(run_dir / set_config.name)
        for item in items:
            result_path = set_dir / f"{int(item['dataset_index']):08d}.json"
            if result_path.exists() and not experiment.overwrite:
                continue
            tasks.append(make_task(set_config, item))

    write_json(
        run_dir / "config.json",
        {
            "experiment_name": experiment.experiment_name,
            "dataset": experiment.dataset,
            "split": experiment.split,
            "seed": experiment.seed,
            "num_samples": experiment.num_samples,
            "dlm_model_name": experiment.dlm_model_name,
            "torch_dtype": experiment.torch_dtype_name,
            "overwrite": experiment.overwrite,
            "local_files_only": experiment.local_files_only,
            "python_executable": experiment.python_executable,
            "timeout_seconds": experiment.timeout_seconds,
            "gpus": list(experiment.gpus),
            "sets": [set_config.name for set_config in all_sets],
        },
    )

    if tasks:
        mp_ctx = mp.get_context("spawn")
        task_queue: mp.Queue = mp_ctx.Queue()
        result_queue: mp.Queue = mp_ctx.Queue()
        for task in tasks:
            task_queue.put(task.to_payload())
        for _ in experiment.gpus:
            task_queue.put(None)
        processes = [
            mp_ctx.Process(
                target=worker_main,
                args=(str(experiment.config_path), gpu, task_queue, result_queue),
                name=f"dream-code-eval-worker-{idx:02d}",
            )
            for idx, gpu in enumerate(experiment.gpus)
        ]
        for process in processes:
            process.start()

        progress = tqdm(total=len(tasks), desc="dream-code-eval", unit="task")
        processed = 0
        try:
            while processed < len(tasks):
                try:
                    result_payload = result_queue.get(timeout=30.0)
                except queue.Empty:
                    failed = [process for process in processes if process.exitcode not in (None, 0)]
                    if failed:
                        details = ", ".join(f"{process.name}: exitcode={process.exitcode}" for process in failed)
                        raise RuntimeError(f"Code eval worker exited before returning all results: {details}")
                    continue
                record = build_result_record(
                    experiment=experiment,
                    item_by_sample_id=item_by_sample_id,
                    result_payload=result_payload,
                )
                set_dir = ensure_dir(run_dir / record["set_name"])
                write_json(set_dir / f"{record['question_index']:08d}.json", record)
                processed += 1
                progress.update(1)
        finally:
            progress.close()
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join()
        gc.collect()

    summary: dict[str, Any] = {}
    for set_config in all_sets:
        set_dir = run_dir / set_config.name
        records = [load_json(path) for path in sorted(set_dir.glob("*.json"))]
        summary[set_config.name] = summarize_records(records)
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
