from __future__ import annotations

import argparse
import csv
import gc
import json
import multiprocessing as mp
import os
import queue
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracelock.common.io_utils import append_jsonl, ensure_dir, load_json, write_json  # noqa: E402
from tracelock.data.dataset_specs import DATASET_SPECS, extract_prompt_text  # noqa: E402
from tracelock.dream.generation import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    GenerationTask,
    GenerationWorkerRunner,
    SetConfig,
    _parse_set_config,
    build_children_map,
    build_set_map,
    iter_all_sets,
    question_filename,
)
from tracelock.evaluation.judge import (  # noqa: E402
    JudgeTask,
    SUPPORTED_TASK_TYPES,
    TASK_TYPE_ACC,
    TASK_TYPE_RANK,
    TASK_TYPE_SCORE,
    compute_rank_map,
    derive_score_order,
    extract_gsm8k_final_answer,
    summarize_acc_results,
    summarize_rank_results,
    summarize_score_results,
)
from tracelock.evaluation.judge_subprocess import run_with_subprocess  # noqa: E402


DEFAULT_EXPERIMENT_CONFIG_PATH = PROJECT_ROOT / "configs" / "eval_gsm8k.json"
DEFAULT_OUTPUT_ROOT_LLM = PROJECT_ROOT / "workspace" / "runs" / "eval"
JUDGE_FIELD_NAMES = (
    "judge_status",
    "judge_elapsed_sec",
    "judge_extract_status",
    "judge_extract_reason",
    "judge_candidate_final_answer",
    "judge_final_status",
    "judge_final_verdict",
    "judge_final_confidence",
    "judge_final_reason",
    "judge_reasoning_status",
    "judge_reasoning_verdict",
    "judge_reasoning_confidence",
    "judge_reasoning_reason",
    "judge_verdict",
    "judge_confidence",
    "judge_reason",
    "judge_rank",
    "judge_is_best",
    "judge_score",
    "judge_score_reason",
    "judge_rank_from_score",
    "judge_is_best_from_score",
    "judge_error",
)
ACC_RELAXED_EXTRACT_MARKER = "dream_relaxed_extract_v1"
ACC_RELAXED_EXTRACT_SUFFIX = (
    "\n\n[Dream evaluation extraction override]\n"
    "When extracting the candidate's final answer for this run, be lenient about output format. "
    "The candidate may omit the requested '#### <answer>' line. If the candidate response contains "
    "a final or last completed numeric result that can reasonably be inferred from the response, "
    "treat it as found and return that numeric result. Do not mark not_found solely because the "
    "candidate omitted '####'. Mark not_found only when no numeric or quantity answer can be inferred."
)


@dataclass(frozen=True)
class JudgeConfig:
    python: str
    model_name: str
    task_type: str
    acc_final_answer_only: bool
    torch_dtype_name: str
    batch_size: int
    max_input_length: int
    max_new_tokens: int
    temperature: float
    json_mode: bool
    acc_relaxed_extract: bool


@dataclass(frozen=True)
class ExperimentConfig:
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
    sets: tuple[SetConfig, ...]
    judge: JudgeConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Dream experiment.config driven LLM judge evaluation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG_PATH)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--sample-manifest", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve_path(value: str | None, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def load_experiment_config(path: Path = DEFAULT_EXPERIMENT_CONFIG_PATH) -> ExperimentConfig:
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text())
    base_dir = config_path.parent
    gpus = tuple(str(item) for item in raw["gpus"])
    if not gpus:
        raise ValueError("experiment.config must define at least one GPU.")
    for gpu in gpus:
        if not gpu.startswith("cuda:"):
            raise ValueError(f"GPU entries must look like 'cuda:N', got {gpu!r}")

    output_root = _resolve_path(raw.get("output_root"), base_dir) or DEFAULT_OUTPUT_ROOT_LLM
    sets = tuple(_parse_set_config(item, base_dir=base_dir) for item in raw["sets"])
    all_sets = iter_all_sets(sets)
    names = [item.name for item in all_sets]
    if len(names) != len(set(names)):
        raise ValueError("All set names, including subsets, must be globally unique.")
    candidate_nums = [item.candidate_num for item in all_sets if item.candidate_num is not None]
    if len(candidate_nums) != len(set(candidate_nums)):
        raise ValueError("All set candidate_num values must be globally unique when provided.")
    candidate_names = [item.candidate_name for item in all_sets if item.candidate_name is not None]
    if len(candidate_names) != len(set(candidate_names)):
        raise ValueError("All set candidate_name values must be globally unique when provided.")

    judge_raw = raw["judge"]
    task_type = str(judge_raw["task_type"]).strip().lower()
    if task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(f"Unsupported judge.task_type: {task_type!r}")
    if task_type == TASK_TYPE_ACC and str(raw["dataset"]) != "gsm8k":
        raise ValueError("judge.task_type=acc currently only supports dataset=gsm8k.")

    judge = JudgeConfig(
        python=str(judge_raw.get("python", sys.executable)),
        model_name=str(judge_raw["model_name"]),
        task_type=task_type,
        acc_final_answer_only=bool(judge_raw.get("acc_final_answer_only", False)),
        torch_dtype_name=str(judge_raw.get("torch_dtype", "bfloat16")),
        batch_size=int(judge_raw.get("batch_size", 1)),
        max_input_length=int(judge_raw.get("max_input_length", 4096)),
        max_new_tokens=int(judge_raw.get("max_new_tokens", 256)),
        temperature=float(judge_raw.get("temperature", 0.0)),
        json_mode=bool(judge_raw.get("json_mode", True)),
        acc_relaxed_extract=bool(judge_raw.get("acc_relaxed_extract", True)),
    )
    if not judge.json_mode:
        raise ValueError("judge.json_mode must be true in v1.")

    return ExperimentConfig(
        config_path=config_path,
        experiment_name=str(raw.get("experiment_name")) if raw.get("experiment_name") is not None else None,
        output_root=output_root,
        gpus=gpus,
        dataset=str(raw["dataset"]),
        split=str(raw.get("split", "train")),
        seed=int(raw.get("seed", 42)),
        num_samples=int(raw["num_samples"]),
        dlm_model_name=str(raw.get("dlm_model_name", "Dream-org/Dream-v0-Instruct-7B")),
        torch_dtype_name=str(raw.get("torch_dtype", "bfloat16")),
        overwrite=bool(raw.get("overwrite", False)),
        local_files_only=bool(raw.get("local_files_only", False)),
        sets=sets,
        judge=judge,
    )


def resolve_run_name(experiment: ExperimentConfig, cli_run_name: str | None) -> str:
    if cli_run_name:
        return cli_run_name
    if experiment.experiment_name:
        return experiment.experiment_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{experiment.config_path.stem}_{timestamp}"


def prepare_run_dir(experiment: ExperimentConfig, run_name: str) -> Path:
    run_dir = experiment.output_root / run_name
    ensure_dir(run_dir)
    return run_dir


def load_dataset_split(experiment: ExperimentConfig):
    spec = DATASET_SPECS[experiment.dataset]
    if spec.subset is None:
        return load_dataset(spec.dataset_name, split=experiment.split)
    return load_dataset(spec.dataset_name, spec.subset, split=experiment.split)


def build_eval_items_from_indices(experiment: ExperimentConfig, dataset: Any, selected_indices: list[int]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for sample_idx in selected_indices:
        record = dataset[int(sample_idx)]
        prompt_text = extract_prompt_text(experiment.dataset, record)
        items.append(
            {
                "sample_id": f"{experiment.dataset}_{experiment.split}_{sample_idx:08d}",
                "dataset": experiment.dataset,
                "split": experiment.split,
                "dataset_index": int(sample_idx),
                "prompt_text": prompt_text,
                "source_record": {
                    key: value
                    for key, value in record.items()
                    if isinstance(value, (str, int, float, bool)) or value is None
                },
            }
        )
    return items


def load_eval_items(experiment: ExperimentConfig) -> list[dict[str, Any]]:
    dataset = load_dataset_split(experiment)
    indices = list(range(len(dataset)))
    rng = random.Random(experiment.seed)
    rng.shuffle(indices)
    selected_indices = indices[: experiment.num_samples]
    return build_eval_items_from_indices(experiment, dataset, selected_indices)


def load_existing_manifest(run_dir: Path) -> dict[str, Any] | None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    return load_json(manifest_path)


def _build_eval_items_from_manifest(experiment: ExperimentConfig, manifest_payload: dict[str, Any], *, source_desc: str) -> list[dict[str, Any]]:
    manifest_dataset = str(manifest_payload.get("dataset"))
    manifest_split = str(manifest_payload.get("split"))
    if manifest_dataset != experiment.dataset or manifest_split != experiment.split:
        raise ValueError(
            f"Sample manifest at {source_desc} uses dataset={manifest_dataset!r}, split={manifest_split!r}, "
            f"but current config requests dataset={experiment.dataset!r}, split={experiment.split!r}."
        )
    sampled_questions = manifest_payload.get("sampled_questions", [])
    if not sampled_questions:
        raise ValueError(f"Sample manifest at {source_desc} is missing sampled_questions.")
    dataset = load_dataset_split(experiment)
    selected_indices = [int(item["question_index"]) for item in sampled_questions]
    return build_eval_items_from_indices(experiment, dataset, selected_indices)


def load_eval_items_for_run(experiment: ExperimentConfig, run_dir: Path, sample_manifest_path: Path | None = None) -> list[dict[str, Any]]:
    if sample_manifest_path is not None:
        manifest_payload = load_json(sample_manifest_path)
        return _build_eval_items_from_manifest(experiment, manifest_payload, source_desc=str(sample_manifest_path))
    existing_manifest = load_existing_manifest(run_dir)
    if existing_manifest is None:
        return load_eval_items(experiment)
    return _build_eval_items_from_manifest(experiment, existing_manifest, source_desc=str(run_dir))


def make_task(set_config: SetConfig, item: dict[str, Any], derived_from_steps: int | None = None) -> GenerationTask:
    return GenerationTask(
        set_name=set_config.name,
        parent_set_name=set_config.parent_name,
        question_index=int(item["dataset_index"]),
        sample_id=str(item["sample_id"]),
        dataset=str(item["dataset"]),
        split=str(item["split"]),
        prompt_text=str(item["prompt_text"]),
        source_record=dict(item["source_record"]),
        gen_method=set_config.gen_method,
        gen_length=int(set_config.gen_length),
        candidate_num=set_config.candidate_num,
        candidate_name=set_config.candidate_name,
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
    experiment = load_experiment_config(Path(config_path))
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


def build_result_record(result_payload: dict[str, Any]) -> dict[str, Any]:
    task = GenerationTask.from_payload(result_payload["task"])
    return {
        "set_name": task.set_name,
        "parent_set_name": task.parent_set_name,
        "question_index": task.question_index,
        "sample_id": task.sample_id,
        "dataset": task.dataset,
        "split": task.split,
        "prompt_text": task.prompt_text,
        "source_record": task.source_record,
        "gen_method": task.gen_method,
        "gen_length": task.gen_length,
        "candidate_num": task.candidate_num,
        "candidate_name": task.candidate_name,
        "temperature": task.temperature,
        "cfg_scale": task.cfg_scale,
        "requested_gen_steps": task.gen_steps,
        "block_len": task.block_len,
        "threshold": task.threshold,
        "pointer_window_size": task.pointer_window_size,
        "tracelock_schedule": task.tracelock_schedule,
        "dynamic_threshold": task.dynamic_threshold,
        "tracelock_threshold": task.tracelock_threshold,
        "tracelock_checkpoint": task.tracelock_checkpoint,
        "tracelock_config": task.tracelock_config,
        "projection_checkpoint": task.projection_checkpoint,
        "transfer_score_source": task.transfer_score_source,
        "derived_from_steps": task.derived_from_steps,
        "status": result_payload["status"],
        "answer": result_payload["answer"],
        "elapsed_sec": result_payload["elapsed_sec"],
        "executed_steps": result_payload["executed_steps"],
        "final_remaining_masks": result_payload["final_remaining_masks"],
        "judge_status": None,
        "judge_elapsed_sec": None,
        "judge_extract_status": None,
        "judge_extract_reason": None,
        "judge_candidate_final_answer": None,
        "judge_final_status": None,
        "judge_final_verdict": None,
        "judge_final_confidence": None,
        "judge_final_reason": None,
        "judge_reasoning_status": None,
        "judge_reasoning_verdict": None,
        "judge_reasoning_confidence": None,
        "judge_reasoning_reason": None,
        "judge_verdict": None,
        "judge_confidence": None,
        "judge_reason": None,
        "judge_rank": None,
        "judge_is_best": None,
        "judge_score": None,
        "judge_score_reason": None,
        "judge_rank_from_score": None,
        "judge_is_best_from_score": None,
        "judge_error": None,
        "error": result_payload["error"],
    }


def build_skipped_record(set_config: SetConfig, item: dict[str, Any], reason: str, *, derived_from_steps: int | None = None) -> dict[str, Any]:
    return {
        "set_name": set_config.name,
        "parent_set_name": set_config.parent_name,
        "question_index": int(item["dataset_index"]),
        "sample_id": str(item["sample_id"]),
        "dataset": str(item["dataset"]),
        "split": str(item["split"]),
        "prompt_text": str(item["prompt_text"]),
        "source_record": dict(item["source_record"]),
        "gen_method": set_config.gen_method,
        "gen_length": int(set_config.gen_length),
        "candidate_num": set_config.candidate_num,
        "candidate_name": set_config.candidate_name,
        "temperature": float(set_config.temperature),
        "cfg_scale": float(set_config.cfg_scale),
        "requested_gen_steps": set_config.gen_steps,
        "block_len": set_config.block_len,
                    "threshold": set_config.threshold,
                    "pointer_window_size": set_config.pointer_window_size,
                    "tracelock_schedule": set_config.tracelock_schedule,
                    "dynamic_threshold": bool(set_config.dynamic_threshold),
        "tracelock_threshold": set_config.tracelock_threshold,
        "tracelock_checkpoint": str(set_config.tracelock_checkpoint) if set_config.tracelock_checkpoint else None,
        "tracelock_config": str(set_config.tracelock_config) if set_config.tracelock_config else None,
        "projection_checkpoint": str(set_config.projection_checkpoint) if set_config.projection_checkpoint else None,
        "transfer_score_source": set_config.transfer_score_source,
        "derived_from_steps": derived_from_steps,
        "status": "skipped",
        "answer": "",
        "elapsed_sec": None,
        "executed_steps": None,
        "final_remaining_masks": None,
        "judge_status": None,
        "judge_elapsed_sec": None,
        "judge_extract_status": None,
        "judge_extract_reason": None,
        "judge_candidate_final_answer": None,
        "judge_final_status": None,
        "judge_final_verdict": None,
        "judge_final_confidence": None,
        "judge_final_reason": None,
        "judge_reasoning_status": None,
        "judge_reasoning_verdict": None,
        "judge_reasoning_confidence": None,
        "judge_reasoning_reason": None,
        "judge_verdict": None,
        "judge_confidence": None,
        "judge_reason": None,
        "judge_rank": None,
        "judge_is_best": None,
        "judge_score": None,
        "judge_score_reason": None,
        "judge_rank_from_score": None,
        "judge_is_best_from_score": None,
        "judge_error": None,
        "error": reason,
    }


def write_result_file(run_dir: Path, record: dict[str, Any]) -> Path:
    set_dir = ensure_dir(run_dir / record["set_name"])
    out_path = set_dir / question_filename(int(record["question_index"]))
    write_json(out_path, record)
    return out_path


def clear_judge_fields(record: dict[str, Any]) -> None:
    for key in JUDGE_FIELD_NAMES:
        record[key] = None


def reset_all_judge_fields(results: dict[tuple[str, int], dict[str, Any]]) -> None:
    for record in results.values():
        clear_judge_fields(record)


def iter_question_candidate_groups(results: dict[tuple[str, int], dict[str, Any]]) -> list[tuple[int, list[dict[str, Any]]]]:
    groups: list[tuple[int, list[dict[str, Any]]]] = []
    question_ids = sorted({question_index for _set_name, question_index in results})
    for question_index in question_ids:
        records = [record for (_set_name, q_idx), record in results.items() if q_idx == question_index]
        candidate_records = [record for record in records if record["status"] == "ok" and record["answer"]]
        if len(candidate_records) < 2:
            continue
        candidate_records.sort(
            key=lambda item: (
                int(item["candidate_num"]) if item.get("candidate_num") is not None else 10**9,
                str(item["set_name"]),
            )
        )
        groups.append((question_index, candidate_records))
    return groups


def queue_skipped_descendants(
    *,
    run_dir: Path,
    children_map: dict[str, tuple[SetConfig, ...]],
    results: dict[tuple[str, int], dict[str, Any]],
    item: dict[str, Any],
    parent_name: str,
    reason: str,
    derived_from_steps: int | None = None,
) -> None:
    for child in children_map.get(parent_name, ()):
        key = (child.name, int(item["dataset_index"]))
        if key in results:
            continue
        record = build_skipped_record(child, item, reason, derived_from_steps=derived_from_steps)
        results[key] = record
        write_result_file(run_dir, record)
        queue_skipped_descendants(
            run_dir=run_dir,
            children_map=children_map,
            results=results,
            item=item,
            parent_name=child.name,
            reason=reason,
            derived_from_steps=derived_from_steps,
        )


def load_existing_results(run_dir: Path, allowed_set_names: set[str], allowed_question_indices: set[int]) -> dict[tuple[str, int], dict[str, Any]]:
    results: dict[tuple[str, int], dict[str, Any]] = {}
    if not run_dir.exists():
        return results
    for set_name in allowed_set_names:
        set_dir = run_dir / set_name
        if not set_dir.is_dir():
            continue
        for path in sorted(set_dir.glob("*.json")):
            record = load_json(path)
            question_index = int(record["question_index"])
            if question_index not in allowed_question_indices:
                continue
            if record.get("status") != "ok":
                continue
            results[(set_name, question_index)] = record
    return results


def refresh_candidate_metadata_from_config(results: dict[tuple[str, int], dict[str, Any]], set_map: dict[str, SetConfig], run_dir: Path) -> int:
    updated = 0
    for key, record in results.items():
        set_name = str(record["set_name"])
        set_config = set_map.get(set_name)
        if set_config is None:
            continue
        new_candidate_num = set_config.candidate_num
        new_candidate_name = set_config.candidate_name
        old_candidate_num = record.get("candidate_num")
        old_candidate_name = record.get("candidate_name")
        if old_candidate_num == new_candidate_num and old_candidate_name == new_candidate_name:
            continue
        record["candidate_num"] = new_candidate_num
        record["candidate_name"] = new_candidate_name
        write_result_file(run_dir, record)
        updated += 1
    return updated


def _csv_value_to_optional(value: str | None) -> Any:
    if value is None or value == "":
        return None
    return value


def restore_acc_judge_fields_from_results_csv(run_dir: Path, results: dict[tuple[str, int], dict[str, Any]]) -> int:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return 0

    restored = 0
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            set_name = row.get("set_name")
            question_index_raw = row.get("question_index")
            if not set_name or question_index_raw in (None, ""):
                continue
            key = (set_name, int(question_index_raw))
            record = results.get(key)
            if record is None:
                continue
            if str(row.get("answer") or "") != str(record.get("answer") or ""):
                continue

            changed = False
            for field_name in JUDGE_FIELD_NAMES:
                value = _csv_value_to_optional(row.get(field_name))
                if value is None or record.get(field_name) is not None:
                    continue
                record[field_name] = value
                changed = True
            if changed:
                write_result_file(run_dir, record)
                restored += 1
    return restored


def apply_acc_final_answer_only_aliases(run_dir: Path, results: dict[tuple[str, int], dict[str, Any]]) -> int:
    updated = 0
    for record in results.values():
        if record.get("judge_final_status") != "ok" or record.get("judge_final_verdict") is None:
            continue

        changed = False
        alias_map = {
            "judge_status": record.get("judge_final_status"),
            "judge_verdict": record.get("judge_final_verdict"),
            "judge_confidence": record.get("judge_final_confidence"),
            "judge_reason": record.get("judge_final_reason"),
        }
        for key, value in alias_map.items():
            if record.get(key) == value:
                continue
            record[key] = value
            changed = True
        if changed:
            write_result_file(run_dir, record)
            updated += 1
    return updated


def schedule_children_for_parent(
    *,
    parent_record: dict[str, Any],
    item: dict[str, Any],
    task_queue: mp.Queue,
    pending_ref: list[int],
    children_map: dict[str, tuple[SetConfig, ...]],
    results: dict[tuple[str, int], dict[str, Any]],
    run_dir: Path,
) -> None:
    children = children_map.get(parent_record["set_name"], ())
    if not children:
        return
    if parent_record["status"] != "ok" or parent_record["executed_steps"] is None or int(parent_record["executed_steps"]) <= 0:
        queue_skipped_descendants(
            run_dir=run_dir,
            children_map=children_map,
            results=results,
            item=item,
            parent_name=parent_record["set_name"],
            reason=f"Parent set {parent_record['set_name']} did not produce usable executed_steps.",
            derived_from_steps=None,
        )
        return
    derived_steps = int(parent_record["executed_steps"])
    for child in children:
        key = (child.name, int(item["dataset_index"]))
        if key in results:
            continue
        task_queue.put(make_task(child, item, derived_from_steps=derived_steps).to_payload())
        pending_ref[0] += 1


def make_generation_progress_bar(*, total: int, initial: int, reused: int) -> tqdm:
    return tqdm(total=total, initial=initial, desc="generation", unit="task", dynamic_ncols=True, postfix={"reused": reused})


def release_generation_memory() -> None:
    gc.collect()
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        return


def log_gpu_memory_snapshot(stage: str) -> None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        print(json.dumps({"stage": stage, "gpus": lines}, ensure_ascii=False), flush=True)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"stage": stage, "gpu_snapshot_error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), flush=True)


def run_generation(experiment: ExperimentConfig, run_dir: Path, items: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    set_map = build_set_map(experiment)
    children_map = build_children_map(experiment)
    item_by_question = {int(item["dataset_index"]): item for item in items}
    all_set_names = [set_config.name for set_config in iter_all_sets(experiment.sets)]
    results = load_existing_results(run_dir, allowed_set_names=set(all_set_names), allowed_question_indices=set(item_by_question))
    refresh_candidate_metadata_from_config(results, set_map, run_dir)
    initial_existing_count = len(results)

    workers: list[mp.Process] = []
    for gpu in experiment.gpus:
        process = ctx.Process(
            target=worker_main,
            args=(str(experiment.config_path), gpu, task_queue, result_queue),
            daemon=True,
            name=f"dream-eval-worker-{gpu.replace(':', '_')}",
        )
        process.start()
        workers.append(process)

    pending = [0]
    progress = make_generation_progress_bar(total=initial_existing_count, initial=initial_existing_count, reused=initial_existing_count)
    try:
        for item in items:
            for set_config in experiment.sets:
                key = (set_config.name, int(item["dataset_index"]))
                if key in results:
                    continue
                task_queue.put(make_task(set_config, item).to_payload())
                pending[0] += 1
                progress.total += 1
                progress.refresh()

        for item in items:
            for set_config in experiment.sets:
                key = (set_config.name, int(item["dataset_index"]))
                existing_record = results.get(key)
                if existing_record is None:
                    continue
                schedule_children_for_parent(
                    parent_record=existing_record,
                    item=item,
                    task_queue=task_queue,
                    pending_ref=pending,
                    children_map=children_map,
                    results=results,
                    run_dir=run_dir,
                )
                progress.total = initial_existing_count + pending[0]
                progress.refresh()

        while pending[0] > 0:
            try:
                payload = result_queue.get(timeout=30.0)
            except queue.Empty:
                failed = [process for process in workers if process.exitcode not in (None, 0)]
                if failed:
                    details = ", ".join(f"{process.name}: exitcode={process.exitcode}" for process in failed)
                    raise RuntimeError(f"Math eval worker exited before returning all results: {details}")
                continue
            pending[0] -= 1
            record = build_result_record(payload)
            question_index = int(record["question_index"])
            key = (record["set_name"], question_index)
            results[key] = record
            write_result_file(run_dir, record)
            progress.update(1)
            item = item_by_question[question_index]
            total_before_children = pending[0]
            schedule_children_for_parent(
                parent_record=record,
                item=item,
                task_queue=task_queue,
                pending_ref=pending,
                children_map=children_map,
                results=results,
                run_dir=run_dir,
            )
            if pending[0] != total_before_children:
                progress.total += pending[0] - total_before_children
                progress.refresh()
    finally:
        progress.close()
        for _ in workers:
            task_queue.put(None)
        for process in workers:
            process.join()
            if process.exitcode not in (0, None):
                raise RuntimeError(f"Worker {process.name} exited with code {process.exitcode}")

    for item in items:
        question_index = int(item["dataset_index"])
        for set_name in all_set_names:
            key = (set_name, question_index)
            if key not in results:
                record = build_skipped_record(build_set_map(experiment)[set_name], item, reason="Task was not executed.")
                results[key] = record
                write_result_file(run_dir, record)
    return results


def build_acc_judge_tasks(results: dict[tuple[str, int], dict[str, Any]]) -> list[JudgeTask]:
    tasks: list[JudgeTask] = []
    for record in results.values():
        if record["status"] != "ok" or not record["answer"]:
            continue
        reference_answer = str(record["source_record"].get("answer", "")).strip()
        tasks.append(
            JudgeTask(
                task_type=TASK_TYPE_ACC,
                question_index=int(record["question_index"]),
                sample_id=str(record["sample_id"]),
                prompt_text=str(record["prompt_text"]),
                set_name=str(record["set_name"]),
                reference_answer=reference_answer,
                reference_final_answer=extract_gsm8k_final_answer(reference_answer),
                candidate_answer=str(record["answer"]),
            )
        )
    return tasks


def build_relaxed_acc_extract_prompt_text(prompt_text: str) -> str:
    return f"{prompt_text.rstrip()}{ACC_RELAXED_EXTRACT_SUFFIX}"


def _is_relaxed_extract_record(record: dict[str, Any]) -> bool:
    reason = str(record.get("judge_extract_reason") or "")
    return ACC_RELAXED_EXTRACT_MARKER in reason


def build_acc_extract_judge_tasks(
    results: dict[tuple[str, int], dict[str, Any]],
    *,
    relaxed_extract: bool,
) -> list[JudgeTask]:
    tasks: list[JudgeTask] = []
    for record in results.values():
        if record["status"] != "ok" or not record["answer"]:
            continue
        if record.get("judge_extract_status") in {"found", "not_found"}:
            if not (
                relaxed_extract
                and record.get("judge_extract_status") == "not_found"
                and not _is_relaxed_extract_record(record)
            ):
                continue
        prompt_text = str(record["prompt_text"])
        if relaxed_extract:
            prompt_text = build_relaxed_acc_extract_prompt_text(prompt_text)
        tasks.append(
            JudgeTask(
                task_type=TASK_TYPE_ACC,
                acc_stage="extract_final_answer",
                question_index=int(record["question_index"]),
                sample_id=str(record["sample_id"]),
                prompt_text=prompt_text,
                set_name=str(record["set_name"]),
                candidate_answer=str(record["answer"]),
            )
        )
    return tasks


def annotate_relaxed_acc_extract_reason(reason: str) -> str:
    clean_reason = str(reason).strip()
    if ACC_RELAXED_EXTRACT_MARKER in clean_reason:
        return clean_reason
    if clean_reason:
        return f"{clean_reason} [{ACC_RELAXED_EXTRACT_MARKER}]"
    return f"[{ACC_RELAXED_EXTRACT_MARKER}]"


def apply_relaxed_acc_extract_compare_fallback(results: dict[tuple[str, int], dict[str, Any]]) -> int:
    updated = 0
    for record in results.values():
        if record["status"] != "ok" or not record.get("answer"):
            continue
        if record.get("judge_extract_status") != "not_found":
            continue
        if not _is_relaxed_extract_record(record):
            continue
        candidate_answer = str(record.get("answer") or "").strip()
        if not candidate_answer:
            continue
        record["judge_extract_status"] = "found"
        record["judge_candidate_final_answer"] = (
            "Full candidate response supplied because relaxed extraction could not isolate a final number:\n"
            f"{candidate_answer}"
        )
        record["judge_extract_reason"] = (
            "Relaxed extraction still returned not_found; passing the full candidate response to the "
            f"final-answer compare judge. [{ACC_RELAXED_EXTRACT_MARKER}]"
        )
        updated += 1
    return updated


def build_acc_compare_judge_tasks(results: dict[tuple[str, int], dict[str, Any]]) -> list[JudgeTask]:
    tasks: list[JudgeTask] = []
    for record in results.values():
        if record["status"] != "ok":
            continue
        if record.get("judge_final_status") == "ok" and record.get("judge_final_verdict") is not None:
            continue
        reference_answer = str(record["source_record"].get("answer", "")).strip()
        tasks.append(
            JudgeTask(
                task_type=TASK_TYPE_ACC,
                acc_stage="compare_final_answer",
                question_index=int(record["question_index"]),
                sample_id=str(record["sample_id"]),
                prompt_text=str(record["prompt_text"]),
                set_name=str(record["set_name"]),
                reference_answer=reference_answer,
                candidate_final_answer=str(record.get("judge_candidate_final_answer") or ""),
            )
        )
    return tasks


def build_acc_reasoning_judge_tasks(results: dict[tuple[str, int], dict[str, Any]]) -> list[JudgeTask]:
    tasks: list[JudgeTask] = []
    for record in results.values():
        if record["status"] != "ok" or not record["answer"]:
            continue
        if record.get("judge_reasoning_status") == "ok" and record.get("judge_reasoning_verdict") is not None:
            continue
        reference_answer = str(record["source_record"].get("answer", "")).strip()
        tasks.append(
            JudgeTask(
                task_type=TASK_TYPE_ACC,
                acc_stage="review_reasoning",
                question_index=int(record["question_index"]),
                sample_id=str(record["sample_id"]),
                prompt_text=str(record["prompt_text"]),
                set_name=str(record["set_name"]),
                reference_answer=reference_answer,
                candidate_answer=str(record["answer"]),
            )
        )
    return tasks


def maybe_shuffle_candidate_records(candidate_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered_records = list(candidate_records)
    if random.random() < (2.0 / 3.0):
        random.shuffle(ordered_records)
    return ordered_records


def build_rank_judge_tasks(results: dict[tuple[str, int], dict[str, Any]]) -> list[JudgeTask]:
    tasks: list[JudgeTask] = []
    for question_index, candidate_records in iter_question_candidate_groups(results):
        candidates: dict[str, str] = {}
        candidate_to_set: dict[str, str] = {}
        ordered_candidate_records = maybe_shuffle_candidate_records(candidate_records)
        for idx, record in enumerate(ordered_candidate_records, start=1):
            candidate_name = str(record.get("candidate_name") or f"candidate_{idx}")
            candidates[candidate_name] = str(record["answer"])
            candidate_to_set[candidate_name] = str(record["set_name"])
        tasks.append(
            JudgeTask(
                task_type=TASK_TYPE_RANK,
                question_index=question_index,
                sample_id=str(candidate_records[0]["sample_id"]),
                prompt_text=str(candidate_records[0]["prompt_text"]),
                candidates=candidates,
                candidate_to_set=candidate_to_set,
            )
        )
    return tasks


def build_score_judge_tasks(results: dict[tuple[str, int], dict[str, Any]]) -> list[JudgeTask]:
    tasks: list[JudgeTask] = []
    for question_index, candidate_records in iter_question_candidate_groups(results):
        candidates: dict[str, str] = {}
        candidate_to_set: dict[str, str] = {}
        ordered_candidate_records = maybe_shuffle_candidate_records(candidate_records)
        for idx, record in enumerate(ordered_candidate_records, start=1):
            candidate_name = str(record.get("candidate_name") or f"candidate_{idx}")
            candidates[candidate_name] = str(record["answer"])
            candidate_to_set[candidate_name] = str(record["set_name"])
        tasks.append(
            JudgeTask(
                task_type=TASK_TYPE_SCORE,
                question_index=question_index,
                sample_id=str(candidate_records[0]["sample_id"]),
                prompt_text=str(candidate_records[0]["prompt_text"]),
                candidates=candidates,
                candidate_to_set=candidate_to_set,
            )
        )
    return tasks


def run_judge(experiment: ExperimentConfig, run_dir: Path, results: dict[tuple[str, int], dict[str, Any]]) -> str | None:
    if experiment.judge.task_type == TASK_TYPE_ACC:
        tasks = build_acc_extract_judge_tasks(
            results,
            relaxed_extract=experiment.judge.acc_relaxed_extract,
        )
    else:
        reset_all_judge_fields(results)
    if experiment.judge.task_type == TASK_TYPE_SCORE:
        tasks = build_score_judge_tasks(results)
    elif experiment.judge.task_type == TASK_TYPE_RANK:
        tasks = build_rank_judge_tasks(results)
    elif experiment.judge.task_type != TASK_TYPE_ACC:
        raise ValueError(f"Unsupported judge task type: {experiment.judge.task_type}")
    if not tasks and experiment.judge.task_type != TASK_TYPE_ACC:
        return None

    print(json.dumps(
        {
            "stage": "judge_start",
            "judge_python": experiment.judge.python,
            "model_name": experiment.judge.model_name,
            "task_type": experiment.judge.task_type,
            "gpus": list(experiment.gpus),
            "batch_size": experiment.judge.batch_size,
            "max_input_length": experiment.judge.max_input_length,
            "max_new_tokens": experiment.judge.max_new_tokens,
            "acc_relaxed_extract": experiment.judge.acc_relaxed_extract,
        },
        ensure_ascii=False,
    ), flush=True)
    log_gpu_memory_snapshot("before_judge")
    raw_path = run_dir / "judge_raw.jsonl"
    if experiment.judge.task_type != TASK_TYPE_ACC and raw_path.exists():
        raw_path.unlink()
    try:
        def _run_judge_payloads(task_items: list[JudgeTask]) -> list[dict[str, Any]]:
            return run_with_subprocess(
                judge_python=experiment.judge.python,
                items=[task.to_payload() for task in task_items],
                batch_size=experiment.judge.batch_size,
                model_name=experiment.judge.model_name,
                torch_dtype_name=experiment.judge.torch_dtype_name,
                temperature=experiment.judge.temperature,
                max_input_length=experiment.judge.max_input_length,
                max_new_tokens=experiment.judge.max_new_tokens,
                gpu_devices=list(experiment.gpus),
                show_progress=True,
            )

        if tasks:
            judge_results = _run_judge_payloads(tasks)
            for payload in judge_results:
                append_jsonl(raw_path, payload)
                task = JudgeTask.from_payload(payload["task"])
                if task.task_type == TASK_TYPE_ACC:
                    key = (str(task.set_name), int(task.question_index))
                    if key not in results:
                        continue
                    results[key]["judge_elapsed_sec"] = payload["elapsed_sec"]
                    if task.acc_stage == "extract_final_answer":
                        results[key]["judge_extract_status"] = payload["status"]
                        results[key]["judge_error"] = payload["error"]
                        if payload["status"] == "ok":
                            parsed = payload["parsed"]
                            results[key]["judge_extract_status"] = parsed["status"]
                            extract_reason = parsed["reason"]
                            if experiment.judge.acc_relaxed_extract:
                                extract_reason = annotate_relaxed_acc_extract_reason(extract_reason)
                            results[key]["judge_extract_reason"] = extract_reason
                            results[key]["judge_candidate_final_answer"] = parsed["final_answer"] or None
                        else:
                            results[key]["judge_status"] = "error"
                    else:
                        results[key]["judge_status"] = payload["status"]
                        results[key]["judge_error"] = payload["error"]
                        if payload["status"] == "ok":
                            parsed = payload["parsed"]
                            if task.acc_stage == "compare_final_answer":
                                results[key]["judge_final_status"] = payload["status"]
                                results[key]["judge_final_verdict"] = parsed["verdict"]
                                results[key]["judge_final_confidence"] = parsed["confidence"]
                                results[key]["judge_final_reason"] = parsed["reason"]
                            elif task.acc_stage == "review_reasoning":
                                results[key]["judge_reasoning_status"] = payload["status"]
                                results[key]["judge_reasoning_verdict"] = parsed["verdict"]
                                results[key]["judge_reasoning_confidence"] = parsed["confidence"]
                                results[key]["judge_reasoning_reason"] = parsed["reason"]
                                results[key]["judge_verdict"] = parsed["verdict"]
                                results[key]["judge_confidence"] = parsed["confidence"]
                                results[key]["judge_reason"] = parsed["reason"]
                    write_result_file(run_dir, results[key])
                elif task.task_type == TASK_TYPE_SCORE:
                    candidate_to_set = task.candidate_to_set or {}
                    touched_keys = [(set_name, int(task.question_index)) for set_name in candidate_to_set.values()]
                    for key in touched_keys:
                        if key in results:
                            results[key]["judge_status"] = payload["status"]
                            results[key]["judge_elapsed_sec"] = payload["elapsed_sec"]
                            results[key]["judge_error"] = payload["error"]
                    if payload["status"] == "ok":
                        parsed = payload["parsed"]
                        _ranking, _ties, rank_map = derive_score_order(parsed["scores"])
                        top_rank = min(rank_map.values())
                        for candidate_name, set_name in candidate_to_set.items():
                            key = (set_name, int(task.question_index))
                            if key not in results:
                                continue
                            results[key]["judge_score"] = parsed["scores"][candidate_name]
                            results[key]["judge_score_reason"] = parsed["reason"]
                            results[key]["judge_rank"] = rank_map[candidate_name]
                            results[key]["judge_is_best"] = bool(rank_map[candidate_name] == top_rank)
                            results[key]["judge_rank_from_score"] = rank_map[candidate_name]
                            results[key]["judge_is_best_from_score"] = bool(rank_map[candidate_name] == top_rank)
                            results[key]["judge_reason"] = parsed["reason"]
                else:
                    candidate_to_set = task.candidate_to_set or {}
                    touched_keys = [(set_name, int(task.question_index)) for set_name in candidate_to_set.values()]
                    for key in touched_keys:
                        if key in results:
                            results[key]["judge_status"] = payload["status"]
                            results[key]["judge_elapsed_sec"] = payload["elapsed_sec"]
                            results[key]["judge_error"] = payload["error"]
                    if payload["status"] == "ok":
                        parsed = payload["parsed"]
                        rank_map = compute_rank_map(parsed["ranking"], parsed["ties"])
                        top_rank = min(rank_map.values())
                        for candidate_name, set_name in candidate_to_set.items():
                            key = (set_name, int(task.question_index))
                            if key not in results:
                                continue
                            results[key]["judge_reason"] = parsed["reason"]
                            results[key]["judge_rank"] = rank_map[candidate_name]
                            results[key]["judge_is_best"] = bool(rank_map[candidate_name] == top_rank)

        if experiment.judge.task_type == TASK_TYPE_ACC:
            if experiment.judge.acc_relaxed_extract:
                fallback_count = apply_relaxed_acc_extract_compare_fallback(results)
                if fallback_count:
                    print(
                        json.dumps(
                            {
                                "stage": "relaxed_acc_extract_compare_fallback",
                                "updated_records": fallback_count,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    for record in results.values():
                        if _is_relaxed_extract_record(record):
                            write_result_file(run_dir, record)
            compare_tasks = build_acc_compare_judge_tasks(results)
            if compare_tasks:
                compare_results = _run_judge_payloads(compare_tasks)
                for payload in compare_results:
                    append_jsonl(raw_path, payload)
                    task = JudgeTask.from_payload(payload["task"])
                    key = (str(task.set_name), int(task.question_index))
                    if key not in results:
                        continue
                    results[key]["judge_elapsed_sec"] = payload["elapsed_sec"]
                    results[key]["judge_final_status"] = payload["status"]
                    results[key]["judge_error"] = payload["error"]
                    if payload["status"] == "ok":
                        parsed = payload["parsed"]
                        results[key]["judge_final_verdict"] = parsed["verdict"]
                        results[key]["judge_final_confidence"] = parsed["confidence"]
                        results[key]["judge_final_reason"] = parsed["reason"]
                        if experiment.judge.acc_final_answer_only:
                            results[key]["judge_status"] = payload["status"]
                            results[key]["judge_verdict"] = parsed["verdict"]
                            results[key]["judge_confidence"] = parsed["confidence"]
                            results[key]["judge_reason"] = parsed["reason"]
                    write_result_file(run_dir, results[key])

            if not experiment.judge.acc_final_answer_only:
                reasoning_tasks = build_acc_reasoning_judge_tasks(results)
                if reasoning_tasks:
                    reasoning_results = _run_judge_payloads(reasoning_tasks)
                    for payload in reasoning_results:
                        append_jsonl(raw_path, payload)
                        task = JudgeTask.from_payload(payload["task"])
                        key = (str(task.set_name), int(task.question_index))
                        if key not in results:
                            continue
                        results[key]["judge_status"] = payload["status"]
                        results[key]["judge_reasoning_status"] = payload["status"]
                        results[key]["judge_elapsed_sec"] = payload["elapsed_sec"]
                        results[key]["judge_error"] = payload["error"]
                        if payload["status"] == "ok":
                            parsed = payload["parsed"]
                            results[key]["judge_reasoning_verdict"] = parsed["verdict"]
                            results[key]["judge_reasoning_confidence"] = parsed["confidence"]
                            results[key]["judge_reasoning_reason"] = parsed["reason"]
                            results[key]["judge_verdict"] = parsed["verdict"]
                            results[key]["judge_confidence"] = parsed["confidence"]
                            results[key]["judge_reason"] = parsed["reason"]
                        write_result_file(run_dir, results[key])
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    log_gpu_memory_snapshot("after_judge")
    return None


def write_ranking_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = list(rows[0].keys())
    seen = set(fieldnames)
    for row in rows[1:]:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_rows(results: dict[tuple[str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(results.values())
    rows.sort(key=lambda item: (item["question_index"], item["set_name"]))
    return rows


def build_manifest(experiment: ExperimentConfig, run_name: str, items: list[dict[str, Any]], *, sample_manifest_path: Path | None = None) -> dict[str, Any]:
    return {
        "run_name": run_name,
        "config_path": str(experiment.config_path),
        "dataset": experiment.dataset,
        "split": experiment.split,
        "seed": experiment.seed,
        "num_samples": len(items),
        "gpus": list(experiment.gpus),
        "output_root": str(experiment.output_root),
        "sample_manifest_path": str(sample_manifest_path) if sample_manifest_path is not None else None,
        "judge": {
            "model_name": experiment.judge.model_name,
            "task_type": experiment.judge.task_type,
            "acc_final_answer_only": experiment.judge.acc_final_answer_only,
            "torch_dtype": experiment.judge.torch_dtype_name,
            "batch_size": experiment.judge.batch_size,
            "max_input_length": experiment.judge.max_input_length,
            "max_new_tokens": experiment.judge.max_new_tokens,
            "temperature": experiment.judge.temperature,
            "acc_relaxed_extract": experiment.judge.acc_relaxed_extract,
        },
        "sampled_questions": [
            {
                "question_index": int(item["dataset_index"]),
                "sample_id": str(item["sample_id"]),
            }
            for item in items
        ],
    }


def build_summary(experiment: ExperimentConfig, results: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    all_set_names = [item.name for item in iter_all_sets(experiment.sets)]
    if experiment.judge.task_type == TASK_TYPE_ACC:
        summary = summarize_acc_results(results, all_set_names)
        if experiment.judge.acc_final_answer_only:
            for set_summary in summary["sets"].values():
                set_summary["judge_success_count"] = set_summary["final_judge_success_count"]
                set_summary["judge_failure_count"] = set_summary["final_judge_failure_count"]
                set_summary["correct_count"] = set_summary["final_answer_correct_count"]
                set_summary["incorrect_count"] = set_summary["final_answer_incorrect_count"]
                set_summary["acc"] = set_summary["final_answer_acc"]
        return summary
    if experiment.judge.task_type == TASK_TYPE_RANK:
        return summarize_rank_results(results, all_set_names)
    if experiment.judge.task_type == TASK_TYPE_SCORE:
        return summarize_score_results(results, all_set_names)
    raise ValueError(f"Unsupported judge task type: {experiment.judge.task_type}")


def main() -> None:
    args = parse_args()
    experiment = load_experiment_config(args.config)
    run_name = resolve_run_name(experiment, args.run_name)
    run_dir = prepare_run_dir(experiment, run_name)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "dataset": experiment.dataset,
                    "split": experiment.split,
                    "num_samples": experiment.num_samples,
                    "sets": [item.name for item in iter_all_sets(experiment.sets)],
                    "judge_task_type": experiment.judge.task_type,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return

    items = load_eval_items_for_run(experiment, run_dir, sample_manifest_path=args.sample_manifest)
    manifest = build_manifest(experiment, run_name, items, sample_manifest_path=args.sample_manifest)
    write_json(run_dir / "manifest.json", manifest)

    results = run_generation(experiment, run_dir, items)
    release_generation_memory()
    if experiment.judge.task_type == TASK_TYPE_ACC:
        restored_count = restore_acc_judge_fields_from_results_csv(run_dir, results)
        if restored_count:
            print(json.dumps({"stage": "restore_acc_judge_cache", "restored_records": restored_count}, ensure_ascii=False), flush=True)
        if experiment.judge.acc_final_answer_only:
            aliased_count = apply_acc_final_answer_only_aliases(run_dir, results)
            if aliased_count:
                print(json.dumps({"stage": "apply_final_answer_only_aliases", "updated_records": aliased_count}, ensure_ascii=False), flush=True)
    judge_error = run_judge(experiment, run_dir, results)
    rows = flatten_rows(results)
    write_ranking_csv(run_dir / "results.csv", rows)
    summary = build_summary(experiment, results)
    if judge_error is not None:
        summary["judge_error"] = judge_error
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
