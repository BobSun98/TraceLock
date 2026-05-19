from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracelock.common.io_utils import load_json  # noqa: E402
from tracelock.common.kodcode_humaneval_like import build_code_generation_prompt  # noqa: E402
from tracelock.common.soft_crop import find_soft_crop_window_from_mask  # noqa: E402
from tracelock.common.transfer_selector import (  # noqa: E402
    TRANSFER_SCORE_SOURCE_CONFIDENCE,
    normalize_transfer_score_source,
    select_transfer_masks,
)
from tracelock.data.dataset_specs import (  # noqa: E402
    DATASET_SPECS,
    extract_prompt_text,
)
from tracelock.models.tracelock import (  # noqa: E402
    TraceLock,
    TraceLockConfig,
    build_tracelock,
    load_tracelock_from_config_payload,
    tracelock_config_from_payload,
    resolve_tracelock_config_payload,
)
from tracelock.dream.dream_utils import (  # noqa: E402
    DEFAULT_MODEL_ID,
    compute_state_ids,
    extract_aligned_last_three_hidden_for_projection,
    load_dream_model,
    load_dream_tokenizer,
    resolve_torch_dtype,
)
from tracelock.dream.unwrapped_generation import (  # noqa: E402
    DreamStepContext,
    diffusion_generate_unwrapped,
)


GEN_METHOD_RANDOM = "random"
GEN_METHOD_LOW_CONFIDENCE = "low_confidence"
GEN_METHOD_NATIVE_ENTROPY = "native_entropy"
GEN_METHOD_FAST_DLM = "fast_dlm"
GEN_METHOD_TRACELOCK = "tracelock"
GEN_METHOD_TRACELOCK_SOFT_CROP = "tracelock_soft_crop"
SUPPORTED_GEN_METHODS = (
    GEN_METHOD_RANDOM,
    GEN_METHOD_LOW_CONFIDENCE,
    GEN_METHOD_NATIVE_ENTROPY,
    GEN_METHOD_FAST_DLM,
    GEN_METHOD_TRACELOCK,
    GEN_METHOD_TRACELOCK_SOFT_CROP,
)

DEFAULT_EXPERIMENT_CONFIG_PATH = PROJECT_ROOT / "configs" / "eval_gsm8k.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "workspace" / "runs" / "eval"
DEFAULT_PROJECTION_CKPT = PROJECT_ROOT / "workspace" / "checkpoints" / "dream-ae-v1" / "best_val_loss.pt"
DEFAULT_FAST_DLM_THRESHOLD = 0.9
TRACELOCK_SCHEDULE_SOFT_CROP = "soft_crop"
TRACELOCK_SCHEDULE_BLOCK = "block"
TRACELOCK_SCHEDULE_FULL = "full"
SUPPORTED_TRACELOCK_SCHEDULES = {
    TRACELOCK_SCHEDULE_SOFT_CROP,
    TRACELOCK_SCHEDULE_BLOCK,
    TRACELOCK_SCHEDULE_FULL,
}


@dataclass(frozen=True)
class SetConfig:
    name: str
    gen_method: str
    gen_length: int
    candidate_num: int | None = None
    candidate_name: str | None = None
    temperature: float = 0.0
    cfg_scale: float = 0.0
    gen_steps: int | None = None
    block_len: int | None = None
    threshold: float | None = None
    pointer_window_size: int | None = None
    tracelock_schedule: str = TRACELOCK_SCHEDULE_SOFT_CROP
    dynamic_threshold: bool = False
    tracelock_threshold: float | None = None
    tracelock_checkpoint: Path | None = None
    tracelock_config: Path | None = None
    projection_checkpoint: Path | None = None
    transfer_score_source: str = TRANSFER_SCORE_SOURCE_CONFIDENCE
    parent_name: str | None = None
    subsets: tuple["SetConfig", ...] = field(default_factory=tuple)


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


@dataclass(frozen=True)
class GenerationTask:
    set_name: str
    parent_set_name: str | None
    question_index: int
    sample_id: str
    dataset: str
    split: str
    prompt_text: str
    source_record: dict[str, Any]
    gen_method: str
    gen_length: int
    candidate_num: int | None
    temperature: float
    cfg_scale: float
    gen_steps: int | None
    block_len: int | None
    threshold: float | None
    pointer_window_size: int | None
    dynamic_threshold: bool
    tracelock_threshold: float | None
    tracelock_checkpoint: str | None
    tracelock_config: str | None
    projection_checkpoint: str | None
    transfer_score_source: str = TRANSFER_SCORE_SOURCE_CONFIDENCE
    derived_from_steps: int | None = None
    candidate_name: str | None = None
    tracelock_schedule: str = TRACELOCK_SCHEDULE_SOFT_CROP

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "GenerationTask":
        return cls(**payload)


def normalize_gen_method(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in SUPPORTED_GEN_METHODS:
        raise ValueError(f"Unsupported gen_method: {value}")
    return normalized


def _resolve_path(value: str | None, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _validate_set_name(name: str) -> None:
    if not name or name in {".", ".."} or "/" in name:
        raise ValueError(f"Invalid set name: {name!r}")


def normalize_tracelock_schedule(value: str | None) -> str:
    if value is None:
        return TRACELOCK_SCHEDULE_SOFT_CROP
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"no_soft_no_block", "none", "global"}:
        normalized = TRACELOCK_SCHEDULE_FULL
    if normalized in {"no_soft_block", "fixed_block"}:
        normalized = TRACELOCK_SCHEDULE_BLOCK
    if normalized not in SUPPORTED_TRACELOCK_SCHEDULES:
        raise ValueError(f"Unsupported tracelock_schedule: {value}")
    return normalized


def _parse_set_config(raw: dict[str, Any], *, base_dir: Path, parent_name: str | None = None) -> SetConfig:
    name = str(raw["name"])
    _validate_set_name(name)
    gen_method = normalize_gen_method(str(raw["gen_method"]))
    gen_length = int(raw["gen_length"])
    if gen_length <= 0:
        raise ValueError(f"Set {name} has invalid gen_length={gen_length}")

    candidate_num = raw.get("candidate_num")
    candidate_name = raw.get("candidate_name")
    temperature = float(raw.get("temperature", 0.0))
    cfg_scale = float(raw.get("cfg_scale", 0.0))
    gen_steps = raw.get("gen_steps")
    block_len = raw.get("block_len", raw.get("block_size"))
    threshold = raw.get("threshold")
    pointer_window_size = raw.get("pointer_window_size")
    tracelock_schedule = normalize_tracelock_schedule(raw.get("tracelock_schedule"))
    dynamic_threshold = bool(raw.get("dynamic_threshold", False))
    tracelock_threshold = raw.get("tracelock_threshold")
    transfer_score_source = normalize_transfer_score_source(str(raw.get("transfer_score_source", TRANSFER_SCORE_SOURCE_CONFIDENCE)))

    if candidate_num is not None:
        candidate_num = int(candidate_num)
    if candidate_name is not None:
        candidate_name = str(candidate_name).strip()
        if not candidate_name:
            raise ValueError(f"Set {name} has empty candidate_name.")
    if gen_steps is not None:
        gen_steps = int(gen_steps)
    if block_len is not None:
        block_len = int(block_len)
    if threshold is not None:
        threshold = float(threshold)
    if pointer_window_size is not None:
        pointer_window_size = int(pointer_window_size)
    if tracelock_threshold is not None:
        tracelock_threshold = float(tracelock_threshold)

    tracelock_checkpoint = _resolve_path(raw.get("tracelock_checkpoint"), base_dir)
    tracelock_config = _resolve_path(raw.get("tracelock_config"), base_dir)
    projection_checkpoint = _resolve_path(raw.get("projection_checkpoint"), base_dir) or DEFAULT_PROJECTION_CKPT

    if gen_method in {
        GEN_METHOD_LOW_CONFIDENCE,
        GEN_METHOD_NATIVE_ENTROPY,
        GEN_METHOD_RANDOM,
        GEN_METHOD_FAST_DLM,
    }:
        if parent_name is None and gen_steps is None:
            raise ValueError(f"Set {name} must define gen_steps.")
        if block_len is None:
            raise ValueError(f"Set {name} must define block_len.")
        if gen_length % block_len != 0:
            raise ValueError(f"Set {name} requires gen_length divisible by block_len.")
        if gen_steps is not None:
            num_blocks = gen_length // block_len
            if gen_steps % num_blocks != 0:
                raise ValueError(f"Set {name} requires gen_steps divisible by num_blocks={num_blocks}.")
        if gen_method == GEN_METHOD_FAST_DLM and threshold is None:
            threshold = DEFAULT_FAST_DLM_THRESHOLD
    else:
        if pointer_window_size is None:
            raise ValueError(f"TraceLock set {name} must define pointer_window_size.")
        if pointer_window_size <= 0:
            raise ValueError(f"TraceLock set {name} has invalid pointer_window_size={pointer_window_size}.")
        if tracelock_schedule == TRACELOCK_SCHEDULE_BLOCK:
            if gen_length % pointer_window_size != 0:
                raise ValueError(
                    f"TraceLock block set {name} requires gen_length divisible by pointer_window_size."
                )
            if gen_steps is not None and gen_steps % (gen_length // pointer_window_size) != 0:
                raise ValueError(
                    f"TraceLock block set {name} requires gen_steps divisible by num_blocks."
                )
        if not dynamic_threshold and tracelock_threshold is None:
            raise ValueError(f"TraceLock set {name} must define tracelock_threshold.")
        if tracelock_checkpoint is None:
            raise ValueError(f"TraceLock set {name} must define tracelock_checkpoint.")
        if tracelock_config is None:
            tracelock_config = tracelock_checkpoint.with_name("config.json")

    subsets = tuple(_parse_set_config(item, base_dir=base_dir, parent_name=name) for item in raw.get("subset", []))
    return SetConfig(
        name=name,
        gen_method=gen_method,
        gen_length=gen_length,
        candidate_num=candidate_num,
        candidate_name=candidate_name,
        temperature=temperature,
        cfg_scale=cfg_scale,
        gen_steps=gen_steps,
        block_len=block_len,
        threshold=threshold,
        pointer_window_size=pointer_window_size,
        tracelock_schedule=tracelock_schedule,
        dynamic_threshold=dynamic_threshold,
        tracelock_threshold=tracelock_threshold,
        tracelock_checkpoint=tracelock_checkpoint,
        tracelock_config=tracelock_config,
        projection_checkpoint=projection_checkpoint,
        transfer_score_source=transfer_score_source,
        parent_name=parent_name,
        subsets=subsets,
    )


def iter_all_sets(sets: tuple[SetConfig, ...]) -> tuple[SetConfig, ...]:
    out: list[SetConfig] = []
    stack = list(sets)
    while stack:
        current = stack.pop(0)
        out.append(current)
        stack[0:0] = list(current.subsets)
    return tuple(out)


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

    output_root = _resolve_path(raw.get("output_root"), base_dir) or DEFAULT_OUTPUT_ROOT
    sets = tuple(_parse_set_config(item, base_dir=base_dir) for item in raw["sets"])
    all_sets = iter_all_sets(sets)
    names = [item.name for item in all_sets]
    if len(names) != len(set(names)):
        raise ValueError("All set names, including subsets, must be globally unique.")

    return ExperimentConfig(
        config_path=config_path,
        experiment_name=str(raw.get("experiment_name")) if raw.get("experiment_name") is not None else None,
        output_root=output_root,
        gpus=gpus,
        dataset=str(raw["dataset"]),
        split=str(raw.get("split", "test")),
        seed=int(raw.get("seed", 42)),
        num_samples=int(raw["num_samples"]),
        dlm_model_name=str(raw.get("dlm_model_name", DEFAULT_MODEL_ID)),
        torch_dtype_name=str(raw.get("torch_dtype", "bfloat16")),
        overwrite=bool(raw.get("overwrite", False)),
        local_files_only=bool(raw.get("local_files_only", False)),
        sets=sets,
    )


def build_set_map(experiment: ExperimentConfig) -> dict[str, SetConfig]:
    return {item.name: item for item in iter_all_sets(experiment.sets)}


def build_children_map(experiment: ExperimentConfig) -> dict[str, tuple[SetConfig, ...]]:
    children: dict[str, list[SetConfig]] = {}
    for item in iter_all_sets(experiment.sets):
        if item.parent_name is not None:
            children.setdefault(item.parent_name, []).append(item)
    return {key: tuple(value) for key, value in children.items()}


def load_dream(model_name: str, dtype: torch.dtype, device: str, *, local_files_only: bool):
    tokenizer = load_dream_tokenizer(model_name, local_files_only=local_files_only)
    if getattr(tokenizer, "padding_side", None) != "left":
        tokenizer.padding_side = "left"
    model = load_dream_model(
        model_name,
        device=device,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    return model, tokenizer


def load_tracelock_model(
    checkpoint_path: Path,
    config_path: Path,
    projection_checkpoint: Path | None,
    device: str,
) -> TraceLock:
    train_config = load_json(config_path)
    model_config = resolve_tracelock_config_payload(train_config)
    uses_precomputed_input = bool(train_config.get("use_ae_data", False))
    use_confidence_feature = bool(train_config.get("use_confidence_feature", False))
    precomputed_input_dim = train_config.get("precomputed_input_dim")
    if precomputed_input_dim is None and uses_precomputed_input:
        precomputed_input_dim = (
            3 * int(model_config["d_tracelock"])
            + 2 * int(model_config["d_tracelock_delta"])
            + (1 if use_confidence_feature else 0)
        )

    if projection_checkpoint is not None:
        projection_payload = torch.load(projection_checkpoint, map_location="cpu", weights_only=False)
        projection_state = projection_payload.get("projection_state", projection_payload)
        hidden_norm_state = projection_state.get("hidden_norm")
        if isinstance(hidden_norm_state, dict) and "weight" in hidden_norm_state:
            inferred_d_model = int(hidden_norm_state["weight"].shape[0])
            if train_config.get("d_model") != inferred_d_model:
                train_config = dict(train_config)
                train_config["d_model"] = inferred_d_model
                model_config = dict(model_config)
                model_config["d_model"] = inferred_d_model

    tracelock_payload: dict[str, Any]
    projection_model: TraceLock | None = None
    if uses_precomputed_input:
        if projection_checkpoint is None:
            raise ValueError(
                f"TraceLock checkpoint {checkpoint_path} expects precomputed AE features, "
                "but no projection_checkpoint was provided."
            )
        tracelock_payload = dict(train_config)
        if "init_model_config" in tracelock_payload and isinstance(tracelock_payload["init_model_config"], dict):
            nested_config = dict(tracelock_payload["init_model_config"])
            nested_config["d_model"] = int(model_config["d_model"])
            nested_config["enable_input_projection_stack"] = False
            nested_config["precomputed_input_dim"] = int(precomputed_input_dim)
            tracelock_payload["init_model_config"] = nested_config
        else:
            tracelock_payload["d_model"] = int(model_config["d_model"])
            tracelock_payload["enable_input_projection_stack"] = False
            tracelock_payload["precomputed_input_dim"] = int(precomputed_input_dim)

        projection_config = tracelock_config_from_payload(
            {
                **model_config,
                "enable_input_projection_stack": True,
                "precomputed_input_dim": None,
            }
        )
        projection_model = build_tracelock(
            projection_config,
            device=device,
            projection_checkpoint=projection_checkpoint,
            freeze_projection=True,
            checkpoint_path=None,
            eval_mode=True,
            print_summary=False,
        )
    else:
        tracelock_payload = train_config

    model = load_tracelock_from_config_payload(
        tracelock_payload,
        device=device,
        projection_checkpoint=None if uses_precomputed_input else projection_checkpoint,
        freeze_projection=True,
        checkpoint_path=checkpoint_path,
        eval_mode=True,
        print_summary=True,
        summary_title=f"DreamTraceLock[{checkpoint_path.name}]",
    )
    model._dream_use_confidence_feature = use_confidence_feature
    model._dream_precomputed_input_dim = (
        None if precomputed_input_dim is None else int(precomputed_input_dim)
    )
    model._dream_uses_precomputed_input = uses_precomputed_input
    model._dream_projection_model = projection_model
    return model


def load_projection_model(checkpoint_path: Path, device: str) -> TraceLock:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    auto_cfg = checkpoint.get("autoencoder_config", {})
    tracelock = TraceLock(
        TraceLockConfig(
            d_model=int(auto_cfg.get("d_model", 4096)),
            d_tracelock=int(auto_cfg.get("d_hidden_bottleneck", 256)),
            d_tracelock_delta=int(auto_cfg.get("d_delta_bottleneck", 32)),
        )
    ).to(device).eval()
    tracelock.load_pretrained_projections(checkpoint_path, freeze=True)
    return tracelock


def build_prompts(tokenizer: AutoTokenizer, prompt_texts: list[str]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for prompt_text in prompt_texts
    ]


def build_single_prompt(tokenizer: AutoTokenizer, prompt_text: str, device: str) -> tuple[str, torch.Tensor, torch.Tensor]:
    prompt = build_prompts(tokenizer, [prompt_text])[0]
    encoded = tokenizer(
        [prompt],
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    return prompt, encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


def finalize_generation_result(
    tokenizer: AutoTokenizer,
    prompt: str,
    prompt_text: str,
    x: torch.Tensor,
    prompt_length: int,
    history: list[dict[str, Any]],
    start_time: float,
) -> dict[str, Any]:
    elapsed_sec = time.perf_counter() - start_time
    answer = tokenizer.batch_decode(x[:, prompt_length:], skip_special_tokens=True)[0].strip()
    return {
        "prompt": prompt,
        "prompt_text": prompt_text,
        "answer": answer,
        "elapsed_sec": elapsed_sec,
        "executed_steps": len(history),
        "history": history,
        "final_remaining_masks": int((x[:, prompt_length:] == int(tokenizer.mask_token_id)).sum().item()),
    }


def question_filename(question_index: int) -> str:
    return f"{question_index:08d}.json"


@dataclass
class HistoryRow:
    global_step: int
    block_idx: int
    step_in_block: int
    window_start: int
    window_end: int
    remaining_masks_before: int
    remaining_masks_after: int
    scheduled_transfer_count: int | None
    actual_transfer_count: int
    tracelock_accept_count: int
    mean_tracelock_prob: float | None
    threshold: float | None = None
    forced_transfer_count: int | None = None
    dynamic_threshold: bool | None = None
    tracelock_threshold: float | None = None
    tracelock_schedule: str | None = None
    transfer_score_source: str | None = None
    fallback_reason: str | None = None
    suppressed_special_accept_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DreamBlockPolicy:
    def __init__(
        self,
        *,
        prompt_length: int,
        gen_length: int,
        steps: int,
        block_length: int,
        mask_token_id: int,
    ) -> None:
        if gen_length % block_length != 0:
            raise ValueError(f"gen_length={gen_length} must be divisible by block_length={block_length}")
        self.prompt_length = int(prompt_length)
        self.gen_length = int(gen_length)
        self.steps = int(steps)
        self.block_length = int(block_length)
        self.mask_token_id = int(mask_token_id)
        self.num_blocks = self.gen_length // self.block_length
        if self.steps % self.num_blocks != 0:
            raise ValueError(f"steps={steps} must be divisible by num_blocks={self.num_blocks}")
        self.steps_per_block = self.steps // self.num_blocks
        self.history_rows: list[dict[str, Any]] = []

    def resolve_block(self, step: int) -> tuple[int, int, int]:
        block_idx = min(int(step) // self.steps_per_block, self.num_blocks - 1)
        block_start = self.prompt_length + block_idx * self.block_length
        block_end = block_start + self.block_length
        return block_idx, block_start, block_end

    def build_current_window_mask(self, context: DreamStepContext) -> tuple[int, int, int, torch.Tensor]:
        block_idx, block_start, block_end = self.resolve_block(int(context.step))
        current_window_mask = torch.zeros_like(context.mask_index, dtype=torch.bool)
        current_window_mask[:, block_start:block_end] = context.mask_index[:, block_start:block_end]
        return block_idx, block_start, block_end, current_window_mask

    def __call__(self, context: DreamStepContext, x_after: torch.Tensor) -> torch.Tensor:
        _, _, block_end = self.resolve_block(int(context.step))
        out = x_after.clone()
        out[:, block_end:] = self.mask_token_id
        return out

    def apply_score_mask(self, step: int, full_scores: torch.Tensor) -> torch.Tensor:
        block_idx, block_start, block_end = self.resolve_block(step)
        del block_idx, block_start
        out = full_scores.clone()
        out[:, block_end:] = -torch.inf
        return out

    def compute_transfer_count(
        self,
        step: int,
        x: torch.Tensor,
        mask_index: torch.Tensor,
        default_count: int,
    ) -> torch.Tensor:
        del mask_index, default_count
        block_idx, block_start, block_end = self.resolve_block(step)
        del block_idx
        remaining_masks = (x[:, block_start:block_end] == self.mask_token_id).sum(dim=1).to(dtype=torch.long)
        step_in_block = int(step) % self.steps_per_block
        remaining_steps = self.steps_per_block - step_in_block
        if remaining_steps <= 0:
            return remaining_masks
        transfer_counts = torch.div(
            remaining_masks + (remaining_steps - 1),
            remaining_steps,
            rounding_mode="floor",
        )
        return torch.where(
            remaining_masks > 0,
            torch.clamp(transfer_counts, min=1),
            torch.zeros_like(transfer_counts),
        )


class DreamBlockHistoryHook(DreamBlockPolicy):
    def __init__(self, *, threshold: float | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.threshold = threshold

    def __call__(self, context: DreamStepContext, x_after: torch.Tensor) -> torch.Tensor:
        block_idx, block_start, block_end, current_window_mask = self.build_current_window_mask(context)
        remaining_before = int(context.mask_index[:, self.prompt_length:self.prompt_length + self.gen_length].sum().item())
        out = x_after.clone()
        forced_transfer_count = 0
        if self.threshold is not None:
            step_token_ids = torch.argmax(context.logits, dim=-1)
            step_token_ids = torch.where(context.mask_index, step_token_ids, context.x)
            out = context.x.clone()
            selected_mask = torch.zeros_like(current_window_mask, dtype=torch.bool)
            masked_scores = torch.where(
                current_window_mask,
                context.full_scores,
                torch.full_like(context.full_scores, -torch.inf),
            )
            accepted_mask = current_window_mask & (context.full_scores >= self.threshold)
            for batch_idx in range(context.x.shape[0]):
                active_count = int(current_window_mask[batch_idx].sum().item())
                if active_count <= 0:
                    continue
                accepted_indices = torch.nonzero(accepted_mask[batch_idx], as_tuple=False).flatten()
                if accepted_indices.numel() > 0:
                    chosen_indices = accepted_indices
                else:
                    fallback_index = torch.argmax(masked_scores[batch_idx]).view(1)
                    chosen_indices = fallback_index
                    forced_transfer_count += int(chosen_indices.numel())
                selected_mask[batch_idx, chosen_indices] = True
            out[selected_mask] = step_token_ids[selected_mask]
        out = super().__call__(context, out)
        actual_transfer_mask = (
            (context.x == self.mask_token_id)
            & (out != self.mask_token_id)
        )
        remaining_after = int((out[:, self.prompt_length:self.prompt_length + self.gen_length] == self.mask_token_id).sum().item())
        self.history_rows.append(
            HistoryRow(
                global_step=int(context.step),
                block_idx=block_idx,
                step_in_block=int(context.step) % self.steps_per_block,
                window_start=block_start - self.prompt_length,
                window_end=block_end - self.prompt_length,
                remaining_masks_before=remaining_before,
                remaining_masks_after=remaining_after,
                scheduled_transfer_count=int(context.number_transfer_tokens),
                actual_transfer_count=int(actual_transfer_mask.sum().item()),
                tracelock_accept_count=0,
                mean_tracelock_prob=None,
                threshold=self.threshold,
                forced_transfer_count=forced_transfer_count if forced_transfer_count > 0 else None,
            ).to_dict()
        )
        return out


class DreamTraceLockHook:
    def __init__(
        self,
        *,
        tracelock: TraceLock,
        prompt_length: int,
        gen_length: int,
        mask_token_id: int,
        pad_token_id: int,
        eos_token_id: int,
        pointer_window_size: int,
        steps: int,
        tracelock_schedule: str,
        threshold: float,
        dynamic_threshold: bool,
        transfer_score_source: str,
    ) -> None:
        self.tracelock = tracelock
        self.prompt_length = int(prompt_length)
        self.gen_length = int(gen_length)
        self.mask_token_id = int(mask_token_id)
        self.pad_token_id = int(pad_token_id)
        self.eos_token_id = int(eos_token_id)
        self.pointer_window_size = int(pointer_window_size)
        self.steps = int(steps)
        self.tracelock_schedule = normalize_tracelock_schedule(tracelock_schedule)
        self.threshold = float(threshold)
        self.dynamic_threshold = bool(dynamic_threshold)
        self.transfer_score_source = normalize_transfer_score_source(transfer_score_source)
        self.history_rows: list[dict[str, Any]] = []
        self.num_blocks = 1
        self.steps_per_block = self.steps
        if self.tracelock_schedule == TRACELOCK_SCHEDULE_BLOCK:
            if self.gen_length % self.pointer_window_size != 0:
                raise ValueError(
                    f"gen_length={self.gen_length} must be divisible by "
                    f"pointer_window_size={self.pointer_window_size} for block schedule."
                )
            self.num_blocks = self.gen_length // self.pointer_window_size
            if self.steps % self.num_blocks != 0:
                raise ValueError(
                    f"steps={self.steps} must be divisible by num_blocks={self.num_blocks} "
                    "for block schedule."
                )
            self.steps_per_block = self.steps // self.num_blocks

    def resolve_window(self, mask_index: torch.Tensor, gen_end: int, step: int) -> tuple[int, int, int, int]:
        if self.tracelock_schedule == TRACELOCK_SCHEDULE_FULL:
            return 0, 0, self.prompt_length, gen_end
        if self.tracelock_schedule == TRACELOCK_SCHEDULE_BLOCK:
            block_idx = min(int(step) // self.steps_per_block, self.num_blocks - 1)
            block_start = self.prompt_length + block_idx * self.pointer_window_size
            block_end = min(gen_end, block_start + self.pointer_window_size)
            return block_idx, int(step) % self.steps_per_block, block_start, block_end
        window_start, window_end = find_soft_crop_window_from_mask(
            mask_index[0],
            prompt_length=self.prompt_length,
            window_width=min(self.pointer_window_size, self.gen_length),
            seq_len=gen_end,
        )
        block_idx = int(max(0, window_start - self.prompt_length) // max(1, self.pointer_window_size))
        return block_idx, int(step), window_start, window_end

    def __call__(self, context: DreamStepContext, _x_after: torch.Tensor) -> torch.Tensor:
        if context.hidden_states is None:
            raise RuntimeError("Dream tracelock generation requires output_hidden_states=True.")
        mask_index = context.mask_index
        generation_mask = mask_index[:, self.prompt_length:self.prompt_length + self.gen_length]
        remaining_before = int(generation_mask.sum().item())
        if not generation_mask.any():
            self.history_rows.append(
                HistoryRow(
                    global_step=int(context.step),
                    block_idx=0,
                    step_in_block=int(context.step),
                    window_start=0,
                    window_end=0,
                    remaining_masks_before=remaining_before,
                    remaining_masks_after=remaining_before,
                    scheduled_transfer_count=0,
                    actual_transfer_count=0,
                    tracelock_accept_count=0,
                    mean_tracelock_prob=None,
                    tracelock_schedule=self.tracelock_schedule,
                ).to_dict()
            )
            return context.x.clone()

        gen_end = self.prompt_length + self.gen_length
        block_idx, step_in_block, window_start, window_end = self.resolve_window(mask_index, gen_end, int(context.step))
        current_window_mask = torch.zeros_like(mask_index, dtype=torch.bool)
        current_window_mask[:, window_start:window_end] = mask_index[:, window_start:window_end]

        # evaluate 阶段喂给 TraceLock 的 hidden，必须与 Dream 当前 step 的 shifted logits 对齐。
        # 否则 TraceLock 会用“原始位置 hidden”去判断“当前位置 proposal”。
        last_three_hidden = extract_aligned_last_three_hidden_for_projection(context.hidden_states)
        step_token_ids = torch.argmax(context.logits, dim=-1)
        probs = torch.softmax(context.logits.to(torch.float64), dim=-1)
        confidence_scores = torch.gather(probs, dim=-1, index=step_token_ids.unsqueeze(-1)).squeeze(-1)
        state_ids = compute_state_ids(
            context.x[0],
            prompt_length=self.prompt_length,
            mask_token_id=self.mask_token_id,
            eos_token_id=self.eos_token_id,
            dtype=torch.long,
        ).unsqueeze(0)
        tracelock_hidden_layers = last_three_hidden[:, :window_end].contiguous()
        projection_model = getattr(self.tracelock, "_dream_projection_model", None)
        if projection_model is not None:
            # 预训练 checkpoint 若使用 AE 特征，这里要先用同一个 projection checkpoint
            # 把 [B, L, 3, D] 转回训练时看到的 precomputed feature。
            with torch.no_grad():
                tracelock_hidden_layers = projection_model.project_hidden_layers(tracelock_hidden_layers)
        if bool(getattr(self.tracelock, "_dream_use_confidence_feature", False)):
            if tracelock_hidden_layers.ndim != 3:
                raise ValueError(
                    "Confidence feature requires precomputed TraceLock inputs with shape [B, L, D]."
                )
            tracelock_hidden_layers = torch.cat(
                [
                    tracelock_hidden_layers,
                    confidence_scores[:, :window_end].to(dtype=tracelock_hidden_layers.dtype).unsqueeze(-1),
                ],
                dim=-1,
            )
        expected_input_dim = getattr(self.tracelock, "_dream_precomputed_input_dim", None)
        if expected_input_dim is not None and tracelock_hidden_layers.ndim == 3:
            actual_input_dim = int(tracelock_hidden_layers.shape[-1])
            if actual_input_dim != int(expected_input_dim):
                raise ValueError(
                    "TraceLock input dim mismatch during Dream evaluation: "
                    f"expected {int(expected_input_dim)}, got {actual_input_dim}."
                )
        tracelock_state_ids = state_ids[:, :window_end].contiguous()
        tracelock_attention_mask = torch.ones(
            (1, window_end),
            dtype=torch.long,
            device=context.x.device,
        )
        tracelock_out = self.tracelock(
            hidden_layers=tracelock_hidden_layers,
            state_ids=tracelock_state_ids,
            attention_mask=tracelock_attention_mask,
        )
        tracelock_probs = torch.zeros_like(mask_index, dtype=tracelock_out["logits"].dtype)
        if self.dynamic_threshold:
            tracelock_probs[:, :window_end] = torch.sigmoid(
                tracelock_out["logits"] - tracelock_out["threshold_logit"].unsqueeze(1)
            )
            active_threshold = self.threshold
        else:
            tracelock_probs[:, :window_end] = torch.sigmoid(tracelock_out["logits"])
            active_threshold = self.threshold

        step_token_ids = torch.where(mask_index, step_token_ids, context.x)
        scheduled_transfer_count = min(int(context.number_transfer_tokens), int(current_window_mask.sum().item()))
        selector_out = select_transfer_masks(
            current_window_mask=current_window_mask,
            scheduled_transfer_count=scheduled_transfer_count,
            transfer_score_source=self.transfer_score_source,
            confidence_scores=confidence_scores,
            tracelock_probs=tracelock_probs,
            threshold=active_threshold,
            current_token_ids=context.x,
            step_token_ids=step_token_ids,
            special_token_ids=tuple(sorted({self.pad_token_id, self.eos_token_id})),
            generation_start_index=self.prompt_length,
            generation_length=self.gen_length,
        )
        transfer_mask = selector_out["scheduled_transfer_mask"] | selector_out["threshold_accept_mask"] | selector_out["fallback_mask"]
        accepted_mask = selector_out["threshold_accept_mask"] | selector_out["fallback_mask"]
        out = context.x.clone()
        out[transfer_mask] = step_token_ids[transfer_mask]
        if self.tracelock_schedule == TRACELOCK_SCHEDULE_BLOCK:
            out[:, window_end:gen_end] = self.mask_token_id
        remaining_after = int((out[:, self.prompt_length:gen_end] == self.mask_token_id).sum().item())
        mean_prob = None
        if current_window_mask[:, window_start:window_end].any():
            mean_prob = float(
                tracelock_probs[:, window_start:window_end][current_window_mask[:, window_start:window_end]].mean().item()
            )
        self.history_rows.append(
            HistoryRow(
                global_step=int(context.step),
                block_idx=block_idx,
                step_in_block=step_in_block,
                window_start=window_start - self.prompt_length,
                window_end=window_end - self.prompt_length,
                remaining_masks_before=remaining_before,
                remaining_masks_after=remaining_after,
                scheduled_transfer_count=scheduled_transfer_count,
                actual_transfer_count=int(transfer_mask.sum().item()),
                tracelock_accept_count=int(accepted_mask.sum().item()),
                mean_tracelock_prob=mean_prob,
                dynamic_threshold=self.dynamic_threshold,
                tracelock_threshold=active_threshold,
                tracelock_schedule=self.tracelock_schedule,
                transfer_score_source=self.transfer_score_source,
                fallback_reason=selector_out["fallback_reason"],
                suppressed_special_accept_count=int(selector_out.get("suppressed_special_accept_count", 0)) or None,
            ).to_dict()
        )
        return out


def run_native_generation(
    model,
    tokenizer,
    prompt_text: str,
    *,
    device: str,
    steps: int,
    gen_length: int,
    block_length: int,
    temperature: float,
    remasking: str,
) -> dict[str, Any]:
    prompt, input_ids, attention_mask = build_single_prompt(tokenizer, prompt_text, device)
    prompt_length = int(input_ids.shape[1])
    hook = DreamBlockHistoryHook(
        prompt_length=prompt_length,
        gen_length=gen_length,
        steps=steps,
        block_length=block_length,
        mask_token_id=int(tokenizer.mask_token_id),
    )
    if remasking == GEN_METHOD_RANDOM:
        transfer_score_type = "random"
    elif remasking == GEN_METHOD_NATIVE_ENTROPY:
        transfer_score_type = "entropy"
    else:
        transfer_score_type = "confidence"
    start_time = time.perf_counter()
    output = diffusion_generate_unwrapped(
        model,
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=gen_length,
        steps=steps,
        temperature=temperature,
        proposal_alg="entropy",
        transfer_score_type=transfer_score_type,
        output_hidden_states=False,
        transfer_scores_hook_func=lambda step, _x, _mask_index, full_scores: hook.apply_score_mask(step, full_scores),
        transfer_count_hook_func=lambda step, x, mask_index, default_count: hook.compute_transfer_count(step, x, mask_index, default_count),
        post_transfer_hook=hook,
    )
    return finalize_generation_result(
        tokenizer,
        prompt,
        prompt_text,
        output["sequences"],
        prompt_length,
        hook.history_rows,
        start_time,
    )


def run_fast_dlm_generation(
    model,
    tokenizer,
    prompt_text: str,
    *,
    device: str,
    steps: int,
    gen_length: int,
    block_length: int,
    temperature: float,
    threshold: float,
) -> dict[str, Any]:
    prompt, input_ids, attention_mask = build_single_prompt(tokenizer, prompt_text, device)
    prompt_length = int(input_ids.shape[1])
    hook = DreamBlockHistoryHook(
        prompt_length=prompt_length,
        gen_length=gen_length,
        steps=steps,
        block_length=block_length,
        mask_token_id=int(tokenizer.mask_token_id),
        threshold=threshold,
    )
    start_time = time.perf_counter()
    output = diffusion_generate_unwrapped(
        model,
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=gen_length,
        steps=steps,
        temperature=temperature,
        proposal_alg="entropy",
        transfer_score_type="confidence",
        transfer_method="fast_dlm",
        fast_dlm_threshold=threshold,
        output_hidden_states=False,
        transfer_scores_hook_func=lambda step, _x, _mask_index, full_scores: hook.apply_score_mask(step, full_scores),
        post_transfer_hook=hook,
    )
    return finalize_generation_result(
        tokenizer,
        prompt,
        prompt_text,
        output["sequences"],
        prompt_length,
        hook.history_rows,
        start_time,
    )


def run_tracelock_generation(
    model,
    tokenizer,
    prompt_text: str,
    *,
    device: str,
    steps: int,
    gen_length: int,
    pointer_window_size: int,
    tracelock_schedule: str,
    temperature: float,
    tracelock: TraceLock,
    dynamic_threshold: bool,
    threshold: float,
    transfer_score_source: str,
) -> dict[str, Any]:
    prompt, input_ids, attention_mask = build_single_prompt(tokenizer, prompt_text, device)
    prompt_length = int(input_ids.shape[1])
    hook = DreamTraceLockHook(
        tracelock=tracelock,
        prompt_length=prompt_length,
        gen_length=gen_length,
        mask_token_id=int(tokenizer.mask_token_id),
        pad_token_id=int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id),
        eos_token_id=int(tokenizer.eos_token_id),
        pointer_window_size=pointer_window_size,
        steps=steps,
        tracelock_schedule=tracelock_schedule,
        threshold=threshold,
        dynamic_threshold=dynamic_threshold,
        transfer_score_source=transfer_score_source,
    )
    start_time = time.perf_counter()
    output = diffusion_generate_unwrapped(
        model,
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=gen_length,
        steps=steps,
        temperature=temperature,
        proposal_alg="entropy",
        output_hidden_states=True,
        post_transfer_hook=hook,
    )
    return finalize_generation_result(
        tokenizer,
        prompt,
        prompt_text,
        output["sequences"],
        prompt_length,
        hook.history_rows,
        start_time,
    )


class GenerationWorkerRunner:
    def __init__(self, experiment: ExperimentConfig, device: str):
        self.experiment = experiment
        self.device = device
        self.dtype = resolve_torch_dtype(experiment.torch_dtype_name)
        self.dlm_model, self.tokenizer = load_dream(
            experiment.dlm_model_name,
            self.dtype,
            self.device,
            local_files_only=experiment.local_files_only,
        )
        self._tracelock_cache: dict[tuple[str, str, str | None], TraceLock] = {}
        self._projection_cache: dict[str, TraceLock] = {}

    def _get_tracelock(self, task: GenerationTask) -> TraceLock:
        if task.tracelock_checkpoint is None or task.tracelock_config is None:
            raise ValueError(f"Task {task.set_name} is missing tracelock paths.")
        cache_key = (task.tracelock_checkpoint, task.tracelock_config, task.projection_checkpoint)
        model = self._tracelock_cache.get(cache_key)
        if model is None:
            model = load_tracelock_model(
                checkpoint_path=Path(task.tracelock_checkpoint),
                config_path=Path(task.tracelock_config),
                projection_checkpoint=Path(task.projection_checkpoint) if task.projection_checkpoint else None,
                device=self.device,
            )
            self._tracelock_cache[cache_key] = model
        return model

    def _get_projection_model(self, task: GenerationTask) -> TraceLock:
        if task.projection_checkpoint is None:
            raise ValueError(f"Task {task.set_name} is missing projection_checkpoint.")
        cache_key = task.projection_checkpoint
        model = self._projection_cache.get(cache_key)
        if model is None:
            model = load_projection_model(Path(task.projection_checkpoint), self.device)
            self._projection_cache[cache_key] = model
        return model

    @torch.no_grad()
    def run_task(self, task: GenerationTask) -> dict[str, Any]:
        if task.gen_method in {GEN_METHOD_RANDOM, GEN_METHOD_LOW_CONFIDENCE, GEN_METHOD_NATIVE_ENTROPY}:
            steps = int(task.derived_from_steps or task.gen_steps or task.gen_length)
            block_len = int(task.block_len or task.gen_length)
            result = run_native_generation(
                self.dlm_model,
                self.tokenizer,
                task.prompt_text,
                device=self.device,
                steps=steps,
                gen_length=task.gen_length,
                block_length=block_len,
                temperature=task.temperature,
                remasking=task.gen_method,
            )
        elif task.gen_method == GEN_METHOD_FAST_DLM:
            steps = int(task.derived_from_steps or task.gen_steps or task.gen_length)
            block_len = int(task.block_len or task.gen_length)
            result = run_fast_dlm_generation(
                self.dlm_model,
                self.tokenizer,
                task.prompt_text,
                device=self.device,
                steps=steps,
                gen_length=task.gen_length,
                block_length=block_len,
                temperature=task.temperature,
                threshold=float(task.threshold if task.threshold is not None else DEFAULT_FAST_DLM_THRESHOLD),
            )
        elif task.gen_method in {GEN_METHOD_TRACELOCK, GEN_METHOD_TRACELOCK_SOFT_CROP}:
            steps = int(task.derived_from_steps or task.gen_steps or task.gen_length)
            pointer_window_size = int(task.pointer_window_size or task.gen_length)
            tracelock_schedule = normalize_tracelock_schedule(task.tracelock_schedule)
            result = run_tracelock_generation(
                self.dlm_model,
                self.tokenizer,
                task.prompt_text,
                device=self.device,
                steps=steps,
                gen_length=task.gen_length,
                pointer_window_size=pointer_window_size,
                tracelock_schedule=tracelock_schedule,
                temperature=task.temperature,
                tracelock=self._get_tracelock(task),
                dynamic_threshold=bool(task.dynamic_threshold),
                threshold=float(task.tracelock_threshold or 0.0),
                transfer_score_source=task.transfer_score_source,
            )
        else:
            raise KeyError(f"Unsupported gen_method: {task.gen_method}")

        result["set_name"] = task.set_name
        result["gen_method"] = task.gen_method
        result["question_index"] = task.question_index
        result["derived_from_steps"] = task.derived_from_steps
        result["tracelock_schedule"] = task.tracelock_schedule
        return result
