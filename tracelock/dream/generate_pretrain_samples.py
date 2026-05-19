from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import multiprocessing as mp
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracelock.common.io_utils import ensure_dir, save_pt, write_json
from tracelock.common.kodcode_humaneval_like import build_kodcode_prompt, load_kodcode_split
from tracelock.common.project import setup_project_root
from tracelock.common.random_utils import set_global_seed
from tracelock.data.dataset_specs import (
    DATASET_SPECS,
    extract_prompt_text,
)
from tracelock.dream.dream_utils import (
    DEFAULT_MODEL_ID,
    STATE_EOT,
    compute_state_ids,
    extract_aligned_last_three_hidden_for_projection,
    load_dream_config,
    load_dream_model,
    load_dream_tokenizer,
    resolve_torch_dtype,
    summarize_special_tokens,
)
from tracelock.dream.unwrapped_generation import DreamStepContext, diffusion_generate_unwrapped
from tracelock.models.tracelock import TraceLock, TraceLockConfig


setup_project_root(PROJECT_ROOT)

DEFAULT_RUN_DIR = PROJECT_ROOT / "workspace" / "traces" / "dream_math_code"
DEFAULT_PROJECTION_CHECKPOINT = PROJECT_ROOT / "workspace" / "checkpoints" / "dream-ae-v1" / "best_val_loss.pt"
FIXED_DATASET_WEIGHTS = {
    "gsm8k": 0.2,
    "kodcode_humaneval_like": 0.2,
    "alpaca_cleaned": 0.4,
}
GEN_LENGTH_CHOICES = (128, 256)
BLOCK_DIVISOR_CHOICES = (8,16,32)
STEP_SAMPLING_CHOICES = ("random",)
CODE_DATASET_KEYS = frozenset({"kodcode_humaneval_like"})
OUTPUT_GROUPS = ("coding", "others")


@dataclass(frozen=True)
class PromptItem:
    sample_id: str
    dataset_key: str
    split: str
    source_index: int
    prompt_text: str
    source_record: dict[str, Any]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Dream projected pretrain shards for before_reranker.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--datasets", nargs="+", default=["gsm8k", "kodcode_humaneval_like", "alpaca_cleaned"])
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num-samples", type=int, default=16000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--devices", nargs="+", default=[f"cuda:{idx}" for idx in range(1)])
    parser.add_argument("--torch-dtype", type=str, default="bfloat16")
    parser.add_argument("--feature-storage-dtype", type=str, default="float16")
    parser.add_argument("--projection-checkpoint", type=Path, default=DEFAULT_PROJECTION_CHECKPOINT)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--gen-length-choices", nargs="+", type=int, default=list(GEN_LENGTH_CHOICES))
    parser.add_argument("--block-divisor-choices", nargs="+", type=int, default=list(BLOCK_DIVISOR_CHOICES))
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Dream diffusion steps. Defaults to max_new_tokens for each sample when unset.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--alg", type=str, default="entropy")
    parser.add_argument("--alg-temp", type=float, default=0.0)
    parser.add_argument("--step-sample-ratio", type=float, default=0.3)
    parser.add_argument("--step-sampling", type=str, choices=STEP_SAMPLING_CHOICES, default="random")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-run-dir-gb", type=float, default=600.0)
    parser.add_argument("--size-check-every-samples", type=int, default=50)
    parser.add_argument(
        "--sample-id-suffix",
        type=str,
        default="",
        help="Optional short suffix appended to each sample_id, useful for avoiding cross-machine filename collisions.",
    )
    return parser.parse_args()


def output_group_for_dataset(dataset_key: str) -> str:
    return "coding" if dataset_key in CODE_DATASET_KEYS else "others"


def normalize_sample_id_suffix(value: str) -> str:
    suffix = str(value).strip()
    if not suffix:
        return ""
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in suffix)
    cleaned = cleaned.strip("_")
    return f"_{cleaned}" if cleaned else ""


def resolve_feature_storage_dtype(name: str) -> torch.dtype:
    return resolve_torch_dtype(name)


def choose_split(sample_id: str, seed: int, val_ratio: float) -> str:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return "val" if bucket < val_ratio else "train"


def choose_step_indices(
    num_steps: int,
    step_sample_ratio: float,
    *,
    sampling: str,
    rng: random.Random,
) -> list[int]:
    kept_steps = max(1, int(round(num_steps * step_sample_ratio)))
    kept_steps = min(kept_steps, num_steps)
    if kept_steps >= num_steps:
        return list(range(num_steps))
    if sampling == "random":
        indices = sorted(rng.sample(range(num_steps), k=kept_steps))
        if indices[0] != 0:
            indices[0] = 0
        if indices[-1] != num_steps - 1:
            indices[-1] = num_steps - 1
        return sorted(set(indices))

    raw = torch.linspace(0, num_steps - 1, steps=kept_steps)
    indices = sorted({int(round(value.item())) for value in raw})
    if indices[-1] != num_steps - 1:
        indices[-1] = num_steps - 1
    if indices[0] != 0:
        indices[0] = 0
    while len(indices) < kept_steps:
        for candidate in range(num_steps):
            if candidate not in indices:
                indices.append(candidate)
                break
    return sorted(indices)


def choose_batch_config(
    *,
    rng: random.Random,
    max_new_tokens: int | None,
    gen_length_choices: list[int],
    steps: int | None,
    block_divisor_choices: list[int],
) -> dict[str, int]:
    gen_len = int(max_new_tokens or rng.choice(list(gen_length_choices)))
    generation_steps = int(steps) if steps is not None else gen_len
    valid_divisors = [
        int(divisor)
        for divisor in block_divisor_choices
        if int(divisor) > 0 and gen_len % int(divisor) == 0 and generation_steps % int(divisor) == 0
    ]
    if not valid_divisors:
        raise ValueError(
            "No valid block divisor for "
            f"gen_len={gen_len}, steps={generation_steps}, choices={list(block_divisor_choices)}"
        )
    block_divisor = int(rng.choice(valid_divisors))
    block_length = gen_len // block_divisor
    num_blocks = block_divisor
    steps_per_block = generation_steps // num_blocks
    return {
        "gen_len": gen_len,
        "steps": generation_steps,
        "block_length": block_length,
        "block_divisor": block_divisor,
        "num_blocks": num_blocks,
        "steps_per_block": steps_per_block,
    }


def load_dataset_split(dataset_key: str, split: str):
    if dataset_key == "kodcode_humaneval_like":
        return load_kodcode_split(split)
    spec = DATASET_SPECS[dataset_key]
    if spec.subset is None:
        return load_dataset(spec.dataset_name, split=split)
    return load_dataset(spec.dataset_name, spec.subset, split=split)


def allocate_sample_counts(dataset_keys: list[str], total_samples: int) -> dict[str, int]:
    if total_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {total_samples}")
    missing_keys = [key for key in dataset_keys if key not in FIXED_DATASET_WEIGHTS]
    if missing_keys:
        raise KeyError(f"Datasets are missing fixed weights: {missing_keys}")
    weights = {key: float(FIXED_DATASET_WEIGHTS[key]) for key in dataset_keys}
    positive_weights = {key: weight for key, weight in weights.items() if weight > 0.0}
    if positive_weights:
        weights = positive_weights
        dataset_keys = [key for key in dataset_keys if key in weights]
    else:
        weights = {key: 1.0 for key in dataset_keys}
    weight_sum = sum(weights.values())
    if weight_sum <= 0.0:
        raise ValueError(f"Dataset weights must sum to a positive value, got {weights}")

    counts: dict[str, int] = {}
    assigned = 0
    remainders: list[tuple[float, str]] = []
    for dataset_key in dataset_keys:
        exact = total_samples * (weights[dataset_key] / weight_sum)
        base = int(math.floor(exact))
        counts[dataset_key] = base
        assigned += base
        remainders.append((exact - base, dataset_key))

    remaining = total_samples - assigned
    for _score, dataset_key in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        counts[dataset_key] += 1
        remaining -= 1
    return counts


def build_prompt_items(
    *,
    dataset_keys: list[str],
    split: str,
    num_samples: int,
    seed: int,
    sample_id_suffix: str = "",
) -> list[PromptItem]:
    items: list[PromptItem] = []
    normalized_suffix = normalize_sample_id_suffix(sample_id_suffix)
    sample_counts = allocate_sample_counts(dataset_keys, num_samples)
    for dataset_offset, dataset_key in enumerate(dataset_keys):
        dataset = load_dataset_split(dataset_key, split)
        rng = random.Random(seed + dataset_offset)
        indices = list(range(len(dataset)))
        rng.shuffle(indices)
        target_count = int(sample_counts.get(dataset_key, 0))
        for sample_idx in indices[:target_count]:
            record = dataset[int(sample_idx)]
            if dataset_key == "kodcode_humaneval_like":
                prompt_text = build_kodcode_prompt(record)
            else:
                prompt_text = extract_prompt_text(dataset_key, record)
            if not prompt_text:
                continue
            items.append(
                PromptItem(
                    sample_id=f"{dataset_key}_{split}_{sample_idx:08d}{normalized_suffix}",
                    dataset_key=dataset_key,
                    split=split,
                    source_index=int(sample_idx),
                    prompt_text=prompt_text,
                    source_record={
                        key: value
                        for key, value in record.items()
                        if isinstance(value, (str, int, float, bool)) or value is None
                    },
                )
            )
    random.Random(seed + 99).shuffle(items)
    return items[:num_samples]


def batch_items(items: list[PromptItem], batch_size: int) -> list[list[PromptItem]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def split_items(items: list[PromptItem], parts: int) -> list[list[PromptItem]]:
    if parts <= 0:
        raise ValueError(f"parts must be positive, got {parts}")
    out: list[list[PromptItem]] = [[] for _ in range(parts)]
    for index, item in enumerate(items):
        out[index % parts].append(item)
    return out


def compute_directory_size_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for path in root.rglob("*"):
        if path.name.startswith(".") or path.suffix == ".tmp" or path.name.endswith(".partial"):
            continue
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            continue
        except OSError:
            continue
    return total


def should_stop_for_size_limit(run_dir: Path, max_run_dir_gb: float) -> bool:
    if max_run_dir_gb <= 0:
        return False
    return compute_directory_size_bytes(run_dir) >= int(max_run_dir_gb * (1024 ** 3))


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


def crop_sample_sequence(
    full_sequence: torch.Tensor,
    *,
    prompt_token_length: int,
    padded_prompt_length: int,
) -> torch.Tensor:
    prompt_part = full_sequence[:prompt_token_length]
    generation_part = full_sequence[padded_prompt_length:]
    return torch.cat([prompt_part, generation_part], dim=0).contiguous()


def crop_padded_tensor(
    tensor: torch.Tensor,
    *,
    prompt_token_length: int,
    padded_prompt_length: int,
) -> torch.Tensor:
    if tensor.ndim == 1:
        return torch.cat([tensor[:prompt_token_length], tensor[padded_prompt_length:]], dim=0).contiguous()
    if tensor.ndim == 2:
        return torch.cat([tensor[:prompt_token_length], tensor[padded_prompt_length:]], dim=0).contiguous()
    raise ValueError(f"Unsupported tensor rank for crop_padded_tensor: {tuple(tensor.shape)}")


class BlockScheduleHook:
    def __init__(
        self,
        *,
        prompt_length: int,
        block_length: int,
        num_blocks: int,
        steps_per_block: int,
        mask_token_id: int,
    ) -> None:
        self.prompt_length = int(prompt_length)
        self.block_length = int(block_length)
        self.num_blocks = int(num_blocks)
        self.steps_per_block = int(steps_per_block)
        self.mask_token_id = int(mask_token_id)

    def __call__(self, context: DreamStepContext, x_after: torch.Tensor) -> torch.Tensor:
        # 训练数据这里不是让 Dream 一次性解完整个 generation 区，
        # 而是强制它按 block 向右推进。
        # 当前 block 右边的 token 即使已经在本 step 被 proposal 出来，也会重新抹回 mask，
        # 这样采样出来的轨迹更接近“局部窗口逐段展开”的训练设定。
        active_block = min(int(context.step) // self.steps_per_block, self.num_blocks - 1)
        boundary = self.prompt_length + (active_block + 1) * self.block_length
        out = x_after.clone()
        out[:, boundary:] = self.mask_token_id
        return out

    def apply_score_mask(self, step: int, full_scores: torch.Tensor) -> torch.Tensor:
        # score 也要同步屏蔽到当前 block 边界为止；
        # 否则虽然 __call__ 会把右边重新盖回 mask，但本 step 的 top-k 预算会被未来 block 抢走。
        active_block = min(int(step) // self.steps_per_block, self.num_blocks - 1)
        boundary = self.prompt_length + (active_block + 1) * self.block_length
        out = full_scores.clone()
        out[:, boundary:] = -torch.inf
        return out

    def compute_transfer_count(self, step: int, x: torch.Tensor, mask_index: torch.Tensor, default_count: int) -> torch.Tensor:
        del mask_index, default_count
        # 在当前 block 内部，把“剩余 mask”均匀摊到这个 block 剩下的 step 上。
        # 这样每个 block 会在自己的 steps_per_block 里逐步填满，而不是前面 step 一下子填完。
        active_block = min(int(step) // self.steps_per_block, self.num_blocks - 1)
        block_start = self.prompt_length + active_block * self.block_length
        block_end = block_start + self.block_length
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


class PretrainCaptureHook:
    def __init__(
        self,
        *,
        projection_model: TraceLock,
        feature_storage_dtype: torch.dtype,
        pad_eos_token_id: int,
        kept_step_indices: list[int],
        block_hook: BlockScheduleHook,
    ) -> None:
        self.projection_model = projection_model
        self.feature_storage_dtype = feature_storage_dtype
        self.pad_eos_token_id = int(pad_eos_token_id)
        self.kept_step_index_set = set(int(idx) for idx in kept_step_indices)
        self.block_hook = block_hook
        self.projected_by_step: dict[int, torch.Tensor] = {}
        self.step_token_ids_by_step: dict[int, torch.Tensor] = {}
        self.step_confidence_by_step: dict[int, torch.Tensor] = {}
        self.state_by_step: dict[int, torch.Tensor] = {}

    def __call__(self, context: DreamStepContext, x_after: torch.Tensor) -> torch.Tensor:
        # 先应用 block hook，再决定是否抓取这个 step 的训练特征。
        # 返回值会作为下一个 step 的 x，因此这里既是“采样控制点”，也是“抓特征的时机”。
        x_captured = self.block_hook(context, x_after)
        if int(context.step) in self.kept_step_index_set:
            if context.hidden_states is None:
                raise RuntimeError("output_hidden_states=True is required for projected pretrain generation.")
            # 这里就是 pretrain 样本里 hidden states 做“位置右移对齐”的入口。
            # context.hidden_states 本身还是 Dream 原始输出；真正的 shift 发生在
            # extract_aligned_last_three_hidden_for_projection(...) 里面。
            # 这样保存下来的 projected_x，才和当前 step 的 shifted logits / label 语义一致。
            last_three_hidden = extract_aligned_last_three_hidden_for_projection(context.hidden_states)
            projected_x = self.projection_model.project_hidden_layers(last_three_hidden)
            # step_token_ids 取的是当前 step 在 shifted logits 下的 argmax proposal，
            # 也就是“如果现在就接受，这个位置会填成什么”。
            step_token_ids = torch.argmax(context.logits, dim=-1)
            # confidence 也必须基于同一份 shifted logits 来算，
            # 否则会和保存下来的 step_token_ids / label 语义错位。
            step_probs = torch.softmax(context.logits.to(torch.float64), dim=-1)
            step_confidence = torch.gather(
                step_probs,
                dim=-1,
                index=step_token_ids.unsqueeze(-1),
            ).squeeze(-1)
            batch_state: list[torch.Tensor] = []
            # Capture state from the same pre-transfer sequence that produced
            # the hidden states and logits for this step.
            # 也就是说 state 必须基于 context.x，而不是已经执行完 transfer 的 x_after。
            x_cpu = context.x.to(dtype=torch.int32).cpu()
            for batch_idx in range(x_cpu.shape[0]):
                batch_state.append(
                    compute_state_ids(
                        x_cpu[batch_idx],
                        prompt_length=int(context.prompt_length),
                        mask_token_id=int(context.mask_token_id),
                        eos_token_id=self.pad_eos_token_id,
                        dtype=torch.int32,
                    )
                )
            self.projected_by_step[int(context.step)] = projected_x.to(dtype=self.feature_storage_dtype).cpu()
            self.step_token_ids_by_step[int(context.step)] = step_token_ids.to(dtype=torch.int32).cpu()
            self.step_confidence_by_step[int(context.step)] = step_confidence.to(dtype=self.feature_storage_dtype).cpu()
            self.state_by_step[int(context.step)] = torch.stack(batch_state, dim=0).contiguous()
        return x_captured


def save_sample(
    *,
    output_root: Path,
    sample: PromptItem,
    split_name: str,
    kept_step_indices: list[int],
    projected_steps: list[torch.Tensor],
    step_token_steps: list[torch.Tensor],
    confidence_steps: list[torch.Tensor],
    state_steps: list[torch.Tensor],
    final_sequence: torch.Tensor,
    prompt_token_length: int,
    generation_meta: dict[str, Any],
    tokenizer,
) -> int:
    final_token_ids = final_sequence.to(dtype=torch.int32).cpu()
    final_answer = tokenizer.decode(final_sequence[prompt_token_length:].tolist(), skip_special_tokens=True).strip()
    if not final_answer:
        return 0

    projected_x = torch.stack(projected_steps, dim=0).contiguous()
    step_token_ids = torch.stack(step_token_steps, dim=0).contiguous()
    confidence = torch.stack(confidence_steps, dim=0).contiguous()
    state = torch.stack(state_steps, dim=0).contiguous()
    # Dream pretrain 的核心监督：
    # 当前 step 的 proposal token，是否等于这条轨迹最终完成后的 token。
    # 所以 label 学的不是“标准下一个词”，而是“这个位置现在是不是已经稳定到可以接受”。
    label = (step_token_ids == final_token_ids.unsqueeze(0)).to(dtype=torch.int32)

    pad_eos_token_id = int(generation_meta["pad_eos_token_id"])
    eot_positions = final_token_ids == pad_eos_token_id
    if eot_positions.any():
        # EOT 只在保存训练样本时额外回填；推理阶段并不依赖这个 state。
        state[:, eot_positions] = int(STATE_EOT)

    split_dir = ensure_dir(output_root / "samples" / split_name)
    meta_dir = ensure_dir(output_root / "samples" / "meta")
    for local_step_idx, actual_step_idx in enumerate(kept_step_indices):
        save_pt(
            split_dir / f"sample_{sample.sample_id}_step_{actual_step_idx:04d}.pt",
            {
                "x": projected_x[local_step_idx].contiguous().clone(),
                "step_token_ids": step_token_ids[local_step_idx].contiguous().clone(),
                "confidence": confidence[local_step_idx].contiguous().clone(),
                "label": label[local_step_idx].contiguous().clone(),
                "state": state[local_step_idx].contiguous().clone(),
                "gen_len": int(generation_meta["gen_len"]),
                "steps": int(generation_meta["steps"]),
                "block_length": int(generation_meta["block_length"]),
                "block_divisor": int(generation_meta["block_divisor"]),
                "num_blocks": int(generation_meta["num_blocks"]),
                "steps_per_block": int(generation_meta["steps_per_block"]),
                "temperature": float(generation_meta["temperature"]),
                "top_p": float(generation_meta["top_p"]),
                "alg": str(generation_meta["alg"]),
                "alg_temp": float(generation_meta["alg_temp"]),
                "model_id": str(generation_meta["model_id"]),
                "projection_checkpoint": str(generation_meta["projection_checkpoint"]),
            },
        )

    write_json(
        meta_dir / f"sample_{sample.sample_id}.json",
        {
            "sample_id": sample.sample_id,
            "split": split_name,
            "prompt": sample.prompt_text,
            "answer": final_answer,
            "dataset_key": sample.dataset_key,
            "dataset_split": sample.split,
            "source_index": sample.source_index,
            "source_record": sample.source_record,
            "prompt_token_length": int(prompt_token_length),
            "gen_len": int(generation_meta["gen_len"]),
            "steps": int(generation_meta["steps"]),
            "block_length": int(generation_meta["block_length"]),
            "block_divisor": int(generation_meta["block_divisor"]),
            "num_blocks": int(generation_meta["num_blocks"]),
            "steps_per_block": int(generation_meta["steps_per_block"]),
            "temperature": float(generation_meta["temperature"]),
            "top_p": float(generation_meta["top_p"]),
            "alg": str(generation_meta["alg"]),
            "alg_temp": float(generation_meta["alg_temp"]),
            "kept_step_indices": list(kept_step_indices),
            "model_id": str(generation_meta["model_id"]),
            "projection_checkpoint": str(generation_meta["projection_checkpoint"]),
        },
    )
    return len(kept_step_indices)


def worker_run(worker_args: dict[str, Any]) -> dict[str, Any]:
    worker_rank = int(worker_args["worker_rank"])
    device = str(worker_args["device"])
    run_dir = Path(worker_args["run_dir"])
    set_global_seed(int(worker_args["seed"]) + worker_rank)

    dtype = resolve_torch_dtype(str(worker_args["torch_dtype"]))
    feature_storage_dtype = resolve_feature_storage_dtype(str(worker_args["feature_storage_dtype"]))

    tokenizer = load_dream_tokenizer(
        str(worker_args["model_id"]),
        local_files_only=bool(worker_args.get("local_files_only", False)),
    )
    model = load_dream_model(
        str(worker_args["model_id"]),
        device=device,
        torch_dtype=dtype,
        local_files_only=bool(worker_args.get("local_files_only", False)),
    )
    projection_model = load_projection_model(Path(worker_args["projection_checkpoint"]), device)

    prompt_items = [PromptItem(**payload) for payload in worker_args["prompt_items"]]
    total_kept = 0
    total_step_files = 0
    total_dropped = 0
    kept_by_group = {group: 0 for group in OUTPUT_GROUPS}
    step_files_by_group = {group: 0 for group in OUTPUT_GROUPS}
    dropped_by_group = {group: 0 for group in OUTPUT_GROUPS}
    stopped_for_size_limit = False
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(worker_args["seed"]) + worker_rank)

    import random

    py_rng = random.Random(int(worker_args["seed"]) + worker_rank)

    print(
        f"[dream-pretrain-worker {worker_rank} {device}] starting num_items={len(prompt_items)} batch_size={worker_args['batch_size']}",
        flush=True,
    )

    for batch in batch_items(prompt_items, int(worker_args["batch_size"])):
        if should_stop_for_size_limit(run_dir, float(worker_args["max_run_dir_gb"])):
            stopped_for_size_limit = True
            break

        # 每个 batch 会随机抽一组 generation 长度 / block 配置，
        # 这样 Dream pretrain 样本天然带有不同 budget 和不同局部展开节奏。
        batch_meta = choose_batch_config(
            rng=py_rng,
            max_new_tokens=worker_args["max_new_tokens"],
            gen_length_choices=list(worker_args["gen_length_choices"]),
            steps=worker_args["steps"],
            block_divisor_choices=list(worker_args["block_divisor_choices"]),
        )
        gen_length = int(batch_meta["gen_len"])
        generation_steps = int(batch_meta["steps"])
        kept_step_indices = choose_step_indices(
            generation_steps,
            float(worker_args["step_sample_ratio"]),
            sampling=str(worker_args["step_sampling"]),
            rng=py_rng,
        )

        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": item.prompt_text}],
                add_generation_prompt=True,
                tokenize=False,
            )
            for item in batch
        ]
        encoded = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        padded_prompt_length = int(input_ids.shape[1])

        # 这里的 prompt_length 用的是 batch padding 后长度，因为 Dream 生成循环里
        # 整个 batch 的 generation 区是接在同一个 padded prompt 右边。
        block_hook = BlockScheduleHook(
            prompt_length=padded_prompt_length,
            block_length=int(batch_meta["block_length"]),
            num_blocks=int(batch_meta["num_blocks"]),
            steps_per_block=int(batch_meta["steps_per_block"]),
            mask_token_id=int(tokenizer.mask_token_id),
        )
        capture_hook = PretrainCaptureHook(
            projection_model=projection_model,
            feature_storage_dtype=feature_storage_dtype,
            pad_eos_token_id=int(tokenizer.pad_token_id),
            kept_step_indices=kept_step_indices,
            block_hook=block_hook,
        )

        with torch.no_grad():
            output = diffusion_generate_unwrapped(
                model,
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_length,
                steps=generation_steps,
                temperature=float(worker_args["temperature"]),
                top_p=float(worker_args["top_p"]),
                proposal_alg=str(worker_args["alg"]),
                alg_temp=float(worker_args["alg_temp"]),
                output_hidden_states=True,
                transfer_scores_hook_func=lambda step, _x, _mask_index, full_scores: block_hook.apply_score_mask(step, full_scores),
                transfer_count_hook_func=lambda step, x, mask_index, default_count: block_hook.compute_transfer_count(step, x, mask_index, default_count),
                post_transfer_hook=capture_hook,
            )

        for sample_idx, sample in enumerate(batch):
            # 单样本保存时，再把 batch padding 的 prompt 部分裁掉，
            # 恢复成“真实 prompt + generation”的紧凑布局。
            prompt_token_length = int(attention_mask[sample_idx].sum().item())
            final_sequence = crop_sample_sequence(
                output["sequences"][sample_idx].detach().cpu(),
                prompt_token_length=prompt_token_length,
                padded_prompt_length=padded_prompt_length,
            )
            per_sample_projected: list[torch.Tensor] = []
            per_sample_step_tokens: list[torch.Tensor] = []
            per_sample_confidence: list[torch.Tensor] = []
            per_sample_state: list[torch.Tensor] = []
            for step_index in kept_step_indices:
                # capture_hook 里缓存的 tensor 仍然带着 batch padding；
                # 保存 .pt 前，需要把无效 padding prompt 裁掉，避免 state / label / x 错位。
                per_sample_projected.append(
                    crop_padded_tensor(
                        capture_hook.projected_by_step[step_index][sample_idx],
                        prompt_token_length=prompt_token_length,
                        padded_prompt_length=padded_prompt_length,
                    )
                )
                per_sample_step_tokens.append(
                    crop_padded_tensor(
                        capture_hook.step_token_ids_by_step[step_index][sample_idx],
                        prompt_token_length=prompt_token_length,
                        padded_prompt_length=padded_prompt_length,
                    )
                )
                per_sample_confidence.append(
                    crop_padded_tensor(
                        capture_hook.step_confidence_by_step[step_index][sample_idx],
                        prompt_token_length=prompt_token_length,
                        padded_prompt_length=padded_prompt_length,
                    )
                )
                per_sample_state.append(
                    crop_padded_tensor(
                        capture_hook.state_by_step[step_index][sample_idx],
                        prompt_token_length=prompt_token_length,
                        padded_prompt_length=padded_prompt_length,
                    )
                )

            split_name = choose_split(sample.sample_id, int(worker_args["seed"]), float(worker_args["val_ratio"]))
            output_group = output_group_for_dataset(sample.dataset_key)
            step_files = save_sample(
                output_root=run_dir / output_group,
                sample=sample,
                split_name=split_name,
                kept_step_indices=kept_step_indices,
                projected_steps=per_sample_projected,
                step_token_steps=per_sample_step_tokens,
                confidence_steps=per_sample_confidence,
                state_steps=per_sample_state,
                final_sequence=final_sequence,
                prompt_token_length=prompt_token_length,
                tokenizer=tokenizer,
                generation_meta={
                    "gen_len": gen_length,
                    "steps": generation_steps,
                    "block_length": int(batch_meta["block_length"]),
                    "block_divisor": int(batch_meta["block_divisor"]),
                    "num_blocks": int(batch_meta["num_blocks"]),
                    "steps_per_block": int(batch_meta["steps_per_block"]),
                    "temperature": float(worker_args["temperature"]),
                    "top_p": float(worker_args["top_p"]),
                    "alg": str(worker_args["alg"]),
                    "alg_temp": float(worker_args["alg_temp"]),
                    "model_id": str(worker_args["model_id"]),
                    "projection_checkpoint": str(worker_args["projection_checkpoint"]),
                    "pad_eos_token_id": int(tokenizer.pad_token_id),
                },
            )
            if step_files > 0:
                total_kept += 1
                total_step_files += step_files
                kept_by_group[output_group] += 1
                step_files_by_group[output_group] += step_files
            else:
                total_dropped += 1
                dropped_by_group[output_group] += 1

            if (
                int(worker_args["size_check_every_samples"]) > 0
                and total_kept > 0
                and total_kept % int(worker_args["size_check_every_samples"]) == 0
                and should_stop_for_size_limit(run_dir, float(worker_args["max_run_dir_gb"]))
            ):
                stopped_for_size_limit = True
                break

        if stopped_for_size_limit:
            break

    summary = {
        "worker_rank": worker_rank,
        "device": device,
        "num_prompt_items": len(prompt_items),
        "samples_kept": total_kept,
        "samples_dropped": total_dropped,
        "step_files": total_step_files,
        "samples_kept_by_group": kept_by_group,
        "samples_dropped_by_group": dropped_by_group,
        "step_files_by_group": step_files_by_group,
        "stopped_for_size_limit": stopped_for_size_limit,
    }
    print(f"[dream-pretrain-worker {worker_rank} {device}] done {json.dumps(summary, ensure_ascii=True)}", flush=True)
    del projection_model
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def _worker_entry(worker_args: dict[str, Any], result_queue: Any) -> None:
    try:
        summary = worker_run(worker_args)
        result_queue.put(("ok", int(worker_args["worker_rank"]), summary))
    except BaseException as exc:  # noqa: BLE001
        result_queue.put(("error", int(worker_args["worker_rank"]), repr(exc)))
        raise


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    run_dir = ensure_dir(args.run_dir)
    for group in OUTPUT_GROUPS:
        ensure_dir(run_dir / group / "samples" / "train")
        ensure_dir(run_dir / group / "samples" / "val")
        ensure_dir(run_dir / group / "samples" / "meta")

    prompt_items = build_prompt_items(
        dataset_keys=list(args.datasets),
        split=args.split,
        num_samples=args.num_samples,
        seed=args.seed,
        sample_id_suffix=args.sample_id_suffix,
    )
    if not prompt_items:
        raise RuntimeError("No prompt items were loaded.")

    write_json(
        run_dir / "config.json",
        {
            **{
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in vars(args).items()
            },
            "output_stage": "before_reranker",
            "output_groups": list(OUTPUT_GROUPS),
            "code_dataset_keys": sorted(CODE_DATASET_KEYS),
            "group_routing": {
                "coding": sorted(CODE_DATASET_KEYS),
                "others": "all remaining dataset keys",
            },
            "sample_roots": {
                "coding": str(run_dir / "coding" / "samples"),
                "others": str(run_dir / "others" / "samples"),
            },
            "steps_default_behavior": "match_gen_length" if args.steps is None else "fixed_override",
            "schedule_strategy": "dream_unwrapped_random_block_divisor_post_transfer_masking",
            "normalized_sample_id_suffix": normalize_sample_id_suffix(args.sample_id_suffix),
        },
    )
    model_summary_tokenizer = load_dream_tokenizer(args.model_id, local_files_only=args.local_files_only)
    model_summary_config = load_dream_config(args.model_id, local_files_only=args.local_files_only)
    write_json(run_dir / "dream_model_summary.json", summarize_special_tokens(model_summary_tokenizer, model_summary_config))

    devices = list(args.devices)
    prompt_shards = split_items(prompt_items, len(devices))
    worker_payloads = []
    for worker_rank, (device, shard) in enumerate(zip(devices, prompt_shards)):
        worker_payloads.append(
            {
                "worker_rank": worker_rank,
                "device": device,
                "run_dir": str(run_dir),
                "prompt_items": [
                    {
                        "sample_id": item.sample_id,
                        "dataset_key": item.dataset_key,
                        "split": item.split,
                        "source_index": item.source_index,
                        "prompt_text": item.prompt_text,
                        "source_record": item.source_record,
                    }
                    for item in shard
                ],
                "batch_size": args.batch_size,
                "seed": args.seed,
                "model_id": args.model_id,
                "torch_dtype": args.torch_dtype,
                "feature_storage_dtype": args.feature_storage_dtype,
                "projection_checkpoint": str(args.projection_checkpoint),
                "max_new_tokens": args.max_new_tokens,
                "gen_length_choices": list(args.gen_length_choices),
                "block_divisor_choices": list(args.block_divisor_choices),
                "steps": args.steps,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "alg": args.alg,
                "alg_temp": args.alg_temp,
                "step_sample_ratio": args.step_sample_ratio,
                "step_sampling": args.step_sampling,
                "val_ratio": args.val_ratio,
                "local_files_only": args.local_files_only,
                "max_run_dir_gb": args.max_run_dir_gb,
                "size_check_every_samples": args.size_check_every_samples,
            }
        )

    if len(devices) == 1:
        worker_summaries = [worker_run(worker_payloads[0])]
    else:
        mp_ctx = mp.get_context("spawn")
        result_queue = mp_ctx.Queue()
        processes = [
            mp_ctx.Process(
                target=_worker_entry,
                args=(payload, result_queue),
                name=f"dream-pretrain-worker-{payload['worker_rank']:02d}",
            )
            for payload in worker_payloads
        ]
        for process in processes:
            process.start()

        results_by_rank: dict[int, dict[str, Any]] = {}
        errors: list[tuple[int, str]] = []
        for _ in worker_payloads:
            status, worker_rank, payload = result_queue.get()
            if status == "ok":
                results_by_rank[int(worker_rank)] = payload
            else:
                errors.append((int(worker_rank), str(payload)))

        for process in processes:
            process.join()

        exit_failures = [
            (payload["worker_rank"], process.exitcode)
            for payload, process in zip(worker_payloads, processes)
            if process.exitcode not in (0, None)
        ]
        if errors or exit_failures:
            raise RuntimeError(
                f"Dream pretrain workers failed. errors={errors} exit_failures={exit_failures}"
            )
        worker_summaries = [results_by_rank[idx] for idx in sorted(results_by_rank)]

    total_kept = sum(int(item["samples_kept"]) for item in worker_summaries)
    total_dropped = sum(int(item["samples_dropped"]) for item in worker_summaries)
    total_step_files = sum(int(item["step_files"]) for item in worker_summaries)
    kept_by_group = {
        group: sum(int(item.get("samples_kept_by_group", {}).get(group, 0)) for item in worker_summaries)
        for group in OUTPUT_GROUPS
    }
    dropped_by_group = {
        group: sum(int(item.get("samples_dropped_by_group", {}).get(group, 0)) for item in worker_summaries)
        for group in OUTPUT_GROUPS
    }
    step_files_by_group = {
        group: sum(int(item.get("step_files_by_group", {}).get(group, 0)) for item in worker_summaries)
        for group in OUTPUT_GROUPS
    }
    stopped_for_size_limit = any(bool(item["stopped_for_size_limit"]) for item in worker_summaries)
    directory_size_bytes = compute_directory_size_bytes(run_dir)

    write_json(
        run_dir / "summary.json",
        {
            "output_stage": "before_reranker",
            "num_prompt_items": len(prompt_items),
            "samples_kept": total_kept,
            "samples_dropped": total_dropped,
            "step_files": total_step_files,
            "samples_kept_by_group": kept_by_group,
            "samples_dropped_by_group": dropped_by_group,
            "step_files_by_group": step_files_by_group,
            "stopped_for_size_limit": stopped_for_size_limit,
            "directory_size_bytes": directory_size_bytes,
            "directory_size_gb": round(directory_size_bytes / (1024 ** 3), 4),
            "devices": devices,
            "worker_summaries": worker_summaries,
        },
    )
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "output_stage": "before_reranker",
                "num_prompt_items": len(prompt_items),
                "samples_kept": total_kept,
                "samples_dropped": total_dropped,
                "step_files": total_step_files,
                "samples_kept_by_group": kept_by_group,
                "samples_dropped_by_group": dropped_by_group,
                "step_files_by_group": step_files_by_group,
                "stopped_for_size_limit": stopped_for_size_limit,
                "directory_size_gb": round(directory_size_bytes / (1024 ** 3), 4),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
