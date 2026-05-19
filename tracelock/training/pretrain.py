from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracelock.common.project import SCRATCH_ROOT, setup_project_root

setup_project_root(PROJECT_ROOT)

from tracelock.common.io_utils import append_jsonl, ensure_dir, write_json  # noqa: E402
from tracelock.training.pretrain_base import (  # noqa: E402
    STATE_GEN,
    _empty_crop_stats,
    apply_random_prefix_crop,
    apply_relative_input_noise,
    apply_batch_feature_noise,
    build_dataloaders as base_build_dataloaders,
    build_eot_loss_weight,
    build_state_loss_mask,
    build_target_keep_loss_weight,
    cleanup_distributed,
    compute_metrics,
    ddp_requires_unused_parameter_detection,
    freeze_projection_stack,
    get_rank,
    is_distributed,
    is_main_process,
    make_config,
    masked_bce_loss,
    merge_crop_stats,
    move_batch_to_device,
    reduce_crop_stats,
    reduce_mean,
    round_metric_values,
    save_checkpoint,
    serialize_args as base_serialize_args,
    set_global_seed,
    setup_distributed,
    unwrap_model,
)
from tracelock.models.tracelock import TraceLock, build_tracelock, print_tracelock_summary  # noqa: E402
from tracelock.data.trace_dataset import (  # noqa: E402
    TraceDirectoryStepDataset,
    list_split_sample_paths,
    tracelock_step_collate,
)


DEFAULT_SAMPLES_DIR = SCRATCH_ROOT / 'traces' / 'dream_math_code' / 'samples'
DEFAULT_OUTPUT_ROOT = SCRATCH_ROOT / 'checkpoints'

# Optional multi-dataset mixing for slim pretraining.
# Leave empty to keep the original single-dataset behavior driven by --samples-dir.
#

TRAIN_DATASET_MIX = {
    SCRATCH_ROOT / 'traces' / 'dream_math_code' / 'samples' : 1.0,
}




# TRAIN_DATASET_MIX: dict[str, float] = {}


class WeightedReplacementSampler(Sampler[int]):
    def __init__(
        self,
        weights: torch.Tensor,
        num_samples: int,
        *,
        seed: int,
    ) -> None:
        if num_samples <= 0:
            raise ValueError(f'num_samples must be positive, got {num_samples}.')
        if weights.numel() == 0:
            raise ValueError('weights must not be empty.')
        self.weights = weights.to(dtype=torch.double)
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(self.weights, self.num_samples, replacement=True, generator=generator)
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples


class DistributedWeightedReplacementSampler(Sampler[int]):
    def __init__(
        self,
        weights: torch.Tensor,
        *,
        num_replicas: int,
        rank: int,
        seed: int,
    ) -> None:
        if weights.numel() == 0:
            raise ValueError('weights must not be empty.')
        if num_replicas <= 0:
            raise ValueError(f'num_replicas must be positive, got {num_replicas}.')
        if not 0 <= rank < num_replicas:
            raise ValueError(f'rank must satisfy 0 <= rank < num_replicas, got rank={rank}, num_replicas={num_replicas}.')
        self.weights = weights.to(dtype=torch.double)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = int(math.ceil(self.weights.numel() / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(self.weights, self.total_size, replacement=True, generator=generator)
        rank_indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(rank_indices.tolist())

    def __len__(self) -> int:
        return self.num_samples


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Slim pretraining entrypoint for TraceLock.')
    parser.add_argument('--samples-dir', type=Path, default=DEFAULT_SAMPLES_DIR)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--run-name', type=str, default='tracelock')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--use-ae-data',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Expect step samples to provide AE-compressed features in `x`.',
    )

    parser.add_argument('--train-batch-size', type=int, default=65 * 1, help='Global train batch size across all ranks.')
    parser.add_argument('--eval-batch-size', type=int, default= 65 * 1, help='Global eval batch size across all ranks.')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers per rank.')

    parser.add_argument('--max-steps', type=int, default=36000) #✅
    parser.add_argument('--eval-every', type=int, default=200)

    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument(
        '--warmup-steps',
        type=int,
        default=0,
        help='Number of optimizer steps used for linear learning-rate warmup before plateau scheduling starts.',
    )
    parser.add_argument(
        '--warmup-init-lr',
        type=float,
        default=1e-5,
        help='Initial learning rate at step 0 when warmup is enabled.',
    )
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--lr-reduce-patience', type=int, default=4)
    parser.add_argument('--early-stop-patience', type=int, default=8)

    parser.add_argument('--device', type=str, default=None)
    parser.add_argument(
        '--init-checkpoint',
        type=Path,
        default=None,
        help='Optional checkpoint used to initialize model weights before training.',
    )
    parser.add_argument('--pretrained-proj-checkpoint', type=Path, default=SCRATCH_ROOT / 'checkpoints' / 'dream-ae-v1' / 'best_val_loss.pt')
    parser.add_argument('--freeze-pretrained-proj', default=True)
    parser.add_argument('--backend', type=str, default='nccl')
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument(
        '--print-model-summary-only',
        action='store_true',
        help='Build the model, print its architecture and parameter counts, then exit.',
    )

    parser.add_argument('--d-model', type=int, default=4096)
    parser.add_argument('--model-type', type=str, choices=('transformer', 'mlp', 'linear'), default='transformer')#✅
    parser.add_argument('--d-tracelock', type=int, default=256)
    parser.add_argument('--d-tracelock-delta', type=int, default=32)
    parser.add_argument('--d-x', type=int, default=384)#✅
    parser.add_argument('--num-encoder-layers', type=int, default=5)#✅
    parser.add_argument('--num-attention-heads', type=int, default=8)#✅
    parser.add_argument('--encoder-ffn-dim', type=int, default=384 * 2)#✅

    parser.add_argument('--mlp-hidden-dim', type=int, default=None)
    parser.add_argument('--mlp-num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max-gen-len', type=int, default=256)
    parser.add_argument(
        '--position-encoding',
        dest='use_position_encoding',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Enable generation position embeddings inside TraceLock.',
    )
    parser.add_argument(
        '--state-encoding',
        dest='use_state_encoding',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Enable state embeddings inside TraceLock.',
    )
    parser.add_argument(
        '--dynamic-threshold',
        dest='use_dynamic_threshold',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Enable a CLS-conditioned dynamic threshold branch.',
    )
    parser.add_argument(
        '--dynamic-threshold-decision-only',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='When dynamic threshold is enabled, train only the decision logits and skip the auxiliary raw-logit BCE pass.',
    )

    parser.add_argument('--target-mask-prob', type=float, default=0) #✅
    parser.add_argument(
        '--target-mask-keep-loss-weight',
        type=float,
        default=1, #✅
        help='Extra multiplicative loss weight applied to STATE_GEN tokens that were not dropped by target_mask_prob.',
    )
    parser.add_argument(
        '--input-noise-ratio',
        type=float,
        default=0.0,
        help='Gaussian noise std as a ratio of the current batch hidden_layers std; train-only.',
    )
    parser.add_argument(
        '--feature-noise-snr-db',
        type=float,
        default=10, #✅
        help='Target SNR in dB for batch-sampled centered generation feature noise applied during training.',
    )
    parser.add_argument(
        '--feature-noise-apply-probability',
        type=float,
        default=0, #✅
        help='Probability of applying batch-sampled generation feature noise to each eligible token during training.',
    )
    parser.add_argument(
        '--enable-random-crop',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument('--random-crop-min-ratio', type=float, default=0.25)
    parser.add_argument('--random-crop-max-ratio', type=float, default=1.0)
    parser.add_argument(
        '--random-crop-apply-to-val',
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        '--val-threshold-scan-ratio',
        type=float,
        default=0.1,
        help='Fraction of validation tokens sampled to scan the best classification threshold before each eval.',
    )
    parser.add_argument(
        '--val-threshold-scan-metric',
        type=str,
        choices=('f1', 'accuracy'),
        default='f1',
        help='Objective used when selecting the best validation threshold from the scan subset.',
    )
    parser.add_argument(
        '--val-state2-only',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Restrict validation loss and metrics to STATE_GEN tokens only, excluding STATE_EOT.',
    )
    parser.add_argument(
        '--two-pass-reweight',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Run a first forward pass to find mispredicted tokens, then reweight them in a second training pass.',
    )
    parser.add_argument(
        '--two-pass-hard-weight',
        type=float,
        default=1.5,
        help='Extra multiplicative weight applied to hard tokens identified by the first pass when two_pass_reweight is enabled.',
    )
    parser.add_argument(
        '--token-class-balance',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='No-op flag. Class-balanced token reweighting is disabled in the current training recipe.',
    )
    parser.add_argument(
        '--token-class-balance-max-pos-weight',
        type=float,
        default=4.0,
        help='No-op argument kept for command-line stability; class-balanced token reweighting is disabled.',
    )
    parser.add_argument(
        '--negative-loss-weight',
        type=float,
        default=2,
        help='Extra multiplicative loss weight for negative STATE_GEN labels.',
    )
    parser.add_argument(
        '--frontier-positive-bias',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Apply position-dependent weighting on visible STATE_GEN positive tokens.',
    )
    parser.add_argument(
        '--frontier-positive-bias-strength',
        type=float,
        default=0.5,
        help='Late-positive decay strength. With the default 0.5, visible positive STATE_GEN weights linearly decay from 1.0 to 0.5.',
    )
    parser.add_argument(
        '--threshold-center-loss-weight',
        type=float,
        default=1e-3,
        help='L2 penalty weight that keeps dynamic threshold logits near 0 so sigmoid(threshold) stays near 0.5 unless the data strongly prefers otherwise.',
    )
    parser.add_argument(
        '--rollout-proxy-threshold',
        type=float,
        default=0.99,
        help='Fixed threshold used by the rollout-like validation proxy on cropped STATE_GEN tokens.',
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def _normalize_dataset_mix(dataset_mix: dict[str, float]) -> dict[Path, float]:
    normalized: dict[Path, float] = {}
    for raw_path, raw_weight in dataset_mix.items():
        path = Path(raw_path)
        weight = float(raw_weight)
        if weight <= 0.0:
            raise ValueError(f'Dataset weight must be > 0 for {path}, got {weight}.')
        normalized[path] = weight
    if not normalized:
        return {}
    weight_sum = sum(normalized.values())
    return {path: weight / weight_sum for path, weight in normalized.items()}


def get_configured_dataset_mix() -> dict[Path, float]:
    return _normalize_dataset_mix(TRAIN_DATASET_MIX)


def get_effective_dataset_mix(args: argparse.Namespace) -> dict[Path, float]:
    dataset_mix = get_configured_dataset_mix()
    if not dataset_mix:
        return {}
    if Path(args.samples_dir).resolve() != DEFAULT_SAMPLES_DIR.resolve():
        return {}
    return dataset_mix


def serialize_args(args: argparse.Namespace) -> dict[str, Any]:
    payload = base_serialize_args(args)
    dataset_mix = get_effective_dataset_mix(args)
    if dataset_mix:
        payload['train_dataset_mix'] = {str(path): weight for path, weight in dataset_mix.items()}
        payload['train_dataset_mix_mode'] = 'weighted_sampling_with_full_val_concat'
    else:
        payload['train_dataset_mix'] = {}
        payload['train_dataset_mix_mode'] = 'single_dataset'
    return payload


def _finalize_precomputed_input_layout(
    args: argparse.Namespace,
    train_dataset: TraceDirectoryStepDataset,
    val_dataset: TraceDirectoryStepDataset,
    stats: dict[str, Any],
) -> None:
    train_input_dim = int(train_dataset.precomputed_input_dim)
    val_input_dim = int(val_dataset.precomputed_input_dim)
    if train_input_dim != val_input_dim:
        raise ValueError(
            f'Train/val precomputed_input_dim mismatch: train={train_input_dim}, val={val_input_dim}.'
        )

    train_use_conf = bool(train_dataset.use_confidence_feature)
    val_use_conf = bool(val_dataset.use_confidence_feature)
    if train_use_conf != val_use_conf:
        raise ValueError(
            f'Train/val confidence feature mismatch: train={train_use_conf}, val={val_use_conf}.'
        )

    args.precomputed_input_dim = train_input_dim
    args.use_confidence_feature = train_use_conf
    stats['precomputed_input_dim'] = train_input_dim
    stats['use_confidence_feature'] = train_use_conf


def validate_args(args: argparse.Namespace) -> None:
    if not args.use_ae_data:
        raise ValueError('pretrain_slim.py expects AE-compressed step features; pass --use-ae-data or omit the flag.')
    if not 0.0 < args.random_crop_min_ratio <= args.random_crop_max_ratio <= 1.0:
        raise ValueError('random crop ratios must satisfy 0 < min <= max <= 1.')
    if not 0.0 < args.val_threshold_scan_ratio <= 1.0:
        raise ValueError('val_threshold_scan_ratio must satisfy 0 < val_threshold_scan_ratio <= 1.')
    if args.input_noise_ratio < 0.0:
        raise ValueError('input_noise_ratio must be >= 0.')
    if args.feature_noise_snr_db < 0.0:
        raise ValueError('feature_noise_snr_db must be >= 0.')
    if not 0.0 <= args.feature_noise_apply_probability <= 1.0:
        raise ValueError('feature_noise_apply_probability must satisfy 0 <= feature_noise_apply_probability <= 1.')
    if args.target_mask_keep_loss_weight <= 0.0:
        raise ValueError('target_mask_keep_loss_weight must be > 0.')
    if args.warmup_steps < 0:
        raise ValueError('warmup_steps must be >= 0.')
    if args.warmup_init_lr < 0.0:
        raise ValueError('warmup_init_lr must be >= 0.')
    if args.model_type == 'mlp' and args.mlp_num_layers <= 0:
        raise ValueError(f'mlp_num_layers must be positive, got {args.mlp_num_layers}.')
    if args.model_type == 'mlp' and args.mlp_hidden_dim is not None and args.mlp_hidden_dim <= 0:
        raise ValueError(f'mlp_hidden_dim must be positive, got {args.mlp_hidden_dim}.')
    if args.use_dynamic_threshold and args.model_type != 'transformer':
        raise ValueError('dynamic_threshold requires --model-type transformer.')
    if args.dynamic_threshold_decision_only and not args.use_dynamic_threshold:
        raise ValueError('dynamic_threshold_decision_only requires --dynamic-threshold.')
    if args.dynamic_threshold_decision_only and args.two_pass_reweight:
        raise ValueError('dynamic_threshold_decision_only cannot be combined with --two-pass-reweight.')
    if args.two_pass_hard_weight < 1.0:
        raise ValueError('two_pass_hard_weight must be >= 1.0.')
    if args.token_class_balance_max_pos_weight < 1.0:
        raise ValueError('token_class_balance_max_pos_weight must be >= 1.0.')
    if args.negative_loss_weight <= 0.0:
        raise ValueError('negative_loss_weight must be > 0.')
    if args.frontier_positive_bias_strength < 0.0:
        raise ValueError('frontier_positive_bias_strength must be >= 0.')
    if args.threshold_center_loss_weight < 0.0:
        raise ValueError('threshold_center_loss_weight must be >= 0.')
    if not 0.0 < args.rollout_proxy_threshold < 1.0:
        raise ValueError('rollout_proxy_threshold must satisfy 0 < rollout_proxy_threshold < 1.')
    dataset_mix = get_effective_dataset_mix(args)
    for samples_dir in dataset_mix:
        train_paths = list_split_sample_paths(samples_dir, 'train')
        val_paths = list_split_sample_paths(samples_dir, 'val')
        if not train_paths:
            raise ValueError(f'Configured mix dataset has no train samples: {samples_dir}')
        if not val_paths:
            raise ValueError(f'Configured mix dataset has no val samples: {samples_dir}')


def _resolve_local_batch_size(global_batch_size: int, world_size: int, name: str) -> int:
    if global_batch_size <= 0:
        raise ValueError(f'{name} must be positive, got {global_batch_size}.')
    if global_batch_size % world_size != 0:
        raise ValueError(
            f'{name}={global_batch_size} must be divisible by world_size={world_size} for distributed training.'
        )
    return global_batch_size // world_size


def _collect_mixed_split_paths(dataset_mix: dict[Path, float], split: str) -> tuple[list[Path], torch.Tensor, dict[str, Any]]:
    all_paths: list[Path] = []
    sample_weights: list[float] = []
    per_dataset_counts: dict[str, int] = {}
    normalized_weights = {str(path): weight for path, weight in dataset_mix.items()}

    for samples_dir, dataset_weight in dataset_mix.items():
        split_paths = list_split_sample_paths(samples_dir, split)
        per_dataset_counts[str(samples_dir)] = len(split_paths)
        all_paths.extend(split_paths)
        if split == 'train':
            sample_weights.extend([dataset_weight / len(split_paths)] * len(split_paths))

    stats = {
        f'{split}_dataset_weights': normalized_weights,
        f'{split}_dataset_sample_counts': per_dataset_counts,
        f'num_{split}_samples': len(all_paths),
    }
    weights_tensor = torch.tensor(sample_weights, dtype=torch.double) if split == 'train' else torch.empty(0, dtype=torch.double)
    return all_paths, weights_tensor, stats


def build_dataloaders(
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader, dict[str, int | float | dict[str, Any]], Sampler[int] | None, DistributedSampler | None]:
    dataset_mix = get_effective_dataset_mix(args)
    world_size = torch.distributed.get_world_size() if is_distributed() else 1
    train_batch_size = _resolve_local_batch_size(args.train_batch_size, world_size, 'train_batch_size')
    eval_batch_size = _resolve_local_batch_size(args.eval_batch_size, world_size, 'eval_batch_size')
    if not dataset_mix:
        train_paths = list_split_sample_paths(args.samples_dir, 'train')
        val_paths = list_split_sample_paths(args.samples_dir, 'val')
        train_dataset = TraceDirectoryStepDataset(args.samples_dir, train_paths)
        val_dataset = TraceDirectoryStepDataset(args.samples_dir, val_paths)

        train_sampler = None
        val_sampler = None
        if is_distributed():
            train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
            val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=args.num_workers,
            collate_fn=tracelock_step_collate,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.num_workers > 0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            collate_fn=tracelock_step_collate,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.num_workers > 0,
        )
        stats: dict[str, int | float | dict[str, Any]] = {
            'num_train_samples': len(train_paths),
            'num_val_samples': len(val_paths),
            'num_train_steps': len(train_dataset),
            'num_val_steps': len(val_dataset),
            'world_size': world_size,
            'train_batch_size_global': args.train_batch_size,
            'train_batch_size_per_rank': train_batch_size,
            'eval_batch_size_global': args.eval_batch_size,
            'eval_batch_size_per_rank': eval_batch_size,
            'train_sampling_mode': 'shuffle' if train_sampler is None else 'distributed_sampler',
            'val_sampling_mode': 'full_split',
        }
        _finalize_precomputed_input_layout(args, train_dataset, val_dataset, stats)
        return train_loader, val_loader, stats, train_sampler, val_sampler

    train_paths, train_weights, train_stats = _collect_mixed_split_paths(dataset_mix, 'train')
    val_paths, _, val_stats = _collect_mixed_split_paths(dataset_mix, 'val')
    train_dataset = TraceDirectoryStepDataset(train_paths)
    val_dataset = TraceDirectoryStepDataset(val_paths)

    if is_distributed():
        train_sampler = DistributedWeightedReplacementSampler(
            train_weights,
            num_replicas=world_size,
            rank=get_rank(),
            seed=args.seed,
        )
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
    else:
        train_sampler = WeightedReplacementSampler(train_weights, len(train_dataset), seed=args.seed)
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=tracelock_step_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        collate_fn=tracelock_step_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    stats = {
        **train_stats,
        **val_stats,
        'num_train_steps': len(train_dataset),
        'num_val_steps': len(val_dataset),
        'world_size': world_size,
        'train_batch_size_global': args.train_batch_size,
        'train_batch_size_per_rank': train_batch_size,
        'eval_batch_size_global': args.eval_batch_size,
        'eval_batch_size_per_rank': eval_batch_size,
        'train_sampling_mode': 'weighted_replacement',
        'val_sampling_mode': 'full_concat',
    }
    _finalize_precomputed_input_layout(args, train_dataset, val_dataset, stats)
    return train_loader, val_loader, stats, train_sampler, val_sampler


def build_two_pass_hard_mask(
    decision_logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    hard_mask_scope: torch.Tensor | None = None,
) -> torch.Tensor:
    predicted = decision_logits >= 0
    target = labels >= 0.5
    hard_mask = (predicted != target) & loss_mask
    if hard_mask_scope is not None:
        hard_mask = hard_mask & hard_mask_scope
    return hard_mask


def build_batch_class_weight(
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    class_balance_scope: torch.Tensor | None = None,
) -> torch.Tensor:
    effective_mask = loss_mask
    apply_mask = loss_mask
    if class_balance_scope is not None:
        effective_mask = effective_mask & class_balance_scope
        apply_mask = class_balance_scope
    valid_labels = labels[effective_mask]
    token_weight = torch.ones_like(labels)
    if valid_labels.numel() == 0:
        return token_weight

    pos_count = valid_labels.sum()
    neg_count = valid_labels.numel() - pos_count
    if pos_count.item() <= 0 or neg_count.item() <= 0:
        return token_weight

    pos_weight = neg_count / pos_count.clamp_min(1.0)
    neg_weight = pos_count / neg_count.clamp_min(1.0)
    balanced_weight = torch.where(
        labels >= 0.5,
        torch.full_like(labels, pos_weight),
        torch.full_like(labels, neg_weight),
    )
    return torch.where(apply_mask, balanced_weight, token_weight)


def cap_positive_class_weight(class_weight: torch.Tensor, labels: torch.Tensor, max_pos_weight: float) -> torch.Tensor:
    if max_pos_weight <= 1.0:
        return class_weight
    pos_mask = labels >= 0.5
    if not pos_mask.any():
        return class_weight
    capped_pos_weight = torch.clamp(class_weight, max=float(max_pos_weight))
    return torch.where(pos_mask, capped_pos_weight, class_weight)


def build_negative_class_weight(
    state_ids: torch.Tensor,
    labels: torch.Tensor,
    negative_loss_weight: float,
) -> torch.Tensor:
    class_weight = torch.ones_like(labels, dtype=torch.float32)
    if negative_loss_weight == 1.0:
        return class_weight
    negative_state2_mask = (state_ids == STATE_GEN) & (labels < 0.5)
    return torch.where(
        negative_state2_mask,
        torch.full_like(class_weight, float(negative_loss_weight)),
        class_weight,
    )


def build_frontier_positive_bias_weight(
    state_ids: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    token_weight = torch.ones_like(labels, dtype=torch.float32)
    if strength <= 0.0:
        return token_weight

    visible_state2_mask = loss_mask & (state_ids == STATE_GEN)
    positive_state2_mask = visible_state2_mask & (labels >= 0.5)
    if not positive_state2_mask.any():
        return token_weight

    relative_position = visible_state2_mask.long().cumsum(dim=1) - 1
    relative_position = relative_position.clamp(min=0)
    visible_count = visible_state2_mask.sum(dim=1, keepdim=True)
    max_relative_position = (visible_count - 1).clamp(min=1).to(torch.float32)
    normalized_position = relative_position.to(torch.float32) / max_relative_position
    positive_weight = 1.0 - float(strength) * normalized_position
    positive_weight = positive_weight.clamp(min=0.0)
    return torch.where(positive_state2_mask, positive_weight, token_weight)


def masked_weighted_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    token_weight: torch.Tensor,
    class_weight: torch.Tensor,
) -> torch.Tensor | None:
    effective_weight = loss_mask.float() * token_weight * class_weight
    weight_sum = effective_weight.sum()
    if weight_sum.item() <= 0:
        return None
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    return (loss * effective_weight).sum() / weight_sum


def compute_thresholded_binary_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    threshold: float,
    precision_beta: float = 0.5,
) -> dict[str, float]:
    valid_logits = logits.detach().float().cpu()
    valid_labels = labels.detach().float().cpu()
    if valid_logits.numel() == 0:
        return {
            'precision': float('nan'),
            'recall': float('nan'),
            'f1': float('nan'),
            'accuracy': float('nan'),
            'specificity': float('nan'),
            'false_positive_rate': float('nan'),
            'negative_predictive_value': float('nan'),
            'precision_weighted_fbeta': float('nan'),
            'accept_rate': float('nan'),
            'positive_rate': float('nan'),
            'token_count': 0.0,
            'true_positive_count': 0.0,
            'false_positive_count': 0.0,
            'false_negative_count': 0.0,
            'true_negative_count': 0.0,
        }

    probs = torch.sigmoid(valid_logits)
    preds = probs >= threshold
    positive = valid_labels >= 0.5
    negative = ~positive
    tp = (preds & positive).sum().item()
    fp = (preds & negative).sum().item()
    fn = ((~preds) & positive).sum().item()
    tn = ((~preds) & negative).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    false_positive_rate = fp / max(fp + tn, 1)
    negative_predictive_value = tn / max(tn + fn, 1)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    f1 = 0.0 if precision + recall == 0.0 else (2.0 * precision * recall) / (precision + recall)
    beta_sq = float(precision_beta) ** 2
    precision_weighted_fbeta = (
        0.0
        if precision + recall == 0.0
        else ((1.0 + beta_sq) * precision * recall) / ((beta_sq * precision) + recall)
    )
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'specificity': specificity,
        'false_positive_rate': false_positive_rate,
        'negative_predictive_value': negative_predictive_value,
        'precision_weighted_fbeta': precision_weighted_fbeta,
        'accept_rate': float(preds.float().mean().item()),
        'positive_rate': float(positive.float().mean().item()),
        'token_count': float(valid_logits.numel()),
        'true_positive_count': float(tp),
        'false_positive_count': float(fp),
        'false_negative_count': float(fn),
        'true_negative_count': float(tn),
    }


def _set_optimizer_lr(optimizer: AdamW, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def _warmup_lr(step: int, args: argparse.Namespace) -> float:
    if args.warmup_steps <= 0:
        return args.learning_rate
    progress = min(max(step, 0), args.warmup_steps) / args.warmup_steps
    return args.warmup_init_lr + (args.learning_rate - args.warmup_init_lr) * progress


def _compute_best_threshold_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    scan_ratio: float,
    scan_metric: str,
    seed: int,
) -> dict[str, float]:
    valid_logits = logits.detach().float().cpu()
    valid_labels = labels.detach().float().cpu()
    if valid_logits.numel() == 0:
        return {
            'best_threshold': float('nan'),
            'threshold_scan_count': 0.0,
            'accuracy_at_best_threshold': float('nan'),
            'precision_at_best_threshold': float('nan'),
            'recall_at_best_threshold': float('nan'),
            'f1_at_best_threshold': float('nan'),
            'threshold_objective_value': float('nan'),
        }

    probs = torch.sigmoid(valid_logits)
    total_count = probs.numel()
    scan_count = max(1, int(total_count * scan_ratio))
    rng = np.random.default_rng(seed)
    scan_indices = rng.choice(total_count, size=scan_count, replace=False)
    scan_probs = probs[scan_indices]
    scan_labels = valid_labels[scan_indices]

    best_threshold = 0.5
    best_objective = float('-inf')
    for threshold in torch.linspace(0.05, 0.95, steps=19):
        scan_preds = (scan_probs >= threshold).float()
        tp = ((scan_preds == 1) & (scan_labels == 1)).sum().item()
        fp = ((scan_preds == 1) & (scan_labels == 0)).sum().item()
        fn = ((scan_preds == 0) & (scan_labels == 1)).sum().item()
        tn = ((scan_preds == 0) & (scan_labels == 0)).sum().item()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        f1 = 0.0 if precision + recall == 0.0 else (2.0 * precision * recall) / (precision + recall)
        objective = f1 if scan_metric == 'f1' else accuracy
        if objective > best_objective:
            best_objective = objective
            best_threshold = float(threshold.item())

    preds = (probs >= best_threshold).float()
    tp = ((preds == 1) & (valid_labels == 1)).sum().item()
    fp = ((preds == 1) & (valid_labels == 0)).sum().item()
    fn = ((preds == 0) & (valid_labels == 1)).sum().item()
    tn = ((preds == 0) & (valid_labels == 0)).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    f1 = 0.0 if precision + recall == 0.0 else (2.0 * precision * recall) / (precision + recall)

    return {
        'best_threshold': best_threshold,
        'threshold_scan_count': float(scan_count),
        'accuracy_at_best_threshold': accuracy,
        'precision_at_best_threshold': precision,
        'recall_at_best_threshold': recall,
        'f1_at_best_threshold': f1,
        'threshold_objective_value': best_objective,
    }


@torch.no_grad()
def evaluate_slim(
    model: TraceLock | DDP,
    dataloader,
    device: str,
    args: argparse.Namespace,
    *,
    global_step: int = 0,
) -> dict[str, float]:
    model.eval()
    local_loss_sum = 0.0
    local_loss_weight_sum = 0.0
    local_logits: list[torch.Tensor] = []
    local_labels: list[torch.Tensor] = []
    local_threshold_logits: list[torch.Tensor] = []
    local_rollout_proxy_logits: list[torch.Tensor] = []
    local_rollout_proxy_labels: list[torch.Tensor] = []
    crop_stats = _empty_crop_stats(device)

    for batch_idx, batch in enumerate(dataloader):
        batch = move_batch_to_device(batch, device)
        hidden_layers = batch['hidden_layers']
        attention_mask, loss_mask, batch_crop_stats = apply_random_prefix_crop(
            batch,
            args,
            apply_crop=args.enable_random_crop and args.random_crop_apply_to_val,
        )
        crop_stats = merge_crop_stats(crop_stats, batch_crop_stats)
        if args.val_state2_only:
            loss_mask = loss_mask & (batch['state'] == STATE_GEN)
        token_weight = build_eot_loss_weight(batch['state'])
        outputs = model(
            hidden_layers=hidden_layers,
            state_ids=batch['state'],
            attention_mask=attention_mask,
        )
        eval_logits = outputs['decision_logits'] if args.use_dynamic_threshold else outputs['logits']
        effective_weight = loss_mask.float() * token_weight
        weight_sum = effective_weight.sum()
        if weight_sum.item() <= 0:
            continue
        token_loss = F.binary_cross_entropy_with_logits(eval_logits, batch['label'], reduction='none')
        local_loss_sum += float((token_loss * effective_weight).sum().item())
        local_loss_weight_sum += float(weight_sum.item())
        local_logits.append(eval_logits[loss_mask].detach().cpu())
        local_labels.append(batch['label'][loss_mask].detach().cpu())
        if args.use_dynamic_threshold:
            local_threshold_logits.append(outputs['threshold_logit'].detach().cpu())

        rollout_rng_devices: list[int] = []
        if batch['sequence_mask'].is_cuda:
            device_index = batch['sequence_mask'].device.index
            rollout_rng_devices = [torch.cuda.current_device() if device_index is None else device_index]
        with torch.random.fork_rng(devices=rollout_rng_devices):
            torch.manual_seed(args.seed + global_step * 100000 + batch_idx)
            rollout_proxy_attention_mask, _, _ = apply_random_prefix_crop(
                batch,
                args,
                apply_crop=args.enable_random_crop,
            )
        rollout_proxy_loss_mask = build_state_loss_mask(batch['state'], rollout_proxy_attention_mask) & (batch['state'] == STATE_GEN)
        if bool(rollout_proxy_loss_mask.any().item()):
            rollout_proxy_outputs = model(
                hidden_layers=hidden_layers,
                state_ids=batch['state'],
                attention_mask=rollout_proxy_attention_mask,
            )
            rollout_proxy_logits = (
                rollout_proxy_outputs['decision_logits']
                if args.use_dynamic_threshold
                else rollout_proxy_outputs['logits']
            )
            local_rollout_proxy_logits.append(rollout_proxy_logits[rollout_proxy_loss_mask].detach().cpu())
            local_rollout_proxy_labels.append(batch['label'][rollout_proxy_loss_mask].detach().cpu())

    loss_stats = torch.tensor([local_loss_sum, local_loss_weight_sum], dtype=torch.float64, device=device)
    if is_distributed():
        torch.distributed.all_reduce(loss_stats, op=torch.distributed.ReduceOp.SUM)

    global_loss = float('nan')
    if loss_stats[1].item() > 0:
        global_loss = float((loss_stats[0] / loss_stats[1]).item())
    crop_metrics = reduce_crop_stats(crop_stats)

    local_payload = {
        'logits': torch.cat(local_logits, dim=0) if local_logits else torch.empty(0, dtype=torch.float32),
        'labels': torch.cat(local_labels, dim=0) if local_labels else torch.empty(0, dtype=torch.float32),
        'threshold_logits': (
            torch.cat(local_threshold_logits, dim=0)
            if local_threshold_logits
            else torch.empty(0, dtype=torch.float32)
        ),
        'rollout_proxy_logits': (
            torch.cat(local_rollout_proxy_logits, dim=0)
            if local_rollout_proxy_logits
            else torch.empty(0, dtype=torch.float32)
        ),
        'rollout_proxy_labels': (
            torch.cat(local_rollout_proxy_labels, dim=0)
            if local_rollout_proxy_labels
            else torch.empty(0, dtype=torch.float32)
        ),
    }

    if is_distributed():
        gathered_payloads: list[dict[str, torch.Tensor] | None] = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(gathered_payloads, local_payload)
    else:
        gathered_payloads = [local_payload]

    if is_main_process():
        logits_parts = [item['logits'] for item in gathered_payloads if item is not None and item['logits'].numel() > 0]
        labels_parts = [item['labels'] for item in gathered_payloads if item is not None and item['labels'].numel() > 0]
        threshold_parts = [
            item['threshold_logits']
            for item in gathered_payloads
            if item is not None and item['threshold_logits'].numel() > 0
        ]
        rollout_proxy_logits_parts = [
            item['rollout_proxy_logits']
            for item in gathered_payloads
            if item is not None and item['rollout_proxy_logits'].numel() > 0
        ]
        rollout_proxy_labels_parts = [
            item['rollout_proxy_labels']
            for item in gathered_payloads
            if item is not None and item['rollout_proxy_labels'].numel() > 0
        ]
        if logits_parts:
            all_logits = torch.cat(logits_parts, dim=0)
            all_labels = torch.cat(labels_parts, dim=0)
            metrics = compute_metrics(all_logits, all_labels)
            if not args.use_dynamic_threshold:
                metrics.update(
                    _compute_best_threshold_metrics(
                        all_logits,
                        all_labels,
                        scan_ratio=args.val_threshold_scan_ratio,
                        scan_metric=args.val_threshold_scan_metric,
                        seed=args.seed + global_step,
                    )
                )
        else:
            metrics = compute_metrics(torch.empty(0), torch.empty(0))
            if not args.use_dynamic_threshold:
                metrics.update(
                    _compute_best_threshold_metrics(
                        torch.empty(0),
                        torch.empty(0),
                        scan_ratio=args.val_threshold_scan_ratio,
                        scan_metric=args.val_threshold_scan_metric,
                        seed=args.seed + global_step,
                    )
                )
        if threshold_parts:
            all_threshold_logits = torch.cat(threshold_parts, dim=0)
            metrics['mean_threshold_logit'] = float(all_threshold_logits.mean().item())
            metrics['std_threshold_logit'] = float(all_threshold_logits.std(unbiased=False).item())
            metrics['mean_threshold_prob'] = float(torch.sigmoid(all_threshold_logits).mean().item())
        else:
            metrics['mean_threshold_logit'] = float('nan')
            metrics['std_threshold_logit'] = float('nan')
            metrics['mean_threshold_prob'] = float('nan')
        if rollout_proxy_logits_parts:
            rollout_proxy_logits = torch.cat(rollout_proxy_logits_parts, dim=0)
            rollout_proxy_labels = torch.cat(rollout_proxy_labels_parts, dim=0)
            rollout_proxy_metrics = compute_thresholded_binary_metrics(
                rollout_proxy_logits,
                rollout_proxy_labels,
                threshold=args.rollout_proxy_threshold,
            )
        else:
            rollout_proxy_metrics = compute_thresholded_binary_metrics(
                torch.empty(0),
                torch.empty(0),
                threshold=args.rollout_proxy_threshold,
            )
        if logits_parts:
            metrics['accept_rate_at_0_5'] = float((torch.sigmoid(all_logits) >= 0.5).float().mean().item())
        else:
            metrics['accept_rate_at_0_5'] = float('nan')
        if args.use_dynamic_threshold:
            metrics.pop('best_threshold', None)
            metrics.pop('threshold_scan_count', None)
            metrics.pop('accuracy_at_best_threshold', None)
            metrics.pop('precision_at_best_threshold', None)
            metrics.pop('recall_at_best_threshold', None)
            metrics.pop('f1_at_best_threshold', None)
            metrics.pop('threshold_objective_value', None)
        metrics['rollout_proxy_threshold'] = float(args.rollout_proxy_threshold)
        metrics['rollout_proxy_precision'] = rollout_proxy_metrics['precision']
        metrics['rollout_proxy_recall'] = rollout_proxy_metrics['recall']
        metrics['rollout_proxy_f1'] = rollout_proxy_metrics['f1']
        metrics['rollout_proxy_f0_5'] = rollout_proxy_metrics['precision_weighted_fbeta']
        metrics['rollout_proxy_accuracy'] = rollout_proxy_metrics['accuracy']
        metrics['rollout_proxy_specificity'] = rollout_proxy_metrics['specificity']
        metrics['rollout_proxy_false_positive_rate'] = rollout_proxy_metrics['false_positive_rate']
        metrics['rollout_proxy_negative_predictive_value'] = rollout_proxy_metrics['negative_predictive_value']
        metrics['rollout_proxy_accept_rate'] = rollout_proxy_metrics['accept_rate']
        metrics['rollout_proxy_positive_rate'] = rollout_proxy_metrics['positive_rate']
        metrics['rollout_proxy_token_count'] = rollout_proxy_metrics['token_count']
        metrics['rollout_proxy_true_positive_count'] = rollout_proxy_metrics['true_positive_count']
        metrics['rollout_proxy_false_positive_count'] = rollout_proxy_metrics['false_positive_count']
        metrics['rollout_proxy_false_negative_count'] = rollout_proxy_metrics['false_negative_count']
        metrics['rollout_proxy_true_negative_count'] = rollout_proxy_metrics['true_negative_count']
        metrics['loss'] = global_loss
        metrics.update(crop_metrics)
    else:
        metrics = {
            'loss': global_loss,
            'auroc': float('nan'),
            'average_precision': float('nan'),
            'accuracy_at_0_5': float('nan'),
            'positive_rate': float('nan'),
            'gen_token_count': 0.0,
            'mean_threshold_logit': float('nan'),
            'std_threshold_logit': float('nan'),
            'mean_threshold_prob': float('nan'),
            'accept_rate_at_0_5': float('nan'),
            'crop_ratio_mean': float('nan'),
            'crop_ratio_min': float('nan'),
            'crop_ratio_max': float('nan'),
            'visible_gen_tokens_mean': float('nan'),
            'rollout_proxy_threshold': float(args.rollout_proxy_threshold),
            'rollout_proxy_precision': float('nan'),
            'rollout_proxy_recall': float('nan'),
            'rollout_proxy_f1': float('nan'),
            'rollout_proxy_f0_5': float('nan'),
            'rollout_proxy_accuracy': float('nan'),
            'rollout_proxy_specificity': float('nan'),
            'rollout_proxy_false_positive_rate': float('nan'),
            'rollout_proxy_negative_predictive_value': float('nan'),
            'rollout_proxy_accept_rate': float('nan'),
            'rollout_proxy_positive_rate': float('nan'),
            'rollout_proxy_token_count': 0.0,
            'rollout_proxy_true_positive_count': 0.0,
            'rollout_proxy_false_positive_count': 0.0,
            'rollout_proxy_false_negative_count': 0.0,
            'rollout_proxy_true_negative_count': 0.0,
        }
        if not args.use_dynamic_threshold:
            metrics.update(
                {
                    'best_threshold': float('nan'),
                    'threshold_scan_count': 0.0,
                    'accuracy_at_best_threshold': float('nan'),
                    'precision_at_best_threshold': float('nan'),
                    'recall_at_best_threshold': float('nan'),
                    'f1_at_best_threshold': float('nan'),
                    'threshold_objective_value': float('nan'),
                }
            )

    if is_distributed():
        metrics_box = [metrics]
        torch.distributed.broadcast_object_list(metrics_box, src=0)
        metrics = metrics_box[0]

    return metrics


def _format_param_count(value: int) -> str:
    return format(value, ',')


def build_training_model(args: argparse.Namespace, *, print_summary: bool = True) -> TraceLock:
    config = make_config(args)
    if args.use_ae_data:
        config.enable_input_projection_stack = False
    model = build_tracelock(
        config,
        projection_checkpoint=args.pretrained_proj_checkpoint if config.enable_input_projection_stack else None,
        freeze_projection=args.freeze_pretrained_proj,
        print_summary=print_summary,
        summary_title='TraceLock',
    )
    if config.enable_input_projection_stack and args.use_ae_data:
        freeze_projection_stack(model)
    return model


def maybe_load_init_checkpoint(model: TraceLock, init_checkpoint: Path | None) -> None:
    if init_checkpoint is None:
        return
    checkpoint = torch.load(init_checkpoint, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model', checkpoint)
    model.load_state_dict(state_dict)


def print_model_summary(args: argparse.Namespace) -> None:
    model = build_training_model(args, print_summary=False)
    print_tracelock_summary(model, title='TraceLock')


def run_training(
    args: argparse.Namespace,
    *,
    dataloader_builder=build_dataloaders,
    extra_run_metadata: dict[str, Any] | None = None,
) -> Path:
    validate_args(args)
    if args.print_model_summary_only:
        print_model_summary(args)
        return args.output_root / args.run_name

    device, local_rank = setup_distributed(args)
    seed = args.seed + get_rank()
    set_global_seed(seed)

    run_dir = ensure_dir(args.output_root / args.run_name)
    train_loader, val_loader, data_stats, train_sampler, _ = dataloader_builder(args)
    if is_main_process():
        write_json(run_dir / 'config.json', serialize_args(args))
        if extra_run_metadata is not None:
            write_json(run_dir / 'lineage.json', extra_run_metadata)
        write_json(run_dir / 'data_stats.json', data_stats)

    model = build_training_model(args).to(device)
    maybe_load_init_checkpoint(model, getattr(args, 'init_checkpoint', None))
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.warmup_steps > 0:
        _set_optimizer_lr(optimizer, args.warmup_init_lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.lr_reduce_patience)

    train_state = {
        'global_step': 0,
        'best_rollout_proxy_f0_5': float('-inf'),
        'best_val_average_precision': float('-inf'),
        'best_rollout_proxy_f1': float('-inf'),
        'epochs_completed': 0,
        'no_improve_evals': 0,
        'world_size': torch.distributed.get_world_size() if is_distributed() else 1,
    }

    if is_distributed():
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=ddp_requires_unused_parameter_detection(unwrap_model(model).config),
        )

    if train_sampler is not None:
        train_sampler.set_epoch(train_state['epochs_completed'])
    train_iter = iter(train_loader)

    train_log_every = 4000
    train_log_loss_sum = 0.0
    train_log_count = 0
    train_log_noise_std_sum = 0.0
    train_log_feature_noise_ratio_sum = 0.0
    train_log_two_pass_hard_ratio_sum = 0.0
    train_log_threshold_loss_sum = 0.0
    train_log_threshold_center_loss_sum = 0.0
    train_log_token_loss_sum = 0.0
    train_crop_stats = _empty_crop_stats(device)

    try:
        while train_state['global_step'] < args.max_steps:
            model.train()
            optimizer.zero_grad(set_to_none=True)

            try:
                batch = next(train_iter)
            except StopIteration:
                train_state['epochs_completed'] += 1
                if train_sampler is not None:
                    train_sampler.set_epoch(train_state['epochs_completed'])
                train_iter = iter(train_loader)
                batch = next(train_iter)

            batch = move_batch_to_device(batch, device)
            hidden_layers, noise_std = apply_relative_input_noise(batch['hidden_layers'], args.input_noise_ratio)
            hidden_layers, feature_noise_ratio = apply_batch_feature_noise(
                hidden_layers,
                batch['state'],
                snr_db=args.feature_noise_snr_db,
                apply_probability=args.feature_noise_apply_probability,
            )
            attention_mask, loss_mask, batch_crop_stats = apply_random_prefix_crop(
                batch,
                args,
                apply_crop=args.enable_random_crop,
            )
            token_weight = build_eot_loss_weight(batch['state'])
            state2_reweight_mask = loss_mask & (batch['state'] == STATE_GEN)
            two_pass_hard_ratio = 0.0
            threshold_loss_value = float('nan')
            threshold_center_loss_value = float('nan')
            token_loss_value = float('nan')

            if args.use_dynamic_threshold:
                threshold_outputs = model(
                    hidden_layers=hidden_layers,
                    state_ids=batch['state'],
                    attention_mask=attention_mask,
                )
                threshold_token_weight = token_weight * build_target_keep_loss_weight(
                    batch['state'],
                    threshold_outputs.get('target_mask_kept'),
                    args.target_mask_keep_loss_weight,
                )
                if args.frontier_positive_bias:
                    threshold_token_weight = threshold_token_weight * build_frontier_positive_bias_weight(
                        batch['state'],
                        batch['label'],
                        loss_mask,
                        args.frontier_positive_bias_strength,
                    )
                threshold_class_weight = build_negative_class_weight(
                    batch['state'],
                    batch['label'],
                    args.negative_loss_weight,
                )
                main_decision_logits = threshold_outputs['logits'] - threshold_outputs['threshold_logit'].unsqueeze(1)
                threshold_loss = masked_weighted_bce_loss(
                    main_decision_logits,
                    batch['label'],
                    loss_mask,
                    threshold_token_weight,
                    threshold_class_weight,
                )
                if threshold_loss is None:
                    continue
                threshold_center_loss = threshold_outputs['threshold_logit'].pow(2).mean()

                hard_mask = build_two_pass_hard_mask(
                    main_decision_logits.detach(),
                    batch['label'],
                    loss_mask,
                    hard_mask_scope=state2_reweight_mask,
                )
                hard_count = int(hard_mask.sum().item())
                loss_count = int(state2_reweight_mask.sum().item())
                two_pass_hard_ratio = (hard_count / loss_count) if loss_count else 0.0

                if args.dynamic_threshold_decision_only:
                    loss = threshold_loss + args.threshold_center_loss_weight * threshold_center_loss
                else:
                    token_outputs = model(
                        hidden_layers=hidden_layers,
                        state_ids=batch['state'],
                        attention_mask=attention_mask,
                    )
                    token_weight = token_weight * build_target_keep_loss_weight(
                        batch['state'],
                        token_outputs.get('target_mask_kept'),
                        args.target_mask_keep_loss_weight,
                    )
                    if args.frontier_positive_bias:
                        token_weight = token_weight * build_frontier_positive_bias_weight(
                            batch['state'],
                            batch['label'],
                            loss_mask,
                            args.frontier_positive_bias_strength,
                        )
                    token_class_weight = build_negative_class_weight(
                        batch['state'],
                        batch['label'],
                        args.negative_loss_weight,
                    )
                    if args.two_pass_reweight:
                        hard_weight = torch.where(
                            hard_mask,
                            torch.full_like(token_weight, float(args.two_pass_hard_weight)),
                            torch.ones_like(token_weight),
                        )
                        token_weight = token_weight * hard_weight

                    token_loss = masked_weighted_bce_loss(
                        token_outputs['logits'],
                        batch['label'],
                        loss_mask,
                        token_weight,
                        token_class_weight,
                    )
                    if token_loss is None:
                        continue

                    loss = (
                        threshold_loss
                        + token_loss
                        + args.threshold_center_loss_weight * threshold_center_loss
                    )
                    token_loss_value = float(token_loss.item())
                threshold_loss_value = float(threshold_loss.item())
                threshold_center_loss_value = float(threshold_center_loss.item())
            else:
                if args.two_pass_reweight:
                    with torch.no_grad():
                        first_pass_outputs = model(
                            hidden_layers=hidden_layers,
                            state_ids=batch['state'],
                            attention_mask=attention_mask,
                        )
                        hard_mask = build_two_pass_hard_mask(
                            first_pass_outputs['logits'],
                            batch['label'],
                            loss_mask,
                            hard_mask_scope=state2_reweight_mask,
                        )
                        hard_count = int(hard_mask.sum().item())
                        loss_count = int(state2_reweight_mask.sum().item())
                        two_pass_hard_ratio = (hard_count / loss_count) if loss_count else 0.0

                    outputs = model(
                        hidden_layers=hidden_layers,
                        state_ids=batch['state'],
                        attention_mask=attention_mask,
                    )
                    token_weight = token_weight * build_target_keep_loss_weight(
                        batch['state'],
                        outputs.get('target_mask_kept'),
                        args.target_mask_keep_loss_weight,
                    )
                    if args.frontier_positive_bias:
                        token_weight = token_weight * build_frontier_positive_bias_weight(
                            batch['state'],
                            batch['label'],
                            loss_mask,
                            args.frontier_positive_bias_strength,
                        )
                    token_class_weight = build_negative_class_weight(
                        batch['state'],
                        batch['label'],
                        args.negative_loss_weight,
                    )
                    hard_weight = torch.where(
                        hard_mask,
                        torch.full_like(token_weight, float(args.two_pass_hard_weight)),
                        torch.ones_like(token_weight),
                    )
                    token_weight = token_weight * hard_weight
                else:
                    outputs = model(
                        hidden_layers=hidden_layers,
                        state_ids=batch['state'],
                        attention_mask=attention_mask,
                    )
                    token_weight = token_weight * build_target_keep_loss_weight(
                        batch['state'],
                        outputs.get('target_mask_kept'),
                        args.target_mask_keep_loss_weight,
                    )
                    if args.frontier_positive_bias:
                        token_weight = token_weight * build_frontier_positive_bias_weight(
                            batch['state'],
                            batch['label'],
                            loss_mask,
                            args.frontier_positive_bias_strength,
                        )
                    token_class_weight = build_negative_class_weight(
                        batch['state'],
                        batch['label'],
                        args.negative_loss_weight,
                    )

                loss = masked_weighted_bce_loss(
                    outputs['logits'],
                    batch['label'],
                    loss_mask,
                    token_weight,
                    token_class_weight,
                )
                token_loss_value = float(loss.item()) if loss is not None else float('nan')
            if loss is None:
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            train_state['global_step'] += 1
            if args.warmup_steps > 0 and train_state['global_step'] <= args.warmup_steps:
                _set_optimizer_lr(optimizer, _warmup_lr(train_state['global_step'], args))

            save_every = int(getattr(args, 'save_every', 0))
            if save_every > 0 and train_state['global_step'] % save_every == 0 and is_main_process():
                save_checkpoint(
                    run_dir / f"Step_{train_state['global_step']}.pt",
                    model,
                    optimizer,
                    scheduler,
                    train_state,
                )
                write_json(run_dir / 'train_state.json', train_state)

            mean_train_loss = reduce_mean(float(loss.item()), device)
            train_log_loss_sum += mean_train_loss
            train_log_count += 1
            train_log_noise_std_sum += reduce_mean(noise_std, device)
            train_log_feature_noise_ratio_sum += reduce_mean(feature_noise_ratio, device)
            train_log_two_pass_hard_ratio_sum += reduce_mean(two_pass_hard_ratio, device)
            if args.use_dynamic_threshold:
                train_log_threshold_loss_sum += reduce_mean(threshold_loss_value, device)
                train_log_threshold_center_loss_sum += reduce_mean(threshold_center_loss_value, device)
            train_log_token_loss_sum += reduce_mean(token_loss_value, device)
            train_crop_stats = merge_crop_stats(train_crop_stats, batch_crop_stats)

            if train_state['global_step'] % train_log_every == 0:
                train_crop_metrics = reduce_crop_stats(train_crop_stats)
                if is_main_process():
                    append_jsonl(
                        run_dir / 'metrics.jsonl',
                        round_metric_values({
                            'split': 'train',
                            'step': train_state['global_step'],
                            'train_loss': train_log_loss_sum / train_log_count,
                            'input_noise_std': train_log_noise_std_sum / train_log_count,
                            'feature_noise_applied_ratio': train_log_feature_noise_ratio_sum / train_log_count,
                            'two_pass_hard_ratio': train_log_two_pass_hard_ratio_sum / train_log_count,
                            'learning_rate': optimizer.param_groups[0]['lr'],
                            'log_window_steps': train_log_count,
                            'threshold_loss': train_log_threshold_loss_sum / train_log_count if args.use_dynamic_threshold else float('nan'),
                            'threshold_center_loss': train_log_threshold_center_loss_sum / train_log_count if args.use_dynamic_threshold else float('nan'),
                            'token_loss': train_log_token_loss_sum / train_log_count,
                            **train_crop_metrics,
                        }),
                    )
                train_log_loss_sum = 0.0
                train_log_count = 0
                train_log_noise_std_sum = 0.0
                train_log_feature_noise_ratio_sum = 0.0
                train_log_two_pass_hard_ratio_sum = 0.0
                train_log_threshold_loss_sum = 0.0
                train_log_threshold_center_loss_sum = 0.0
                train_log_token_loss_sum = 0.0
                train_crop_stats = _empty_crop_stats(device)

            if train_state['global_step'] % args.eval_every == 0:
                val_metrics = evaluate_slim(model, val_loader, device, args, global_step=train_state['global_step'])
                if train_state['global_step'] >= args.warmup_steps:
                    scheduler.step(val_metrics['loss'])
                if is_main_process():
                    append_jsonl(
                        run_dir / 'metrics.jsonl',
                        round_metric_values({'split': 'val', 'step': train_state['global_step'], **val_metrics}),
                    )

                improved = False
                if val_metrics['rollout_proxy_f0_5'] > train_state['best_rollout_proxy_f0_5']:
                    train_state['best_rollout_proxy_f0_5'] = val_metrics['rollout_proxy_f0_5']
                    if is_main_process():
                        save_checkpoint(run_dir / 'best_rollout_proxy_f0_5.pt', model, optimizer, scheduler, train_state)
                    improved = True
                if val_metrics['average_precision'] > train_state['best_val_average_precision']:
                    train_state['best_val_average_precision'] = val_metrics['average_precision']
                    if is_main_process():
                        save_checkpoint(run_dir / 'best_val_average_precision.pt', model, optimizer, scheduler, train_state)
                    improved = True
                if val_metrics['rollout_proxy_f1'] > train_state['best_rollout_proxy_f1']:
                    train_state['best_rollout_proxy_f1'] = val_metrics['rollout_proxy_f1']
                    if is_main_process():
                        save_checkpoint(run_dir / 'best_rollout_proxy_f1.pt', model, optimizer, scheduler, train_state)
                    improved = True

                if improved:
                    train_state['no_improve_evals'] = 0
                else:
                    train_state['no_improve_evals'] += 1

                if is_main_process():
                    write_json(run_dir / 'train_state.json', train_state)

                if train_state['no_improve_evals'] >= args.early_stop_patience:
                    if is_main_process():
                        print(f"Early stopping at step {train_state['global_step']}")
                    break

        if train_log_count > 0:
            train_crop_metrics = reduce_crop_stats(train_crop_stats)
            if is_main_process():
                append_jsonl(
                    run_dir / 'metrics.jsonl',
                    round_metric_values({
                        'split': 'train',
                        'step': train_state['global_step'],
                        'train_loss': train_log_loss_sum / train_log_count,
                        'input_noise_std': train_log_noise_std_sum / train_log_count,
                        'feature_noise_applied_ratio': train_log_feature_noise_ratio_sum / train_log_count,
                        'two_pass_hard_ratio': train_log_two_pass_hard_ratio_sum / train_log_count,
                        'learning_rate': optimizer.param_groups[0]['lr'],
                        'log_window_steps': train_log_count,
                        'threshold_loss': train_log_threshold_loss_sum / train_log_count if args.use_dynamic_threshold else float('nan'),
                        'threshold_center_loss': train_log_threshold_center_loss_sum / train_log_count if args.use_dynamic_threshold else float('nan'),
                        'token_loss': train_log_token_loss_sum / train_log_count,
                        **train_crop_metrics,
                    }),
                )

        if is_main_process():
            write_json(run_dir / 'train_state.json', train_state)
            print('training finished', train_state)
    finally:
        cleanup_distributed()

    return run_dir


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == '__main__':
    main()
