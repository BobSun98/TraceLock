from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracelock.common.project import SCRATCH_ROOT, setup_project_root

setup_project_root(PROJECT_ROOT)

from tracelock.common.io_utils import append_jsonl, ensure_dir, write_json
from tracelock.common.random_utils import set_global_seed
from tracelock.data.trace_dataset import (
    TraceDirectoryStepDataset,
    list_split_sample_paths,
    tracelock_step_collate,
)
from tracelock.models.tracelock import TraceLock, TraceLockConfig, build_tracelock


from sklearn.metrics import average_precision_score, roc_auc_score


DEFAULT_SAMPLES_DIR = SCRATCH_ROOT / 'pretrain' / 'samples'
DEFAULT_OUTPUT_ROOT = SCRATCH_ROOT / 'checkpoints'
STATE_PROMPT = 0
STATE_LOCKED = 1
STATE_GEN = 2
STATE_EOT = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Pretrain the TraceLock model with optional DDP.')
    parser.add_argument('--samples-dir', type=Path, default=DEFAULT_SAMPLES_DIR)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--run-name', type=str, default='tracelock')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--use-ae-data',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Expect step samples to provide precomputed AE features in `x`. This training path requires AE features.',
    )

    parser.add_argument('--train-batch-size', type=int, default=3, help='Global train batch size across all ranks.')
    parser.add_argument('--eval-batch-size', type=int, default=16, help='Global eval batch size across all ranks.')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers per rank.')

    parser.add_argument('--grad-accum-steps', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=8000)
    parser.add_argument('--eval-every', type=int, default=400)
    parser.add_argument('--save-every', type=int, default=50000000 + 1)

    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--loss-type', type=str, choices=('bce', 'focal'), default='bce')
    parser.add_argument('--focal-gamma', type=float, default=4)
    parser.add_argument(
        '--focal-pos-weight',
        type=float,
        default=1.0,
        help='Multiplicative weight for positive labels when using focal loss.',
    )
    parser.add_argument(
        '--focal-neg-weight',
        type=float,
        default=2,
        help='Multiplicative weight for negative labels when using focal loss.',
    )

    parser.add_argument('--lr-reduce-patience', type=int, default=2)
    parser.add_argument('--early-stop-patience', type=int, default=8)

    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--resume', type=Path, default=None)
    parser.add_argument('--pretrained-proj-checkpoint', type=Path, default=SCRATCH_ROOT / 'checkpoints' / 'dream-ae-v1' / 'best_val_loss.pt')
    parser.add_argument('--freeze-pretrained-proj', default=True)
    parser.add_argument('--backend', type=str, default='nccl')
    parser.add_argument('--local-rank', type=int, default=-1)

    parser.add_argument('--d-model', type=int, default=4096)
    parser.add_argument('--model-type', type=str, choices=('transformer', 'mlp', 'linear'), default='transformer')
    parser.add_argument('--d-tracelock', type=int, default=256)
    parser.add_argument('--d-tracelock-delta', type=int, default=32)
    parser.add_argument('--d-x', type=int, default=256)
    parser.add_argument('--num-encoder-layers', type=int, default=2)
    parser.add_argument('--num-attention-heads', type=int, default=4)
    parser.add_argument('--encoder-ffn-dim', type=int, default=768)
    parser.add_argument('--mlp-hidden-dim', type=int, default=None)
    parser.add_argument('--mlp-num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max-gen-len', type=int, default=512)
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
        default=False,
        help='Enable a CLS-conditioned dynamic decision threshold head.',
    )

    parser.add_argument('--target-mask-prob', type=float, default=0.5)
    parser.add_argument(
        '--target-mask-keep-loss-weight',
        type=float,
        default=1.0,
        help='Extra multiplicative loss weight applied to STATE_GEN tokens that were not dropped by target_mask_prob.',
    )
    parser.add_argument(
        '--input-noise-ratio',
        type=float,
        default=0.5,
        help='Gaussian noise std as a ratio of the current batch hidden_layers std; train-only.',
    )
    parser.add_argument(
        '--feature-ablation',
        type=str,
        choices=('none', 'zero'),
        default='none',
        help='Optional ablation applied to hidden_layers before the model forward pass.',
    )
    parser.add_argument('--enable-random-crop', default= True )
    parser.add_argument('--random-crop-min-ratio', type=float, default=0.25)
    parser.add_argument('--random-crop-max-ratio', type=float, default=1.0)
    parser.add_argument(
        '--random-crop-apply-to-val',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        '--feature-ablation-apply-to-val',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Apply feature ablation during validation as well as training.',
    )
    parser.add_argument(
        '--position-loss-decay',
        type=float,
        default=0.99,
        help=(
            'Exponential decay base for generation-region position weighting. Relative positions are counted across '
            'all non-prompt tokens. STATE_GEN and STATE_EOT tokens contribute to the loss.'
        ),
    )
    parser.add_argument(
        '--position-loss-decay-apply-to-val',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Apply generation-region position loss decay during validation as well as training.',
    )
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> TraceLockConfig:
    return TraceLockConfig(
        model_type=args.model_type,
        d_model=args.d_model,
        d_tracelock=args.d_tracelock,
        d_tracelock_delta=args.d_tracelock_delta,
        d_x=args.d_x,
        precomputed_input_dim=getattr(args, 'precomputed_input_dim', None),
        max_gen_len=args.max_gen_len,
        num_encoder_layers=args.num_encoder_layers,
        num_attention_heads=args.num_attention_heads,
        encoder_ffn_dim=args.encoder_ffn_dim,
        dropout=args.dropout,
        target_mask_prob=args.target_mask_prob,
        use_position_encoding=args.use_position_encoding,
        use_state_encoding=args.use_state_encoding,
        mlp_hidden_dim=args.mlp_hidden_dim if args.mlp_hidden_dim is not None else args.d_x,
        mlp_num_layers=args.mlp_num_layers,
        use_dynamic_threshold=args.use_dynamic_threshold,
    )


def serialize_args(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        else:
            payload[key] = value
    return payload


def round_metric_values(payload: dict[str, Any], digits: int = 4) -> dict[str, Any]:
    rounded: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, float):
            rounded[key] = round(value, digits)
        else:
            rounded[key] = value
    return rounded


def masked_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    position_weight: torch.Tensor,
) -> torch.Tensor | None:
    token_weight = loss_mask.float() * position_weight
    weight_sum = token_weight.sum()
    if weight_sum.item() == 0:
        return None
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    return (loss * token_weight).sum() / weight_sum


def masked_focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    position_weight: torch.Tensor,
    gamma: float,
    pos_weight: float,
    neg_weight: float,
) -> torch.Tensor | None:
    token_weight = loss_mask.float() * position_weight
    weight_sum = token_weight.sum()
    if weight_sum.item() == 0:
        return None

    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    probs = torch.sigmoid(logits)
    pt = probs * labels + (1.0 - probs) * (1.0 - labels)
    focal_factor = (1.0 - pt).pow(gamma)
    class_weight = labels * pos_weight + (1.0 - labels) * neg_weight
    loss = class_weight * focal_factor * bce
    return (loss * token_weight).sum() / weight_sum


def masked_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    position_weight: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor | None:
    if args.loss_type == 'bce':
        return masked_bce_loss(logits, labels, loss_mask, position_weight)
    return masked_focal_loss(
        logits,
        labels,
        loss_mask,
        position_weight,
        gamma=args.focal_gamma,
        pos_weight=args.focal_pos_weight,
        neg_weight=args.focal_neg_weight,
    )


def build_position_loss_weight(
    state_ids: torch.Tensor,
    decay: float,
) -> torch.Tensor:
    if decay == 1.0:
        return torch.ones_like(state_ids, dtype=torch.float32)

    generation_region_mask = state_ids != STATE_PROMPT
    relative_position = generation_region_mask.long().cumsum(dim=1) - 1
    relative_position = relative_position.clamp(min=0)

    decay_tensor = torch.tensor(decay, dtype=torch.float32, device=state_ids.device)
    position_weight = torch.pow(decay_tensor, relative_position.to(torch.float32))
    position_weight = torch.where(
        generation_region_mask,
        position_weight,
        torch.ones_like(position_weight),
    )
    return position_weight


def build_state_loss_mask(state_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    loss_state_mask = (state_ids == STATE_GEN) | (state_ids == STATE_EOT)
    return loss_state_mask & attention_mask


def build_eot_loss_weight(state_ids: torch.Tensor) -> torch.Tensor:
    token_weight = torch.ones_like(state_ids, dtype=torch.float32)
    eot_mask = state_ids == STATE_EOT
    if not eot_mask.any():
        return token_weight

    eot_count = eot_mask.sum(dim=1, keepdim=True).clamp(min=1).to(torch.float32)
    per_eot_weight = 1.0 / eot_count
    token_weight = torch.where(eot_mask, per_eot_weight, token_weight)
    return token_weight


def build_target_keep_loss_weight(
    state_ids: torch.Tensor,
    target_mask_kept: torch.Tensor | None,
    keep_loss_weight: float,
) -> torch.Tensor:
    token_weight = torch.ones_like(state_ids, dtype=torch.float32)
    if keep_loss_weight == 1.0 or target_mask_kept is None:
        return token_weight

    kept_gen_mask = (state_ids == STATE_GEN) & target_mask_kept
    if not kept_gen_mask.any():
        return token_weight
    return torch.where(
        kept_gen_mask,
        torch.full_like(token_weight, keep_loss_weight),
        token_weight,
    )


@torch.no_grad()
def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    valid_logits = logits.detach().float().cpu()
    valid_labels = labels.detach().float().cpu()
    if valid_logits.numel() == 0:
        return {
            'loss': float('nan'),
            'auroc': float('nan'),
            'average_precision': float('nan'),
            'accuracy_at_0_5': float('nan'),
            'positive_rate': float('nan'),
            'gen_token_count': 0.0,
        }

    probs = torch.sigmoid(valid_logits)
    preds = (probs >= 0.5).float()
    accuracy = (preds == valid_labels).float().mean().item()
    positive_rate = valid_labels.mean().item()

    auroc = float('nan')
    ap = float('nan')
    if roc_auc_score is not None and valid_labels.unique().numel() > 1:
        auroc = float(roc_auc_score(valid_labels.numpy(), probs.numpy()))
    if average_precision_score is not None and valid_labels.unique().numel() > 1:
        ap = float(average_precision_score(valid_labels.numpy(), probs.numpy()))

    return {
        'auroc': auroc,
        'average_precision': ap,
        'accuracy_at_0_5': accuracy,
        'positive_rate': positive_rate,
        'gen_token_count': float(valid_logits.numel()),
    }


def move_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def validate_args(args: argparse.Namespace) -> None:
    if not args.use_ae_data:
        raise ValueError('pretrain.py expects AE-compressed step features; pass --use-ae-data or omit the flag.')
    if not 0.0 < args.random_crop_min_ratio <= args.random_crop_max_ratio <= 1.0:
        raise ValueError(
            'random crop ratios must satisfy 0 < random_crop_min_ratio <= random_crop_max_ratio <= 1.'
        )
    if args.input_noise_ratio < 0.0:
        raise ValueError('input_noise_ratio must be >= 0.')
    if args.target_mask_keep_loss_weight <= 0.0:
        raise ValueError('target_mask_keep_loss_weight must be > 0.')
    if not 0.0 < args.position_loss_decay <= 1.0:
        raise ValueError('position_loss_decay must satisfy 0 < position_loss_decay <= 1.')
    if args.model_type == 'mlp' and args.mlp_num_layers <= 0:
        raise ValueError(f'mlp_num_layers must be positive, got {args.mlp_num_layers}.')
    if args.model_type == 'mlp' and args.mlp_hidden_dim is not None and args.mlp_hidden_dim <= 0:
        raise ValueError(f'mlp_hidden_dim must be positive, got {args.mlp_hidden_dim}.')
    if args.use_dynamic_threshold and args.model_type != 'transformer':
        raise ValueError('dynamic_threshold requires --model-type transformer.')


def _empty_crop_stats(device: str) -> dict[str, float]:
    return {
        'crop_ratio_sum': 0.0,
        'visible_gen_tokens_sum': 0.0,
        'sample_count': 0.0,
        'crop_ratio_min': float('inf'),
        'crop_ratio_max': float('-inf'),
        'device': device,
    }


def apply_random_prefix_crop(
    batch: dict[str, Any],
    args: argparse.Namespace,
    *,
    apply_crop: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    attention_mask = batch['sequence_mask']
    loss_mask = build_state_loss_mask(batch['state'], attention_mask)
    stats = _empty_crop_stats(str(attention_mask.device))

    if not apply_crop:
        return attention_mask, loss_mask, stats

    batch_size, seq_len = attention_mask.shape
    cropped_attention_mask = attention_mask.clone()
    prompt_lengths = batch['prompt_length'].to(torch.int64)
    gen_lengths = batch['gen_len'].to(torch.int64)
    valid_gen = gen_lengths > 0

    if valid_gen.any():
        ratios = torch.empty(batch_size, device=attention_mask.device, dtype=torch.float32)
        ratios.uniform_(args.random_crop_min_ratio, args.random_crop_max_ratio)
        visible_gen = torch.round(ratios * gen_lengths.to(torch.float32)).to(torch.int64)
        visible_gen = torch.where(valid_gen, visible_gen.clamp(min=1), visible_gen)
        visible_gen = torch.minimum(visible_gen, gen_lengths)
        position_ids = torch.arange(seq_len, device=attention_mask.device).unsqueeze(0)
        cropped_attention_mask.zero_()
        cropped_attention_mask[~valid_gen] = attention_mask[~valid_gen]

        target_positive_mask = (batch['state'] == STATE_GEN) & (batch['label'] >= 0.5) & attention_mask
        gen_position_ids = position_ids - prompt_lengths.unsqueeze(1)
        effective_visible_gen = visible_gen.clone()

        for sample_idx in range(batch_size):
            if not bool(valid_gen[sample_idx].item()):
                continue

            prompt_len = int(prompt_lengths[sample_idx].item())
            prefix_end = int(visible_gen[sample_idx].item())
            full_gen_len = int(gen_lengths[sample_idx].item())

            sample_target_mask = target_positive_mask[sample_idx]
            in_prefix_target = sample_target_mask & (gen_position_ids[sample_idx] >= 0) & (gen_position_ids[sample_idx] < prefix_end)
            if not bool(in_prefix_target.any().item()):
                # If the sampled prefix contains no positive STATE_GEN token, fall back
                # to the full sequence instead of biasing the boundary onto a future positive.
                prefix_end = full_gen_len

            effective_visible_gen[sample_idx] = prefix_end

            sample_prompt_mask = position_ids[0] < prompt_len
            sample_prefix_mask = (
                (gen_position_ids[sample_idx] >= 0)
                & (gen_position_ids[sample_idx] < prefix_end)
                & attention_mask[sample_idx]
            )
            cropped_attention_mask[sample_idx] = sample_prompt_mask | sample_prefix_mask

        loss_mask = build_state_loss_mask(batch['state'], cropped_attention_mask)

        valid_ratios = effective_visible_gen[valid_gen].to(torch.float32) / gen_lengths[valid_gen].to(torch.float32)
        stats = {
            'crop_ratio_sum': float(valid_ratios.sum().item()),
            'visible_gen_tokens_sum': float(effective_visible_gen[valid_gen].sum().item()),
            'sample_count': float(valid_gen.sum().item()),
            'crop_ratio_min': float(valid_ratios.min().item()),
            'crop_ratio_max': float(valid_ratios.max().item()),
            'device': str(attention_mask.device),
        }

    return cropped_attention_mask, loss_mask, stats


def merge_crop_stats(accumulator: dict[str, float], update: dict[str, float]) -> dict[str, float]:
    accumulator['crop_ratio_sum'] += update['crop_ratio_sum']
    accumulator['visible_gen_tokens_sum'] += update['visible_gen_tokens_sum']
    accumulator['sample_count'] += update['sample_count']
    accumulator['crop_ratio_min'] = min(accumulator['crop_ratio_min'], update['crop_ratio_min'])
    accumulator['crop_ratio_max'] = max(accumulator['crop_ratio_max'], update['crop_ratio_max'])
    return accumulator


def finalize_crop_stats(stats: dict[str, float]) -> dict[str, float]:
    count = stats['sample_count']
    if count <= 0:
        return {
            'crop_ratio_mean': float('nan'),
            'crop_ratio_min': float('nan'),
            'crop_ratio_max': float('nan'),
            'visible_gen_tokens_mean': float('nan'),
        }
    return {
        'crop_ratio_mean': stats['crop_ratio_sum'] / count,
        'crop_ratio_min': stats['crop_ratio_min'],
        'crop_ratio_max': stats['crop_ratio_max'],
        'visible_gen_tokens_mean': stats['visible_gen_tokens_sum'] / count,
    }


def reduce_crop_stats(stats: dict[str, float]) -> dict[str, float]:
    if not is_distributed():
        return finalize_crop_stats(stats)

    device = stats['device']
    sum_tensor = torch.tensor(
        [stats['crop_ratio_sum'], stats['visible_gen_tokens_sum'], stats['sample_count']],
        dtype=torch.float64,
        device=device,
    )
    min_tensor = torch.tensor([stats['crop_ratio_min']], dtype=torch.float64, device=device)
    max_tensor = torch.tensor([stats['crop_ratio_max']], dtype=torch.float64, device=device)
    dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(min_tensor, op=dist.ReduceOp.MIN)
    dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)

    reduced = {
        'crop_ratio_sum': float(sum_tensor[0].item()),
        'visible_gen_tokens_sum': float(sum_tensor[1].item()),
        'sample_count': float(sum_tensor[2].item()),
        'crop_ratio_min': float(min_tensor.item()),
        'crop_ratio_max': float(max_tensor.item()),
        'device': device,
    }
    return finalize_crop_stats(reduced)


def apply_relative_input_noise(hidden_layers: torch.Tensor, noise_ratio: float) -> tuple[torch.Tensor, float]:
    if noise_ratio <= 0.0:
        return hidden_layers, 0.0

    base_std = float(hidden_layers.detach().float().std(unbiased=False).item())
    if base_std <= 0.0:
        return hidden_layers, 0.0

    noise_std = base_std * noise_ratio
    noise = torch.randn_like(hidden_layers) * noise_std
    return hidden_layers + noise, noise_std


def apply_batch_feature_noise(
    hidden_layers: torch.Tensor,
    state_ids: torch.Tensor,
    *,
    snr_db: float,
    apply_probability: float,
    source_states: tuple[int, ...] = (1, 2),
) -> tuple[torch.Tensor, float]:
    if snr_db <= 0.0 or apply_probability <= 0.0:
        return hidden_layers, 0.0
    if hidden_layers.ndim != 3:
        return hidden_layers, 0.0

    candidate_mask = torch.zeros_like(state_ids, dtype=torch.bool)
    for state_value in source_states:
        candidate_mask |= state_ids == state_value

    candidate_indices = candidate_mask.nonzero(as_tuple=False)
    candidate_count = int(candidate_indices.shape[0])
    if candidate_count == 0:
        return hidden_layers, 0.0

    apply_mask = candidate_mask & (torch.rand_like(hidden_layers[..., 0]) < apply_probability)
    apply_indices = apply_mask.nonzero(as_tuple=False)
    apply_count = int(apply_indices.shape[0])
    if apply_count == 0:
        return hidden_layers, 0.0

    sampled_pool_indices = torch.randint(
        low=0,
        high=candidate_count,
        size=(apply_count,),
        device=hidden_layers.device,
    )
    sampled_feature_indices = candidate_indices.index_select(0, sampled_pool_indices)
    noise_features = hidden_layers[sampled_feature_indices[:, 0], sampled_feature_indices[:, 1]]
    centered_noise_features = noise_features - noise_features.mean(dim=-1, keepdim=True)

    noisy_hidden_layers = hidden_layers.clone()
    target_features = noisy_hidden_layers[apply_indices[:, 0], apply_indices[:, 1]]

    signal_power = target_features.pow(2).mean(dim=-1, keepdim=True)
    noise_power = centered_noise_features.pow(2).mean(dim=-1, keepdim=True)
    snr_linear = 10.0 ** (snr_db / 10.0)
    target_noise_power = signal_power / snr_linear
    safe_noise_power = noise_power.clamp_min(torch.finfo(centered_noise_features.dtype).eps)
    scale = torch.sqrt(target_noise_power / safe_noise_power)
    scaled_noise = centered_noise_features * scale

    noisy_hidden_layers[apply_indices[:, 0], apply_indices[:, 1]] = target_features + scaled_noise

    applied_ratio = apply_count / candidate_count
    return noisy_hidden_layers, float(applied_ratio)


def apply_feature_ablation(hidden_layers: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == 'none':
        return hidden_layers
    if mode == 'zero':
        return torch.zeros_like(hidden_layers)
    raise ValueError(f'Unsupported feature ablation mode: {mode}')


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed(args: argparse.Namespace) -> tuple[str, int]:
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank_env = int(os.environ.get('LOCAL_RANK', '-1'))
    local_rank = local_rank_env if local_rank_env >= 0 else args.local_rank

    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError('Distributed training requires CUDA.')
        if local_rank < 0:
            raise RuntimeError('LOCAL_RANK is required for distributed training.')
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=args.backend)
        device = f'cuda:{local_rank}'
        return device, local_rank

    if args.device is not None:
        return args.device, 0
    if torch.cuda.is_available():
        return 'cuda', 0
    return 'cpu', 0


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def resolve_local_batch_size(global_batch_size: int, world_size: int, label: str) -> int:
    if global_batch_size < world_size:
        raise ValueError(f'{label} ({global_batch_size}) must be >= world size ({world_size}).')
    if global_batch_size % world_size != 0:
        raise ValueError(f'{label} ({global_batch_size}) must be divisible by world size ({world_size}).')
    return global_batch_size // world_size


def unwrap_model(model: TraceLock | DDP) -> TraceLock:
    return model.module if isinstance(model, DDP) else model


def ddp_requires_unused_parameter_detection(config: TraceLockConfig) -> bool:
    return (not config.use_position_encoding) or (not config.use_state_encoding)


def freeze_projection_stack(model: TraceLock) -> None:
    for module in (model.hidden_norm, model.delta_norm, model.hidden_proj, model.delta_proj):
        for parameter in module.parameters():
            parameter.requires_grad = False


def reduce_mean(value: float, device: str) -> float:
    if not is_distributed():
        return value
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return float(tensor.item())


def build_dataloaders(
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader, dict[str, int], DistributedSampler | None, DistributedSampler | None]:
    train_paths = list_split_sample_paths(args.samples_dir, 'train')
    val_paths = list_split_sample_paths(args.samples_dir, 'val')
    train_dataset = TraceDirectoryStepDataset(args.samples_dir, train_paths)
    val_dataset = TraceDirectoryStepDataset(args.samples_dir, val_paths)

    world_size = get_world_size()
    train_batch_size = resolve_local_batch_size(args.train_batch_size, world_size, 'train_batch_size')
    eval_batch_size = resolve_local_batch_size(args.eval_batch_size, world_size, 'eval_batch_size')

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
    stats = {
        'num_train_samples': len(train_paths),
        'num_val_samples': len(val_paths),
        'num_train_steps': len(train_dataset),
        'num_val_steps': len(val_dataset),
        'world_size': world_size,
        'train_batch_size_global': args.train_batch_size,
        'train_batch_size_per_rank': train_batch_size,
        'eval_batch_size_global': args.eval_batch_size,
        'eval_batch_size_per_rank': eval_batch_size,
    }
    return train_loader, val_loader, stats, train_sampler, val_sampler


@torch.no_grad()
def evaluate(model: TraceLock | DDP, dataloader: DataLoader, device: str, args: argparse.Namespace) -> dict[str, float]:
    model.eval()
    local_loss_sum = 0.0
    local_loss_count = 0
    local_logits: list[torch.Tensor] = []
    local_labels: list[torch.Tensor] = []
    crop_stats = _empty_crop_stats(device)
    apply_crop = args.enable_random_crop and args.random_crop_apply_to_val
    decay = args.position_loss_decay if args.position_loss_decay_apply_to_val else 1.0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        hidden_layers = batch['hidden_layers']
        if args.feature_ablation_apply_to_val:
            hidden_layers = apply_feature_ablation(hidden_layers, args.feature_ablation)
        attention_mask, loss_mask, batch_crop_stats = apply_random_prefix_crop(batch, args, apply_crop=apply_crop)
        crop_stats = merge_crop_stats(crop_stats, batch_crop_stats)
        position_weight = build_position_loss_weight(batch['state'], decay) * build_eot_loss_weight(batch['state'])
        outputs = model(
            hidden_layers=hidden_layers,
            state_ids=batch['state'],
            attention_mask=attention_mask,
        )
        loss = masked_classification_loss(outputs['logits'], batch['label'], loss_mask, position_weight, args)
        if loss is None:
            continue
        local_loss_sum += float(loss.item())
        local_loss_count += 1
        local_logits.append(outputs['logits'][loss_mask].detach().cpu())
        local_labels.append(batch['label'][loss_mask].detach().cpu())

    loss_stats = torch.tensor([local_loss_sum, float(local_loss_count)], dtype=torch.float64, device=device)
    if is_distributed():
        dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)

    global_loss = float('nan')
    if loss_stats[1].item() > 0:
        global_loss = float((loss_stats[0] / loss_stats[1]).item())
    crop_metrics = reduce_crop_stats(crop_stats)

    local_payload = {
        'logits': torch.cat(local_logits, dim=0) if local_logits else torch.empty(0, dtype=torch.float32),
        'labels': torch.cat(local_labels, dim=0) if local_labels else torch.empty(0, dtype=torch.float32),
    }

    if is_distributed():
        gathered_payloads: list[dict[str, torch.Tensor] | None] = [None for _ in range(get_world_size())]
        dist.all_gather_object(gathered_payloads, local_payload)
    else:
        gathered_payloads = [local_payload]

    if is_main_process():
        logits_parts = [item['logits'] for item in gathered_payloads if item is not None and item['logits'].numel() > 0]
        labels_parts = [item['labels'] for item in gathered_payloads if item is not None and item['labels'].numel() > 0]
        if logits_parts:
            metrics = compute_metrics(torch.cat(logits_parts, dim=0), torch.cat(labels_parts, dim=0))
        else:
            metrics = compute_metrics(torch.empty(0), torch.empty(0))
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
            'crop_ratio_mean': float('nan'),
            'crop_ratio_min': float('nan'),
            'crop_ratio_max': float('nan'),
            'visible_gen_tokens_mean': float('nan'),
        }

    if is_distributed():
        metrics_box = [metrics]
        dist.broadcast_object_list(metrics_box, src=0)
        metrics = metrics_box[0]

    return metrics


def save_checkpoint(
    checkpoint_path: Path,
    model: TraceLock | DDP,
    optimizer: AdamW,
    scheduler: ReduceLROnPlateau,
    train_state: dict[str, Any],
) -> None:
    checkpoint = {
        'model': unwrap_model(model).state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'train_state': train_state,
    }
    torch.save(checkpoint, checkpoint_path)


def main() -> None:
    args = parse_args()
    validate_args(args)
    device, local_rank = setup_distributed(args)
    seed = args.seed + get_rank()
    set_global_seed(seed)

    run_dir = ensure_dir(args.output_root / args.run_name)
    if is_main_process():
        write_json(run_dir / 'config.json', serialize_args(args))

    train_loader, val_loader, data_stats, train_sampler, _ = build_dataloaders(args)
    if is_main_process():
        write_json(run_dir / 'data_stats.json', data_stats)

    model = build_tracelock(
        make_config(args),
        device=device,
        projection_checkpoint=args.pretrained_proj_checkpoint,
        freeze_projection=args.freeze_pretrained_proj,
        print_summary=True,
        summary_title='TraceLock',
    )
    if args.use_ae_data:
        freeze_projection_stack(model)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.lr_reduce_patience)

    train_state = {
        'global_step': 0,
        'best_val_loss': float('inf'),
        'best_val_average_precision': float('-inf'),
        'epochs_completed': 0,
        'no_improve_evals': 0,
        'world_size': get_world_size(),
    }

    if args.resume is not None and args.resume.exists():
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        train_state.update(checkpoint['train_state'])

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

    train_log_every = 200
    train_log_loss_sum = 0.0
    train_log_count = 0
    train_log_noise_std_sum = 0.0
    train_crop_stats = _empty_crop_stats(device)

    try:
        while train_state['global_step'] < args.max_steps:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            accumulated_noise_std = 0.0
            valid_micro_steps = 0
            step_crop_stats = _empty_crop_stats(device)

            for _ in range(args.grad_accum_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_state['epochs_completed'] += 1
                    if train_sampler is not None:
                        train_sampler.set_epoch(train_state['epochs_completed'])
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                batch = move_batch_to_device(batch, device)
                hidden_layers = apply_feature_ablation(batch['hidden_layers'], args.feature_ablation)
                hidden_layers, noise_std = apply_relative_input_noise(hidden_layers, args.input_noise_ratio)
                attention_mask, loss_mask, batch_crop_stats = apply_random_prefix_crop(
                    batch,
                    args,
                    apply_crop=args.enable_random_crop,
                )
                step_crop_stats = merge_crop_stats(step_crop_stats, batch_crop_stats)
                position_weight = (
                    build_position_loss_weight(batch['state'], args.position_loss_decay)
                    * build_eot_loss_weight(batch['state'])
                )
                outputs = model(
                    hidden_layers=hidden_layers,
                    state_ids=batch['state'],
                    attention_mask=attention_mask,
                )
                position_weight = position_weight * build_target_keep_loss_weight(
                    batch['state'],
                    outputs.get('target_mask_kept'),
                    args.target_mask_keep_loss_weight,
                )
                loss = masked_classification_loss(
                    outputs['logits'],
                    batch['label'],
                    loss_mask,
                    position_weight,
                    args,
                )
                if loss is None:
                    continue
                (loss / args.grad_accum_steps).backward()
                accumulated_loss += float(loss.item())
                accumulated_noise_std += noise_std
                valid_micro_steps += 1

            if valid_micro_steps == 0:
                continue

            nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            train_state['global_step'] += 1

            local_train_loss = accumulated_loss / valid_micro_steps
            mean_train_loss = reduce_mean(local_train_loss, device)
            train_log_loss_sum += mean_train_loss
            train_log_count += 1
            train_log_noise_std_sum += reduce_mean(accumulated_noise_std / valid_micro_steps, device)
            train_crop_stats = merge_crop_stats(train_crop_stats, step_crop_stats)
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
                            'learning_rate': optimizer.param_groups[0]['lr'],
                            'log_window_steps': train_log_count,
                            **train_crop_metrics,
                        }),
                    )
                train_log_loss_sum = 0.0
                train_log_count = 0
                train_log_noise_std_sum = 0.0
                train_crop_stats = _empty_crop_stats(device)

            if train_state['global_step'] % args.eval_every == 0:
                val_metrics = evaluate(model, val_loader, device, args)
                scheduler.step(val_metrics['loss'])
                if is_main_process():
                    append_jsonl(
                        run_dir / 'metrics.jsonl',
                        round_metric_values({'split': 'val', 'step': train_state['global_step'], **val_metrics}),
                    )

                improved = False
                if val_metrics['loss'] < train_state['best_val_loss']:
                    train_state['best_val_loss'] = val_metrics['loss']
                    if is_main_process():
                        save_checkpoint(run_dir / 'best_val_loss.pt', model, optimizer, scheduler, train_state)
                    improved = True
                if val_metrics['average_precision'] > train_state['best_val_average_precision']:
                    train_state['best_val_average_precision'] = val_metrics['average_precision']
                    if is_main_process():
                        save_checkpoint(run_dir / 'best_val_average_precision.pt', model, optimizer, scheduler, train_state)
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

            if train_state['global_step'] % args.save_every == 0 and is_main_process():
                save_checkpoint(run_dir / f"step_{train_state['global_step']:07d}.pt", model, optimizer, scheduler, train_state)
                write_json(run_dir / 'train_state.json', train_state)

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
                        'learning_rate': optimizer.param_groups[0]['lr'],
                        'log_window_steps': train_log_count,
                        **train_crop_metrics,
                    }),
                )
        if is_main_process():
            write_json(run_dir / 'train_state.json', train_state)
            print('training finished', train_state)
    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()
