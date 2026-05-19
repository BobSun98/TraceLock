from __future__ import annotations

import os
from typing import Literal

import torch


TRANSFER_SCORE_SOURCE_CONFIDENCE = "confidence"
TRANSFER_SCORE_SOURCE_TRACELOCK_SCORE = "tracelock_score"
TRANSFER_SCORE_SOURCES = (
    TRANSFER_SCORE_SOURCE_CONFIDENCE,
    TRANSFER_SCORE_SOURCE_TRACELOCK_SCORE,
)
DEFAULT_TRANSFER_SCORE_SOURCE = TRANSFER_SCORE_SOURCE_TRACELOCK_SCORE
TransferScoreSource = Literal["confidence", "tracelock_score"]


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    return float(value)


# Global safety gate for inference experiments.
# When enabled, TraceLock-driven PAD/EOS accepts are rejected until confidence has
# accepted the first PAD/EOS boundary; after that TraceLock may clear the tail.
SUPPRESS_TRACELOCK_SPECIAL_ACCEPTS = _env_flag("TRACELOCK_SUPPRESS_SPECIAL_ACCEPTS", True)
TRACELOCK_SPECIAL_SUPPRESS_POSITION_RATIO = _env_float("TRACELOCK_SPECIAL_SUPPRESS_POSITION_RATIO", 1.0)
ALLOW_TRACELOCK_SPECIAL_AFTER_FIRST = _env_flag("TRACELOCK_ALLOW_SPECIAL_AFTER_FIRST", True)

# Short aliases used by older local launch scripts in this release tree.
SUPPRESS_EARLY_TRACELOCK_SPECIAL_ACCEPTS = SUPPRESS_TRACELOCK_SPECIAL_ACCEPTS
EARLY_TRACELOCK_SPECIAL_POSITION_RATIO = TRACELOCK_SPECIAL_SUPPRESS_POSITION_RATIO


def normalize_transfer_score_source(value: str) -> TransferScoreSource:
    normalized = value.strip().lower()
    if normalized not in TRANSFER_SCORE_SOURCES:
        raise ValueError(f"Unsupported transfer_score_source: {value}")
    return normalized  # type: ignore[return-value]


def _score_tensor_for_source(
    *,
    transfer_score_source: TransferScoreSource,
    confidence_scores: torch.Tensor,
    tracelock_probs: torch.Tensor,
) -> torch.Tensor:
    if transfer_score_source == TRANSFER_SCORE_SOURCE_CONFIDENCE:
        return confidence_scores
    if transfer_score_source == TRANSFER_SCORE_SOURCE_TRACELOCK_SCORE:
        return tracelock_probs
    raise ValueError(f"Unsupported transfer_score_source: {transfer_score_source}")


def select_transfer_masks(
    *,
    current_window_mask: torch.Tensor,
    scheduled_transfer_count: int,
    transfer_score_source: TransferScoreSource,
    confidence_scores: torch.Tensor,
    tracelock_probs: torch.Tensor,
    threshold: float,
    max_threshold_accept_count: int = 4,
    current_token_ids: torch.Tensor | None = None,
    step_token_ids: torch.Tensor | None = None,
    special_token_ids: tuple[int, ...] | None = None,
    generation_start_index: int | None = None,
    generation_length: int | None = None,
) -> dict[str, torch.Tensor | str | bool | int]:
    score_tensor = _score_tensor_for_source(
        transfer_score_source=transfer_score_source,
        confidence_scores=confidence_scores,
        tracelock_probs=tracelock_probs,
    )
    masked_scores = torch.where(current_window_mask, score_tensor, torch.full_like(score_tensor, float("-inf")))

    scheduled_transfer_mask = torch.zeros_like(current_window_mask, dtype=torch.bool)
    if scheduled_transfer_count > 0:
        _, select_index = torch.topk(masked_scores[0], k=scheduled_transfer_count)
        scheduled_transfer_mask[0, select_index] = True

    threshold_hit_mask = current_window_mask & (tracelock_probs >= threshold)
    threshold_accept_mask = threshold_hit_mask.clone()
    cap_applied = False
    if max_threshold_accept_count is not None:
        if max_threshold_accept_count <= 0:
            raise ValueError(f"max_threshold_accept_count must be positive, got {max_threshold_accept_count}")
        for batch_idx in range(current_window_mask.shape[0]):
            hit_indices = torch.nonzero(threshold_hit_mask[batch_idx], as_tuple=False).flatten()
            if int(hit_indices.numel()) <= max_threshold_accept_count:
                continue
            cap_applied = True
            threshold_accept_mask[batch_idx] = False
            hit_scores = tracelock_probs[batch_idx, hit_indices]
            _, topk_order = torch.topk(hit_scores, k=max_threshold_accept_count)
            kept_indices = hit_indices[topk_order]
            threshold_accept_mask[batch_idx, kept_indices] = True

    suppressed_special_accept_count = 0
    suppressed_pad_eos_accept_mask = torch.zeros_like(current_window_mask, dtype=torch.bool)
    if (
        SUPPRESS_TRACELOCK_SPECIAL_ACCEPTS
        and step_token_ids is not None
        and special_token_ids
        and generation_start_index is not None
        and generation_length is not None
        and generation_length > 0
    ):
        generation_start = int(generation_start_index)
        suppress_end = generation_start + max(1, int(generation_length * TRACELOCK_SPECIAL_SUPPRESS_POSITION_RATIO))
        position_ids = torch.arange(current_window_mask.shape[1], device=current_window_mask.device).unsqueeze(0)
        suppress_generation_mask = (position_ids >= generation_start) & (position_ids < suppress_end)
        special_token_mask = torch.zeros_like(current_window_mask, dtype=torch.bool)
        for token_id in special_token_ids:
            special_token_mask |= step_token_ids == int(token_id)
        existing_special_mask = torch.zeros_like(current_window_mask, dtype=torch.bool)
        if current_token_ids is not None:
            generation_end = generation_start + int(generation_length)
            generation_position_mask = (position_ids >= generation_start) & (position_ids < generation_end)
            for token_id in special_token_ids:
                existing_special_mask |= current_token_ids == int(token_id)
            existing_special_mask &= generation_position_mask
        if not (ALLOW_TRACELOCK_SPECIAL_AFTER_FIRST and existing_special_mask.any()):
            tracelock_driven_mask = threshold_accept_mask.clone()
            if transfer_score_source == TRANSFER_SCORE_SOURCE_TRACELOCK_SCORE:
                tracelock_driven_mask |= scheduled_transfer_mask
            suppressed_pad_eos_accept_mask = tracelock_driven_mask & suppress_generation_mask & special_token_mask
            suppressed_special_accept_count = int(suppressed_pad_eos_accept_mask.sum().item())
            if suppressed_special_accept_count > 0:
                threshold_accept_mask = threshold_accept_mask & ~suppressed_pad_eos_accept_mask
                scheduled_transfer_mask = scheduled_transfer_mask & ~suppressed_pad_eos_accept_mask

    fallback_mask = torch.zeros_like(current_window_mask, dtype=torch.bool)
    fallback_reason = "none"
    if not threshold_accept_mask.any() and not scheduled_transfer_mask.any() and current_window_mask.any():
        fallback_scores = torch.where(
            current_window_mask,
            # confidence_scores,
            tracelock_probs,
            torch.full_like(confidence_scores, float("-inf")),
        )
        fallback_index = int(torch.argmax(fallback_scores[0]).item())
        fallback_mask[0, fallback_index] = True
        if suppressed_special_accept_count > 0:
            fallback_reason = "tracelock_special_suppressed_confidence_fallback"
        else:
            fallback_reason = "selector_top1_no_threshold_hit"

    return {
        "scheduled_transfer_mask": scheduled_transfer_mask,
        "threshold_hit_mask": threshold_hit_mask,
        "threshold_accept_mask": threshold_accept_mask,
        "suppressed_pad_eos_accept_mask": suppressed_pad_eos_accept_mask,
        "fallback_mask": fallback_mask,
        "fallback_reason": fallback_reason,
        "cap_applied": cap_applied,
        "suppressed_special_accept_count": suppressed_special_accept_count,
    }
