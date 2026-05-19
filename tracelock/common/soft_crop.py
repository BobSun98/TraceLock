from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


STATE_PROMPT = 0
STATE_LOCKED = 1
STATE_GEN = 2

SOFT_CROP_WINDOW_WIDTH_KEYS = ("window_width", "block_length", "gen_len")


def resolve_soft_crop_window_width(
    payload: Mapping[str, Any],
    *,
    fallback_gen_len: int | None = None,
) -> int:
    for key in SOFT_CROP_WINDOW_WIDTH_KEYS:
        value = payload.get(key)
        if value is None:
            continue
        window_width = int(value)
        if window_width > 0:
            return window_width

    if fallback_gen_len is not None and fallback_gen_len > 0:
        return int(fallback_gen_len)
    raise KeyError("Could not resolve soft-crop window width from payload.")


def find_soft_crop_window_from_state(
    state_1d: torch.Tensor,
    *,
    prompt_length: int,
    window_width: int,
    seq_len: int | None = None,
) -> tuple[int, int]:
    if seq_len is None:
        seq_len = int(state_1d.shape[0])
    if window_width <= 0:
        raise ValueError("window_width must be positive.")

    gen_positions = torch.nonzero(state_1d[:seq_len] == STATE_GEN, as_tuple=False).squeeze(-1)
    if gen_positions.numel() == 0:
        return int(prompt_length), int(prompt_length)

    window_start = int(gen_positions[0].item())
    window_end = min(int(seq_len), window_start + int(window_width))
    return window_start, window_end


def find_soft_crop_window_from_mask(
    mask_index_1d: torch.Tensor,
    *,
    prompt_length: int,
    window_width: int,
    seq_len: int | None = None,
) -> tuple[int, int]:
    if seq_len is None:
        seq_len = int(mask_index_1d.shape[0])
    if window_width <= 0:
        raise ValueError("window_width must be positive.")

    masked_positions = torch.nonzero(mask_index_1d[prompt_length:seq_len], as_tuple=False).squeeze(-1)
    if masked_positions.numel() == 0:
        return int(prompt_length), int(prompt_length)

    window_start = int(prompt_length + masked_positions[0].item())
    window_end = min(int(seq_len), window_start + int(window_width))
    return window_start, window_end


def build_soft_crop_attention_mask(batch: dict[str, Any]) -> torch.Tensor:
    attention_mask = batch["sequence_mask"].clone()
    batch_size, seq_len = attention_mask.shape
    position_ids = torch.arange(seq_len, device=attention_mask.device).unsqueeze(0)

    for batch_idx in range(batch_size):
        valid_len = int(batch["seq_len"][batch_idx].item())
        prompt_len = int(batch["prompt_length"][batch_idx].item())
        window_width = int(batch["window_width"][batch_idx].item())
        state = batch["state"][batch_idx, :valid_len]

        _, window_end = find_soft_crop_window_from_state(
            state,
            prompt_length=prompt_len,
            window_width=window_width,
            seq_len=valid_len,
        )
        if window_end <= prompt_len:
            continue

        keep_mask = position_ids[0] < window_end
        attention_mask[batch_idx] &= keep_mask

    return attention_mask
