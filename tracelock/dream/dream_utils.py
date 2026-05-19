from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModel, AutoTokenizer


DEFAULT_MODEL_ID = "Dream-org/Dream-v0-Instruct-7B"
STATE_PROMPT = 0
STATE_LOCKED = 1
STATE_GEN = 2
STATE_EOT = 3


def resolve_torch_dtype(name: str) -> torch.dtype:
    if not hasattr(torch, name):
        raise ValueError(f"Unsupported torch dtype: {name}")
    dtype = getattr(torch, name)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"{name} is not a torch dtype.")
    return dtype


def load_dream_tokenizer(model_id: str, *, local_files_only: bool = False):
    try:
        return AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
    except Exception:
        repo_path = Path(snapshot_download(model_id, local_files_only=local_files_only))
        tokenizer_path = repo_path / "tokenization_dream.py"
        spec = importlib.util.spec_from_file_location("dream_tokenization_dream", tokenizer_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load Dream tokenizer module from {tokenizer_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.DreamTokenizer.from_pretrained(str(repo_path))


def load_dream_model(
    model_id: str,
    *,
    device: str,
    torch_dtype: torch.dtype,
    local_files_only: bool = False,
):
    return AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
    ).to(device).eval()


def load_dream_config(
    model_id: str,
    *,
    local_files_only: bool = False,
):
    return AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )


def build_generation_attention_mask(
    attention_mask_2d: torch.Tensor | None,
    total_length: int,
) -> tuple[str | torch.Tensor, torch.Tensor | None]:
    if attention_mask_2d is None:
        return "full", None
    if not torch.any(attention_mask_2d == 0):
        return "full", None

    padded_mask = torch.nn.functional.pad(
        attention_mask_2d,
        (0, total_length - attention_mask_2d.shape[1]),
        value=1.0,
    )
    tok_idx = padded_mask.long().cumsum(-1) - 1
    tok_idx.masked_fill_(padded_mask == 0, 1)
    full_mask = torch.logical_and(
        padded_mask.unsqueeze(1).unsqueeze(-2),
        padded_mask.unsqueeze(1).unsqueeze(-1),
    )
    return full_mask, tok_idx


def shift_hidden_positions(hidden_states: torch.Tensor) -> torch.Tensor:
    if hidden_states.ndim < 2:
        raise ValueError(f"hidden_states must have at least 2 dims, got {tuple(hidden_states.shape)}")
    seq_len = int(hidden_states.shape[1])
    if seq_len <= 1:
        return hidden_states.contiguous()
    # Dream 的 logits 在生成时会做 [a,b,c,d] -> [a,a,b,c] 的位置对齐。
    # 这里对 hidden 采用同样的规则，避免 TraceLock 看到的特征和当前位置 proposal 语义不一致。
    return torch.cat([hidden_states[:, :1], hidden_states[:, :-1]], dim=1).contiguous()


def extract_aligned_last_three_hidden_for_projection(hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
    if len(hidden_states) < 3:
        raise ValueError(f"Expected at least 3 hidden state tensors, got {len(hidden_states)}")
    # 输出给 TraceLock 投影层的格式是 [B, L, 3, D]。
    last_three = torch.stack(hidden_states[-3:], dim=1).permute(0, 2, 1, 3).contiguous()
    # 这里真正执行 hidden 的位置对齐：
    # 原始序列位置 [a,b,c,d] 会变成 [a,a,b,c]，
    # 与 Dream 生成时 logits 的 shift 规则保持一致。
    aligned = shift_hidden_positions(last_three)
    return aligned.contiguous()


def extract_aligned_last_three_hidden_for_trace(hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
    if len(hidden_states) < 3:
        raise ValueError(f"Expected at least 3 hidden state tensors, got {len(hidden_states)}")
    # trace 样本沿用旧格式 [3, B, L, D]，这里只在对齐后再换回 trace 需要的布局。
    aligned = extract_aligned_last_three_hidden_for_projection(hidden_states)
    return aligned.permute(2, 0, 1, 3).contiguous()


def crop_sample_sequence(
    full_sequence: torch.Tensor,
    *,
    prompt_token_length: int,
    padded_prompt_length: int,
) -> torch.Tensor:
    prompt_part = full_sequence[:prompt_token_length]
    generation_part = full_sequence[padded_prompt_length:]
    return torch.cat([prompt_part, generation_part], dim=0).contiguous()


def compute_state_ids(
    token_ids: torch.Tensor,
    *,
    prompt_length: int,
    mask_token_id: int,
    eos_token_id: int,
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    state_ids = torch.full(token_ids.shape, STATE_GEN, dtype=dtype, device=token_ids.device)
    state_ids[:prompt_length] = STATE_PROMPT

    generation_tokens = token_ids[prompt_length:]
    generation_states = torch.where(
        generation_tokens == mask_token_id,
        torch.full_like(generation_tokens, STATE_GEN, dtype=dtype),
        torch.full_like(generation_tokens, STATE_LOCKED, dtype=dtype),
    )
    state_ids[prompt_length:] = generation_states
    return state_ids


def summarize_special_tokens(tokenizer, model_or_config) -> dict[str, Any]:
    return {
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "mask_token": getattr(tokenizer, "mask_token", None),
        "mask_token_id": getattr(tokenizer, "mask_token_id", None),
        "config_bos_token_id": getattr(model_or_config, "bos_token_id", None),
        "config_eos_token_id": getattr(model_or_config, "eos_token_id", None),
        "config_pad_token_id": getattr(model_or_config, "pad_token_id", None),
        "config_mask_token_id": getattr(model_or_config, "mask_token_id", None),
        "padding_side": tokenizer.padding_side,
        "hidden_size": getattr(model_or_config, "hidden_size", None),
        "num_hidden_layers": getattr(model_or_config, "num_hidden_layers", None),
    }
