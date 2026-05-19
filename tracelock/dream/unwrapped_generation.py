from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from tracelock.dream.dream_utils import build_generation_attention_mask


PROPOSAL_ALG_ORIGIN = "origin"
PROPOSAL_ALG_MASKGIT_PLUS = "maskgit_plus"
PROPOSAL_ALG_TOPK_MARGIN = "topk_margin"
PROPOSAL_ALG_ENTROPY = "entropy"
SUPPORTED_PROPOSAL_ALGS = (
    PROPOSAL_ALG_ORIGIN,
    PROPOSAL_ALG_MASKGIT_PLUS,
    PROPOSAL_ALG_TOPK_MARGIN,
    PROPOSAL_ALG_ENTROPY,
)

TRANSFER_METHOD_SCHEDULED_TOPK = "scheduled_topk"
TRANSFER_METHOD_FAST_DLM = "fast_dlm"
SUPPORTED_TRANSFER_METHODS = (
    TRANSFER_METHOD_SCHEDULED_TOPK,
    TRANSFER_METHOD_FAST_DLM,
)

TRANSFER_SCORE_ENTROPY = "entropy"
TRANSFER_SCORE_CONFIDENCE = "confidence"
TRANSFER_SCORE_MARGIN = "margin"
TRANSFER_SCORE_RANDOM = "random"
SUPPORTED_TRANSFER_SCORES = (
    TRANSFER_SCORE_ENTROPY,
    TRANSFER_SCORE_CONFIDENCE,
    TRANSFER_SCORE_MARGIN,
    TRANSFER_SCORE_RANDOM,
)


def top_p_logits(logits: torch.Tensor, top_p: float | None = None) -> torch.Tensor:
    if top_p is None or top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    return logits.masked_fill(mask, torch.finfo(logits.dtype).min)


def top_k_logits(logits: torch.Tensor, top_k: int | None = None) -> torch.Tensor:
    if top_k is None:
        return logits
    top_k = min(int(top_k), logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    return logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)


def sample_tokens(
    logits: torch.Tensor,
    *,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
    margin_confidence: bool = False,
    neg_entropy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)

    probs = torch.softmax(logits, dim=-1)
    if temperature > 0:
        try:
            x0 = torch.distributions.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except Exception:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        confidence = sorted_probs[:, 0] - sorted_probs[:, 1]

    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    return confidence, x0


def compute_token_scores(
    logits: torch.Tensor,
    *,
    score_type: str,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    random_generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if score_type == TRANSFER_SCORE_CONFIDENCE:
        return sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
    if score_type == TRANSFER_SCORE_MARGIN:
        return sample_tokens(
            logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            margin_confidence=True,
        )
    if score_type == TRANSFER_SCORE_ENTROPY:
        return sample_tokens(
            logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            neg_entropy=True,
        )
    if score_type == TRANSFER_SCORE_RANDOM:
        _, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
        scores = torch.rand(
            x0.shape,
            generator=random_generator,
            device=x0.device,
            dtype=torch.float32,
        ).to(dtype=logits.dtype)
        return scores, x0
    raise ValueError(f"Unsupported score_type: {score_type}")


def resolve_transfer_score_type(proposal_alg: str, transfer_score_type: str | None) -> str:
    if transfer_score_type is not None:
        normalized = str(transfer_score_type).strip().lower()
        if normalized not in SUPPORTED_TRANSFER_SCORES:
            raise ValueError(f"Unsupported transfer_score_type: {transfer_score_type}")
        return normalized
    if proposal_alg == PROPOSAL_ALG_MASKGIT_PLUS:
        return TRANSFER_SCORE_CONFIDENCE
    if proposal_alg == PROPOSAL_ALG_TOPK_MARGIN:
        return TRANSFER_SCORE_MARGIN
    if proposal_alg == PROPOSAL_ALG_ENTROPY:
        return TRANSFER_SCORE_ENTROPY
    raise ValueError(f"proposal_alg={proposal_alg} requires an explicit transfer_score_type")


def resolve_proposal_alg(proposal_alg: str) -> str:
    normalized = str(proposal_alg).strip().lower()
    if normalized not in SUPPORTED_PROPOSAL_ALGS:
        raise ValueError(f"Unsupported proposal_alg: {proposal_alg}")
    return normalized


def resolve_transfer_method(transfer_method: str) -> str:
    normalized = str(transfer_method).strip().lower()
    if normalized not in SUPPORTED_TRANSFER_METHODS:
        raise ValueError(f"Unsupported transfer_method: {transfer_method}")
    return normalized


def build_full_length_masked_values(
    *,
    x: torch.Tensor,
    mask_index: torch.Tensor,
    masked_values: torch.Tensor,
    fill_value: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    full = torch.full_like(x, fill_value=fill_value, dtype=dtype)
    full[mask_index] = masked_values
    return full


def apply_selected_transfers(
    *,
    x: torch.Tensor,
    mask_index: torch.Tensor,
    transfer_index: torch.Tensor,
    x0_masked: torch.Tensor,
    mask_token_id: int,
) -> torch.Tensor:
    x_next = x.clone()
    if transfer_index.ndim != 2:
        raise ValueError(f"transfer_index must have shape [B, N], got {tuple(transfer_index.shape)}")
    x_proposal = torch.full_like(x_next, fill_value=int(mask_token_id))
    x_proposal[mask_index] = x0_masked
    row_indices = torch.arange(x_next.size(0), device=x_next.device).unsqueeze(1).expand_as(transfer_index)
    x_next[row_indices, transfer_index] = x_proposal[row_indices, transfer_index]
    return x_next


def apply_selected_transfers_mask(
    *,
    x: torch.Tensor,
    mask_index: torch.Tensor,
    transfer_mask: torch.Tensor,
    x0_masked: torch.Tensor,
    mask_token_id: int,
) -> torch.Tensor:
    x_next = x.clone()
    if transfer_mask.shape != x.shape:
        raise ValueError(f"transfer_mask must have shape {tuple(x.shape)}, got {tuple(transfer_mask.shape)}")
    x_proposal = torch.full_like(x_next, fill_value=int(mask_token_id))
    x_proposal[mask_index] = x0_masked
    x_next[transfer_mask] = x_proposal[transfer_mask]
    return x_next


def select_transfer_mask_topk(
    full_scores: torch.Tensor,
    *,
    number_transfer_tokens: int | torch.Tensor,
    alg_temp: float | None,
) -> torch.Tensor:
    batch_size, seq_len = full_scores.shape
    if isinstance(number_transfer_tokens, torch.Tensor):
        transfer_counts = number_transfer_tokens.to(device=full_scores.device, dtype=torch.long).view(batch_size)
    else:
        transfer_counts = torch.full(
            (batch_size,),
            fill_value=int(number_transfer_tokens),
            device=full_scores.device,
            dtype=torch.long,
        )
    transfer_counts = transfer_counts.clamp(min=0, max=seq_len)
    max_transfer_tokens = int(transfer_counts.max().item()) if transfer_counts.numel() > 0 else 0
    if max_transfer_tokens <= 0:
        return torch.zeros_like(full_scores, dtype=torch.bool)

    if alg_temp is None or alg_temp == 0:
        _, transfer_index = torch.topk(full_scores, max_transfer_tokens, dim=-1)
    else:
        sampled_scores = F.softmax(full_scores / alg_temp, dim=-1)
        transfer_index = torch.multinomial(sampled_scores, num_samples=max_transfer_tokens)

    rank_mask = (
        torch.arange(max_transfer_tokens, device=full_scores.device)
        .unsqueeze(0)
        .expand(batch_size, max_transfer_tokens)
        < transfer_counts.unsqueeze(1)
    )
    transfer_mask = torch.zeros_like(full_scores, dtype=torch.bool)
    transfer_mask.scatter_(1, transfer_index, rank_mask)
    return transfer_mask


def select_transfer_mask_threshold(
    full_scores: torch.Tensor,
    *,
    mask_index: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    transfer_mask = mask_index & (full_scores >= threshold)
    if transfer_mask.any():
        return transfer_mask

    masked_scores = torch.where(
        mask_index,
        full_scores,
        torch.full_like(full_scores, torch.finfo(full_scores.dtype).min),
    )
    fallback_index = torch.argmax(masked_scores, dim=1, keepdim=True)
    fallback_mask = torch.zeros_like(transfer_mask, dtype=torch.bool)
    fallback_mask.scatter_(1, fallback_index, True)
    return fallback_mask & mask_index


@dataclass
class DreamStepContext:
    step: int
    steps: int
    x: torch.Tensor
    logits: torch.Tensor
    mask_index: torch.Tensor
    x0_masked: torch.Tensor
    full_scores: torch.Tensor
    number_transfer_tokens: int
    prompt_length: int
    mask_token_id: int
    attention_mask: str | torch.Tensor
    position_ids: torch.Tensor | None
    hidden_states: tuple[torch.Tensor, ...] | None = None


TransferPostProcessor = Callable[[DreamStepContext, torch.Tensor], torch.Tensor]


@torch.no_grad()
def diffusion_generate_unwrapped(
    model,
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    max_new_tokens: int,
    steps: int,
    eps: float = 1e-3,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
    proposal_alg: str = PROPOSAL_ALG_ENTROPY,
    transfer_method: str = TRANSFER_METHOD_SCHEDULED_TOPK,
    transfer_score_type: str | None = None,
    alg_temp: float | None = 0.0,
    fast_dlm_threshold: float | None = None,
    random_seed: int | None = None,
    output_history: bool = False,
    output_hidden_states: bool = False,
    return_hidden_trace: bool = False,
    generation_logits_hook_func: Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    generation_tokens_hook_func: Callable[[int | None, torch.Tensor, torch.Tensor | None], torch.Tensor] | None = None,
    transfer_scores_hook_func: Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    transfer_count_hook_func: Callable[[int, torch.Tensor, torch.Tensor, int], int | torch.Tensor] | None = None,
    post_transfer_hook: TransferPostProcessor | None = None,
) -> dict[str, object]:
    # 这是 Dream diffusion 生成的“裸循环”版本：
    # 1. 先把 generation 区全部补成 mask_token；
    # 2. 每个 step 用当前 x 跑一次 Dream，得到当前位置的 proposal logits；
    # 3. 在 mask 位置上采样/打分，挑一部分 token 从 masked 变成 accepted；
    # 4. 可选地通过 hook 在 transfer 前后插入额外逻辑（例如 block schedule、特征抓取）。
    #
    # 这里返回的 sequences 是最后一步的完整序列；如果中途所有 mask 都被填完，会提前结束。
    proposal_alg = resolve_proposal_alg(proposal_alg)
    transfer_method = resolve_transfer_method(transfer_method)
    transfer_score_type = resolve_transfer_score_type(proposal_alg, transfer_score_type)
    if transfer_method == TRANSFER_METHOD_FAST_DLM and fast_dlm_threshold is None:
        raise ValueError("fast_dlm_threshold is required when transfer_method='fast_dlm'")

    hook_logits = generation_logits_hook_func or (lambda _step, _x, logits: logits)
    hook_tokens = generation_tokens_hook_func or (lambda _step, x, _logits: x)
    hook_scores = transfer_scores_hook_func or (lambda _step, _x, _mask_index, full_scores: full_scores)
    hook_transfer_count = transfer_count_hook_func or (lambda _step, _x, _mask_index, default_count: default_count)
    generation_mask_token_id = getattr(getattr(model, "generation_config", None), "mask_token_id", None)
    config_mask_token_id = getattr(getattr(model, "config", None), "mask_token_id", None)
    mask_token_id = generation_mask_token_id if generation_mask_token_id is not None else config_mask_token_id
    if mask_token_id is None:
        raise ValueError("Dream model does not expose mask_token_id on generation_config or config.")
    mask_token_id = int(mask_token_id)
    prompt_length = int(input_ids.shape[1])
    total_length = prompt_length + int(max_new_tokens)

    # x 的长度从一开始就固定成 prompt + generation。
    # prompt 部分来自输入，generation 部分先全部填成 mask_token，后面逐步被 proposal 替换。
    x = F.pad(input_ids, (0, total_length - prompt_length), value=mask_token_id)
    full_attention_mask, position_ids = build_generation_attention_mask(attention_mask, total_length=total_length)
    timesteps = torch.linspace(1, eps, steps + 1, device=x.device)
    histories: list[torch.Tensor] | None = [] if output_history else None
    hidden_trace: list[tuple[torch.Tensor, ...]] | None = [] if return_hidden_trace else None
    need_hidden_states = bool(output_hidden_states or return_hidden_trace)

    random_generator = None
    if random_seed is not None:
        random_generator = torch.Generator(device=x.device)
        random_generator.manual_seed(int(random_seed))

    x = hook_tokens(None, x, None)
    for step_idx in range(steps):
        # 这里的 mask_index 是“当前 step 仍然待生成的位置”。
        # 后面的 logits 打分、transfer 预算、hook 都围绕这批位置展开。
        mask_index = x == mask_token_id
        model_outputs = model(
            input_ids=x,
            attention_mask=full_attention_mask,
            position_ids=position_ids,
            output_hidden_states=need_hidden_states,
            return_dict=True,
        )
        # Dream 原始 logits 是标准 next-token 形式；这里手动右移一位，
        # 让 position i 的 logits 对齐到“当前位置应该填什么 token”的语义。
        # 后续 pretrain 里 step_token_ids / hidden 对齐都建立在这个 shifted 语义上。
        logits = torch.cat([model_outputs.logits[:, :1], model_outputs.logits[:, :-1]], dim=1)
        logits = hook_logits(step_idx, x, logits)
        mask_logits = logits[mask_index]

        t = timesteps[step_idx]
        s = timesteps[step_idx + 1]

        if proposal_alg == PROPOSAL_ALG_ORIGIN:
            # origin 分支不显式比较 score，只按 diffusion 日程随机决定
            # 这一步要从多少个 masked 位置里“放出” proposal。
            p_transfer = 1 - s / t if step_idx < steps - 1 else 1
            x0_masked = torch.full_like(mask_logits[:, 0], fill_value=mask_token_id, dtype=torch.long)
            transfer_mask_local = torch.rand(
                x0_masked.shape,
                generator=random_generator,
                device=x.device,
            ) < p_transfer
            if transfer_mask_local.any():
                _, sampled_ids = sample_tokens(
                    mask_logits[transfer_mask_local],
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                x0_masked[transfer_mask_local] = sampled_ids
            x_after = x.clone()
            x_after[mask_index] = x0_masked
            full_scores = torch.zeros_like(x, dtype=logits.dtype)
        else:
            # 非 origin 分支会先对所有 mask 位置给一个 proposal token，再给每个 proposal 一个 score。
            # 后面是“从这些 proposal 里接受多少、接受哪些”的问题。
            scores_masked, x0_masked = compute_token_scores(
                mask_logits,
                score_type=transfer_score_type,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                random_generator=random_generator,
            )
            # 默认 transfer 预算来自 diffusion schedule：当前剩余 mask 中，本 step 应该转正多少个。
            # 如果外面传了 hook，可以把它改成 block-wise 或别的动态预算。
            num_mask_token = mask_index.sum() / mask_index.shape[0]
            default_transfer_count = int(num_mask_token * (1 - s / t)) if step_idx < steps - 1 else int(num_mask_token)
            number_transfer_tokens = hook_transfer_count(step_idx, x, mask_index, default_transfer_count)
            full_scores = build_full_length_masked_values(
                x=x,
                mask_index=mask_index,
                masked_values=scores_masked,
                fill_value=-torch.inf,
                dtype=logits.dtype,
            )
            full_scores = hook_scores(step_idx, x, mask_index, full_scores)

            if transfer_method == TRANSFER_METHOD_SCHEDULED_TOPK:
                # scheduled_topk: 只接受 score 最高的若干个 proposal。
                transfer_mask = select_transfer_mask_topk(
                    full_scores,
                    number_transfer_tokens=number_transfer_tokens,
                    alg_temp=alg_temp,
                )
                x_after = apply_selected_transfers_mask(
                    x=x,
                    mask_index=mask_index,
                    transfer_mask=transfer_mask,
                    x0_masked=x0_masked,
                    mask_token_id=mask_token_id,
                )
            else:
                # fast_dlm: 不按 quota，而是所有 score 超过阈值的位置都直接接受。
                transfer_mask = select_transfer_mask_threshold(
                    full_scores,
                    mask_index=mask_index,
                    threshold=float(fast_dlm_threshold),
                )
                x_after = x.clone()
                proposed_full = torch.full_like(x_after, fill_value=mask_token_id)
                proposed_full[mask_index] = x0_masked
                x_after[transfer_mask] = proposed_full[transfer_mask]

        # context 里保存的是“本 step 做 transfer 之前”的视角：
        # x / hidden / logits / state 都对应同一个 pre-transfer 序列。
        # 这点对 pretrain 样本抓取很关键，因为 label/state 都要和这一步看到的 proposal 对齐。
        context = DreamStepContext(
            step=step_idx,
            steps=steps,
            x=x,
            logits=logits,
            mask_index=mask_index,
            x0_masked=x0_masked,
            full_scores=full_scores,
            number_transfer_tokens=0
            if proposal_alg == PROPOSAL_ALG_ORIGIN
            else (
                int(number_transfer_tokens.max().item())
                if isinstance(number_transfer_tokens, torch.Tensor)
                else int(number_transfer_tokens)
            ),
            prompt_length=prompt_length,
            mask_token_id=mask_token_id,
            attention_mask=full_attention_mask,
            position_ids=position_ids,
            hidden_states=tuple(model_outputs.hidden_states) if output_hidden_states else None,
        )
        if post_transfer_hook is not None:
            # post_transfer_hook 既可以修改 x_after，也可以顺便把当前 step 的 hidden / token / state 抓下来。
            x_after = post_transfer_hook(context, x_after)
        x = hook_tokens(step_idx, x_after, logits)

        if histories is not None:
            histories.append(x.clone())
        if hidden_trace is not None:
            hidden_trace.append(tuple(layer.detach().clone() for layer in model_outputs.hidden_states))
        if not (x[:, prompt_length:] == mask_token_id).any():
            # generation 区已经没有 mask，说明整条样本已经生成完成。
            break

    return {
        "sequences": x,
        "history": histories,
        "hidden_trace": hidden_trace,
    }
