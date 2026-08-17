"""Hugging Face greedy generate with industry TTFT (prefill included)."""

from __future__ import annotations

import time
from typing import Any

import torch
from transformers.models.xlstm.modeling_xlstm import xLSTMCache

from vllm_xlstm.load import cuda_sync
from vllm_xlstm.metrics import RawTiming, timing_from_parts


def _reset_cache(cache: xLSTMCache) -> None:
    """Zero (C, n, m) in place so CUDA-graph tensor identities stay valid."""
    cache.seqlen_offset = 0
    for layer in cache.rnn_state:
        for tensor in cache.rnn_state[layer]:
            tensor.zero_()


def _module_device(module: Any) -> torch.device:
    """Device of the first parameter in ``module``."""
    return next(module.parameters()).device


def new_cache(model: Any, batch_size: int) -> xLSTMCache:
    """
    Allocate a static xLSTM cache.

    When the model is split with ``device_map``, each layer's ``(C, n, m)`` is
    placed on that block's device so decode does not hang on a cross-GPU copy.
    """
    embed = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else model
    device = _module_device(embed)
    dtype = next(model.parameters()).dtype
    cache = xLSTMCache(model.config, max_batch_size=batch_size, device=device, dtype=dtype)
    blocks = getattr(getattr(model, "backbone", model), "blocks", None)
    if blocks is not None:
        for idx, block in enumerate(blocks):
            block_dev = _module_device(block)
            cache.rnn_state[idx] = tuple(tensor.to(block_dev) for tensor in cache.rnn_state[idx])
    return cache


def greedy_prefill_decode(
    model: Any,
    input_ids: torch.Tensor,
    *,
    n_gen: int,
    cache: xLSTMCache | None = None,
    decode_step: Any | None = None,
) -> tuple[torch.Tensor, RawTiming]:
    """
    Greedy generate with explicit prefill then token-by-token decode.

    TTFT is request start → first generated token (the argmax of the prefill
    last position). Subsequent tokens use ``decode_step`` if given, else a
    plain ``model`` forward.

    Args:
        model: ``xLSTMForCausalLM`` (or compatible) in eval mode.
        input_ids: Prompt ids ``[B, S]``.
        n_gen: Number of new tokens.
        cache: Optional pre-allocated cache; created when omitted.
        decode_step: Optional ``(token_buf, cache) -> next_token [B, 1]``.

    Returns:
        Generated ids ``[B, n_gen]`` and the raw timing for this call.
    """
    batch, n_prompt = input_ids.shape
    embed = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else model
    input_ids = input_ids.to(_module_device(embed))
    if cache is None:
        cache = new_cache(model, batch)
    else:
        _reset_cache(cache)

    generated = torch.empty(batch, n_gen, dtype=torch.long, device=input_ids.device)
    cuda_sync()
    t_start = time.perf_counter()
    with torch.inference_mode():
        out = model(input_ids=input_ids, cache_params=cache, use_cache=True)
        first = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated[:, :1] = first
        cuda_sync()
        t_first = time.perf_counter()
        token = first
        for i in range(1, n_gen):
            if decode_step is not None:
                token = decode_step(token, cache)
            else:
                out = model(input_ids=token, cache_params=cache, use_cache=True)
                token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated[:, i : i + 1] = token
        cuda_sync()
        t_end = time.perf_counter()
    timing = timing_from_parts(
        t_start=t_start,
        t_first=t_first,
        t_end=t_end,
        n_prompt=n_prompt,
        n_gen=n_gen,
        batch_size=batch,
    )
    return generated, timing
