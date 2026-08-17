"""Greedy 7B parity: HF-native vs vLLM-xLSTM (gate for published speed)."""

from __future__ import annotations

from typing import Any

import torch

from vllm_xlstm.graph_engine import GraphDecodeEngine
from vllm_xlstm.hf_engine import greedy_prefill_decode
from vllm_xlstm.load import cuda_sync


def greedy_parity(
    hf_model: Any,
    vllm_engine: GraphDecodeEngine,
    tokenizer: Any,
    *,
    prompt: str = "The capital of France is",
    n_tokens: int = 16,
) -> dict[str, Any]:
    """
    Compare greedy token ids from HF-native (or the provided HF model) vs graph engine.

    Args:
        hf_model: Reference Hugging Face causal LM.
        vllm_engine: CUDA-graph / custom-op engine.
        tokenizer: Shared tokenizer.
        prompt: Short text prompt.
        n_tokens: Generated tokens (plan default: 16).

    Returns:
        Dict with ``ok``, token lists, and a gate string.
    """
    device = next(hf_model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    with torch.inference_mode():
        hf_tok, _ = greedy_prefill_decode(hf_model, input_ids, n_gen=n_tokens)
        cuda_sync()
        v_tok, _, mode = vllm_engine.generate(input_ids, n_gen=n_tokens)
        cuda_sync()
    hf_ids = hf_tok[0].tolist()
    v_ids = v_tok[0].tolist()
    n_match = sum(a == b for a, b in zip(hf_ids, v_ids, strict=False))
    ok = hf_ids == v_ids
    decoded_hf = tokenizer.decode(hf_ids, skip_special_tokens=True)
    decoded_v = tokenizer.decode(v_ids, skip_special_tokens=True)
    return {
        "ok": ok,
        "n_tokens": n_tokens,
        "n_match": n_match,
        "hf_ids": hf_ids,
        "vllm_ids": v_ids,
        "hf_text": decoded_hf,
        "vllm_text": decoded_v,
        "decode_mode": mode,
        "prompt": prompt,
        "gate": "PASS" if ok else "FAIL: greedy 16-token mismatch; speed claims not published",
    }
