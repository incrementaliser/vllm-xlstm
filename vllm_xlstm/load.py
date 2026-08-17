"""Load NX-AI xLSTM-7b for HF baselines and the vLLM-xLSTM engine."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def resolve_model_src(model_id: str, hf_dir: Path | None) -> str:
    """Prefer a local Hub snapshot path when it exists."""
    if hf_dir is not None and hf_dir.is_dir():
        return str(hf_dir)
    return model_id


def load_tokenizer(model_src: str) -> Any:
    """Load the tokenizer and ensure a pad token exists."""
    tokenizer = AutoTokenizer.from_pretrained(model_src, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_hf_xlstm(
    model_src: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    step_kernel: str = "native",
    device_map: str | dict[str, int] | None = None,
) -> Any:
    """
    Load ``xLSTMForCausalLM`` with inference-mode mLSTM kernels.

    Args:
        model_src: Hub id or local snapshot.
        dtype: Parameter dtype.
        step_kernel: ``native`` or ``triton``.
        device_map: ``"cuda"``, ``"auto"``, or an explicit map.

    Returns:
        Eval-mode causal LM on the requested device(s).
    """
    config = AutoConfig.from_pretrained(model_src, trust_remote_code=True)
    config.mode = "inference"
    config.return_last_states = True
    config.step_kernel = step_kernel
    if step_kernel == "triton":
        config.chunkwise_kernel = "chunkwise--triton_xl_chunk"
        config.sequence_kernel = "native_sequence__triton"
    else:
        config.chunkwise_kernel = "chunkwise--native_autograd"
        config.sequence_kernel = "native_sequence__native"
    if device_map is None:
        device_map = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs: dict[str, Any] = {
        "config": config,
        "trust_remote_code": True,
        "dtype": dtype,
    }
    # ``from_pretrained(..., device_map="cuda:0")`` is unreliable; ``.to`` after load.
    move_to: str | dict[str, int] | None = None
    if isinstance(device_map, str) and device_map.startswith("cuda"):
        move_to = device_map
    else:
        kwargs["device_map"] = device_map
    try:
        model = AutoModelForCausalLM.from_pretrained(model_src, **kwargs)
    except TypeError:
        kwargs.pop("dtype", None)
        kwargs["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(model_src, **kwargs)
    if move_to is not None:
        model.to(move_to)
    model.eval()
    return model


def peak_mem_mib() -> float | None:
    """Max allocated CUDA memory across visible devices, in MiB."""
    if not torch.cuda.is_available():
        return None
    peak = 0.0
    for idx in range(torch.cuda.device_count()):
        peak = max(peak, torch.cuda.max_memory_allocated(idx) / (1024 * 1024))
    return round(peak, 1)


def reset_peak_mem() -> None:
    """Reset CUDA peak-memory stats on all devices."""
    if not torch.cuda.is_available():
        return
    for idx in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(idx)


def clear_cuda() -> None:
    """Drop Python refs' GPU tensors as far as the caching allocator allows."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cuda_sync() -> None:
    """Synchronize all visible CUDA devices (no-op on CPU)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gpu_count() -> int:
    """Visible CUDA device count."""
    return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
