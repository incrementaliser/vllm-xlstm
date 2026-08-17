"""Speed-plan prompts and fixed-length tokenization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def load_speed_prompts(path: Path | None = None) -> list[dict[str, Any]]:
    """Load short/medium/long prompt dicts from the package JSON."""
    from vllm_xlstm.paths import speed_prompts_path

    p = path or speed_prompts_path()
    return json.loads(p.read_text(encoding="utf-8"))


def fixed_length_ids(
    tokenizer: Any,
    text: str,
    length: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build a 1×length token tensor from ``text`` (truncate or pad).

    Args:
        tokenizer: HF tokenizer.
        text: Source prompt text.
        length: Exact token count.
        device: Torch device.

    Returns:
        ``input_ids`` of shape ``[1, length]``.
    """
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"][0].tolist()
    bos = tokenizer.bos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (
        tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    )
    if bos is not None and (not ids or ids[0] != bos):
        ids = [bos] + ids
    if len(ids) >= length:
        ids = ids[:length]
    else:
        ids = ids + [pad] * (length - len(ids))
    return torch.tensor([ids], device=device, dtype=torch.long)


def expand_batch(ids: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Repeat a 1×S prompt to batch_size×S."""
    if batch_size == 1:
        return ids
    return ids.repeat(batch_size, 1)
