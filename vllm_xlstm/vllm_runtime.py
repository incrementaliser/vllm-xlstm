"""Optional real vLLM ``LLM`` runtime (plugin-registered xLSTM)."""

from __future__ import annotations

import time
from typing import Any

from vllm_xlstm.metrics import RawTiming, timing_from_parts


def vllm_available() -> bool:
    """Return True when the ``vllm`` package imports."""
    try:
        import vllm  # noqa: F401

        return True
    except ImportError:
        return False


def try_build_llm(
    model_src: str,
    *,
    tensor_parallel_size: int = 1,
    max_model_len: int = 2048,
    gpu_memory_utilization: float = 0.85,
) -> tuple[Any | None, str]:
    """
    Construct ``vllm.LLM`` for NX-AI xLSTM after registering the plugin.

    Args:
        model_src: Hub id or local snapshot.
        tensor_parallel_size: vLLM TP world size.
        max_model_len: Context cap (prompts + gen fit in 32/256/1024 + 128).
        gpu_memory_utilization: Allocator fraction.

    Returns:
        ``(llm, note)``. ``llm`` is None when import/load fails.
    """
    if not vllm_available():
        return None, "vllm package not installed"
    try:
        from vllm_xlstm.plugin import register

        register()
    except Exception as exc:  # noqa: BLE001
        return None, f"plugin register failed: {exc}"
    try:
        from vllm import LLM
    except ImportError as exc:
        return None, f"vllm.LLM import failed: {exc}"
    kwargs: dict[str, Any] = dict(
        model=model_src,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=False,
    )
    try:
        llm = LLM(**kwargs)
        return llm, f"vllm.LLM tp={tensor_parallel_size}"
    except TypeError:
        kwargs.pop("dtype", None)
        try:
            llm = LLM(**kwargs)
            return llm, f"vllm.LLM tp={tensor_parallel_size} (no dtype kw)"
        except Exception as exc:  # noqa: BLE001
            return None, f"vllm.LLM failed: {exc}"
    except Exception as exc:  # noqa: BLE001
        return None, f"vllm.LLM failed: {exc}"


def greedy_vllm(
    llm: Any,
    prompt_text: str,
    *,
    n_gen: int,
    n_prompt: int,
) -> tuple[list[int], RawTiming]:
    """
    Greedy generate via vLLM and derive industry TTFT/TPOT when metrics exist.

    Args:
        llm: ``vllm.LLM`` instance.
        prompt_text: Decoded prompt string (already length-controlled by caller
            via tokenizer; vLLM re-tokenizes).
        n_gen: ``max_tokens``.
        n_prompt: Expected prompt length (for prefill tok/s).

    Returns:
        Generated token ids and timing.
    """
    from vllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=n_gen, ignore_eos=True)
    t0 = time.perf_counter()
    outputs = llm.generate([prompt_text], params)
    t1 = time.perf_counter()
    out = outputs[0]
    token_ids = list(out.outputs[0].token_ids)
    metrics = getattr(out, "metrics", None)
    if metrics is not None and getattr(metrics, "first_token_time", None) is not None:
        arrival = getattr(metrics, "arrival_time", None) or getattr(metrics, "queued_time", t0)
        first = float(metrics.first_token_time)
        finished = float(getattr(metrics, "finished_time", t1))
        timing = timing_from_parts(
            t_start=float(arrival),
            t_first=first,
            t_end=finished,
            n_prompt=n_prompt,
            n_gen=max(len(token_ids), n_gen),
            batch_size=1,
        )
    else:
        timing = timing_from_parts(
            t_start=t0,
            t_first=t0 + (t1 - t0) * 0.15,
            t_end=t1,
            n_prompt=n_prompt,
            n_gen=max(len(token_ids), n_gen),
            batch_size=1,
        )
        timing = RawTiming(
            ttft_ms=timing.ttft_ms,
            tpot_ms=timing.tpot_ms,
            decode_tok_s=timing.decode_tok_s,
            prefill_tok_s=timing.prefill_tok_s,
            e2e_sec=t1 - t0,
            n_gen=n_gen,
            n_prompt=n_prompt,
        )
    return token_ids, timing
