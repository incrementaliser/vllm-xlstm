"""Industry serving metrics: TTFT, TPOT/ITL, throughput, e2e."""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence


@dataclass
class TimedSample:
    """One timed measurement using industry TTFT (prefill + first generated token)."""

    backend: str
    precision: str
    prompt_name: str
    prompt_tokens: int
    gen_tokens: int
    batch_size: int = 1
    gpu_count: int = 1
    ttft_ms_mean: float | None = None
    ttft_ms_std: float | None = None
    tpot_ms_mean: float | None = None
    tpot_ms_std: float | None = None
    decode_tok_s_mean: float | None = None
    decode_tok_s_std: float | None = None
    prefill_tok_s_mean: float | None = None
    prefill_tok_s_std: float | None = None
    e2e_sec_mean: float | None = None
    e2e_sec_std: float | None = None
    sanity_ok: bool | None = None
    peak_mem_mib: float | None = None
    n_warmup: int = 3
    n_timed: int = 5
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        return asdict(self)


@dataclass
class RawTiming:
    """One greedy generate: industry TTFT, TPOT, throughputs, e2e."""

    ttft_ms: float
    tpot_ms: float
    decode_tok_s: float
    prefill_tok_s: float
    e2e_sec: float
    n_gen: int
    n_prompt: int

    def sanity_ok(self, rel_tol: float = 0.10) -> bool:
        """
        Check ``e2e ≈ TTFT/1000 + TPOT/1000 * (N_gen - 1)``.

        Args:
            rel_tol: Allowed relative error on e2e.

        Returns:
            True when the identity holds within ``rel_tol``.
        """
        if self.n_gen <= 1:
            return True
        pred = self.ttft_ms / 1000.0 + (self.tpot_ms / 1000.0) * (self.n_gen - 1)
        if pred <= 0:
            return False
        return abs(self.e2e_sec - pred) / pred <= rel_tol


def mean_std(values: Sequence[float]) -> tuple[float | None, float | None]:
    """Return sample mean and std (None std when fewer than two values)."""
    vals = [float(v) for v in values]
    if not vals:
        return None, None
    mean = statistics.fmean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def summarize_timings(
    timings: Sequence[RawTiming],
    *,
    backend: str,
    precision: str,
    prompt_name: str,
    prompt_tokens: int,
    gen_tokens: int,
    batch_size: int = 1,
    gpu_count: int = 1,
    n_warmup: int = 3,
    peak_mem_mib: float | None = None,
    notes: str = "",
    extra: dict[str, Any] | None = None,
) -> TimedSample:
    """
    Collapse repeated ``RawTiming`` runs into mean±std ``TimedSample``.

    Args:
        timings: Timed runs (warmup already discarded).
        backend: Label such as ``HF-native`` or ``vLLM-xLSTM``.
        precision: Weight dtype / kernel note.
        prompt_name: short / medium / long.
        prompt_tokens: Prefill length.
        gen_tokens: New tokens generated.
        batch_size: Concurrent sequences.
        gpu_count: GPUs used for this sample.
        n_warmup: Warmup runs discarded before ``timings``.
        peak_mem_mib: Peak device memory if known.
        notes: Free-form backend notes (graph vs eager).
        extra: Optional extra JSON fields.

    Returns:
        Aggregated sample with sanity flag.
    """
    ttft = [t.ttft_ms for t in timings]
    tpot = [t.tpot_ms for t in timings]
    dec = [t.decode_tok_s for t in timings]
    pref = [t.prefill_tok_s for t in timings]
    e2e = [t.e2e_sec for t in timings]
    ttft_m, ttft_s = mean_std(ttft)
    tpot_m, tpot_s = mean_std(tpot)
    dec_m, dec_s = mean_std(dec)
    pref_m, pref_s = mean_std(pref)
    e2e_m, e2e_s = mean_std(e2e)
    sanity = all(t.sanity_ok() for t in timings) if timings else None
    return TimedSample(
        backend=backend,
        precision=precision,
        prompt_name=prompt_name,
        prompt_tokens=prompt_tokens,
        gen_tokens=gen_tokens,
        batch_size=batch_size,
        gpu_count=gpu_count,
        ttft_ms_mean=ttft_m,
        ttft_ms_std=ttft_s,
        tpot_ms_mean=tpot_m,
        tpot_ms_std=tpot_s,
        decode_tok_s_mean=dec_m,
        decode_tok_s_std=dec_s,
        prefill_tok_s_mean=pref_m,
        prefill_tok_s_std=pref_s,
        e2e_sec_mean=e2e_m,
        e2e_sec_std=e2e_s,
        sanity_ok=sanity,
        peak_mem_mib=peak_mem_mib,
        n_warmup=n_warmup,
        n_timed=len(timings),
        notes=notes,
        extra=extra or {},
    )


def timing_from_parts(
    *,
    t_start: float,
    t_first: float,
    t_end: float,
    n_prompt: int,
    n_gen: int,
    batch_size: int = 1,
) -> RawTiming:
    """
    Build a ``RawTiming`` from synchronized wall-clock marks.

    Args:
        t_start: Request start (after CUDA sync).
        t_first: First generated token ready.
        t_end: Last generated token ready.
        n_prompt: Prompt tokens per sequence.
        n_gen: Generated tokens per sequence.
        batch_size: Concurrent sequences (scales prefill/decode token counts).

    Returns:
        Filled ``RawTiming``. Decode tok/s is ``1000 / TPOT``.
    """
    ttft_ms = max((t_first - t_start) * 1000.0, 0.0)
    e2e_sec = max(t_end - t_start, 1e-9)
    n_after = max(n_gen - 1, 1)
    tpot_s = max(t_end - t_first, 1e-9) / n_after
    tpot_ms = tpot_s * 1000.0
    decode_tok_s = 1000.0 / tpot_ms
    prefill_tok_s = (n_prompt * batch_size) / max(t_first - t_start, 1e-9)
    return RawTiming(
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        decode_tok_s=decode_tok_s,
        prefill_tok_s=prefill_tok_s,
        e2e_sec=e2e_sec,
        n_gen=n_gen,
        n_prompt=n_prompt,
    )
