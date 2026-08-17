"""Markdown report for HF vs vLLM-xLSTM speed runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from vllm_xlstm.metrics import TimedSample


def _fmt(val: float | None, digits: int = 2) -> str:
    """Format a float or em-dash."""
    if val is None:
        return "—"
    return f"{val:.{digits}f}"


def _pm(mean: float | None, std: float | None, digits: int = 2) -> str:
    """Format mean±std."""
    if mean is None:
        return "—"
    if std is None:
        return _fmt(mean, digits)
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def rows_to_markdown(rows: Sequence[TimedSample]) -> str:
    """Render a metrics table (TTFT, TPOT, tok/s, e2e)."""
    header = (
        "| backend | gpus | prompt | tokens | batch | TTFT ms ↓ | TPOT ms ↓ | "
        "decode tok/s ↑ | prefill tok/s ↑ | e2e s ↓ | sanity |"
    )
    sep = "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"
    lines = [header, sep]
    for row in rows:
        lines.append(
            "| {backend} | {gpus} | {prompt} | {pt} | {bs} | {ttft} | {tpot} | {dec} | {pref} | {e2e} | {san} |".format(
                backend=row.backend,
                gpus=row.gpu_count,
                prompt=row.prompt_name,
                pt=row.prompt_tokens,
                bs=row.batch_size,
                ttft=_pm(row.ttft_ms_mean, row.ttft_ms_std, 1),
                tpot=_pm(row.tpot_ms_mean, row.tpot_ms_std, 2),
                dec=_pm(row.decode_tok_s_mean, row.decode_tok_s_std, 2),
                pref=_pm(row.prefill_tok_s_mean, row.prefill_tok_s_std, 1),
                e2e=_pm(row.e2e_sec_mean, row.e2e_sec_std, 3),
                san="ok" if row.sanity_ok else ("fail" if row.sanity_ok is False else "—"),
            )
        )
    return "\n".join(lines)


def _pick(
    rows: Sequence[TimedSample],
    *,
    backend: str,
    prompt_name: str = "medium",
    batch_size: int = 1,
    gpu_count: int | None = None,
) -> TimedSample | None:
    """Find a sample matching backend/prompt/batch/gpu."""
    for row in rows:
        if row.backend != backend or row.prompt_name != prompt_name:
            continue
        if row.batch_size != batch_size:
            continue
        if gpu_count is not None and row.gpu_count != gpu_count:
            continue
        return row
    return None


def write_report(
    path: Path,
    *,
    env: dict[str, Any],
    parity: dict[str, Any],
    speed_rows: Sequence[TimedSample],
    batch_rows: Sequence[TimedSample],
    plot_paths: Sequence[Path],
    iterate_notes: Sequence[str],
    vllm_engine_note: str,
    tp_size: int,
) -> Path:
    """
    Write ``report.md`` with 1-GPU gate and optional 2-GPU section.

    Args:
        path: Destination markdown path (reproduce snippet uses ``path.parent.name``).
        env: ``env.json`` payload.
        parity: Greedy parity dict.
        speed_rows: Single-stream samples.
        batch_rows: Batch matrix samples.
        plot_paths: Written PNGs.
        iterate_notes: How decode was sped up (graphs, compile, …).
        vllm_engine_note: Real ``vllm.LLM`` status.
        tp_size: Requested tensor-parallel size.

    Returns:
        ``path``.
    """
    hf = _pick(speed_rows, backend="HF-native", gpu_count=1) or _pick(
        speed_rows, backend="HF-native"
    )
    vllm = _pick(speed_rows, backend="vLLM-xLSTM", gpu_count=1) or _pick(
        speed_rows, backend="vLLM-xLSTM"
    )
    parity_ok = bool(parity.get("ok"))
    gate_pass = False
    gate_line = "not evaluated"
    if hf and vllm and hf.decode_tok_s_mean and vllm.decode_tok_s_mean:
        gate_pass = bool(parity_ok and vllm.decode_tok_s_mean > hf.decode_tok_s_mean)
        if not parity_ok:
            gate_line = (
                f"FAIL (parity): speed not published. "
                f"HF-native {hf.decode_tok_s_mean:.2f} tok/s vs "
                f"vLLM-xLSTM {vllm.decode_tok_s_mean:.2f} tok/s (medium, gen=128)."
            )
        elif gate_pass:
            gate_line = (
                f"PASS: vLLM-xLSTM {vllm.decode_tok_s_mean:.2f} tok/s > "
                f"HF-native {hf.decode_tok_s_mean:.2f} tok/s "
                f"(medium, gen=128, 1 GPU)."
            )
        else:
            gate_line = (
                f"FAIL: vLLM-xLSTM {vllm.decode_tok_s_mean:.2f} tok/s ≤ "
                f"HF-native {hf.decode_tok_s_mean:.2f} tok/s "
                f"(medium, gen=128, 1 GPU)."
            )

    run_name = path.parent.name
    lines = [
        "# vLLM xLSTM speed report",
        "",
        "Reproduce:",
        "",
        "```bash",
        "uv run xlstm-vllm-speed",
        "# two GPUs:",
        "uv run xlstm-vllm-speed-tp2",
        f"cat artifacts/runs/{run_name}/report.md",
        "```",
        "",
        "Metrics glossary: `docs/metrics.md` (TTFT includes prefill; TPOT is ms/token; "
        "decode tok/s = 1000/TPOT; e2e lower is better).",
        "",
        "## Environment",
        "",
        "```json",
        __import__("json").dumps(env, indent=2),
        "```",
        "",
        f"- Real vLLM engine: {vllm_engine_note}",
        f"- Requested TP size: {tp_size}",
        "",
        "## Parity gate (greedy 16 tokens)",
        "",
        f"- ok: `{parity_ok}`",
        f"- gate: {parity.get('gate', 'n/a')}",
        f"- decode mode: {parity.get('decode_mode', 'n/a')}",
        f"- match {parity.get('n_match', '?')}/{parity.get('n_tokens', 16)}",
        "",
        "## 1-GPU success gate",
        "",
        f"**{gate_line}**",
        "",
        "The gate is medium prompt / gen=128 / **vLLM-xLSTM decode tok/s > HF-native** "
        "on one GPU. Two-GPU TP is reported below but is not required to pass.",
        "",
        "## Iterate notes",
        "",
    ]
    if iterate_notes:
        lines.extend(f"- {n}" for n in iterate_notes)
    else:
        lines.append("- (none)")
    lines += [
        "",
        "## Single-stream (batch=1)",
        "",
        rows_to_markdown(speed_rows),
        "",
        "## Batch throughput (medium, gen=128)",
        "",
    ]
    if batch_rows:
        lines.append(rows_to_markdown(batch_rows))
    else:
        lines.append("_not run_")
    tp2 = [r for r in list(speed_rows) + list(batch_rows) if r.gpu_count >= 2]
    lines += [
        "",
        "## 2-GPU (tensor parallel / device_map)",
        "",
    ]
    if tp2:
        lines.append(
            "7B already fits one A40 (~14 GiB). TP=2 is a capacity / serving "
            "measurement. NCCL can make batch=1 decode **worse**; batch throughput "
            "may still improve."
        )
        lines.append("")
        lines.append(rows_to_markdown(tp2))
    else:
        lines.append(
            "_Not run in this job (use `uv run xlstm-vllm-speed-tp2` with 2 GPUs)._"
        )
    lines += ["", "## Plots", ""]
    if plot_paths:
        for p in plot_paths:
            lines.append(f"- `{p.name}`")
    else:
        lines.append("_none_")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
