"""Plots for TTFT, TPOT, decode tok/s, e2e, and batch throughput."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from vllm_xlstm.metrics import TimedSample


def write_plots(
    plots_dir: Path,
    speed_rows: Sequence[TimedSample],
    batch_rows: Sequence[TimedSample] | None = None,
) -> list[Path]:
    """
    Write PNGs. Never plot TPOT on a tok/s axis.

    Args:
        plots_dir: Destination directory.
        speed_rows: Single-stream (and TP overlay) samples.
        batch_rows: Optional batch-matrix samples.

    Returns:
        Paths of written figures (empty if matplotlib is missing).
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[vllm-xlstm] matplotlib not installed; skipping plots")
        return []

    written: list[Path] = []
    batch_rows = list(batch_rows or [])
    single = [
        r
        for r in speed_rows
        if r.batch_size == 1 and r.decode_tok_s_mean is not None and r.prompt_name != "error"
    ]

    def _key(row: TimedSample) -> str:
        gp = f"{row.gpu_count}gpu"
        return f"{row.backend}/{row.precision}/{gp}"

    def _line(
        *,
        filename: str,
        ylabel: str,
        title: str,
        rows: list[TimedSample],
        y_attr: str,
        yerr_attr: str,
    ) -> None:
        usable = [r for r in rows if getattr(r, y_attr) is not None]
        if not usable:
            return
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for key in sorted({_key(r) for r in usable}):
            subset = sorted(
                [r for r in usable if _key(r) == key],
                key=lambda x: x.prompt_tokens,
            )
            ax.errorbar(
                [r.prompt_tokens for r in subset],
                [getattr(r, y_attr) for r in subset],
                yerr=[getattr(r, yerr_attr) or 0 for r in subset],
                marker="o",
                label=key,
            )
        ax.set_xlabel("Input length (tokens)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
        out = plots_dir / filename
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        written.append(out)

    _line(
        filename="ttft_ms_vs_input_len.png",
        ylabel="TTFT (ms) — lower is better",
        title="Time to first token (includes prefill)",
        rows=single,
        y_attr="ttft_ms_mean",
        yerr_attr="ttft_ms_std",
    )
    _line(
        filename="tpot_ms_vs_input_len.png",
        ylabel="TPOT (ms/token) — lower is better",
        title="Inter-token latency after the first token",
        rows=single,
        y_attr="tpot_ms_mean",
        yerr_attr="tpot_ms_std",
    )
    _line(
        filename="decode_tok_s_vs_input_len.png",
        ylabel="Decode tok/s — higher is better",
        title="Decode throughput (1000 / TPOT)",
        rows=single,
        y_attr="decode_tok_s_mean",
        yerr_attr="decode_tok_s_std",
    )
    _line(
        filename="e2e_sec_vs_input_len.png",
        ylabel="E2E latency (s) — lower is better",
        title="End-to-end latency (prefill + 128 generated tokens)",
        rows=single,
        y_attr="e2e_sec_mean",
        yerr_attr="e2e_sec_std",
    )
    _line(
        filename="prefill_tok_s_vs_input_len.png",
        ylabel="Prefill tok/s — higher is better",
        title="Prefill throughput",
        rows=single,
        y_attr="prefill_tok_s_mean",
        yerr_attr="prefill_tok_s_std",
    )

    if batch_rows:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        usable_b = [r for r in batch_rows if r.decode_tok_s_mean is not None]
        for key in sorted({_key(r) for r in usable_b}):
            subset = sorted(
                [r for r in usable_b if _key(r) == key],
                key=lambda x: x.batch_size,
            )
            ax.errorbar(
                [r.batch_size for r in subset],
                [r.decode_tok_s_mean * r.batch_size for r in subset],
                yerr=[(r.decode_tok_s_std or 0) * r.batch_size for r in subset],
                marker="o",
                label=key,
            )
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Aggregate new-token tok/s — higher is better")
        ax.set_title("Batch decode throughput (medium prompt, gen=128)")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
        out = plots_dir / "batch_tok_s_vs_batch_size.png"
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        written.append(out)

    gpu_counts = {r.gpu_count for r in single}
    if len(gpu_counts) >= 2:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        medium = [r for r in single if r.prompt_name == "medium"]
        labels = [f"{r.backend}\n{r.gpu_count} GPU" for r in medium]
        vals = [r.decode_tok_s_mean or 0 for r in medium]
        ax.bar(range(len(medium)), vals)
        ax.set_xticks(range(len(medium)), labels, fontsize=8)
        ax.set_ylabel("Decode tok/s")
        ax.set_title("1-GPU vs 2-GPU decode (medium, gen=128)")
        ax.grid(True, axis="y", alpha=0.3)
        out = plots_dir / "decode_1gpu_vs_2gpu.png"
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        written.append(out)

    return written
