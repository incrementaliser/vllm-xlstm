"""CLI: HF vs vLLM-xLSTM speed report → ``artifacts/runs/<DATE_TIME_EXPERIMENT>/``."""

from __future__ import annotations

import argparse
from pathlib import Path

from vllm_xlstm.io_util import tee_run_log
from vllm_xlstm.paths import default_hf_dir, new_run_dir


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse flags shared by 1-GPU and TP2 entry points."""
    p = argparse.ArgumentParser(
        description="HF vs vLLM-xLSTM speed report (industry TTFT/TPOT/throughput/e2e)."
    )
    p.add_argument(
        "--hf-dir",
        type=Path,
        default=None,
        help="Local HF snapshot (default: Hub cache snapshot if present).",
    )
    p.add_argument("--model-id", type=str, default="NX-AI/xLSTM-7b")
    p.add_argument("--gen-tokens", type=int, default=128)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--timed", type=int, default=5)
    p.add_argument(
        "--tp",
        "--tensor-parallel-size",
        dest="tensor_parallel_size",
        type=int,
        default=1,
        help="1 (default) or 2 for xlstm-vllm-speed-tp2.",
    )
    p.add_argument(
        "--experiment",
        type=str,
        default=None,
        help=(
            "Slug for artifacts/runs/<DATE_TIME_EXPERIMENT>/ "
            "(default: vllm-speed, or vllm-speed-tp2 for the TP2 entry point)."
        ),
    )
    p.add_argument("--skip-hf-native", action="store_true")
    p.add_argument("--skip-hf-triton", action="store_true")
    p.add_argument("--skip-vllm", action="store_true")
    p.add_argument("--skip-batch", action="store_true")
    p.add_argument("--skip-parity", action="store_true")
    p.add_argument("--skip-vllm-engine", action="store_true")
    return p.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    default_experiment: str = "vllm-speed",
) -> None:
    """CLI entry: 1-GPU (or ``--tp 2``) report into a DATE_TIME_EXPERIMENT directory."""
    from vllm_xlstm.bench import run_experiment

    args = _parse(argv)
    experiment = args.experiment or default_experiment
    run_dir = new_run_dir(experiment)
    hf_dir = args.hf_dir
    if hf_dir is None:
        hf_dir = default_hf_dir()
    with tee_run_log(run_dir):
        print(f"[vllm-speed] run directory: {run_dir}")
        print(f"[vllm-speed] tp={args.tensor_parallel_size} hf_dir={hf_dir}")
        run_experiment(
            run_dir,
            model_id=args.model_id,
            hf_dir=hf_dir,
            gen_tokens=args.gen_tokens,
            warmup=args.warmup,
            timed=args.timed,
            tensor_parallel_size=args.tensor_parallel_size,
            skip_hf_native=args.skip_hf_native,
            skip_hf_triton=args.skip_hf_triton,
            skip_vllm=args.skip_vllm,
            skip_batch=args.skip_batch,
            skip_parity=args.skip_parity,
            try_vllm_engine=not args.skip_vllm_engine,
        )


def main_tp2() -> None:
    """CLI entry: same protocol with ``tensor_parallel_size=2`` (needs 2 GPUs)."""
    import sys

    extra = list(sys.argv[1:])
    if "--tp" not in extra and "--tensor-parallel-size" not in extra:
        extra = ["--tp", "2", *extra]
    main(extra, default_experiment="vllm-speed-tp2")
