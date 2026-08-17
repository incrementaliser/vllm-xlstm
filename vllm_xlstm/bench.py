"""Collect env metadata and run the HF vs vLLM-xLSTM speed matrix."""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from typing import Any, Callable

import torch

from vllm_xlstm.graph_engine import GraphDecodeEngine, try_compile_step
from vllm_xlstm.hf_engine import greedy_prefill_decode
from vllm_xlstm.load import (
    clear_cuda,
    cuda_sync,
    gpu_count,
    load_hf_xlstm,
    load_tokenizer,
    peak_mem_mib,
    reset_peak_mem,
    resolve_model_src,
)
from vllm_xlstm.metrics import TimedSample, summarize_timings
from vllm_xlstm.parity import greedy_parity
from vllm_xlstm.plots import write_plots
from vllm_xlstm.prompts import expand_batch, fixed_length_ids, load_speed_prompts
from vllm_xlstm.report import write_report
from vllm_xlstm.vllm_runtime import try_build_llm


def collect_env_metadata() -> dict[str, Any]:
    """Hardware / software metadata for ``env.json``."""
    meta: dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        import torch as _torch

        meta["torch"] = _torch.__version__
        meta["cuda_available"] = _torch.cuda.is_available()
        if _torch.cuda.is_available():
            meta["gpu_name"] = _torch.cuda.get_device_name(0)
            meta["gpu_count"] = _torch.cuda.device_count()
            props = _torch.cuda.get_device_properties(0)
            meta["gpu_total_mem_mib"] = round(props.total_memory / (1024 * 1024), 1)
            names = [_torch.cuda.get_device_name(i) for i in range(_torch.cuda.device_count())]
            meta["gpu_names"] = names
    except ImportError:
        meta["torch"] = None
    try:
        import transformers

        meta["transformers"] = transformers.__version__
    except ImportError:
        meta["transformers"] = None
    try:
        import vllm

        meta["vllm"] = getattr(vllm, "__version__", "unknown")
    except ImportError:
        meta["vllm"] = None
    try:
        import xlstm

        meta["xlstm"] = getattr(xlstm, "__version__", "unknown")
    except ImportError:
        meta["xlstm"] = None
    try:
        import mlstm_kernels

        meta["mlstm_kernels"] = getattr(mlstm_kernels, "__version__", "present")
    except ImportError:
        meta["mlstm_kernels"] = None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
            timeout=10,
        )
        meta["nvidia_smi"] = out.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return meta


def _timed_loop(
    fn: Callable[[], Any],
    *,
    warmup: int,
    timed: int,
) -> list[Any]:
    """Run ``fn`` warmup+timed times; return timed results."""
    for _ in range(max(warmup, 0)):
        fn()
        cuda_sync()
    out = []
    for _ in range(timed):
        out.append(fn())
        cuda_sync()
    return out


def bench_hf_matrix(
    model: Any,
    tokenizer: Any,
    prompts: list[dict[str, Any]],
    *,
    backend: str,
    precision: str,
    gen_tokens: int,
    warmup: int,
    timed: int,
    gpu_n: int,
    device: torch.device,
) -> list[TimedSample]:
    """Single-stream industry metrics for a Hugging Face model."""
    rows: list[TimedSample] = []
    reset_peak_mem()
    for spec in prompts:
        ids = fixed_length_ids(tokenizer, spec["prompt"], int(spec["approx_tokens"]), device)

        def _once(ids: torch.Tensor = ids) -> Any:
            _tok, timing = greedy_prefill_decode(model, ids, n_gen=gen_tokens)
            return timing

        timings = _timed_loop(_once, warmup=warmup, timed=timed)
        rows.append(
            summarize_timings(
                timings,
                backend=backend,
                precision=precision,
                prompt_name=str(spec["name"]),
                prompt_tokens=int(spec["approx_tokens"]),
                gen_tokens=gen_tokens,
                batch_size=1,
                gpu_count=gpu_n,
                n_warmup=warmup,
                peak_mem_mib=peak_mem_mib(),
            )
        )
        print(
            f"[bench] {backend} {spec['name']}: "
            f"decode={rows[-1].decode_tok_s_mean:.2f} tok/s "
            f"TTFT={rows[-1].ttft_ms_mean:.1f} ms "
            f"TPOT={rows[-1].tpot_ms_mean:.2f} ms"
        )
    return rows


def bench_engine_matrix(
    engine: GraphDecodeEngine,
    tokenizer: Any,
    prompts: list[dict[str, Any]],
    *,
    backend: str,
    precision: str,
    gen_tokens: int,
    warmup: int,
    timed: int,
    gpu_n: int,
    device: torch.device,
    notes: str = "",
) -> list[TimedSample]:
    """Single-stream industry metrics for the CUDA-graph engine."""
    rows: list[TimedSample] = []
    engine.prepare(1)
    reset_peak_mem()
    for spec in prompts:
        ids = fixed_length_ids(tokenizer, spec["prompt"], int(spec["approx_tokens"]), device)

        def _once(ids: torch.Tensor = ids) -> Any:
            _tok, timing, _mode = engine.generate(ids, n_gen=gen_tokens)
            return timing

        timings = _timed_loop(_once, warmup=warmup, timed=timed)
        rows.append(
            summarize_timings(
                timings,
                backend=backend,
                precision=precision,
                prompt_name=str(spec["name"]),
                prompt_tokens=int(spec["approx_tokens"]),
                gen_tokens=gen_tokens,
                batch_size=1,
                gpu_count=gpu_n,
                n_warmup=warmup,
                peak_mem_mib=peak_mem_mib(),
                notes=notes or engine.decode_mode,
            )
        )
        print(
            f"[bench] {backend} {spec['name']} ({engine.decode_mode}): "
            f"decode={rows[-1].decode_tok_s_mean:.2f} tok/s "
            f"TTFT={rows[-1].ttft_ms_mean:.1f} ms "
            f"TPOT={rows[-1].tpot_ms_mean:.2f} ms"
        )
    return rows


def bench_batch_matrix(
    run_one: Callable[[torch.Tensor, int], Any],
    tokenizer: Any,
    medium: dict[str, Any],
    *,
    backend: str,
    precision: str,
    gen_tokens: int,
    warmup: int,
    timed: int,
    gpu_n: int,
    device: torch.device,
    batches: tuple[int, ...] = (1, 2, 4, 8),
    notes: str = "",
) -> list[TimedSample]:
    """Batch 1/2/4/8 on the medium prompt."""
    rows: list[TimedSample] = []
    base = fixed_length_ids(tokenizer, medium["prompt"], int(medium["approx_tokens"]), device)
    for bs in batches:
        ids = expand_batch(base, bs)

        def _once(ids: torch.Tensor = ids, bs: int = bs) -> Any:
            return run_one(ids, bs)

        timings = _timed_loop(_once, warmup=max(1, warmup // 2), timed=max(2, timed // 2))
        rows.append(
            summarize_timings(
                timings,
                backend=backend,
                precision=precision,
                prompt_name="medium",
                prompt_tokens=int(medium["approx_tokens"]),
                gen_tokens=gen_tokens,
                batch_size=bs,
                gpu_count=gpu_n,
                n_warmup=max(1, warmup // 2),
                peak_mem_mib=peak_mem_mib(),
                notes=notes,
            )
        )
        print(
            f"[bench] {backend} batch={bs}: "
            f"decode={rows[-1].decode_tok_s_mean:.2f} tok/s/stream "
            f"agg~{(rows[-1].decode_tok_s_mean or 0) * bs:.1f} tok/s"
        )
    return rows


def _medium_decode(rows: list[TimedSample], backend: str) -> float | None:
    """Medium-prompt decode tok/s for ``backend``."""
    for row in rows:
        if row.backend == backend and row.prompt_name == "medium" and row.batch_size == 1:
            return row.decode_tok_s_mean
    return None


def iterate_vllm_engine(
    model: Any,
    tokenizer: Any,
    prompts: list[dict[str, Any]],
    *,
    hf_native_decode: float | None,
    gen_tokens: int,
    warmup: int,
    timed: int,
    gpu_n: int,
    device: torch.device,
) -> tuple[GraphDecodeEngine, list[TimedSample], list[str], dict[str, Any]]:
    """
    Try CUDA graphs ± custom ops ± compile until decode beats HF-native.

    Returns:
        Winning engine, its speed rows, iterate notes, and parity-vs-triton dict.
    """
    notes: list[str] = []
    medium = next(p for p in prompts if p["name"] == "medium")
    ids_med = fixed_length_ids(tokenizer, medium["prompt"], int(medium["approx_tokens"]), device)

    def _score(engine: GraphDecodeEngine) -> float:
        engine.prepare(1)
        _tok, timing, mode = engine.generate(ids_med, n_gen=gen_tokens)
        notes.append(f"probe decode={timing.decode_tok_s:.2f} tok/s mode={mode}")
        return timing.decode_tok_s

    candidates: list[tuple[str, GraphDecodeEngine]] = []
    # Same weights; graphs around HF triton (no extra wrapper).
    e_graph = GraphDecodeEngine(model, use_triton=True, patch_ops=False)
    candidates.append(("cuda_graph/unpatched", e_graph))
    best_engine = e_graph
    best_score = _score(e_graph)
    best_label = "cuda_graph/unpatched"
    notes.append(f"baseline graph (no extra op patch): {best_score:.2f} tok/s")

    if hf_native_decode is not None and best_score <= hf_native_decode:
        e_op = GraphDecodeEngine(model, use_triton=True, patch_ops=True)
        s_op = _score(e_op)
        notes.append(f"custom-op + graph: {s_op:.2f} tok/s")
        if s_op > best_score:
            best_engine, best_score, best_label = e_op, s_op, "cuda_graph/custom_op"

    if hf_native_decode is not None and best_score <= hf_native_decode:
        if try_compile_step(best_engine):
            best_engine._sessions.clear()
            s_c = _score(best_engine)
            notes.append(f"torch.compile: {s_c:.2f} tok/s")
            if s_c > best_score:
                best_score, best_label = s_c, best_label + "+compile"

    if hf_native_decode is None:
        notes.append("HF-native decode unavailable; keeping fastest probe")
    elif best_score > hf_native_decode:
        notes.append(
            f"WIN: {best_label} {best_score:.2f} tok/s > HF-native {hf_native_decode:.2f}"
        )
    else:
        notes.append(
            f"still behind HF-native ({hf_native_decode:.2f}); publishing {best_label} "
            f"{best_score:.2f} tok/s"
        )

    rows = bench_engine_matrix(
        best_engine,
        tokenizer,
        prompts,
        backend="vLLM-xLSTM",
        precision=f"bf16/{best_label}",
        gen_tokens=gen_tokens,
        warmup=warmup,
        timed=timed,
        gpu_n=gpu_n,
        device=device,
        notes=best_label,
    )
    par = greedy_parity(model, best_engine, tokenizer, n_tokens=16)
    par["iterate_best"] = best_label
    par["iterate_probe_tok_s"] = best_score
    return best_engine, rows, notes, par


def _run_tp2_overlay(
    *,
    model_src: str,
    tokenizer: Any,
    prompts: list[dict[str, Any]],
    dtype: torch.dtype,
    gen_tokens: int,
    warmup: int,
    timed: int,
    skip_batch: bool,
    speed_rows: list[TimedSample],
    batch_rows: list[TimedSample],
) -> tuple[list[TimedSample], list[TimedSample]]:
    """HF ``device_map=auto`` and vLLM-xLSTM on two GPUs (capacity, not the gate)."""
    print("[bench] 2-GPU overlay (device_map=auto)…")
    vis = gpu_count()
    medium = next(p for p in prompts if p["name"] == "medium")
    native = load_hf_xlstm(
        model_src, dtype=dtype, step_kernel="native", device_map="auto"
    )
    n_dev = next(native.get_input_embeddings().parameters()).device
    print(f"[bench] HF-native devices: embed={n_dev} last={next(reversed(list(native.parameters()))).device}")
    speed_rows.extend(
        bench_hf_matrix(
            native,
            tokenizer,
            prompts,
            backend="HF-native",
            precision="bf16/native/device_map",
            gen_tokens=gen_tokens,
            warmup=warmup,
            timed=timed,
            gpu_n=vis,
            device=n_dev,
        )
    )
    del native
    clear_cuda()
    triton = load_hf_xlstm(
        model_src, dtype=dtype, step_kernel="triton", device_map="auto"
    )
    t_dev = next(triton.parameters()).device
    speed_rows.extend(
        bench_hf_matrix(
            triton,
            tokenizer,
            prompts,
            backend="HF-triton",
            precision="bf16/triton/device_map",
            gen_tokens=gen_tokens,
            warmup=warmup,
            timed=timed,
            gpu_n=vis,
            device=t_dev,
        )
    )
    engine = GraphDecodeEngine(triton, use_triton=True, patch_ops=False)
    speed_rows.extend(
        bench_engine_matrix(
            engine,
            tokenizer,
            prompts,
            backend="vLLM-xLSTM",
            precision="bf16/device_map",
            gen_tokens=gen_tokens,
            warmup=warmup,
            timed=timed,
            gpu_n=vis,
            device=t_dev,
            notes="tp2-eager-or-graph",
        )
    )
    if not skip_batch:
        def _eng(ids: torch.Tensor, bs: int) -> Any:
            engine.prepare(bs)
            _t, timing, _m = engine.generate(ids, n_gen=gen_tokens)
            return timing

        batch_rows.extend(
            bench_batch_matrix(
                _eng,
                tokenizer,
                medium,
                backend="vLLM-xLSTM",
                precision="bf16/device_map",
                gen_tokens=gen_tokens,
                warmup=warmup,
                timed=timed,
                gpu_n=vis,
                device=t_dev,
                notes="tp2",
            )
        )
    del triton
    del engine
    clear_cuda()
    return speed_rows, batch_rows


def run_experiment(
    run_dir: Path,
    *,
    model_id: str,
    hf_dir: Path | None,
    gen_tokens: int = 128,
    warmup: int = 3,
    timed: int = 5,
    tensor_parallel_size: int = 1,
    skip_hf_native: bool = False,
    skip_hf_triton: bool = False,
    skip_vllm: bool = False,
    skip_batch: bool = False,
    skip_parity: bool = False,
    try_vllm_engine: bool = True,
) -> dict[str, Any]:
    """
    Full protocol: HF-native, HF-triton, vLLM-xLSTM, optional TP2 / vLLM.LLM.

    Args:
        run_dir: ``DATE_TIME_EXPERIMENT`` artifacts directory.
        model_id: Hub id fallback.
        hf_dir: Local snapshot if present.
        gen_tokens: New tokens (default 128).
        warmup: Discarded runs.
        timed: Timed runs.
        tensor_parallel_size: 1 or 2.
        skip_*: Toggles for debugging.
        try_vllm_engine: Attempt real ``vllm.LLM`` in addition to the plugin engine.

    Returns:
        Summary dict written to ``results/summary.json``.
    """
    from vllm_xlstm.io_util import write_json

    if not torch.cuda.is_available():
        raise SystemExit(
            "xlstm-vllm-speed needs a CUDA GPU. Submit: sbatch scripts/job.sh "
            "(or scripts/job_tp2.sh for two GPUs)."
        )
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    model_src = resolve_model_src(model_id, hf_dir)
    env = collect_env_metadata()
    env["model_src"] = model_src
    env["tensor_parallel_size"] = tensor_parallel_size
    write_json(run_dir / "env.json", env)
    prompts = load_speed_prompts()
    tokenizer = load_tokenizer(model_src)
    vis_gpu = gpu_count()
    gpu_n = vis_gpu if vis_gpu else 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    speed_rows: list[TimedSample] = []
    batch_rows: list[TimedSample] = []
    iterate_notes: list[str] = []
    parity: dict[str, Any] = {"ok": False, "gate": "not run"}
    native_tokens: list[int] | None = None
    hf_native_decode: float | None = None
    vllm_engine_note = "not attempted"
    engine: GraphDecodeEngine | None = None

    if not skip_hf_native:
        print("[bench] loading HF-native…")
        native = load_hf_xlstm(
            model_src,
            dtype=dtype,
            step_kernel="native",
            device_map="cuda:0" if torch.cuda.is_available() else None,
        )
        n_gpu = 1
        try:
            if not skip_parity:
                p_ids = tokenizer("The capital of France is", return_tensors="pt")["input_ids"].to(
                    next(native.parameters()).device
                )
                tok, _ = greedy_prefill_decode(native, p_ids, n_gen=16)
                native_tokens = tok[0].tolist()
            speed_rows.extend(
                bench_hf_matrix(
                    native,
                    tokenizer,
                    prompts,
                    backend="HF-native",
                    precision="bf16/native",
                    gen_tokens=gen_tokens,
                    warmup=warmup,
                    timed=timed,
                    gpu_n=n_gpu,
                    device=next(native.parameters()).device,
                )
            )
            hf_native_decode = _medium_decode(speed_rows, "HF-native")
        finally:
            del native
            clear_cuda()

    triton_model = None
    if not skip_hf_triton or not skip_vllm:
        print("[bench] loading HF-triton…")
        triton_model = load_hf_xlstm(
            model_src,
            dtype=dtype,
            step_kernel="triton",
            device_map="cuda:0" if torch.cuda.is_available() else None,
        )
        t_device = next(triton_model.parameters()).device
        n_gpu = 1
        if not skip_hf_triton:
            speed_rows.extend(
                bench_hf_matrix(
                    triton_model,
                    tokenizer,
                    prompts,
                    backend="HF-triton",
                    precision="bf16/triton",
                    gen_tokens=gen_tokens,
                    warmup=warmup,
                    timed=timed,
                    gpu_n=n_gpu,
                    device=t_device,
                )
            )

    if not skip_vllm and triton_model is not None:
        engine, v_rows, iterate_notes, par_engine = iterate_vllm_engine(
            triton_model,
            tokenizer,
            prompts,
            hf_native_decode=hf_native_decode,
            gen_tokens=gen_tokens,
            warmup=warmup,
            timed=timed,
            gpu_n=1,
            device=next(triton_model.parameters()).device,
        )
        speed_rows.extend(v_rows)
        if native_tokens is not None:
            p_ids = tokenizer("The capital of France is", return_tensors="pt")["input_ids"].to(
                next(triton_model.parameters()).device
            )
            v_tok, _, mode = engine.generate(p_ids, n_gen=16)
            v_ids = v_tok[0].tolist()
            ok_native = v_ids == native_tokens
            ok_kernel = bool(par_engine.get("ok"))
            parity = {
                "ok": ok_native or ok_kernel,
                "ok_native": ok_native,
                "ok_same_kernel": ok_kernel,
                "hf_ids": native_tokens,
                "vllm_ids": v_ids,
                "n_tokens": 16,
                "n_match": sum(a == b for a, b in zip(native_tokens, v_ids, strict=False)),
                "decode_mode": mode,
                "gate": (
                    "PASS"
                    if ok_native
                    else (
                        "PASS_KERNEL (matches HF-triton, not native kernels)"
                        if ok_kernel
                        else "FAIL: greedy mismatch; speed claims not published"
                    )
                ),
            }
        else:
            parity = par_engine
        if not skip_batch:
            def _eng(ids: torch.Tensor, bs: int) -> Any:
                engine.prepare(bs)
                _t, timing, _m = engine.generate(ids, n_gen=gen_tokens)
                return timing

            medium = next(p for p in prompts if p["name"] == "medium")
            batch_rows.extend(
                bench_batch_matrix(
                    _eng,
                    tokenizer,
                    medium,
                    backend="vLLM-xLSTM",
                    precision=v_rows[0].precision if v_rows else "bf16",
                    gen_tokens=gen_tokens,
                    warmup=warmup,
                    timed=timed,
                    gpu_n=1,
                    device=next(triton_model.parameters()).device,
                    notes=engine.decode_mode,
                )
            )
            if not skip_hf_triton:
                def _hf(ids: torch.Tensor, bs: int) -> Any:
                    _t, timing = greedy_prefill_decode(triton_model, ids, n_gen=gen_tokens)
                    return timing

                batch_rows.extend(
                    bench_batch_matrix(
                        _hf,
                        tokenizer,
                        medium,
                        backend="HF-triton",
                        precision="bf16/triton",
                        gen_tokens=gen_tokens,
                        warmup=warmup,
                        timed=timed,
                        gpu_n=n_gpu,
                        device=next(triton_model.parameters()).device,
                    )
                )

    if try_vllm_engine:
        llm, vllm_engine_note = try_build_llm(
            model_src, tensor_parallel_size=tensor_parallel_size
        )
        if llm is not None:
            iterate_notes.append(f"vllm.LLM constructed: {vllm_engine_note}")
            del llm
            clear_cuda()
        else:
            iterate_notes.append(f"vllm.LLM skipped: {vllm_engine_note}")
    else:
        vllm_engine_note = "skipped by flag"

    if triton_model is not None:
        del triton_model
        engine = None
        clear_cuda()

    if tensor_parallel_size >= 2 and vis_gpu >= 2:
        iterate_notes.append("2-GPU overlay: HF device_map=auto + vLLM-xLSTM (graphs likely disabled)")
        try:
            speed_rows, batch_rows = _run_tp2_overlay(
                model_src=model_src,
                tokenizer=tokenizer,
                prompts=prompts,
                dtype=dtype,
                gen_tokens=gen_tokens,
                warmup=warmup,
                timed=timed,
                skip_batch=skip_batch,
                speed_rows=speed_rows,
                batch_rows=batch_rows,
            )
        except Exception as exc:  # noqa: BLE001
            iterate_notes.append(f"2-GPU overlay failed: {exc}")
            print(f"[bench] 2-GPU overlay failed: {exc}")
            clear_cuda()

    write_json(run_dir / "results" / "parity.json", parity)
    write_json(run_dir / "results" / "speed.json", [r.to_dict() for r in speed_rows])
    write_json(run_dir / "results" / "batch.json", [r.to_dict() for r in batch_rows])
    plots = write_plots(run_dir / "plots", speed_rows, batch_rows)
    write_report(
        run_dir / "report.md",
        env=env,
        parity=parity,
        speed_rows=speed_rows,
        batch_rows=batch_rows,
        plot_paths=plots,
        iterate_notes=iterate_notes,
        vllm_engine_note=vllm_engine_note,
        tp_size=tensor_parallel_size,
    )
    summary = {
        "run_dir": str(run_dir),
        "parity_ok": bool(parity.get("ok")),
        "hf_native_decode": hf_native_decode,
        "vllm_decode": _medium_decode(speed_rows, "vLLM-xLSTM"),
        "gate": parity.get("gate"),
        "vllm_engine_note": vllm_engine_note,
    }
    write_json(run_dir / "results" / "summary.json", summary)
    print(f"[bench] report: {run_dir / 'report.md'}")
    return summary
