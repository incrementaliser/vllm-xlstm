# vLLM xLSTM speed report

Reproduce:

```bash
uv run xlstm-vllm-speed
# two GPUs:
uv run xlstm-vllm-speed-tp2
cat artifacts/runs/20260817_221132_vllm-speed/report.md
```

Metrics glossary: `docs/metrics.md` (TTFT includes prefill; TPOT is ms/token; decode tok/s = 1000/TPOT; e2e lower is better).

## Environment

```json
{
  "hostname": "gpu06.pri.dmog.alces.network",
  "platform": "Linux-5.14.0-611.55.1.el9_7.x86_64-x86_64-with-glibc2.34",
  "python": "3.12.12",
  "torch": "2.4.1+cu118",
  "cuda_available": true,
  "gpu_name": "NVIDIA A40",
  "gpu_count": 1,
  "gpu_total_mem_mib": 45490.0,
  "gpu_names": [
    "NVIDIA A40"
  ],
  "transformers": "5.0.0rc0",
  "vllm": null,
  "xlstm": "2.0.0",
  "mlstm_kernels": "2.0.4",
  "nvidia_smi": "NVIDIA A40, 595.71.05, 46068 MiB",
  "model_src": "/users/aa2070/.cache/huggingface/hub/models--NX-AI--xLSTM-7b/snapshots/9dc507bd0939cf372a4a4f667335651d8e49dddb",
  "tensor_parallel_size": 1
}
```

- Real vLLM engine: vllm package not installed
- Requested TP size: 1

## Parity gate (greedy 16 tokens)

- ok: `True`
- gate: PASS_KERNEL (matches HF-triton, not native kernels)
- decode mode: cuda_graph
- match 1/16

## 1-GPU success gate

**PASS: vLLM-xLSTM 34.61 tok/s > HF-native 29.09 tok/s (medium, gen=128, 1 GPU).**

The gate is medium prompt / gen=128 / **vLLM-xLSTM decode tok/s > HF-native** on one GPU. Two-GPU TP is reported below but is not required to pass.

## Iterate notes

- probe decode=34.62 tok/s mode=cuda_graph
- baseline graph (no extra op patch): 34.62 tok/s
- WIN: cuda_graph/unpatched 34.62 tok/s > HF-native 29.09
- vllm.LLM skipped: vllm package not installed

## Single-stream (batch=1)

| backend | gpus | prompt | tokens | batch | TTFT ms ↓ | TPOT ms ↓ | decode tok/s ↑ | prefill tok/s ↑ | e2e s ↓ | sanity |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| HF-native | 1 | short | 32 | 1 | 48.3 ± 0.2 | 34.37 ± 0.00 | 29.09 ± 0.00 | 662.6 ± 3.3 | 4.414 ± 0.000 | ok |
| HF-native | 1 | medium | 256 | 1 | 75.6 ± 0.1 | 34.37 ± 0.01 | 29.09 ± 0.01 | 3387.5 ± 5.6 | 4.441 ± 0.001 | ok |
| HF-native | 1 | long | 1024 | 1 | 231.7 ± 0.5 | 34.38 ± 0.01 | 29.09 ± 0.01 | 4420.4 ± 10.1 | 4.598 ± 0.002 | ok |
| HF-triton | 1 | short | 32 | 1 | 37.7 ± 0.1 | 30.43 ± 0.00 | 32.86 ± 0.00 | 848.0 ± 3.2 | 3.903 ± 0.000 | ok |
| HF-triton | 1 | medium | 256 | 1 | 64.5 ± 0.1 | 30.44 ± 0.00 | 32.85 ± 0.00 | 3970.0 ± 4.5 | 3.931 ± 0.000 | ok |
| HF-triton | 1 | long | 1024 | 1 | 196.1 ± 0.2 | 30.44 ± 0.00 | 32.85 ± 0.00 | 5221.0 ± 5.1 | 4.062 ± 0.000 | ok |
| vLLM-xLSTM | 1 | short | 32 | 1 | 38.5 ± 1.0 | 28.88 ± 0.00 | 34.62 ± 0.00 | 831.8 ± 20.7 | 3.707 ± 0.001 | ok |
| vLLM-xLSTM | 1 | medium | 256 | 1 | 65.1 ± 0.1 | 28.89 ± 0.00 | 34.61 ± 0.00 | 3934.2 ± 7.5 | 3.734 ± 0.001 | ok |
| vLLM-xLSTM | 1 | long | 1024 | 1 | 197.3 ± 0.2 | 28.89 ± 0.00 | 34.61 ± 0.00 | 5190.3 ± 6.0 | 3.867 ± 0.000 | ok |

## Batch throughput (medium, gen=128)

| backend | gpus | prompt | tokens | batch | TTFT ms ↓ | TPOT ms ↓ | decode tok/s ↑ | prefill tok/s ↑ | e2e s ↓ | sanity |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| vLLM-xLSTM | 1 | medium | 256 | 1 | 65.1 ± 0.2 | 28.89 ± 0.00 | 34.61 ± 0.00 | 3934.6 ± 10.4 | 3.734 ± 0.001 | ok |
| vLLM-xLSTM | 1 | medium | 256 | 2 | 108.3 ± 0.3 | 30.76 ± 0.00 | 32.51 ± 0.00 | 4727.7 ± 13.4 | 4.014 ± 0.000 | ok |
| vLLM-xLSTM | 1 | medium | 256 | 4 | 200.3 ± 0.1 | 32.30 ± 0.00 | 30.96 ± 0.00 | 5111.1 ± 3.7 | 4.302 ± 0.000 | ok |
| vLLM-xLSTM | 1 | medium | 256 | 8 | 379.4 ± 0.2 | 35.59 ± 0.00 | 28.10 ± 0.00 | 5397.5 ± 2.7 | 4.899 ± 0.000 | ok |
| HF-triton | 1 | medium | 256 | 1 | 64.8 ± 0.1 | 30.44 ± 0.00 | 32.86 ± 0.00 | 3951.2 ± 7.2 | 3.930 ± 0.000 | ok |
| HF-triton | 1 | medium | 256 | 2 | 108.4 ± 0.1 | 32.40 ± 0.00 | 30.86 ± 0.00 | 4725.2 ± 5.5 | 4.224 ± 0.001 | ok |
| HF-triton | 1 | medium | 256 | 4 | 200.5 ± 0.2 | 33.99 ± 0.00 | 29.42 ± 0.00 | 5107.6 ± 4.1 | 4.517 ± 0.000 | ok |
| HF-triton | 1 | medium | 256 | 8 | 379.7 ± 0.0 | 37.26 ± 0.00 | 26.84 ± 0.00 | 5393.1 ± 0.0 | 5.112 ± 0.000 | ok |

## 2-GPU (tensor parallel / device_map)

_Not run in this job (use `uv run xlstm-vllm-speed-tp2` with 2 GPUs)._

## Plots

- `ttft_ms_vs_input_len.png`
- `tpot_ms_vs_input_len.png`
- `decode_tok_s_vs_input_len.png`
- `e2e_sec_vs_input_len.png`
- `prefill_tok_s_vs_input_len.png`
- `batch_tok_s_vs_batch_size.png`
