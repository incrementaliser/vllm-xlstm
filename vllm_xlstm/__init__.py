"""Out-of-tree vLLM xLSTM plugin and HF vs CUDA-graph speed benches.

Install registers ``xLSTMForCausalLM`` via ``vllm.general_plugins``. The
measured speed path is ``GraphDecodeEngine`` (Hugging Face weights +
``mlstm_kernels`` + PyTorch CUDA graphs). Full ``vllm.LLM`` / ``vllm serve``
is not production-ready yet.

Reproduce with::

    uv run xlstm-vllm-speed
    uv run xlstm-vllm-speed-tp2
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
