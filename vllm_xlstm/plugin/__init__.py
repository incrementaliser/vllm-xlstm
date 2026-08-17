"""vLLM general plugin: register ``xLSTMForCausalLM``."""

from __future__ import annotations


def register() -> None:
    """
    Register the out-of-tree xLSTM causal LM with vLLM when vLLM is installed.

    Uses a lazy import path so worker processes do not initialize CUDA at
    import time (fork-safe).
    """
    try:
        from vllm import ModelRegistry
    except ImportError:
        return
    arch = "xLSTMForCausalLM"
    try:
        supported = ModelRegistry.get_supported_archs()
    except Exception:
        supported = ()
    if arch in supported:
        return
    ModelRegistry.register_model(arch, "vllm_xlstm.plugin.model:XlstmForCausalLM")
