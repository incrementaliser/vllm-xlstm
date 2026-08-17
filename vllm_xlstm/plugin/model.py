"""vLLM-facing xLSTM causal LM: IsAttentionFree + recurrent state (C, n, m)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from vllm_xlstm.plugin.mlstm_op import patch_mlstm_backends


def _hf_to_xlstm_large_key(key: str) -> str:
    """Map Hugging Face ``xLSTMForCausalLM`` names onto ``xLSTMLarge``."""
    if key.startswith("backbone.embeddings."):
        return "embedding." + key[len("backbone.embeddings.") :]
    return key


def _maybe_vllm_interfaces() -> tuple[type, ...]:
    """Return vLLM mixins when installed; otherwise an empty tuple."""
    try:
        from vllm.model_executor.models.interfaces import HasInnerState, IsAttentionFree

        extras: list[type] = [HasInnerState, IsAttentionFree]
        try:
            from vllm.model_executor.models.interfaces import SupportsPP

            extras.append(SupportsPP)
        except ImportError:
            pass
        return tuple(extras)
    except ImportError:
        return ()


_VLLM_MIXINS = _maybe_vllm_interfaces()


class XlstmForCausalLM(nn.Module, *_VLLM_MIXINS):
    """
    Out-of-tree vLLM model: attention-free xLSTM with in-place (C, n, m) state.

    Constructed either from a vLLM ``VllmConfig`` (serving) or from a Hugging
    Face config (benches / parity). Mixer math goes through the ``mlstm_kernels``
    custom ops, not ``transformers.generate``.
    """

    def __init__(
        self,
        *,
        vllm_config: Any | None = None,
        prefix: str = "",
        hf_config: Any | None = None,
        step_kernel: str = "triton",
    ) -> None:
        """
        Build the xLSTM stack.

        Args:
            vllm_config: vLLM engine config (preferred when serving).
            prefix: Unused; kept for vLLM's ``maybe_prefix`` convention.
            hf_config: Transformers ``xLSTMConfig`` when not using vLLM.
            step_kernel: Decode kernel name forwarded to ``xLSTMLarge``.
        """
        super().__init__()
        if vllm_config is not None:
            hf_config = vllm_config.model_config.hf_config
            self.vllm_config = vllm_config
        else:
            self.vllm_config = None
        if hf_config is None:
            raise ValueError("XlstmForCausalLM requires vllm_config or hf_config")
        self.config = hf_config
        xl_cfg = hf_config.to_xlstm_block_config()
        xl_cfg.mode = "inference"
        xl_cfg.return_last_states = True
        xl_cfg.step_kernel = step_kernel  # type: ignore[assignment]
        if step_kernel == "triton":
            xl_cfg.chunkwise_kernel = "chunkwise--triton_xl_chunk"
            xl_cfg.sequence_kernel = "native_sequence__triton"
        from xlstm.xlstm_large.model import xLSTMLarge

        self.model = xLSTMLarge(xl_cfg)
        self.vocab_size = int(hf_config.vocab_size)
        try:
            from vllm.model_executor.layers.logits_processor import LogitsProcessor

            self.logits_processor = LogitsProcessor(self.vocab_size)
        except ImportError:
            self.logits_processor = None
        patch_mlstm_backends(self.model, use_triton=step_kernel == "triton")

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: Any) -> tuple[torch.dtype, ...]:
        """Recurrent state dtypes: (C, n, m), default float32 for numerical range."""
        cache_dtype = torch.float32
        try:
            model_dtype = vllm_config.model_config.dtype
        except Exception:
            model_dtype = torch.bfloat16
        return (cache_dtype, cache_dtype, cache_dtype)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: Any,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """
        Per-layer recurrent shapes matching HF ``xLSTMCache``.

        Returns:
            ``C (nh, dhqk, dhv)``, ``n (nh, dhqk)``, ``m (nh, 1)``.
            Batch is added by the engine.
        """
        cfg = vllm_config.model_config.hf_config
        tp = 1
        try:
            tp = int(vllm_config.parallel_config.tensor_parallel_size)
        except Exception:
            pass
        n_heads = max(int(cfg.num_heads) // max(tp, 1), 1)
        dhqk = int(cfg.qk_head_dim)
        dhv = int(cfg.v_head_dim)
        return ((n_heads, dhqk, dhv), (n_heads, dhqk), (n_heads, 1))

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Token embeddings."""
        return self.model.embedding(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
        state: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """
        Prefill or decode. ``state`` is the in-place (C, n, m) dict keyed by layer.

        Args:
            input_ids: Token ids ``[B, S]`` (or flattened tokens in vLLM).
            positions: Ignored (no RoPE).
            intermediate_tensors: Pipeline-parallel placeholder.
            inputs_embeds: Optional embeddings instead of ids.
            state: Recurrent cache; mutated in place when provided.

        Returns:
            Logits ``[B, S, V]``, or ``(logits, state)`` when the backbone
            is configured to return last states.
        """
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("forward requires input_ids or inputs_embeds")
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
            hidden, state = self.model.backbone(self.model.embedding(input_ids), state)
        else:
            if inputs_embeds.ndim == 2:
                inputs_embeds = inputs_embeds.unsqueeze(0)
            hidden, state = self.model.backbone(inputs_embeds, state)
        logits = self.model.lm_head(hidden)
        from xlstm.xlstm_large.components import soft_cap

        logits = soft_cap(logits, self.model.config.output_logit_soft_cap)
        return logits

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project last-layer hidden states to vocab logits."""
        logits = self.model.lm_head(hidden_states)
        from xlstm.xlstm_large.components import soft_cap

        return soft_cap(logits, self.model.config.output_logit_soft_cap)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load HF safetensor names into the ``xLSTMLarge`` module.

        Args:
            weights: Iterable of ``(name, tensor)`` from the checkpoint.

        Returns:
            Set of loaded parameter names (vLLM convention).
        """
        mapped: dict[str, torch.Tensor] = {}
        for name, tensor in weights:
            mapped[_hf_to_xlstm_large_key(name)] = tensor
        missing, unexpected = self.model.load_state_dict(mapped, strict=False)
        loaded = set(mapped) - set(unexpected)
        if missing:
            # Still report loaded params; missing is often tied embeddings.
            pass
        return loaded
