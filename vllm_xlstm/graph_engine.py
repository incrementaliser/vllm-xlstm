"""CUDA-graph decode engine: HF xLSTM weights + mLSTM custom ops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vllm_xlstm.hf_engine import greedy_prefill_decode, new_cache
from vllm_xlstm.metrics import RawTiming
from vllm_xlstm.plugin.mlstm_op import patch_mlstm_backends


@dataclass
class GraphSession:
    """Captured decode graph and the static buffers it reads/writes."""

    graph: torch.cuda.CUDAGraph
    token_buf: torch.Tensor
    cache: Any
    decode_mode: str


class GraphDecodeEngine:
    """
    Serving-style greedy decode: chunkwise prefill, CUDA-graph replay for S=1.

    This is the 1-GPU speed path used when vLLM's own engine cannot load
    xLSTM yet (or as the plugin's decode loop). It is not ``model.generate()``.
    """

    def __init__(
        self,
        model: Any,
        *,
        use_triton: bool = True,
        patch_ops: bool = True,
    ) -> None:
        """
        Args:
            model: Loaded ``xLSTMForCausalLM`` (triton kernels recommended).
            use_triton: Patch backends to the Triton step custom op.
            patch_ops: If False, leave ``mLSTMBackend`` implementations unchanged.
        """
        self.model = model.eval()
        self.n_patched = (
            patch_mlstm_backends(model, use_triton=use_triton) if patch_ops else 0
        )
        self._sessions: dict[int, GraphSession | None] = {}
        self.decode_mode = "eager"

    def prepare(self, batch_size: int) -> str:
        """Capture graphs (or decide eager) before timed runs.

        Args:
            batch_size: Static batch for the CUDA graph.

        Returns:
            ``cuda_graph`` or ``eager``.
        """
        session = self._session(batch_size)
        self.decode_mode = "cuda_graph" if session is not None else "eager"
        return self.decode_mode

    def _eager_step(self, token: torch.Tensor, cache: Any) -> torch.Tensor:
        """One greedy decode step without a CUDA graph."""
        out = self.model(input_ids=token, cache_params=cache, use_cache=True)
        return out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    def _capture(self, batch_size: int) -> GraphSession | None:
        """
        Capture a CUDA graph for a fixed batch of single-token steps.

        Returns:
            Session or ``None`` when capture is not possible (CPU / multi-device).
        """
        if not torch.cuda.is_available():
            return None
        devices = {p.device for p in self.model.parameters()}
        if len(devices) != 1 or next(iter(devices)).type != "cuda":
            return None
        cache = new_cache(self.model, batch_size)
        token_buf = torch.zeros(batch_size, 1, dtype=torch.long, device=token_device(self.model))
        with torch.inference_mode():
            for _ in range(3):
                self._eager_step(token_buf, cache)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    nxt = self._eager_step(token_buf, cache)
                    token_buf.copy_(nxt)
            except Exception as exc:  # noqa: BLE001
                print(f"[vllm-xlstm] CUDA graph capture failed (batch={batch_size}): {exc}")
                return None
            torch.cuda.synchronize()
        return GraphSession(graph=graph, token_buf=token_buf, cache=cache, decode_mode="cuda_graph")

    def _session(self, batch_size: int) -> GraphSession | None:
        """Return a captured graph for ``batch_size``, capturing on first use."""
        if batch_size not in self._sessions:
            self._sessions[batch_size] = self._capture(batch_size)
        return self._sessions[batch_size]

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        n_gen: int,
    ) -> tuple[torch.Tensor, RawTiming, str]:
        """
        Greedy generate with industry TTFT.

        Args:
            input_ids: Prompt ids ``[B, S]``.
            n_gen: New tokens.

        Returns:
            Generated ids, timing, and decode mode (``cuda_graph`` or ``eager``).
        """
        batch = int(input_ids.shape[0])
        session = self._session(batch)
        if session is None:
            tokens, timing = greedy_prefill_decode(
                self.model, input_ids, n_gen=n_gen, decode_step=self._eager_step
            )
            return tokens, timing, "eager"

        def _graph_step(token: torch.Tensor, cache: Any) -> torch.Tensor:
            """Replay the captured graph; ``cache`` is the static session cache."""
            session.token_buf.copy_(token)
            session.graph.replay()
            return session.token_buf

        tokens, timing = greedy_prefill_decode(
            self.model,
            input_ids,
            n_gen=n_gen,
            cache=session.cache,
            decode_step=_graph_step,
        )
        return tokens, timing, "cuda_graph"


def token_device(model: Any) -> torch.device:
    """Device of the first parameter (embedding / lm_head)."""
    return next(model.parameters()).device


def try_compile_step(engine: GraphDecodeEngine) -> bool:
    """
    Optionally ``torch.compile`` the eager decode step when graphs are unavailable.

    Args:
        engine: Engine whose ``_eager_step`` may be replaced.

    Returns:
        True if compile succeeded.
    """
    if not torch.cuda.is_available():
        return False
    try:
        compiled = torch.compile(engine._eager_step, mode="reduce-overhead", fullgraph=False)
        engine._eager_step = compiled  # type: ignore[method-assign]
        engine.decode_mode = "torch_compile"
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[vllm-xlstm] torch.compile skipped: {exc}")
        return False
