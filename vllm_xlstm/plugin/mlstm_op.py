"""CUDA-graph-compatible custom ops wrapping NX-AI ``mlstm_kernels``.

Prefill uses the chunkwise kernel; decode (S=1) uses the recurrent step kernel.
Both are registered as PyTorch custom ops so CUDA graphs treat them as opaque
nodes instead of tracing through Triton internals.
"""

from __future__ import annotations

from typing import Any

import torch

_OPS_READY = False


def _step_kernel(use_triton: bool) -> Any:
    """Return the NX-AI mLSTM step kernel (triton or native)."""
    from mlstm_kernels.torch import get_mlstm_step_kernel

    name = "triton" if use_triton else "native"
    try:
        return get_mlstm_step_kernel(name)
    except ValueError:
        return get_mlstm_step_kernel("native")


def _chunkwise_kernel() -> Any:
    """Return the chunkwise TFLA kernel, falling back to native autograd."""
    from mlstm_kernels.torch import get_mlstm_kernel

    for name in ("chunkwise--triton_xl_chunk", "chunkwise--triton_limit_chunk"):
        try:
            return get_mlstm_kernel(name)
        except ValueError:
            continue
    return get_mlstm_kernel("chunkwise--native_autograd")


def _mlstm_step_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor,
    f: torch.Tensor,
    c_state: torch.Tensor,
    n_state: torch.Tensor,
    m_state: torch.Tensor,
    eps: float,
    use_triton: bool,
) -> torch.Tensor:
    """Run one mLSTM step and copy new (C, n, m) into the provided state tensors."""
    kernel = _step_kernel(use_triton)
    h, (c_new, n_new, m_new) = kernel(
        q=q,
        k=k,
        v=v,
        i=i,
        f=f,
        c=c_state,
        n=n_state,
        m=m_state,
        eps=eps,
        dtype_state=c_state.dtype,
    )
    c_state.copy_(c_new)
    n_state.copy_(n_new)
    m_state.copy_(m_new)
    return h


def _mlstm_step_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor,
    f: torch.Tensor,
    c_state: torch.Tensor,
    n_state: torch.Tensor,
    m_state: torch.Tensor,
    eps: float,
    use_triton: bool,
) -> torch.Tensor:
    """Meta/fake implementation for tracing and CUDA-graph capture."""
    return torch.empty_like(v)


def _mlstm_chunkwise_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor,
    f: torch.Tensor,
    c_state: torch.Tensor,
    n_state: torch.Tensor,
    m_state: torch.Tensor,
    eps: float,
    chunk_size: int,
) -> torch.Tensor:
    """Run chunkwise mLSTM over a sequence and update (C, n, m) in place."""
    kernel = _chunkwise_kernel()
    h, (c_new, n_new, m_new) = kernel(
        q=q,
        k=k,
        v=v,
        i=i,
        f=f,
        c_initial=c_state,
        n_initial=n_state,
        m_initial=m_state,
        chunk_size=chunk_size,
        return_last_states=True,
        eps=eps,
        autocast_kernel_dtype=q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16,
    )
    c_state.copy_(c_new)
    n_state.copy_(n_new)
    m_state.copy_(m_new)
    seq = q.shape[2]
    return h[:, :, :seq, :]


def _mlstm_chunkwise_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor,
    f: torch.Tensor,
    c_state: torch.Tensor,
    n_state: torch.Tensor,
    m_state: torch.Tensor,
    eps: float,
    chunk_size: int,
) -> torch.Tensor:
    """Meta/fake implementation returning hidden states of shape (B, NH, S, DHV)."""
    return torch.empty_like(v)


def ensure_custom_ops() -> None:
    """Register ``xlstm_vllm::mlstm_step`` and ``xlstm_vllm::mlstm_chunkwise`` once."""
    global _OPS_READY
    if _OPS_READY:
        return
    try:
        torch.library.define(
            "xlstm_vllm::mlstm_step",
            "(Tensor q, Tensor k, Tensor v, Tensor i, Tensor f, "
            "Tensor(a!) c_state, Tensor(b!) n_state, Tensor(c!) m_state, "
            "float eps, bool use_triton) -> Tensor",
        )
        torch.library.impl("xlstm_vllm::mlstm_step", "CompositeExplicitAutograd", _mlstm_step_impl)
        torch.library.register_fake("xlstm_vllm::mlstm_step", _mlstm_step_fake)
    except RuntimeError as exc:
        if "already" not in str(exc).lower():
            # Fallback: torch.library.custom_op decorator path (PyTorch 2.4+).
            _register_via_custom_op()
            _OPS_READY = True
            return
    try:
        torch.library.define(
            "xlstm_vllm::mlstm_chunkwise",
            "(Tensor q, Tensor k, Tensor v, Tensor i, Tensor f, "
            "Tensor(a!) c_state, Tensor(b!) n_state, Tensor(c!) m_state, "
            "float eps, int chunk_size) -> Tensor",
        )
        torch.library.impl(
            "xlstm_vllm::mlstm_chunkwise",
            "CompositeExplicitAutograd",
            _mlstm_chunkwise_impl,
        )
        torch.library.register_fake("xlstm_vllm::mlstm_chunkwise", _mlstm_chunkwise_fake)
    except RuntimeError as exc:
        if "already" not in str(exc).lower():
            _register_via_custom_op()
    _OPS_READY = True


def _register_via_custom_op() -> None:
    """Register ops with ``torch.library.custom_op`` when ``define`` is unavailable."""

    @torch.library.custom_op("xlstm_vllm::mlstm_step", mutates_args=("c_state", "n_state", "m_state"))
    def mlstm_step(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        i: torch.Tensor,
        f: torch.Tensor,
        c_state: torch.Tensor,
        n_state: torch.Tensor,
        m_state: torch.Tensor,
        eps: float,
        use_triton: bool,
    ) -> torch.Tensor:
        """Decode-step custom op wrapping ``mlstm_kernels``."""
        return _mlstm_step_impl(q, k, v, i, f, c_state, n_state, m_state, eps, use_triton)

    @mlstm_step.register_fake
    def _(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        i: torch.Tensor,
        f: torch.Tensor,
        c_state: torch.Tensor,
        n_state: torch.Tensor,
        m_state: torch.Tensor,
        eps: float,
        use_triton: bool,
    ) -> torch.Tensor:
        return _mlstm_step_fake(q, k, v, i, f, c_state, n_state, m_state, eps, use_triton)

    @torch.library.custom_op(
        "xlstm_vllm::mlstm_chunkwise",
        mutates_args=("c_state", "n_state", "m_state"),
    )
    def mlstm_chunkwise(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        i: torch.Tensor,
        f: torch.Tensor,
        c_state: torch.Tensor,
        n_state: torch.Tensor,
        m_state: torch.Tensor,
        eps: float,
        chunk_size: int,
    ) -> torch.Tensor:
        """Prefill custom op wrapping chunkwise ``mlstm_kernels``."""
        return _mlstm_chunkwise_impl(
            q, k, v, i, f, c_state, n_state, m_state, eps, chunk_size
        )

    @mlstm_chunkwise.register_fake
    def _(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        i: torch.Tensor,
        f: torch.Tensor,
        c_state: torch.Tensor,
        n_state: torch.Tensor,
        m_state: torch.Tensor,
        eps: float,
        chunk_size: int,
    ) -> torch.Tensor:
        return _mlstm_chunkwise_fake(
            q, k, v, i, f, c_state, n_state, m_state, eps, chunk_size
        )


def mlstm_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor,
    f: torch.Tensor,
    c_state: torch.Tensor,
    n_state: torch.Tensor,
    m_state: torch.Tensor,
    *,
    eps: float = 1e-6,
    use_triton: bool = True,
) -> torch.Tensor:
    """
    Recurrent mLSTM step (decode). Mutates ``c_state``, ``n_state``, ``m_state``.

    Args:
        q, k, v: Query/key/value of shape ``(B, NH, DH)``.
        i, f: Input/forget gate pre-activations ``(B, NH, 1)``.
        c_state: Cell state ``(B, NH, DHQK, DHV)``.
        n_state: Normalizer ``(B, NH, DHQK)``.
        m_state: Max state ``(B, NH, 1)``.
        eps: Numerical stability epsilon.
        use_triton: Prefer the Triton step kernel.

    Returns:
        Hidden state ``(B, NH, DHV)``.
    """
    ensure_custom_ops()
    return torch.ops.xlstm_vllm.mlstm_step(
        q, k, v, i, f, c_state, n_state, m_state, eps, use_triton
    )


def mlstm_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor,
    f: torch.Tensor,
    c_state: torch.Tensor,
    n_state: torch.Tensor,
    m_state: torch.Tensor,
    *,
    eps: float = 1e-6,
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    Chunkwise mLSTM (prefill). Mutates ``c_state``, ``n_state``, ``m_state``.

    Args:
        q, k, v: Sequence tensors ``(B, NH, S, DH)``.
        i, f: Gate pre-activations ``(B, NH, S)``.
        c_state, n_state, m_state: Recurrent state buffers.
        eps: Numerical stability epsilon.
        chunk_size: TFLA chunk length.

    Returns:
        Hidden states ``(B, NH, S, DHV)``.
    """
    ensure_custom_ops()
    return torch.ops.xlstm_vllm.mlstm_chunkwise(
        q, k, v, i, f, c_state, n_state, m_state, eps, chunk_size
    )


def patch_mlstm_backends(model: torch.nn.Module, *, use_triton: bool = True) -> int:
    """
    Route every ``mLSTMBackend.forward`` through the custom ops.

    Args:
        model: Module tree that contains ``mLSTMBackend`` instances.
        use_triton: Decode-step kernel preference.

    Returns:
        Number of backends patched.
    """
    from types import MethodType

    from mlstm_kernels.torch.backend_module import mLSTMBackend

    ensure_custom_ops()
    patched = 0

    def _forward(
        self: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        i: torch.Tensor,
        f: torch.Tensor,
        c_initial: torch.Tensor | None = None,
        n_initial: torch.Tensor | None = None,
        m_initial: torch.Tensor | None = None,
        return_last_states: bool | None = None,
        mode: str | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Custom-op mLSTM forward (prefill vs decode)."""
        bsz, n_heads, seqlen, dhqk = q.shape
        dhv = v.shape[-1]
        device = q.device
        dtype_state = (
            c_initial.dtype
            if c_initial is not None
            else torch.float32
        )
        if c_initial is None:
            c_initial = torch.zeros(bsz, n_heads, dhqk, dhv, device=device, dtype=dtype_state)
        if n_initial is None:
            n_initial = torch.zeros(bsz, n_heads, dhqk, device=device, dtype=dtype_state)
        if m_initial is None:
            m_initial = torch.zeros(bsz, n_heads, 1, device=device, dtype=dtype_state)
        eps = float(getattr(self.config, "eps", 1e-6))
        chunk_size = int(getattr(self.config, "chunk_size", 64))
        if seqlen == 1:
            i_step = i if i.ndim == 3 else i.unsqueeze(-1)
            f_step = f if f.ndim == 3 else f.unsqueeze(-1)
            if i_step.shape[-1] != 1:
                i_step = i_step[..., :1]
            if f_step.shape[-1] != 1:
                f_step = f_step[..., :1]
            h = mlstm_step(
                q.squeeze(2),
                k.squeeze(2),
                v.squeeze(2),
                i_step,
                f_step,
                c_initial,
                n_initial,
                m_initial,
                eps=eps,
                use_triton=use_triton,
            )
            hidden = h.unsqueeze(2)
        else:
            i_seq = i.squeeze(-1) if i.ndim == 4 else i
            f_seq = f.squeeze(-1) if f.ndim == 4 else f
            hidden = mlstm_chunkwise(
                q,
                k,
                v,
                i_seq,
                f_seq,
                c_initial,
                n_initial,
                m_initial,
                eps=eps,
                chunk_size=chunk_size,
            )
        return hidden, (c_initial, n_initial, m_initial)

    for module in model.modules():
        if isinstance(module, mLSTMBackend):
            module.forward = MethodType(_forward, module)
            patched += 1
    return patched
