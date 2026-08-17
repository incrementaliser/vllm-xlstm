"""mLSTM mixer layer: projections + custom-op step/chunkwise (C, n, m)."""

from __future__ import annotations

import torch
from torch import nn

from vllm_xlstm.plugin.mlstm_op import mlstm_chunkwise, mlstm_step


def _soft_cap(x: torch.Tensor, cap: float) -> torch.Tensor:
    """Apply the NX-AI soft cap used on xLSTM gates."""
    if cap <= 0:
        return x
    return cap * torch.tanh(x / cap)


class MlstmMixer(nn.Module):
    """
    Attention-free mLSTM mixer with in-place recurrent state ``(C, n, m)``.

    Linear maps stay in PyTorch; the mixer core is the CUDA-graph-friendly
    custom op wrapping ``mlstm_kernels``.
    """

    def __init__(
        self,
        *,
        embedding_dim: int,
        num_heads: int,
        qk_dim: int,
        v_dim: int,
        gate_soft_cap: float = 15.0,
        eps: float = 1e-6,
        chunk_size: int = 64,
        use_bias: bool = False,
        use_triton: bool = True,
    ) -> None:
        """
        Args:
            embedding_dim: Residual stream width.
            num_heads: mLSTM heads.
            qk_dim: Total query/key dim (split across heads).
            v_dim: Total value dim.
            gate_soft_cap: Soft cap on i/f pre-activations.
            eps: Kernel epsilon.
            chunk_size: Prefill chunk length.
            use_bias: Linear bias (7B uses False except gates).
            use_triton: Prefer Triton step kernel.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.gate_soft_cap = gate_soft_cap
        self.eps = eps
        self.chunk_size = chunk_size
        self.use_triton = use_triton
        self.q = nn.Linear(embedding_dim, qk_dim, bias=use_bias)
        self.k = nn.Linear(embedding_dim, qk_dim, bias=use_bias)
        self.v = nn.Linear(embedding_dim, v_dim, bias=use_bias)
        self.ogate_preact = nn.Linear(embedding_dim, v_dim, bias=use_bias)
        self.igate_preact = nn.Linear(embedding_dim, num_heads, bias=True)
        self.fgate_preact = nn.Linear(embedding_dim, num_heads, bias=True)
        self.out_proj = nn.Linear(v_dim, embedding_dim, bias=use_bias)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Hidden states ``[B, S, D]``.
            state: Optional ``(C, n, m)``; allocated at zeros when omitted.

        Returns:
            Residual-stream update and updated ``(C, n, m)``.
        """
        bsz, seqlen, _ = x.shape
        q = self.q(x).reshape(bsz, seqlen, self.num_heads, -1).transpose(1, 2)
        k = self.k(x).reshape(bsz, seqlen, self.num_heads, -1).transpose(1, 2)
        v = self.v(x).reshape(bsz, seqlen, self.num_heads, -1).transpose(1, 2)
        i = _soft_cap(self.igate_preact(x), self.gate_soft_cap).transpose(1, 2)
        f = _soft_cap(self.fgate_preact(x), self.gate_soft_cap).transpose(1, 2)
        o = torch.sigmoid(self.ogate_preact(x))
        dhqk = q.shape[-1]
        dhv = v.shape[-1]
        if state is None:
            c_state = torch.zeros(bsz, self.num_heads, dhqk, dhv, device=x.device, dtype=torch.float32)
            n_state = torch.zeros(bsz, self.num_heads, dhqk, device=x.device, dtype=torch.float32)
            m_state = torch.zeros(bsz, self.num_heads, 1, device=x.device, dtype=torch.float32)
        else:
            c_state, n_state, m_state = state
        if seqlen == 1:
            h = mlstm_step(
                q.squeeze(2),
                k.squeeze(2),
                v.squeeze(2),
                i if i.ndim == 3 else i.unsqueeze(-1),
                f if f.ndim == 3 else f.unsqueeze(-1),
                c_state,
                n_state,
                m_state,
                eps=self.eps,
                use_triton=self.use_triton,
            )
            h = h.transpose(1, 2).reshape(bsz, seqlen, -1)
        else:
            h = mlstm_chunkwise(
                q,
                k,
                v,
                i,
                f,
                c_state,
                n_state,
                m_state,
                eps=self.eps,
                chunk_size=self.chunk_size,
            )
            h = h.transpose(1, 2).reshape(bsz, seqlen, -1)
        y = self.out_proj(o * h)
        return y, (c_state, n_state, m_state)

    def extra_repr(self) -> str:
        """Summarize mixer dims for debugging."""
        return (
            f"d={self.embedding_dim}, heads={self.num_heads}, "
            f"qk={self.qk_dim}, v={self.v_dim}, triton={self.use_triton}"
        )
