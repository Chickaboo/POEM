"""Causal self-attention layers with RoPE or absolute-position inputs."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor, base: float = 10_000.0) -> torch.Tensor:
    head_dim = x.size(-1)
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")
    seq_len = x.size(-2)
    device = x.device
    dtype = x.dtype
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.repeat_interleave(freqs, 2, dim=-1).to(dtype=dtype)
    cos = emb.cos()[None, None, :, :]
    sin = emb.sin()[None, None, :, :]
    return (x * cos) + (_rotate_half(x) * sin)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        dropout: float = 0.0,
        use_rope: bool = True,
        rope_base: float = 10_000.0,
    ) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"d_model={dim} must be divisible by n_heads={n_heads}")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.dropout = float(dropout)
        self.use_rope = use_rope
        self.rope_base = rope_base
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.use_rope:
            q = apply_rope(q, self.rope_base)
            k = apply_rope(k, self.rope_base)
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.out(attn)


class RoPECausalSelfAttention(CausalSelfAttention):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0, rope_base: float = 10_000.0) -> None:
        super().__init__(dim, n_heads, dropout=dropout, use_rope=True, rope_base=rope_base)


class AbsoluteCausalSelfAttention(CausalSelfAttention):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__(dim, n_heads, dropout=dropout, use_rope=False)
