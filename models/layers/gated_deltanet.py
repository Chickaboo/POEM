"""Sequential Gated DeltaNet layer.

This implements the core gated delta rule from Yang et al., "Gated Delta
Networks: Improving Mamba2 with Delta Rule" (ICLR 2025) as an explicit loop.
The update follows Eq. 10 in the paper:

    S_t = alpha_t * S_{t-1} * (I - beta_t k_t k_t^T) + beta_t v_t k_t^T

with L2-normalized q/k vectors and a post-recurrence output gate.  The official
GPU implementation adds optimized chunk kernels and short convolutions; POEM
keeps this reference layer plain CPU PyTorch for correctness and debugging.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models.layers.norm import RMSNorm


def sequential_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the gated delta recurrence.

    Args:
        q, k, v: tensors shaped [batch, heads, seq, head_dim].
        alpha, beta: tensors shaped [batch, heads, seq] with values in [0, 1].

    Returns:
        outputs shaped [batch, heads, seq, head_dim] and the final recurrent
        state shaped [batch, heads, head_dim, head_dim].
    """

    batch, heads, seq_len, head_dim = q.shape
    state = q.new_zeros(batch, heads, head_dim, head_dim)
    outputs: list[torch.Tensor] = []
    for step in range(seq_len):
        q_t = q[:, :, step]
        k_t = k[:, :, step]
        v_t = v[:, :, step]
        alpha_t = alpha[:, :, step].clamp(0.0, 1.0)
        beta_t = beta[:, :, step].clamp(0.0, 1.0)
        old_value = torch.einsum("bhvk,bhk->bhv", state, k_t)
        erase = torch.einsum("bhv,bhk->bhvk", old_value, k_t)
        write = torch.einsum("bhv,bhk->bhvk", v_t, k_t)
        state = alpha_t[..., None, None] * (state - beta_t[..., None, None] * erase)
        state = state + beta_t[..., None, None] * write
        outputs.append(torch.einsum("bhvk,bhk->bhv", state, q_t))
    return torch.stack(outputs, dim=2), state


class GatedDeltaNetLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by n_heads={n_heads}")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.alpha_proj = nn.Linear(dim, n_heads, bias=True)
        self.beta_proj = nn.Linear(dim, n_heads, bias=True)
        self.gate_proj = nn.Linear(dim, dim, bias=False)
        self.out_norm = RMSNorm(dim)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        q = F.normalize(q, p=2.0, dim=-1)
        k = F.normalize(k, p=2.0, dim=-1)
        alpha = torch.exp(-F.softplus(self.alpha_proj(x))).transpose(1, 2)
        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)
        recurrent, _ = sequential_gated_delta_rule(q, k, v, alpha, beta)
        recurrent = recurrent.transpose(1, 2).contiguous().view(batch, seq_len, self.dim)
        gated = self.out_norm(recurrent) * F.silu(self.gate_proj(x))
        return self.out_proj(self.dropout(gated))
