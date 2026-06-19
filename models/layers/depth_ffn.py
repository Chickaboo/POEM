"""Depth-modulated FFN for POEM's recurrent block."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DepthFFN(nn.Module):
    """SwiGLU FFN with a learned per-loop bias added before the SiLU gate."""

    def __init__(self, dim: int, hidden_dim: int, max_loops: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_proj = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.step_embed = nn.Embedding(max_loops, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, step_index: int) -> torch.Tensor:
        gate, value = self.in_proj(x).chunk(2, dim=-1)
        step_bias = self.step_embed.weight[int(step_index)][None, None, :]
        x = F.silu(gate + step_bias) * value
        return self.out_proj(self.dropout(x))
