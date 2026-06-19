"""Feed-forward layers."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_proj = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.out_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.in_proj(x).chunk(2, dim=-1)
        x = F.silu(gate) * value
        return self.out_proj(self.dropout(x))
