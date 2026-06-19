"""Normalization layers."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMSNorm keeps the pre-norm blocks lightweight for small CPU models."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight
