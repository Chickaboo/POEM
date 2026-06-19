"""ACT-style halting head."""

from __future__ import annotations

import torch
from torch import nn


class ACTHaltingHead(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))
