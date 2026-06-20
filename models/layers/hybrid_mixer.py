"""Hybrid sequence mixer for POEM Candidate F.

The mixer reserves most channels for Gated DeltaNet style recurrent mixing and
the remainder for dense causal RoPE attention.  When flash-linear-attention is
installed, the GDN branch uses its chunk-parallel GPU kernel; otherwise it falls
back to POEM's sequential reference GDN for local CPU tests and debugging.
"""

from __future__ import annotations

import torch
from torch import nn

from models.layers.attention import RoPECausalSelfAttention
from models.layers.gated_deltanet import GatedDeltaNetLayer


class OptionalFlashGatedDeltaNet(nn.Module):
    def __init__(
        self,
        dim: int,
        fallback_heads: int,
        dropout: float = 0.0,
        use_flash: bool = True,
        fla_num_heads: int | None = None,
        fla_head_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.using_flash = False
        self.flash_error: str | None = None
        if use_flash:
            try:
                from fla.layers import GatedDeltaNet as FLAGatedDeltaNet

                kwargs = {
                    "hidden_size": dim,
                    "mode": "chunk",
                    "use_gate": True,
                    "use_short_conv": True,
                }
                if fla_num_heads is not None:
                    kwargs["num_heads"] = fla_num_heads
                if fla_head_dim is not None:
                    kwargs["head_dim"] = fla_head_dim
                self.layer = FLAGatedDeltaNet(**kwargs)
                self.using_flash = True
            except Exception as exc:  # pragma: no cover - depends on optional CUDA package
                self.flash_error = str(exc)
                self.layer = GatedDeltaNetLayer(dim, fallback_heads, dropout=dropout)
        else:
            self.layer = GatedDeltaNetLayer(dim, fallback_heads, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.layer(x)
        if isinstance(output, tuple):
            return output[0]
        return output


class HybridGDNRoPEMixer(nn.Module):
    def __init__(
        self,
        dim: int,
        gdn_dim: int,
        attn_dim: int,
        gdn_heads: int,
        attn_heads: int,
        dropout: float = 0.0,
        rope_base: float = 10_000.0,
        use_flash_gdn: bool = True,
        fla_gdn_heads: int | None = None,
        fla_gdn_head_dim: int | None = None,
    ) -> None:
        super().__init__()
        if gdn_dim + attn_dim != dim:
            raise ValueError("gdn_dim + attn_dim must equal dim")
        self.gdn_in = nn.Linear(dim, gdn_dim, bias=False)
        self.attn_in = nn.Linear(dim, attn_dim, bias=False)
        self.gdn = OptionalFlashGatedDeltaNet(
            gdn_dim,
            fallback_heads=gdn_heads,
            dropout=dropout,
            use_flash=use_flash_gdn,
            fla_num_heads=fla_gdn_heads,
            fla_head_dim=fla_gdn_head_dim,
        )
        self.attn = RoPECausalSelfAttention(
            attn_dim,
            attn_heads,
            dropout=dropout,
            rope_base=rope_base,
        )
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    @property
    def using_flash_gdn(self) -> bool:
        return self.gdn.using_flash

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gdn_out = self.gdn(self.gdn_in(x))
        attn_out = self.attn(self.attn_in(x))
        return self.out(self.dropout(torch.cat([gdn_out, attn_out], dim=-1)))
