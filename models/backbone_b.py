"""Candidate B: Attention+GDN fusion without recursion."""

from __future__ import annotations

import torch
from torch import nn

from models.backbone_d import _sample_event
from models.config import POEMConfig
from models.embeddings import FieldedOutputHeads, FieldedTokenEmbedding, POEMModelOutput, fielded_cross_entropy
from models.layers.attention import RoPECausalSelfAttention
from models.layers.ffn import SwiGLUFeedForward
from models.layers.gated_deltanet import GatedDeltaNetLayer
from models.layers.norm import RMSNorm
from tokenizer.vocab import PIECE_START_EVENT, TYPE_PIECE_END


class FusionBlock(nn.Module):
    def __init__(self, config: POEMConfig) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(config.d_model)
        self.attn = RoPECausalSelfAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            rope_base=config.rope_base,
        )
        self.norm_gdn = RMSNorm(config.d_model)
        self.gdn = GatedDeltaNetLayer(config.d_model, config.n_heads, dropout=config.dropout)
        self.norm_ffn = RMSNorm(config.d_model)
        self.ffn = SwiGLUFeedForward(config.d_model, config.ffn_hidden_dim, dropout=config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x))
        x = x + self.gdn(self.norm_gdn(x))
        x = x + self.ffn(self.norm_ffn(x))
        return x


class PoemFusionNoRecursion(nn.Module):
    def __init__(self, config: POEMConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = FieldedTokenEmbedding(config.d_model, dropout=config.dropout)
        self.prelude = nn.ModuleList([FusionBlock(config) for _ in range(2)])
        self.middle = FusionBlock(config)
        self.coda = FusionBlock(config)
        self.norm = RMSNorm(config.d_model)
        self.heads = FieldedOutputHeads(config.d_model)

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor | None = None) -> POEMModelOutput:
        x = self.embed(token_ids)
        for layer in self.prelude:
            x = layer(x)
        x = self.middle(x)
        x = self.coda(x)
        logits = self.heads(self.norm(x))
        loss = fielded_cross_entropy(logits, targets) if targets is not None else None
        return POEMModelOutput(logits=logits, loss=loss, metrics={})

    @torch.no_grad()
    def generate(
        self,
        seed_token: tuple[int, int, int, int, int] = PIECE_START_EVENT,
        max_len: int = 160,
        temperature: float = 1.0,
    ) -> list[tuple[int, int, int, int, int]]:
        self.eval()
        device = next(self.parameters()).device
        events = [tuple(seed_token)]
        for _ in range(max_len - 1):
            x = torch.tensor(events, dtype=torch.long, device=device)[None, :, :]
            output = self.forward(x)
            event = _sample_event(output.logits, temperature)
            events.append(event)
            if event[0] == TYPE_PIECE_END:
                break
        return events
