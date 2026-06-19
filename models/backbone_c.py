"""Candidate C: pure RoPE attention Transformer baseline."""

from __future__ import annotations

import torch
from torch import nn

from models.backbone_d import TransformerBlock, _sample_event
from models.config import POEMConfig
from models.embeddings import FieldedOutputHeads, FieldedTokenEmbedding, POEMModelOutput, fielded_cross_entropy
from models.layers.norm import RMSNorm
from tokenizer.vocab import PIECE_START_EVENT, TYPE_PIECE_END


class PoemRoPETransformer(nn.Module):
    def __init__(self, config: POEMConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = FieldedTokenEmbedding(config.d_model, dropout=config.dropout)
        self.layers = nn.ModuleList([TransformerBlock(config, use_rope=True) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        self.heads = FieldedOutputHeads(config.d_model)

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor | None = None) -> POEMModelOutput:
        x = self.embed(token_ids)
        for layer in self.layers:
            x = layer(x)
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
