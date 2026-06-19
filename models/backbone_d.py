"""Candidate D: absolute-position naive Transformer baseline."""

from __future__ import annotations

import torch
from torch import nn

from models.config import POEMConfig
from models.embeddings import FieldedOutputHeads, FieldedTokenEmbedding, POEMModelOutput, fielded_cross_entropy
from models.layers.attention import AbsoluteCausalSelfAttention
from models.layers.ffn import SwiGLUFeedForward
from models.layers.norm import RMSNorm
from tokenizer.vocab import PAD_EVENT, PIECE_START_EVENT, TYPE_NOTE, TYPE_PIECE_END


class TransformerBlock(nn.Module):
    def __init__(self, config: POEMConfig, use_rope: bool = False) -> None:
        super().__init__()
        from models.layers.attention import CausalSelfAttention

        self.norm_attn = RMSNorm(config.d_model)
        if use_rope:
            self.attn = CausalSelfAttention(
                config.d_model,
                config.n_heads,
                dropout=config.dropout,
                use_rope=True,
                rope_base=config.rope_base,
            )
        else:
            self.attn = AbsoluteCausalSelfAttention(config.d_model, config.n_heads, dropout=config.dropout)
        self.norm_ffn = RMSNorm(config.d_model)
        self.ffn = SwiGLUFeedForward(config.d_model, config.ffn_hidden_dim, dropout=config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x))
        x = x + self.ffn(self.norm_ffn(x))
        return x


class PoemAbsoluteTransformer(nn.Module):
    def __init__(self, config: POEMConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = FieldedTokenEmbedding(config.d_model, dropout=config.dropout)
        self.absolute_pos = nn.Embedding(config.max_seq_len, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config, use_rope=False) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        self.heads = FieldedOutputHeads(config.d_model)

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor | None = None) -> POEMModelOutput:
        if token_ids.size(1) > self.config.max_seq_len:
            raise ValueError(f"Sequence length {token_ids.size(1)} exceeds max_seq_len={self.config.max_seq_len}")
        positions = torch.arange(token_ids.size(1), device=token_ids.device)
        x = self.embed(token_ids) + self.absolute_pos(positions)[None, :, :]
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


def _sample_event(logits: dict[str, torch.Tensor], temperature: float) -> tuple[int, int, int, int, int]:
    temp = max(float(temperature), 1e-4)

    def sample(name: str) -> int:
        probs = torch.softmax(logits[name][0, -1] / temp, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    token_type = sample("type")
    if token_type == TYPE_NOTE:
        return (token_type, sample("pitch"), sample("duration"), sample("velocity"), sample("position"))
    if token_type == TYPE_PIECE_END:
        return (token_type, PAD_EVENT[1], PAD_EVENT[2], PAD_EVENT[3], PAD_EVENT[4])
    return (token_type, PAD_EVENT[1], PAD_EVENT[2], PAD_EVENT[3], PAD_EVENT[4])
