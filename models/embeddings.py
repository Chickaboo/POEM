"""Token embedding and output heads for summed-field POEM events."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from tokenizer.vocab import (
    DUR_PAD,
    N_DUR,
    N_PITCH,
    N_POS,
    N_TOKEN_TYPES,
    N_VEL,
    PITCH_PAD,
    POS_PAD,
    TYPE_NOTE,
    TYPE_PAD,
    VEL_PAD,
)


class POEMModelOutput(NamedTuple):
    logits: dict[str, torch.Tensor]
    loss: torch.Tensor | None = None
    metrics: dict[str, torch.Tensor | float] | None = None


class FieldedTokenEmbedding(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.type_emb = nn.Embedding(N_TOKEN_TYPES, dim)
        self.pitch_emb = nn.Embedding(N_PITCH, dim, padding_idx=PITCH_PAD)
        self.duration_emb = nn.Embedding(N_DUR, dim, padding_idx=DUR_PAD)
        self.velocity_emb = nn.Embedding(N_VEL, dim, padding_idx=VEL_PAD)
        self.position_emb = nn.Embedding(N_POS, dim, padding_idx=POS_PAD)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        token_type, pitch, duration, velocity, position = tokens.long().unbind(dim=-1)
        x = (
            self.type_emb(token_type)
            + self.pitch_emb(pitch)
            + self.duration_emb(duration)
            + self.velocity_emb(velocity)
            + self.position_emb(position)
        )
        return self.dropout(x)


class FieldedOutputHeads(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.type_head = nn.Linear(dim, N_TOKEN_TYPES, bias=False)
        self.pitch_head = nn.Linear(dim, N_PITCH, bias=False)
        self.duration_head = nn.Linear(dim, N_DUR, bias=False)
        self.velocity_head = nn.Linear(dim, N_VEL, bias=False)
        self.position_head = nn.Linear(dim, N_POS, bias=False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "type": self.type_head(x),
            "pitch": self.pitch_head(x),
            "duration": self.duration_head(x),
            "velocity": self.velocity_head(x),
            "position": self.position_head(x),
        }


def fielded_cross_entropy(
    logits: dict[str, torch.Tensor],
    targets: torch.Tensor,
    extra_loss: torch.Tensor | None = None,
) -> torch.Tensor:
    target_type = targets[..., 0].long()
    loss = F.cross_entropy(
        logits["type"].reshape(-1, logits["type"].size(-1)),
        target_type.reshape(-1),
        ignore_index=TYPE_PAD,
    )
    note_mask = target_type == TYPE_NOTE
    if note_mask.any():
        for field_index, field_name in enumerate(("pitch", "duration", "velocity", "position"), start=1):
            loss = loss + F.cross_entropy(logits[field_name][note_mask], targets[..., field_index][note_mask].long())
    if extra_loss is not None:
        loss = loss + extra_loss
    return loss


@torch.no_grad()
def fielded_loss_breakdown(logits: dict[str, torch.Tensor], targets: torch.Tensor) -> dict[str, float]:
    target_type = targets[..., 0].long()
    valid_mask = target_type != TYPE_PAD
    note_mask = target_type == TYPE_NOTE
    metrics: dict[str, float] = {
        "valid_tokens": float(valid_mask.sum().item()),
        "note_tokens": float(note_mask.sum().item()),
        "note_fraction": float(note_mask.sum().item() / max(1, valid_mask.sum().item())),
    }
    if valid_mask.any():
        type_logits = logits["type"][valid_mask]
        type_targets = target_type[valid_mask]
        metrics["type_loss"] = float(F.cross_entropy(type_logits, type_targets).detach())
        metrics["type_acc"] = float((type_logits.argmax(dim=-1) == type_targets).float().mean().detach())
    if note_mask.any():
        for field_index, field_name in enumerate(("pitch", "duration", "velocity", "position"), start=1):
            field_logits = logits[field_name][note_mask]
            field_targets = targets[..., field_index][note_mask].long()
            metrics[f"{field_name}_loss"] = float(F.cross_entropy(field_logits, field_targets).detach())
            metrics[f"{field_name}_acc"] = float((field_logits.argmax(dim=-1) == field_targets).float().mean().detach())
    return metrics
