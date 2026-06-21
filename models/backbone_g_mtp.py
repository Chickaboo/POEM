"""Candidate G-MTP: Candidate G backbone with multi-token prediction heads."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models.backbone_d import _sample_event
from models.backbone_g import PoemDenseRoPETransformer
from models.config import POEMConfig
from models.embeddings import FieldedOutputHeads, POEMModelOutput
from tokenizer.vocab import (
    PAD_EVENT,
    PIECE_START_EVENT,
    TYPE_NOTE,
    TYPE_PAD,
    TYPE_PIECE_END,
)


def build_mtp_targets(primary_targets: torch.Tensor, horizon: int) -> torch.Tensor:
    """Build offset targets where index 0 is the ordinary next-token target."""
    if horizon < 1:
        raise ValueError(f"mtp_horizon must be >= 1, got {horizon}")
    targets = primary_targets.new_tensor(PAD_EVENT).view(1, 1, 5).repeat(horizon, *primary_targets.shape[:2], 1)
    targets[0] = primary_targets
    for offset_index in range(1, horizon):
        shift = offset_index
        if shift < primary_targets.size(1):
            targets[offset_index, :, :-shift] = primary_targets[:, shift:]
    return targets


def safe_fielded_cross_entropy(logits: dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
    target_type = targets[..., 0].long()
    valid_mask = target_type != TYPE_PAD
    loss = logits["type"].sum() * 0.0
    if valid_mask.any():
        loss = loss + F.cross_entropy(logits["type"][valid_mask], target_type[valid_mask])
    note_mask = target_type == TYPE_NOTE
    if note_mask.any():
        for field_index, field_name in enumerate(("pitch", "duration", "velocity", "position"), start=1):
            loss = loss + F.cross_entropy(logits[field_name][note_mask], targets[..., field_index][note_mask].long())
    return loss


def mtp_loss(
    logits_by_offset: list[dict[str, torch.Tensor]],
    primary_targets: torch.Tensor,
    aux_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    mtp_targets = build_mtp_targets(primary_targets, len(logits_by_offset))
    losses: dict[str, torch.Tensor] = {}
    total = logits_by_offset[0]["type"].sum() * 0.0
    for index, logits in enumerate(logits_by_offset):
        offset = index + 1
        offset_loss = safe_fielded_cross_entropy(logits, mtp_targets[index])
        losses[f"mtp_offset_{offset}_loss"] = offset_loss
        total = total + offset_loss if offset == 1 else total + aux_weight * offset_loss
    losses["mtp_primary_loss"] = losses["mtp_offset_1_loss"]
    losses["mtp_total_loss"] = total
    return total, losses, mtp_targets


class PoemDenseRoPETransformerMTP(nn.Module):
    def __init__(self, config: POEMConfig) -> None:
        super().__init__()
        self.config = config
        self.horizon = max(1, int(config.mtp_horizon))
        self.aux_weight = float(config.mtp_aux_weight)
        self.backbone = PoemDenseRoPETransformer(config)
        # Keep auxiliary heads lightweight: they branch directly from Candidate G's
        # final hidden state rather than duplicating transformer blocks per offset.
        self.aux_heads = nn.ModuleList(
            [FieldedOutputHeads(config.d_model) for _ in range(max(0, self.horizon - 1))]
        )

    def _hidden_states(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.backbone.embed(token_ids)
        for layer in self.backbone.layers:
            x = layer(x)
        return self.backbone.norm(x)

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor | None = None) -> POEMModelOutput:
        hidden = self._hidden_states(token_ids)
        logits_by_offset = [self.backbone.heads(hidden)]
        logits_by_offset.extend(head(hidden) for head in self.aux_heads)
        metrics: dict[str, torch.Tensor | float] = {
            "dense_rope_layers": torch.tensor(float(len(self.backbone.layers)), device=hidden.device),
            "mtp_horizon": torch.tensor(float(self.horizon), device=hidden.device),
            "mtp_aux_weight": torch.tensor(float(self.aux_weight), device=hidden.device),
        }
        loss = None
        if targets is not None:
            loss, loss_metrics, _ = mtp_loss(logits_by_offset, targets, self.aux_weight)
            metrics.update(loss_metrics)
        return POEMModelOutput(logits=logits_by_offset[0], loss=loss, metrics=metrics)

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
