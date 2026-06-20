"""Candidate A: Attention+GDN fusion with adaptive recursion."""

from __future__ import annotations

import torch
from torch import nn

from models.backbone_b import FusionBlock
from models.backbone_d import _sample_event
from models.config import POEMConfig
from models.embeddings import FieldedOutputHeads, FieldedTokenEmbedding, POEMModelOutput, fielded_cross_entropy
from models.layers.attention import RoPECausalSelfAttention
from models.layers.depth_ffn import DepthFFN
from models.layers.gated_deltanet import GatedDeltaNetLayer
from models.layers.halting import ACTHaltingHead
from models.layers.norm import RMSNorm
from tokenizer.vocab import PIECE_START_EVENT, TYPE_PAD, TYPE_PIECE_END


class AdaptiveRecurrentBlock(nn.Module):
    def __init__(self, config: POEMConfig, halt_threshold: float | None = None) -> None:
        super().__init__()
        self.config = config
        self.halt_threshold = config.halt_threshold if halt_threshold is None else halt_threshold
        self.epsilon = nn.Parameter(torch.full((config.max_loops, config.d_model), 0.1))
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
        self.depth_ffn = DepthFFN(config.d_model, config.ffn_hidden_dim, config.max_loops, dropout=config.dropout)
        self.halt = ACTHaltingHead(config.d_model)

    def forward(
        self,
        state: torch.Tensor,
        encoding: torch.Tensor,
        active_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        batch, seq_len, _ = state.shape
        valid = active_mask.bool() if active_mask is not None else torch.ones(batch, seq_len, dtype=torch.bool, device=state.device)
        active = valid.clone()
        accumulated = state.new_zeros(batch, seq_len)
        loops_taken = state.new_zeros(batch, seq_len)
        halt_values: list[torch.Tensor] = []
        actual_steps = 0

        for step in range(self.config.max_loops):
            actual_steps = step + 1
            proposed = state + self.epsilon[step][None, None, :] * encoding
            proposed = proposed + self.attn(self.norm_attn(proposed))
            proposed = proposed + self.gdn(self.norm_gdn(proposed))
            proposed = proposed + self.depth_ffn(self.norm_ffn(proposed), step)
            halt_prob = self.halt(proposed).squeeze(-1)
            if not torch.isfinite(halt_prob).all():
                raise FloatingPointError("ACT halting probability produced NaN or Inf")
            halt_values.append(halt_prob.detach())
            update_mask = active[..., None]
            state = torch.where(update_mask, proposed, state)
            loops_taken = loops_taken + active.float()
            accumulated = torch.where(active, accumulated + halt_prob, accumulated)
            active = valid & (accumulated < self.halt_threshold)
            if not active.any():
                break

        valid_count = valid.float().sum().clamp_min(1.0)
        avg_loops = (loops_taken * valid.float()).sum() / valid_count
        halt_stack = torch.stack(halt_values, dim=0)
        metrics: dict[str, torch.Tensor | float] = {
            "avg_loops_per_token": avg_loops.detach(),
            "actual_recurrent_steps": state.new_tensor(float(actual_steps)),
            "halt_prob_mean": halt_stack.mean().detach(),
            "halt_prob_max": halt_stack.max().detach(),
        }
        return state, metrics


class PoemAdaptiveFusion(nn.Module):
    def __init__(self, config: POEMConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = FieldedTokenEmbedding(config.d_model, dropout=config.dropout)
        self.prelude = nn.ModuleList([FusionBlock(config) for _ in range(2)])
        self.recurrent = AdaptiveRecurrentBlock(config)
        self.coda = FusionBlock(config)
        self.norm = RMSNorm(config.d_model)
        self.heads = FieldedOutputHeads(config.d_model)

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor | None = None) -> POEMModelOutput:
        x = self.embed(token_ids)
        for layer in self.prelude:
            x = layer(x)
        encoding = x.detach()
        active_mask = token_ids[..., 0] != TYPE_PAD
        state, metrics = self.recurrent(x, encoding, active_mask=active_mask)
        x = self.coda(state)
        logits = self.heads(self.norm(x))
        extra_loss = None
        if targets is not None:
            loops = metrics["avg_loops_per_token"]
            assert isinstance(loops, torch.Tensor)
            extra_loss = loops.to(x.device) * self.config.ponder_loss_weight
        loss = fielded_cross_entropy(logits, targets, extra_loss=extra_loss) if targets is not None else None
        return POEMModelOutput(logits=logits, loss=loss, metrics=metrics)

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
