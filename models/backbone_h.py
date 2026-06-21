"""Candidate H: HRM-style dense RoPE hierarchy.

This adapts the public HRM-Text recurrence pattern to POEM's fielded symbolic
music model: a slow H-level dense RoPE Transformer receives the fast L-level
state, while the L-level repeatedly executes with the current H-level plan.
Both levels keep full quadratic RoPE attention.
"""

from __future__ import annotations

import torch
from torch import nn

from models.backbone_d import TransformerBlock, _sample_event
from models.config import POEMConfig
from models.embeddings import FieldedOutputHeads, FieldedTokenEmbedding, POEMModelOutput, fielded_cross_entropy
from models.layers.norm import RMSNorm
from tokenizer.vocab import PIECE_START_EVENT, TYPE_PIECE_END


class DenseRoPEStack(nn.Module):
    def __init__(self, config: POEMConfig, n_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(config, use_rope=True) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class HRMRecurrentLevel(nn.Module):
    def __init__(self, config: POEMConfig, n_layers: int) -> None:
        super().__init__()
        self.injection_gain = nn.Parameter(torch.ones(config.d_model))
        self.core = DenseRoPEStack(config, n_layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor) -> torch.Tensor:
        injected = hidden_states + self.injection_gain[None, None, :] * input_injection
        return self.core(injected)


class PoemHRMDenseRoPE(nn.Module):
    def __init__(self, config: POEMConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = FieldedTokenEmbedding(config.d_model, dropout=config.dropout)
        layers_per_level = max(1, config.n_layers // 2) if config.hrm_half_layers else max(1, config.n_layers)
        self.h_level = HRMRecurrentLevel(config, layers_per_level)
        self.l_level = HRMRecurrentLevel(config, layers_per_level)
        self.norm = RMSNorm(config.d_model)
        self.heads = FieldedOutputHeads(config.d_model)
        z_l_init = torch.empty(config.d_model)
        nn.init.trunc_normal_(z_l_init, std=1.0)
        self.register_buffer("z_l_init", z_l_init, persistent=True)

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor | None = None) -> POEMModelOutput:
        z_h = self.embed(token_ids)
        z_l = self.z_l_init.to(dtype=z_h.dtype, device=z_h.device)[None, None, :].expand_as(z_h)

        total_l_steps = self.config.hrm_h_cycles * self.config.hrm_l_cycles
        bp_steps = max(1, min(int(self.config.hrm_bp_steps), self.config.hrm_h_cycles + total_l_steps))
        h_bp_steps = min(self.config.hrm_h_cycles, max(0, bp_steps - 1))
        l_bp_steps = bp_steps - h_bp_steps

        l_step = 0
        for h_step in range(self.config.hrm_h_cycles):
            for _ in range(self.config.hrm_l_cycles):
                l_step += 1
                allow_grad = l_step > total_l_steps - l_bp_steps
                with torch.set_grad_enabled(torch.is_grad_enabled() and allow_grad):
                    z_l = self.l_level(z_l, z_h)
            allow_grad = h_step >= self.config.hrm_h_cycles - h_bp_steps
            with torch.set_grad_enabled(torch.is_grad_enabled() and allow_grad):
                z_h = self.h_level(z_h, z_l)

        logits = self.heads(self.norm(z_h))
        loss = fielded_cross_entropy(logits, targets) if targets is not None else None
        metrics = {
            "hrm_h_cycles": torch.tensor(float(self.config.hrm_h_cycles), device=z_h.device),
            "hrm_l_cycles": torch.tensor(float(self.config.hrm_l_cycles), device=z_h.device),
            "hrm_bp_steps": torch.tensor(float(bp_steps), device=z_h.device),
            "hrm_effective_dense_passes": torch.tensor(
                float((self.config.hrm_h_cycles + total_l_steps) * len(self.h_level.core.layers)),
                device=z_h.device,
            ),
        }
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
