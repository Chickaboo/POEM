"""Candidate F: hybrid-head recursive GDN + dense RoPE attention."""

from __future__ import annotations

import torch
from torch import nn

from models.backbone_d import _sample_event
from models.config import POEMConfig
from models.embeddings import FieldedOutputHeads, FieldedTokenEmbedding, POEMModelOutput, fielded_cross_entropy
from models.layers.depth_ffn import DepthFFN
from models.layers.ffn import SwiGLUFeedForward
from models.layers.halting import ACTHaltingHead
from models.layers.hybrid_mixer import HybridGDNRoPEMixer
from models.layers.norm import RMSNorm
from tokenizer.vocab import PIECE_START_EVENT, TYPE_PAD, TYPE_PIECE_END


class HybridBlock(nn.Module):
    def __init__(self, config: POEMConfig, depth_ffn: bool = False) -> None:
        super().__init__()
        self.norm_mixer = RMSNorm(config.d_model)
        self.mixer = HybridGDNRoPEMixer(
            dim=config.d_model,
            gdn_dim=config.hybrid_gdn_dim,
            attn_dim=config.hybrid_attn_dim,
            gdn_heads=config.hybrid_gdn_heads,
            attn_heads=config.hybrid_attn_heads,
            dropout=config.dropout,
            rope_base=config.rope_base,
            use_flash_gdn=config.use_flash_gdn,
            fla_gdn_heads=config.hybrid_fla_gdn_heads,
            fla_gdn_head_dim=config.hybrid_fla_gdn_head_dim,
            use_short_conv=config.hybrid_use_short_conv,
        )
        self.norm_ffn = RMSNorm(config.d_model)
        self.depth_ffn = depth_ffn
        if depth_ffn:
            self.ffn = DepthFFN(config.d_model, config.ffn_hidden_dim, config.max_loops, dropout=config.dropout)
        else:
            self.ffn = SwiGLUFeedForward(config.d_model, config.ffn_hidden_dim, dropout=config.dropout)

    def forward(self, x: torch.Tensor, step_index: int | None = None) -> torch.Tensor:
        x = x + self.mixer(self.norm_mixer(x))
        if self.depth_ffn:
            if step_index is None:
                raise ValueError("step_index is required for depth-modulated HybridBlock")
            x = x + self.ffn(self.norm_ffn(x), step_index)
        else:
            x = x + self.ffn(self.norm_ffn(x))
        return x


class HybridAdaptiveRecurrentBlock(nn.Module):
    def __init__(self, config: POEMConfig) -> None:
        super().__init__()
        self.config = config
        self.halt_threshold = config.halt_threshold
        self.epsilon = nn.Parameter(torch.full((config.max_loops, config.d_model), 0.1))
        self.block = HybridBlock(config, depth_ffn=True)
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
            proposed = self.block(proposed, step_index=step)
            halt_prob = self.halt(proposed).squeeze(-1)
            if not torch.isfinite(halt_prob).all():
                raise FloatingPointError("ACT halting probability produced NaN or Inf")
            halt_values.append(halt_prob.detach())
            state = torch.where(active[..., None], proposed, state)
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


class PoemHybridRecursive(nn.Module):
    def __init__(self, config: POEMConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = FieldedTokenEmbedding(config.d_model, dropout=config.dropout)
        self.prelude = nn.ModuleList([HybridBlock(config) for _ in range(config.hybrid_prelude_layers)])
        self.recurrent = HybridAdaptiveRecurrentBlock(config)
        self.coda = nn.ModuleList([HybridBlock(config) for _ in range(config.hybrid_coda_layers)])
        self.norm = RMSNorm(config.d_model)
        self.heads = FieldedOutputHeads(config.d_model)
        self._validate_flash_gdn_requirement()

    def hybrid_gdn_status(self) -> dict[str, object]:
        mixers = [module for module in self.modules() if isinstance(module, HybridGDNRoPEMixer)]
        flash_count = sum(1 for mixer in mixers if mixer.using_flash_gdn)
        errors = [mixer.flash_error for mixer in mixers if mixer.flash_error]
        return {
            "hybrid_mixers": len(mixers),
            "flash_gdn_mixers": flash_count,
            "fallback_gdn_mixers": len(mixers) - flash_count,
            "short_conv": bool(self.config.hybrid_use_short_conv),
            "flash_errors": sorted(set(errors)),
        }

    def _validate_flash_gdn_requirement(self) -> None:
        if not getattr(self.config, "require_flash_gdn", False):
            return
        status = self.hybrid_gdn_status()
        if int(status["fallback_gdn_mixers"]) == 0:
            return
        errors = status["flash_errors"]
        details = "; ".join(str(error) for error in errors) if errors else "unknown import/initialization error"
        raise RuntimeError(
            "Candidate F was configured with require_flash_gdn=True, but at least one "
            f"hybrid mixer fell back to POEM's sequential GDN. Install a working "
            f"flash-linear-attention CUDA build before training. Import error: {details}"
        )

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor | None = None) -> POEMModelOutput:
        x = self.embed(token_ids)
        for layer in self.prelude:
            x = layer(x)
        encoding = x.detach()
        active_mask = token_ids[..., 0] != TYPE_PAD
        state, metrics = self.recurrent(x, encoding, active_mask=active_mask)
        for layer in self.coda:
            state = layer(state)
        logits = self.heads(self.norm(state))
        extra_loss = None
        if targets is not None:
            loops = metrics["avg_loops_per_token"]
            assert isinstance(loops, torch.Tensor)
            extra_loss = loops.to(state.device) * self.config.ponder_loss_weight
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
