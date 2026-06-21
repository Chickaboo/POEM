"""Flat-token continuation models using POEM Candidate G/H backbones."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from models.backbone_d import TransformerBlock
from models.backbone_h import HRMRecurrentLevel
from models.layers.norm import RMSNorm


class ContinuationOutput(NamedTuple):
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    metrics: dict[str, torch.Tensor | float] | None = None


@dataclass
class ContinuationConfig:
    model_type: str = "G_CONT"
    vocab_size: int = 374
    pad_id: int = 360
    bos_id: int = 361
    eos_id: int = 362
    event_size: int = 4
    d_model: int = 640
    n_heads: int = 10
    n_layers: int = 8
    ffn_multiplier: float = 8.0 / 3.0
    dropout: float = 0.1
    max_seq_len: int = 2048
    rope_base: float = 10_000.0
    hrm_h_cycles: int = 2
    hrm_l_cycles: int = 3
    hrm_bp_steps: int = 5
    hrm_half_layers: bool = True

    @property
    def ffn_hidden_dim(self) -> int:
        hidden = int(round(self.d_model * self.ffn_multiplier))
        return max(16, ((hidden + 15) // 16) * 16)


def continuation_config_for_model_type(model_type: str, smoke_test: bool = False) -> ContinuationConfig:
    normalized = str(model_type).upper()
    if normalized not in {"G_CONT", "H_CONT"}:
        raise ValueError(f"Unknown continuation model_type {model_type!r}; expected G_CONT or H_CONT")
    config = ContinuationConfig(model_type=normalized)
    if smoke_test:
        config.d_model = 128
        config.n_heads = 4
        config.n_layers = 2
        config.ffn_multiplier = 2.0
        config.dropout = 0.0
        config.max_seq_len = 128
        config.hrm_h_cycles = 1
        config.hrm_l_cycles = 2
        config.hrm_bp_steps = 2
    return config


def continuation_config_from_dict(data: dict) -> ContinuationConfig:
    config = continuation_config_for_model_type(str(data.get("model_type", "G_CONT")), smoke_test=False)
    for key, value in data.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def build_continuation_model(config: ContinuationConfig) -> nn.Module:
    normalized = str(config.model_type).upper()
    if normalized == "G_CONT":
        return ContinuationDenseRoPE(config)
    if normalized == "H_CONT":
        return ContinuationHRMDenseRoPE(config)
    raise ValueError(f"Unsupported continuation model_type {config.model_type!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def continuation_targets(
    token_ids: torch.Tensor,
    seed_length: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    targets = torch.full_like(token_ids, fill_value=int(ignore_index))
    if token_ids.size(1) > 1:
        targets[:, :-1] = token_ids[:, 1:]
    context_len = max(0, int(seed_length) - 1)
    if context_len > 0:
        targets[:, :context_len] = int(ignore_index)
    return targets


def slot_allowed_mask(seq_len: int, vocab_size: int, device: torch.device) -> torch.Tensor:
    slot_ids = (torch.arange(seq_len, device=device, dtype=torch.long) + 1) % 4
    allowed = torch.zeros((seq_len, vocab_size), dtype=torch.bool, device=device)
    ranges = {
        0: (0, 128),
        1: (128, 216),
        2: (216, 344),
        3: (344, 360),
    }
    for slot, (start, end) in ranges.items():
        allowed[slot_ids == slot, max(0, start) : min(vocab_size, end)] = True
    empty = ~allowed.any(dim=-1)
    if empty.any():
        allowed[empty] = True
    return allowed


def apply_slot_mask(logits: torch.Tensor, targets: torch.Tensor | None = None, ignore_index: int = -100) -> torch.Tensor:
    allowed = slot_allowed_mask(logits.size(1), logits.size(2), logits.device)
    masked = logits.masked_fill(~allowed[None, :, :], -1.0e4)
    if targets is not None:
        valid = (targets != int(ignore_index)) & (targets >= 0) & (targets < logits.size(-1))
        if valid.any():
            batch_idx, time_idx = torch.nonzero(valid, as_tuple=True)
            class_idx = targets[batch_idx, time_idx].long()
            masked[batch_idx, time_idx, class_idx] = logits[batch_idx, time_idx, class_idx]
    return masked


def continuation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    slot_aware: bool = True,
) -> tuple[torch.Tensor, int]:
    if bool(slot_aware):
        logits = apply_slot_mask(logits, targets=targets, ignore_index=ignore_index)
    valid = targets != int(ignore_index)
    valid_count = int(valid.sum().item())
    if valid_count <= 0:
        return logits.sum() * 0.0, 0
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=int(ignore_index),
        label_smoothing=float(label_smoothing),
    )
    return loss, valid_count


@torch.no_grad()
def continuation_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    slot_aware: bool = True,
) -> float:
    if bool(slot_aware):
        logits = apply_slot_mask(logits, targets=targets, ignore_index=ignore_index)
    valid = targets != int(ignore_index)
    valid_count = int(valid.sum().item())
    if valid_count <= 0:
        return 0.0
    return float(((logits.argmax(dim=-1) == targets) & valid).sum().item() / valid_count)


def sample_next_token(
    logits: torch.Tensor,
    *,
    context_length: int,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 0.95,
) -> torch.Tensor:
    logits = logits.float()
    vocab_size = logits.size(-1)
    next_slot = int(context_length) % 4
    ranges = {
        0: (0, 128),
        1: (128, 216),
        2: (216, 344),
        3: (344, 360),
    }
    allowed = torch.zeros(vocab_size, dtype=torch.bool, device=logits.device)
    start, end = ranges[next_slot]
    allowed[max(0, start) : min(vocab_size, end)] = True
    logits = logits.masked_fill(~allowed, -1.0e4)
    logits = logits / max(float(temperature), 1.0e-4)
    if int(top_k) > 0 and int(top_k) < vocab_size:
        threshold = torch.topk(logits, int(top_k)).values[..., -1]
        logits = logits.masked_fill(logits < threshold, -1.0e4)
    if float(top_p) < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > float(top_p)
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -1.0e4)
        logits = torch.full_like(logits, -1.0e4).scatter(-1, sorted_indices, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


class ContinuationDenseRoPE(nn.Module):
    def __init__(self, config: ContinuationConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_id)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([TransformerBlock(config, use_rope=True) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor | None = None) -> ContinuationOutput:
        if token_ids.size(1) > self.config.max_seq_len:
            token_ids = token_ids[:, -self.config.max_seq_len :]
            if targets is not None:
                targets = targets[:, -self.config.max_seq_len :]
        x = self.dropout(self.embed(token_ids.long()))
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        metrics: dict[str, torch.Tensor | float] = {
            "dense_rope_layers": torch.tensor(float(len(self.layers)), device=token_ids.device)
        }
        if targets is not None:
            loss, valid = continuation_loss(logits, targets)
            metrics["valid_loss_tokens"] = torch.tensor(float(valid), device=token_ids.device)
        return ContinuationOutput(logits=logits, loss=loss, metrics=metrics)

    @torch.no_grad()
    def generate(self, seed_tokens: list[int] | torch.Tensor, max_new_tokens: int, **sample_kwargs) -> list[int]:
        self.eval()
        device = next(self.parameters()).device
        if isinstance(seed_tokens, torch.Tensor):
            tokens = seed_tokens.to(device=device, dtype=torch.long).flatten().tolist()
        else:
            tokens = [int(token) for token in seed_tokens]
        for _ in range(int(max_new_tokens)):
            x = torch.tensor(tokens[-self.config.max_seq_len :], dtype=torch.long, device=device)[None, :]
            output = self.forward(x)
            next_token = sample_next_token(output.logits[0, -1], context_length=len(tokens), **sample_kwargs)
            tokens.append(int(next_token.item()))
        return tokens


class ContinuationHRMDenseRoPE(nn.Module):
    def __init__(self, config: ContinuationConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_id)
        self.dropout = nn.Dropout(config.dropout)
        layers_per_level = max(1, config.n_layers // 2) if config.hrm_half_layers else max(1, config.n_layers)
        self.h_level = HRMRecurrentLevel(config, layers_per_level)
        self.l_level = HRMRecurrentLevel(config, layers_per_level)
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        z_l_init = torch.empty(config.d_model)
        nn.init.trunc_normal_(z_l_init, std=1.0)
        self.register_buffer("z_l_init", z_l_init, persistent=True)

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor | None = None) -> ContinuationOutput:
        if token_ids.size(1) > self.config.max_seq_len:
            token_ids = token_ids[:, -self.config.max_seq_len :]
            if targets is not None:
                targets = targets[:, -self.config.max_seq_len :]
        z_h = self.dropout(self.embed(token_ids.long()))
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
        logits = self.lm_head(self.norm(z_h))
        loss = None
        metrics: dict[str, torch.Tensor | float] = {
            "hrm_h_cycles": torch.tensor(float(self.config.hrm_h_cycles), device=token_ids.device),
            "hrm_l_cycles": torch.tensor(float(self.config.hrm_l_cycles), device=token_ids.device),
            "hrm_bp_steps": torch.tensor(float(bp_steps), device=token_ids.device),
        }
        if targets is not None:
            loss, valid = continuation_loss(logits, targets)
            metrics["valid_loss_tokens"] = torch.tensor(float(valid), device=token_ids.device)
        return ContinuationOutput(logits=logits, loss=loss, metrics=metrics)

    @torch.no_grad()
    def generate(self, seed_tokens: list[int] | torch.Tensor, max_new_tokens: int, **sample_kwargs) -> list[int]:
        self.eval()
        device = next(self.parameters()).device
        if isinstance(seed_tokens, torch.Tensor):
            tokens = seed_tokens.to(device=device, dtype=torch.long).flatten().tolist()
        else:
            tokens = [int(token) for token in seed_tokens]
        for _ in range(int(max_new_tokens)):
            x = torch.tensor(tokens[-self.config.max_seq_len :], dtype=torch.long, device=device)[None, :]
            output = self.forward(x)
            next_token = sample_next_token(output.logits[0, -1], context_length=len(tokens), **sample_kwargs)
            tokens.append(int(next_token.item()))
        return tokens
