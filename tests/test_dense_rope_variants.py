from __future__ import annotations

import torch

from models.build import build_model, count_parameters
from models.config import config_for_model_type
from tokenizer.vocab import TYPE_NOTE


def _note_batch(batch_size: int = 2, seq_len: int = 12) -> torch.Tensor:
    batch = torch.empty(batch_size, seq_len, 5, dtype=torch.long)
    batch[..., 0] = TYPE_NOTE
    batch[..., 1] = torch.randint(0, 128, (batch_size, seq_len), dtype=torch.long)
    batch[..., 2] = torch.randint(0, 32, (batch_size, seq_len), dtype=torch.long)
    batch[..., 3] = torch.randint(0, 16, (batch_size, seq_len), dtype=torch.long)
    batch[..., 4] = torch.randint(0, 16, (batch_size, seq_len), dtype=torch.long)
    return batch


def test_candidate_g_dense_rope_forward_backward() -> None:
    config = config_for_model_type("G", smoke_test=True)
    model = build_model(config)
    batch = _note_batch()

    output = model(batch[:, :-1], batch[:, 1:])

    assert output.loss is not None
    assert torch.isfinite(output.loss)
    assert output.metrics["dense_rope_layers"] == float(config.n_layers)
    output.loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


def test_candidate_h_hrm_dense_rope_forward_backward() -> None:
    config = config_for_model_type("H", smoke_test=True)
    model = build_model(config)
    batch = _note_batch()

    output = model(batch[:, :-1], batch[:, 1:])

    assert output.loss is not None
    assert torch.isfinite(output.loss)
    assert output.metrics["hrm_h_cycles"] == float(config.hrm_h_cycles)
    assert output.metrics["hrm_l_cycles"] == float(config.hrm_l_cycles)
    output.loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


def test_hrm_variant_adds_recurrence_without_sharing_non_hrm_class() -> None:
    dense = build_model(config_for_model_type("G", smoke_test=True))
    hrm = build_model(config_for_model_type("H", smoke_test=True))

    assert dense.__class__.__name__ == "PoemDenseRoPETransformer"
    assert hrm.__class__.__name__ == "PoemHRMDenseRoPE"
    assert count_parameters(hrm) > 0
