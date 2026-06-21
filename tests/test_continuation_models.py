from __future__ import annotations

import torch

from models.continuation import (
    apply_slot_mask,
    build_continuation_model,
    continuation_config_for_model_type,
    continuation_targets,
    count_parameters,
)


def test_continuation_targets_mask_seed_context() -> None:
    token_ids = torch.tensor([[0, 128, 216, 344, 1, 129, 217, 345]], dtype=torch.long)
    targets = continuation_targets(token_ids, seed_length=4)

    assert targets.tolist() == [[-100, -100, -100, 1, 129, 217, 345, -100]]


def test_slot_mask_allows_expected_quad_classes() -> None:
    logits = torch.zeros(1, 4, 374)
    masked = apply_slot_mask(logits)

    assert masked[0, 0, 128] == 0.0
    assert masked[0, 0, 0] < -1000.0
    assert masked[0, 1, 216] == 0.0
    assert masked[0, 2, 344] == 0.0
    assert masked[0, 3, 0] == 0.0


def test_continuation_g_and_h_smoke_forward_backward() -> None:
    for model_type in ("G_CONT", "H_CONT"):
        config = continuation_config_for_model_type(model_type, smoke_test=True)
        model = build_continuation_model(config)
        token_ids = torch.tensor([[0, 128, 216, 344, 1, 129, 217, 345]], dtype=torch.long)
        targets = continuation_targets(token_ids, seed_length=4)

        output = model(token_ids, targets)

        assert output.loss is not None
        assert torch.isfinite(output.loss)
        output.loss.backward()
        assert any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


def test_continuation_default_profiles_are_around_40m() -> None:
    g = build_continuation_model(continuation_config_for_model_type("G_CONT"))
    h = build_continuation_model(continuation_config_for_model_type("H_CONT"))

    assert 39_000_000 <= count_parameters(g) <= 41_000_000
    assert 39_000_000 <= count_parameters(h) <= 41_000_000
