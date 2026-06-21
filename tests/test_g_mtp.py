from __future__ import annotations

import math

import torch

from models.backbone_g_mtp import build_mtp_targets, mtp_loss
from models.build import build_model, count_parameters
from models.config import config_for_model_type
from tokenizer.vocab import (
    DUR_PAD,
    N_DUR,
    N_PITCH,
    N_POS,
    N_TOKEN_TYPES,
    N_VEL,
    PAD_EVENT,
    POS_PAD,
    TYPE_NOTE,
    TYPE_PAD,
    VEL_PAD,
)


def _note(pitch: int) -> list[int]:
    return [TYPE_NOTE, pitch, 1, 2, 3]


def test_mtp_targets_shift_and_mask_sequence_boundaries() -> None:
    primary_targets = torch.tensor(
        [[_note(60), _note(61), _note(62), [TYPE_PAD, 128, DUR_PAD, VEL_PAD, POS_PAD]]],
        dtype=torch.long,
    )

    targets = build_mtp_targets(primary_targets, horizon=4)

    assert targets.shape == (4, 1, 4, 5)
    assert targets[0, 0].tolist() == primary_targets[0].tolist()
    assert targets[1, 0].tolist() == [_note(61), _note(62), list(PAD_EVENT), list(PAD_EVENT)]
    assert targets[2, 0].tolist() == [_note(62), list(PAD_EVENT), list(PAD_EVENT), list(PAD_EVENT)]
    assert targets[3, 0].tolist() == [list(PAD_EVENT), list(PAD_EVENT), list(PAD_EVENT), list(PAD_EVENT)]


def test_mtp_loss_combines_primary_and_weighted_auxiliary_losses() -> None:
    primary_targets = torch.tensor(
        [[_note(60), _note(61), [TYPE_PAD, 128, DUR_PAD, VEL_PAD, POS_PAD]]],
        dtype=torch.long,
    )
    batch_size, seq_len = primary_targets.shape[:2]
    logits_by_offset = [
        {
            "type": torch.zeros(batch_size, seq_len, N_TOKEN_TYPES),
            "pitch": torch.zeros(batch_size, seq_len, N_PITCH),
            "duration": torch.zeros(batch_size, seq_len, N_DUR),
            "velocity": torch.zeros(batch_size, seq_len, N_VEL),
            "position": torch.zeros(batch_size, seq_len, N_POS),
        }
        for _ in range(3)
    ]

    total, losses, _ = mtp_loss(logits_by_offset, primary_targets, aux_weight=0.5)

    uniform_note_loss = sum(math.log(size) for size in (N_TOKEN_TYPES, N_PITCH, N_DUR, N_VEL, N_POS))
    assert torch.isclose(losses["mtp_offset_1_loss"], torch.tensor(uniform_note_loss))
    assert torch.isclose(losses["mtp_offset_2_loss"], torch.tensor(uniform_note_loss))
    assert torch.isclose(losses["mtp_offset_3_loss"], torch.tensor(0.0))
    assert torch.isclose(total, torch.tensor(uniform_note_loss * 1.5))


def test_candidate_g_mtp_forward_backward_and_parameter_delta() -> None:
    config = config_for_model_type("G_MTP", smoke_test=True)
    config.mtp_horizon = 4
    model = build_model(config)
    batch = torch.tensor([[_note(60), _note(61), _note(62), _note(63), _note(64)]], dtype=torch.long)

    output = model(batch[:, :-1], batch[:, 1:])

    assert output.loss is not None
    assert torch.isfinite(output.loss)
    assert output.metrics["mtp_horizon"] == float(config.mtp_horizon)
    assert output.metrics["mtp_primary_loss"] <= output.metrics["mtp_total_loss"]
    output.loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)

    g_params = count_parameters(build_model(config_for_model_type("G", smoke_test=True)))
    g_mtp_params = count_parameters(model)
    assert g_mtp_params > g_params
    assert g_mtp_params - g_params == (config.mtp_horizon - 1) * config.d_model * (
        N_TOKEN_TYPES + N_PITCH + N_DUR + N_VEL + N_POS
    )
