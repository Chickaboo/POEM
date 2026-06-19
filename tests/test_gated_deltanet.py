from __future__ import annotations

import torch

from models.layers.gated_deltanet import sequential_gated_delta_rule


def test_sequential_gated_delta_rule_matches_worked_example() -> None:
    q = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]])
    k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]]])
    v = torch.tensor([[[[2.0, 0.0], [0.0, 3.0], [4.0, 0.0]]]])
    alpha = torch.tensor([[[0.5, 1.0, 0.25]]])
    beta = torch.tensor([[[1.0, 0.5, 1.0]]])

    outputs, final_state = sequential_gated_delta_rule(q, k, v, alpha, beta)

    expected_outputs = torch.tensor([[[[2.0, 0.0], [0.0, 1.5], [4.0, 0.375]]]])
    expected_state = torch.tensor([[[[4.0, 0.0], [0.0, 0.375]]]])
    torch.testing.assert_close(outputs, expected_outputs)
    torch.testing.assert_close(final_state, expected_state)
