from __future__ import annotations

import torch

from models.backbone_a import AdaptiveRecurrentBlock
from models.config import config_for_model_type


def test_recurrent_block_enforces_max_loop_ceiling() -> None:
    config = config_for_model_type("A", smoke_test=True)
    config.max_loops = 2
    config.halt_threshold = 0.99
    block = AdaptiveRecurrentBlock(config)
    with torch.no_grad():
        block.halt.linear.weight.zero_()
        block.halt.linear.bias.fill_(-50.0)
    x = torch.randn(1, 3, config.d_model)
    _, metrics = block(x, x.detach(), active_mask=torch.ones(1, 3, dtype=torch.bool))
    assert metrics["actual_recurrent_steps"] == 2.0
    assert torch.isclose(metrics["avg_loops_per_token"], torch.tensor(2.0))
