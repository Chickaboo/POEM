import torch

from models.build import build_model
from models.config import config_for_model_type
from tokenizer.vocab import TYPE_NOTE


def test_candidate_f_smoke_forward_backward() -> None:
    config = config_for_model_type("F", smoke_test=True)
    model = build_model(config)
    batch = torch.randint(0, 4, (2, 12, 5), dtype=torch.long)
    batch[..., 0] = TYPE_NOTE
    batch[..., 1] = torch.randint(0, 128, (2, 12), dtype=torch.long)
    batch[..., 2] = torch.randint(0, 32, (2, 12), dtype=torch.long)
    batch[..., 3] = torch.randint(0, 16, (2, 12), dtype=torch.long)
    batch[..., 4] = torch.randint(0, 16, (2, 12), dtype=torch.long)

    output = model(batch[:, :-1], batch[:, 1:])

    assert output.loss is not None
    assert torch.isfinite(output.loss)
    assert "avg_loops_per_token" in output.metrics
    output.loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
