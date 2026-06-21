"""Model factory for POEM candidates."""

from __future__ import annotations

from torch import nn

from models.backbone_a import PoemAdaptiveFusion
from models.backbone_b import PoemFusionNoRecursion
from models.backbone_c import PoemRoPETransformer
from models.backbone_d import PoemAbsoluteTransformer
from models.backbone_e import PoemGDNAblation
from models.backbone_f import PoemHybridRecursive
from models.backbone_g import PoemDenseRoPETransformer
from models.backbone_h import PoemHRMDenseRoPE
from models.config import POEMConfig, config_for_model_type


def build_model(config: POEMConfig | None = None, model_type: str | None = None, smoke_test: bool = False) -> nn.Module:
    if config is None:
        if model_type is None:
            model_type = "D"
        config = config_for_model_type(model_type, smoke_test=smoke_test)
    normalized = config.model_type.upper()
    if normalized == "A":
        return PoemAdaptiveFusion(config)
    if normalized == "C":
        return PoemRoPETransformer(config)
    if normalized == "D":
        return PoemAbsoluteTransformer(config)
    if normalized == "B":
        return PoemFusionNoRecursion(config)
    if normalized == "E":
        return PoemGDNAblation(config)
    if normalized == "F":
        return PoemHybridRecursive(config)
    if normalized == "G":
        return PoemDenseRoPETransformer(config)
    if normalized == "H":
        return PoemHRMDenseRoPE(config)
    raise NotImplementedError(f"Candidate {normalized} has not been wired yet")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
