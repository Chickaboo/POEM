"""Configuration dataclasses for POEM backbone candidates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class POEMConfig:
    model_type: str = "D"
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 5
    ffn_multiplier: float = 8.0 / 3.0
    dropout: float = 0.1
    max_seq_len: int = 256
    max_loops: int = 6
    halt_threshold: float = 0.99
    ponder_loss_weight: float = 1e-3
    rope_base: float = 10_000.0
    gdn_state_dim: int | None = None
    use_rope: bool = True
    use_absolute_pos: bool = False

    @property
    def ffn_hidden_dim(self) -> int:
        hidden = int(round(self.d_model * self.ffn_multiplier))
        return max(16, ((hidden + 15) // 16) * 16)


@dataclass
class PoemBackboneA(POEMConfig):
    model_type: str = "A"
    d_model: int = 384
    n_heads: int = 6
    max_loops: int = 6
    use_rope: bool = True
    use_absolute_pos: bool = False


@dataclass
class PoemBackboneB(POEMConfig):
    model_type: str = "B"
    d_model: int = 384
    n_heads: int = 6
    use_rope: bool = True
    use_absolute_pos: bool = False


@dataclass
class PoemBackboneC(POEMConfig):
    model_type: str = "C"
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 6
    use_rope: bool = True
    use_absolute_pos: bool = False


@dataclass
class PoemBackboneD(POEMConfig):
    model_type: str = "D"
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 6
    use_rope: bool = False
    use_absolute_pos: bool = True


@dataclass
class PoemBackboneE(POEMConfig):
    model_type: str = "E"
    d_model: int = 448
    n_heads: int = 7
    n_layers: int = 4
    use_rope: bool = True
    use_absolute_pos: bool = False


def config_for_model_type(model_type: str, smoke_test: bool = False) -> POEMConfig:
    normalized = model_type.upper()
    config_cls = {
        "A": PoemBackboneA,
        "B": PoemBackboneB,
        "C": PoemBackboneC,
        "D": PoemBackboneD,
        "E": PoemBackboneE,
    }.get(normalized)
    if config_cls is None:
        raise ValueError(f"Unknown model_type {model_type!r}; expected A, B, C, D, or E")
    config = config_cls()
    if smoke_test:
        config.d_model = 64
        config.n_heads = 4
        config.n_layers = 2 if normalized in {"C", "D"} else 1
        config.ffn_multiplier = 2.0
        config.dropout = 0.0
        config.max_seq_len = 160
        config.max_loops = 2
        config.halt_threshold = 0.5
    return config


def config_from_dict(data: dict) -> POEMConfig:
    config = config_for_model_type(str(data.get("model_type", "D")), smoke_test=False)
    for key, value in data.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config
