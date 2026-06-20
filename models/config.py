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
    hybrid_gdn_dim: int = 288
    hybrid_attn_dim: int = 96
    hybrid_gdn_heads: int = 6
    hybrid_attn_heads: int = 3
    hybrid_fla_gdn_heads: int | None = 3
    hybrid_fla_gdn_head_dim: int | None = 72
    hybrid_prelude_layers: int = 2
    hybrid_coda_layers: int = 2
    use_flash_gdn: bool = True
    require_flash_gdn: bool = False
    hybrid_use_short_conv: bool = False
    hybrid_fla_mode: str = "chunk"

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


@dataclass
class PoemBackboneF(POEMConfig):
    model_type: str = "F"
    d_model: int = 384
    n_heads: int = 6
    max_loops: int = 6
    hybrid_gdn_dim: int = 288
    hybrid_attn_dim: int = 96
    hybrid_gdn_heads: int = 6
    hybrid_attn_heads: int = 3
    # flash-linear-attention recommends num_heads * head_dim ~= 0.75 * hidden_size
    # for gated GDN layers. 3 * 72 = 216 = 0.75 * 288.
    hybrid_fla_gdn_heads: int | None = 3
    hybrid_fla_gdn_head_dim: int | None = 72
    hybrid_prelude_layers: int = 2
    hybrid_coda_layers: int = 2
    use_flash_gdn: bool = True
    require_flash_gdn: bool = False
    hybrid_use_short_conv: bool = False
    hybrid_fla_mode: str = "chunk"
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
        "F": PoemBackboneF,
    }.get(normalized)
    if config_cls is None:
        raise ValueError(f"Unknown model_type {model_type!r}; expected A, B, C, D, E, or F")
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
        if normalized == "F":
            config.hybrid_gdn_dim = 48
            config.hybrid_attn_dim = 16
            config.hybrid_gdn_heads = 3
            config.hybrid_attn_heads = 1
            config.hybrid_fla_gdn_heads = 3
            config.hybrid_fla_gdn_head_dim = 12
            config.hybrid_prelude_layers = 1
            config.hybrid_coda_layers = 1
            config.use_flash_gdn = False
            config.require_flash_gdn = False
            config.hybrid_use_short_conv = False
            config.hybrid_fla_mode = "chunk"
    return config


def config_from_dict(data: dict) -> POEMConfig:
    config = config_for_model_type(str(data.get("model_type", "D")), smoke_test=False)
    for key, value in data.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config
