"""Generate POEM MIDI samples from a checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from models.build import build_model
from models.config import config_from_dict
from tokenizer.tokens_to_midi import tokens_to_midi


def describe_generation_model(model, config) -> None:
    model_type = str(config.model_type).upper()
    if model_type == "G":
        print(
            "Generation model: Candidate G dense RoPE "
            f"(layers={config.n_layers}, d_model={config.d_model}, heads={config.n_heads})",
            flush=True,
        )
        return
    if model_type == "G_MTP":
        print(
            "Generation model: Candidate G-MTP dense RoPE "
            f"(layers={config.n_layers}, d_model={config.d_model}, heads={config.n_heads}, "
            f"mtp_horizon={config.mtp_horizon}; using primary head only)",
            flush=True,
        )
        return
    if model_type == "H":
        layers_per_level = len(model.h_level.core.layers)
        print(
            "Generation model: Candidate H HRM dense RoPE "
            f"(H_cycles={config.hrm_h_cycles}, L_cycles={config.hrm_l_cycles}, "
            f"layers_per_level={layers_per_level}, d_model={config.d_model}, heads={config.n_heads})",
            flush=True,
        )
        return
    print(f"Generation model: Candidate {model_type}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--max_len", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--tempo", type=float, default=120.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--require_flash_gdn",
        action="store_true",
        help="For Candidate F checkpoints trained with FLA, fail early if FLA is unavailable.",
    )
    parser.add_argument(
        "--fla_mode",
        choices=["checkpoint", "chunk", "fused_recurrent"],
        default="checkpoint",
        help="Override Candidate F's FLA runtime mode for generation.",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = config_from_dict(checkpoint["config"])
    if args.require_flash_gdn and hasattr(config, "require_flash_gdn"):
        config.require_flash_gdn = True
        config.use_flash_gdn = True
    if args.fla_mode != "checkpoint" and hasattr(config, "hybrid_fla_mode"):
        config.hybrid_fla_mode = args.fla_mode
    model = build_model(config)
    describe_generation_model(model, config)
    status_fn = getattr(model, "hybrid_gdn_status", None)
    if callable(status_fn):
        print(f"Hybrid GDN status: {status_fn()}", flush=True)
    try:
        model.load_state_dict(checkpoint["model_state"])
    except RuntimeError as exc:
        if str(config.model_type).upper() == "F":
            raise RuntimeError(
                "Could not load Candidate F checkpoint into the current runtime. "
                "If this checkpoint was trained with flash-linear-attention, local "
                "generation also needs a compatible FLA install and should be run with "
                "`--require_flash_gdn`. The sequential fallback is useful for smoke tests, "
                "but it is not weight-compatible with FLA-trained checkpoints."
            ) from exc
        raise
    model.to(device)
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for index in range(args.num_samples):
        events = model.generate(max_len=args.max_len, temperature=args.temperature)
        output_path = args.output_dir / f"poem-{config.model_type.lower()}-sample-{index:03d}.mid"
        tokens_to_midi(events, output_path, tempo=args.tempo)
        print(output_path)


if __name__ == "__main__":
    main()
