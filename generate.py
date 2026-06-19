"""Generate POEM MIDI samples from a checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from models.build import build_model
from models.config import config_from_dict
from tokenizer.tokens_to_midi import tokens_to_midi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--max_len", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--tempo", type=float, default=120.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = config_from_dict(checkpoint["config"])
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state"])
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
