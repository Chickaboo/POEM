"""Generate POEM MIDI samples from a checkpoint."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

from continuation.tokenizer_custom import CustomDeltaTokenizer
from models.build import build_model
from models.config import config_from_dict
from models.continuation import (
    build_continuation_model,
    continuation_config_from_dict,
)
from tokenizer.tokens_to_midi import tokens_to_midi


try:
    from prompt_toolkit import prompt as prompt_toolkit_prompt
    from prompt_toolkit.completion import PathCompleter
except Exception:  # pragma: no cover - optional local nicety
    prompt_toolkit_prompt = None
    PathCompleter = None


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


def load_checkpoint_portable(path: Path, device: torch.device) -> dict:
    if sys.platform.startswith("win"):
        import pathlib

        pathlib.PosixPath = pathlib.WindowsPath
    return torch.load(path, map_location=device, weights_only=False)


def prompt_text(message: str, default: str = "", path: bool = False) -> str:
    suffix = f" [{default}]" if default else ""
    full_message = f"{message}{suffix}: "
    if prompt_toolkit_prompt is not None:
        completer = PathCompleter(expanduser=True) if path and PathCompleter is not None else None
        value = prompt_toolkit_prompt(full_message, default="", completer=completer)
    else:
        value = input(full_message)
    value = value.strip().strip('"')
    return value if value else default


def prompt_int(message: str, default: int, minimum: int = 1) -> int:
    while True:
        raw = prompt_text(message, str(default))
        try:
            return max(int(minimum), int(raw))
        except ValueError:
            print(f"Enter an integer >= {minimum}.")


def prompt_float(message: str, default: float, minimum: float = 0.0, maximum: float | None = None) -> float:
    while True:
        raw = prompt_text(message, str(default))
        try:
            value = max(float(minimum), float(raw))
            return min(value, float(maximum)) if maximum is not None else value
        except ValueError:
            print(f"Enter a number >= {minimum}.")


def configure_continuation_generation(args: argparse.Namespace) -> dict[str, float | int]:
    params: dict[str, float | int] = {
        "seed_tokens": int(args.seed_tokens),
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k),
    }
    print("\nContinuation generation parameters")
    while True:
        for key, value in params.items():
            print(f"  {key}: {value}")
        action = prompt_text("Press Enter to accept, type a parameter name to edit, or q to quit", "")
        if not action:
            return params
        if action.lower() in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        if action not in params:
            print(f"Unknown parameter {action!r}. Choices: {', '.join(params)}")
            continue
        if action in {"seed_tokens", "max_new_tokens", "top_k"}:
            params[action] = prompt_int(action, int(params[action]), minimum=1)
        elif action == "top_p":
            params[action] = prompt_float(action, float(params[action]), minimum=0.01, maximum=1.0)
        else:
            params[action] = prompt_float(action, float(params[action]), minimum=0.01)


def continuation_output_path(output_dir: Path, checkpoint_stem: str, seed_path: Path) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    safe_seed = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in seed_path.stem)
    return output_dir / "continuation" / checkpoint_stem / f"{safe_seed}-{timestamp}.mid"


def prepare_seed_tokens(tokenizer: CustomDeltaTokenizer, seed_path: Path, seed_tokens: int) -> list[int]:
    tokens = [int(token) for token in tokenizer.encode(seed_path)]
    if not tokens:
        raise RuntimeError(f"Seed MIDI produced no tokens: {seed_path}")
    event_size = max(1, int(tokenizer.event_size))
    keep = min(len(tokens), max(event_size, int(seed_tokens)))
    keep -= keep % event_size
    keep = max(event_size, keep)
    return tokens[-keep:]


def generate_continuation_once(
    *,
    model,
    tokenizer: CustomDeltaTokenizer,
    seed_path: Path,
    output_dir: Path,
    checkpoint_stem: str,
    params: dict[str, float | int],
) -> Path:
    seed_tokens = prepare_seed_tokens(tokenizer, seed_path, int(params["seed_tokens"]))
    generated = model.generate(
        seed_tokens=seed_tokens,
        max_new_tokens=int(params["max_new_tokens"]),
        temperature=float(params["temperature"]),
        top_p=float(params["top_p"]),
        top_k=int(params["top_k"]),
    )
    output_path = continuation_output_path(output_dir, checkpoint_stem, seed_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.decode(generated, output_path=output_path)
    return output_path


def run_continuation_generation(args: argparse.Namespace, checkpoint: dict, device: torch.device) -> None:
    config = continuation_config_from_dict(checkpoint["config"])
    model = build_continuation_model(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    tokenizer_path = args.tokenizer_path if args.tokenizer_path is not None else Path("custom_tokenizer.json")
    tokenizer = CustomDeltaTokenizer.load(str(tokenizer_path))
    print(
        "Generation model: "
        f"{config.model_type} continuation "
        f"(layers={config.n_layers}, d_model={config.d_model}, heads={config.n_heads}, "
        f"vocab={config.vocab_size})",
        flush=True,
    )
    print(f"Tokenizer: {tokenizer_path} (vocab={tokenizer.vocab_size}, event_size={tokenizer.event_size})")
    params = configure_continuation_generation(args) if args.interactive else {
        "seed_tokens": int(args.seed_tokens),
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k),
    }
    output_dir = args.output_dir
    if args.seed_midi is not None:
        output_path = generate_continuation_once(
            model=model,
            tokenizer=tokenizer,
            seed_path=args.seed_midi,
            output_dir=output_dir,
            checkpoint_stem=args.checkpoint.stem,
            params=params,
        )
        print(output_path)
        return

    print("\nEnter a seed MIDI path. Use Tab for path completion when prompt_toolkit is installed.")
    print("Type q to quit.\n")
    while True:
        raw_seed = prompt_text("Seed MIDI", path=True)
        if raw_seed.lower() in {"q", "quit", "exit"}:
            return
        if not raw_seed:
            continue
        seed_path = Path(raw_seed).expanduser()
        if not seed_path.exists():
            print(f"Seed path not found: {seed_path}")
            continue
        try:
            output_path = generate_continuation_once(
                model=model,
                tokenizer=tokenizer,
                seed_path=seed_path,
                output_dir=output_dir,
                checkpoint_stem=args.checkpoint.stem,
                params=params,
            )
        except Exception as exc:
            print(f"Generation failed: {type(exc).__name__}: {exc}")
            continue
        print(f"Wrote {output_path}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--output_dir", type=Path, default=Path("samples"))
    parser.add_argument("--max_len", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--seed_tokens", type=int, default=1024)
    parser.add_argument("--seed_midi", type=Path, default=None)
    parser.add_argument("--tokenizer_path", type=Path, default=None)
    parser.add_argument("--interactive", action=argparse.BooleanOptionalAction, default=True)
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
    checkpoint = load_checkpoint_portable(args.checkpoint, device)
    raw_model_type = str(checkpoint.get("config", {}).get("model_type", "")).upper()
    if raw_model_type in {"G_CONT", "H_CONT"}:
        run_continuation_generation(args, checkpoint, device)
        return
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
