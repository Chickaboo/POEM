"""Kaggle launcher for Pulse88 POEM continuation models."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_VARIANTS = ["G_CONT", "H_CONT"]


def run(command: list[str], cwd: Path) -> None:
    print("\n$ " + " ".join(command), flush=True)
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    subprocess.run(command, cwd=cwd, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_dir", type=Path, default=Path.cwd())
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("/kaggle/input/datasets/chickaboomcmurtrie/pulse88-tokenized-500k"),
    )
    parser.add_argument("--hf_repo_id", required=True)
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accumulation_steps", type=int, default=4)
    parser.add_argument("--seed_length", type=int, default=1024)
    parser.add_argument("--continuation_length", type=int, default=1024)
    parser.add_argument("--max_pieces", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--max_hours", type=float, default=11.75)
    parser.add_argument("--output_dir", type=Path, default=Path("/kaggle/working/continuation_checkpoints"))
    parser.add_argument("--metrics_dir", type=Path, default=Path("/kaggle/working/continuation_metrics"))
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--val_interval", type=int, default=1000)
    parser.add_argument("--checkpoint_interval", type=int, default=2500)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    if not args.hf_token:
        raise RuntimeError("HF token is required. Set HF_TOKEN or pass --hf_token.")
    if not args.data_root.exists():
        raise FileNotFoundError(f"Pulse88 tokenized data root not found: {args.data_root}")

    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required in the Kaggle notebook.") from exc

    HfApi(token=args.hf_token).create_repo(
        repo_id=args.hf_repo_id,
        repo_type="model",
        private=bool(args.private),
        exist_ok=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    for variant in args.variants:
        elapsed_hours = (time.time() - start) / 3600.0
        if elapsed_hours >= float(args.max_hours):
            print(f"Stopping before {variant}; max_hours={args.max_hours} reached.", flush=True)
            break
        command = [
            sys.executable,
            "-u",
            "train_continuation.py",
            "--model_type",
            str(variant),
            "--data_root",
            str(args.data_root),
            "--output_dir",
            str(args.output_dir),
            "--metrics_dir",
            str(args.metrics_dir),
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--grad_accumulation_steps",
            str(args.grad_accumulation_steps),
            "--seed_length",
            str(args.seed_length),
            "--continuation_length",
            str(args.continuation_length),
            "--max_pieces",
            str(args.max_pieces),
            "--max_steps",
            str(args.max_steps),
            "--num_workers",
            str(args.num_workers),
            "--val_interval",
            str(args.val_interval),
            "--checkpoint_interval",
            str(args.checkpoint_interval),
            "--device",
            "cuda",
            "--amp",
            "--amp_dtype",
            "float16",
            "--data_parallel",
            "--hf_repo_id",
            str(args.hf_repo_id),
            "--hf_token",
            str(args.hf_token),
        ]
        run(command, args.repo_dir)


if __name__ == "__main__":
    main()
