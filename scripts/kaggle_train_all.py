"""Run the full POEM Kaggle training workflow.

This script is meant for a Kaggle notebook with two attached datasets:

- the POEM code repository mirror
- the Beautiful-Motifs MIDI dataset

It trains D, C, E, B, and A in sequence, writes checkpoint/metric artifacts,
generates five MIDI samples per completed candidate, and uploads artifacts to a
Hugging Face model repository when HF credentials are provided.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


MODEL_ORDER = ["D", "C", "E", "B", "A"]


def run(command: list[str], cwd: Path) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def upload_file(path: Path, repo_id: str | None, token: str | None, path_in_repo: str) -> None:
    if not repo_id or not token or not path.exists():
        return
    from huggingface_hub import HfApi

    HfApi(token=token).upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path_in_repo.replace("\\", "/"),
        repo_id=repo_id,
        repo_type="model",
    )


def upload_folder_files(folder: Path, repo_id: str | None, token: str | None, path_in_repo: str) -> None:
    if not folder.exists():
        return
    for path in folder.rglob("*"):
        if path.is_file():
            upload_file(path, repo_id, token, f"{path_in_repo}/{path.relative_to(folder).as_posix()}")


def write_model_card(path: Path, repo_id: str, epochs: int, model_order: list[str]) -> None:
    text = f"""# POEM

This repository stores POEM symbolic melody model checkpoints, metrics, and generated MIDI samples.

## Training

- Dataset: Beautiful-Motifs short motifs
- Epochs per candidate: {epochs}
- Candidate order: {", ".join(model_order)}
- Metrics: per-candidate `metrics/summary.json`, `train_history.json`, `val_history.json`, and checkpoint-level JSON files
- Samples: five MIDI generations per completed candidate under `samples/`

## Layout

```text
poem-a/
  checkpoints/
  metrics/
  samples/
poem-b/
...
comparison/summary.json
```
"""
    path.write_text(text, encoding="utf-8")


def latest_checkpoint(model_dir: Path, model_type: str) -> Path:
    best = model_dir / f"poem-{model_type.lower()}-best.pt"
    final = model_dir / f"poem-{model_type.lower()}-final.pt"
    if final.exists():
        return final
    if best.exists():
        return best
    candidates = sorted(model_dir.glob(f"poem-{model_type.lower()}-*.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found for model {model_type} in {model_dir}")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_dir", type=Path, default=Path.cwd())
    parser.add_argument("--data_dir", required=True, type=Path)
    parser.add_argument("--hf_repo_id", required=True)
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--variants", nargs="+", default=MODEL_ORDER)
    parser.add_argument("--cache_path", type=Path, default=Path("/kaggle/working/cache/poem-short-token-cache.pt"))
    parser.add_argument("--output_dir", type=Path, default=Path("/kaggle/working/checkpoints"))
    parser.add_argument("--metrics_dir", type=Path, default=Path("/kaggle/working/metrics"))
    parser.add_argument("--samples_dir", type=Path, default=Path("/kaggle/working/samples"))
    parser.add_argument("--val_interval", type=int, default=2000)
    parser.add_argument("--checkpoint_interval_steps", type=int, default=5000)
    parser.add_argument("--checkpoint_interval_minutes", type=float, default=20.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_hours", type=float, default=11.75)
    parser.add_argument("--samples_per_model", type=int, default=5)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    if not args.hf_token:
        raise RuntimeError("HF token is required. Set HF_TOKEN or pass --hf_token.")

    from huggingface_hub import HfApi

    api = HfApi(token=args.hf_token)
    api.create_repo(args.hf_repo_id, repo_type="model", private=args.private, exist_ok=True)
    args.cache_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    args.samples_dir.mkdir(parents=True, exist_ok=True)

    card_path = Path("/kaggle/working/POEM_MODEL_CARD.md")
    write_model_card(card_path, args.hf_repo_id, args.epochs, args.variants)
    upload_file(card_path, args.hf_repo_id, args.hf_token, "README.md")

    if not args.cache_path.exists():
        run(
            [
                sys.executable,
                "-u",
                "scripts/pretokenize.py",
                "--data_dir",
                str(args.data_dir),
                "--output",
                str(args.cache_path),
                "--log_interval",
                "5000",
            ],
            args.repo_dir,
        )
    upload_file(args.cache_path, args.hf_repo_id, args.hf_token, "data/poem-short-token-cache.pt")

    start = time.time()
    completed: list[dict] = []
    for model_type in args.variants:
        elapsed_hours = (time.time() - start) / 3600.0
        if elapsed_hours >= args.max_hours:
            print(f"Stopping before {model_type}; max_hours={args.max_hours} reached.", flush=True)
            break

        run(
            [
                sys.executable,
                "-u",
                "train.py",
                "--model_type",
                model_type,
                "--data_dir",
                str(args.data_dir),
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--token_cache",
                str(args.cache_path),
                "--output_dir",
                str(args.output_dir),
                "--metrics_dir",
                str(args.metrics_dir),
                "--val_interval",
                str(args.val_interval),
                "--checkpoint_interval_steps",
                str(args.checkpoint_interval_steps),
                "--checkpoint_interval_minutes",
                str(args.checkpoint_interval_minutes),
                "--device",
                "cuda",
                "--amp",
                "--amp_dtype",
                "float16",
                "--data_parallel",
                "--num_workers",
                str(args.num_workers),
                "--hf_repo_id",
                args.hf_repo_id,
                *(["--hf_private"] if args.private else []),
            ],
            args.repo_dir,
        )

        model_name = f"poem-{model_type.lower()}"
        ckpt = latest_checkpoint(args.output_dir / model_name, model_type)
        model_samples_dir = args.samples_dir / model_name
        run(
            [
                sys.executable,
                "-u",
                "generate.py",
                "--checkpoint",
                str(ckpt),
                "--num_samples",
                str(args.samples_per_model),
                "--output_dir",
                str(model_samples_dir),
            ],
            args.repo_dir,
        )
        upload_folder_files(model_samples_dir, args.hf_repo_id, args.hf_token, f"{model_name}/samples")
        summary_path = args.metrics_dir / model_name / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        completed.append({"model_type": model_type, "checkpoint": str(ckpt), "summary": summary})

        comparison_dir = Path("/kaggle/working/comparison")
        comparison_dir.mkdir(parents=True, exist_ok=True)
        comparison_path = comparison_dir / "summary.json"
        comparison_path.write_text(json.dumps({"completed": completed}, indent=2), encoding="utf-8")
        upload_file(comparison_path, args.hf_repo_id, args.hf_token, "comparison/summary.json")


if __name__ == "__main__":
    main()
