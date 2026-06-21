"""Run the full POEM Kaggle training workflow.

This script is meant for a Kaggle notebook with two attached datasets:

- the POEM-BASE code repository mirror
- the Beautiful-Motifs MIDI dataset

It trains H and G by default: the HRM dense RoPE candidate plus its matched
non-HRM dense RoPE control. It writes
checkpoint/metric artifacts locally, generates five MIDI samples per completed
candidate, and uploads each completed candidate folder to Hugging Face in a
single commit to avoid Hub commit rate limits.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


MODEL_ORDER = ["H", "G"]
DEFAULT_BATCH_BY_MODEL = {
    "A": 32,
    "B": 64,
    "C": 256,
    "D": 256,
    "E": 256,
    "F": 128,
    "G": 64,
    "G_MTP": 64,
    "H": 64,
}


def run(command: list[str], cwd: Path) -> None:
    print("\n$ " + " ".join(command), flush=True)
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TRITON_CACHE_AUTOTUNING", "0")
    env.setdefault("FLA_CACHE_MODE", "disabled")
    subprocess.run(command, cwd=cwd, check=True, env=env)


def batch_size_for_model(args: argparse.Namespace, model_type: str) -> int:
    overrides = {
        "A": args.batch_size_a,
        "B": args.batch_size_b,
        "C": args.batch_size_c,
        "D": args.batch_size_d,
        "E": args.batch_size_e,
        "F": args.batch_size_f,
        "G": args.batch_size_g,
        "G_MTP": args.batch_size_g_mtp,
        "H": args.batch_size_h,
    }
    override = overrides.get(model_type)
    if override is not None:
        return override
    if args.batch_size is not None:
        return args.batch_size
    return DEFAULT_BATCH_BY_MODEL[model_type]


def upload_single_commit(
    folder: Path,
    repo_id: str | None,
    token: str | None,
    path_in_repo: str,
    commit_message: str,
) -> None:
    if not repo_id or not token or not folder.exists():
        return
    from huggingface_hub import HfApi

    HfApi(token=token).upload_folder(
        folder_path=str(folder),
        path_in_repo=path_in_repo.replace("\\", "/"),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )


def stage_model_artifacts(
    staging_root: Path,
    model_name: str,
    checkpoint_dir: Path,
    metrics_dir: Path,
    samples_dir: Path,
) -> Path:
    import shutil

    target = staging_root / model_name
    if target.exists():
        shutil.rmtree(target)
    (target / "checkpoints").mkdir(parents=True, exist_ok=True)
    (target / "metrics").mkdir(parents=True, exist_ok=True)
    (target / "samples").mkdir(parents=True, exist_ok=True)
    if checkpoint_dir.exists():
        shutil.copytree(checkpoint_dir, target / "checkpoints", dirs_exist_ok=True)
    if metrics_dir.exists():
        shutil.copytree(metrics_dir, target / "metrics", dirs_exist_ok=True)
    if samples_dir.exists():
        shutil.copytree(samples_dir, target / "samples", dirs_exist_ok=True)
    return target


def write_model_card(path: Path, repo_id: str, epochs: int, model_order: list[str]) -> None:
    text = f"""# POEM-BASE

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
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--batch_size_a", type=int, default=32)
    parser.add_argument("--batch_size_b", type=int, default=64)
    parser.add_argument("--batch_size_c", type=int, default=256)
    parser.add_argument("--batch_size_d", type=int, default=256)
    parser.add_argument("--batch_size_e", type=int, default=256)
    parser.add_argument("--batch_size_f", type=int, default=128)
    parser.add_argument("--batch_size_g", type=int, default=64)
    parser.add_argument("--batch_size_g_mtp", type=int, default=None)
    parser.add_argument("--batch_size_h", type=int, default=64)
    parser.add_argument("--batch_size_f_single_gpu", type=int, default=64)
    parser.add_argument("--variants", nargs="+", default=MODEL_ORDER)
    parser.add_argument("--cache_path", type=Path, default=Path("/kaggle/working/cache/poem-short-token-cache.pt"))
    parser.add_argument("--output_dir", type=Path, default=Path("/kaggle/working/checkpoints"))
    parser.add_argument("--metrics_dir", type=Path, default=Path("/kaggle/working/metrics"))
    parser.add_argument("--samples_dir", type=Path, default=Path("/kaggle/working/samples"))
    parser.add_argument("--staging_dir", type=Path, default=Path("/kaggle/working/hf_upload"))
    parser.add_argument("--val_interval", type=int, default=2000)
    parser.add_argument("--checkpoint_interval_steps", type=int, default=5000)
    parser.add_argument("--checkpoint_interval_minutes", type=float, default=20.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pretokenize_workers", type=int, default=8)
    parser.add_argument("--max_hours", type=float, default=11.75)
    parser.add_argument("--samples_per_model", type=int, default=5)
    parser.add_argument("--mtp_horizon", type=int, default=4)
    parser.add_argument("--mtp_aux_weight", type=float, default=0.3)
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
    args.staging_dir.mkdir(parents=True, exist_ok=True)

    card_path = Path("/kaggle/working/POEM_MODEL_CARD.md")
    write_model_card(card_path, args.hf_repo_id, args.epochs, args.variants)
    readme_stage = args.staging_dir / "_repo_readme"
    readme_stage.mkdir(parents=True, exist_ok=True)
    (readme_stage / "README.md").write_text(card_path.read_text(encoding="utf-8"), encoding="utf-8")
    upload_single_commit(readme_stage, args.hf_repo_id, args.hf_token, ".", "Update POEM-BASE README")

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
                "--workers",
                str(args.pretokenize_workers),
            ],
            args.repo_dir,
        )
    # Do not upload the token cache here; it is large and not needed for model comparison artifacts.

    start = time.time()
    completed: list[dict] = []
    for model_type in args.variants:
        elapsed_hours = (time.time() - start) / 3600.0
        if elapsed_hours >= args.max_hours:
            print(f"Stopping before {model_type}; max_hours={args.max_hours} reached.", flush=True)
            break
        model_batch_size = batch_size_for_model(args, model_type)
        if model_type.upper() == "F" and model_batch_size > args.batch_size_f_single_gpu:
            print(
                f"Reducing candidate F batch_size from {model_batch_size} to "
                f"{args.batch_size_f_single_gpu} because FLA GDN runs in single-GPU mode.",
                flush=True,
            )
            model_batch_size = args.batch_size_f_single_gpu
        print(f"Training candidate {model_type} with batch_size={model_batch_size}", flush=True)

        train_command = [
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
            str(model_batch_size),
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
        ]
        if model_type.upper() == "F":
            train_command.append("--require_flash_gdn")
        if model_type.upper() == "G_MTP":
            train_command.extend(["--mtp_horizon", str(args.mtp_horizon)])
            train_command.extend(["--mtp_aux_weight", str(args.mtp_aux_weight)])
        run(train_command, args.repo_dir)

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
        summary_path = args.metrics_dir / model_name / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        completed.append({"model_type": model_type, "checkpoint": str(ckpt), "summary": summary})
        staged_model = stage_model_artifacts(
            args.staging_dir,
            model_name,
            args.output_dir / model_name,
            args.metrics_dir / model_name,
            model_samples_dir,
        )
        upload_single_commit(
            staged_model,
            args.hf_repo_id,
            args.hf_token,
            model_name,
            f"Upload completed {model_name}",
        )

        comparison_dir = Path("/kaggle/working/comparison")
        comparison_dir.mkdir(parents=True, exist_ok=True)
        comparison_path = comparison_dir / "summary.json"
        comparison_path.write_text(json.dumps({"completed": completed}, indent=2), encoding="utf-8")
        comparison_stage = args.staging_dir / "comparison"
        comparison_stage.mkdir(parents=True, exist_ok=True)
        (comparison_stage / "summary.json").write_text(comparison_path.read_text(encoding="utf-8"), encoding="utf-8")
        upload_single_commit(
            comparison_stage,
            args.hf_repo_id,
            args.hf_token,
            "comparison",
            "Update comparison summary",
        )


if __name__ == "__main__":
    main()
