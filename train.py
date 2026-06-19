"""Train POEM candidate backbones."""

from __future__ import annotations

import argparse
import json
import math
import os
import itertools
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.build import build_model, count_parameters
from models.config import config_for_model_type
from models.embeddings import fielded_loss_breakdown
from tokenizer.vocab import TYPE_PAD
from training.compute_budget import DEFAULT_TRAIN_EPOCHS, format_budget_plan, plan_epoch_budget
from training.data import (
    MIDITokenDataset,
    collate_token_sequences,
    count_tokenized_events,
    discover_motif_files,
    load_token_cache,
    split_files,
)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def is_main_process() -> bool:
    return True


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu()) if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def args_payload(args: argparse.Namespace) -> dict:
    payload = vars(args).copy()
    if payload.get("hf_token"):
        payload["hf_token"] = "***"
    return payload


def maybe_upload_file(path: Path, repo_id: str | None, token: str | None, path_in_repo: str) -> None:
    if not repo_id or not token or not path.exists():
        return
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        print(f"Hugging Face upload skipped; huggingface_hub unavailable: {exc}", flush=True)
        return
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path_in_repo.replace("\\", "/"),
        repo_id=repo_id,
        repo_type="model",
    )


def maybe_create_hf_repo(repo_id: str | None, token: str | None, private: bool) -> None:
    if not repo_id or not token:
        return
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        print(f"Hugging Face repo setup skipped; huggingface_hub unavailable: {exc}", flush=True)
        return
    HfApi(token=token).create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lowered = name.lower()
        if param.ndim < 2 or "norm" in lowered or "emb" in lowered or lowered.endswith("bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    if args.lr_schedule == "none":
        return None
    warmup_steps = args.warmup_steps
    if warmup_steps <= 0:
        warmup_steps = int(total_steps * args.warmup_ratio)
    warmup_steps = max(1, warmup_steps)
    total_steps = max(total_steps, warmup_steps + 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    max_batches: int = 20,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    breakdown_totals: dict[str, float] = {}
    breakdown_count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            batch = batch.to(device, non_blocking=True)
            if batch.size(1) < 2:
                continue
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                output = model(batch[:, :-1], batch[:, 1:])
            if output.loss is not None:
                loss_tensor = output.loss.mean() if output.loss.ndim > 0 else output.loss
                losses.append(float(loss_tensor.detach()))
                breakdown = fielded_loss_breakdown(output.logits, batch[:, 1:])
                for key, value in breakdown.items():
                    breakdown_totals[key] = breakdown_totals.get(key, 0.0) + float(value)
                breakdown_count += 1
            if batch_index + 1 >= max_batches:
                break
    model.train()
    metrics = {key: value / max(1, breakdown_count) for key, value in breakdown_totals.items()}
    metrics["loss"] = sum(losses) / max(1, len(losses))
    metrics["ppl"] = math.exp(min(metrics["loss"], 20.0))
    return metrics


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config,
    step: int,
    val_loss: float | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": unwrap_model(model).state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": vars(config),
            "step": step,
            "val_loss": val_loss,
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    torch.set_num_threads(max(1, min(args.threads, torch.get_num_threads())))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    config = config_for_model_type(args.model_type, smoke_test=args.smoke_test)
    model = build_model(config)
    param_count = count_parameters(model)
    model.to(device)
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"Using DataParallel on {torch.cuda.device_count()} CUDA devices", flush=True)
    print(f"Device: {device}, amp={use_amp}, amp_dtype={args.amp_dtype}", flush=True)

    files = discover_motif_files(args.data_dir, include_long_motifs=args.include_long_motifs)
    train_files, val_files = split_files(files, val_fraction=args.val_fraction, seed=args.seed, smoke_test=args.smoke_test)
    token_cache = load_token_cache(args.token_cache if args.token_cache.exists() else None)
    print(f"POEM candidate {config.model_type}: {param_count:,} trainable parameters", flush=True)
    print(f"Files: train={len(train_files)}, val={len(val_files)}, include_long={args.include_long_motifs}", flush=True)
    if token_cache:
        print(f"Loaded pretokenized cache: {args.token_cache} ({len(token_cache)} files)", flush=True)
    else:
        print("No pretokenized cache loaded; falling back to MIDI parsing.", flush=True)
    print("Counting tokenized events for the chosen training split...", flush=True)
    dataset_tokens = count_tokenized_events(train_files, max_seq_len=config.max_seq_len, token_cache=token_cache)
    selected_epochs = 1 if args.smoke_test else args.epochs
    plan = plan_epoch_budget(dataset_tokens, param_count, selected_epochs=selected_epochs)
    if args.smoke_test:
        print("Smoke test mode: using tiny model/data and a fixed short step budget.", flush=True)
    print(format_budget_plan(plan), flush=True)
    maybe_create_hf_repo(args.hf_repo_id, args.hf_token, args.hf_private)

    train_dataset = MIDITokenDataset(
        train_files,
        pitch_transpose_aug=args.pitch_transpose_aug,
        tempo_aug=args.tempo_aug,
        cache_tokens=True,
        max_seq_len=config.max_seq_len,
        token_cache=token_cache,
    )
    val_dataset = MIDITokenDataset(val_files, cache_tokens=True, max_seq_len=config.max_seq_len, token_cache=token_cache)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_token_sequences,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_token_sequences,
    )

    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and args.amp_dtype == "float16")
    max_steps = args.max_steps
    if max_steps is None and args.smoke_test:
        max_steps = 80
    planned_epochs = plan.selected_epochs
    best_val = float("inf")
    start_time = time.time()
    last_checkpoint_time = start_time
    recent_losses: list[float] = []
    first_losses: list[float] = []
    tokens_seen = 0
    step = 0
    train_history: list[dict] = []
    val_history: list[dict] = []
    checkpoint_history: list[dict] = []
    model_dir_name = f"poem-{config.model_type.lower()}"
    metrics_dir = args.metrics_dir / model_dir_name
    checkpoint_dir = args.output_dir / model_dir_name
    total_steps = max_steps if max_steps is not None else planned_epochs * max(1, len(train_loader))
    scheduler = build_lr_scheduler(optimizer, args, total_steps)
    print(
        f"Optimizer: AdamW lr={args.lr}, matrix_weight_decay={args.weight_decay}, "
        f"schedule={args.lr_schedule}, total_steps={total_steps}",
        flush=True,
    )

    for epoch in range(planned_epochs):
        epoch_start = time.time()
        for batch in train_loader if max_steps is None else itertools.cycle(train_loader):
            batch = batch.to(device, non_blocking=True)
            if batch.size(1) < 2:
                continue
            step += 1
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                output = model(batch[:, :-1], batch[:, 1:])
            if output.loss is None:
                raise RuntimeError("Model did not return a training loss")
            loss_tensor = output.loss.mean() if output.loss.ndim > 0 else output.loss
            if scaler.is_enabled():
                scaler.scale(loss_tensor).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_tensor.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

            token_count = int((batch[:, 1:, 0] != TYPE_PAD).sum().item())
            tokens_seen += token_count
            loss_value = float(loss_tensor.detach())
            current_lr = float(optimizer.param_groups[0]["lr"])
            recent_losses.append(loss_value)
            if len(first_losses) < 10:
                first_losses.append(loss_value)
            if len(recent_losses) > 20:
                recent_losses.pop(0)

            elapsed = max(time.time() - start_time, 1e-6)
            tokens_per_second = tokens_seen / elapsed
            if step % args.log_interval == 0 or step == 1:
                metrics = output.metrics or {}
                loops = metrics.get("avg_loops_per_token")
                loop_text = f", loops/token={float(loops):.2f}" if loops is not None else ""
                train_record = {
                    "model_type": config.model_type,
                    "step": step,
                    "epoch": epoch + 1,
                    "planned_epochs": planned_epochs,
                    "train_loss": loss_value,
                    "train_ppl": math.exp(min(loss_value, 20.0)),
                    "tokens_seen": tokens_seen,
                    "tokens_per_second": tokens_per_second,
                    "elapsed_seconds": elapsed,
                    "avg_loops_per_token": float(loops) if loops is not None else None,
                    "lr": current_lr,
                }
                train_history.append(train_record)
                print(
                    f"step={step} epoch={epoch + 1}/{planned_epochs} "
                    f"loss={loss_value:.4f} lr={current_lr:.2e} tokens/sec={tokens_per_second:.1f}{loop_text}",
                    flush=True,
                )

            if step % args.val_interval == 0:
                val_metrics = evaluate(model, val_loader, device=device, use_amp=use_amp, amp_dtype=amp_dtype)
                val_loss = val_metrics["loss"]
                val_record = {
                    "model_type": config.model_type,
                    "step": step,
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "val_ppl": val_metrics["ppl"],
                    "elapsed_seconds": time.time() - start_time,
                    "field_metrics": val_metrics,
                }
                val_history.append(val_record)
                print(f"validation step={step} val_loss={val_loss:.4f}", flush=True)
                if val_loss < best_val:
                    best_val = val_loss
                    ckpt_path = checkpoint_dir / f"{model_dir_name}-best.pt"
                    save_checkpoint(ckpt_path, model, optimizer, config, step, val_loss)
                    write_checkpoint_metrics(
                        metrics_dir,
                        checkpoint_history,
                        ckpt_path,
                        config.model_type,
                        step,
                        epoch + 1,
                        loss_value,
                        val_loss,
                        tokens_seen,
                        tokens_per_second,
                        start_time,
                        train_history,
                        val_history,
                        args,
                    )
                    upload_artifacts(args, ckpt_path, metrics_dir, model_dir_name)

            now = time.time()
            if step % args.checkpoint_interval_steps == 0 or (
                now - last_checkpoint_time >= args.checkpoint_interval_minutes * 60.0
            ):
                ckpt_path = checkpoint_dir / f"{model_dir_name}-{step}.pt"
                save_checkpoint(ckpt_path, model, optimizer, config, step, best_val if best_val < float("inf") else None)
                write_checkpoint_metrics(
                    metrics_dir,
                    checkpoint_history,
                    ckpt_path,
                    config.model_type,
                    step,
                    epoch + 1,
                    loss_value,
                    best_val if best_val < float("inf") else None,
                    tokens_seen,
                    tokens_per_second,
                    start_time,
                    train_history,
                    val_history,
                    args,
                )
                upload_artifacts(args, ckpt_path, metrics_dir, model_dir_name)
                last_checkpoint_time = now

            if max_steps is not None and step >= max_steps:
                first = sum(first_losses) / max(1, len(first_losses))
                last = sum(recent_losses[-10:]) / max(1, min(10, len(recent_losses)))
                print(f"smoke summary: first_loss_avg={first:.4f}, last_loss_avg={last:.4f}", flush=True)
                return
        if max_steps is None:
            val_metrics = evaluate(model, val_loader, device=device, use_amp=use_amp, amp_dtype=amp_dtype)
            val_loss = val_metrics["loss"]
            val_history.append(
                {
                    "model_type": config.model_type,
                    "step": step,
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "val_ppl": val_metrics["ppl"],
                    "elapsed_seconds": time.time() - start_time,
                    "field_metrics": val_metrics,
                }
            )
            print(f"end epoch={epoch + 1} val_loss={val_loss:.4f}", flush=True)
            if val_loss < best_val:
                best_val = val_loss
                ckpt_path = checkpoint_dir / f"{model_dir_name}-best.pt"
                save_checkpoint(ckpt_path, model, optimizer, config, step, val_loss)
                write_checkpoint_metrics(
                    metrics_dir,
                    checkpoint_history,
                    ckpt_path,
                    config.model_type,
                    step,
                    epoch + 1,
                    train_history[-1]["train_loss"] if train_history else float("nan"),
                    val_loss,
                    tokens_seen,
                    tokens_seen / max(time.time() - start_time, 1e-6),
                    start_time,
                    train_history,
                    val_history,
                    args,
                )
                upload_artifacts(args, ckpt_path, metrics_dir, model_dir_name)
            write_summary_metrics(
                metrics_dir,
                checkpoint_history,
                config.model_type,
                step,
                epoch + 1,
                best_val if best_val < float("inf") else val_loss,
                tokens_seen,
                start_time,
                train_history,
                val_history,
                args,
            )

    if max_steps is None:
        final_path = checkpoint_dir / f"{model_dir_name}-final.pt"
        save_checkpoint(final_path, model, optimizer, config, step, best_val if best_val < float("inf") else None)
        write_checkpoint_metrics(
            metrics_dir,
            checkpoint_history,
            final_path,
            config.model_type,
            step,
            planned_epochs,
            train_history[-1]["train_loss"] if train_history else float("nan"),
            best_val if best_val < float("inf") else None,
            tokens_seen,
            tokens_seen / max(time.time() - start_time, 1e-6),
            start_time,
            train_history,
            val_history,
            args,
        )
        upload_artifacts(args, final_path, metrics_dir, model_dir_name)


def write_checkpoint_metrics(
    metrics_dir: Path,
    checkpoint_history: list[dict],
    ckpt_path: Path,
    model_type: str,
    step: int,
    epoch: int,
    train_loss: float,
    val_loss: float | None,
    tokens_seen: int,
    tokens_per_second: float,
    start_time: float,
    train_history: list[dict],
    val_history: list[dict],
    args: argparse.Namespace,
) -> None:
    payload = {
        "model_type": model_type,
        "checkpoint": str(ckpt_path),
        "step": step,
        "epoch": epoch,
        "train_loss": train_loss,
        "train_ppl": math.exp(min(train_loss, 20.0)) if math.isfinite(train_loss) else None,
        "val_loss": val_loss,
        "val_ppl": math.exp(min(val_loss, 20.0)) if val_loss is not None and math.isfinite(val_loss) else None,
        "tokens_seen": tokens_seen,
        "tokens_per_second": tokens_per_second,
        "elapsed_seconds": time.time() - start_time,
        "args": args_payload(args),
    }
    checkpoint_history.append(payload)
    write_json(metrics_dir / "checkpoints" / f"{ckpt_path.stem}.json", payload)
    write_summary_metrics(
        metrics_dir,
        checkpoint_history,
        model_type,
        step,
        epoch,
        val_loss,
        tokens_seen,
        start_time,
        train_history,
        val_history,
        args,
    )


def write_summary_metrics(
    metrics_dir: Path,
    checkpoint_history: list[dict],
    model_type: str,
    step: int,
    epoch: int,
    best_val_loss: float | None,
    tokens_seen: int,
    start_time: float,
    train_history: list[dict],
    val_history: list[dict],
    args: argparse.Namespace,
) -> None:
    write_json(metrics_dir / "train_history.json", {"records": train_history})
    write_json(metrics_dir / "val_history.json", {"records": val_history})
    write_json(
        metrics_dir / "summary.json",
        {
            "model_type": model_type,
            "step": step,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "best_val_ppl": math.exp(min(best_val_loss, 20.0)) if best_val_loss is not None and math.isfinite(best_val_loss) else None,
            "tokens_seen": tokens_seen,
            "elapsed_seconds": time.time() - start_time,
            "checkpoints": checkpoint_history,
            "args": args_payload(args),
        },
    )


def upload_artifacts(args: argparse.Namespace, ckpt_path: Path, metrics_dir: Path, model_dir_name: str) -> None:
    if not args.hf_repo_id or not args.hf_token:
        return
    base = f"{model_dir_name}"
    maybe_upload_file(ckpt_path, args.hf_repo_id, args.hf_token, f"{base}/checkpoints/{ckpt_path.name}")
    for metrics_file in metrics_dir.rglob("*.json"):
        rel = metrics_file.relative_to(metrics_dir).as_posix()
        maybe_upload_file(metrics_file, args.hf_repo_id, args.hf_token, f"{base}/metrics/{rel}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_type", choices=["A", "B", "C", "D", "E"], default="D")
    parser.add_argument("--data_dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAIN_EPOCHS)
    parser.add_argument("--include_long_motifs", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--pitch_transpose_aug", action="store_true")
    parser.add_argument("--tempo_aug", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--lr_schedule", choices=["cosine", "none"], default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--min_lr_ratio", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--val_fraction", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--val_interval", type=int, default=2000)
    parser.add_argument("--checkpoint_interval_steps", type=int, default=5000)
    parser.add_argument("--checkpoint_interval_minutes", type=float, default=20.0)
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--token_cache", type=Path, default=Path("cache/poem-short-token-cache.pt"))
    parser.add_argument("--metrics_dir", type=Path, default=Path("metrics"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--data_parallel", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--hf_repo_id", default=None)
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--hf_private", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
