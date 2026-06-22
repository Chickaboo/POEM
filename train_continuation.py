"""Train Pulse88 symbolic continuation models with POEM G/H backbones."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.continuation import (
    build_continuation_model,
    continuation_accuracy,
    continuation_config_for_model_type,
    continuation_targets,
    count_parameters,
)
from training.continuation_data import (
    Pulse88ContinuationDataset,
    collate_fixed_windows,
    load_pulse88_manifest,
    train_val_split,
)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lowered = name.lower()
        if param.ndim < 2 or "norm" in lowered or lowered.endswith("bias") or "embed" in lowered:
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_ratio: float,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, int(round(int(total_steps) * float(warmup_ratio))))
    total_steps = max(int(total_steps), warmup_steps + 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, float(step + 1) / float(warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return float(min_lr_ratio) + (1.0 - float(min_lr_ratio)) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def scalar_loss(loss: torch.Tensor) -> torch.Tensor:
    return loss.mean() if loss.ndim > 0 else loss


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    seed_length: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    loss_total = 0.0
    token_total = 0
    acc_total = 0.0
    acc_batches = 0
    for batch_index, batch in enumerate(loader):
        token_ids = batch["token_ids"].to(device, non_blocking=True)
        targets = continuation_targets(token_ids, seed_length=seed_length)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            output = model(token_ids, targets)
        if output.loss is not None:
            loss_tensor = scalar_loss(output.loss)
            valid = int((targets != -100).sum().item())
            loss_total += float(loss_tensor.detach()) * float(valid)
            token_total += valid
            acc_total += continuation_accuracy(output.logits, targets)
            acc_batches += 1
        if batch_index + 1 >= int(max_batches):
            break
    model.train()
    loss = loss_total / max(1, token_total)
    return {
        "loss": loss,
        "ppl": math.exp(min(loss, 20.0)),
        "accuracy": acc_total / max(1, acc_batches),
        "tokens": float(token_total),
    }


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    config,
    step: int,
    epoch: int,
    best_val_loss: float | None,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": unwrap_model(model).state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "config": vars(config),
            "step": int(step),
            "epoch": int(epoch),
            "best_val_loss": best_val_loss,
            "args": vars(args),
        },
        path,
    )


def maybe_upload_folder(folder: Path, repo_id: str | None, token: str | None, path_in_repo: str) -> None:
    if not repo_id or not token or not folder.exists():
        return
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        print(f"Hugging Face upload skipped; huggingface_hub unavailable: {exc}", flush=True)
        return
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=path_in_repo.strip("/"),
        commit_message=f"Upload {path_in_repo.strip('/')}",
    )


def train(args: argparse.Namespace) -> None:
    torch.set_num_threads(max(1, min(int(args.threads), torch.get_num_threads())))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    total_length = int(args.seed_length) + int(args.continuation_length)
    config = continuation_config_for_model_type(args.model_type, smoke_test=bool(args.smoke_test))
    config.max_seq_len = total_length if not args.smoke_test else min(config.max_seq_len, total_length)
    config.dropout = float(args.dropout)

    records = load_pulse88_manifest(
        args.data_root,
        max_pieces=int(args.max_pieces),
        min_tokens=total_length,
    )
    train_records, val_records = train_val_split(records, val_fraction=float(args.val_fraction), seed=int(args.seed))
    train_dataset = Pulse88ContinuationDataset(
        train_records,
        seed_length=int(args.seed_length),
        continuation_length=int(args.continuation_length),
        event_size=int(config.event_size),
        seed=int(args.seed),
    )
    val_dataset = Pulse88ContinuationDataset(
        val_records,
        seed_length=int(args.seed_length),
        continuation_length=int(args.continuation_length),
        event_size=int(config.event_size),
        seed=int(args.seed) + 1,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(args.num_workers) > 0,
        collate_fn=collate_fixed_windows,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(args.num_workers) > 0,
        collate_fn=collate_fixed_windows,
    )

    model = build_continuation_model(config)
    param_count = count_parameters(model)
    model.to(device)
    if bool(args.data_parallel) and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"Using DataParallel on {torch.cuda.device_count()} CUDA devices", flush=True)

    print(f"Continuation {config.model_type}: {param_count:,} trainable parameters", flush=True)
    print(
        f"Data: records={len(records):,} train={len(train_dataset):,} val={len(val_dataset):,} "
        f"window={total_length} seed={args.seed_length} continuation={args.continuation_length}",
        flush=True,
    )
    print(f"Device: {device}, amp={use_amp}, amp_dtype={args.amp_dtype}", flush=True)

    optimizer = build_optimizer(model, lr=float(args.lr), weight_decay=float(args.weight_decay))
    updates_per_epoch = math.ceil(len(train_loader) / max(1, int(args.grad_accumulation_steps)))
    total_steps = int(args.max_steps) if args.max_steps else max(1, int(args.epochs) * updates_per_epoch)
    scheduler = build_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_ratio=float(args.warmup_ratio),
        min_lr_ratio=float(args.min_lr_ratio),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(use_amp and amp_dtype == torch.float16))

    model_name = str(config.model_type).lower()
    run_dir = args.output_dir / model_name
    metrics_dir = args.metrics_dir / model_name
    checkpoint_dir = run_dir / "checkpoints"
    history: list[dict] = []
    val_history: list[dict] = []
    best_val = float("inf")
    step = 0
    tokens_seen = 0
    start_time = time.time()
    recent_losses: list[float] = []
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(int(args.epochs)):
        for batch_index, batch in enumerate(train_loader, start=1):
            token_ids = batch["token_ids"].to(device, non_blocking=True)
            targets = continuation_targets(token_ids, seed_length=int(args.seed_length))
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                output = model(token_ids, targets)
                loss = output.loss
            if loss is None:
                raise RuntimeError("Model did not return a loss")
            loss = scalar_loss(loss)
            scaled_loss = loss / max(1, int(args.grad_accumulation_steps))
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            should_step = batch_index % int(args.grad_accumulation_steps) == 0 or batch_index == len(train_loader)
            valid_tokens = int((targets != -100).sum().item())
            tokens_seen += valid_tokens
            recent_losses.append(float(loss.detach()))
            if len(recent_losses) > 20:
                recent_losses.pop(0)
            if should_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step += 1

                elapsed = max(time.time() - start_time, 1e-6)
                tokens_per_second = tokens_seen / elapsed
                if step % int(args.log_interval) == 0 or step == 1:
                    train_loss = sum(recent_losses) / max(1, len(recent_losses))
                    record = {
                        "model_type": config.model_type,
                        "epoch": epoch + 1,
                        "step": step,
                        "train_loss": train_loss,
                        "train_ppl": math.exp(min(train_loss, 20.0)),
                        "tokens_seen": tokens_seen,
                        "tokens_per_second": tokens_per_second,
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "elapsed_seconds": elapsed,
                    }
                    history.append(record)
                    print(
                        f"step={step} epoch={epoch + 1}/{args.epochs} "
                        f"loss={train_loss:.4f} lr={record['lr']:.2e} tokens/sec={tokens_per_second:.1f}",
                        flush=True,
                    )

                if step % int(args.val_interval) == 0:
                    val_metrics = evaluate(
                        model,
                        val_loader,
                        device=device,
                        seed_length=int(args.seed_length),
                        use_amp=use_amp,
                        amp_dtype=amp_dtype,
                        max_batches=int(args.val_batches),
                    )
                    val_history.append({"step": step, "epoch": epoch + 1, **val_metrics})
                    print(
                        f"validation step={step} val_loss={val_metrics['loss']:.4f} "
                        f"val_ppl={val_metrics['ppl']:.2f} acc={val_metrics['accuracy']:.4f}",
                        flush=True,
                    )
                    if val_metrics["loss"] < best_val:
                        best_val = float(val_metrics["loss"])
                        save_checkpoint(
                            checkpoint_dir / f"{model_name}-best.pt",
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            config=config,
                            step=step,
                            epoch=epoch + 1,
                            best_val_loss=best_val,
                            args=args,
                        )

                if step % int(args.checkpoint_interval) == 0:
                    save_checkpoint(
                        checkpoint_dir / f"{model_name}-{step}.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        config=config,
                        step=step,
                        epoch=epoch + 1,
                        best_val_loss=best_val if best_val < float("inf") else None,
                        args=args,
                    )
                if int(args.max_steps) > 0 and step >= int(args.max_steps):
                    break
        if int(args.max_steps) > 0 and step >= int(args.max_steps):
            break

    final_metrics = evaluate(
        model,
        val_loader,
        device=device,
        seed_length=int(args.seed_length),
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        max_batches=int(args.val_batches),
    )
    val_history.append({"step": step, "epoch": epoch + 1, **final_metrics})
    best_val = min(best_val, float(final_metrics["loss"]))
    save_checkpoint(
        checkpoint_dir / f"{model_name}-final.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        config=config,
        step=step,
        epoch=epoch + 1,
        best_val_loss=best_val,
        args=args,
    )
    summary = {
        "model_type": config.model_type,
        "params": param_count,
        "config": vars(config),
        "data_root": str(args.data_root),
        "records": len(records),
        "train_records": len(train_dataset),
        "val_records": len(val_dataset),
        "seed_length": int(args.seed_length),
        "continuation_length": int(args.continuation_length),
        "steps": int(step),
        "tokens_seen": int(tokens_seen),
        "best_val_loss": float(best_val),
        "final_val": final_metrics,
        "history": history,
        "val_history": val_history,
        "elapsed_seconds": time.time() - start_time,
    }
    write_json(metrics_dir / "summary.json", summary)
    write_json(metrics_dir / "train_history.json", {"records": history})
    write_json(metrics_dir / "val_history.json", {"records": val_history})
    print(
        f"complete model={config.model_type} steps={step} best_val_loss={best_val:.4f} "
        f"final_val_loss={final_metrics['loss']:.4f}",
        flush=True,
    )
    maybe_upload_folder(checkpoint_dir, args.hf_repo_id, args.hf_token, f"continuation/{model_name}/checkpoints")
    maybe_upload_folder(metrics_dir, args.hf_repo_id, args.hf_token, f"continuation/{model_name}/metrics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_type", choices=["G_CONT", "H_CONT"], default="G_CONT")
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("continuation_checkpoints"))
    parser.add_argument("--metrics_dir", type=Path, default=Path("continuation_metrics"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--max_pieces", type=int, default=0)
    parser.add_argument("--seed_length", type=int, default=1024)
    parser.add_argument("--continuation_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--min_lr_ratio", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val_fraction", type=float, default=0.03)
    parser.add_argument("--val_batches", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--val_interval", type=int, default=1000)
    parser.add_argument("--checkpoint_interval", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--data_parallel", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--hf_repo_id", default=None)
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
