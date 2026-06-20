"""Pretokenize the Beautiful-Motifs MIDI files for faster POEM training."""

from __future__ import annotations

import argparse
import os
import time
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokenizer.midi_to_tokens import tokenize_midi
from training.data import cache_key_for_path, discover_motif_files, events_to_tensor, save_token_cache


def tokenize_one(path: Path) -> tuple[str, str, torch.Tensor, int, str | None]:
    try:
        result = tokenize_midi(path)
        tensor = events_to_tensor(result.events)
        return str(path), cache_key_for_path(path), tensor, int(tensor.shape[0]), None
    except Exception as exc:  # pragma: no cover - dataset hygiene reporting
        return str(path), cache_key_for_path(path), torch.empty((0, 5), dtype=torch.uint8), 0, str(exc)


def log_progress(index: int, total: int, total_events: int, start: float) -> None:
    elapsed = max(time.time() - start, 1e-6)
    print(
        f"pretokenized {index}/{total} files, "
        f"events={total_events}, files/sec={index / elapsed:.1f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("cache/poem-short-token-cache.pt"))
    parser.add_argument("--include_long_motifs", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes for MIDI tokenization. Use 0 or 1 for sequential mode.",
    )
    parser.add_argument("--chunksize", type=int, default=64)
    args = parser.parse_args()

    files = discover_motif_files(args.data_dir, include_long_motifs=args.include_long_motifs)
    if args.limit is not None:
        files = files[: args.limit]
    records = {}
    failures: list[tuple[str, str]] = []
    total_events = 0
    start = time.time()
    workers = max(0, int(args.workers))
    if workers == 0:
        workers = 1
    print(f"pretokenize workers: {workers}", flush=True)

    if workers <= 1:
        for index, path in enumerate(files, start=1):
            source, key, tensor, event_count, error = tokenize_one(path)
            if error is None:
                records[key] = tensor
                total_events += event_count
            else:
                failures.append((source, error))
            if index == 1 or index % args.log_interval == 0:
                log_progress(index, len(files), total_events, start)
    else:
        max_workers = min(workers, os.cpu_count() or workers)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(tokenize_one, files, chunksize=max(1, args.chunksize))
            for index, (source, key, tensor, event_count, error) in enumerate(results, start=1):
                if error is None:
                    records[key] = tensor
                    total_events += event_count
                else:
                    failures.append((source, error))
                if index == 1 or index % args.log_interval == 0:
                    log_progress(index, len(files), total_events, start)

    metadata = {
        "data_dir": str(args.data_dir.resolve()),
        "include_long_motifs": args.include_long_motifs,
        "file_count": len(files),
        "cached_count": len(records),
        "total_events": total_events,
        "failures": failures[:100],
    }
    save_token_cache(args.output, records, metadata)
    elapsed = max(time.time() - start, 1e-6)
    print(f"wrote {args.output}", flush=True)
    print(f"cached files: {len(records)}/{len(files)}", flush=True)
    print(f"total events: {total_events}", flush=True)
    print(f"elapsed: {elapsed:.1f}s", flush=True)
    if failures:
        print(f"failures: {len(failures)}", flush=True)
        for path, message in failures[:5]:
            print(f"  {path}: {message}", flush=True)


if __name__ == "__main__":
    main()
