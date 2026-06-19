"""Pretokenize the Beautiful-Motifs MIDI files for faster POEM training."""

from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokenizer.midi_to_tokens import tokenize_midi
from training.data import cache_key_for_path, discover_motif_files, events_to_tensor, save_token_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("cache/poem-short-token-cache.pt"))
    parser.add_argument("--include_long_motifs", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=1000)
    args = parser.parse_args()

    files = discover_motif_files(args.data_dir, include_long_motifs=args.include_long_motifs)
    if args.limit is not None:
        files = files[: args.limit]
    records = {}
    failures: list[tuple[str, str]] = []
    total_events = 0
    start = time.time()

    for index, path in enumerate(files, start=1):
        try:
            result = tokenize_midi(path)
            tensor = events_to_tensor(result.events)
            records[cache_key_for_path(path)] = tensor
            total_events += int(tensor.shape[0])
        except Exception as exc:  # pragma: no cover - dataset hygiene reporting
            failures.append((str(path), str(exc)))
        if index == 1 or index % args.log_interval == 0:
            elapsed = max(time.time() - start, 1e-6)
            print(
                f"pretokenized {index}/{len(files)} files, "
                f"events={total_events}, files/sec={index / elapsed:.1f}",
                flush=True,
            )

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
