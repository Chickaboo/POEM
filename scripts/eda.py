"""Dataset and token-level EDA for POEM."""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.inspect_dataset import inspect_dataset
from tokenizer.midi_to_tokens import tokenize_midi
from tokenizer.vocab import DURATION_BUCKET_EDGES_BEATS, total_vocab_rows


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * pct))
    return ordered[index]


def format_stats(values: list[float], unit: str = "") -> str:
    if not values:
        return "n/a"
    suffix = f" {unit}" if unit else ""
    return (
        f"mean={statistics.fmean(values):.2f}{suffix}, "
        f"median={statistics.median(values):.2f}{suffix}, "
        f"p95={percentile(values, 0.95):.2f}{suffix}"
    )


def run_token_eda(files: list[Path], sample_size: int, full_token_count: bool, seed: int) -> None:
    rng = random.Random(seed)
    token_files = files if full_token_count else rng.sample(files, min(sample_size, len(files)))
    seq_lengths: list[float] = []
    mono_notes: list[float] = []
    source_notes: list[float] = []
    note_density: list[float] = []
    tempos: list[float] = []
    failures: list[tuple[Path, str]] = []

    for path in token_files:
        try:
            result = tokenize_midi(path)
        except Exception as exc:  # pragma: no cover - dataset hygiene reporting
            failures.append((path, str(exc)))
            continue
        seq_lengths.append(float(len(result.events)))
        mono_notes.append(float(result.monophonic_note_count))
        source_notes.append(float(result.source_note_count))
        tempos.append(float(result.tempo))
        if result.duration_seconds > 0:
            note_density.append(result.monophonic_note_count / result.duration_seconds)

    label = "full chosen subset" if full_token_count else f"sample of {len(token_files)} files"
    print(f"\nToken-level stats ({label}, {len(seq_lengths)} parsed):")
    print(f"  summed embedding vocabulary rows: {total_vocab_rows()}")
    print(f"  sequence length: {format_stats(seq_lengths, 'events')}")
    print(f"  monophonic notes per clip: {format_stats(mono_notes, 'notes')}")
    print(f"  source notes per clip before melody extraction: {format_stats(source_notes, 'notes')}")
    print(f"  monophonic note density: {format_stats(note_density, 'notes/sec')}")
    print(f"  onset-derived tempo estimate: {format_stats(tempos, 'BPM')}")
    if seq_lengths:
        if full_token_count:
            print(f"  exact total dataset tokens/events: {int(sum(seq_lengths))}")
        else:
            estimated = int(statistics.fmean(seq_lengths) * len(files))
            print(f"  estimated total dataset tokens/events from sample: {estimated}")
            print("  use --full_token_count for the exact token count used by compute-budget planning.")
    print("\nDuration bucket edges in beats:")
    print("  " + ", ".join(f"{edge:.4f}" for edge in DURATION_BUCKET_EDGES_BEATS))
    if failures:
        print(f"\nTokenization failures: {len(failures)}")
        for path, message in failures[:5]:
            print(f"  {path.name}: {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True, type=Path)
    parser.add_argument("--include_long_motifs", action="store_true")
    parser.add_argument("--duration_sample_size", type=int, default=200)
    parser.add_argument("--token_sample_size", type=int, default=1000)
    parser.add_argument("--full_token_count", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    categories = inspect_dataset(args.data_dir, sample_size=args.duration_sample_size, seed=args.seed)
    short_files = categories.get("short", [])
    long_files = categories.get("long", [])
    chosen = list(short_files)
    if args.include_long_motifs:
        chosen.extend(long_files)
    print("\nChosen subset for POEM primary training:")
    print(f"  short motifs: {len(short_files)}")
    print(f"  long motifs included: {args.include_long_motifs} ({len(long_files)} available)")
    print(f"  total chosen files: {len(chosen)}")
    run_token_eda(chosen, args.token_sample_size, args.full_token_count, args.seed)


if __name__ == "__main__":
    main()
