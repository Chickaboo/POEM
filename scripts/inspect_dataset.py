"""Inspect the Beautiful-Motifs MIDI directory before tokenization.

The project intentionally discovers the local directory layout instead of
assuming the Hugging Face dataset structure.  Duration sampling uses
pretty_midi because it exposes note start/end times directly and keeps this
script focused on musical properties rather than raw MIDI event plumbing.
"""

from __future__ import annotations

import argparse
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pretty_midi


MIDI_SUFFIXES = {".mid", ".midi"}


def find_midi_files(data_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in MIDI_SUFFIXES
    )


def classify_motif(path: Path) -> str:
    parts = [part.lower() for part in path.parts]
    name = path.name.lower()
    if any("short" in part for part in parts) or "short" in name:
        return "short"
    if any("long" in part for part in parts) or "long" in name:
        return "long"
    return "unknown"


def print_tree(root: Path, max_depth: int = 3, max_entries_per_dir: int = 40) -> None:
    print(f"Directory tree (top {max_depth} levels):")

    def walk(directory: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            print("  " * depth + "[permission denied]")
            return
        shown = entries[:max_entries_per_dir]
        for entry in shown:
            suffix = "/" if entry.is_dir() else ""
            print("  " * depth + f"{entry.name}{suffix}")
            if entry.is_dir():
                walk(entry, depth + 1)
        if len(entries) > len(shown):
            print("  " * depth + f"... ({len(entries) - len(shown)} more entries)")

    print(f"{root.name}/")
    walk(root, 1)


def category_evidence(files: Iterable[Path], data_dir: Path) -> None:
    by_parent = Counter()
    by_category = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    for path in files:
        rel = path.relative_to(data_dir)
        by_parent[str(rel.parent)] += 1
        category = classify_motif(path)
        by_category[category] += 1
        if len(examples[category]) < 5:
            examples[category].append(str(rel))

    print("\nMIDI counts by apparent category:")
    for category, count in sorted(by_category.items()):
        print(f"  {category}: {count}")

    print("\nLargest MIDI-containing folders:")
    for folder, count in by_parent.most_common(12):
        print(f"  {folder}: {count}")

    print("\nClassification evidence:")
    if by_category["short"] or by_category["long"]:
        print("  Short/long is distinguishable by folder or filename text.")
    else:
        print("  Short/long was not distinguishable by folder or filename text.")
    for category in ("short", "long", "unknown"):
        if examples.get(category):
            print(f"  {category} examples:")
            for example in examples[category]:
                print(f"    {example}")


def parse_midi_stats(path: Path) -> tuple[float, int]:
    midi = pretty_midi.PrettyMIDI(str(path))
    notes = [
        note
        for instrument in midi.instruments
        if not instrument.is_drum
        for note in instrument.notes
    ]
    note_count = len(notes)
    duration = max((note.end for note in notes), default=midi.get_end_time())
    return float(duration), note_count


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * pct))
    return ordered[index]


def sample_category_stats(
    category: str, files: list[Path], sample_size: int, seed: int
) -> None:
    if not files:
        print(f"\n{category}: no files to sample")
        return
    rng = random.Random(seed)
    sample = rng.sample(files, min(sample_size, len(files)))
    durations: list[float] = []
    note_counts: list[int] = []
    failures: list[tuple[Path, str]] = []
    for path in sample:
        try:
            duration, note_count = parse_midi_stats(path)
        except Exception as exc:  # pragma: no cover - dataset hygiene reporting
            failures.append((path, str(exc)))
            continue
        durations.append(duration)
        note_counts.append(note_count)

    print(f"\n{category} duration sample ({len(durations)}/{len(sample)} parsed):")
    if durations:
        print(
            "  seconds: "
            f"mean={statistics.fmean(durations):.2f}, "
            f"median={statistics.median(durations):.2f}, "
            f"p95={percentile(durations, 0.95):.2f}"
        )
        print(
            "  note count: "
            f"mean={statistics.fmean(note_counts):.1f}, "
            f"median={statistics.median(note_counts):.1f}, "
            f"p95={percentile([float(n) for n in note_counts], 0.95):.0f}"
        )
    if failures:
        print(f"  parse failures: {len(failures)}")
        for path, message in failures[:5]:
            print(f"    {path.name}: {message}")


def inspect_dataset(data_dir: Path, sample_size: int = 200, seed: int = 1337) -> dict[str, list[Path]]:
    data_dir = data_dir.expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    print(f"Dataset directory: {data_dir}")
    print_tree(data_dir)

    midi_files = find_midi_files(data_dir)
    print(f"\nTotal MIDI files: {len(midi_files)}")
    category_evidence(midi_files, data_dir)

    by_category: dict[str, list[Path]] = defaultdict(list)
    for path in midi_files:
        by_category[classify_motif(path)].append(path)

    for category in ("short", "long", "unknown"):
        sample_category_stats(category, by_category.get(category, []), sample_size, seed)

    chosen = len(by_category.get("short", []))
    print("\nDefault training subset:")
    print(f"  short motifs only: {chosen} files")
    if chosen and abs(chosen - 70_000) / 70_000 > 0.1:
        print("  note: this differs meaningfully from the rough 70k prior estimate.")
    print("  pass --include_long_motifs in training scripts to include long motifs later.")
    return by_category


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True, type=Path)
    parser.add_argument("--sample_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    inspect_dataset(args.data_dir, args.sample_size, args.seed)


if __name__ == "__main__":
    main()
