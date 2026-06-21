"""Convert generated MIDI samples into compact note-letter text."""

from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path

import pretty_midi


NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def note_name(pitch: int) -> str:
    octave = int(pitch) // 12 - 1
    return f"{NOTE_NAMES[int(pitch) % 12]}{octave}"


def duration_label(beats: float) -> str:
    fraction = Fraction(max(beats, 0.0)).limit_denominator(32)
    names = {
        Fraction(4, 1): "whole",
        Fraction(3, 1): "dotted-half",
        Fraction(2, 1): "half",
        Fraction(3, 2): "dotted-quarter",
        Fraction(1, 1): "quarter",
        Fraction(3, 4): "dotted-eighth",
        Fraction(1, 2): "eighth",
        Fraction(3, 8): "dotted-sixteenth",
        Fraction(1, 4): "sixteenth",
        Fraction(1, 8): "thirty-second",
    }
    return names.get(fraction, f"{fraction} beats")


def beat_position(beat: float) -> str:
    bar = int(beat // 4) + 1
    within_bar = beat - ((bar - 1) * 4)
    return f"{bar}:{within_bar + 1:.3f}"


def midi_to_text(path: Path) -> str:
    midi = pretty_midi.PrettyMIDI(str(path))
    tempi = midi.get_tempo_changes()[1]
    tempo = float(tempi[0]) if len(tempi) else 120.0
    notes = sorted(
        (note for instrument in midi.instruments if not instrument.is_drum for note in instrument.notes),
        key=lambda note: (note.start, note.pitch),
    )
    lines = [
        f"=== {path.name} ===",
        f"tempo_bpm: {tempo:.2f}",
        f"notes: {len(notes)}",
        "",
        "letter_sequence:",
    ]
    sequence: list[str] = []
    events: list[str] = []
    previous_end_beat = 0.0
    for note in notes:
        start_beat = note.start * tempo / 60.0
        end_beat = note.end * tempo / 60.0
        duration_beats = max(0.0, end_beat - start_beat)
        if start_beat - previous_end_beat > 0.01:
            rest_beats = start_beat - previous_end_beat
            sequence.append(f"Rest({duration_label(rest_beats)})")
            events.append(
                f"{beat_position(previous_end_beat)}  Rest  dur={duration_label(rest_beats)}"
            )
        name = note_name(note.pitch)
        dur = duration_label(duration_beats)
        sequence.append(f"{name}({dur})")
        events.append(
            f"{beat_position(start_beat)}  {name:<4} dur={dur:<14} velocity={note.velocity}"
        )
        previous_end_beat = max(previous_end_beat, end_beat)
    lines.append(" ".join(sequence) if sequence else "(no notes)")
    lines.extend(["", "timed_events:"])
    lines.extend(events or ["(no notes)"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    midi_files = sorted(args.input_dir.glob("*.mid"))
    if not midi_files:
        raise FileNotFoundError(f"No .mid files found in {args.input_dir}")
    output = args.output or args.input_dir / "samples_as_note_text.txt"
    output.write_text("\n\n".join(midi_to_text(path) for path in midi_files) + "\n", encoding="utf-8")
    print(f"Wrote {len(midi_files)} samples to {output}")


if __name__ == "__main__":
    main()
