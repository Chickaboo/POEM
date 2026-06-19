"""Token-to-MIDI conversion for POEM generations and tokenizer tests."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pretty_midi

from tokenizer.vocab import (
    POSITIONS_PER_BAR,
    STEPS_PER_BEAT,
    TYPE_BAR,
    TYPE_NOTE,
    TYPE_PAD,
    TYPE_PIECE_END,
    TYPE_PIECE_START,
    dequantize_duration_beats,
    dequantize_velocity,
    is_rest_type,
    rest_type_to_bucket,
)


def _as_event_tuple(event: Sequence[int]) -> tuple[int, int, int, int, int]:
    if hasattr(event, "tolist"):
        event = event.tolist()
    if len(event) != 5:
        raise ValueError(f"Expected event with 5 fields, got {event}")
    return tuple(int(x) for x in event)  # type: ignore[return-value]


def tokens_to_pretty_midi(
    events: Iterable[Sequence[int]],
    tempo: float = 120.0,
    program: int = 0,
) -> pretty_midi.PrettyMIDI:
    midi = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))
    instrument = pretty_midi.Instrument(program=int(program), is_drum=False, name="POEM")
    seconds_per_step = 60.0 / float(tempo) / STEPS_PER_BEAT
    current_step = 0

    for raw_event in events:
        token_type, pitch, duration_bucket, velocity_bucket, position = _as_event_tuple(raw_event)
        if token_type in (TYPE_PAD, TYPE_PIECE_START):
            continue
        if token_type == TYPE_PIECE_END:
            break
        if token_type == TYPE_BAR:
            current_step = ((current_step // POSITIONS_PER_BAR) + 1) * POSITIONS_PER_BAR
            continue
        if is_rest_type(token_type):
            rest_beats = dequantize_duration_beats(rest_type_to_bucket(token_type))
            current_step += max(1, int(round(rest_beats * STEPS_PER_BEAT)))
            continue
        if token_type != TYPE_NOTE:
            continue

        position = int(position) % POSITIONS_PER_BAR
        bar_start = (current_step // POSITIONS_PER_BAR) * POSITIONS_PER_BAR
        start_step = bar_start + position
        if start_step < current_step:
            start_step += POSITIONS_PER_BAR
        duration_beats = dequantize_duration_beats(duration_bucket)
        duration_steps = max(1, int(round(duration_beats * STEPS_PER_BEAT)))
        end_step = start_step + duration_steps
        instrument.notes.append(
            pretty_midi.Note(
                velocity=dequantize_velocity(velocity_bucket),
                pitch=max(0, min(127, int(pitch))),
                start=start_step * seconds_per_step,
                end=end_step * seconds_per_step,
            )
        )
        current_step = end_step

    instrument.notes.sort(key=lambda note: (note.start, note.pitch))
    midi.instruments.append(instrument)
    return midi


def tokens_to_midi(
    events: Iterable[Sequence[int]],
    output_path: str | Path,
    tempo: float = 120.0,
    program: int = 0,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi = tokens_to_pretty_midi(events, tempo=tempo, program=program)
    midi.write(str(output_path))
    return output_path
