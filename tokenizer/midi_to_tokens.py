"""MIDI-to-token conversion for POEM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pretty_midi

from tokenizer.vocab import (
    BAR_EVENT,
    DUR_PAD,
    PIECE_END_EVENT,
    PIECE_START_EVENT,
    POS_PAD,
    POSITIONS_PER_BAR,
    STEPS_PER_BEAT,
    VEL_PAD,
    note_event,
    quantize_duration_beats,
    quantize_velocity,
    rest_bucket_to_type,
)


@dataclass(frozen=True)
class QuantizedNote:
    start_step: int
    duration_steps: int
    pitch: int
    velocity: int


@dataclass(frozen=True)
class MidiTokenizationResult:
    events: list[tuple[int, int, int, int, int]]
    tempo: float
    duration_seconds: float
    source_note_count: int
    monophonic_note_count: int


def estimate_musical_tempo(midi: pretty_midi.PrettyMIDI) -> float:
    """Prefer onset-derived tempo because these files often have plain 60 BPM metadata."""

    try:
        tempo = float(midi.estimate_tempo())
    except Exception:
        tempo = 0.0
    if tempo > 0.0 and tempo < 400.0:
        return tempo
    _, tempi = midi.get_tempo_changes()
    if len(tempi):
        tempo = float(tempi[0])
        if tempo > 0.0:
            return tempo
    return 120.0


def _source_notes(midi: pretty_midi.PrettyMIDI) -> list[pretty_midi.Note]:
    return sorted(
        (
            note
            for instrument in midi.instruments
            if not instrument.is_drum
            for note in instrument.notes
        ),
        key=lambda note: (float(note.start), -int(note.pitch), -float(note.end)),
    )


def extract_monophonic_notes(
    midi: pretty_midi.PrettyMIDI,
    transpose: int = 0,
) -> tuple[list[QuantizedNote], float, int, float]:
    tempo = estimate_musical_tempo(midi)
    seconds_per_beat = 60.0 / tempo
    notes = _source_notes(midi)
    grouped: dict[int, pretty_midi.Note] = {}

    for note in notes:
        start_beats = float(note.start) / seconds_per_beat
        start_step = max(0, int(round(start_beats * STEPS_PER_BEAT)))
        existing = grouped.get(start_step)
        if existing is None or (note.pitch, note.velocity, note.end - note.start) > (
            existing.pitch,
            existing.velocity,
            existing.end - existing.start,
        ):
            grouped[start_step] = note

    melody: list[QuantizedNote] = []
    for start_step, note in sorted(grouped.items()):
        pitch = int(note.pitch) + int(transpose)
        if pitch < 0 or pitch > 127:
            continue
        duration_beats = max((float(note.end) - float(note.start)) / seconds_per_beat, 1.0 / STEPS_PER_BEAT)
        duration_steps = max(1, int(round(duration_beats * STEPS_PER_BEAT)))
        melody.append(
            QuantizedNote(
                start_step=start_step,
                duration_steps=duration_steps,
                pitch=pitch,
                velocity=int(note.velocity),
            )
        )

    clipped: list[QuantizedNote] = []
    for index, note in enumerate(melody):
        next_start = melody[index + 1].start_step if index + 1 < len(melody) else None
        duration_steps = note.duration_steps
        if next_start is not None:
            duration_steps = min(duration_steps, max(1, next_start - note.start_step))
        clipped.append(
            QuantizedNote(
                start_step=note.start_step,
                duration_steps=max(1, duration_steps),
                pitch=note.pitch,
                velocity=note.velocity,
            )
        )

    duration_seconds = max((float(note.end) for note in notes), default=midi.get_end_time())
    return clipped, tempo, len(notes), float(duration_seconds)


def tokenize_midi(path: str | Path, transpose: int = 0) -> MidiTokenizationResult:
    midi = pretty_midi.PrettyMIDI(str(path))
    melody, tempo, source_note_count, duration_seconds = extract_monophonic_notes(midi, transpose)
    events: list[tuple[int, int, int, int, int]] = [PIECE_START_EVENT]
    current_step = 0

    for note in melody:
        note_bar = note.start_step // POSITIONS_PER_BAR
        current_bar = current_step // POSITIONS_PER_BAR
        while current_bar < note_bar:
            events.append(BAR_EVENT)
            current_bar += 1
            current_step = current_bar * POSITIONS_PER_BAR

        if note.start_step > current_step:
            gap_steps = note.start_step - current_step
            gap_beats = gap_steps / STEPS_PER_BEAT
            rest_bucket = quantize_duration_beats(gap_beats)
            events.append((rest_bucket_to_type(rest_bucket), 128, DUR_PAD, VEL_PAD, POS_PAD))
            current_step = note.start_step

        duration_beats = note.duration_steps / STEPS_PER_BEAT
        duration_bucket = quantize_duration_beats(duration_beats)
        position = note.start_step % POSITIONS_PER_BAR
        events.append(
            note_event(
                pitch=note.pitch,
                duration_bucket=duration_bucket,
                velocity_bucket=quantize_velocity(note.velocity),
                position=position,
            )
        )
        current_step = max(current_step, note.start_step + note.duration_steps)

    events.append(PIECE_END_EVENT)
    return MidiTokenizationResult(
        events=events,
        tempo=tempo,
        duration_seconds=duration_seconds,
        source_note_count=source_note_count,
        monophonic_note_count=len(melody),
    )


def midi_to_tokens(path: str | Path, transpose: int = 0) -> list[tuple[int, int, int, int, int]]:
    return tokenize_midi(path, transpose=transpose).events
