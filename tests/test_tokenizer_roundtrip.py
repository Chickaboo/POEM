from __future__ import annotations

import pretty_midi

from tokenizer.midi_to_tokens import tokenize_midi
from tokenizer.tokens_to_midi import tokens_to_pretty_midi
from tokenizer.vocab import STEPS_PER_BEAT


def make_test_midi() -> pretty_midi.PrettyMIDI:
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    instrument = pretty_midi.Instrument(program=0)
    for pitch, start, end, velocity in [
        (60, 0.00, 0.25, 64),
        (64, 0.50, 0.75, 72),
        (67, 1.00, 1.50, 80),
        (72, 2.00, 2.50, 96),
    ]:
        instrument.notes.append(
            pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end)
        )
    midi.instruments.append(instrument)
    return midi


def notes(midi: pretty_midi.PrettyMIDI) -> list[pretty_midi.Note]:
    return sorted(
        [note for instrument in midi.instruments for note in instrument.notes],
        key=lambda note: note.start,
    )


def test_tokenizer_roundtrip_preserves_pitch_and_quantized_time(tmp_path) -> None:
    source_path = tmp_path / "source.mid"
    source = make_test_midi()
    source.write(str(source_path))

    result = tokenize_midi(source_path)
    reconstructed = tokens_to_pretty_midi(result.events, tempo=result.tempo)

    source_notes = notes(source)
    reconstructed_notes = notes(reconstructed)
    assert [note.pitch for note in reconstructed_notes] == [note.pitch for note in source_notes]

    seconds_per_step = 60.0 / result.tempo / STEPS_PER_BEAT
    tolerance = seconds_per_step * 1.5
    for original, recovered in zip(source_notes, reconstructed_notes, strict=True):
        assert abs(original.start - recovered.start) <= tolerance
        assert abs((original.end - original.start) - (recovered.end - recovered.start)) <= tolerance
