"""Data loading utilities for POEM."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from scripts.inspect_dataset import classify_motif, find_midi_files
from tokenizer.midi_to_tokens import midi_to_tokens, tokenize_midi
from tokenizer.vocab import (
    DUR_PAD,
    PAD_EVENT,
    POS_PAD,
    TYPE_NOTE,
    VEL_PAD,
    dequantize_duration_beats,
    is_rest_type,
    note_event,
    quantize_duration_beats,
    rest_bucket_to_type,
    rest_type_to_bucket,
)


TokenSequence = list[tuple[int, int, int, int, int]]
TokenCache = dict[str, torch.Tensor]


def discover_motif_files(data_dir: Path, include_long_motifs: bool = False) -> list[Path]:
    files = []
    for path in find_midi_files(data_dir):
        category = classify_motif(path)
        if category == "short" or (include_long_motifs and category == "long"):
            files.append(path)
    return sorted(files)


def split_files(
    files: Sequence[Path],
    val_fraction: float = 0.03,
    seed: int = 1337,
    smoke_test: bool = False,
) -> tuple[list[Path], list[Path]]:
    rng = random.Random(seed)
    shuffled = list(files)
    rng.shuffle(shuffled)
    if smoke_test:
        shuffled = shuffled[:60]
        return shuffled[:50], shuffled[50:60]
    val_count = max(1, int(round(len(shuffled) * val_fraction)))
    return shuffled[val_count:], shuffled[:val_count]


def transpose_events(
    events: Sequence[Sequence[int]],
    shift: int,
) -> list[tuple[int, int, int, int, int]]:
    shifted: list[tuple[int, int, int, int, int]] = []
    for raw_event in events:
        token_type, pitch, duration, velocity, position = (int(x) for x in raw_event)
        if token_type == TYPE_NOTE:
            pitch = max(0, min(127, pitch + shift))
        shifted.append((token_type, pitch, duration, velocity, position))
    return shifted


def scale_timing_events(
    events: Sequence[Sequence[int]],
    factor: float,
) -> list[tuple[int, int, int, int, int]]:
    scaled: list[tuple[int, int, int, int, int]] = []
    for raw_event in events:
        token_type, pitch, duration, velocity, position = (int(x) for x in raw_event)
        if token_type == TYPE_NOTE:
            new_bucket = quantize_duration_beats(dequantize_duration_beats(duration) * factor)
            scaled.append(note_event(pitch, new_bucket, velocity, position))
        elif is_rest_type(token_type):
            rest_bucket = rest_type_to_bucket(token_type)
            new_bucket = quantize_duration_beats(dequantize_duration_beats(rest_bucket) * factor)
            scaled.append((rest_bucket_to_type(new_bucket), 128, DUR_PAD, VEL_PAD, POS_PAD))
        else:
            scaled.append((token_type, pitch, duration, velocity, position))
    return scaled


class MIDITokenDataset(Dataset):
    def __init__(
        self,
        files: Sequence[Path],
        pitch_transpose_aug: bool = False,
        tempo_aug: bool = False,
        cache_tokens: bool = True,
        max_seq_len: int | None = None,
        token_cache: TokenCache | None = None,
    ) -> None:
        self.files = list(files)
        self.pitch_transpose_aug = pitch_transpose_aug
        self.tempo_aug = tempo_aug
        self.cache_tokens = cache_tokens
        self.max_seq_len = max_seq_len
        self.token_cache = token_cache or {}
        self._cache: dict[Path, TokenSequence] = {}

    def __len__(self) -> int:
        return len(self.files)

    def _load_events(self, path: Path) -> TokenSequence:
        if self.cache_tokens and path in self._cache:
            return self._cache[path]
        cache_key = cache_key_for_path(path)
        if cache_key in self.token_cache:
            events = tensor_to_events(self.token_cache[cache_key])
        else:
            events = midi_to_tokens(path)
        if self.max_seq_len is not None and len(events) > self.max_seq_len:
            events = events[: self.max_seq_len - 1] + [events[-1]]
        if self.cache_tokens:
            self._cache[path] = events
        return events

    def __getitem__(self, index: int) -> torch.Tensor:
        events = self._load_events(self.files[index])
        if self.pitch_transpose_aug:
            shift = random.randint(-5, 5)
            events = transpose_events(events, shift)
        if self.tempo_aug:
            events = scale_timing_events(events, random.uniform(0.9, 1.1))
        return torch.tensor(events, dtype=torch.long)


def collate_token_sequences(batch: Sequence[torch.Tensor]) -> torch.Tensor:
    max_len = max(item.size(0) for item in batch)
    output = torch.tensor(PAD_EVENT, dtype=torch.long).repeat(len(batch), max_len, 1)
    for index, item in enumerate(batch):
        output[index, : item.size(0)] = item
    return output


def cache_key_for_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").lower()


def events_to_tensor(events: Sequence[Sequence[int]]) -> torch.Tensor:
    return torch.tensor(events, dtype=torch.uint8)


def tensor_to_events(tensor: torch.Tensor) -> TokenSequence:
    return [tuple(int(value) for value in row) for row in tensor.cpu().tolist()]


def load_token_cache(path: Path | None) -> TokenCache:
    if path is None:
        return {}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records = payload.get("records", {})
    return {str(key): value.to(dtype=torch.uint8, device="cpu") for key, value in records.items()}


def save_token_cache(path: Path, records: TokenCache, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "records": records}, path)


def count_tokenized_events(
    files: Sequence[Path],
    max_seq_len: int | None = None,
    token_cache: TokenCache | None = None,
) -> int:
    total = 0
    token_cache = token_cache or {}
    for path in files:
        cache_key = cache_key_for_path(path)
        if cache_key in token_cache:
            length = int(token_cache[cache_key].shape[0])
        else:
            result = tokenize_midi(path)
            length = len(result.events)
        if max_seq_len is not None and length > max_seq_len:
            length = max_seq_len
        total += length
    return total
