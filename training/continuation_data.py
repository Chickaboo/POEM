"""Dataset loading for Pulse88 CustomDeltaTokenizer continuation windows."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def find_manifest_paths(root: Path) -> list[Path]:
    candidates = [
        root / "metadata" / "manifest.json",
        root / "manifest.json",
    ]
    found = [path for path in candidates if path.exists()]
    found.extend(sorted(root.glob("**/metadata/manifest.json")))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in found:
        key = str(path.resolve()).lower()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _resolve_npz_path(raw_path: str, manifest_path: Path, data_root: Path) -> Path | None:
    raw = Path(str(raw_path))
    probes: list[Path] = []
    if raw.is_absolute():
        probes.append(raw)
    else:
        probes.extend(
            [
                data_root / raw,
                manifest_path.parent / raw,
                manifest_path.parent.parent / raw,
                manifest_path.parent.parent / raw.name,
                manifest_path.parent.parent / "data" / raw.name,
            ]
        )
    for probe in probes:
        if probe.exists() and probe.is_file():
            return probe.resolve()
    return None


def load_pulse88_manifest(
    data_root: Path,
    *,
    max_pieces: int = 0,
    min_tokens: int = 1024,
) -> list[dict[str, Any]]:
    manifest_paths = find_manifest_paths(data_root)
    if not manifest_paths:
        raise FileNotFoundError(f"No Pulse88 manifest.json found under {data_root}")

    records: list[dict[str, Any]] = []
    skipped_unresolved = 0
    skipped_short = 0
    for manifest_path in manifest_paths:
        payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, list):
            continue
        for row in payload:
            if not isinstance(row, dict):
                continue
            raw_npz = str(row.get("npz_path", "")).strip()
            if not raw_npz and str(row.get("tokens_path", "")).strip():
                raw_npz = str(row.get("tokens_path", "")).strip()
            if not raw_npz:
                skipped_unresolved += 1
                continue
            npz_path = _resolve_npz_path(raw_npz, manifest_path, data_root)
            if npz_path is None:
                skipped_unresolved += 1
                continue
            length = int(row.get("length", row.get("tokens", -1)))
            if length <= 0:
                with np.load(npz_path, allow_pickle=False) as pack:
                    length = int(pack["tokens"].shape[0])
            if length < int(min_tokens):
                skipped_short += 1
                continue
            records.append(
                {
                    "piece_id": str(row.get("md5", npz_path.stem) or npz_path.stem),
                    "source_path": str(row.get("source_path", "")),
                    "npz_path": str(npz_path),
                    "length": int(length),
                    "manifest_path": str(manifest_path),
                }
            )
            if int(max_pieces) > 0 and len(records) >= int(max_pieces):
                return records

    if len(records) < 2:
        raise RuntimeError(
            "Need at least two eligible Pulse88 token files after filtering "
            f"(kept={len(records)}, skipped_unresolved={skipped_unresolved}, skipped_short={skipped_short})."
        )
    return records


def train_val_split(
    records: Sequence[dict[str, Any]],
    *,
    val_fraction: float = 0.05,
    seed: int = 1337,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(records)
    random.Random(int(seed)).shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * float(val_fraction))))
    val_count = min(val_count, len(shuffled) - 1)
    return shuffled[val_count:], shuffled[:val_count]


class Pulse88ContinuationDataset(Dataset):
    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        *,
        seed_length: int,
        continuation_length: int,
        event_size: int = 4,
        seed: int = 1337,
    ) -> None:
        self.records = list(records)
        self.seed_length = int(seed_length)
        self.continuation_length = int(continuation_length)
        self.total_length = self.seed_length + self.continuation_length
        self.event_size = max(1, int(event_size))
        self.rng = random.Random(int(seed))
        if self.seed_length % self.event_size != 0:
            raise ValueError(f"seed_length must be divisible by event_size={self.event_size}")
        if self.continuation_length % self.event_size != 0:
            raise ValueError(f"continuation_length must be divisible by event_size={self.event_size}")
        self.records = [record for record in self.records if int(record["length"]) >= self.total_length]
        if not self.records:
            raise RuntimeError(f"No records are long enough for window length {self.total_length}")

    def __len__(self) -> int:
        return len(self.records)

    def _window_start(self, length: int) -> int:
        max_start = max(0, int(length) - self.total_length)
        raw = self.rng.randint(0, max_start) if max_start > 0 else 0
        snapped = raw - (raw % self.event_size)
        if snapped > max_start:
            snapped = max_start - (max_start % self.event_size)
        return max(0, int(snapped))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        with np.load(Path(str(record["npz_path"])), allow_pickle=False) as pack:
            tokens = np.asarray(pack["tokens"], dtype=np.int64)
        start = self._window_start(int(tokens.shape[0]))
        window = tokens[start : start + self.total_length]
        seed = window[: self.seed_length]
        continuation = window[self.seed_length :]
        return {
            "seed": torch.from_numpy(seed.astype(np.int64, copy=False)),
            "continuation": torch.from_numpy(continuation.astype(np.int64, copy=False)),
            "token_ids": torch.from_numpy(window.astype(np.int64, copy=False)),
        }


def collate_fixed_windows(batch: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "seed": torch.stack([item["seed"] for item in batch], dim=0),
        "continuation": torch.stack([item["continuation"] for item in batch], dim=0),
        "token_ids": torch.stack([item["token_ids"] for item in batch], dim=0),
    }
