"""Vocabulary and quantization constants for POEM.

POEM uses one event per musical step, but note events are represented by
separate categorical fields whose embeddings are summed.  This keeps the
effective vocabulary small instead of building a pitch-duration-velocity-position
cross product.
"""

from __future__ import annotations

import bisect
import math


EVENT_COLUMNS = ("type", "pitch", "duration", "velocity", "position")

TYPE_NOTE = 0
TYPE_REST_OFFSET = 1
NUM_REST_TYPES = 32
TYPE_BAR = TYPE_REST_OFFSET + NUM_REST_TYPES
TYPE_PIECE_START = TYPE_BAR + 1
TYPE_PIECE_END = TYPE_PIECE_START + 1
TYPE_PAD = TYPE_PIECE_END + 1
N_TOKEN_TYPES = TYPE_PAD + 1

PITCH_PAD = 128
N_PITCH = 129
DUR_PAD = 32
N_DUR = 33
VEL_PAD = 16
N_VEL = 17
POS_PAD = 16
N_POS = 17

PAD_EVENT = (TYPE_PAD, PITCH_PAD, DUR_PAD, VEL_PAD, POS_PAD)
PIECE_START_EVENT = (TYPE_PIECE_START, PITCH_PAD, DUR_PAD, VEL_PAD, POS_PAD)
PIECE_END_EVENT = (TYPE_PIECE_END, PITCH_PAD, DUR_PAD, VEL_PAD, POS_PAD)
BAR_EVENT = (TYPE_BAR, PITCH_PAD, DUR_PAD, VEL_PAD, POS_PAD)

BEATS_PER_BAR = 4
POSITIONS_PER_BAR = 16
STEPS_PER_BEAT = POSITIONS_PER_BAR // BEATS_PER_BAR

# 32 log-spaced duration buckets in beats.  The minimum is a 1/32-note-like
# grace bucket, while the maximum covers two 4/4 bars.  Bucket centers are the
# geometric means of adjacent edges.
DURATION_MIN_BEATS = 1.0 / 32.0
DURATION_MAX_BEATS = 8.0
DURATION_BUCKET_EDGES_BEATS = tuple(
    DURATION_MIN_BEATS
    * (DURATION_MAX_BEATS / DURATION_MIN_BEATS) ** (i / NUM_REST_TYPES)
    for i in range(NUM_REST_TYPES + 1)
)
DURATION_BUCKET_CENTERS_BEATS = tuple(
    math.sqrt(DURATION_BUCKET_EDGES_BEATS[i] * DURATION_BUCKET_EDGES_BEATS[i + 1])
    for i in range(NUM_REST_TYPES)
)


def is_rest_type(token_type: int) -> bool:
    return TYPE_REST_OFFSET <= int(token_type) < TYPE_REST_OFFSET + NUM_REST_TYPES


def rest_type_to_bucket(token_type: int) -> int:
    if not is_rest_type(token_type):
        raise ValueError(f"Token type {token_type} is not a REST_<bucket> token")
    return int(token_type) - TYPE_REST_OFFSET


def rest_bucket_to_type(bucket: int) -> int:
    bucket = int(bucket)
    if bucket < 0 or bucket >= NUM_REST_TYPES:
        raise ValueError(f"Duration bucket out of range: {bucket}")
    return TYPE_REST_OFFSET + bucket


def quantize_duration_beats(duration_beats: float) -> int:
    value = max(float(duration_beats), DURATION_MIN_BEATS)
    if value >= DURATION_MAX_BEATS:
        return NUM_REST_TYPES - 1
    return bisect.bisect_right(DURATION_BUCKET_EDGES_BEATS, value) - 1


def dequantize_duration_beats(bucket: int) -> float:
    bucket = min(max(int(bucket), 0), NUM_REST_TYPES - 1)
    return DURATION_BUCKET_CENTERS_BEATS[bucket]


def quantize_velocity(velocity: int) -> int:
    return min(max(int(velocity), 0), 127) * 16 // 128


def dequantize_velocity(bucket: int) -> int:
    bucket = min(max(int(bucket), 0), 15)
    return int(round((bucket + 0.5) * 128 / 16))


def note_event(pitch: int, duration_bucket: int, velocity_bucket: int, position: int) -> tuple[int, int, int, int, int]:
    return (
        TYPE_NOTE,
        min(max(int(pitch), 0), 127),
        min(max(int(duration_bucket), 0), NUM_REST_TYPES - 1),
        min(max(int(velocity_bucket), 0), 15),
        int(position) % POSITIONS_PER_BAR,
    )


def total_vocab_rows() -> int:
    return N_TOKEN_TYPES + N_PITCH + N_DUR + N_VEL + N_POS
