"""Shared pieces of the dataset-v2 recording kit: categories, text measures and the gates.

Pure functions only, so the selection logic can be tested on a PC without a GPU.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

VOWELS = frozenset("aeəıioöuü")
_LOWER = str.maketrans({"I": "ı", "İ": "i"})
_NOT_WORD = re.compile(r"[^\w\s]", re.UNICODE)

# Yes/no questions, including ones marked only by intonation and tag questions.
# The pilot's ear test (144 takes, 24 questions) showed the good ones rise on the
# syllable before "-mı" and then fall, so they are not gated on a final rise; among
# passing takes the one with the largest rise-then-fall is preferred (see rise_fall_st).
RISING = frozenset({"A1", "A2", "A3"})
# Paragraphs and long sentences may run up to 25 s; everything else keeps v1's limit.
LONG = frozenset({"G1", "G2", "L"})
MAX_SECONDS_SHORT = 14.5
MAX_SECONDS_LONG = 25.0
MIN_SECONDS = 1.0
MIN_PACE, MAX_PACE = 2.6, 8.5  # syllables per second over the whole clip (v1 rule)
PACE_CHECK_MIN_SYLLABLES = 6
MIN_PEAK = 0.02
MAX_CLIPPED = 0.001
QUESTION_TAIL_S = 0.8  # where the pre-particle peak is looked for
FRAME_S = 0.01
MAX_CER = {"default": 0.02, "J": 0.05, "G1": 0.03, "G2": 0.03}
NO_CER = frozenset({"I"})  # Whisper writes numbers as digits; checked by pace, voice and ear


def az_lower(text: str) -> str:
    return text.translate(_LOWER).lower()


def word_list(text: str) -> list[str]:
    return _NOT_WORD.sub(" ", az_lower(text)).split()


def syllables(text: str) -> int:
    return sum(character in VOWELS for character in az_lower(text))


def edit_distance(a, b) -> int:
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        diagonal, row[0] = row[0], i
        for j, y in enumerate(b, start=1):
            diagonal, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, diagonal + (x != y))
    return row[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = "".join(word_list(reference)), "".join(word_list(hypothesis))
    return edit_distance(list(ref), list(hyp)) / max(len(ref), 1)


def final_movement_st(f0: list[float]) -> float:
    """Pitch of the last voiced stretch against the stretch before it, in semitones.

    The windows shrink for short questions so a two-word clip still gets a value.
    """
    voiced = [value for value in f0 if value > 0]
    n = len(voiced)
    tail = min(15, max(5, n // 6))
    before = min(40, max(10, n // 3))
    if n < tail + before:
        return math.nan
    last = sorted(voiced[-tail:])[tail // 2]
    earlier = sorted(voiced[-tail - before : -tail])[before // 2]
    return 12 * math.log2(last / earlier)


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, the same as numpy's default."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def rise_fall_st(f0: list[float], frame_s: float = FRAME_S) -> tuple[float, float]:
    """The question shape from a cleaned pitch track (frames of frame_s seconds, 0 = unvoiced).

    Returns (peak_rise, end_drop) in semitones: the held peak of the last QUESTION_TAIL_S
    seconds of voicing against the median of the rest, and the fall from that peak to the end.
    Both are NaN when fewer than 20 frames are voiced.
    """
    frames = [(index, 12 * math.log2(value / 100)) for index, value in enumerate(f0) if value > 0]
    if len(frames) < 20:
        return math.nan, math.nan
    # Whole frames, so the window edge does not depend on floating-point rounding.
    first_tail = frames[-1][0] - round(QUESTION_TAIL_S / frame_s)
    tail = [st for index, st in frames if index >= first_tail]
    body = [st for index, st in frames if index < first_tail]
    base = _percentile(body if len(body) >= 10 else [st for _, st in frames], 50)
    peak = _percentile(tail, 90)
    end = _percentile([st for _, st in frames[-5:]], 50)
    return peak - base, peak - end


@dataclass(frozen=True)
class Gates:
    """Thresholds from v1's distribution (see calibrate.py), plus the fixed rules."""

    min_similarity: float
    min_buzz: float
    max_ring: float
    min_range_st: float
    exclamation_range_st: float

    @classmethod
    def from_dict(cls, data: dict) -> "Gates":
        return cls(**{field: float(data[field]) for field in cls.__dataclass_fields__})


def rejection(cat: str, text: str, score: dict, gates: Gates) -> str | None:
    """Why a take fails, or None when it passes every gate."""
    seconds = score["seconds"]
    limit = MAX_SECONDS_LONG if cat in LONG else MAX_SECONDS_SHORT
    if seconds < MIN_SECONDS:
        return "short"
    if seconds > limit:
        return "long"
    if score["peak"] < MIN_PEAK:
        return "silent"
    if score["clipped"] > MAX_CLIPPED:
        return "clipped"
    count = syllables(text)
    if count >= PACE_CHECK_MIN_SYLLABLES and not MIN_PACE <= count / seconds <= MAX_PACE:
        return "pace"
    if cat not in NO_CER and score["cer"] > MAX_CER.get(cat, MAX_CER["default"]):
        return "misread"
    if score["similarity"] < gates.min_similarity:
        return "voice"
    if score["buzz"] < gates.min_buzz:
        return "buzz"
    if score["ring"] > gates.max_ring:
        return "ring"
    if not score["range_st"] >= gates.min_range_st:  # NaN fails too
        return "monotone"
    if cat == "C" and not score["range_st"] >= gates.exclamation_range_st:
        return "flat_exclamation"
    return None


def preference(cat: str, take_order: int, score: dict) -> tuple:
    """Sort key among passing takes: fewer misreadings, then for questions the larger
    rise-then-fall, then a cleaner sound and a closer voice; the earlier take wins a tie."""
    drop = score.get("end_drop_st", math.nan) if cat in RISING else 0.0
    shape = -drop if drop == drop else 0.0  # NaN ranks like no fall
    return (round(score["cer"], 3), round(shape, 1), -round(score["buzz"], 1), -round(score["similarity"], 3), take_order)
