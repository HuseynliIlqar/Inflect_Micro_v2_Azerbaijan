"""The gates and the take preference, without a GPU."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import Gates, cer, final_movement_st, preference, rejection, rise_fall_st, syllables  # noqa: E402

GATES = Gates(min_similarity=0.6, min_buzz=8.0, max_ring=0.25, min_range_st=8.0, exclamation_range_st=14.0)
GOOD = {"seconds": 3.0, "peak": 0.5, "clipped": 0.0, "cer": 0.0, "similarity": 0.8, "buzz": 10.0,
        "ring": 0.1, "range_st": 15.0, "final_st": 3.0, "peak_rise_st": 3.0, "end_drop_st": 6.0}
TEXT = "Sabah bizə gələcəksənmi?"  # 9 syllables


def test_a_good_take_passes() -> None:
    assert rejection("A1", TEXT, GOOD, GATES) is None


def test_questions_are_not_gated_on_a_final_rise() -> None:
    # By ear, good "-mı" questions peak before the particle and fall on it.
    assert rejection("A1", TEXT, {**GOOD, "final_st": -1.2}, GATES) is None
    assert rejection("A1", TEXT, {**GOOD, "final_st": math.nan, "end_drop_st": math.nan}, GATES) is None


def test_each_gate_has_its_reason() -> None:
    cases = {
        "short": {"seconds": 0.5}, "long": {"seconds": 16.0}, "silent": {"peak": 0.01},
        "clipped": {"clipped": 0.01}, "pace": {"seconds": 0.95 * 9 / 2.0},
        "misread": {"cer": 0.05}, "voice": {"similarity": 0.5}, "buzz": {"buzz": 7.0},
        "ring": {"ring": 0.3}, "monotone": {"range_st": 5.0},
    }
    for reason, change in cases.items():
        got = rejection("A1", TEXT, {**GOOD, **change}, GATES)
        assert got in (reason, "short"), (reason, got)


def test_long_categories_get_25_seconds() -> None:
    long_text = " ".join(["Bu, uzun bir cümlədir"] * 12) + "."
    assert rejection("L", long_text, {**GOOD, "seconds": 20.0}, GATES) is None
    assert rejection("D1", long_text, {**GOOD, "seconds": 20.0}, GATES) == "long"


def test_numbers_skip_the_whisper_gate_and_rare_words_get_more_room() -> None:
    assert rejection("I", "Beşinci mərtəbədə yaşayırıq.", {**GOOD, "cer": 0.3}, GATES) is None
    assert rejection("J", "Arıçı pətəyi açdı və bala baxdı.", {**GOOD, "cer": 0.04}, GATES) is None


def test_exclamation_needs_a_wide_range() -> None:
    assert rejection("C", "Vay, nə gözəl mənzərədir!", {**GOOD, "range_st": 12.0}, GATES) == "flat_exclamation"


def test_preference_orders_takes() -> None:
    a = (0, {**GOOD, "cer": 0.01})
    b = (1, {**GOOD, "cer": 0.0})
    c = (2, {**GOOD, "cer": 0.0, "end_drop_st": 11.0})
    best = min([a, b, c], key=lambda item: preference("A1", item[0], item[1]))
    assert best[0] == 2  # same reading, larger rise-then-fall
    flat = (0, {**GOOD, "end_drop_st": math.nan})
    assert min([flat, b], key=lambda item: preference("A1", item[0], item[1]))[0] == 1  # NaN ranks like no fall
    # Outside questions the shape does not count.
    assert min([(1, GOOD), (0, {**GOOD, "end_drop_st": 0.0})], key=lambda item: preference("D1", item[0], item[1]))[0] == 0
    tie = min([(1, GOOD), (0, GOOD)], key=lambda item: preference("D1", item[0], item[1]))
    assert tie[0] == 0  # take A wins a tie


def test_text_measures() -> None:
    assert syllables(TEXT) == 9
    assert cer("Qapını bağladınmı?", "qapını bağladınmı") == 0.0
    rising = [180.0] * 60 + [220.0] * 15
    assert final_movement_st(rising) > 3
    assert math.isnan(final_movement_st([180.0] * 8))


def test_rise_fall_shape() -> None:
    hz = lambda st: 100 * 2 ** (st / 12)  # noqa: E731
    # 1.5 s level, a peak 6 st up for 0.5 s, then a fall to 2 st below the start.
    question = [hz(0)] * 150 + [hz(6)] * 50 + [hz(-2)] * 20
    rise, drop = rise_fall_st(question)
    assert 5.5 < rise < 6.5 and 7.5 < drop < 8.5
    level = [hz(0)] * 200
    assert rise_fall_st(level) == (0.0, 0.0)
    # Unvoiced frames keep their time slot: the tail is measured in seconds, not voiced frames.
    gapped = [hz(0)] * 150 + [0.0] * 100 + [hz(4)] * 30
    assert abs(rise_fall_st(gapped)[0] - 4) < 0.01
    assert all(math.isnan(x) for x in rise_fall_st([hz(0)] * 10))
