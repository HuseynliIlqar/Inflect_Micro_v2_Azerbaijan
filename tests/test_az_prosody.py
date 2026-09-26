"""Tests for the Azerbaijani stress-placement layer."""

from __future__ import annotations

import pytest

from aztts.az_prosody import PRIMARY, SECONDARY, restress


def stresses(phonemes: str) -> list[str]:
    """The stress mark carried by each word, or "" when it carries none."""
    marks = []
    for word in phonemes.split():
        if PRIMARY in word:
            marks.append(PRIMARY)
        elif SECONDARY in word:
            marks.append(SECONDARY)
        else:
            marks.append("")
    return marks


class TestFallback:
    def test_word_count_mismatch_is_left_alone(self):
        phonemes = f"b{PRIMARY}u d{PRIMARY}ir"
        assert restress(phonemes, "Bu bir söz artıqdır") == phonemes

    def test_empty_phonemes_are_left_alone(self):
        assert restress("", "Salam") == ""

    def test_unknown_level_is_rejected(self):
        with pytest.raises(ValueError, match="safe"):
            restress("a", "a", level="loud")


class TestSafeLevel:
    def test_clitic_loses_its_accent(self):
        phonemes = f"b{PRIMARY}u d\u0292{PRIMARY}ox"
        assert stresses(restress(phonemes, "Bu çox")) == ["", PRIMARY]

    def test_weak_head_is_demoted_not_dropped(self):
        phonemes = f"on{PRIMARY}un ev{PRIMARY}i"
        assert stresses(restress(phonemes, "onun evi")) == [SECONDARY, PRIMARY]

    def test_content_words_are_untouched(self):
        phonemes = f"toj{PRIMARY}a a\u0301d{PRIMARY}\u00e6t"
        assert restress(phonemes, "toya adət") == phonemes

    def test_punctuation_does_not_hide_a_clitic(self):
        phonemes = f"v{PRIMARY}\u00e6, d\u0292{PRIMARY}ox"
        assert stresses(restress(phonemes, "və, çox")) == ["", PRIMARY]

    def test_capitalised_clitic_is_recognised(self):
        phonemes = f"b{PRIMARY}u d\u0292{PRIMARY}ox"
        assert stresses(restress(phonemes, "Bu çox"))[0] == ""

    def test_safe_level_ignores_participles(self):
        phonemes = f"cit{PRIMARY}ab oxuj{PRIMARY}an ad{PRIMARY}am"
        assert restress(phonemes, "kitab oxuyan adam") == phonemes


class TestWideLevel:
    def test_postposition_loses_its_accent(self):
        phonemes = f"b{PRIMARY}iz yt\u0283{PRIMARY}yn"
        marks = stresses(restress(phonemes, "biz üçün", level="wide"))
        assert marks == ["", ""]

    def test_participle_is_demoted_mid_phrase(self):
        phonemes = f"cit{PRIMARY}ab oxuj{PRIMARY}an ad{PRIMARY}am"
        marks = stresses(restress(phonemes, "kitab oxuyan adam", level="wide"))
        assert marks == [PRIMARY, SECONDARY, PRIMARY]

    def test_ablative_noun_is_not_mistaken_for_a_participle(self):
        phonemes = f"n\u00e6sill\u00e6rd{PRIMARY}\u00e6n \u0261\u00e6l{PRIMARY}\u00e6n d\u00e6j{PRIMARY}\u00e6r"
        marks = stresses(restress(phonemes, "nəsillərdən gələn dəyər", level="wide"))
        assert marks == [PRIMARY, SECONDARY, PRIMARY]

    def test_possessive_is_not_mistaken_for_an_adjective(self):
        phonemes = f"dil{PRIMARY}i jo\u0279{PRIMARY}u ev{PRIMARY}i"
        assert restress(phonemes, "dili yolu evi", level="wide") == phonemes

    def test_long_adjective_is_demoted(self):
        phonemes = f"\u0261ijm\u00e6tl{PRIMARY}i \u00e6n\u00e6n{PRIMARY}\u00e6"
        marks = stresses(restress(phonemes, "qiymətli ənənə", level="wide"))
        assert marks == [SECONDARY, PRIMARY]

    def test_phrase_final_word_keeps_its_accent(self):
        phonemes = f"{PRIMARY}æks etdiɹ{PRIMARY}æn, b{PRIMARY}u"
        marks = stresses(restress(phonemes, "əks etdirən, bu", level="wide"))
        assert marks == [PRIMARY, PRIMARY, ""]

    def test_last_word_keeps_its_accent(self):
        phonemes = f"cit{PRIMARY}ab oxuj{PRIMARY}an"
        marks = stresses(restress(phonemes, "kitab oxuyan", level="wide"))
        assert marks == [PRIMARY, PRIMARY]


class TestDemotionMode:
    def test_secondary_can_be_switched_off(self):
        phonemes = f"cit{PRIMARY}ab oxuj{PRIMARY}an ad{PRIMARY}am"
        marks = stresses(
            restress(phonemes, "kitab oxuyan adam", level="wide", to_secondary=False)
        )
        assert marks == [PRIMARY, "", PRIMARY]


class TestLengthMarkIsNeverTouched:
    def test_phonemic_length_survives(self):
        phonemes = f"sa\u02d0\u026b{PRIMARY}am b{PRIMARY}u"
        result = restress(phonemes, "salam bu", level="wide")
        assert result.count("\u02d0") == 1
