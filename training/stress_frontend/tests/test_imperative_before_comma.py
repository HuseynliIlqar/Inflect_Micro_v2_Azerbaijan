"""A bare -ma/-mə verb form before a comma: negative imperative or verbal noun?

"Bu kitabı ÁLma, o birini al" is an order, stressed before the negative. In
"qidalanma, idman və yuxu" the same shape is a noun in a list and keeps the
final stress. Dataset v2 holds about 260 such imperatives, so the rule decides
how they are labelled for training.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from az_stress import Lexicon, stress_rule, word_context  # noqa: E402

LEXICON = Lexicon(
    verbs=frozenset({
        "al", "tələs", "qorx", "burax", "danış", "oy", "tök", "anlaş", "qidalan", "get", "gəl", "öyrət",
        "toxun", "oxu", "birləş", "proqramlaşdır", "çıx", "sın",
    }),
    lemmas=frozenset({
        "kitab", "ev", "armud", "gilas", "idman", "yuxu", "hörmət", "müzakirə", "pişik", "səbət", "məktəb", "buz",
    }),
    forms=frozenset({"məktəblər", "üstün"}),
)


def stress_in(sentence: str, word: str) -> int | None:
    return stress_rule(word, LEXICON, **word_context(sentence, word, LEXICON))[0]


@pytest.mark.parametrize(
    ("sentence", "word", "syllable"),
    [
        ("Bu kitabı alma, o birini al.", "alma", 0),
        ("Tələsmə, qızım, avtobus hələ gəlməyib.", "tələsmə", 1),
        ("Ata, buraxma, yıxılaram!", "buraxma", 1),
        ("Qorxma, körpü möhkəmdir.", "qorxma", 0),
        ("Mən dönənə qədər evdən heç yerə getmə, qızım.", "getmə", 0),
        ("Əsəbi olanda heç kimlə danışma, əvvəl sakitləş.", "danışma", 1),
        # A closing exclamation or verb after the comma is not a two-item list.
        ("Pişiyə toxunma, dırmaqlayar!", "toxunma", 1),
        ("Buzun üstünə çıxma, sınar!", "çıxma", 0),
        # A form of address set off by a comma anywhere in the sentence.
        ("Ay qızım, bu qədər oxuma, gözlərin yorulacaq.", "oxuma", 1),
    ],
)
def test_order_before_a_comma_is_stressed_before_the_negative(sentence: str, word: str, syllable: int) -> None:
    assert stress_in(sentence, word) == syllable


@pytest.mark.parametrize(
    ("sentence", "word"),
    [
        ("Qidalanma, idman və yuxu vacibdir.", "qidalanma"),
        ("Mən alma, armud və gilas aldım.", "alma"),
        ("Bu şeylər ştamplama, oyma, tökmə üsulları ilə hazırlanır.", "oyma"),
        ("Bu şeylər ştamplama, oyma, tökmə üsulları ilə hazırlanır.", "tökmə"),
        ("Etimad və anlaşma, hörmət vacibdir.", "anlaşma"),
        ("Hekayə danışma, müzakirə, öyrətmə daxildir.", "danışma"),
        # "və" three words on still marks a list (found in dataset v2 and v1).
        ("Məktəblərdə proqramlaşdırma, robot texnikası və maliyyə tədris olunacaq.", "proqramlaşdırma"),
        ("Birləşmə, hərbi hissə və bölmələr yeniləndi.", "birləşmə"),
        # "Babam" without a comma is the subject, not an address.
        ("Babam kənddən bir səbət alma, bir banka bal və kökə gətirərdi.", "alma"),
    ],
)
def test_noun_in_a_list_keeps_the_default(sentence: str, word: str) -> None:
    # A verbal noun keeps eSpeak's choice (None) or the final -ə rule's last syllable,
    # never the order's stress on the syllable before -ma/-mə.
    last = sum(character in "aıoueəiöü" for character in word) - 1
    assert stress_in(sentence, word) in (None, last)


def test_sentence_final_rule_is_unchanged() -> None:
    assert stress_in("Sabah bizə gəlmə.", "gəlmə") == 0


def test_context_carries_the_new_flag() -> None:
    assert word_context("Bu kitabı alma, o birini al.", "alma", LEXICON)["clause_imperative"] is True
    assert word_context("Mən alma, armud və gilas aldım.", "alma", LEXICON)["clause_imperative"] is False
