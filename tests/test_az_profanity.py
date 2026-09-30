"""The obscenity filter: what it bleeps, and -- as much -- what it leaves alone.

A false positive is worse than a miss here: bleeping `şikayət` or `götürmək`
breaks an ordinary sentence, so the clean-word list is as long as the dirty one.
"""

from __future__ import annotations

import pytest

from aztts import normalize_az
from aztts.az_chunk import chunk_text
from aztts.az_profanity import BLEEP, bleep_segments, censor_az, is_profane
from aztts.engine import BLEEP_SECONDS, bleep


@pytest.mark.parametrize(
    "word",
    [
        # stems with their inflections
        "sik", "sikim", "sikdir", "siktir", "sikərəm", "sikib", "sikiş",
        "qəhbə", "qəhbəsən", "qəhbələr", "qəhbəxana",
        "amcıq", "amcığı", "amına", "amını",
        "göt", "götü", "götünə", "götdən", "götverən",
        "yarraq", "yarrağı", "qancıq", "orospu", "peysər", "pezəvəng",
        "dəyyus", "yavşaq", "gicdıllaq", "gijdıllaq", "sürtük",
        "piç", "piçlər", "itoğlu", "köpəkoğlu",
        # Russian and Turkish loans, written in Latin letters
        "blyat", "dalbayob", "dolbayob", "pidaras", "pidor", "naxuy", "nahuy",
        "pizdec", "suka", "mudak", "amk",
        # the same loans in Cyrillic
        "блять", "сука", "хуй", "пиздец", "мудак",
    ],
)
def test_obscenities_are_caught(word: str) -> None:
    assert is_profane(word), word


@pytest.mark.parametrize(
    "word",
    [
        # share `sik` / `şik`
        "sikkə", "sikkələr", "siklon", "sikl", "şikayət", "sikayet", "şikəst",
        "şikar", "şik",
        # share `göt`
        "götür", "götürmək", "götürdü", "götürülüb", "Göteborq",
        # share `am`
        "amma", "amin", "Amerika", "ambar", "amil", "amal",
        # share `yarraq`, `mudak`, `xuy`, `piç`, `qancıq`, `suka`
        "yaraq", "yarpaq", "müdafiə", "müdir", "huy", "huyu", "pişik",
        "qan", "qanun", "sukan",
        # ordinary words from the demo sentences
        "Salam", "maşında", "işləyir", "kitabı", "maraqlı", "küçələri",
    ],
)
def test_clean_words_are_left_alone(word: str) -> None:
    assert not is_profane(word), word


def test_case_does_not_matter_and_uses_azerbaijani_casing() -> None:
    assert is_profane("SİKDİR")
    assert is_profane("Qəhbə")
    assert not is_profane("ŞİKAYƏT")


@pytest.mark.parametrize("word", ["qehbe", "amciq", "gotune", "s1kdir", "$ikdir", "siiiikdir"])
def test_common_spellings_without_azerbaijani_letters_are_caught(word: str) -> None:
    assert is_profane(word), word


def test_a_sentence_keeps_everything_but_the_obscenity() -> None:
    assert censor_az("Sən qəhbəsən, bildin?") == f"Sən {BLEEP}, bildin?"


def test_neighbouring_obscenities_become_one_bleep() -> None:
    assert censor_az("Siktir get, qəhbə, dalbayob!") == f"{BLEEP} get, {BLEEP}!"
    assert censor_az("sikim qəhbə yarraq") == BLEEP


def test_a_suffix_after_an_apostrophe_or_hyphen_goes_with_the_word() -> None:
    assert censor_az("Bu göt'ə bax.") == f"Bu {BLEEP} bax."
    assert censor_az("qəhbə-lər gəldi") == f"{BLEEP} gəldi"


def test_a_clean_hyphenated_word_is_untouched() -> None:
    assert censor_az("Bakı-Gəncə yolu") == "Bakı-Gəncə yolu"


def test_clean_text_is_returned_unchanged() -> None:
    text = "Şikayətinizi götürdük, sikkəni isə Amerikaya göndərdik."
    assert censor_az(text) == text


def test_empty_text() -> None:
    assert censor_az("") == ""
    assert censor_az("   ") == "   "


def test_the_bleep_survives_normalisation_and_chunking() -> None:
    normalised = normalize_az(censor_az("Qəhbə 25% artım etdi."))
    assert normalised == f"{BLEEP} iyirmi beş faiz artım etdi."
    assert BLEEP in chunk_text(normalised)[0]


@pytest.mark.parametrize(
    ("chunk", "expected"),
    [
        ("Salam.", ("Salam.",)),
        # a mark stranded at the head of a piece has nothing to close
        (f"Sən {BLEEP}, bildin?", ("Sən", BLEEP, "bildin?")),
        (f"{BLEEP}!", (BLEEP,)),
        (f"{BLEEP} get.", (BLEEP, "get.")),
        (f"a {BLEEP} b {BLEEP}", ("a", BLEEP, "b", BLEEP)),
    ],
)
def test_bleep_segments_split_a_chunk_for_synthesis(chunk: str, expected: tuple[str, ...]) -> None:
    assert bleep_segments(chunk) == expected


def test_the_bleep_is_a_quiet_tone_that_starts_and_ends_in_silence() -> None:
    tone = bleep(24000)
    assert tone.dtype.name == "float32"
    assert tone.size == round(24000 * BLEEP_SECONDS)
    assert 0.2 < abs(tone).max() <= 0.25
    assert tone[0] == 0.0 and abs(tone[-1]) < 1e-6


@pytest.mark.parametrize(
    "text",
    [
        # a whole-form entry is matched with its own letters, not folded ones
        "Amina gəldi", "Amini dedi", "Got it", "PIC mikrokontroller",
        # a price is not a word written in digits
        "$1k qazandı", "10k qaçış", "1.5k",
        # names that begin like the `sik` stem
        "Sikandar", "Sikorski",
    ],
)
def test_names_prices_and_foreign_words_are_left_alone(text: str) -> None:
    assert censor_az(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "göt",          # decomposed ö, as macOS and some clipboards send it
        "piç",          # decomposed ç
        "s­ik",          # soft hyphen
        "si​k",          # zero-width space
        "q‍əhbə",        # zero-width joiner
        "﻿sikdir",       # byte-order mark
    ],
)
def test_invisible_characters_and_decomposed_letters_do_not_hide_a_word(text: str) -> None:
    assert censor_az(text) == BLEEP
