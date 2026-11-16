"""Tests for the Azerbaijani stress rules. Syllable indices count from 0 at the start of the word."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from az_stress import Lexicon, lexical_stress, place_stress, restress_sentence, stress_rule  # noqa: E402

LEXICON = Lexicon(
    verbs=frozenset({"gəl", "get", "yaz", "bil", "oxu", "bildir", "yandır", "ol", "et", "de", "ye", "gör", "başla", "deyil", "istə", "düşün"}),
    lemmas=frozenset({"müəllim", "həkim", "gözəl", "kitab", "maşın", "qələm", "adam", "salam", "tamam", "tələbə", "ev", "uşaq", "nadir", "kimi", "tam", "qara", "yaşıl", "düz", "ağ", "uzun", "alıq", "aran", "mərə", "i", "ri", "bəyaz", "yeni", "sürət", "sadə", "ürəyin", "siz", "mən", "tək"}),
    forms=frozenset({"evdə", "müəllim", "həkim", "kitab", "maşın", "qələm", "uşaq", "gözəl", "tələbə"}),
)


def stress(word: str, question: bool = False) -> int | None:
    return lexical_stress(word, LEXICON, question=question)


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("gəlmədi", 0),
        ("getmədi", 0),
        ("yazmayacağıq", 0),
        ("bilmirəm", 0),
        ("oxumur", 1),
        ("gəlməz", 0),
        ("getmədən", 0),
        ("gəlməyib", 0),
    ],
)
def test_negative_verbs_stress_the_syllable_before_the_negative(word: str, expected: int) -> None:
    assert stress(word) == expected


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("gəlirəm", 1),
        ("gələcəyəm", 2),
        ("gəlmişəm", 1),
        ("oxumusan", 2),
        ("gəlirsən", 1),
        ("gəlirlər", 1),
        ("gəlirdi", 1),
        ("gələrkən", 1),
        ("gəlibdir", 1),
    ],
)
def test_unstressed_endings_after_a_tense_marker_leave_the_stress_on_the_marker(word: str, expected: int) -> None:
    assert stress(word) == expected


@pytest.mark.parametrize("word", ["gəldi", "gəlir", "gələcək", "gəldim", "gəlmək", "bildirdi", "yandır"])
def test_verb_forms_with_final_stress_are_left_alone(word: str) -> None:
    assert stress(word) is None


def test_question_particle_is_unstressed() -> None:
    assert stress("gəldinmi", question=True) == 1
    assert stress("gəlirsənmi", question=True) == 1
    assert stress("kitabdırmı", question=True) == 1


def test_question_particle_needs_a_question() -> None:
    assert stress("kitabdırmı") is None
    assert stress("kimi", question=True) is None


@pytest.mark.parametrize(("word", "expected"), [("gəlməsi", None), ("gəlməkdən", None), ("gəlməyə", 2), ("gəlmə", 1)])
def test_verbal_nouns_are_not_read_as_negatives(word: str, expected: int | None) -> None:
    assert stress(word) == expected


def test_negative_imperative_and_conditional() -> None:
    assert stress("gəlməsin") == 0
    assert stress("gəlməsə") == 0


@pytest.mark.parametrize(
    ("word", "expected"),
    [("müəllimdir", 2), ("evdədir", 1), ("gözəldir", 1), ("həkiməm", 1), ("tələbəyəm", 2), ("həkimsən", 1), ("maşınla", 1), ("qələmlə", 1), ("uşaqkən", 1)],
)
def test_unstressed_endings_on_nouns(word: str, expected: int) -> None:
    assert stress(word) == expected


@pytest.mark.parametrize("word", ["adam", "salam", "tamam", "nadir", "kitab", "kitablar"])
def test_dictionary_words_are_not_split_into_stem_and_ending(word: str) -> None:
    assert stress(word) is None


@pytest.mark.parametrize(
    ("word", "expected"),
    [("sonra", 0), ("indi", 0), ("amma", 0), ("ancaq", 0), ("lakin", 0), ("çünki", 0), ("dünən", 0), ("deyil", 0), ("Şəki", 0), ("Quba", 0), ("Şəkidə", 0), ("Türkiyə", 0), ("Türkiyədə", 0), ("burada", 0), ("Ona", 0)],
)
def test_exception_words(word: str, expected: int) -> None:
    assert stress(word) == expected


@pytest.mark.parametrize(
    ("word", "expected"),
    [("Rusiya", 0), ("Almaniya", 1), ("texnologiya", 2), ("Rusiyada", 0), ("Amerika", 1), ("politika", 1), ("texnika", 0)],
)
def test_loanwords_in_iya_and_ika(word: str, expected: int) -> None:
    assert stress(word) == expected


@pytest.mark.parametrize("word", ["niyə", "Türkiyə", "bika"])
def test_words_that_only_look_like_loans_are_not_treated_as_loans(word: str) -> None:
    assert stress_rule(word, LEXICON)[1] != "loan"


@pytest.mark.parametrize(("word", "expected"), [("qapqara", 0), ("yamyaşıl", 0), ("dümdüz", 0), ("apağ", 0), ("upuzun", 0), ("bəmbəyaz", 0), ("yepyeni", 0)])
def test_intensive_adjectives_stress_the_first_syllable(word: str, expected: int) -> None:
    assert stress(word) == expected


@pytest.mark.parametrize(
    ("token", "syllable", "expected"),
    [
        ("ɟælmædˈi", 0, "ɟˈælmædi"),
        ("citabɫˈar", 0, "cˈitabɫar"),
        ("myællimdˈir", 2, "myællˈimdir"),
        ("jaːˈɯʃ", 0, "jˈaːɯʃ"),
        ("andʒˈax", 0, "ˈandʒax"),
        ("andʒax", 1, "andʒˈax"),
        ("oxumusanmˈɯ?", 2, "oxumˈusanmɯ?"),
        ("anˈa,", 0, "ˈana,"),
    ],
)
def test_place_stress_moves_the_mark_to_the_syllable_onset(token: str, syllable: int, expected: str) -> None:
    assert place_stress(token, syllable) == expected


def test_place_stress_refuses_a_syllable_that_does_not_exist() -> None:
    assert place_stress("citˈab", 5) is None


def test_restress_sentence_fixes_words_and_keeps_the_rest() -> None:
    text = "Sən bu kitabı oxumusanmı?"
    phonemes = "sæn bu citabˈɯ oxumusanmˈɯ?"
    assert restress_sentence(text, phonemes, LEXICON) == "sæn bu citabˈɯ oxumˈusanmɯ?"


def test_restress_sentence_drops_the_accent_of_clitics() -> None:
    assert restress_sentence("Mən və sən", "mˈæn vˈæ sæn", LEXICON) == "mæn væ sæn"


def test_restress_sentence_leaves_a_misaligned_pair_unchanged() -> None:
    assert restress_sentence("bir iki", "bˈir", LEXICON) == "bˈir"


def test_restress_sentence_skips_words_whose_vowels_do_not_line_up() -> None:
    assert restress_sentence("gəlmədi", "ɟælmdˈi", LEXICON) == "ɟælmdˈi"


@pytest.mark.parametrize("word", ["aralıq", "iri", "aparan", "mərmərə", "əsər", "aman"])
def test_words_that_only_look_intensive_are_not_treated_as_intensive(word: str) -> None:
    assert stress_rule(word, LEXICON)[1] != "intensive"


@pytest.mark.parametrize(("word", "expected"), [("edirəm", 1), ("gedirəm", 1), ("deyirəm", 1), ("yeyirəm", 1), ("etmədim", 0), ("gedəcəyik", 2)])
def test_verb_stems_that_change_before_a_vowel(word: str, expected: int) -> None:
    assert stress(word) == expected


@pytest.mark.parametrize(("word", "expected"), [("deyilmi", 0), ("deyildir", 0), ("deyilsə", 0)])
def test_exception_words_keep_their_stress_before_unstressed_endings(word: str, expected: int) -> None:
    assert stress(word, question=True) == expected


def test_restress_sentence_ignores_standalone_dashes_that_espeak_drops() -> None:
    text = "Dövlətlər - xanlıqlar yarandı."
    phonemes = "dœvlætlˈær xanɫɯɡɫˈar jaɾandˈɯ."
    assert restress_sentence(text, phonemes, LEXICON) == phonemes
    assert restress_sentence("Sonra - gəlmədi.", "sonrˈa ɟælmædˈi.", LEXICON) == "sˈonra ɟˈælmædi."


@pytest.mark.parametrize(("word", "expected"), [("ildə", 1), ("ölkə", 1), ("şəkildə", 2), ("istifadə", 3), ("edə", 1), ("gecə", 1), ("məsələ", 2), ("düşüncə", 2)])
def test_words_ending_in_schwa_are_stressed_at_the_end(word: str, expected: int) -> None:
    assert stress(word) == expected


@pytest.mark.parametrize("word", ["necə", "belə", "hələ", "mənə", "sənə", "bizə", "görə", "niyə", "neçə", "öncə"])
def test_schwa_words_that_really_have_an_earlier_stress_keep_espeak(word: str) -> None:
    assert stress(word) is None


@pytest.mark.parametrize(("word", "expected"), [("sizcə", 0), ("sadəcə", 1), ("ürəyincə", 2), ("təkcə", 0)])
def test_the_adverbial_ending_ca_is_unstressed(word: str, expected: int) -> None:
    assert stress(word) == expected


def test_the_comitative_keeps_the_stress_before_it() -> None:
    assert stress("sürətlə") == 1


def test_lenkeran_and_derived_adjectives_are_final() -> None:
    assert stress("Lənkəran") is None
    assert stress("Rusiyalı") is None


def test_place_names_keep_their_stress_with_case_endings() -> None:
    assert restress_sentence("Mən Türkiyədə yaşayıram.", "mæn tyrcˈijædæ jaʃajˈɯɾam.", LEXICON) == "mæn tˈyrcijædæ jaʃajˈɯɾam."


def test_a_bare_negative_imperative_at_the_end_of_a_sentence_stresses_the_stem() -> None:
    lexicon = Lexicon(verbs=frozenset({"tərpət", "qidalan"}), lemmas=frozenset(), forms=frozenset())
    assert stress_rule("tərpətmə", lexicon, sentence_final=True)[0] == 1
    assert stress_rule("tərpətmə", lexicon)[0] == 2
    assert restress_sentence("Qidalanma, idman.", "ɡidaɫanmˈa, idmˈan.", lexicon) == "ɡidaɫanmˈa, idmˈan."


def test_ganja_is_final() -> None:
    assert stress("Gəncə") == 1
    assert stress("Gəncədə") == 2


def test_restress_sentence_handles_a_comma_that_espeak_keeps_as_its_own_token() -> None:
    assert restress_sentence("Sonra , gəlmədi", "sonrˈa , ɟælmædˈi", LEXICON) == "sˈonra , ɟˈælmædi"


def test_restress_sentence_refuses_a_pairing_where_many_vowel_counts_differ() -> None:
    assert restress_sentence("sonra gəlmədi indi", "sonrˈa ɟˈæl ˈi", LEXICON) == "sonrˈa ɟˈæl ˈi"


@pytest.mark.parametrize(
    ("word", "expected"),
    [("nəticəsində", 3), ("sayəsində", 2), ("ərzində", 1), ("üzərində", 2), ("içində", 1), ("haqqında", 1),
     ("arasında", 2), ("tərəfindən", 2), ("üzərinə", 2), ("altından", 1)],
)
def test_postpositions_stress_the_syllable_before_the_case_ending(word: str, expected: int) -> None:
    assert stress(word) == expected


def test_ordinary_nouns_with_the_same_endings_stay_final() -> None:
    assert stress("şəhərində") == 3


@pytest.mark.parametrize(("word", "expected"), [("burada", 0), ("harada", 0), ("orada", 0)])
def test_place_words_keep_the_first_syllable(word: str, expected: int) -> None:
    assert stress(word) == expected
