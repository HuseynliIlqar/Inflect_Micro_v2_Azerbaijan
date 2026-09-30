"""Tests for Azerbaijani text normalisation."""

from __future__ import annotations

import pytest

from aztts.az_text import (
    az_lower,
    harmonise,
    normalize_az,
    number_to_words,
    ordinal_to_words,
    roman_to_int,
    spell_acronym,
)


class TestRomanNumerals:
    @pytest.mark.parametrize(
        ("token", "expected"),
        [("II", 2), ("III", 3), ("IV", 4), ("IX", 9), ("XI", 11), ("XL", 40), ("L", None)],
    )
    def test_values(self, token: str, expected: int | None) -> None:
        assert roman_to_int(token) == expected

    @pytest.mark.parametrize("token", ["DVD", "CD", "MIX", "SMS", "", "IIII", "AB"])
    def test_rejects_non_numerals(self, token: str) -> None:
        assert roman_to_int(token) is None

    def test_single_letters_are_never_numerals(self) -> None:
        # "I" and "V" are far more likely to be letters than numbers.
        assert roman_to_int("I") is None
        assert roman_to_int("V") is None


class TestNumberWords:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0, "sıfır"), (5, "beş"), (15, "on beş"), (2024, "iki min iyirmi dörd")],
    )
    def test_cardinals(self, value: int, expected: str) -> None:
        assert number_to_words(value) == expected

    @pytest.mark.parametrize(
        ("value", "expected"), [(1, "birinci"), (2, "ikinci"), (5, "beşinci")]
    )
    def test_ordinals(self, value: int, expected: str) -> None:
        assert ordinal_to_words(value) == expected


class TestAzerbaijaniLowercase:
    def test_dotted_and_dotless_i(self) -> None:
        assert az_lower("İSTİ") == "isti"
        assert az_lower("ISIQ") == "ısıq"


class TestAcronyms:
    def test_spelled_letter_by_letter(self) -> None:
        assert spell_acronym("ATM") == "a te em"

    def test_known_acronym_uses_settled_form(self) -> None:
        assert "a em e a" in normalize_az("AMEA-nın qərarı")

    def test_unknown_acronym_is_spelled(self) -> None:
        assert normalize_az("ATM kartı") == "a te em kartı"

    def test_long_uppercase_word_is_left_alone(self) -> None:
        # Six letters or more is emphasis, not an acronym.
        assert normalize_az("DİQQƏTLİ ol") == "DİQQƏTLİ ol"


class TestRomanInText:
    def test_the_headline_case(self) -> None:
        assert normalize_az("II Dünya müharibəsi") == "İkinci Dünya müharibəsi"

    def test_lowercase_context_keeps_lowercase(self) -> None:
        assert normalize_az("fəsil XI") == "fəsil on birinci"

    def test_capital_only_at_a_sentence_start(self) -> None:
        assert normalize_az("bu XX əsrdə") == "bu iyirminci əsrdə"
        assert normalize_az("Bitdi. XX əsr") == "Bitdi. İyirminci əsr"

    def test_roman_with_ordinal_suffix(self) -> None:
        assert normalize_az("XX-ci əsr") == "İyirminci əsr"


class TestNumbers:
    def test_percent_after(self) -> None:
        assert normalize_az("25% artım") == "iyirmi beş faiz artım"

    def test_percent_before(self) -> None:
        assert normalize_az("%25 artım") == "iyirmi beş faiz artım"

    def test_decimal_with_comma(self) -> None:
        # The denominator comes first: "onda beş", five tenths.
        assert normalize_az("1,5 metr") == "bir tam onda beş metr"

    def test_group_separators_are_removed(self) -> None:
        assert normalize_az("1.000.000 nəfər") == "bir milyon nəfər"

    def test_negative_number(self) -> None:
        assert normalize_az("-3 dərəcə") == "mənfi üç dərəcə"

    def test_range(self) -> None:
        assert normalize_az("10-15 nəfər") == "on ilə on beş nəfər"


class TestSuffixedNumbers:
    def test_ordinal_suffix_becomes_ordinal_word(self) -> None:
        assert normalize_az("5-ci sıra") == "beşinci sıra"

    def test_case_suffix_is_kept_verbatim(self) -> None:
        # The written suffix already carries the right vowel harmony.
        assert normalize_az("10-da qaldı") == "onda qaldı"

    def test_year_with_ordinal_suffix(self) -> None:
        assert normalize_az("2024-cü ildə") == "iki min iyirmi dördüncü ildə"


class TestUnits:
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("5 kq", "beş kiloqram"),
            ("3 km", "üç kilometr"),
            ("250 ml", "iki yüz əlli millilitr"),
            ("16 GB", "on altı giqabayt"),
        ],
    )
    def test_units(self, source: str, expected: str) -> None:
        assert normalize_az(source) == expected

    def test_unit_letter_inside_a_word_is_untouched(self) -> None:
        assert normalize_az("5 mərtəbə") == "beş mərtəbə"


class TestCurrency:
    def test_code_after_amount(self) -> None:
        assert normalize_az("25 AZN") == "iyirmi beş manat"

    def test_symbol_before_amount(self) -> None:
        assert normalize_az("$25") == "iyirmi beş dollar"


class TestDatesAndTimes:
    def test_full_date(self) -> None:
        assert normalize_az("15.03.2024") == "on beş mart iki min iyirmi dördüncü il"

    def test_slash_date(self) -> None:
        assert normalize_az("01/12/1999") == "bir dekabr min doqquz yüz doxsan doqquzuncu il"

    def test_impossible_date_is_left_as_digits(self) -> None:
        assert "mart" not in normalize_az("40.03.2024")

    def test_time_on_the_hour(self) -> None:
        assert normalize_az("saat 09:00") == "saat doqquz"

    def test_time_with_minutes(self) -> None:
        assert normalize_az("saat 14:30") == "saat on dörd otuz"

    def test_time_with_leading_zero_minute(self) -> None:
        assert normalize_az("14:05") == "on dörd sıfır beş"


class TestAbbreviations:
    def test_expansion(self) -> None:
        assert normalize_az("alma, armud və s.") == "alma, armud və sairə"

    def test_abbreviation_period_is_not_a_sentence_end(self) -> None:
        result = normalize_az("Kitab, dəftər və s. gətir.")
        assert result == "Kitab, dəftər və sairə gətir."

    def test_meselen(self) -> None:
        assert normalize_az("məs. belə") == "məsələn belə"


class TestDanglingSuffixes:
    """A written "-suffix" must be absorbed, never left hanging on a dash."""

    def test_year_suffix_after_a_full_date(self) -> None:
        assert normalize_az("15.03.2024-cü ildə") == (
            "on beş mart iki min iyirmi dördüncü ildə"
        )

    def test_case_suffix_after_a_time(self) -> None:
        assert normalize_az("09:00-da") == "doqquzda"

    def test_case_suffix_after_a_time_with_minutes(self) -> None:
        assert normalize_az("14:30-da") == "on dörd otuzda"

    def test_case_suffix_after_an_acronym(self) -> None:
        assert normalize_az("ATM-də") == "a te emdə"

    def test_case_suffix_after_a_known_acronym(self) -> None:
        assert normalize_az("AMEA-nın") == "a em e anın"

    @pytest.mark.parametrize(
        "source", ["15.03.2024-cü ildə", "09:00-da", "ATM-də", "AMEA-nın"]
    )
    def test_no_dash_survives(self, source: str) -> None:
        assert "-" not in normalize_az(source)


class TestPhoneNumbers:
    def test_read_group_by_group(self) -> None:
        result = normalize_az("+994 50 123 45 67")
        assert result == "doqquz yüz doxsan dörd əlli yüz iyirmi üç qırx beş altmış yeddi"
        assert "min" not in result

    @pytest.mark.parametrize("source", ["050 123 45 67", "0501234567", "(050) 123-45-67"])
    def test_local_number_keeps_its_zero(self, source: str) -> None:
        assert normalize_az(source) == "sıfır əlli yüz iyirmi üç qırx beş altmış yeddi"


class TestSymbols:
    def test_ampersand(self) -> None:
        assert normalize_az("Ali & Vəli") == "Ali və Vəli"

    def test_number_sign(self) -> None:
        assert normalize_az("№ 5") == "nömrə beş"


class TestTidying:
    def test_curly_quotes_and_ellipsis(self) -> None:
        assert normalize_az("“Salam…”") == '"Salam."'

    def test_whitespace_is_collapsed(self) -> None:
        assert normalize_az("  Salam   dünya  ") == "Salam dünya"

    def test_space_before_punctuation_is_removed(self) -> None:
        assert normalize_az("Salam , dünya") == "Salam, dünya"

    @pytest.mark.parametrize("source", ["", "   ", "\n"])
    def test_empty_input(self, source: str) -> None:
        assert normalize_az(source) == ""


class TestIdempotence:
    @pytest.mark.parametrize(
        "source",
        [
            "II Dünya müharibəsi",
            "25% artım oldu",
            "5-ci sıra",
            "Salam, necəsən?",
        ],
    )
    def test_normalising_twice_changes_nothing(self, source: str) -> None:
        once = normalize_az(source)
        assert normalize_az(once) == once


class TestPlainTextIsUntouched:
    @pytest.mark.parametrize(
        "source",
        [
            "Salam, bu gün hava çox gözəldir.",
            "Sən bu kitabı oxumusan?",
            "Payız gəlmişdi və küçələr saralmış yarpaqlarla örtülmüşdü.",
        ],
    )
    def test_ordinary_prose_survives_unchanged(self, source: str) -> None:
        assert normalize_az(source) == source
