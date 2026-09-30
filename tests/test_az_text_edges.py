"""Edge cases of Azerbaijani text normalisation: how numbers are read aloud."""

from __future__ import annotations

import pytest

from aztts.az_text import harmonise, normalize_az


class TestDatesReadAloud:
    """A date is read "bir sentyabr ... otuz doqquzuncu il", never "birinci"."""

    @pytest.mark.parametrize(
        "source", ["01/09/1939", "01.09.1939", "1/9/1939", "01-09-1939", "1939-09-01"]
    )
    def test_every_written_form(self, source: str) -> None:
        assert normalize_az(source) == "bir sentyabr min doqquz yüz otuz doqquzuncu il"

    def test_the_reported_sentence(self) -> None:
        assert normalize_az(
            "II Dünya müharibəsi 01/09/1939 tarixində başladı və 25% artım oldu."
        ) == (
            "İkinci Dünya müharibəsi bir sentyabr min doqquz yüz otuz doqquzuncu il "
            "tarixində başladı və iyirmi beş faiz artım oldu."
        )

    def test_il_is_not_doubled(self) -> None:
        assert normalize_az("01.09.1939 ildə") == (
            "bir sentyabr min doqquz yüz otuz doqquzuncu ildə"
        )

    def test_case_suffix_moves_onto_il(self) -> None:
        assert normalize_az("01.09.1939-dan") == (
            "bir sentyabr min doqquz yüz otuz doqquzuncu ildən"
        )

    def test_ordinal_suffix_on_the_year(self) -> None:
        assert normalize_az("01/09/1939-cu ildə") == (
            "bir sentyabr min doqquz yüz otuz doqquzuncu ildə"
        )

    def test_two_digit_year(self) -> None:
        assert normalize_az("31/12/99") == "otuz bir dekabr doxsan doqquzuncu il"

    def test_day_and_month_only(self) -> None:
        assert normalize_az("15/08-də") == "on beş avqustda"

    def test_padded_day_before_a_month_name(self) -> None:
        assert normalize_az("01 may") == "bir may"
        assert normalize_az("01 İyun") == "bir İyun"

    def test_impossible_date_is_read_part_by_part(self) -> None:
        assert normalize_az("32.13.2020") == "otuz iki nöqtə on üç nöqtə iki min iyirmi"


class TestFractionsAndDecimals:
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("3,14", "üç tam yüzdə on dörd"),
            ("0.001", "sıfır tam mində bir"),
            ("1.000,5", "min tam onda beş"),
            ("1,000,000", "bir milyon"),
            ("1/2 hissə", "ikidə bir hissə"),
            ("3/4", "dörddə üç"),
            ("2/3-si", "üçdə ikisi"),
            ("24/7", "iyirmi dörd bölü yeddi"),
        ],
    )
    def test_read(self, source: str, expected: str) -> None:
        assert normalize_az(source) == expected


class TestRangesAndSuffixes:
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("10-15%", "on ilə on beş faiz"),
            ("2,5% artım", "iki tam onda beş faiz artım"),
            ("25%-ə qədər", "iyirmi beş faizə qədər"),
            (
                "1941-1945-ci illər",
                "min doqquz yüz qırx bir ilə min doqquz yüz qırx beşinci illər",
            ),
            ("2,5-3,5 kq", "iki tam onda beş ilə üç tam onda beş kiloqram"),
            ("2,5-dən çox", "iki tam onda beşdən çox"),
            ("5 km-dən", "beş kilometrdən"),
            ("10 km/saat", "on kilometr saatda"),
        ],
    )
    def test_read(self, source: str, expected: str) -> None:
        assert normalize_az(source) == expected


class TestCurrencyEdges:
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("5$", "beş dollar"),
            ("10 AZN-dən", "on manatdan"),
            ("10 AZN-ə", "on manata"),
            ("3 mlrd AZN", "üç milyard manat"),
            ("$2,5 mln", "iki tam onda beş milyon dollar"),
            ("5-10 AZN", "beş ilə on manat"),
            ("AZN ilə", "manat ilə"),
        ],
    )
    def test_read(self, source: str, expected: str) -> None:
        assert normalize_az(source) == expected

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("19,99 AZN", "on doqquz manat doxsan doqquz qəpik"),
            ("19,99 AZN-dən", "on doqquz manat doxsan doqquz qəpikdən"),
            ("$1.50", "bir dollar əlli sent"),
            ("0,50 AZN", "əlli qəpik"),
            ("5,00 AZN", "beş manat"),
            ("12,5 AZN", "on iki tam onda beş manat"),
        ],
    )
    def test_two_decimals_are_coins(self, source: str, expected: str) -> None:
        assert normalize_az(source) == expected


class TestHarmonise:
    @pytest.mark.parametrize(
        ("word", "suffix", "expected"),
        [
            ("manat", "dən", "manatdan"),
            ("manat", "ya", "manata"),
            ("faiz", "a", "faizə"),
            ("avro", "a", "avroya"),
            ("il", "dan", "ildən"),
            ("metr", "un", "metrin"),
            ("dollar", None, "dollar"),
        ],
    )
    def test_suffix_follows_the_spoken_word(
        self, word: str, suffix: str | None, expected: str
    ) -> None:
        assert harmonise(word, suffix) == expected


class TestNumberEdges:
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("007", "sıfır sıfır yeddi"),
            ("Su-27", "Su iyirmi yeddi"),
            ("COVID-19", "kovid on doqquz"),
            ("+5 dərəcə", "üstəgəl beş dərəcə"),
            ("30° isti", "otuz dərəcə isti"),
            ("Hesab 3:2 oldu.", "Hesab üç iki oldu."),
            ("2x3", "iki vurulsun üç"),
            ("24:00", "iyirmi dörd"),
            ("00:00", "sıfır sıfır"),
            ("1.2.3 versiyası", "bir nöqtə iki nöqtə üç versiyası"),
        ],
    )
    def test_read(self, source: str, expected: str) -> None:
        assert normalize_az(source) == expected


class TestLoneRomanNumerals:
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("I Pyotr", "Birinci Pyotr"),
            ("V əsrdə", "Beşinci əsrdə"),
            ("Mən I qrupdayam.", "Mən birinci qrupdayam."),
            ("V-ci", "Beşinci"),
            ("XIX-XX əsrlər", "On doqquzuncu ilə iyirminci əsrlər"),
        ],
    )
    def test_numeral(self, source: str, expected: str) -> None:
        assert normalize_az(source) == expected

    @pytest.mark.parametrize("source", ["V hərfi", "X şirkəti", "I love"])
    def test_letter_stays_a_letter(self, source: str) -> None:
        assert normalize_az(source) == source


class TestIdempotence:
    @pytest.mark.parametrize(
        "source",
        ["01/09/1939", "10 AZN-dən", "3/4", "+994 50 123 45 67", "XIX-XX əsrlər"],
    )
    def test_normalising_twice_changes_nothing(self, source: str) -> None:
        once = normalize_az(source)
        assert normalize_az(once) == once
