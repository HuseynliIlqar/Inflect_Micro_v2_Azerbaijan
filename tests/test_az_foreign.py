"""Foreign words written in Latin spelling are said the Azerbaijani way.

Without this step eSpeak reads "Microsoft" by Azerbaijani letter rules
("midʒrosoft") and "Google" with a soft g. The spoken forms are language data in
``FOREIGN_WORDS`` (``aztts/az_tables.py``).
"""

from __future__ import annotations

import pytest

from aztts.az_tables import FOREIGN_WORDS
from aztts.az_text import normalize_az


class TestForeignWords:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Microsoft və Google şirkətləri rəqabət aparır.", "Maykrosoft və Qugl şirkətləri rəqabət aparır."),
            ("Bu videonu YouTube-da gördüm.", "Bu videonu Yutubda gördüm."),
            ("Mən Instagram-a şəkil yüklədim.", "Mən İnstaqrama şəkil yüklədim."),
            ("Yeni iPhone-u aldım.", "Yeni ayfonu aldım."),
            ("Wi-Fi şifrəsi nədir?", "Vayfay şifrəsi nədir?"),
            ("Sənə e-mail göndərdim.", "Sənə imeyl göndərdim."),
            ("Dərslər online keçirilir.", "Dərslər onlayn keçirilir."),
        ],
    )
    def test_spoken_form(self, text: str, expected: str) -> None:
        assert normalize_az(text) == expected

    def test_suffix_follows_the_spoken_word(self) -> None:
        # The writer harmonised "-a" to "Google"; the spoken "Qugl" keeps a back vowel.
        assert normalize_az("Google-a yazdım.") == "Qugla yazdım."
        # "Microsoft-un": the spoken form ends in a consonant, the suffix stays.
        assert normalize_az("Microsoft-un məhsulu") == "Maykrosoftun məhsulu"

    def test_capital_i_is_english_not_azerbaijani(self) -> None:
        # Lower-casing "Instagram" the Azerbaijani way would give "ınstagram"
        # and miss the table; the lookup must use English case rules.
        assert normalize_az("INSTAGRAM səhifəsi") == "İnstaqram səhifəsi"

    def test_lowercase_word_capitalised_at_sentence_start(self) -> None:
        assert normalize_az("iPhone yenidir.") == "Ayfon yenidir."

    def test_numbers_after_the_word_are_still_read(self) -> None:
        assert normalize_az("Yeni iPhone 15 aldım.") == "Yeni ayfon on beş aldım."

    @pytest.mark.parametrize(
        "text",
        [
            "Googlelamaq yeni sözdür.",      # longer word: not the brand
            "Qugl və Maykrosoft.",           # already Azerbaijani spelling
            "Bolt ilə getdim.",              # not in the table on purpose
        ],
    )
    def test_leaves_other_words_alone(self, text: str) -> None:
        assert normalize_az(text) == text

    def test_table_is_lowercase_latin(self) -> None:
        for written, spoken in FOREIGN_WORDS.items():
            assert written == written.lower() and written.isascii(), written
            assert spoken and spoken == spoken.strip(), written
