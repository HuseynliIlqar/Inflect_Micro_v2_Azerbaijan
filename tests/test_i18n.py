"""The two label tables must stay in step."""

from __future__ import annotations

import pytest

from webui import i18n


def test_both_languages_have_the_same_keys() -> None:
    azerbaijani = set(i18n.table("az"))
    english = set(i18n.table("en"))
    assert azerbaijani == english, {
        "only in az": sorted(azerbaijani - english),
        "only in en": sorted(english - azerbaijani),
    }


@pytest.mark.parametrize("language", ("az", "en"))
def test_no_label_is_empty(language: str) -> None:
    empty = [key for key, text in i18n.table(language).items() if not text.strip()]
    assert not empty


@pytest.mark.parametrize("language", ("az", "en"))
def test_every_language_has_a_display_name(language: str) -> None:
    assert i18n.LANGUAGE_NAMES[language].strip()


def test_the_default_language_exists() -> None:
    assert i18n.DEFAULT_LANGUAGE in i18n.LANGUAGES


def test_an_unknown_language_falls_back_to_the_default() -> None:
    assert i18n.table("de") is i18n.table(i18n.DEFAULT_LANGUAGE)


def test_an_unknown_key_echoes_itself_instead_of_raising() -> None:
    assert i18n.label("az", "no_such_label") == "no_such_label"


@pytest.mark.parametrize("language", ("az", "en"))
def test_the_statistics_line_formats(language: str) -> None:
    line = i18n.label(
        language,
        "stats_template",
        seconds=1.5,
        elapsed=0.5,
        realtime=3.0,
        chunks=2,
        rate=24_000,
    )
    assert "{" not in line
    assert "3.0" in line and "24000" in line


@pytest.mark.parametrize("language", ("az", "en"))
def test_a_missing_format_argument_does_not_crash(language: str) -> None:
    assert i18n.label(language, "error_failed", wrong_name="x")


@pytest.mark.parametrize("language", ("az", "en"))
def test_every_prosody_level_has_a_label(language: str) -> None:
    for level in ("off", "safe", "wide"):
        key = f"prosody_{level}"
        assert i18n.label(language, key) != key


def test_the_tables_are_read_only() -> None:
    with pytest.raises(TypeError):
        i18n.table("az")["title"] = "changed"  # type: ignore[index]


@pytest.mark.parametrize("language", ("az", "en"))
def test_the_chunk_count_is_formatted(language: str) -> None:
    assert "1" in i18n.chunk_count(language, 1)
    assert "7" in i18n.chunk_count(language, 7)


def test_english_pluralises_the_chunk_count() -> None:
    assert i18n.chunk_count("en", 1) == "1 chunk"
    assert i18n.chunk_count("en", 2) == "2 chunks"


def test_azerbaijani_does_not_pluralise_after_a_numeral() -> None:
    assert i18n.chunk_count("az", 1) == "1 hissə"
    assert i18n.chunk_count("az", 2) == "2 hissə"


@pytest.mark.parametrize("language", ("az", "en"))
def test_the_voice_selector_offers_both_voices(language: str) -> None:
    choices = i18n.voice_choices(language, True)
    assert [value for _, value in choices] == ["az", "en"]
    assert all(text.strip() for text, _ in choices)


@pytest.mark.parametrize("language", ("az", "en"))
def test_a_missing_english_model_is_said_out_loud(language: str) -> None:
    available, missing = (
        i18n.voice_choices(language, True)[1][0],
        i18n.voice_choices(language, False)[1][0],
    )
    assert missing != available
    assert len(missing) > len(available)
    assert "\n" not in missing


@pytest.mark.parametrize("language", ("az", "en"))
def test_the_download_instruction_is_a_runnable_command(language: str) -> None:
    assert "training/scripts/download_model.py" in i18n.label(
        language, "voice_en_missing"
    )


@pytest.mark.parametrize("language", ("az", "en"))
def test_the_git_lfs_hint_names_the_command(language: str) -> None:
    assert "git lfs pull" in i18n.label(language, "voice_en_pointer")
