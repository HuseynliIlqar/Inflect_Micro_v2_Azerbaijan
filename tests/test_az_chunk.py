"""Tests for Azerbaijani sentence chunking."""

from __future__ import annotations

import pytest

from aztts.az_chunk import DEFAULT_MAX_WORDS, chunk_text

LONG = (
    "Payız gəlmişdi və şəhərin küçələri saralmış yarpaqlarla örtülmüşdü, "
    "amma hava hələ də isti idi və insanlar parklarda gəzişirdilər."
)


class TestBasics:
    def test_empty_input(self) -> None:
        assert chunk_text("") == ()
        assert chunk_text("   ") == ()

    def test_short_text_is_one_chunk(self) -> None:
        assert chunk_text("Salam, necəsən?") == ("Salam, necəsən?",)

    def test_rejects_zero_max_words(self) -> None:
        with pytest.raises(ValueError):
            chunk_text("Salam.", max_words=0)


class TestSentenceSplitting:
    def test_full_stops(self) -> None:
        assert chunk_text("Salam. Necəsən.") == ("Salam.", "Necəsən.")

    def test_question_and_exclamation(self) -> None:
        assert chunk_text("Getdin? Bəli! Yaxşı.") == ("Getdin?", "Bəli!", "Yaxşı.")

    def test_punctuation_is_kept(self) -> None:
        # The package derives its pause length from the trailing mark.
        for chunk in chunk_text("Salam. Necəsən?"):
            assert chunk[-1] in ".?"


class TestLongSentences:
    def test_every_chunk_respects_the_limit(self) -> None:
        for chunk in chunk_text(LONG):
            assert len(chunk.split()) <= DEFAULT_MAX_WORDS

    def test_no_words_are_lost(self) -> None:
        joined = " ".join(chunk_text(LONG))
        assert joined.split() == LONG.split()

    def test_comma_is_preferred_over_a_hard_cut(self) -> None:
        text = " ".join(["söz"] * 10) + ", " + " ".join(["kəlmə"] * 10)
        chunks = chunk_text(text)
        assert chunks[0].endswith(",")

    def test_conjunction_split_when_no_comma(self) -> None:
        text = " ".join(["söz"] * 10) + " və " + " ".join(["kəlmə"] * 10)
        chunks = chunk_text(text)
        assert any(chunk.startswith("və") for chunk in chunks)

    def test_hard_split_when_nothing_else_helps(self) -> None:
        text = " ".join(["söz"] * 40)
        chunks = chunk_text(text, max_words=10)
        assert len(chunks) == 4
        assert all(len(chunk.split()) == 10 for chunk in chunks)


class TestCustomLimit:
    @pytest.mark.parametrize("limit", [5, 10, 15, 25])
    def test_limit_is_honoured(self, limit: int) -> None:
        for chunk in chunk_text(LONG, max_words=limit):
            assert len(chunk.split()) <= limit

    def test_a_generous_limit_leaves_sentences_whole(self) -> None:
        assert len(chunk_text(LONG, max_words=100)) == 1


class TestWhitespace:
    def test_newlines_and_runs_are_collapsed(self) -> None:
        assert chunk_text("Salam\n\n  dünya") == ("Salam dünya",)

    def test_no_chunk_is_blank_or_padded(self) -> None:
        for chunk in chunk_text("Salam.   Necəsən?   "):
            assert chunk == chunk.strip()
            assert chunk
