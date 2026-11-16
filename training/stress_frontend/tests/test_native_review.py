"""The native speaker's choices on 130 words must all come out of the frontend."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from az_stress import WEAK_ACCENT, Lexicon, stress_rule, word_context  # noqa: E402

HERE = Path(__file__).resolve().parent
LEXICON_DIR = HERE.parent / "lexicon"


def _read(name: str) -> frozenset[str]:
    return frozenset((LEXICON_DIR / f"{name}.txt").read_text(encoding="utf-8").split())


def _syllable(form: str) -> int:
    parts = form.split("-")
    lower = lambda s: s.translate(str.maketrans({"I": "ı", "İ": "i"})).lower()  # noqa: E731
    return next(index for index, part in enumerate(parts) if part != lower(part))


def _cases() -> list[tuple[str, str, int, int]]:
    items = {item["word"]: item for item in json.loads((HERE / "stress_items.json").read_text(encoding="utf-8"))}
    key = json.loads((HERE / "stress_key.json").read_text(encoding="utf-8"))
    cases = []
    for line in (HERE.parent / "native_review" / "review_answers.txt").read_text(encoding="utf-8").split():
        word = line.replace("-", "").translate(str.maketrans({"I": "ı", "İ": "i"})).lower()
        item = items[word]
        order = key[item["id"]]["order"]
        espeak = _syllable(item["options"][order.index("espeak")])
        cases.append((item["sentence"], word, espeak, _syllable(line)))
    return cases


LEXICON = Lexicon(verbs=_read("verbs"), lemmas=_read("lemmas"), forms=_read("forms"))


@pytest.mark.parametrize(("sentence", "word", "espeak", "expected"), _cases())
def test_frontend_agrees_with_the_native_speaker(sentence: str, word: str, espeak: int, expected: int) -> None:
    context = word_context(sentence, word)
    syllable, _ = stress_rule(word, LEXICON, **context)
    assert (espeak if syllable is None else syllable) == expected


def _second_cases() -> list[tuple[str, str, int, int]]:
    folder = HERE.parent / "native_review" / "review2"
    items = {item["word"]: item for item in json.loads((folder / "stress_items.json").read_text(encoding="utf-8"))}
    key = json.loads((folder / "stress_key.json").read_text(encoding="utf-8"))
    cases = []
    for line in (HERE.parent / "native_review" / "review2_answers.txt").read_text(encoding="utf-8").split():
        word = line.replace("-", "").translate(str.maketrans({"I": "ı", "İ": "i"})).lower()
        item = items[word]
        cases.append((item["sentence"], word, key[item["id"]]["espeak"], _syllable(line)))
    return cases


@pytest.mark.parametrize(("sentence", "word", "espeak", "expected"), _second_cases())
def test_frontend_agrees_with_the_second_review(sentence: str, word: str, espeak: int | None, expected: int) -> None:
    if espeak is None:
        # "ilə" carries only eSpeak's weak accent on its first syllable, which the frontend keeps.
        assert word in WEAK_ACCENT and expected == 0
        return
    context = word_context(sentence, word)
    syllable, _ = stress_rule(word, LEXICON, **context)
    assert (espeak if syllable is None else syllable) == expected
