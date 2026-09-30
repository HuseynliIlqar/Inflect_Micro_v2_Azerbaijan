"""Primitives shared by every Azerbaijani normalisation rule.

Casing with the dotted/dotless ``i``, vowel harmony for a re-attached suffix,
numbers as words, and the regex fragments the rules build their patterns from.
Every function takes a string and returns a new one; nothing is mutated.
"""

from __future__ import annotations

import re
from types import MappingProxyType

from num2words import num2words

# Optional Azerbaijani suffix written after a dash: "2024-cü", "ATM-də".
SUFFIX = r"(?:-([^\W\d_][\wəçğıöşü]*))?"

# A number, and a number or a range of them: "2,5", "10-15".
NUMBER = r"\d+(?:[.,]\d+)?"

_VOWELS = "aıoueəiöü"
_BACK_VOWELS = "aıou"
# The four-way vowel (ı/i/u/ü) that follows each vowel.
_FOUR_WAY = MappingProxyType(
    {"a": "ı", "ı": "ı", "o": "u", "u": "u", "e": "i", "ə": "i", "i": "i", "ö": "ü", "ü": "ü"}
)


def az_lower(text: str) -> str:
    """Lower-case with the Azerbaijani dotted/dotless ``i`` rules."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def az_capitalise(text: str) -> str:
    """Capitalise the first letter; ``i`` becomes ``İ``, not ``I``."""
    if not text:
        return text
    head = "İ" if text[0] == "i" else text[0].upper()
    return head + text[1:]


def either_case(word: str) -> str:
    """``iyun`` -> ``[iİ]yun``: a capital without re.IGNORECASE, whose handling
    of ``İ`` differs between Python and JavaScript."""
    head = word[0]
    upper = "İ" if head == "i" else head.upper()
    return f"[{head}{upper}]{word[1:]}"


def number_to_words(value: int) -> str:
    return num2words(value, lang="az")


def ordinal_to_words(value: int) -> str:
    return num2words(value, lang="az", to="ordinal")


def attach_suffix(word: str, suffix: str | None) -> str:
    """Glue a written suffix onto a spoken word: ``doqquz`` + ``da`` -> ``doqquzda``.

    The suffix already carries the right vowel harmony because the writer chose
    it, so no morphology is needed here -- only the dash has to disappear.
    """
    return word if not suffix else f"{word}{suffix}"


def harmonise(word: str, suffix: str | None) -> str:
    """Re-attach a written suffix to a *different* spoken word.

    ``10 AZN-dən`` is written against the letters but said as ``manatdan``, so
    the suffix's vowels follow the new word (a/ə two-way, ı/i/u/ü four-way) and
    the buffer consonant is added or dropped: ``faiz`` + ``a`` -> ``faizə``,
    ``manat`` + ``ya`` -> ``manata``.
    """
    if not suffix:
        return word
    base = az_lower(word)
    ending = az_lower(suffix)
    if base[-1] in _VOWELS and ending[0] in _VOWELS:
        ending = ("y" if ending[0] in "aə" else "n") + ending
    elif (
        base[-1] not in _VOWELS
        and len(ending) > 1
        and ending[0] in "yn"
        and ending[1] in _VOWELS
    ):
        ending = ending[1:]
    last = next((c for c in reversed(base) if c in _VOWELS), "a")
    harmonised = []
    for character in ending:
        if character in "aə":
            character = "a" if last in _BACK_VOWELS else "ə"
        elif character in "ıiuü":
            character = _FOUR_WAY[last]
        if character in _VOWELS:
            last = character
        harmonised.append(character)
    return word + "".join(harmonised)


def _decimal_to_words(whole: str, fraction: str) -> str:
    """``1,5`` -> ``bir tam onda beş``; falls back to digits when very long."""
    places = {1: "onda", 2: "yüzdə", 3: "mində"}.get(len(fraction))
    head = number_to_words(int(whole))
    if places is None:
        digits = " ".join(number_to_words(int(d)) for d in fraction)
        return f"{head} tam {digits}"
    return f"{head} tam {places} {number_to_words(int(fraction))}"


def speak_number(raw: str) -> str:
    """An integer, or a decimal written with ``,`` or ``.``."""
    if "," in raw or "." in raw:
        whole, fraction = re.split(r"[.,]", raw, maxsplit=1)
        return _decimal_to_words(whole, fraction)
    return number_to_words(int(raw))


def speak_digits(raw: str) -> str:
    """``007`` -> ``sıfır sıfır yeddi``: leading zeros are read, not dropped."""
    rest = raw.lstrip("0")
    zeros = ["sıfır"] * (len(raw) - len(rest))
    return " ".join(zeros + ([number_to_words(int(rest))] if rest else []))
