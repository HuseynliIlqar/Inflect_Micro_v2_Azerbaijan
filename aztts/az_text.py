"""Azerbaijani text normalisation for the TTS frontend.

The adaptation model was trained on ordinary prose, so digits, Roman numerals,
abbreviations and symbols reach it as bare characters and come out wrong --
``II`` is spoken as "ı ı", ``25%`` drops the percent sign entirely.  This module
rewrites such spans into the words a reader would actually say, and is meant to
run *before* the deployment frontend turns text into phonemes.

``normalize_az()`` is the pipeline.  Its rules live in three places: dates,
times and phone numbers in ``az_dates.py``; money, percentages, units,
fractions and ranges in ``az_amounts.py``; everything else here.  The word
tables are in ``az_tables.py`` and the shared primitives in ``az_words.py``.

Every function takes a string and returns a new one; nothing is mutated.

    >>> normalize_az("II qrupda 25% artım oldu.")
    'İkinci qrupda iyirmi beş faiz artım oldu.'
"""

from __future__ import annotations

import re
from types import MappingProxyType

from .az_amounts import (
    replace_currency,
    replace_fractions,
    replace_percent,
    replace_ranges,
    replace_units,
)
from .az_dates import (
    replace_dates,
    replace_dotted_chains,
    replace_phone_numbers,
    replace_times,
)
from .az_tables import (
    ABBREVIATIONS,
    ACRONYMS,
    LETTER_NAMES,
    ORDINAL_SUFFIXES,
    ROMAN_NOUNS,
    SYMBOLS,
)
from .az_words import (
    NUMBER,
    SUFFIX,
    attach_suffix,
    az_capitalise,
    az_lower,
    harmonise,
    number_to_words,
    ordinal_to_words,
    speak_digits,
    speak_number,
)

__all__ = [
    "normalize_az",
    "spell_acronym",
    "az_lower",
    "az_capitalise",
    "roman_to_int",
    "number_to_words",
    "ordinal_to_words",
    "harmonise",
]

_ROMAN_VALUES = MappingProxyType({"I": 1, "V": 5, "X": 10, "L": 50})
_LONE_ROMAN = MappingProxyType({"I": 1, "V": 5, "X": 10})
# Deliberately limited to I/V/X/L so ordinary upper-case words (DVD, MIX, CD)
# are never mistaken for numerals.
_ROMAN_RE = re.compile(r"^(?=[IVXL]{2,})(XL|L?X{0,4})(IX|IV|V?I{0,3})$")
_ROMAN_MAX = 50


def roman_to_int(token: str) -> int | None:
    """Return the value of a Roman numeral, or ``None`` if it is not one."""
    if not _ROMAN_RE.match(token):
        return None
    total = 0
    previous = 0
    for character in reversed(token):
        value = _ROMAN_VALUES[character]
        total = total - value if value < previous else total + value
        previous = max(previous, value)
    return total if 0 < total <= _ROMAN_MAX else None


def spell_acronym(token: str) -> str:
    """Read an acronym one letter at a time: ``ATM`` -> ``a te em``."""
    return " ".join(LETTER_NAMES.get(letter, letter) for letter in az_lower(token))


def _at_sentence_start(text: str, index: int) -> bool:
    before = text[:index].rstrip()
    return not before or before[-1] in ".!?"


def _join_groups(match: re.Match[str]) -> str:
    return re.sub(r"[ .,]", "", match.group(0))


def _strip_group_separators(text: str) -> str:
    """``1 000 000``, ``1.000.000`` and ``1,000,000`` become plain digit runs.

    A leading ``0`` is never a thousands group (``0.001`` is a decimal), and a
    single ``,000`` stays a decimal comma, which is the Azerbaijani convention.
    """
    text = re.sub(r"(?<![\d.,])[1-9]\d{0,2}(?: \d{3})+(?!\d)", _join_groups, text)
    text = re.sub(r"(?<![\d.,])[1-9]\d{0,2}(?:\.\d{3})+(?!\d|\.\d)", _join_groups, text)
    return re.sub(r"(?<![\d.,])[1-9]\d{0,2}(?:,\d{3}){2,}(?!\d|,\d)", _join_groups, text)


def _replace_abbreviations(text: str) -> str:
    for written, spoken in ABBREVIATIONS.items():
        text = re.sub(rf"(?<!\w){re.escape(written)}", spoken, text, flags=re.IGNORECASE)
    return text


def _replace_operators(text: str) -> str:
    """A score or ratio (``3:2``) is two numbers; ``2x3`` is a product."""
    text = re.sub(r"(?<=\d):(?=\d)", " ", text)
    return re.sub(r"(?<=\d)\s?[xх×]\s?(?=\d)", " × ", text)


def _replace_roman(text: str) -> str:
    """``II`` -> ``ikinci``, capitalised only where a sentence starts."""

    def word_at(match: re.Match[str], value: int) -> str:
        word = ordinal_to_words(value)
        return az_capitalise(word) if _at_sentence_start(match.string, match.start()) else word

    def span(match: re.Match[str]) -> str:
        first = roman_to_int(match.group(1)) or _LONE_ROMAN.get(match.group(1))
        second = roman_to_int(match.group(2)) or _LONE_ROMAN.get(match.group(2))
        if first is None or second is None:
            return match.group(0)
        return f"{word_at(match, first)} ilə {ordinal_to_words(second)}"

    def numeral(match: re.Match[str]) -> str:
        value = roman_to_int(match.group(1))
        return match.group(0) if value is None else word_at(match, value)

    def lone(match: re.Match[str]) -> str:
        token, suffix, following = match.group(1), match.group(2), match.group(3) or ""
        is_numeral = (
            bool(suffix)
            or az_lower(following).startswith(ROMAN_NOUNS)
            or (token == "I" and following[:1].isupper())
        )
        return word_at(match, _LONE_ROMAN[token]) if is_numeral else match.group(0)

    ordinal = r"(?:-(?:cı|ci|cu|cü))?"
    text = re.sub(rf"\b([IVXL]+)\s?[-–]\s?([IVXL]+)\b{ordinal}", span, text)
    text = re.sub(rf"\b([IVXL]{{2,}})\b{ordinal}", numeral, text)
    return re.sub(r"(?<![\w.])([IVX])\b(-(?:cı|ci|cu|cü))?(?=(?:\s+([^\W\d_]+))?)", lone, text)


def _replace_suffixed_numbers(text: str) -> str:
    """``5-ci`` -> ``beşinci``; ``10-da`` -> ``onda`` (the suffix already harmonises)."""

    def speak(match: re.Match[str]) -> str:
        value, suffix = match.group(1), match.group(2)
        if az_lower(suffix) in ORDINAL_SUFFIXES and value.isdigit():
            return ordinal_to_words(int(value))
        return f"{speak_number(value)}{suffix}"

    return re.sub(rf"(?<![\w.,])({NUMBER})-([^\W\d_][\wəçğıöşü]*)", speak, text)


def _replace_bare_numbers(text: str) -> str:
    # "Su-27", "COVID-19": that dash joins a name to a number; it is no minus.
    text = re.sub(r"(?<=[^\W\d_])-(?=\d)", " ", text)
    text = re.sub(r"(?<![\w+])\+(?=\d)", "üstəgəl ", text)

    def speak(match: re.Match[str]) -> str:
        sign, raw = match.group(1), match.group(2)
        if raw.isdigit() and len(raw) > 1 and raw.startswith("0"):
            spoken = speak_digits(raw)
        else:
            spoken = speak_number(raw)
        return f"mənfi {spoken}" if sign else spoken

    return re.sub(rf"(?:(?<!\w)(-))?({NUMBER})", speak, text)


def _replace_acronyms(text: str) -> str:
    def speak(match: re.Match[str]) -> str:
        token, suffix = match.group(1), match.group(2)
        spoken = ACRONYMS.get(token) or spell_acronym(token)
        return attach_suffix(spoken, suffix)

    return re.sub(rf"\b([A-ZƏÇĞİÖŞÜ]{{2,5}}){SUFFIX}\b", speak, text)


def _replace_symbols(text: str) -> str:
    for symbol, spoken in SYMBOLS.items():
        text = text.replace(symbol, spoken)
    return text


def _tidy(text: str) -> str:
    """Straighten quotes and dashes, then collapse whitespace."""
    for source, target in (
        ("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"'),
        ("–", "-"), ("—", "-"), ("…", "."), (" ", " "),
    ):
        text = text.replace(source, target)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return " ".join(text.split())


def normalize_az(text: str) -> str:
    """Rewrite digits, numerals, units and abbreviations as spoken Azerbaijani.

    The order matters: abbreviations first so their full stops never look like
    sentence ends, then the number forms from most specific (dates, times) to
    least (bare digits), and acronyms last so Roman numerals win the ambiguity.
    """
    if not text or not text.strip():
        return ""
    steps = (
        _tidy,
        _replace_abbreviations,
        replace_phone_numbers,
        _strip_group_separators,
        replace_dates,
        replace_dotted_chains,
        replace_times,
        _replace_operators,
        replace_currency,
        replace_percent,
        replace_units,
        _replace_roman,
        replace_fractions,
        replace_ranges,
        _replace_suffixed_numbers,
        _replace_bare_numbers,
        _replace_acronyms,
        _replace_symbols,
        _tidy,
    )
    for step in steps:
        text = step(text)
    return text
