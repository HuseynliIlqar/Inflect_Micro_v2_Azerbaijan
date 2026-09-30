"""Dates, times and phone numbers read the way a person says them aloud.

A date's day is a cardinal and its year an ordinal plus ``il``
(``01/09/1939`` -> ``bir sentyabr ... otuz doqquzuncu il``); a phone number is
read group by group with its zeros; a time is hours then minutes.  These run
early in ``normalize_az()`` because their digits would otherwise be taken for
decimals, ranges or fractions.
"""

from __future__ import annotations

import re

from .az_tables import MONTHS, ORDINAL_SUFFIXES
from .az_words import (
    SUFFIX,
    attach_suffix,
    az_lower,
    either_case,
    harmonise,
    number_to_words,
    ordinal_to_words,
    speak_digits,
)

_MONTH_RE = "|".join(either_case(month) for month in MONTHS)
# "il" right after a date already names the year: "01.09.1939 ildə".
_YEAR_WORD_RE = re.compile(r"\s+il(?:in|də|dən|i|ə)?\b")


def _phone_groups(digits: str) -> list[str]:
    """Split an unbroken run the way numbers are dictated: ``... 123 45 67``."""
    if len(digits) <= 3:
        return [digits] if digits else []
    if len(digits) == 4:
        return [digits[:2], digits[2:]]
    if len(digits) >= 7:
        return _phone_groups(digits[:-7]) + [digits[-7:-4], digits[-4:-2], digits[-2:]]
    return _phone_groups(digits[:-2]) + [digits[-2:]]


def replace_phone_numbers(text: str) -> str:
    """Read a phone number group by group, zeros included:
    ``050 123 45 67`` -> ``sıfır əlli yüz iyirmi üç qırx beş altmış yeddi``."""

    def speak(match: re.Match[str]) -> str:
        written = re.findall(r"\d+", match.group(0))
        groups = [group for run in written for group in _phone_groups(run)]
        return " ".join(speak_digits(group) for group in groups)

    text = re.sub(r"\+\d[\d\s\-()]{7,}\d", speak, text)
    return re.sub(
        r"(?<![\d+])\(?0\d{2}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\d)", speak, text
    )


def _speak_date(
    day: int, month: int, year: int | None, suffix: str | None, *, year_word_follows: bool
) -> str | None:
    """``1, 9, 1939`` -> ``bir sentyabr min doqquz yüz otuz doqquzuncu il``.

    The day is a cardinal ("bir sentyabr", never "birinci sentyabr") and the
    year an ordinal followed by ``il``, the way a date is read aloud.
    """
    if not (1 <= day <= 31 and 1 <= month <= 12):
        return None
    head = f"{number_to_words(day)} {MONTHS[month - 1]}"
    if year is None:
        return harmonise(head, suffix)
    spoken_year = ordinal_to_words(year)
    if suffix and az_lower(suffix) in ORDINAL_SUFFIXES:
        return f"{head} {spoken_year}"
    if suffix:
        return f"{head} {spoken_year} {harmonise('il', suffix)}"
    if year_word_follows:
        return f"{head} {spoken_year}"
    return f"{head} {spoken_year} il"


def replace_dates(text: str) -> str:
    def speak(
        match: re.Match[str], day: int, month: int, year: int | None, suffix: str | None
    ) -> str:
        follows = bool(_YEAR_WORD_RE.match(match.string, match.end()))
        spoken = _speak_date(day, month, year, suffix, year_word_follows=follows)
        return match.group(0) if spoken is None else spoken

    def iso(match: re.Match[str]) -> str:
        year, month, day, suffix = match.group(1, 2, 3, 4)
        return speak(match, int(day), int(month), int(year), suffix)

    def day_month_year(match: re.Match[str]) -> str:
        day, separator, month, year, suffix = match.group(1, 2, 3, 4, 5)
        if separator == "-" and len(year) != 4:
            return match.group(0)
        return speak(match, int(day), int(month), int(year), suffix)

    def day_month(match: re.Match[str]) -> str:
        return speak(match, int(match.group(1)), int(match.group(2)), None, match.group(3))

    text = re.sub(rf"(?<![\d.,])(\d{{4}})-(\d{{2}})-(\d{{2}})(?!\d){SUFFIX}\b", iso, text)
    text = re.sub(
        rf"(?<![\d.,/])(\d{{1,2}})([./-])(\d{{1,2}})\2(\d{{4}}|\d{{2}})(?!\d|[./]\d){SUFFIX}\b",
        day_month_year,
        text,
    )
    # "15/08": a zero-padded second part is a month, never a denominator.
    text = re.sub(rf"(?<![\d.,/])(\d{{1,2}})/(0[1-9])(?![\d/]){SUFFIX}", day_month, text)
    # "01 may": the zero is padding, not a digit to read.
    return re.sub(rf"(?<![\d.,])0(\d)(?=\s+(?:{_MONTH_RE}))", r"\1", text)


def replace_dotted_chains(text: str) -> str:
    """Versions and addresses, or a date that failed validation:
    ``1.2.3`` -> ``bir nöqtə iki nöqtə üç``."""
    return re.sub(
        r"(?<![\d.,])\d+(?:\.\d+){2,}(?!\d)",
        lambda m: " nöqtə ".join(number_to_words(int(p)) for p in m.group(0).split(".")),
        text,
    )


def replace_times(text: str) -> str:
    def speak(match: re.Match[str]) -> str:
        hour, minute = int(match.group(1)), int(match.group(2))
        suffix = match.group(3)
        if minute > 59 or hour > 24 or (hour == 24 and minute):
            return match.group(0)
        if minute == 0:
            spoken = "sıfır sıfır" if hour == 0 else number_to_words(hour)
            return attach_suffix(spoken, suffix)
        tail = attach_suffix(number_to_words(minute), suffix)
        if minute < 10:
            return f"{number_to_words(hour)} sıfır {tail}"
        return f"{number_to_words(hour)} {tail}"

    return re.sub(rf"\b(\d{{1,2}}):(\d{{2}}){SUFFIX}\b", speak, text)
