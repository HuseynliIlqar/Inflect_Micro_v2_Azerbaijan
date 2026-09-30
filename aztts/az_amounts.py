"""Amounts with something attached: money, percentages, units, fractions, ranges.

Each rule turns the attachment into a word and, where a suffix was written
against a symbol (``AZN-dən``, ``%-ə``), re-harmonises it onto that word.
The numbers themselves mostly stay digits here, so later steps in
``normalize_az()`` read them the same way as everywhere else.
"""

from __future__ import annotations

import re

from .az_tables import CURRENCIES, CURRENCY_SUBUNITS, ORDINAL_SUFFIXES, UNITS
from .az_words import (
    NUMBER,
    SUFFIX,
    attach_suffix,
    az_lower,
    harmonise,
    number_to_words,
    ordinal_to_words,
    speak_number,
)

_AMOUNT = rf"{NUMBER}(?:\s?[-–]\s?{NUMBER})?"
# A scale word written between an amount and its currency: "5 mln AZN".
_SCALE = r"(?:\s?(?:mln|mlrd|trln|milyon|milyard|trilyon|min)\.?)?"


def _money(amount: str, currency: str, suffix: str | None) -> str:
    """``19,99`` manat -> ``19 manat 99 qəpik``.  Digits stay digits, so ranges
    and plain decimals are read by the same steps as everywhere else."""
    coins = re.fullmatch(r"(\d+)[.,](\d{2})", amount)
    if not coins or currency not in CURRENCY_SUBUNITS:
        return f"{amount} {harmonise(currency, suffix)}"
    whole, cents = coins.group(1), int(coins.group(2))
    coin = f"{cents} {harmonise(CURRENCY_SUBUNITS[currency], suffix)}"
    if not cents:
        return f"{whole} {harmonise(currency, suffix)}"
    if not int(whole):
        return coin
    return f"{whole} {currency} {coin}"


def replace_currency(text: str) -> str:
    """Put the currency after the amount as a word."""
    symbols = re.escape("".join(k for k in CURRENCIES if not k.isalpha()))
    codes = "|".join(k for k in CURRENCIES if k.isalpha())

    text = re.sub(
        rf"([{symbols}])\s?({_AMOUNT}{_SCALE})",
        lambda m: _money(m.group(2), CURRENCIES[m.group(1)], None),
        text,
    )
    text = re.sub(
        rf"(?<![\w.,])({_AMOUNT}{_SCALE})\s?({codes}|[{symbols}]){SUFFIX}(?!\w)",
        lambda m: _money(m.group(1), CURRENCIES[az_lower(m.group(2))], m.group(3)),
        text,
        flags=re.IGNORECASE,
    )
    # A code on its own: "AZN ilə ödəniş".
    return re.sub(
        rf"\b(AZN|USD|EUR|GBP|RUB){SUFFIX}\b",
        lambda m: harmonise(CURRENCIES[az_lower(m.group(1))], m.group(2)),
        text,
    )


def replace_percent(text: str) -> str:
    text = re.sub(rf"%\s?({_AMOUNT})", r"\1 faiz", text)
    return re.sub(
        rf"({_AMOUNT})\s?%{SUFFIX}",
        lambda m: f"{m.group(1)} {harmonise('faiz', m.group(2))}",
        text,
    )


def replace_units(text: str) -> str:
    keys = sorted(UNITS, key=len, reverse=True)
    pattern = "|".join(re.escape(key) for key in keys)

    def speak(match: re.Match[str]) -> str:
        spoken = UNITS[az_lower(match.group(2))]
        return f"{match.group(1)} {harmonise(spoken, match.group(3))}"

    return re.sub(rf"(\d)\s?({pattern}){SUFFIX}(?!\w)", speak, text, flags=re.IGNORECASE)


def replace_fractions(text: str) -> str:
    """``3/4`` -> ``dörddə üç``; a numerator not smaller than the denominator
    is a division: ``24/7`` -> ``iyirmi dörd bölü yeddi``."""

    def speak(match: re.Match[str]) -> str:
        top, bottom = int(match.group(1)), int(match.group(2))
        suffix = match.group(3)
        if 0 < top < bottom:
            head = harmonise(number_to_words(bottom), "da")
            return f"{head} {attach_suffix(number_to_words(top), suffix)}"
        return f"{number_to_words(top)} bölü {attach_suffix(number_to_words(bottom), suffix)}"

    return re.sub(rf"(?<![\d/.,])(\d+)/(\d+)(?![\d/]|[.,]\d){SUFFIX}", speak, text)


def replace_ranges(text: str) -> str:
    """``10-15`` -> ``on ilə on beş``.  A suffix belongs to the second number:
    ``1941-1945-ci illər`` -> ``... qırx bir ilə ... qırx beşinci illər``."""

    def speak(match: re.Match[str]) -> str:
        first, second, suffix = match.group(1, 2, 3)
        if suffix and az_lower(suffix) in ORDINAL_SUFFIXES and second.isdigit():
            tail = ordinal_to_words(int(second))
        else:
            tail = attach_suffix(speak_number(second), suffix)
        return f"{speak_number(first)} ilə {tail}"

    return re.sub(
        rf"(?<![\w.,])({NUMBER})\s?[-–]\s?({NUMBER}){SUFFIX}\b", speak, text
    )
