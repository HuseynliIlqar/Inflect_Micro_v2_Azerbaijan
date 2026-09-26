"""Azerbaijani text normalisation for the TTS frontend.

The adaptation model was trained on ordinary prose, so digits, Roman numerals,
abbreviations and symbols reach it as bare characters and come out wrong --
``II`` is spoken as "ı ı", ``25%`` drops the percent sign entirely.  This module
rewrites such spans into the words a reader would actually say, and is meant to
run *before* the deployment frontend turns text into phonemes.

Every function takes a string and returns a new one; nothing is mutated.

    >>> normalize_az("II qrupda 25% artım oldu.")
    'İkinci qrupda iyirmi beş faiz artım oldu.'
"""

from __future__ import annotations

import re
from types import MappingProxyType

from num2words import num2words

__all__ = [
    "normalize_az",
    "spell_acronym",
    "az_capitalise",
    "roman_to_int",
    "number_to_words",
    "ordinal_to_words",
]

# Azerbaijani letter names, used when an acronym is read one letter at a time.
LETTER_NAMES = MappingProxyType(
    {
        "a": "a", "b": "be", "c": "ce", "ç": "çe", "d": "de", "e": "e",
        "ə": "ə", "f": "fe", "g": "ge", "ğ": "ğe", "h": "he", "x": "xe",
        "ı": "ı", "i": "i", "j": "je", "k": "ka", "q": "qe", "l": "el",
        "m": "em", "n": "en", "o": "o", "ö": "ö", "p": "pe", "r": "er",
        "s": "se", "ş": "şe", "t": "te", "u": "u", "ü": "ü", "v": "ve",
        "y": "ye", "z": "ze",
    }
)

MONTHS = (
    "yanvar", "fevral", "mart", "aprel", "may", "iyun",
    "iyul", "avqust", "sentyabr", "oktyabr", "noyabr", "dekabr",
)

# Written abbreviation -> spoken form.  Applied before anything else so their
# full stops never look like sentence boundaries to the chunker.
ABBREVIATIONS = MappingProxyType(
    {
        "və s.": "və sairə",
        "və b.": "və başqaları",
        "və i.a.": "və iləaxır",
        "məs.": "məsələn",
        "e.ə.": "eramızdan əvvəl",
        "b.e.ə.": "bizim eradan əvvəl",
        "prof.": "professor",
        "dos.": "dosent",
        "akad.": "akademik",
        "dr.": "doktor",
        "küç.": "küçə",
        "şəh.": "şəhər",
        "r-nu": "rayonu",
        "mkr.": "mikrorayon",
        "səh.": "səhifə",
        "bax:": "bax,",
    }
)

# Unit abbreviation -> spoken form, matched only when a number precedes it.
UNITS = MappingProxyType(
    {
        "kq": "kiloqram", "q": "qram", "t": "ton", "mq": "milliqram",
        "km": "kilometr", "m": "metr", "sm": "santimetr", "mm": "millimetr",
        "l": "litr", "ml": "millilitr",
        "san": "saniyə", "dəq": "dəqiqə", "st": "ədəd",
        "kv.m": "kvadrat metr", "kub.m": "kub metr", "ha": "hektar",
        "kvt": "kilovatt", "vt": "vatt", "hz": "herts",
        "gb": "giqabayt", "mb": "meqabayt", "kb": "kilobayt", "tb": "terabayt",
        "°c": "dərəcə selsi", "°f": "dərəcə farenheyt",
    }
)

CURRENCIES = MappingProxyType(
    {
        "azn": "manat", "₼": "manat",
        "usd": "dollar", "$": "dollar",
        "eur": "avro", "€": "avro",
        "gbp": "funt sterlinq", "£": "funt sterlinq",
        "rub": "rubl", "₽": "rubl",
        "try": "Türkiyə lirəsi", "₺": "Türkiyə lirəsi",
    }
)

SYMBOLS = MappingProxyType(
    {
        "&": " və ", "№": " nömrə ", "§": " paraqraf ",
        "×": " vurulsun ", "÷": " bölünsün ", "±": " artı mənfi ",
        "≈": " təxminən ", "≤": " kiçik və ya bərabər ",
        "≥": " böyük və ya bərabər ", "=": " bərabərdir ",
    }
)

# Acronyms with a settled spoken form; anything else short and upper-case is
# spelled out letter by letter.
ACRONYMS = MappingProxyType(
    {
        "AMEA": "a em e a",
        "ADA": "ada",
        "ASAN": "asan",
        "NATO": "nato",
        "UNESCO": "yunesko",
        "UNICEF": "yunisef",
        "COVID": "kovid",
        "PIN": "pin",
        "SIM": "sim",
    }
)

ORDINAL_SUFFIXES = frozenset(
    {
        "cı", "ci", "cu", "cü",
        "ncı", "nci", "ncu", "ncü",
        "ıncı", "inci", "uncu", "üncü",
    }
)

_ROMAN_VALUES = MappingProxyType({"I": 1, "V": 5, "X": 10, "L": 50})
# Deliberately limited to I/V/X/L so ordinary upper-case words (DVD, MIX, CD)
# are never mistaken for numerals.
_ROMAN_RE = re.compile(r"^(?=[IVXL]{2,})(XL|L?X{0,4})(IX|IV|V?I{0,3})$")
_ROMAN_MAX = 50

# Optional Azerbaijani suffix written after a dash: "2024-cü", "ATM-də".
_SUFFIX = r"(?:-([^\W\d_][\wəçğıöşü]*))?"


def az_lower(text: str) -> str:
    """Lower-case with the Azerbaijani dotted/dotless ``i`` rules."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def az_capitalise(text: str) -> str:
    """Capitalise the first letter; ``i`` becomes ``İ``, not ``I``."""
    if not text:
        return text
    head = "İ" if text[0] == "i" else text[0].upper()
    return head + text[1:]


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


def number_to_words(value: int) -> str:
    return num2words(value, lang="az")


def ordinal_to_words(value: int) -> str:
    return num2words(value, lang="az", to="ordinal")


def spell_acronym(token: str) -> str:
    """Read an acronym one letter at a time: ``ATM`` -> ``a te em``."""
    return " ".join(LETTER_NAMES.get(letter, letter) for letter in az_lower(token))


def _attach_suffix(word: str, suffix: str | None) -> str:
    """Glue a written suffix onto a spoken word: ``doqquz`` + ``da`` -> ``doqquzda``.

    The suffix already carries the right vowel harmony because the writer chose
    it, so no morphology is needed here -- only the dash has to disappear.
    """
    return word if not suffix else f"{word}{suffix}"


def _decimal_to_words(whole: str, fraction: str) -> str:
    """``1,5`` -> ``bir tam beş onda``; falls back to digits when very long."""
    places = {1: "onda", 2: "yüzdə", 3: "mində"}.get(len(fraction))
    head = number_to_words(int(whole))
    if places is None:
        digits = " ".join(number_to_words(int(d)) for d in fraction)
        return f"{head} tam {digits}"
    return f"{head} tam {number_to_words(int(fraction))} {places}"


def _strip_group_separators(text: str) -> str:
    """``1 000 000`` and ``1.000.000`` become plain digit runs."""
    text = re.sub(r"(?<=\d)[  ](?=\d{3}\b)", "", text)
    return re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", text)


def _replace_abbreviations(text: str) -> str:
    for written, spoken in ABBREVIATIONS.items():
        text = re.sub(rf"(?<!\w){re.escape(written)}", spoken, text, flags=re.IGNORECASE)
    return text


def _replace_phone_numbers(text: str) -> str:
    def speak(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        return " ".join(number_to_words(int(digit)) for digit in digits)

    return re.sub(r"\+\d[\d\s\-()]{7,}\d", speak, text)


def _replace_dates(text: str) -> str:
    def speak(match: re.Match[str]) -> str:
        day, month, year = (int(part) for part in match.group(1, 2, 3))
        suffix = match.group(4)
        if not (1 <= day <= 31 and 1 <= month <= 12):
            return match.group(0)
        if suffix and az_lower(suffix) in ORDINAL_SUFFIXES:
            spoken_year = ordinal_to_words(year)
        else:
            spoken_year = _attach_suffix(number_to_words(year), suffix)
        return f"{ordinal_to_words(day)} {MONTHS[month - 1]} {spoken_year}"

    return re.sub(
        rf"\b(\d{{1,2}})[./](\d{{1,2}})[./](\d{{4}}){_SUFFIX}\b", speak, text
    )


def _replace_times(text: str) -> str:
    def speak(match: re.Match[str]) -> str:
        hour, minute = int(match.group(1)), int(match.group(2))
        suffix = match.group(3)
        if hour > 23 or minute > 59:
            return match.group(0)
        if minute == 0:
            return _attach_suffix(number_to_words(hour), suffix)
        tail = _attach_suffix(number_to_words(minute), suffix)
        if minute < 10:
            return f"{number_to_words(hour)} sıfır {tail}"
        return f"{number_to_words(hour)} {tail}"

    return re.sub(rf"\b(\d{{1,2}}):(\d{{2}}){_SUFFIX}\b", speak, text)


def _replace_currency(text: str) -> str:
    symbols = "".join(k for k in CURRENCIES if not k.isalpha())
    codes = "|".join(k for k in CURRENCIES if k.isalpha())
    amount = r"\d+(?:[.,]\d+)?"

    def speak_number(raw: str) -> str:
        if "," in raw or "." in raw:
            whole, fraction = re.split(r"[.,]", raw, maxsplit=1)
            return _decimal_to_words(whole, fraction)
        return number_to_words(int(raw))

    text = re.sub(
        rf"([{re.escape(symbols)}])\s?({amount})",
        lambda m: f"{speak_number(m.group(2))} {CURRENCIES[m.group(1)]}",
        text,
    )
    return re.sub(
        rf"\b({amount})\s?({codes}|[{re.escape(symbols)}])\b",
        lambda m: f"{speak_number(m.group(1))} {CURRENCIES[az_lower(m.group(2))]}",
        text,
        flags=re.IGNORECASE,
    )


def _replace_percent(text: str) -> str:
    text = re.sub(r"%\s?(\d+)", lambda m: f"{number_to_words(int(m.group(1)))} faiz", text)
    return re.sub(r"(\d+)\s?%", lambda m: f"{number_to_words(int(m.group(1)))} faiz", text)


def _replace_units(text: str) -> str:
    keys = sorted(UNITS, key=len, reverse=True)
    pattern = "|".join(re.escape(key) for key in keys)

    def speak(match: re.Match[str]) -> str:
        return f"{match.group(1)} {UNITS[az_lower(match.group(2))]}"

    return re.sub(
        rf"(\d)\s?({pattern})(?![\wə])", speak, text, flags=re.IGNORECASE
    )


def _replace_roman(text: str) -> str:
    def speak(match: re.Match[str]) -> str:
        value = roman_to_int(match.group(1))
        if value is None:
            return match.group(0)
        word = ordinal_to_words(value)
        return az_capitalise(word) if match.group(0)[0].isupper() else word

    return re.sub(r"\b([IVXL]{2,})\b(?:-(?:cı|ci|cu|cü))?", speak, text)


def _replace_suffixed_numbers(text: str) -> str:
    """``5-ci`` -> ``beşinci``; ``10-da`` -> ``onda`` (the suffix already harmonises)."""

    def speak(match: re.Match[str]) -> str:
        value, suffix = int(match.group(1)), match.group(2)
        if az_lower(suffix) in ORDINAL_SUFFIXES:
            return ordinal_to_words(value)
        return f"{number_to_words(value)}{suffix}"

    return re.sub(r"\b(\d+)-([^\W\d_][\wəçğıöşü]*)", speak, text)


def _replace_ranges(text: str) -> str:
    return re.sub(
        r"\b(\d+)\s?[-–]\s?(\d+)\b",
        lambda m: f"{number_to_words(int(m.group(1)))} ilə "
        f"{number_to_words(int(m.group(2)))}",
        text,
    )


def _replace_bare_numbers(text: str) -> str:
    def speak(match: re.Match[str]) -> str:
        raw = match.group(0)
        if "," in raw or "." in raw:
            whole, fraction = re.split(r"[.,]", raw, maxsplit=1)
            return _decimal_to_words(whole, fraction)
        return number_to_words(int(raw))

    return re.sub(r"-?\d+(?:[.,]\d+)?", speak, text)


def _replace_acronyms(text: str) -> str:
    def speak(match: re.Match[str]) -> str:
        token, suffix = match.group(1), match.group(2)
        spoken = ACRONYMS.get(token) or spell_acronym(token)
        return _attach_suffix(spoken, suffix)

    return re.sub(rf"\b([A-ZƏÇĞİÖŞÜ]{{2,5}}){_SUFFIX}\b", speak, text)


def _replace_symbols(text: str) -> str:
    for symbol, spoken in SYMBOLS.items():
        text = text.replace(symbol, spoken)
    return text


def _tidy(text: str) -> str:
    """Straighten quotes and dashes, then collapse whitespace."""
    for source, target in (
        ("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"'),
        ("–", "-"), ("—", "-"), ("…", "."), (" ", " "),
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
        _replace_phone_numbers,
        _strip_group_separators,
        _replace_dates,
        _replace_times,
        _replace_currency,
        _replace_percent,
        _replace_units,
        _replace_roman,
        _replace_suffixed_numbers,
        _replace_ranges,
        _replace_bare_numbers,
        _replace_acronyms,
        _replace_symbols,
        _tidy,
    )
    for step in steps:
        text = step(text)
    return text
