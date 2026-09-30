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
    "harmonise",
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
        "km/saat": "kilometr saatda", "km/s": "kilometr saniyədə",
        "m/s": "metr saniyədə",
        "mln": "milyon", "mlrd": "milyard", "trln": "trilyon",
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

# The coin an amount's two decimal places count: "19,99 AZN" is said
# "on doqquz manat doxsan doqquz qəpik".
CURRENCY_SUBUNITS = MappingProxyType(
    {
        "manat": "qəpik", "dollar": "sent", "avro": "sent", "rubl": "qəpik",
        "funt sterlinq": "pens", "Türkiyə lirəsi": "quruş",
    }
)

SYMBOLS = MappingProxyType(
    {
        "&": " və ", "№": " nömrə ", "§": " paraqraf ",
        "×": " vurulsun ", "÷": " bölünsün ", "±": " artı mənfi ",
        "≈": " təxminən ", "≤": " kiçik və ya bərabər ",
        "≥": " böyük və ya bərabər ", "=": " bərabərdir ",
        "°": " dərəcə ",
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

# Nouns after which a lone I, V or X is a numeral: "V əsr", "X sinif".  A lone
# letter is otherwise left alone -- it is far more often just a letter.
ROMAN_NOUNS = (
    "əsr", "sinif", "kurs", "hissə", "fəsil", "bölmə", "cild", "maddə",
    "dərəcə", "qrup", "rüb", "yarımil", "mərtəbə",
)

ORDINAL_SUFFIXES = frozenset(
    {
        "cı", "ci", "cu", "cü",
        "ncı", "nci", "ncu", "ncü",
        "ıncı", "inci", "uncu", "üncü",
    }
)

_ROMAN_VALUES = MappingProxyType({"I": 1, "V": 5, "X": 10, "L": 50})
_LONE_ROMAN = MappingProxyType({"I": 1, "V": 5, "X": 10})
# Deliberately limited to I/V/X/L so ordinary upper-case words (DVD, MIX, CD)
# are never mistaken for numerals.
_ROMAN_RE = re.compile(r"^(?=[IVXL]{2,})(XL|L?X{0,4})(IX|IV|V?I{0,3})$")
_ROMAN_MAX = 50

# Optional Azerbaijani suffix written after a dash: "2024-cü", "ATM-də".
_SUFFIX = r"(?:-([^\W\d_][\wəçğıöşü]*))?"

# A number, and a number or a range of them: "2,5", "10-15".
_NUMBER = r"\d+(?:[.,]\d+)?"
_AMOUNT = rf"{_NUMBER}(?:\s?[-–]\s?{_NUMBER})?"
# A scale word written between an amount and its currency: "5 mln AZN".
_SCALE = r"(?:\s?(?:mln|mlrd|trln|milyon|milyard|trilyon|min)\.?)?"

_VOWELS = "aıoueəiöü"
_BACK_VOWELS = "aıou"
# The four-way vowel (ı/i/u/ü) that follows each vowel.
_FOUR_WAY = MappingProxyType(
    {"a": "ı", "ı": "ı", "o": "u", "u": "u", "e": "i", "ə": "i", "i": "i", "ö": "ü", "ü": "ü"}
)


def _either_case(word: str) -> str:
    """``iyun`` -> ``[iİ]yun``: a capital without re.IGNORECASE, whose handling
    of ``İ`` differs between Python and JavaScript."""
    head = word[0]
    upper = "İ" if head == "i" else head.upper()
    return f"[{head}{upper}]{word[1:]}"


_MONTH_RE = "|".join(_either_case(month) for month in MONTHS)
# "il" right after a date already names the year: "01.09.1939 ildə".
_YEAR_WORD_RE = re.compile(r"\s+il(?:in|də|dən|i|ə)?\b")


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


def _speak_number(raw: str) -> str:
    """An integer, or a decimal written with ``,`` or ``.``."""
    if "," in raw or "." in raw:
        whole, fraction = re.split(r"[.,]", raw, maxsplit=1)
        return _decimal_to_words(whole, fraction)
    return number_to_words(int(raw))


def _speak_digits(raw: str) -> str:
    """``007`` -> ``sıfır sıfır yeddi``: leading zeros are read, not dropped."""
    rest = raw.lstrip("0")
    zeros = ["sıfır"] * (len(raw) - len(rest))
    return " ".join(zeros + ([number_to_words(int(rest))] if rest else []))


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


def _phone_groups(digits: str) -> list[str]:
    """Split an unbroken run the way numbers are dictated: ``... 123 45 67``."""
    if len(digits) <= 3:
        return [digits] if digits else []
    if len(digits) == 4:
        return [digits[:2], digits[2:]]
    if len(digits) >= 7:
        return _phone_groups(digits[:-7]) + [digits[-7:-4], digits[-4:-2], digits[-2:]]
    return _phone_groups(digits[:-2]) + [digits[-2:]]


def _replace_phone_numbers(text: str) -> str:
    """Read a phone number group by group, zeros included:
    ``050 123 45 67`` -> ``sıfır əlli yüz iyirmi üç qırx beş altmış yeddi``."""

    def speak(match: re.Match[str]) -> str:
        written = re.findall(r"\d+", match.group(0))
        groups = [group for run in written for group in _phone_groups(run)]
        return " ".join(_speak_digits(group) for group in groups)

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


def _replace_dates(text: str) -> str:
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

    text = re.sub(rf"(?<![\d.,])(\d{{4}})-(\d{{2}})-(\d{{2}})(?!\d){_SUFFIX}\b", iso, text)
    text = re.sub(
        rf"(?<![\d.,/])(\d{{1,2}})([./-])(\d{{1,2}})\2(\d{{4}}|\d{{2}})(?!\d|[./]\d){_SUFFIX}\b",
        day_month_year,
        text,
    )
    # "15/08": a zero-padded second part is a month, never a denominator.
    text = re.sub(rf"(?<![\d.,/])(\d{{1,2}})/(0[1-9])(?![\d/]){_SUFFIX}", day_month, text)
    # "01 may": the zero is padding, not a digit to read.
    return re.sub(rf"(?<![\d.,])0(\d)(?=\s+(?:{_MONTH_RE}))", r"\1", text)


def _replace_dotted_chains(text: str) -> str:
    """Versions and addresses, or a date that failed validation:
    ``1.2.3`` -> ``bir nöqtə iki nöqtə üç``."""
    return re.sub(
        r"(?<![\d.,])\d+(?:\.\d+){2,}(?!\d)",
        lambda m: " nöqtə ".join(number_to_words(int(p)) for p in m.group(0).split(".")),
        text,
    )


def _replace_times(text: str) -> str:
    def speak(match: re.Match[str]) -> str:
        hour, minute = int(match.group(1)), int(match.group(2))
        suffix = match.group(3)
        if minute > 59 or hour > 24 or (hour == 24 and minute):
            return match.group(0)
        if minute == 0:
            spoken = "sıfır sıfır" if hour == 0 else number_to_words(hour)
            return _attach_suffix(spoken, suffix)
        tail = _attach_suffix(number_to_words(minute), suffix)
        if minute < 10:
            return f"{number_to_words(hour)} sıfır {tail}"
        return f"{number_to_words(hour)} {tail}"

    return re.sub(rf"\b(\d{{1,2}}):(\d{{2}}){_SUFFIX}\b", speak, text)


def _replace_operators(text: str) -> str:
    """A score or ratio (``3:2``) is two numbers; ``2x3`` is a product."""
    text = re.sub(r"(?<=\d):(?=\d)", " ", text)
    return re.sub(r"(?<=\d)\s?[xх×]\s?(?=\d)", " × ", text)


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


def _replace_currency(text: str) -> str:
    """Put the currency after the amount as a word."""
    symbols = re.escape("".join(k for k in CURRENCIES if not k.isalpha()))
    codes = "|".join(k for k in CURRENCIES if k.isalpha())

    text = re.sub(
        rf"([{symbols}])\s?({_AMOUNT}{_SCALE})",
        lambda m: _money(m.group(2), CURRENCIES[m.group(1)], None),
        text,
    )
    text = re.sub(
        rf"(?<![\w.,])({_AMOUNT}{_SCALE})\s?({codes}|[{symbols}]){_SUFFIX}(?!\w)",
        lambda m: _money(m.group(1), CURRENCIES[az_lower(m.group(2))], m.group(3)),
        text,
        flags=re.IGNORECASE,
    )
    # A code on its own: "AZN ilə ödəniş".
    return re.sub(
        rf"\b(AZN|USD|EUR|GBP|RUB){_SUFFIX}\b",
        lambda m: harmonise(CURRENCIES[az_lower(m.group(1))], m.group(2)),
        text,
    )


def _replace_percent(text: str) -> str:
    text = re.sub(rf"%\s?({_AMOUNT})", r"\1 faiz", text)
    return re.sub(
        rf"({_AMOUNT})\s?%{_SUFFIX}",
        lambda m: f"{m.group(1)} {harmonise('faiz', m.group(2))}",
        text,
    )


def _replace_units(text: str) -> str:
    keys = sorted(UNITS, key=len, reverse=True)
    pattern = "|".join(re.escape(key) for key in keys)

    def speak(match: re.Match[str]) -> str:
        spoken = UNITS[az_lower(match.group(2))]
        return f"{match.group(1)} {harmonise(spoken, match.group(3))}"

    return re.sub(rf"(\d)\s?({pattern}){_SUFFIX}(?!\w)", speak, text, flags=re.IGNORECASE)


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


def _replace_fractions(text: str) -> str:
    """``3/4`` -> ``dörddə üç``; a numerator not smaller than the denominator
    is a division: ``24/7`` -> ``iyirmi dörd bölü yeddi``."""

    def speak(match: re.Match[str]) -> str:
        top, bottom = int(match.group(1)), int(match.group(2))
        suffix = match.group(3)
        if 0 < top < bottom:
            head = harmonise(number_to_words(bottom), "da")
            return f"{head} {_attach_suffix(number_to_words(top), suffix)}"
        return f"{number_to_words(top)} bölü {_attach_suffix(number_to_words(bottom), suffix)}"

    return re.sub(rf"(?<![\d/.,])(\d+)/(\d+)(?![\d/]|[.,]\d){_SUFFIX}", speak, text)


def _replace_ranges(text: str) -> str:
    """``10-15`` -> ``on ilə on beş``.  A suffix belongs to the second number:
    ``1941-1945-ci illər`` -> ``... qırx bir ilə ... qırx beşinci illər``."""

    def speak(match: re.Match[str]) -> str:
        first, second, suffix = match.group(1, 2, 3)
        if suffix and az_lower(suffix) in ORDINAL_SUFFIXES and second.isdigit():
            tail = ordinal_to_words(int(second))
        else:
            tail = _attach_suffix(_speak_number(second), suffix)
        return f"{_speak_number(first)} ilə {tail}"

    return re.sub(
        rf"(?<![\w.,])({_NUMBER})\s?[-–]\s?({_NUMBER}){_SUFFIX}\b", speak, text
    )


def _replace_suffixed_numbers(text: str) -> str:
    """``5-ci`` -> ``beşinci``; ``10-da`` -> ``onda`` (the suffix already harmonises)."""

    def speak(match: re.Match[str]) -> str:
        value, suffix = match.group(1), match.group(2)
        if az_lower(suffix) in ORDINAL_SUFFIXES and value.isdigit():
            return ordinal_to_words(int(value))
        return f"{_speak_number(value)}{suffix}"

    return re.sub(rf"(?<![\w.,])({_NUMBER})-([^\W\d_][\wəçğıöşü]*)", speak, text)


def _replace_bare_numbers(text: str) -> str:
    # "Su-27", "COVID-19": that dash joins a name to a number; it is no minus.
    text = re.sub(r"(?<=[^\W\d_])-(?=\d)", " ", text)
    text = re.sub(r"(?<![\w+])\+(?=\d)", "üstəgəl ", text)

    def speak(match: re.Match[str]) -> str:
        sign, raw = match.group(1), match.group(2)
        if raw.isdigit() and len(raw) > 1 and raw.startswith("0"):
            spoken = _speak_digits(raw)
        else:
            spoken = _speak_number(raw)
        return f"mənfi {spoken}" if sign else spoken

    return re.sub(rf"(?:(?<!\w)(-))?({_NUMBER})", speak, text)


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
        _replace_dotted_chains,
        _replace_times,
        _replace_operators,
        _replace_currency,
        _replace_percent,
        _replace_units,
        _replace_roman,
        _replace_fractions,
        _replace_ranges,
        _replace_suffixed_numbers,
        _replace_bare_numbers,
        _replace_acronyms,
        _replace_symbols,
        _tidy,
    )
    for step in steps:
        text = step(text)
    return text
