"""Find obscenities in Azerbaijani text and mark them for a bleep.

`censor_az` replaces every obscene word with `BLEEP`; the engine speaks the
text around it and plays a tone where it stands, the way broadcast television
does. It runs before `normalize_az`, on the text as the reader wrote it.

    >>> censor_az("Sən qəhbəsən, bildin?")
    'Sən █, bildin?'

The list targets obscenities, not every unkind word: `eşşək` is an animal and
`axmaq` is in children's books, so a news article or a fairy tale reads as
written. A false positive breaks an ordinary sentence, which is why every stem
that collides with a clean word (`sik` -> `sikkə`, `göt` -> `götürmək`) either
lists those words as exceptions or matches only its own inflected forms.

Stem matching folds away what people type instead of Azerbaijani letters
(`qehbe`, `amciq`), digits and symbols used as letters (`s1kdir`, `$ikdir`) and
letters held down for emphasis (`siiiikdir`). `ş` is deliberately not folded to
`s`: `şikayət` would otherwise read as `sik...`. Whole forms are matched with
their own letters, because folded they are names and English words: `amına`
would be `Amina`, `piç` would be `PIC`, `göt` would be `got`. A price such as
`$1k` is an amount, not a word written in digits.

The text is NFC-composed and stripped of invisible format characters first, so
a decomposed `ö` or a zero-width space inside a word does not hide it. This is
a filter for ordinary text, not a guarantee: `s i k` spelled out still passes.
"""

from __future__ import annotations

import re
import unicodedata

from .az_words import az_lower

__all__ = ["BLEEP", "bleep_segments", "censor_az", "is_profane"]

# One character with no letters in it, so no step of `normalize_az` touches it
# and `chunk_text` counts it as one word.
BLEEP = "█"

# Language data: anything starting with one of these is obscene...
PROFANE_STEMS: tuple[str, ...] = (
    "sik", "qəhbə", "gəhbə", "amcı", "götverən", "götbaş", "yarraq", "yarrağ",
    "qancıq", "qancığ", "orospu", "oruspu", "peysər", "pezəvəng", "pezevenk",
    "dəyyus", "dəyus", "yavşaq", "yavsaq", "yavşağ", "gicdıllaq", "gijdıllaq",
    "sürtük", "itoğlu", "köpəkoğlu", "köpəyoğlu",
    "blyat", "blyad", "bılyat", "dalbayob", "dolbayob", "dalbayeb", "dolbaeb",
    "pidar", "pidor", "pizd", "yeba", "mudak", "mudaq", "mudil",
    "бля", "пизд", "хуй", "хуе", "хуё", "ебат", "ебан", "ёбан", "мудак",
    "пидор", "пидар",
)

# ...unless it starts with one of these clean words, which share the letters.
CLEAN_PREFIXES: tuple[str, ...] = (
    "sikk", "sikl", "sikh", "sikay", "sikest", "sikar", "sikan", "sikor",
)

# Language data: forms matched whole, for stems too short or too common to
# match by prefix (`göt` starts `Göteborq`, `am` starts `amma`). They are not
# folded, so the spellings without Azerbaijani letters are listed where they
# are unambiguous -- never a bare `got`.
PROFANE_WORDS: frozenset[str] = frozenset({
    "göt", "götü", "götün", "götünü", "götünə", "götündə", "götündən",
    "götə", "götdə", "götdən", "götlər", "götləri", "götlük",
    "gotu", "gotun", "gotunu", "gotune", "gotunde", "gotunden",
    "gotde", "gotden", "gotler", "gotleri", "gotluk",
    "amına", "amını", "amk",
    "piç", "piçi", "piçin", "piçlər",
    "suka", "sukalar", "naxuy", "nahuy", "xuy",
    "сука", "суки",
})

# Digits and symbols standing in for letters, and letters typed as their
# nearest Latin neighbour. Applied after `az_lower`, for stems only.
_FOLD = str.maketrans({
    "ə": "e", "ı": "i", "ö": "o", "ü": "u", "ç": "c", "ğ": "g",
    "0": "o", "1": "i", "3": "e", "4": "a", "@": "a", "$": "s",
})

# A word, with any suffix written after an apostrophe or hyphen.
_TOKEN_RE = re.compile(r"[\w@$]+(?:['’\-][\w@$]+)*")
_PART_RE = re.compile(r"['’\-]")
_HELD_RE = re.compile(r"(.)\1{2,}")
# An amount, not a word: `$1k`, `10k`, `2x`.
_AMOUNT_RE = re.compile(r"[$@]?\d+(?:[.,]\d+)*[^\W\d_]{0,2}")
_RUN_RE = re.compile(rf"{BLEEP}(?:[\s,;:\-]+{BLEEP})+")
# What is left of a piece once a bleep is cut out of it: marks at its head
# closed the word the bleep replaced, and a piece of marks alone says nothing.
_LEADING_MARKS_RE = re.compile(r"^[\s,.;:!?\-]+")
_SPEAKABLE_RE = re.compile(r"\w")


def _visible(text: str) -> str:
    """NFC-composed, without soft hyphens, zero-width spaces and the like."""
    composed = unicodedata.normalize("NFC", text)
    return "".join(c for c in composed if unicodedata.category(c) != "Cf")


def _plain(word: str) -> str:
    """Lower-cased with held letters let go: the form whole words match in."""
    return _HELD_RE.sub(r"\1", az_lower(word))


def _skeleton(word: str) -> str:
    """The plain form, folded as well: the form stems match in."""
    return _plain(az_lower(word).translate(_FOLD))


_STEMS = tuple(_skeleton(stem) for stem in PROFANE_STEMS)
_CLEAN = tuple(_skeleton(prefix) for prefix in CLEAN_PREFIXES)
_WORDS = frozenset(_plain(word) for word in PROFANE_WORDS)


def is_profane(word: str) -> bool:
    """Whether one word, or any part of a hyphenated word, is obscene."""
    for part in _PART_RE.split(_visible(word)):
        if not part or _AMOUNT_RE.fullmatch(part):
            continue
        if _plain(part) in _WORDS:
            return True
        skeleton = _skeleton(part)
        if skeleton.startswith(_STEMS) and not skeleton.startswith(_CLEAN):
            return True
    return False


def censor_az(text: str) -> str:
    """Return the text with each obscene word replaced by `BLEEP`.

    Obscenities next to each other -- `siktir, qəhbə` -- become one bleep, so
    the listener hears a single tone rather than a stutter. Invisible format
    characters are dropped from the whole text; they have no sound anyway.
    """
    marked = _TOKEN_RE.sub(
        lambda match: BLEEP if is_profane(match.group(0)) else match.group(0),
        _visible(text),
    )
    return _RUN_RE.sub(BLEEP, marked)


def bleep_segments(chunk: str) -> tuple[str, ...]:
    """Cut a chunk into the pieces to speak and the `BLEEP`s between them.

    >>> bleep_segments("Sən █, bildin?")
    ('Sən', '█', 'bildin?')
    """
    segments: list[str] = []
    for index, piece in enumerate(chunk.split(BLEEP)):
        if index:
            segments.append(BLEEP)
        spoken = _LEADING_MARKS_RE.sub("", piece).strip()
        if _SPEAKABLE_RE.search(spoken):
            segments.append(spoken)
    return tuple(segments)
