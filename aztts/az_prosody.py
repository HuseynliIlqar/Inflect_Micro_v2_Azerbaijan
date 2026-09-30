"""Re-place the stress marks eSpeak puts on Azerbaijani phonemes.

eSpeak stresses the final syllable of nearly every word, so a sentence reaches
the model as a flat row of equal accents.  Measured on ``prepared/az``:

    155,175 words -> 92.2% carry a primary stress, only 7.8% carry none.

Natural speech reduces modifiers, participles and clitics so that the remaining
accents stand out.  This module rebuilds that contrast by editing the phoneme
string before it reaches the model.

Every edit stays inside what the checkpoint saw during training:

* dropping ``PRIMARY`` is how the 7.8% unstressed words already look
  (``və``, ``bu``, ``da``, ``nə``, ``o`` ...);
* demoting ``PRIMARY`` to ``SECONDARY`` is attested on 409 distinct words
  (``ˌonbˈeʃ``, ``onˌun``, ``ˌilæ`` ...).

The length mark ``ː`` is deliberately never touched: in this dataset it is
phonemic, not prosodic (``saːɫˈam`` = *salam*, ``jaːɯʃ`` = *yağış*), so using it
for emphasis would change the word.

    >>> restress("bˈu, dʒˈox vadʒibdˈir.", "Bu, çox vacibdir.")
    'bu, dʒˈox vadʒibdˈir.'
"""

from __future__ import annotations

import re

from .az_words import az_lower

__all__ = [
    "restress",
    "PRIMARY",
    "SECONDARY",
    "CLITICS",
    "WEAK_HEADS",
]

PRIMARY = "\u02c8"
SECONDARY = "\u02cc"

# Words the training data already shows without a primary stress.  Dropping the
# accent from these is fully in distribution.
CLITICS = frozenset(
    {
        "və", "bu", "o", "da", "də", "nə", "ya", "həm",
        "biz", "siz", "mən", "sən", "ki", "hər",
    }
)

# Words the training data shows carrying a secondary stress instead of a
# primary one.
WEAK_HEADS = frozenset(
    {
        "ilə", "isə", "tam",
        "onun", "onların", "onlar", "mənim", "sənin", "sizin", "bizim",
    }
)

# Postpositions and conjunctions that behave like clitics in speech but that
# eSpeak stresses anyway.  Only used at level "wide".
_WIDE_WEAK = frozenset(
    {
        "üçün", "kimi", "qədər", "görə", "üzrə", "barədə", "haqqında",
        "sonra", "əvvəl", "arasında", "tərəfindən", "boyu", "başqa",
        "amma", "ancaq", "lakin", "çünki", "yaxud", "əgər", "yəni",
        "daha", "artıq", "belə", "elə", "isə",
    }
)

# Participle endings.  In an Azerbaijani relative clause the participle is the
# head and carries no accent of its own -- the accent sits on what precedes it.
# The ``(?<![dt])`` guard keeps the ablative case out: ``nəsillərdən`` is a noun,
# not a participle.  The handful of real participles that end in ``-dən`` are
# listed separately.
_PARTICIPLE_RE = re.compile(
    r"(?:[yn][aə]n|(?<![dt])[aə]n|m[ıiuü]ş|dığı|diyi|duğu|düyü)$"
)
_PARTICIPLE_WORDS = frozenset({"edən", "gedən", "döndən", "ödən"})

# Attributive adjective endings.  Adjective + noun puts the accent on the noun.
# ``-lı/-li`` is required to be at least six letters long because the possessive
# ("dili", "yolu") wears the same ending on much shorter stems.
_ADJECTIVE_RE = re.compile(r"(?:s[ıiuü]z|[ıiuü]lı|ki)$")
_ADJECTIVE_LONG_RE = re.compile(r"l[ıiuü]$")
_ADJECTIVE_MIN_LETTERS = 6

# Trailing punctuation that ends a phrase; a word in this position keeps its
# accent because it carries the phrase's nuclear stress.
_PHRASE_END_RE = re.compile(r"[,.;:!?…—]$")

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _bare(word: str) -> str:
    """The lowercase letters of ``word``, without punctuation."""
    match = _WORD_RE.search(word)
    return az_lower(match.group(0)) if match else ""


def _demote(token: str, *, to_secondary: bool) -> str:
    """Replace the primary stress in ``token`` with a weaker mark, or none."""
    if PRIMARY not in token:
        return token
    return token.replace(PRIMARY, SECONDARY if to_secondary else "", 1)


def _decide(
    index: int,
    words: list[str],
    *,
    level: str,
) -> str:
    """Return "keep", "drop" or "demote" for the word at ``index``."""
    bare = _bare(words[index])
    if not bare:
        return "keep"
    if bare in CLITICS:
        return "drop"
    if bare in WEAK_HEADS:
        return "demote"
    if level != "wide":
        return "keep"
    if bare in _WIDE_WEAK:
        return "drop"
    # Morphology only applies mid-phrase: the last word before a comma or a full
    # stop carries the nuclear accent and must keep it.
    at_phrase_end = index == len(words) - 1 or _PHRASE_END_RE.search(words[index])
    if at_phrase_end:
        return "keep"
    if index > 0 and (bare in _PARTICIPLE_WORDS or _PARTICIPLE_RE.search(bare)):
        return "demote"
    if _ADJECTIVE_RE.search(bare):
        return "demote"
    if len(bare) >= _ADJECTIVE_MIN_LETTERS and _ADJECTIVE_LONG_RE.search(bare):
        return "demote"
    return "keep"


def restress(
    phonemes: str,
    text: str,
    *,
    level: str = "safe",
    to_secondary: bool = True,
) -> str:
    """Rewrite the stress marks of ``phonemes`` using ``text`` for the words.

    ``phonemes`` must be the eSpeak output for ``text`` with words separated by
    spaces, which is what the deployment frontend produces.  ``level`` is
    ``"safe"`` (clitics only, every edit attested in the training data) or
    ``"wide"`` (also postpositions, participles and attributive adjectives).
    ``to_secondary`` keeps a weak accent instead of removing it outright.

    The phoneme string is returned unchanged when the two sides do not line up
    word for word, so a frontend change can never corrupt the output.
    """
    if level not in {"safe", "wide"}:
        raise ValueError('level must be "safe" or "wide"')
    phoneme_words = phonemes.split()
    words = text.split()
    if not phoneme_words or len(phoneme_words) != len(words):
        return phonemes
    rebuilt = []
    for index, token in enumerate(phoneme_words):
        action = _decide(index, words, level=level)
        if action == "drop":
            rebuilt.append(_demote(token, to_secondary=False))
        elif action == "demote":
            rebuilt.append(_demote(token, to_secondary=to_secondary))
        else:
            rebuilt.append(token)
    return " ".join(rebuilt)
