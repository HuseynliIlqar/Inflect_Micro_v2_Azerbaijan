"""Split Azerbaijani text into chunks short enough for a 9.36M model.

The exported package already breaks text at sentence ends and only cuts a
sentence further once it passes 280 characters -- roughly forty words.  This
model loses intonation long before that: the tail of a long clause flattens and
word endings get clipped.  Splitting nearer fifteen words keeps every chunk
inside the range the model handles well.

Punctuation is preserved on each chunk because the package derives its inter
chunk pause from the trailing mark.

    >>> chunk_text("Salam. Necəsən?")
    ('Salam.', 'Necəsən?')
"""

from __future__ import annotations

import re

__all__ = ["chunk_text", "DEFAULT_MAX_WORDS"]

DEFAULT_MAX_WORDS = 15

# Sentence-final marks, kept with the sentence they close.
_SENTENCE_RE = re.compile(r"(?<=[.!?;:])\s+")

# Conjunctions that carry a natural breath before them; used as a second-choice
# split point when a clause has no comma to cut at.
_CONJUNCTIONS = (
    "və", "amma", "ancaq", "lakin", "çünki", "ki", "ya", "yaxud",
    "ona görə", "buna görə", "həmçinin", "yəni",
)


def _words(text: str) -> tuple[str, ...]:
    return tuple(text.split())


def _split_at_commas(sentence: str) -> tuple[str, ...]:
    """Cut after commas, keeping the comma so the pause survives."""
    parts = re.split(r"(?<=,)\s+", sentence)
    return tuple(part for part in (p.strip() for p in parts) if part)


def _split_at_conjunctions(clause: str, max_words: int) -> tuple[str, ...]:
    """Break before a conjunction near the middle, when one is available."""
    words = _words(clause)
    if len(words) <= max_words:
        return (clause,)
    lowered = tuple(word.strip(",.;:!?").lower() for word in words)
    candidates = [
        index
        for index, word in enumerate(lowered)
        if word in _CONJUNCTIONS and 2 <= index <= len(words) - 3
    ]
    if not candidates:
        return _split_hard(clause, max_words)
    middle = len(words) // 2
    cut = min(candidates, key=lambda index: abs(index - middle))
    head = " ".join(words[:cut])
    tail = " ".join(words[cut:])
    return _split_at_conjunctions(head, max_words) + _split_at_conjunctions(tail, max_words)


def _split_hard(clause: str, max_words: int) -> tuple[str, ...]:
    """Last resort: cut on a fixed word count."""
    words = _words(clause)
    return tuple(
        " ".join(words[start : start + max_words])
        for start in range(0, len(words), max_words)
    )


def chunk_text(text: str, max_words: int = DEFAULT_MAX_WORDS) -> tuple[str, ...]:
    """Return the text as chunks of at most ``max_words`` words each.

    Split points are tried in order of how natural the resulting pause sounds:
    sentence end, then comma, then conjunction, then a plain word count.
    """
    if max_words < 1:
        raise ValueError("max_words must be at least 1")
    normalized = " ".join(text.split())
    if not normalized:
        return ()
    chunks: list[str] = []
    for sentence in _SENTENCE_RE.split(normalized):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(_words(sentence)) <= max_words:
            chunks.append(sentence)
            continue
        for clause in _split_at_commas(sentence):
            chunks.extend(_split_at_conjunctions(clause, max_words))
    return tuple(chunk for chunk in chunks if chunk)
