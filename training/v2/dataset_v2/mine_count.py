"""How many usable dataset-v2 sentences the local text sources already hold, per category,
after removing everything already in the current dataset.

    python mine_count.py
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

SOURCES = Path("D:/ktts/teacher/corpus/modern_raw")
EXISTING = Path("D:/ktts/teacher/dataset/metadata.jsonl")
AZ_LETTERS = set("abcçdeəfgğhxıijkqlmnoöprsştuüvyzABCÇDEƏFGĞHXIİJKQLMNOÖPRSŞTUÜVYZ")
QUESTION_WORDS = re.compile(r"\b(nə|kim|harada|haraya|haradan|niyə|necə|nə vaxt|hansı|neçə|kimin|nəyə|nədən|kimə|nəyi|kimi)\b", re.I)
YES_NO = re.compile(r"\w(mı|mi|mu|mü)\b|\b(mı|mi|mu|mü)\b", re.I)
FIRST_PERSON = re.compile(r"\w{2,}(ım|im|um|üm|am|əm)\b")


def norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def clean(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return (letters and sum(c in AZ_LETTERS for c in letters) / len(letters) > 0.98
            and not re.search(r"\d", text) and 3 <= len(text.split()) <= 30
            and text[-1] in ".!?" and "  " not in text)


def category(text: str) -> str:
    words = len(text.split())
    if text.endswith("?"):
        if "yoxsa" in text.lower():
            return "alternative_question"
        if QUESTION_WORDS.search(text):
            return "wh_question"
        return "yes_no_question" if YES_NO.search(text) else "other_question"
    if text.endswith("!"):
        return "exclamation"
    if FIRST_PERSON.search(text):
        return "first_person_m"
    return "statement_6_10" if 6 <= words <= 10 else "statement_other"


def main() -> None:
    existing = {norm(json.loads(line)["text"]) for line in EXISTING.read_text(encoding="utf-8").splitlines() if line.strip()}
    counts, seen = Counter(), set()
    for path in sorted(SOURCES.glob("*.jsonl")):
        per = Counter()
        for line in path.read_text(encoding="utf-8").splitlines():
            text = json.loads(line).get("text", "").strip()
            key = norm(text)
            if not text or key in existing or key in seen or not clean(text):
                continue
            seen.add(key)
            per[category(text)] += 1
        print(path.name, dict(per))
        counts.update(per)
    print("TOTAL", dict(counts))


if __name__ == "__main__":
    main()
