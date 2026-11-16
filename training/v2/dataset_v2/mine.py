"""Mine dataset-v2 candidate sentences from the local text sources, with strict quality filters,
and sort them into the categories the report asks for.

    python mine.py            -> pool/<category>.jsonl and pool/summary.json

Filters (all must pass): proper Azerbaijani spelling (sentences of 4+ words need at least one
of ə ı ö ü ş ç ğ, and none of the ASCII transliterations such as "ucun"), a single sentence,
no digits, e-mails, links, "Name:" chat prefixes, all-caps acronyms or Cyrillic, 3-25 words,
and not already in the current dataset (or another candidate).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = Path("D:/ktts/teacher/corpus/modern_raw")
EXISTING = Path("D:/ktts/teacher/dataset/metadata.jsonl")
OUT = HERE / "pool"

AZ_SPECIAL = set("əıöüşçğƏIÖÜŞÇĞİ")
ASCII_TELLS = re.compile(r"\b(ucun|uchun|bilersiniz|melumat|chox|cox|ve|bu gun|gore|ile|olaraq|teqdim|edin|sizin|isteyirem|isterdim|xahis)\b", re.I)
CHAT_PREFIX = re.compile(r"^[\wƏəİıÖöÜüŞşÇçĞğ.]+( [\wƏəİıÖöÜüŞşÇçĞğ.]+)?:\s")
ACRONYM = re.compile(r"\b[A-ZƏÖÜŞÇĞİ]{2,}\b")
INNER_BREAK = re.compile(r"[.!?…]\s+\S")
QUESTION_WORDS = re.compile(r"\b(nə|kim|harada|haraya|haradan|niyə|necə|hansı|neçə|kimin|nəyə|nədən|kimə|nəyi|nəçi)\b", re.I)
YES_NO = re.compile(r"(mı|mi|mu|mü)\b", re.I)
FIRST_PERSON = re.compile(r"\w{2,}(dım|dim|dum|düm|tım|tim|tum|tüm|ram|rəm|yam|yəm|am|əm|ım|im|um|üm)\b", re.I)
SECOND_PERSON = re.compile(r"\w{2,}(dın|din|dun|dün|tın|tin|tun|tün|san|sən|sınız|siniz|sunuz|sünüz)\b", re.I)
LABIAL_START = re.compile(r"\b[bpmBPM]\w+")
UVULAR_START = re.compile(r"\b[qQ]\w+")


def norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def passes(text: str) -> bool:
    words = text.split()
    if not 3 <= len(words) <= 25 or text[-1] not in ".!?":
        return False
    if re.search(r"[\d@/\\_#*<>{}\[\]|=+]|http|www|[а-яА-Я]", text):
        return False
    if CHAT_PREFIX.match(text) or ACRONYM.search(text) or INNER_BREAK.search(text[:-1]):
        return False
    if re.search(r"\b[wW]\w*", text):
        return False
    if len(words) >= 4 and not (AZ_SPECIAL & set(text)):
        return False
    return not ASCII_TELLS.search(text) or bool(AZ_SPECIAL & set(text)) and not re.search(r"\b(ucun|uchun|cox|chox)\b", text, re.I)


def categories(text: str) -> list[str]:
    words = len(text.split())
    lower = text.lower()
    found = []
    if text.endswith("?"):
        if "yoxsa" in lower:
            found.append("alternative_question")
        elif QUESTION_WORDS.search(text):
            found.append("wh_question")
        elif YES_NO.search(text):
            found.append("yes_no_question")
        else:
            found.append("intonation_question")
    elif text.endswith("!"):
        found.append("exclamation")
    else:
        if FIRST_PERSON.search(text) and SECOND_PERSON.search(text):
            found.append("first_and_second_person")
        elif FIRST_PERSON.search(text):
            found.append("first_person")
        if len(LABIAL_START.findall(text)) >= 3 or len(UVULAR_START.findall(text)) >= 2:
            found.append("consonant_dense")
        if 6 <= words <= 10:
            found.append("statement_6_10")
    return found or ["statement_other"]


def main() -> None:
    existing = {norm(json.loads(line)["text"]) for line in EXISTING.read_text(encoding="utf-8").splitlines() if line.strip()}
    OUT.mkdir(exist_ok=True)
    pools: dict[str, list[dict]] = {}
    seen: set[str] = set()
    per_source = Counter()
    for path in sorted(SOURCES.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            text = " ".join(json.loads(line).get("text", "").split())
            key = norm(text)
            if not text or key in existing or key in seen or not passes(text):
                continue
            seen.add(key)
            per_source[path.stem] += 1
            for cat in categories(text):
                pools.setdefault(cat, []).append({"text": text, "source": path.stem})
    summary = {"per_source": dict(per_source), "per_category": {k: len(v) for k, v in sorted(pools.items())}}
    for cat, items in pools.items():
        with (OUT / f"{cat}.jsonl").open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
