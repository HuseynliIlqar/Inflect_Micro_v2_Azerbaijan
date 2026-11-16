"""Check dataset-v2 sentence files before they are merged.

    python check_sentences.py out/A1_*.jsonl [more files...]

Each line must be {"cat": "<code>", "text": "<sentence or paragraph>"}. Reports, per file:
lines, bad lines with the reason, duplicates (inside the files, against the current dataset and
the 471 test sentences), average words, and the share of words the model never saw in training
(new-word share). Prints at most 25 problem lines. Exit code 0 always; read the report.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from glob import glob
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET = Path("D:/ktts/teacher/dataset/metadata.jsonl")
TESTSET = Path("D:/ktts/avaz-train/results/asr_eval/rows.json")
VOCAB = HERE / "train_vocab.txt"

LETTERS = "abcçdeəfgğhxıijkqlmnoöprsştuüvyzABCÇDEƏFGĞHXIİJKQLMNOÖPRSŞTUÜVYZ"
ALLOWED = set(LETTERS + " ,.!?:;—-«»\"…")
_LOWER = str.maketrans({"I": "ı", "İ": "i"})
WORD_LIMITS = {"A3": (8, 25)}
DEFAULT_LIMITS = (2, 25)


def words_of(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", text.translate(_LOWER).lower())


def key(text: str) -> str:
    return " ".join(words_of(text))


def limits(cat: str) -> tuple[int, int]:
    if cat in WORD_LIMITS:
        return WORD_LIMITS[cat]
    if cat.startswith("G"):
        return 25, 60
    if cat.startswith("L"):
        return 16, 35
    return DEFAULT_LIMITS


def problems(cat: str, text: str) -> list[str]:
    found = []
    if not text or text[-1] not in ".!?…":
        found.append("must end with . ! ? or …")
    bad = sorted(set(text) - ALLOWED)
    if bad:
        found.append(f"characters not allowed: {''.join(bad)}")
    if re.search(r"\b[A-ZƏÖÜŞÇĞİ]{2,}\b", text):
        found.append("all-caps acronym")
    if "  " in text or text != text.strip():
        found.append("extra spaces")
    lo, hi = limits(cat)
    n = len(text.split())
    if not lo <= n <= hi:
        found.append(f"{n} words, expected {lo}-{hi}")
    if text[:1].islower():
        found.append("starts with a lowercase letter")
    return found


def main() -> None:
    files = [Path(p) for arg in sys.argv[1:] for p in sorted(glob(arg))]
    known = {key(json.loads(l)["text"]) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()}
    known |= {key(r["text"]) for r in json.loads(TESTSET.read_text(encoding="utf-8"))}
    vocab = {line.split("\t")[0] for line in VOCAB.read_text(encoding="utf-8").splitlines()}
    seen: dict[str, str] = {}
    shown = 0
    total = Counter()
    for path in files:
        stats = Counter()
        word_total = new_words = 0
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            stats["lines"] += 1
            try:
                row = json.loads(line)
                cat, text = row["cat"], row["text"]
            except (ValueError, KeyError) as error:
                issues = [f"bad JSON line: {error}"]
                cat = text = ""
            else:
                issues = problems(cat, text)
                k = key(text)
                if k in known:
                    issues.append("already in the current dataset or test set")
                elif k in seen:
                    issues.append(f"duplicate of {seen[k]}")
                else:
                    seen[k] = f"{path.name}:{number}"
                ws = words_of(text)
                word_total += len(ws)
                new_words += sum(w not in vocab for w in ws)
            if issues:
                stats["bad"] += 1
                if shown < 25:
                    print(f"  {path.name}:{number}: {'; '.join(issues)} | {text[:90]}")
                    shown += 1
        share = new_words / max(word_total, 1)
        print(f"{path.name}: {stats['lines']} lines, {stats['bad']} bad, "
              f"{word_total / max(stats['lines'], 1):.1f} words/line, new-word share {share:.1%}")
        total.update(stats)
    print(f"TOTAL: {total['lines']} lines, {total['bad']} bad, {len(seen)} unique")


if __name__ == "__main__":
    main()
