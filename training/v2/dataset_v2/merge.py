"""Merge the checked dataset-v2 files into one list for recording.

    python merge.py   -> sentences_v2.jsonl, sentences_v2.txt, summary.json

Drops lines the checker would reject and duplicates (first one wins), gives every line an id
like A1_00042, and estimates the recorded length per category from the teacher's measured
speaking rate (5.77 syllables per second) plus half a second of edge silence.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from check_sentences import DATASET, TESTSET, key, problems  # noqa: E402

VOWELS = set("aeəıioöuüAEƏIİOÖUÜ")
SYLLABLES_PER_SECOND = 5.77
EDGE_SECONDS = 0.5


def seconds(text: str) -> float:
    return sum(ch in VOWELS for ch in text) / SYLLABLES_PER_SECOND + EDGE_SECONDS


def main() -> None:
    known = {key(json.loads(l)["text"]) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()}
    known |= {key(r["text"]) for r in json.loads(TESTSET.read_text(encoding="utf-8"))}
    seen, kept, dropped = set(), [], defaultdict(int)
    for path in sorted((HERE / "out").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                dropped["bad json"] += 1
                continue
            cat, text = row.get("cat", ""), row.get("text", "")
            k = key(text)
            if problems(cat, text):
                dropped["checker"] += 1
            elif k in known or k in seen:
                dropped["duplicate"] += 1
            else:
                seen.add(k)
                kept.append({"cat": cat, "text": text})
    counters, summary = defaultdict(int), defaultdict(lambda: {"lines": 0, "hours": 0.0})
    with (HERE / "sentences_v2.jsonl").open("w", encoding="utf-8") as jl, (HERE / "sentences_v2.txt").open("w", encoding="utf-8") as txt:
        for row in sorted(kept, key=lambda r: r["cat"]):
            counters[row["cat"]] += 1
            row = {"id": f"{row['cat']}_{counters[row['cat']]:05d}", **row}
            jl.write(json.dumps(row, ensure_ascii=False) + "\n")
            txt.write(f"{row['id']}\t{row['text']}\n")
            summary[row["cat"]]["lines"] += 1
            summary[row["cat"]]["hours"] += seconds(row["text"]) / 3600
    result = {"categories": {k: {"lines": v["lines"], "hours": round(v["hours"], 2)} for k, v in sorted(summary.items())},
              "total_lines": len(kept), "total_hours": round(sum(v["hours"] for v in summary.values()), 2),
              "dropped": dict(dropped)}
    (HERE / "summary.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
