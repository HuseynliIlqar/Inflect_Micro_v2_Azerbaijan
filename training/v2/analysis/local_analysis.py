"""Extra breakdowns from the downloaded measurements (no audio work):
question kinds (yes/no with -mı/-mi/-mu/-mü vs question-word questions), how steady the voice
is from clip to clip (pitch and loudness spread), and the sentence-length effect on pitch range.

    python local_analysis.py D:/ktts/avaz-train/results/asr_eval
"""
from __future__ import annotations

import json
import math
import re
import statistics
import sys
from pathlib import Path

QUESTION_WORDS = re.compile(r"\b(nə|kim|harada|haraya|haradan|niyə|necə|nə vaxt|hansı|neçə|kimin|nəyə|nədən|kimə)\b", re.I)
YES_NO = re.compile(r"(mı|mi|mu|mü)(\b|[?,])", re.I)


def question_kind(text: str) -> str | None:
    if not text.rstrip().endswith("?"):
        return None
    if QUESTION_WORDS.search(text):
        return "question_word"
    return "yes_no" if YES_NO.search(text) else "other"


def finite(values):
    return [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]


def main() -> None:
    folder = Path(sys.argv[1])
    rows = json.loads((folder / "rows.json").read_text(encoding="utf-8"))
    voice = json.loads((folder / "voice_rows.json").read_text(encoding="utf-8"))
    versions = sorted({r["version"] for r in voice})
    by = {(r["version"], r["index"]): r for r in voice}
    report = {"question_kinds": {}, "steadiness": {}}
    kinds = {i: question_kind(row["text"]) for i, row in enumerate(rows)}
    for version in versions:
        out = {}
        for kind in ("yes_no", "question_word", "other"):
            idx = [i for i, k in kinds.items() if k == kind]
            values = finite(by[(version, i)].get("final_st") for i in idx)
            out[kind] = {"clips": len(idx), "final_st_median": round(statistics.median(values), 2) if values else None}
        report["question_kinds"][version] = out
        f0 = finite(r.get("f0_median") for r in voice if r["version"] == version)
        semitones = [12 * math.log2(v / 100) for v in f0]
        rates = finite(r.get("rate") for r in voice if r["version"] == version)
        report["steadiness"][version] = {
            "pitch_median_spread_st": round(statistics.pstdev(semitones), 3),
            "rate_spread_rel": round(statistics.pstdev(rates) / statistics.mean(rates), 3),
        }
    print(json.dumps(report, indent=1, ensure_ascii=False))
    (folder.parent / "local_analysis.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
