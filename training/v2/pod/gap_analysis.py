"""Three follow-up checks for the next training's plan.

1. Where in the sentence Whisper loses words (first / middle / last), model against recordings.
2. Silence before and after the speech in training clips and in model output.
3. What the training set is made of: sentence types, lengths, hours, word-final consonants.

    /workspace/venv_asr/bin/python gap_analysis.py [--out /workspace/asr_eval] [--prepared /workspace/prepared]
Writes <out>/gap_analysis.json.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from asr_analysis import align, length_bin, sentence_type
from eval_asr import word_list

MODEL_VERSION = "step-00030000"
FRAME = 0.01
SILENCE_DB = 40.0


def position_errors(rows: list[dict], version: str) -> dict:
    seen, wrong = Counter(), Counter()
    for row in rows:
        ref = word_list(row["text"])
        position = 0
        for op, r, _ in align(ref, word_list(row[version])):
            if r is None:
                continue
            slot = "first" if position == 0 else "last" if position == len(ref) - 1 else "middle"
            seen[slot] += 1
            wrong[slot] += op != "ok"
            position += 1
    return {slot: round(wrong[slot] / seen[slot], 4) for slot in ("first", "middle", "last")}


def edge_silence(path: str) -> tuple[float, float]:
    wave, rate = sf.read(path, dtype="float32")
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    hop = int(rate * FRAME)
    count = len(wave) // hop
    db = 20 * np.log10(np.sqrt(np.mean(wave[: count * hop].reshape(count, hop) ** 2, axis=1)) + 1e-9)
    loud = np.flatnonzero(db > db.max() - SILENCE_DB)
    return float(loud[0] * FRAME), float((count - 1 - loud[-1]) * FRAME)


def silence_stats(paths: list[str]) -> dict:
    pairs = [edge_silence(p) for p in paths]
    lead, tail = [p[0] for p in pairs], [p[1] for p in pairs]
    return {"lead_median_s": round(statistics.median(lead), 3), "tail_median_s": round(statistics.median(tail), 3),
            "lead_under_50ms": round(sum(x < 0.05 for x in lead) / len(lead), 3),
            "tail_under_50ms": round(sum(x < 0.05 for x in tail) / len(tail), 3)}


def training_set(prepared: Path) -> dict:
    rows = [json.loads(line) for line in (prepared / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    hours = sum(r["duration_seconds"] for r in rows) / 3600
    types, lengths, seconds_by_length = Counter(), Counter(), Counter()
    endings = Counter()
    for r in rows:
        text = r["normalized_text"]
        types[sentence_type(text)] += 1
        lengths[length_bin(text)] += 1
        seconds_by_length[length_bin(text)] += r["duration_seconds"]
        for word in word_list(text):
            endings[word[-1]] += 1
    total_words = sum(endings.values())
    return {"clips": len(rows), "hours": round(hours, 2),
            "types": {k: round(v / len(rows), 4) for k, v in types.items()},
            "length_clips": {k: round(v / len(rows), 4) for k, v in sorted(lengths.items())},
            "length_hours": {k: round(v / 3600, 2) for k, v in sorted(seconds_by_length.items())},
            "duration_median_s": round(statistics.median(r["duration_seconds"] for r in rows), 2),
            "duration_p90_s": round(float(np.percentile([r["duration_seconds"] for r in rows], 90)), 2),
            "word_final_letters": {k: round(v / total_words, 4) for k, v in endings.most_common(12)}}, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/workspace/asr_eval"))
    parser.add_argument("--prepared", type=Path, default=Path("/workspace/prepared"))
    args = parser.parse_args()
    asr_rows = json.loads((args.out / "asr_rows.json").read_text(encoding="utf-8"))
    clean = [r for r in asr_rows if not any(ch.isdigit() for v in r if v != "text" for ch in r[v])]
    composition, train_rows = training_set(args.prepared)
    sample = random.Random(0).sample(train_rows, 1500)
    report = {
        "position_wer": {v: position_errors(clean, v) for v in ("dataset", "base", MODEL_VERSION)},
        "silence": {
            "training_clips": silence_stats([str(args.prepared / r["audio"]) for r in sample]),
            "model_validation": silence_stats([str(args.out / MODEL_VERSION / f"{i:04d}.wav") for i in range(len(asr_rows))]),
        },
        "training_set": composition,
    }
    (args.out / "gap_analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
