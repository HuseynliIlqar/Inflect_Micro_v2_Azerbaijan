"""Build dataset v2 from the selection, in v1's layout, ready to train next to it.

    python finalize.py --run /workspace/rec --out /workspace/dataset_v2

Copies each chosen take to <out>/audio/az-01-main/v2_<id>.wav (the "v2_" prefix keeps the
names apart from v1's 000000.wav ...), and writes:
- metadata.jsonl: {"audio", "text", "speaker"} rows, exactly v1's format;
- clip_map.jsonl: file, sentence id, category, take, seconds and the take's scores;
- qc_summary.json: counts, hours, drop reasons and edge silence;
- checksums.sha256: for verifying the download.

Edge silence stays as recorded. The pilot's chosen takes already match v1 on the same
measure (lead median 0.28 s vs v1's 0.23 s, trail 0.31 s vs 0.31 s), so trimming would only
add a processing step v1 never had.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

KIT = Path(__file__).resolve().parent
SPEAKER = "az-01-main"
SILENCE_FLOOR_DB = -40.0
SCORE_KEYS = ("seconds", "cer", "buzz", "ring", "similarity", "range_st", "final_st", "peak_rise_st", "end_drop_st")


def dataset_rows(selection: list[dict], rows: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """metadata.jsonl and clip_map.jsonl rows for the chosen takes, sorted by sentence id."""
    metadata, clip_map = [], []
    for item in sorted((s for s in selection if "take" in s), key=lambda s: s["id"]):
        name = f"v2_{item['id']}.wav"
        metadata.append({"audio": f"audio/{SPEAKER}/{name}", "text": rows[item["id"]]["text"], "speaker": SPEAKER})
        clip_map.append({"file": name, "id": item["id"], "cat": item["cat"], "take": item["take"],
                         "source": f"takes/{item['take']}/{item['id']}.wav"})
    return metadata, clip_map


def edge_silence(path: Path) -> tuple[float, float]:
    """Seconds before the first and after the last 10 ms frame within 40 dB of the loudest."""
    wave, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = wave.mean(axis=1)
    hop = rate // 100
    frames = len(mono) // hop
    rms = np.sqrt(np.mean(mono[: frames * hop].reshape(frames, hop) ** 2, axis=1) + 1e-12)
    loud = np.where(20 * np.log10(rms / rms.max()) > SILENCE_FLOOR_DB)[0]
    return loud[0] / 100, (frames - 1 - loud[-1]) / 100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--selection", default="selection")
    parser.add_argument("--scores", default="scores")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    rows = {r["id"]: r for r in map(json.loads, (KIT / "sentences_v2.jsonl").read_text(encoding="utf-8").splitlines())}
    selection = [json.loads(line) for line in (args.run / args.selection / "selection.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((args.run / args.selection / "summary.json").read_text(encoding="utf-8"))
    takes = summary["takes"]
    scores = {t: {s["id"]: s for s in map(json.loads, (args.run / args.scores / f"{t}.jsonl").read_text(encoding="utf-8").splitlines())}
              for t in takes}
    metadata, clip_map = dataset_rows(selection, rows)

    audio_dir = args.out / "audio" / SPEAKER
    audio_dir.mkdir(parents=True, exist_ok=True)

    def place(entry: dict) -> tuple[str, float, float]:
        target = audio_dir / entry["file"]
        shutil.copyfile(args.run / entry["source"], target)
        lead, trail = edge_silence(target)
        return sha256(target), lead, trail

    with ThreadPoolExecutor(args.workers) as pool:
        placed = list(pool.map(place, clip_map))
    for entry, (_, lead, trail) in zip(clip_map, placed):
        score = scores[entry["take"]][entry["id"]]
        entry.update({key: score.get(key) for key in SCORE_KEYS}, lead_s=lead, trail_s=trail)

    write_jsonl(args.out / "metadata.jsonl", metadata)
    write_jsonl(args.out / "clip_map.jsonl", clip_map)
    (args.out / "checksums.sha256").write_text(
        "".join(f"{digest}  audio/{SPEAKER}/{entry['file']}\n" for entry, (digest, _, _) in zip(clip_map, placed)),
        newline="\n")

    dropped = [s for s in selection if "take" not in s]
    qc = {
        "sentences": len(selection), "kept": len(metadata), "dropped": len(dropped),
        "hours": round(sum(e["seconds"] for e in clip_map) / 3600, 2),
        "kept_per_category": dict(sorted(Counter(e["cat"] for e in clip_map).items())),
        "chosen_take": dict(sorted(Counter(e["take"] for e in clip_map).items())),
        "drop_reasons": dict(Counter(r for s in dropped for r in set(s["dropped"].values()))),
        "edge_silence_median_s": {"lead": round(statistics.median(e["lead_s"] for e in clip_map), 3),
                                  "trail": round(statistics.median(e["trail_s"] for e in clip_map), 3)},
        "gates": summary["gates"], "question_takes": summary.get("question_takes", takes),
        "bias_guard": summary["bias_guard"],
    }
    (args.out / "qc_summary.json").write_text(json.dumps(qc, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
    print(json.dumps({k: qc[k] for k in ("sentences", "kept", "dropped", "hours", "edge_silence_median_s")}))


def write_jsonl(path: Path, items: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
