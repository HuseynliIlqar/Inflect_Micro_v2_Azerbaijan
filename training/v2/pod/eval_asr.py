"""Transcribe the clips from render_eval.py with the Azerbaijani Whisper model and score word and
character errors per version, plus the original recordings as the ceiling.

    LD_LIBRARY_PATH=... /workspace/venv_asr/bin/python eval_asr.py [--out /workspace/asr_eval]

Same model, decoding settings and text normalisation as the earlier dataset check
(teacher/kit/eval_whisper.py and metrics_core.py). Writes <out>/asr_rows.json and
<out>/asr_summary.json.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from faster_whisper import WhisperModel

MODEL = "hasan0v/whisper-turbo-az-ct2"
BROKEN_WER = 0.5
_NOT_WORD = re.compile(r"[^\w\s]", re.UNICODE)
_LOWER = str.maketrans({"I": "ı", "İ": "i"})


def word_list(text: str) -> list[str]:
    return _NOT_WORD.sub(" ", text.translate(_LOWER).lower()).split()


def edit_distance(a, b) -> int:
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        diagonal, row[0] = row[0], i
        for j, y in enumerate(b, start=1):
            diagonal, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, diagonal + (x != y))
    return row[-1]


def score(pairs: list[tuple[str, str]]) -> dict:
    words = [(word_list(r), word_list(h)) for r, h in pairs]
    chars = [("".join(r), "".join(h)) for r, h in words]
    word_errors = [edit_distance(r, h) for r, h in words]
    broken = sum(e / max(len(r), 1) > BROKEN_WER or not h for e, (r, h) in zip(word_errors, words))
    return {"clips": len(pairs),
            "wer": sum(word_errors) / max(sum(len(r) for r, _ in words), 1),
            "cer": sum(edit_distance(list(r), list(h)) for r, h in chars) / max(sum(len(r) for r, _ in chars), 1),
            "broken": broken}


def transcribe(engine: WhisperModel, path: str) -> str:
    segments, _ = engine.transcribe(path, language="az", beam_size=5, vad_filter=False,
                                    condition_on_previous_text=False)
    return " ".join(segment.text.strip() for segment in segments)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/workspace/asr_eval"))
    args = parser.parse_args()
    rows = json.loads((args.out / "rows.json").read_text(encoding="utf-8"))
    versions = {"dataset": [row["target"] for row in rows]}
    for folder in sorted(p for p in args.out.iterdir() if p.is_dir()):
        versions[folder.name] = [str(folder / f"{i:04d}.wav") for i in range(len(rows))]

    engine = WhisperModel(MODEL, device="cuda", compute_type="float16")
    hyps = {}
    for name, paths in versions.items():
        hyps[name] = [transcribe(engine, path) for path in paths]
        print(f"transcribed {name}", flush=True)

    table = [{"text": row["text"], **{name: hyps[name][i] for name in versions}} for i, row in enumerate(rows)]
    (args.out / "asr_rows.json").write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    summary = {name: score([(row["text"], hyps[name][i]) for i, row in enumerate(rows)]) for name in versions}
    (args.out / "asr_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
