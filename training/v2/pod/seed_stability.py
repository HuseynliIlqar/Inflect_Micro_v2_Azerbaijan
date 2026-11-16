"""Random or systematic? The same sentences read with five seeds at two sampling-noise levels.

A sentence that fails with every seed points at the model or the frontend; one that fails with
some seeds points at sampling noise.

    render (training venv): python seed_stability.py render
    score  (venv_asr):      python seed_stability.py score
Writes /workspace/asr_eval/seeds/<variation>/<row>-<seed>.wav and /workspace/asr_eval/seed_stability.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path("/workspace/asr_eval")
SEEDS = range(5)
VARIATIONS = (0.667, 0.333)
STEP = 3  # every third number-free validation row


def chosen_rows() -> list[int]:
    rows = json.loads((OUT / "asr_rows.json").read_text(encoding="utf-8"))
    clean = [i for i, r in enumerate(rows) if not any(ch.isdigit() for k, v in r.items() if k != "text" for ch in v)]
    return clean[::STEP]


def render() -> None:
    import soundfile as sf

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, "/workspace/base")
    from inference import InflectTTS
    from listen import load_weights

    rows = json.loads((OUT / "rows.json").read_text(encoding="utf-8"))
    engine = InflectTTS("/workspace/base", device="cuda")
    load_weights(engine, Path("/workspace/runs/polish/exports/model-step-00030000.pth"))
    for variation in VARIATIONS:
        folder = OUT / "seeds" / f"{variation}"
        folder.mkdir(parents=True, exist_ok=True)
        for i in chosen_rows():
            for seed in SEEDS:
                rate, wave = engine.synthesize(rows[i]["text"], phonemes=rows[i]["phonemes"], variation=variation, seed=seed)
                sf.write(folder / f"{i:04d}-{seed}.wav", wave, rate)
    print("rendered seeds", flush=True)


def score() -> None:
    from faster_whisper import WhisperModel

    from eval_asr import MODEL, edit_distance, transcribe, word_list

    rows = json.loads((OUT / "asr_rows.json").read_text(encoding="utf-8"))
    engine = WhisperModel(MODEL, device="cuda", compute_type="float16")
    report = {}
    for variation in VARIATIONS:
        folder = OUT / "seeds" / f"{variation}"
        per_sentence = []
        for i in chosen_rows():
            ref = word_list(rows[i]["text"])
            fails = [edit_distance(ref, word_list(transcribe(engine, str(folder / f"{i:04d}-{seed}.wav")))) > 0
                     for seed in SEEDS]
            per_sentence.append({"row": i, "text": rows[i]["text"], "failed_seeds": sum(fails),
                                 "dataset_ok": edit_distance(ref, word_list(rows[i]["dataset"])) == 0})
        n = len(per_sentence)
        report[str(variation)] = {
            "sentences": n,
            "never_fail": round(sum(s["failed_seeds"] == 0 for s in per_sentence) / n, 3),
            "sometimes_fail": round(sum(0 < s["failed_seeds"] < len(SEEDS) for s in per_sentence) / n, 3),
            "always_fail": round(sum(s["failed_seeds"] == len(SEEDS) for s in per_sentence) / n, 3),
            "always_fail_but_dataset_ok": [s["text"] for s in per_sentence if s["failed_seeds"] == len(SEEDS) and s["dataset_ok"]],
            "clip_error_rate": round(sum(s["failed_seeds"] for s in per_sentence) / (n * len(SEEDS)), 3),
        }
    (OUT / "seed_stability.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    {"render": render, "score": score}[sys.argv[1]]()
