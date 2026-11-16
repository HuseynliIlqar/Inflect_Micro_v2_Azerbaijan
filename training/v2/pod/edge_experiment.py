"""Is the model's weakness at sentence edges fixable at inference time?

    render (training venv): python edge_experiment.py render
    score  (venv_asr):      python edge_experiment.py score

Variants of the 30k checkpoint on the same validation sentences:
  lead-pause: a comma pause symbol before the first phoneme, so the first word no longer starts the sequence
  var-0.33:   half the sampling noise (variation 0.333 instead of 0.667)
Scored on the same 416 number-free rows as asr_analysis.py, against the existing baseline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path("/workspace/asr_eval")
CHECKPOINT = Path("/workspace/runs/polish/exports/model-step-00030000.pth")
BASELINE = "step-00030000"
VARIANTS = {"exp-lead-pause": {"prefix": ", ", "variation": 0.667},
            "exp-var-0.33": {"prefix": "", "variation": 0.333},
            "exp-var-0.2": {"prefix": "", "variation": 0.2},
            "exp-var-0.5": {"prefix": "", "variation": 0.5}}


def render() -> None:
    import soundfile as sf

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, "/workspace/base")
    from inference import InflectTTS
    from listen import SEED, load_weights

    rows = json.loads((OUT / "rows.json").read_text(encoding="utf-8"))
    engine = InflectTTS("/workspace/base", device="cuda")
    load_weights(engine, CHECKPOINT)
    for name, setting in VARIANTS.items():
        folder = OUT / name
        if folder.is_dir():
            continue
        folder.mkdir()
        for index, row in enumerate(rows):
            rate, wave = engine.synthesize(row["text"], phonemes=setting["prefix"] + row["phonemes"],
                                           variation=setting["variation"], seed=SEED)
            sf.write(folder / f"{index:04d}.wav", wave, rate)
        print(f"rendered {name}", flush=True)


def score() -> None:
    from faster_whisper import WhisperModel

    from asr_analysis import length_bin, letter_errors, wer_of
    from eval_asr import MODEL, transcribe
    from gap_analysis import position_errors

    rows = json.loads((OUT / "asr_rows.json").read_text(encoding="utf-8"))
    clean_index = [i for i, r in enumerate(rows) if not any(ch.isdigit() for k, v in r.items() if k != "text" for ch in v)]
    engine = WhisperModel(MODEL, device="cuda", compute_type="float16")
    for name in VARIANTS:
        cache = OUT / name / "hyps.json"
        if not cache.exists():
            cache.write_text(json.dumps({i: transcribe(engine, str(OUT / name / f"{i:04d}.wav")) for i in clean_index},
                                        ensure_ascii=False), encoding="utf-8")
        hyps = json.loads(cache.read_text(encoding="utf-8"))
        for i in clean_index:
            rows[i][name] = hyps[str(i)]
    clean = [rows[i] for i in clean_index]
    short = [r for r in clean if length_bin(r["text"]) == "1-5"]
    report = {}
    for version in ("dataset", BASELINE, *VARIANTS):
        letters, _ = letter_errors(clean, version)
        report[version] = {"wer_clean": wer_of(clean, version)["wer"], "wer_short": wer_of(short, version)["wer"],
                           "position": position_errors(clean, version),
                           "m_error": round(letters["m"][1], 4), "q_error": round(letters["q"][1], 4)}
    (OUT / "edge_experiment.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    {"render": render, "score": score}[sys.argv[1]]()
