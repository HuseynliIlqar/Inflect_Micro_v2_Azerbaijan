"""Hard-text test through the shipped packages: numbers, dates, abbreviations, foreign words,
long words and sentences, punctuation, questions, consonant drills, one-word replies and
stress homographs (hardtext.json). The package does its own normalisation and phonemisation,
exactly as a user's text would go.

    render (training venv, once per package):
        python hardtext.py render --package /workspace/final/avaz-v2 --name new [--variation 0.667]
        python hardtext.py render --package /workspace/base --name base
    score  (venv_asr): python hardtext.py score
Writes /workspace/hardtext/<name>/NN.wav, frontend.json per version and hardtext.json (all results).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parent
OUT = Path("/workspace/hardtext")


def render(package: Path, name: str, variation: float) -> None:
    import soundfile as sf

    sys.path.insert(0, str(package))
    from inflect_vits_frontend import run_vits_frontend
    from inference import InflectTTS

    items = json.loads((KIT / "hardtext.json").read_text(encoding="utf-8"))
    engine = InflectTTS(package, device="cuda")
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    frontend = []
    for index, item in enumerate(items):
        try:  # a package that cannot read a sentence is itself a finding: record it and go on
            rate, wave = engine.synthesize(item["text"], variation=variation, seed=0)
            sf.write(folder / f"{index:02d}.wav", wave, rate)
            output = run_vits_frontend(item["text"])
            frontend.append({"normalized": output.normalized_text, "phonemes": output.phoneme_text})
        except Exception as error:
            frontend.append({"error": f"{type(error).__name__}: {error}"})
    (folder / "frontend.json").write_text(json.dumps(frontend, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"rendered {name}: {len(items)}", flush=True)


def score() -> None:
    from faster_whisper import WhisperModel

    from eval_asr import MODEL, edit_distance, transcribe, word_list

    items = json.loads((KIT / "hardtext.json").read_text(encoding="utf-8"))
    engine = WhisperModel(MODEL, device="cuda", compute_type="float16")
    names = sorted(p.name for p in OUT.iterdir() if p.is_dir())
    fronts = {n: json.loads((OUT / n / "frontend.json").read_text(encoding="utf-8")) for n in names}
    results = []
    for index, item in enumerate(items):
        row = {"cat": item["cat"], "text": item["text"]}
        for name in names:
            wav = OUT / name / f"{index:02d}.wav"
            if not wav.exists():
                row[name] = {"hyp": "", "wer_vs_normalized": None, **fronts[name][index]}
                continue
            hyp = transcribe(engine, str(wav))
            ref = word_list(fronts[name][index].get("normalized") or item["text"])
            row[name] = {"hyp": hyp, "wer_vs_normalized": round(edit_distance(ref, word_list(hyp)) / max(len(ref), 1), 3),
                         **fronts[name][index]}
        results.append(row)
    (OUT / "hardtext.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    for row in results:
        print(f"[{row['cat']}] {row['text']}")
        for name in names:
            r = row[name]
            print(f"   {name:9s} norm: {r.get('normalized', r.get('error'))}")
            print(f"   {'':9s} heard: {r['hyp']}  (wer {r['wer_vs_normalized']})")
            if r.get("phonemes"):
                print(f"   {'':9s} phon: {r['phonemes']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("render", "score"))
    parser.add_argument("--package", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--variation", type=float, default=0.667)
    args = parser.parse_args()
    if args.mode == "render":
        render(args.package, args.name, args.variation)
    else:
        score()
