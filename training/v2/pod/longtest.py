"""Read long paragraphs with the old Avaz and with a new checkpoint, for a side-by-side listen.

    python longtest.py --checkpoint /workspace/runs/polish/exports/model-step-00030000.pth \
                       [--out /workspace/longtest] [--base /workspace/base]

Both versions get the same sentence split, the same pauses and the same seeds; only the model
(and its frontend: eSpeak for the old Avaz, the stress frontend for the new one) differs.
Writes <out>/<name>/NN.wav, <out>/texts.json and <out>/metrics.json. CPU only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT))
from listen import SEED, VARIATION, _summary, load_engine, load_stress_frontend, load_weights  # noqa: E402

PARAGRAPH_PAUSE = 0.55  # seconds between paragraphs, on top of the sentence pauses


def read_text(engine, paragraphs: list[str], frontend) -> np.ndarray:
    from inference import boundary_pause_seconds, split_text  # the package runtime, on sys.path via load_engine

    rate = engine.sample_rate
    pieces, index = [], 0
    for p_index, paragraph in enumerate(paragraphs):
        if p_index:
            pieces.append(np.zeros(round(rate * PARAGRAPH_PAUSE), np.float32))
        chunks = split_text(paragraph)
        for c_index, chunk in enumerate(chunks):
            if c_index:
                pieces.append(np.zeros(round(rate * boundary_pause_seconds(chunks[c_index - 1])), np.float32))
            if frontend is None:
                _, wave = engine.synthesize(chunk, variation=VARIATION, seed=SEED + index)
            else:
                phonemes = frontend.phonemize(frontend.normalize(chunk))
                _, wave = engine.synthesize(chunk, phonemes=phonemes, variation=VARIATION, seed=SEED + index)
            pieces.append(np.asarray(wave, dtype=np.float32))
            index += 1
    return np.concatenate(pieces)


def render_all(engine, texts: list[dict], folder: Path, frontend) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for number, text in enumerate(texts):
        path = folder / f"{number:02d}.wav"
        sf.write(path, read_text(engine, text["paragraphs"], frontend), engine.sample_rate)
        paths.append(path)
        print(f"{folder.name}/{path.name}", flush=True)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("/workspace/longtest"))
    parser.add_argument("--base", type=Path, default=Path("/workspace/base"))
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    texts = json.loads((KIT / "longtest_texts.json").read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "texts.json").write_text(json.dumps(texts, ensure_ascii=False, indent=1), encoding="utf-8")

    engine = load_engine(args.base)
    versions = {"base": render_all(engine, texts, args.out / "base", None)}
    load_weights(engine, args.checkpoint)
    new_name = args.checkpoint.stem.replace("model-", "")
    versions[new_name] = render_all(engine, texts, args.out / new_name, load_stress_frontend())

    metrics = {name: [dict(_summary([path]), seconds=round(sf.info(path).duration, 1)) for path in paths]
               for name, paths in versions.items()}
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
