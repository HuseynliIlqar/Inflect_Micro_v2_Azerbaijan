"""Give the validation section its "before training" and "target" clips.

    python validation_before.py --run /workspace/runs/main

The toolkit's validation reads the first row of prepared/validation.jsonl (batch 1, no
shuffle) at every validation step. This writes, next to those clips:
  validation/before.wav  the same sentence read by the old Avaz with its own frontend
  validation/target.wav  the dataset recording of it (the teacher TTS), what the model learns from
  validation/sentence.json  the text
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import soundfile as sf

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT))
from listen import SEED, VARIATION, load_engine  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, default=Path("/workspace/prepared"))
    parser.add_argument("--base", type=Path, default=Path("/workspace/base"))
    args = parser.parse_args()

    with (args.prepared / "validation.jsonl").open(encoding="utf-8") as handle:
        row = json.loads(handle.readline())
    folder = args.run / "validation"
    folder.mkdir(parents=True, exist_ok=True)
    engine = load_engine(args.base)
    rate, wave = engine.synthesize(row["text"], variation=VARIATION, seed=SEED)
    sf.write(folder / "before.wav.tmp", wave, rate, format="WAV")
    (folder / "before.wav.tmp").rename(folder / "before.wav")
    shutil.copyfile(args.prepared / row["audio"], folder / "target.wav")
    (folder / "sentence.json").write_text(json.dumps({"text": row["text"]}, ensure_ascii=False), encoding="utf-8")
    print(f"wrote before.wav ({len(wave) / rate:.2f} s) and target.wav for: {row['text']}")


if __name__ == "__main__":
    main()
