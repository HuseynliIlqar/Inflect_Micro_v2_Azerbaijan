"""Read every held-out validation sentence with the old Avaz and with chosen checkpoints, for the
pronunciation check (eval_asr.py transcribes them afterwards).

    python render_eval.py --checkpoint /workspace/runs/polish/exports/model-step-00025000.pth \
                          --checkpoint /workspace/runs/polish/exports/model-step-00030000.pth

Writes <out>/<version>/NNNN.wav and <out>/rows.json (text and the original recording per row).
The new models read the stored stress phonemes from validation.jsonl; the old Avaz reads the
text with its own eSpeak frontend. Same seed and variation for every version.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT))
from listen import SEED, VARIATION, load_weights  # noqa: E402


def load_rows(prepared: Path) -> list[dict]:
    rows = []
    for line in (prepared / "validation.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows.append({"text": row["normalized_text"], "phonemes": row["phonemes"],
                         "target": str(prepared / row["audio"])})
    return rows


def render(engine, rows: list[dict], folder: Path, use_phonemes: bool) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        phonemes = row["phonemes"] if use_phonemes else None
        rate, wave = engine.synthesize(row["text"], phonemes=phonemes, variation=VARIATION, seed=SEED)
        sf.write(folder / f"{index:04d}.wav", wave, rate)
    print(f"rendered {folder.name}: {len(rows)} clips", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, default=Path("/workspace/asr_eval"))
    parser.add_argument("--base", type=Path, default=Path("/workspace/base"))
    parser.add_argument("--prepared", type=Path, default=Path("/workspace/prepared"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    rows = load_rows(args.prepared)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rows.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    sys.path.insert(0, str(args.base))
    from inference import InflectTTS  # the package's own runtime

    engine = InflectTTS(args.base, device=args.device)
    render(engine, rows, args.out / "base", use_phonemes=False)
    for checkpoint in args.checkpoint:
        load_weights(engine, checkpoint)
        render(engine, rows, args.out / checkpoint.stem.replace("model-", ""), use_phonemes=True)


if __name__ == "__main__":
    main()
