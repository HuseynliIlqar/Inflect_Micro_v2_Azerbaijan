"""Track the pitch of every clip of a dataset and pack it into one file for training.

    python extract_pitch.py --audio-dir D:/ktts/teacher/dataset/audio/az-01-main --out pitch_v1.npz

Writes <out> with:
- stems: clip file names without extension (the prepared dataset keeps them in its file names);
- offsets: int64, clip i's frames are f0[offsets[i]:offsets[i + 1]];
- f0: float32 Hz per model frame (hop 256 at 24 kHz), 0 where unvoiced;
and <out>.json with counts and the voice's pitch distribution.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_track import track  # noqa: E402


def _one(path: str) -> np.ndarray:
    wave, rate = sf.read(path, dtype="float64", always_2d=True)
    return track(wave.mean(axis=1), rate)


def summarise(f0: np.ndarray, clips: int) -> dict:
    voiced = f0[f0 > 0]
    semitones = 12 * np.log2(voiced / np.median(voiced))
    return {"clips": clips, "frames": int(len(f0)), "hours": round(len(f0) * 256 / 24000 / 3600, 2),
            "voiced_share": round(float(len(voiced) / len(f0)), 4),
            "median_hz": round(float(np.median(voiced)), 2),
            "p1_p5_p95_p99_semitones": [round(float(np.percentile(semitones, q)), 2) for q in (1, 5, 95, 99)]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.wav")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    paths = sorted(args.audio_dir.glob(args.pattern))
    if not paths:
        raise SystemExit(f"no {args.pattern} in {args.audio_dir}")
    started = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        tracks = list(pool.map(_one, [str(p) for p in paths], chunksize=32))
    offsets = np.concatenate(([0], np.cumsum([len(t) for t in tracks]))).astype(np.int64)
    f0 = np.concatenate(tracks).astype(np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, stems=np.array([p.stem for p in paths]), offsets=offsets, f0=f0)
    summary = {**summarise(f0, len(paths)), "seconds_taken": round(time.time() - started, 1),
               "audio_dir": str(args.audio_dir)}
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
