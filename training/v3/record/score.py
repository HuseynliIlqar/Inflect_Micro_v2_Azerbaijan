"""Measure every clip of a take: length, level, Whisper reading, voice, buzz, ring and pitch.

    python score.py --take A --takes-dir /workspace/rec/takes --out /workspace/rec/scores \
                    --centroid /workspace/rec/v1_centroid.npy

Writes <out>/<take>.jsonl, one object per clip. Clips already scored are skipped, so a rerun
resumes. The GPU part (Whisper, ECAPA) runs in this process and the CPU part (buzz, Praat
pitch) in a process pool.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT))
from common import cer, final_movement_st, rise_fall_st  # noqa: E402

WHISPER = "hasan0v/whisper-turbo-az-ct2"
CHUNK = 400
LOG = logging.getLogger("score")


def level_facts(path: str) -> dict:
    wave, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = wave.mean(axis=1)
    return {"seconds": round(len(mono) / rate, 3), "peak": round(float(np.max(np.abs(mono))), 4),
            "clipped": round(float(np.mean(np.abs(mono) > 0.999)), 5)}


def cpu_facts(path: str) -> dict:
    """Buzz and ring (buzz_measure.py, as in every earlier test) and the Praat pitch measures,
    including the question shape (rise_fall_st) from the cleaned track."""
    import buzz_measure
    import parselmouth
    from pitch import robust_f0

    buzz = buzz_measure.measure(path)
    wave, rate = sf.read(path, dtype="float64", always_2d=True)
    sound = parselmouth.Sound(wave.mean(axis=1), sampling_frequency=rate)
    f0 = sound.to_pitch_ac(time_step=0.01, pitch_floor=70, pitch_ceiling=450).selected_array["frequency"]
    voiced = f0[f0 > 0]
    spread = float(12 * np.log2(np.percentile(voiced, 95) / np.percentile(voiced, 5))) if len(voiced) >= 20 else math.nan
    peak_rise, end_drop = rise_fall_st(list(robust_f0(sound)[0]))
    return {**level_facts(path), "buzz": round(float(buzz["h2-4k"]), 3), "ring": round(float(buzz["ring_share"]), 4),
            "range_st": round(spread, 3), "final_st": round(final_movement_st(list(f0)), 3),
            "peak_rise_st": round(peak_rise, 3), "end_drop_st": round(end_drop, 3)}


class GpuMeasures:
    """Whisper transcription and ECAPA voice embedding."""

    def __init__(self) -> None:
        import torch
        from faster_whisper import WhisperModel
        from speechbrain.inference.speaker import EncoderClassifier

        self.torch = torch
        self.whisper = WhisperModel(WHISPER, device="cuda", compute_type="float16")
        self.ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                    savedir="/workspace/hf/ecapa", run_opts={"device": "cuda"})

    def transcribe(self, path: str) -> str:
        segments, _ = self.whisper.transcribe(path, language="az", beam_size=5, vad_filter=False,
                                              condition_on_previous_text=False)
        return " ".join(segment.text.strip() for segment in segments)

    def embed(self, path: str) -> np.ndarray:
        import torchaudio

        wave, rate = sf.read(path, dtype="float32", always_2d=True)
        tensor = torchaudio.functional.resample(self.torch.from_numpy(wave.mean(axis=1))[None], rate, 16000)
        vector = self.ecapa.encode_batch(tensor.to("cuda")).squeeze().cpu().numpy()
        return vector / np.linalg.norm(vector)


def measure(items: list[tuple[str, str, str]], gpu: GpuMeasures, centroid: np.ndarray | None,
            workers: int) -> list[dict]:
    """items: (id, text, wav path). Returns one score dict per item, in order."""
    with Pool(workers) as pool:
        cpu = pool.map(cpu_facts, [path for _, _, path in items], chunksize=4)
    rows = []
    for (clip_id, text, path), facts in zip(items, cpu):
        hypothesis = gpu.transcribe(path)
        vector = gpu.embed(path)
        similarity = float(vector @ centroid) if centroid is not None else math.nan
        rows.append({"id": clip_id, **facts, "hyp": hypothesis, "cer": round(cer(text, hypothesis), 4),
                     "similarity": round(similarity, 4), "embedding": vector.round(5).tolist() if centroid is None else None})
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--take", required=True)
    parser.add_argument("--takes-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--centroid", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    rows = {r["id"]: r for r in map(json.loads, (KIT / "sentences_v2.jsonl").read_text(encoding="utf-8").splitlines())}
    folder = args.takes_dir / args.take
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / f"{args.take}.jsonl"
    done = {json.loads(line)["id"] for line in target.read_text(encoding="utf-8").splitlines()} if target.exists() else set()
    todo = [(p.stem, rows[p.stem]["text"], str(p)) for p in sorted(folder.glob("*.wav"))
            if not p.stem.endswith(".part") and p.stem not in done and p.stem in rows]
    LOG.info("take %s: %d clips to score", args.take, len(todo))
    centroid = np.load(args.centroid)
    gpu = GpuMeasures()
    for start in range(0, len(todo), CHUNK):
        scored = measure(todo[start : start + CHUNK], gpu, centroid, args.workers)
        with target.open("a", encoding="utf-8") as handle:
            for row in scored:
                row.pop("embedding", None)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        LOG.info("take %s: %d / %d scored", args.take, min(start + CHUNK, len(todo)), len(todo))


if __name__ == "__main__":
    main()
