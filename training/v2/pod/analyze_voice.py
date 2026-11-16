"""Voice and prosody measurements on the validation sentences, for the original recordings
(dataset), the old Avaz (base) and new checkpoints: same text for every version.

    /workspace/venv_asr/bin/python analyze_voice.py [--out /workspace/asr_eval]

Per clip: articulation rate, pauses, pitch (median, range, final movement), jitter, shimmer,
harmonics-to-noise ratio and a band spectrum. Writes <out>/voice_rows.json and
<out>/voice_summary.json (medians per version and sentence type, pairwise gaps to the dataset,
mean band spectrum).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf
from parselmouth.praat import call

VOWELS = frozenset("aeəıioöuü")
FRAME = 0.01
SILENCE_DB = 40.0  # a frame is silent this far below the clip's loudest 10 ms frame
MIN_PAUSE = 0.08
F0_FLOOR, F0_CEIL = 70.0, 450.0
TAIL, BEFORE = 15, 40  # voiced frames for the final pitch movement
BAND_EDGES = (50, 250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000)
_LOWER = str.maketrans({"I": "ı", "İ": "i"})
SCALARS = ("rate", "pauses", "pause_seconds", "f0_median", "f0_range_st", "final_st",
           "voiced_share", "jitter", "shimmer", "hnr")


def sentence_type(text: str) -> str:
    end = text.rstrip()[-1:]
    return {"?": "question", "!": "exclamation"}.get(end, "statement")


def length_bin(text: str) -> str:
    words = len(text.split())
    return "1-5" if words <= 5 else "6-10" if words <= 10 else "11-15" if words <= 15 else "16+"


def _frames_db(wave: np.ndarray, rate: int) -> np.ndarray:
    hop = int(rate * FRAME)
    count = len(wave) // hop
    frames = wave[: count * hop].reshape(count, hop)
    return 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-9)


def timing(wave: np.ndarray, rate: int, text: str) -> dict:
    db = _frames_db(wave, rate)
    loud = np.flatnonzero(db > db.max() - SILENCE_DB)
    start, end = int(loud[0]), int(loud[-1])
    silent = db[start:end + 1] <= db.max() - SILENCE_DB
    pauses, run = [], 0
    for flag in list(silent) + [False]:
        if flag:
            run += 1
        else:
            if run * FRAME >= MIN_PAUSE:
                pauses.append(run * FRAME)
            run = 0
    speech = (end - start + 1) * FRAME - sum(pauses)
    syllables = sum(ch in VOWELS for ch in text.translate(_LOWER).lower())
    return {"rate": syllables / max(speech, 1e-3), "pauses": len(pauses), "pause_seconds": sum(pauses),
            "span": (start, end)}


def pitch_and_quality(wave: np.ndarray, rate: int) -> dict:
    sound = parselmouth.Sound(wave.astype(np.float64), sampling_frequency=rate)
    f0 = sound.to_pitch_ac(time_step=FRAME, pitch_floor=F0_FLOOR, pitch_ceiling=F0_CEIL).selected_array["frequency"]
    voiced = f0[f0 > 0]
    out = {"voiced_share": float(len(voiced) / max(len(f0), 1)), "f0_median": math.nan,
           "f0_range_st": math.nan, "final_st": math.nan}
    if len(voiced) >= 20:
        out["f0_median"] = float(np.median(voiced))
        out["f0_range_st"] = float(12 * np.log2(np.percentile(voiced, 95) / np.percentile(voiced, 5)))
    if len(voiced) >= TAIL + BEFORE:
        out["final_st"] = float(12 * np.log2(np.median(voiced[-TAIL:]) / np.median(voiced[-TAIL - BEFORE:-TAIL])))
    points = call(sound, "To PointProcess (periodic, cc)", F0_FLOOR, F0_CEIL)
    out["jitter"] = float(call(points, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
    out["shimmer"] = float(call([sound, points], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
    harmonicity = sound.to_harmonicity_cc(time_step=FRAME, minimum_pitch=F0_FLOOR, silence_threshold=0.1,
                                          periods_per_window=1.0)
    out["hnr"] = float(call(harmonicity, "Get mean", 0, 0))
    return out


def band_spectrum(wave: np.ndarray, rate: int, span: tuple[int, int]) -> list[float]:
    """Power share per band in dB over the speech part, so loudness differences cancel out."""
    hop = int(rate * FRAME)
    speech = wave[span[0] * hop:(span[1] + 1) * hop]
    n_fft = 2048
    if len(speech) < n_fft:
        speech = np.pad(speech, (0, n_fft - len(speech)))
    window = np.hanning(n_fft)
    starts = range(0, len(speech) - n_fft + 1, 512)
    power = np.mean([np.abs(np.fft.rfft(speech[s:s + n_fft] * window)) ** 2 for s in starts], axis=0)
    freqs = np.fft.rfftfreq(n_fft, 1 / rate)
    total = power[(freqs >= BAND_EDGES[0]) & (freqs < BAND_EDGES[-1])].sum()
    return [float(10 * np.log10(power[(freqs >= lo) & (freqs < hi)].sum() / total + 1e-12))
            for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:])]


def measure(job: tuple[str, int, str, str]) -> dict:
    version, index, path, text = job
    wave, rate = sf.read(path, dtype="float32")
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    row = {"version": version, "index": index, "type": sentence_type(text), "length": length_bin(text)}
    try:
        times = timing(wave, rate, text)
        row.update({k: v for k, v in times.items() if k != "span"})
        row.update(pitch_and_quality(wave, rate))
        row["bands"] = band_spectrum(wave, rate, times["span"])
    except Exception as error:  # one bad clip must not stop the run; it is reported in the rows
        row["error"] = repr(error)
    return row


def _median(values) -> float:
    finite = [v for v in values if isinstance(v, float | int) and not math.isnan(v)]
    return round(statistics.median(finite), 4) if finite else math.nan


def summarize(rows: list[dict], versions: list[str]) -> dict:
    by_version = {v: [r for r in rows if r["version"] == v and "error" not in r] for v in versions}
    summary = {"medians": {}, "by_type": {}, "pairwise_vs_dataset": {}, "bands_db": {},
               "band_edges": BAND_EDGES, "errors": sum("error" in r for r in rows)}
    for version, group in by_version.items():
        summary["medians"][version] = {k: _median(r.get(k) for r in group) for k in SCALARS}
        summary["by_type"][version] = {
            kind: {"clips": sum(r["type"] == kind for r in group),
                   "final_st": _median(r.get("final_st") for r in group if r["type"] == kind),
                   "f0_range_st": _median(r.get("f0_range_st") for r in group if r["type"] == kind)}
            for kind in ("statement", "question", "exclamation")}
        summary["bands_db"][version] = [round(float(np.mean([r["bands"][b] for r in group])), 2)
                                        for b in range(len(BAND_EDGES) - 1)]
    data = {r["index"]: r for r in by_version.get("dataset", [])}
    for version, group in by_version.items():
        if version == "dataset":
            continue
        pairs = [(r, data[r["index"]]) for r in group if r["index"] in data]
        summary["pairwise_vs_dataset"][version] = {
            "rate_ratio": _median(r["rate"] / d["rate"] for r, d in pairs),
            "f0_range_gap_st": _median(r["f0_range_st"] - d["f0_range_st"] for r, d in pairs),
            "final_gap_st_questions": _median(r["final_st"] - d["final_st"] for r, d in pairs if r["type"] == "question"),
            "pause_count_gap": _median(r["pauses"] - d["pauses"] for r, d in pairs),
            "hnr_gap_db": _median(r["hnr"] - d["hnr"] for r, d in pairs),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/workspace/asr_eval"))
    parser.add_argument("--versions", nargs="+", default=["base", "step-00025000", "step-00030000"])
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    rows = json.loads((args.out / "rows.json").read_text(encoding="utf-8"))
    jobs = [("dataset", i, row["target"], row["text"]) for i, row in enumerate(rows)]
    for version in args.versions:
        jobs += [(version, i, str(args.out / version / f"{i:04d}.wav"), row["text"]) for i, row in enumerate(rows)]
    with Pool(args.workers) as pool:
        measured = pool.map(measure, jobs, chunksize=8)
    (args.out / "voice_rows.json").write_text(json.dumps(measured), encoding="utf-8")
    summary = summarize(measured, ["dataset", *args.versions])
    (args.out / "voice_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
