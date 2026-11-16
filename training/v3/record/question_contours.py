"""Look closer at the pitch at the end of yes/no questions before trusting the rise gate.

    python question_contours.py --takes-dir /workspace/rec/takes --takes A B C [D E F] --ids pilot_ids.txt

For every A1-A3 take: the current final movement, the peak of the last 0.8 s against the
median of the rest (a rise on the syllable before "-mı" counts even if "-mı" itself falls),
the drop from that peak to the end, and whether the tracker jumped by more than 9 semitones
between neighbouring frames (an octave error). Writes question_contours.json next to the takes.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT))
from common import RISING, final_movement_st, rise_fall_st  # noqa: E402
from pitch import robust_f0  # noqa: E402

JUMP_ST = 9.0


def contour(path: str) -> dict:
    wave, rate = sf.read(path, dtype="float64", always_2d=True)
    sound = parselmouth.Sound(wave.mean(axis=1), sampling_frequency=rate)
    f0, _ = robust_f0(sound)
    voiced = f0 > 0
    if voiced.sum() < 20:
        return {"final_st": float("nan"), "peak_rise_st": float("nan"), "end_drop_st": float("nan"), "octave_jump": False}
    st = 12 * np.log2(f0[voiced] / 100)
    peak_rise, end_drop = rise_fall_st(list(f0))
    return {"final_st": round(final_movement_st(list(f0)), 2), "peak_rise_st": round(peak_rise, 2),
            "end_drop_st": round(end_drop, 2), "octave_jump": bool((np.abs(np.diff(st)) > JUMP_ST).any())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--takes-dir", type=Path, required=True)
    parser.add_argument("--takes", nargs="+", required=True)
    parser.add_argument("--ids", type=Path, required=True)
    args = parser.parse_args()
    rows = {r["id"]: r for r in map(json.loads, (KIT / "sentences_v2.jsonl").read_text(encoding="utf-8").splitlines())}
    ids = [i for i in args.ids.read_text(encoding="utf-8").split() if rows[i]["cat"] in RISING]
    jobs = [(i, t, str(args.takes_dir / t / f"{i}.wav")) for i in ids for t in args.takes if (args.takes_dir / t / f"{i}.wav").exists()]
    with Pool(8) as pool:
        measured = pool.map(contour, [path for _, _, path in jobs])
    result = [{"id": i, "take": t, **m} for (i, t, _), m in zip(jobs, measured)]
    (args.takes_dir / "question_contours.json").write_text(json.dumps(result, indent=1))
    finite = lambda key: [r[key] for r in result if r[key] == r[key]]  # noqa: E731
    by_sentence = {}
    for r in result:
        by_sentence.setdefault(r["id"], []).append(r)
    summary = {
        "takes": len(result), "octave_jump_share": round(sum(r["octave_jump"] for r in result) / len(result), 3),
        "final_st_p50": round(statistics.median(finite("final_st")), 2),
        "peak_rise_p50": round(statistics.median(finite("peak_rise_st")), 2),
        "end_drop_p50": round(statistics.median(finite("end_drop_st")), 2),
        "rise_fall_share": round(sum(r["peak_rise_st"] >= 2 and r["final_st"] < 1 for r in result) / len(result), 3),
        "sentences_with_final_rise": sum(any(r["final_st"] >= 1 for r in rs) for rs in by_sentence.values()),
        "sentences_with_peak_rise": sum(any(r["peak_rise_st"] >= 2 for r in rs) for rs in by_sentence.values()),
        "sentences": len(by_sentence),
    }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
