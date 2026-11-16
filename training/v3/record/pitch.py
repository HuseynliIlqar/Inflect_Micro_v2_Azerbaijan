"""Pitch tracking shared by score.py and question_contours.py (needs Praat via parselmouth)."""
from __future__ import annotations

import numpy as np
import parselmouth


def robust_f0(sound: parselmouth.Sound) -> tuple[np.ndarray, np.ndarray]:
    """Two passes: the first finds the voice's median, the second caps the ceiling at 1.8x it
    (about 10 semitones) so the tracker cannot jump an octave up; then a 5-frame median filter.
    Returns (f0 per 10 ms frame with 0 for unvoiced, frame times)."""
    first = sound.to_pitch_ac(time_step=0.01, pitch_floor=70, pitch_ceiling=450).selected_array["frequency"]
    median = float(np.median(first[first > 0])) if (first > 0).sum() >= 10 else 200.0
    pitch = sound.to_pitch_ac(time_step=0.01, pitch_floor=max(60.0, median / 2.2), pitch_ceiling=min(600.0, median * 1.8))
    f0 = pitch.selected_array["frequency"]
    smooth = f0.copy()
    for i in range(len(f0)):
        window = f0[max(0, i - 2) : i + 3]
        window = window[window > 0]
        smooth[i] = np.median(window) if f0[i] > 0 and len(window) else 0.0
    return smooth, pitch.xs()
