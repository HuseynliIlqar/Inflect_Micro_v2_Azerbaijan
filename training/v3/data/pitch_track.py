"""Pitch per model frame: the target the pitch part (stage 5) learns and the decoder (stage 6) uses.

The model's frames come from the VITS spectrogram: the audio is reflect-padded by
(filter - hop) / 2 = 384 samples and cut with hop 256 and window 1024, without centring. Frame i
therefore spans samples [256 i - 384, 256 i + 640) of the clip, its centre is sample 256 i + 128,
and a clip of N samples has N // 256 frames.

The tracker is Praat's autocorrelation in the two passes the take selection used
(record/pitch.py): the first finds the clip's median, the second bounds the search to
[median / 2.2, median x 1.8] so it cannot jump an octave, then a 5-frame median filter. Praat's
values are moved onto the model's frame centres: log-linear between two voiced Praat frames,
the nearest frame's value otherwise, and unvoiced (0) where the nearest Praat frame is unvoiced.
"""
from __future__ import annotations

import numpy as np
import parselmouth

SAMPLE_RATE = 24000
HOP = 256
FIRST_FLOOR, FIRST_CEILING = 70.0, 450.0


def frame_count(samples: int) -> int:
    return samples // HOP


def frame_centres(frames: int) -> np.ndarray:
    return (np.arange(frames) * HOP + HOP / 2) / SAMPLE_RATE


def _median_filter_voiced(f0: np.ndarray) -> np.ndarray:
    smooth = np.zeros_like(f0)
    for i in np.flatnonzero(f0 > 0):
        window = f0[max(0, i - 2): i + 3]
        smooth[i] = np.median(window[window > 0])
    return smooth


def praat_track(sound: parselmouth.Sound, step: float) -> tuple[np.ndarray, np.ndarray]:
    """(f0 with 0 for unvoiced, frame times) on Praat's own grid."""
    first = sound.to_pitch_ac(time_step=step, pitch_floor=FIRST_FLOOR, pitch_ceiling=FIRST_CEILING)
    values = first.selected_array["frequency"]
    median = float(np.median(values[values > 0])) if (values > 0).sum() >= 10 else 200.0
    pitch = sound.to_pitch_ac(time_step=step, pitch_floor=max(60.0, median / 2.2),
                              pitch_ceiling=min(600.0, median * 1.8))
    return _median_filter_voiced(pitch.selected_array["frequency"]), pitch.xs()


def to_model_frames(f0: np.ndarray, times: np.ndarray, frames: int) -> np.ndarray:
    """Move a track from Praat's grid onto the model's frame centres."""
    out = np.zeros(frames, dtype=np.float32)
    if len(times) == 0:
        return out
    centres = frame_centres(frames)
    right = np.clip(np.searchsorted(times, centres), 1, max(1, len(times) - 1))
    left = right - 1
    if len(times) == 1:
        left = right = np.zeros_like(right)
    nearest = np.where(np.abs(times[left] - centres) <= np.abs(times[right] - centres), left, right)
    step = HOP / SAMPLE_RATE
    voiced = (f0[nearest] > 0) & (np.abs(times[nearest] - centres) <= step)  # no extrapolation past the track
    both = (f0[left] > 0) & (f0[right] > 0) & (right != left)
    span = np.where(right != left, times[right] - times[left], 1.0)
    weight = np.clip((centres - times[left]) / span, 0.0, 1.0)
    log_left = np.log(np.where(f0[left] > 0, f0[left], 1.0))
    log_right = np.log(np.where(f0[right] > 0, f0[right], 1.0))
    interpolated = np.exp(log_left + weight * (log_right - log_left))
    out[voiced] = np.where(both, interpolated, f0[nearest])[voiced]
    return out


def track(wave: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Mono float audio -> f0 in Hz per model frame (0 = unvoiced), length len(wave) // 256."""
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz audio, got {sample_rate}")
    sound = parselmouth.Sound(np.asarray(wave, dtype=np.float64), sampling_frequency=sample_rate)
    f0, times = praat_track(sound, HOP / SAMPLE_RATE)
    return to_model_frames(f0, times, frame_count(len(wave)))
