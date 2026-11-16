"""Pitch per model frame: frame geometry, voicing and timing on synthetic tones. Needs parselmouth."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
parselmouth = pytest.importorskip("parselmouth")

from pitch_track import HOP, SAMPLE_RATE, frame_centres, frame_count, to_model_frames, track  # noqa: E402


def _tone(seconds: float, hz: float) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return 0.3 * sum(np.sin(2 * np.pi * hz * k * t) / k for k in range(1, 6))


def test_frames_match_the_vits_spectrogram() -> None:
    for samples in (24000, 24255, 24256, 36001):
        padded = samples + 2 * (1024 - HOP) // 2
        assert frame_count(samples) == (padded - 1024) // HOP + 1
    assert frame_centres(2)[1] * SAMPLE_RATE == 256 + 128


def test_a_tone_after_silence_is_voiced_at_its_pitch_and_only_there() -> None:
    wave = np.concatenate([np.zeros(SAMPLE_RATE // 2), _tone(1.0, 200.0), np.zeros(SAMPLE_RATE // 4)])
    f0 = track(wave)
    assert len(f0) == len(wave) // HOP
    centres = frame_centres(len(f0))
    inside = (centres > 0.56) & (centres < 1.44)
    assert np.all(np.abs(12 * np.log2(f0[inside] / 200.0)) < 0.3)
    assert np.all(f0[centres < 0.44] == 0) and np.all(f0[centres > 1.56] == 0)


def test_a_pitch_step_lands_on_the_right_frame() -> None:
    wave = np.concatenate([_tone(0.75, 150.0), _tone(0.75, 250.0)])
    f0 = track(wave)
    centres = frame_centres(len(f0))
    voiced = f0 > 0
    high = voiced & (f0 > 200)
    first_high = centres[np.flatnonzero(high)[0]]
    assert abs(first_high - 0.75) <= 2 * HOP / SAMPLE_RATE


def test_moving_onto_model_frames_interpolates_in_log_and_keeps_unvoiced() -> None:
    step = HOP / SAMPLE_RATE
    centres = frame_centres(6)
    f0 = np.array([100.0, 400.0, 0.0, 0.0, 200.0, 200.0])
    later = to_model_frames(f0, centres + step / 4, 6)  # Praat's grid a quarter frame later
    assert later[0] == pytest.approx(100.0)                       # before the first Praat frame: its value
    assert later[1] == pytest.approx(100.0 * 4 ** 0.75, rel=1e-5)  # 3/4 of the way from 100 to 400, in log
    assert later[2] == 0.0 and later[3] == 0.0                    # the nearest Praat frame is unvoiced
    assert later[4] == pytest.approx(200.0) and later[5] == pytest.approx(200.0)
    earlier = to_model_frames(f0, centres - step / 4, 6)
    assert earlier[1] == pytest.approx(400.0)                     # nearest is voiced, its neighbour is not
    assert earlier[2] == 0.0


def test_frames_far_from_the_track_stay_unvoiced() -> None:
    step = HOP / SAMPLE_RATE
    times = np.array([0.5, 0.5 + step])
    out = to_model_frames(np.array([180.0, 180.0]), times, frame_count(SAMPLE_RATE))
    centres = frame_centres(len(out))
    assert np.all(out[np.abs(centres - 0.5 - step / 2) > 2 * step] == 0)
