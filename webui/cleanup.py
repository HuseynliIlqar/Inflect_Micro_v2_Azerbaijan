"""Remove the vocoder's resonant buzz from a single clip, in memory.

`tools/audio_postprocess.py` does the same job better: it averages the spectrum
over a whole folder, so a peak has to survive many sentences before it counts as
a resonance, and it filters through ffmpeg. That is the right tool for a batch.

The interface needs something narrower -- one clip, already in memory, no ffmpeg
on the machine -- so this module works on the array with scipy, which is already
a dependency. Two consequences worth knowing:

- a single clip is a noisier estimate, so the prominence threshold here is
  stricter and at most two notches are applied;
- the level is matched by peak normalisation, not by the LUFS measurement
  ffmpeg's `loudnorm` performs.

    >>> import numpy as np
    >>> rate = 24_000
    >>> t = np.arange(rate, dtype=np.float32) / rate
    >>> audio = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    >>> cleaned, found = autoclean(audio, rate)
    >>> cleaned.dtype == np.float32 and cleaned.shape == audio.shape
    True
"""

from __future__ import annotations

import numpy as np
from scipy import signal

__all__ = ["autoclean", "clean", "find_resonances", "spectrum"]

WINDOW = 4096
HOP = 2048
# Stricter than the 4.0 dB used across a folder: one clip is a noisier estimate
# and a false notch is more audible than a missed one.
PROMINENCE_DB = 6.0
# Speech formants live below this; the buzz sits above it.
MIN_HZ = 1500.0
MAX_NOTCHES = 2
NOTCH_Q = 12.0
HIGHPASS_HZ = 70.0
LOWPASS_HZ = 11_000.0
PEAK = 0.95


def spectrum(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean magnitude spectrum in dB, averaged over the clip's frames."""
    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    if mono.size < WINDOW:
        raise ValueError("not enough audio to analyse")
    window = np.hanning(WINDOW).astype(np.float32)
    frames = [
        np.abs(np.fft.rfft(mono[start : start + WINDOW] * window))
        for start in range(0, mono.size - WINDOW, HOP)
    ]
    if not frames:
        raise ValueError("not enough audio to analyse")
    magnitude = 20.0 * np.log10(np.mean(frames, axis=0) + 1e-12)
    return np.fft.rfftfreq(WINDOW, 1.0 / sample_rate), magnitude


def _smooth(values: np.ndarray, width: int = 41) -> np.ndarray:
    """Moving average: the baseline a real peak has to beat."""
    kernel = np.ones(width, dtype=np.float64) / width
    padded = np.pad(values, width // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: values.size]


def find_resonances(
    audio: np.ndarray, sample_rate: int, limit: int = MAX_NOTCHES
) -> tuple[float, ...]:
    """The strongest narrow peaks above `MIN_HZ`, loudest first, in Hz."""
    try:
        frequencies, magnitude = spectrum(audio, sample_rate)
    except ValueError:
        return ()
    excess = magnitude - _smooth(magnitude)
    ceiling = sample_rate / 2.0 * 0.95
    peaks = [
        (float(frequencies[i]), float(excess[i]))
        for i in range(1, excess.size - 1)
        if MIN_HZ <= frequencies[i] <= ceiling
        and excess[i] >= PROMINENCE_DB
        and excess[i] > excess[i - 1]
        and excess[i] >= excess[i + 1]
    ]
    peaks.sort(key=lambda peak: peak[1], reverse=True)
    return tuple(hertz for hertz, _ in peaks[:limit])


def clean(
    audio: np.ndarray,
    sample_rate: int,
    *,
    notches: tuple[float, ...] = (),
    highpass: float = HIGHPASS_HZ,
    lowpass: float = LOWPASS_HZ,
    peak: float = PEAK,
) -> np.ndarray:
    """Return a filtered copy: high pass, notches, low pass, peak normalise.

    The input is never modified. Filtering is zero phase (`filtfilt`), so the
    timing of the speech is untouched.
    """
    filtered = np.asarray(audio, dtype=np.float64).reshape(-1).copy()
    if filtered.size == 0:
        return np.zeros(0, dtype=np.float32)
    nyquist = sample_rate / 2.0
    # `filtfilt` needs more samples than three times the filter order.
    usable = filtered.size > 3 * 8

    if usable and 0 < highpass < nyquist:
        sos = signal.butter(2, highpass / nyquist, btype="highpass", output="sos")
        filtered = signal.sosfiltfilt(sos, filtered)
    for hertz in notches:
        if usable and MIN_HZ <= hertz < nyquist:
            numerator, denominator = signal.iirnotch(hertz / nyquist, NOTCH_Q)
            filtered = signal.filtfilt(numerator, denominator, filtered)
    if usable and 0 < lowpass < nyquist:
        sos = signal.butter(4, lowpass / nyquist, btype="lowpass", output="sos")
        filtered = signal.sosfiltfilt(sos, filtered)

    loudest = float(np.max(np.abs(filtered))) if filtered.size else 0.0
    if loudest > 1e-6:
        filtered = filtered * (peak / loudest)
    return np.clip(filtered, -1.0, 1.0).astype(np.float32)


def autoclean(
    audio: np.ndarray, sample_rate: int
) -> tuple[np.ndarray, tuple[float, ...]]:
    """Find this clip's resonances and filter them out.

    Returns the cleaned audio and the frequencies that were notched, so the
    interface can say what it did instead of claiming magic.
    """
    notches = find_resonances(audio, sample_rate)
    return clean(audio, sample_rate, notches=notches), notches
