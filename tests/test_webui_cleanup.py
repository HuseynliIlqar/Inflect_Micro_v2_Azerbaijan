"""The single-clip resonance filter."""

from __future__ import annotations

import numpy as np
import pytest

from webui.cleanup import autoclean, clean, find_resonances, spectrum

RATE = 24_000


def tone(hertz: float, seconds: float = 1.0, amplitude: float = 0.3) -> np.ndarray:
    time = np.arange(int(RATE * seconds), dtype=np.float32) / RATE
    return (amplitude * np.sin(2 * np.pi * hertz * time)).astype(np.float32)


def noise(seconds: float = 1.0, amplitude: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (amplitude * rng.standard_normal(int(RATE * seconds))).astype(np.float32)


# -- analysis ----------------------------------------------------------------


def test_the_spectrum_covers_the_audible_range() -> None:
    frequencies, magnitude = spectrum(tone(440.0), RATE)
    assert frequencies[0] == 0.0
    assert frequencies[-1] == pytest.approx(RATE / 2)
    assert frequencies.shape == magnitude.shape


def test_a_clip_shorter_than_the_window_is_refused() -> None:
    with pytest.raises(ValueError):
        spectrum(np.zeros(100, dtype=np.float32), RATE)


def test_a_buzz_on_top_of_noise_is_found() -> None:
    audio = noise() + tone(3120.0, amplitude=0.25)
    found = find_resonances(audio, RATE)
    assert found, "the injected 3120 Hz tone should stand out"
    assert min(abs(hertz - 3120.0) for hertz in found) < 40.0


def test_plain_noise_has_no_resonance() -> None:
    assert find_resonances(noise(), RATE) == ()


def test_a_low_tone_is_left_alone() -> None:
    # Speech formants live below 1500 Hz and must not be notched.
    assert find_resonances(noise() + tone(300.0), RATE) == ()


def test_too_short_to_analyse_returns_nothing_rather_than_raising() -> None:
    assert find_resonances(np.zeros(10, dtype=np.float32), RATE) == ()


def test_at_most_two_notches_are_returned() -> None:
    audio = noise() + tone(2000.0) + tone(3120.0) + tone(5000.0) + tone(7000.0)
    assert len(find_resonances(audio, RATE)) <= 2


# -- filtering ---------------------------------------------------------------


def test_the_input_is_not_modified() -> None:
    audio = noise() + tone(3120.0)
    before = audio.copy()
    clean(audio, RATE, notches=(3120.0,))
    assert np.array_equal(audio, before)


def test_the_result_keeps_shape_and_type() -> None:
    audio = tone(440.0)
    cleaned = clean(audio, RATE)
    assert cleaned.shape == audio.shape
    assert cleaned.dtype == np.float32
    assert np.all(np.abs(cleaned) <= 1.0)


def test_a_notch_removes_the_buzz() -> None:
    audio = noise() + tone(3120.0, amplitude=0.25)
    cleaned = clean(audio, RATE, notches=(3120.0,))
    frequencies, before = spectrum(audio, RATE)
    _, after = spectrum(cleaned, RATE)
    at_buzz = int(np.argmin(np.abs(frequencies - 3120.0)))
    assert after[at_buzz] < before[at_buzz] - 10.0


def test_the_high_pass_removes_rumble() -> None:
    audio = tone(440.0) + tone(25.0, amplitude=0.3)
    cleaned = clean(audio, RATE)
    frequencies, before = spectrum(audio, RATE)
    _, after = spectrum(cleaned, RATE)
    at_rumble = int(np.argmin(np.abs(frequencies - 25.0)))
    assert after[at_rumble] < before[at_rumble] - 6.0


def test_speech_range_energy_survives() -> None:
    audio = tone(440.0)
    frequencies, before = spectrum(audio, RATE)
    _, after = spectrum(clean(audio, RATE), RATE)
    at_tone = int(np.argmin(np.abs(frequencies - 440.0)))
    assert after[at_tone] > before[at_tone] - 3.0


def test_a_quiet_clip_is_brought_up() -> None:
    cleaned = clean(tone(440.0, amplitude=0.01), RATE)
    assert float(np.max(np.abs(cleaned))) == pytest.approx(0.95, abs=0.05)


def test_silence_stays_silent() -> None:
    cleaned = clean(np.zeros(RATE, dtype=np.float32), RATE)
    assert float(np.max(np.abs(cleaned))) == 0.0


def test_an_empty_clip_returns_an_empty_array() -> None:
    cleaned = clean(np.zeros(0, dtype=np.float32), RATE)
    assert cleaned.size == 0 and cleaned.dtype == np.float32


def test_autoclean_reports_what_it_notched() -> None:
    audio = noise() + tone(3120.0, amplitude=0.25)
    cleaned, notches = autoclean(audio, RATE)
    assert cleaned.dtype == np.float32
    assert notches and min(abs(hertz - 3120.0) for hertz in notches) < 40.0


def test_autoclean_on_clean_audio_notches_nothing() -> None:
    _, notches = autoclean(noise(), RATE)
    assert notches == ()
