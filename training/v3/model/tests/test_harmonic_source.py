"""The harmonic source: pitch, phase continuity, cutoff, voicing and the zero start. CPU only."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmonic_source import HarmonicSource, source_injections  # noqa: E402

SR, HOP = 24000, 256


def _peak_hz(signal: torch.Tensor) -> float:
    spectrum = torch.fft.rfft(signal * torch.hann_window(len(signal)))
    return float(torch.argmax(spectrum.abs())) * SR / len(signal)


def test_the_fundamental_has_the_requested_pitch() -> None:
    source = HarmonicSource().eval()
    f0 = torch.full((1, 1, 94), 200.0)  # about 1 s
    fundamental = source.harmonics(f0, noise=False)[0, 0]
    assert len(fundamental) == 94 * HOP
    assert abs(_peak_hz(fundamental) - 200.0) < 2.0


def test_phase_is_continuous_across_frames_when_pitch_moves() -> None:
    source = HarmonicSource().eval()
    f0 = torch.linspace(120.0, 320.0, 200).view(1, 1, 200)
    cycles = source.cycles(f0)[0, 0].double()
    step = torch.remainder(cycles[1:] - cycles[:-1], 1.0)
    expected = f0.repeat_interleave(HOP, dim=2)[0, 0, 1:].double() / SR
    assert torch.max(torch.abs(step - expected)) < 1e-4


def test_harmonics_stop_at_the_cutoff() -> None:
    source = HarmonicSource().eval()
    sines = source.harmonics(torch.full((1, 1, 40), 300.0), noise=False)[0]
    energy = sines.pow(2).mean(dim=1)
    assert torch.all(energy[:19] > 1e-4)   # 300 x 19 = 5,700 Hz
    assert torch.all(energy[20:] == 0)     # 300 x 20 = 6,000 Hz and up


def test_unvoiced_frames_carry_only_noise() -> None:
    source = HarmonicSource().eval()
    f0 = torch.zeros(1, 1, 10)
    f0[:, :, 5:] = 150.0
    quiet = source.harmonics(f0, noise=False)[0]
    assert torch.all(quiet[:, : 5 * HOP] == 0)
    torch.manual_seed(0)
    noisy = source.harmonics(f0, noise=True)[0]
    assert noisy[:, : 5 * HOP].std() > 0.02  # sine_amp / 3


def test_source_has_one_channel_per_sample_and_is_bounded() -> None:
    source = HarmonicSource()
    out = source(torch.full((2, 1, 12), 180.0))
    assert out.shape == (2, 1, 12 * HOP)
    assert torch.all(out.abs() <= 1)


def test_injections_reach_each_stage_rate_and_start_at_zero() -> None:
    rates = [8, 8, 2, 2]
    convs = source_injections(rates, 320)
    signal = torch.randn(1, 1, 20 * HOP)
    lengths, channels = [], []
    for conv in convs:
        out = conv(signal)
        lengths.append(out.size(2))
        channels.append(out.size(1))
        assert torch.all(out == 0)
    assert lengths == [20 * 8, 20 * 64, 20 * 128, 20 * 256]
    assert channels == [160, 80, 40, 20]
