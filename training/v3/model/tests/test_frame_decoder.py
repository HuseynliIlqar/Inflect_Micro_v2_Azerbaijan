import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frame_decoder import FHOP, SR, FrameDecoder, sine_bank  # noqa: E402
from frame_losses import Discriminators, SpectralLosses, d_loss, g_loss, harmonicity_loss, modulation_loss  # noqa: E402

V2_DECODER_PARAMS = 5_766_000


def test_sine_bank_length_and_silence_when_unvoiced() -> None:
    f0 = torch.zeros(2, 10)
    bank, cycles, voiced = sine_bank(f0)
    assert bank.shape == (2, 10 * FHOP) and cycles.shape == bank.shape
    assert torch.all(bank == 0) and torch.all(voiced == 0)


def test_sine_bank_peaks_at_the_pitch_and_mutes_high_harmonics() -> None:
    f0 = torch.full((1, 40), 200.0)
    bank, _, _ = sine_bank(f0)
    spectrum = torch.fft.rfft(bank[0] * torch.hann_window(bank.shape[1])).abs()  # windowed: no leakage
    freqs = torch.fft.rfftfreq(bank.shape[1], 1 / SR)
    assert abs(float(freqs[spectrum.argmax()]) % 200.0) < 5 or abs(float(freqs[spectrum.argmax()]) % 200.0 - 200) < 5
    assert spectrum[freqs > 11200].max() < 1e-3 * spectrum.max()
    high, _, _ = sine_bank(torch.full((1, 8), 6000.0))  # only the fundamental is under 11 kHz
    s = torch.fft.rfft(high[0] * torch.hann_window(high.shape[1])).abs()
    fr = torch.fft.rfftfreq(high.shape[1], 1 / SR)
    assert abs(float(fr[s.argmax()]) - 6000) < 50 and s[fr > 9000].max() < 1e-2 * s.max()


def test_sine_bank_is_continuous_across_frames_with_a_changing_pitch() -> None:
    f0 = torch.linspace(150, 260, 30)[None]
    _, cycles, _ = sine_bank(f0)
    wave = torch.sin(2 * math.pi * cycles[0].double())
    step = (wave[1:] - wave[:-1]).abs().max()
    assert step < 2 * math.pi * 270 / SR * 1.05


@pytest.fixture(scope="module")
def decoder() -> FrameDecoder:
    torch.manual_seed(0)
    return FrameDecoder()


def test_decoder_output_length_finiteness_and_budget(decoder) -> None:
    z = torch.randn(2, 192, 24)
    f0 = torch.cat([torch.zeros(2, 4), torch.full((2, 20), 180.0)], 1)
    y = decoder(z, f0[:, None])
    assert y.shape == (2, 24 * FHOP) and torch.isfinite(y).all()
    params = sum(p.numel() for p in decoder.parameters())
    assert params < V2_DECODER_PARAMS, params


def test_decoder_is_deterministic_for_given_noise_and_trains_end_to_end(decoder) -> None:
    z = torch.randn(1, 192, 16)
    f0 = torch.full((1, 16), 190.0)
    noise = torch.randn(1, 16 * FHOP)
    assert torch.equal(decoder(z, f0, noise), decoder(z, f0, noise))
    decoder.zero_grad()
    decoder(z, f0, noise).pow(2).mean().backward()
    for name in ("inp.weight", "blocks.0.pw1.weight", "head.out.weight", "head.out_b.weight"):
        grad = dict(decoder.named_parameters())[name].grad
        assert grad is not None and grad.abs().sum() > 0, name


def test_losses_vanish_for_identical_audio_and_are_finite_otherwise(decoder) -> None:
    torch.manual_seed(1)
    t = torch.randn(2, 32 * FHOP) * 0.1
    y = torch.randn(2, 32 * FHOP) * 0.1
    f0 = torch.full((2, 32), 200.0)
    spectral = SpectralLosses()
    lm, ls = spectral(t, t)
    assert float(lm) == pytest.approx(0, abs=1e-6) and float(ls) == pytest.approx(0, abs=1e-6)
    assert float(harmonicity_loss(t, t, f0)) == pytest.approx(0, abs=1e-6)
    _, cycles, voiced = sine_bank(f0)
    assert float(modulation_loss(t, t, cycles, voiced)) == pytest.approx(0, abs=1e-6)
    for value in (*spectral(y, t), harmonicity_loss(y, t, f0), modulation_loss(y, t, cycles, voiced)):
        assert torch.isfinite(value) and float(value) > 0


def test_critics_and_their_losses_run() -> None:
    D = Discriminators()
    real, fake = torch.randn(2, 8 * FHOP), torch.randn(2, 8 * FHOP)
    assert torch.isfinite(d_loss(D, real, fake))
    adv, fm = g_loss(D, real, fake)
    assert torch.isfinite(adv) and torch.isfinite(fm)
