"""Reconstruction and adversarial losses for the Avaz decoder run.

multi_scale_mel_loss follows descript-audio-codec / BigVGAN-v2 (MIT): log10 mel magnitudes at
seven window lengths (32 ... 2048 samples, hop = window / 4), L1 summed over scales. Short
windows judge onsets and sibilants, long windows judge harmonics.

linear_stft_loss is the multi-resolution STFT loss of Parallel WaveGAN: spectral convergence plus
L1 of log magnitudes on a linear frequency axis, so the 4-12 kHz band (where the 6 kHz ring sits)
keeps full weight instead of being squeezed by the mel scale.

The adversarial losses are least-squares GAN, as in VITS and BigVGAN.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import torch
import torch.nn.functional as F

MEL_WINDOWS = (32, 64, 128, 256, 512, 1024, 2048)
MEL_BANDS = (5, 10, 20, 40, 80, 160, 320)
STFT_RESOLUTIONS = ((512, 128, 512), (1024, 256, 1024), (2048, 512, 2048))
CLAMP = 1.0e-5


@lru_cache(maxsize=None)
def _mel_basis(sample_rate: int, n_fft: int, n_mels: int) -> torch.Tensor:
    import torchaudio

    # Slaney scale and norm, as librosa.filters.mel uses by default (BigVGAN-v2 / DAC).
    return torchaudio.functional.melscale_fbanks(
        n_freqs=n_fft // 2 + 1, f_min=0.0, f_max=sample_rate / 2, n_mels=n_mels,
        sample_rate=sample_rate, norm="slaney", mel_scale="slaney",
    ).T.contiguous()  # [n_mels, n_freqs]


def _magnitude(wave: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
    window = torch.hann_window(win, device=wave.device, dtype=torch.float32)
    spec = torch.stft(wave.float(), n_fft=n_fft, hop_length=hop, win_length=win, window=window,
                      center=True, return_complex=True)
    return spec.abs()


def multi_scale_mel_loss(fake: torch.Tensor, real: torch.Tensor, sample_rate: int,
                         windows: Sequence[int] = MEL_WINDOWS, bands: Sequence[int] = MEL_BANDS) -> torch.Tensor:
    """fake, real: [batch, 1, samples] or [batch, samples]."""
    fake, real = fake.reshape(fake.shape[0], -1), real.reshape(real.shape[0], -1)
    total = fake.new_zeros((), dtype=torch.float32)
    for window, n_mels in zip(windows, bands):
        basis = _mel_basis(sample_rate, window, n_mels).to(fake.device)
        mel_fake = torch.matmul(basis, _magnitude(fake, window, window // 4, window))
        mel_real = torch.matmul(basis, _magnitude(real, window, window // 4, window))
        total = total + F.l1_loss(torch.log10(mel_fake.clamp_min(CLAMP)), torch.log10(mel_real.clamp_min(CLAMP)))
    return total


def linear_stft_loss(fake: torch.Tensor, real: torch.Tensor,
                     resolutions: Sequence[Sequence[int]] = STFT_RESOLUTIONS) -> torch.Tensor:
    """Mean over resolutions of spectral convergence + log-magnitude L1."""
    fake, real = fake.reshape(fake.shape[0], -1), real.reshape(real.shape[0], -1)
    total = fake.new_zeros((), dtype=torch.float32)
    for n_fft, hop, win in resolutions:
        mag_fake = _magnitude(fake, n_fft, hop, win).clamp_min(CLAMP)
        mag_real = _magnitude(real, n_fft, hop, win).clamp_min(CLAMP)
        convergence = torch.linalg.norm(mag_real - mag_fake) / torch.linalg.norm(mag_real).clamp_min(CLAMP)
        log_l1 = F.l1_loss(torch.log(mag_fake), torch.log(mag_real))
        total = total + convergence + log_l1
    return total / len(resolutions)


def discriminator_loss(real_scores: Sequence[torch.Tensor], fake_scores: Sequence[torch.Tensor]) -> torch.Tensor:
    return sum(torch.mean((1.0 - r.float()) ** 2) + torch.mean(f.float() ** 2)
               for r, f in zip(real_scores, fake_scores))


def generator_loss(fake_scores: Sequence[torch.Tensor]) -> torch.Tensor:
    return sum(torch.mean((1.0 - f.float()) ** 2) for f in fake_scores)


def feature_loss(real_maps: Sequence[Sequence[torch.Tensor]], fake_maps: Sequence[Sequence[torch.Tensor]]) -> torch.Tensor:
    return 2.0 * sum(torch.mean(torch.abs(r.float().detach() - f.float()))
                     for group_r, group_f in zip(real_maps, fake_maps) for r, f in zip(group_r, group_f))
