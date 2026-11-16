"""Losses and critics for the frame-rate decoder (frame_decoder.py).

Adapted from Paradee (https://github.com/sahilmahendrakar/paradee, training/scripts/train_vhead.py and
discriminators.py), Apache License 2.0, by Sahil Mahendrakar, at Avaz's 256-sample frames:
- spectral_losses: log-mel L1 plus multi-resolution STFT (spectral convergence + log magnitude), including
  64- and 128-sample windows, shorter than a pitch period, so the loss sees how energy pulses within a period;
- harmonicity_loss: energy on multiples of the pitch against halfway between them, per 1 kHz band, 1-9 kHz,
  on steady voiced frames. Spectrogram losses alone leave the upper harmonics too clean, which sounds buzzy;
- modulation_loss: how the loudness of the upper bands pulses with the pitch (depth and timing, 1x and 2x);
- Discriminators: compact multi-period + multi-resolution spectrogram critics, LS-GAN, feature matching.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.nn.utils import weight_norm

from frame_decoder import FHOP, SR


class SpectralLosses(nn.Module):
    def __init__(self, fine: bool = True):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=1024, hop_length=256, n_mels=80)
        self.sizes = (64, 128, 512, 1024, 2048) if fine else (512, 1024, 2048)
        for n in self.sizes:
            self.register_buffer(f"w{n}", torch.hann_window(n), persistent=False)

    def forward(self, y: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lm = F.l1_loss(torch.log(self.mel(y).clamp_min(1e-5)), torch.log(self.mel(t).clamp_min(1e-5)))
        ls = 0.0
        for n in self.sizes:
            w = getattr(self, f"w{n}")
            Y = torch.stft(y, n, n // 4, window=w, return_complex=True).abs()
            T = torch.stft(t, n, n // 4, window=w, return_complex=True).abs()
            ls = ls + (T - Y).norm() / T.norm().clamp_min(1e-5) + F.l1_loss(torch.log(Y.clamp_min(1e-5)),
                                                                           torch.log(T.clamp_min(1e-5)))
        return lm, ls / len(self.sizes)


def harmonicity_loss(y: torch.Tensor, t: torch.Tensor, f0: torch.Tensor, n: int = 2048, hop: int = 512) -> torch.Tensor:
    """f0 [B, frames] in Hz at 256-sample frames."""
    w = torch.hann_window(n, device=y.device)
    nfr = y.shape[-1] // hop + 1
    idx = (torch.arange(nfr, device=y.device) * hop // FHOP).clamp(max=f0.shape[1] - 1)
    lo, hi = (idx - 2).clamp(min=0), (idx + 2).clamp(max=f0.shape[1] - 1)
    f = f0[:, idx]
    steady = (torch.minimum(torch.minimum(f0[:, lo], f), f0[:, hi]) > 100).float()
    fr = torch.arange(n // 2 + 1, device=y.device) * SR / n
    ratio = fr[None, :, None] / f.clamp_min(60)[:, None]
    frac = (ratio - ratio.round()).abs()
    on, off = (frac < 0.15).float(), ((frac - 0.5).abs() < 0.15).float()
    bands = torch.stack([((fr >= 1000 * k) & (fr < 1000 * (k + 1))).float() for k in range(1, 9)])
    out = []
    for x in (y, t):
        P = torch.stft(x, n, hop, window=w, return_complex=True).abs().pow(2)
        out.append(torch.log((torch.einsum("bft,kf->bkt", P * on, bands) + 1e-8)
                             / (torch.einsum("bft,kf->bkt", P * off, bands) + 1e-8)))
    return ((out[0] - out[1]).abs() * steady[:, None]).sum() / (steady.sum() * 8).clamp_min(1)


MOD_BANDS = [(4, 8), (8, 12), (12, 16), (16, 21), (21, 29)]  # 375 Hz bins: 1.5-3, 3-4.5, 4.5-6, 6-8, 8-11 kHz


def modulation_loss(y: torch.Tensor, t: torch.Tensor, cycles: torch.Tensor, voiced: torch.Tensor,
                    n: int = 64, hop: int = 16, win: int = 75) -> torch.Tensor:
    """cycles, voiced [B, samples]: the sine bank's phase and voicing (FrameDecoder.head.last)."""
    w = torch.hann_window(n, device=y.device)
    k = y.shape[-1] // hop + 1
    ph = 2 * torch.pi * F.pad(cycles, (0, hop))[:, ::hop][:, :k]
    v = F.pad(voiced, (0, hop))[:, ::hop][:, :k]
    carriers = torch.stack([fn(m * ph) for m in (1, 2) for fn in (torch.cos, torch.sin)], 1)
    depth = []
    for x in (y, t):
        X = torch.stft(x, n, hop, window=w, return_complex=True).abs()
        e = torch.stack([X[:, a:b].sum(1) for a, b in MOD_BANDS], 1)
        num = F.avg_pool1d((e[:, :, None] * carriers[:, None]).flatten(1, 2), win, win // 2)
        den = F.avg_pool1d(e, win, win // 2).clamp_min(1e-5)
        depth.append(2 * num.view(e.shape[0], len(MOD_BANDS), 4, -1) / den[:, :, None])
    mask = (F.avg_pool1d(v[:, None], win, win // 2) > 0.99).float()[:, :, None]
    return ((depth[0] - depth[1]).abs() * mask).sum() / (mask.sum() * 20).clamp_min(1)


class PeriodD(nn.Module):
    def __init__(self, period: int, ch=(32, 64, 128, 256)):
        super().__init__()
        self.period = period
        c = 1
        self.convs = nn.ModuleList()
        for o in ch:
            self.convs.append(weight_norm(nn.Conv2d(c, o, (5, 1), (3, 1), padding=(2, 0))))
            c = o
        self.post = weight_norm(nn.Conv2d(c, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        b, t = x.shape
        p = self.period
        if t % p:
            x = F.pad(x, (0, p - t % p), "reflect")
        x = x.view(b, 1, -1, p)
        feats = []
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            feats.append(x)
        x = self.post(x)
        feats.append(x)
        return x.flatten(1), feats


class ResD(nn.Module):
    def __init__(self, n_fft: int, ch=(32, 64, 128, 128)):
        super().__init__()
        self.n_fft, self.hop = n_fft, n_fft // 4
        c = 1
        self.convs = nn.ModuleList()
        for o in ch:
            self.convs.append(weight_norm(nn.Conv2d(c, o, (3, 9), (1, 2), padding=(1, 4))))
            c = o
        self.post = weight_norm(nn.Conv2d(c, 1, (3, 3), 1, padding=(1, 1)))
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def forward(self, x):
        s = torch.stft(x, self.n_fft, self.hop, self.n_fft, window=self.window, return_complex=True).abs()[:, None]
        feats = []
        for c in self.convs:
            s = F.leaky_relu(c(s), 0.1)
            feats.append(s)
        s = self.post(s)
        feats.append(s)
        return s.flatten(1), feats


class Discriminators(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11), n_ffts=(512, 1024, 2048)):
        super().__init__()
        self.ds = nn.ModuleList([PeriodD(p) for p in periods] + [ResD(n) for n in n_ffts])

    def forward(self, x):
        return [d(x) for d in self.ds]


def d_loss(D: Discriminators, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    loss = 0.0
    for (r, _), (f, _) in zip(D(real), D(fake.detach())):
        loss = loss + ((r - 1) ** 2).mean() + (f ** 2).mean()
    return loss


def g_loss(D: Discriminators, real: torch.Tensor, fake: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    adv, fm = 0.0, 0.0
    for (_, rf), (f, ff) in zip(D(real), D(fake)):
        adv = adv + ((f - 1) ** 2).mean()
        fm = fm + sum(F.l1_loss(a, b.detach()) for a, b in zip(ff, rf)) / len(ff)
    return adv, fm
