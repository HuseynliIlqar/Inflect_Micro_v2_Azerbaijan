"""Frame-rate source-filter decoder: VITS latent + pitch -> waveform, with no upsampling layers.

Adapted from Paradee (https://github.com/sahilmahendrakar/paradee, training/scripts/vhead.py, head "v2"),
Apache License 2.0, by Sahil Mahendrakar. Changes for Avaz:
- frames are 256 samples (the VITS hop, 93.75 a second) instead of 300, so the STFT is n_fft 512, hop 128;
- the trunk is a ConvNeXt stack on the VITS latent z and the pitch, instead of Kokoro's AdaIN trunk;
- the sine bank's running phase is accumulated in float64 on the device.

How it works. The trunk reads z and the pitch per frame. The head predicts, for two STFT frames per latent
frame, a smooth filter on a clean sine bank built from the pitch (64 harmonics, phase-locked to the pitch by
construction, muted above 11 kHz and where unvoiced): gain and phase at 80 mel-spaced points and one pulse
shift per frame. Noise comes from three gains: plain white noise, noise that pulses with the pitch, and a
short-window noise path (3 ms steps) for consonant bursts. One inverse STFT gives the waveform. Nothing is
upsampled, so the fixed resonances of a transposed-convolution vocoder (Avaz v2's ~9 kHz ring) cannot form.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 24000
FHOP = 256  # samples per latent frame
F0_REF_HZ = 178.0


def sine_bank(f0: torch.Tensor, harmonics: int = 64, fhop: int = FHOP, sr: int = SR, ceiling: float = 11000.0,
              c0: torch.Tensor | None = None):
    """f0 [B, frames] in Hz (0 = unvoiced) -> (bank [B, samples], phase in cycles [B, samples], voiced [B, samples]).

    The fundamental's phase is the running sum of cycles per frame, interpolated linearly between frame centres
    (sample 256 i + 128); harmonic k has phase k times that. c0 [B]: cycles already run before this crop."""
    b, n = f0.shape
    rate = f0.double() * fhop / sr  # cycles per frame
    start = torch.cumsum(rate, 1)
    if c0 is not None:
        start = start + c0.double()[:, None]
    start = start % 1.0
    advance = F.pad(rate[:, 1:], (0, 1))
    pos = (torch.arange(n * fhop, device=f0.device, dtype=torch.float64) + 0.5) / fhop - 0.5
    index = pos.floor().clamp(0, n - 1).long()
    frac = (pos - index).clamp(0, 1)
    cycles = (start[:, index] + frac * advance[:, index]).float()
    f0s = f0.float().repeat_interleave(fhop, 1)
    k = torch.arange(1, harmonics + 1, device=f0.device, dtype=torch.float32)[None, :, None]
    amp = ((f0s[:, None] * k < ceiling) & (f0s[:, None] > 10)).float()
    bank = (torch.sin(2 * math.pi * k * cycles[:, None]) * amp).sum(1) * 0.1
    return bank, cycles, (f0s > 10).float()


class ConvNeXt(nn.Module):
    def __init__(self, dim: int, mult: int = 3):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, dim * mult)
        self.pw2 = nn.Linear(dim * mult, dim)
        self.gamma = nn.Parameter(torch.full((dim,), 1e-2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(self.dw(x).transpose(1, 2))
        return x + (self.gamma * self.pw2(F.gelu(self.pw1(h)))).transpose(1, 2)


def warp_matrix(bins: int, points: int, sr: int = SR) -> torch.Tensor:
    """[bins, points] linear-interpolation weights from points evenly spaced on a mel-like axis to linear STFT bins."""
    f = torch.linspace(0, sr / 2, bins)
    mel = lambda x: torch.log1p(x / 700)  # noqa: E731
    pos = mel(f) / mel(torch.tensor(sr / 2.0)) * (points - 1)
    i = pos.floor().clamp(0, points - 2).long()
    w = pos - i
    m = torch.zeros(bins, points)
    m[torch.arange(bins), i] = 1 - w
    m[torch.arange(bins), i + 1] += w
    return m


class SourceFilterHead(nn.Module):
    """Paradee's head "v2" at Avaz's frame size: see the module docstring."""

    def __init__(self, in_ch: int, dim: int = 160, blocks: int = 4, n_fft: int = 512, hop: int = 128,
                 harmonics: int = 64, mod: int = 2, points: int = 80, burst: bool = True, burst_points: int = 24,
                 burst_fft: int = 128, burst_hop: int = 64):
        super().__init__()
        self.n_fft, self.hop, self.harmonics, self.mod, self.points = n_fft, hop, harmonics, mod, points
        self.up, self.bins = FHOP // hop, n_fft // 2 + 1
        self.burst, self.burst_points, self.burst_fft, self.burst_hop = burst, burst_points, burst_fft, burst_hop
        self.burst_up = FHOP // burst_hop
        self.inp = nn.Conv1d(in_ch, dim, 7, padding=3)
        self.norm0 = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([ConvNeXt(dim) for _ in range(blocks)])
        self.norm1 = nn.LayerNorm(dim)
        self.channels = 4 + 2 * mod  # harmonic log-gain, phase cos, phase sin, noise log-gain, pulsing-noise gains
        self.out = nn.Linear(dim, self.up * (self.channels * points + 1))
        nn.init.trunc_normal_(self.out.weight, std=0.02)
        with torch.no_grad():
            bias = self.out.bias.view(self.up, -1)
            bias.zero_()
            g = bias[:, : self.channels * points].view(self.up, self.channels, points)
            g[:, 0] = -3.0
            g[:, 1] = 1.0
            g[:, 3] = -6.0
        if burst:
            self.out_b = nn.Linear(dim, self.burst_up * burst_points)
            nn.init.trunc_normal_(self.out_b.weight, std=0.02)
            nn.init.constant_(self.out_b.bias, -7.0)
            self.register_buffer("wb", torch.hann_window(burst_fft), persistent=False)
            self.register_buffer("Mb", warp_matrix(burst_fft // 2 + 1, burst_points), persistent=False)
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.register_buffer("M", warp_matrix(self.bins, points), persistent=False)
        self.last: dict = {}

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stft(x, self.n_fft, self.hop, window=self.window, return_complex=True)

    def forward(self, x: torch.Tensor, f0: torch.Tensor, noise: torch.Tensor | None = None,
                c0: torch.Tensor | None = None) -> torch.Tensor:
        b, _, n = x.shape
        frames = n * self.up
        samples = n * FHOP
        h = self.norm0(self.inp(x).transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            h = block(h)
        hn = self.norm1(h.transpose(1, 2))
        o = self.out(hn).view(b, frames, self.channels * self.points + 1)
        o = torch.cat([o, o[:, -1:]], 1)  # a centred STFT has one more frame
        g = (o[..., :-1].reshape(b, frames + 1, self.channels, self.points) @ self.M.T).permute(0, 2, 3, 1)
        gh, gn = torch.exp(g[:, 0].clamp(max=6)), torch.exp(g[:, 3].clamp(max=6))
        u = torch.complex(g[:, 1], g[:, 2])
        u = u / u.abs().clamp_min(1e-4)
        with torch.no_grad():
            e, cycles, voiced = sine_bank(f0, self.harmonics, c0=c0)
            E = self.stft(e)
            nz = torch.randn_like(e) if noise is None else noise
            Nz = self.stft(nz)
            Nm = [self.stft(nz * voiced * fn(2 * math.pi * m * cycles))
                  for m in range(1, self.mod + 1) for fn in (torch.cos, torch.sin)]
            f0f = torch.cat([f0.repeat_interleave(self.up, 1), f0[:, -1:]], 1).clamp_min(60)
            ratio = torch.linspace(0, SR / 2, self.bins, device=x.device)[None, :, None] / f0f[:, None]
        shift = 0.5 * torch.tanh(o[..., -1])[:, None]  # pulse shift, in cycles
        Hh = gh * u * torch.polar(torch.ones_like(ratio), 2 * math.pi * ratio * shift)
        self.last = dict(gh=gh, gn=gn, cycles=cycles, voiced=voiced)
        for j in range(2 * self.mod):
            Nz = Nz + g[:, 4 + j] * Nm[j]
        y = torch.istft(E * Hh + Nz * gn, self.n_fft, self.hop, window=self.window, length=samples)
        if self.burst:
            gb = torch.exp((self.out_b(hn).view(b, n * self.burst_up, self.burst_points) @ self.Mb.T).clamp(max=6))
            gb = gb.transpose(1, 2)
            gb = torch.cat([gb, gb[..., -1:]], -1)
            with torch.no_grad():
                Nb = torch.stft(nz.flip(-1), self.burst_fft, self.burst_hop, window=self.wb, return_complex=True)
            y = y + torch.istft(Nb * gb, self.burst_fft, self.burst_hop, window=self.wb, length=samples)
        return y


class FrameDecoder(nn.Module):
    """ConvNeXt trunk on (z, pitch) at the latent frame rate + the source-filter head. Output [B, frames * 256]."""

    def __init__(self, in_channels: int = 192, dim: int = 256, blocks: int = 8, head_dim: int = 160,
                 head_blocks: int = 4, f0_ref_hz: float = F0_REF_HZ, **head_kw):
        super().__init__()
        self.f0_ref_hz = f0_ref_hz
        self.inp = nn.Conv1d(in_channels, dim, 7, padding=3)
        self.pitch_in = nn.Conv1d(2, dim, 7, padding=3)
        self.norm0 = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([ConvNeXt(dim) for _ in range(blocks)])
        self.head = SourceFilterHead(dim, dim=head_dim, blocks=head_blocks, **head_kw)

    def forward(self, z: torch.Tensor, f0: torch.Tensor, noise: torch.Tensor | None = None,
                c0: torch.Tensor | None = None) -> torch.Tensor:
        """z [B, C, T]; f0 [B, T] or [B, 1, T] in Hz, 0 where unvoiced."""
        if f0.dim() == 3:
            f0 = f0[:, 0]
        f0 = f0.float()
        voiced = (f0 > 10).float()
        octaves = torch.log2(f0.clamp_min(1.0) / self.f0_ref_hz) * voiced
        h = self.inp(z) + self.pitch_in(torch.stack([octaves, voiced], 1))
        h = self.norm0(h.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            h = block(h)
        return self.head(h, f0 * voiced, noise, c0)
