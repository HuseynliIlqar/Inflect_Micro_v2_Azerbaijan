"""Stage 6: a harmonic source for v2's decoder, in the neural source-filter (NSF) style.

Wang, Takaki and Yamagishi, "Neural source-filter waveform models" (2019); the HiFi-GAN form is
the "NSF-HiFiGAN" used by DiffSinger and so-vits-svc. From the pitch per frame, the source builds
sine harmonics at the sample rate, mixes them with one learnt linear layer and adds noise where
the frame is unvoiced. Each upsampling stage of the decoder then adds the source, downsampled to
its own rate by a strided convolution.

Why: v2's decoder has to invent the harmonics from the latent alone, and it lags the teacher in
harmonic clarity between 2 and 4 kHz (buzz score 8.8 against 10.6). Harmonics up to 6 kHz are
given here; above that the decoder keeps generating from the latent.

Phase is computed per frame (an exclusive cumulative sum of each frame's fractional cycles) and
then within the frame, so float32 stays exact to about 1e-4 cycles over a long sentence. It uses
only CumSum, Floor, Sin and reshapes, so it exports to ONNX. The convolutions that inject the
source start at zero, so the decoder starts exactly as v2's.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def _frac(x: torch.Tensor) -> torch.Tensor:
    return x - torch.floor(x)


class HarmonicSource(nn.Module):
    def __init__(self, sample_rate: int = 24000, hop: int = 256, harmonics: int = 48, cutoff_hz: float = 6000.0,
                 sine_amp: float = 0.1, noise_std: float = 0.003) -> None:
        super().__init__()
        self.sample_rate, self.hop, self.cutoff_hz = sample_rate, hop, cutoff_hz
        self.sine_amp, self.noise_std = sine_amp, noise_std
        self.register_buffer("orders", torch.arange(1, harmonics + 1, dtype=torch.float32).view(1, harmonics, 1),
                             persistent=False)
        self.register_buffer("steps", (torch.arange(hop, dtype=torch.float32) + 1).view(1, 1, 1, hop) / sample_rate,
                             persistent=False)
        self.merge = nn.Linear(harmonics, 1)

    def cycles(self, f0: torch.Tensor) -> torch.Tensor:
        """f0 [b, 1, t] in Hz -> the fundamental's phase in cycles, in [0, 1), [b, 1, t * hop]."""
        per_frame = _frac(f0 * self.hop / self.sample_rate)
        start = _frac(torch.cumsum(per_frame, dim=2) - per_frame)
        within = start.unsqueeze(-1) + f0.unsqueeze(-1) * self.steps
        return _frac(within).flatten(2)

    def harmonics(self, f0: torch.Tensor, noise: bool = True) -> torch.Tensor:
        """f0 [b, 1, t] -> [b, harmonics, t * hop]: sines on voiced samples, noise elsewhere."""
        f0_up = f0.repeat_interleave(self.hop, dim=2)
        voiced = (f0_up > 0).to(f0.dtype)
        phase = _frac(self.cycles(f0) * self.orders)
        if self.training:
            offsets = torch.rand(f0.size(0), self.orders.size(1), 1, device=f0.device, dtype=f0.dtype)
            offsets[:, 0] = 0.0  # the fundamental keeps phase 0, as in NSF
            phase = phase + offsets
        below_cutoff = (f0_up * self.orders < self.cutoff_hz).to(f0.dtype)
        sines = self.sine_amp * torch.sin(2 * math.pi * phase) * below_cutoff * voiced
        if noise:
            amplitude = voiced * self.noise_std + (1 - voiced) * self.sine_amp / 3
            sines = sines + amplitude * torch.randn_like(sines)
        return sines

    def forward(self, f0: torch.Tensor, noise: bool = True) -> torch.Tensor:
        """f0 [b, 1, t] in Hz (0 = unvoiced) -> source [b, 1, t * hop]."""
        mixed = self.merge(self.harmonics(f0, noise).transpose(1, 2))
        return torch.tanh(mixed).transpose(1, 2)


def source_injections(upsample_rates: list[int], upsample_initial_channel: int) -> nn.ModuleList:
    """One zero-initialised convolution per upsampling stage, from the sample rate down to the stage's."""
    convs = nn.ModuleList()
    for i in range(len(upsample_rates)):
        channels = upsample_initial_channel // (2 ** (i + 1))
        stride = math.prod(upsample_rates[i + 1:])
        conv = (nn.Conv1d(1, channels, 2 * stride, stride=stride, padding=stride // 2) if stride > 1
                else nn.Conv1d(1, channels, 1))
        nn.init.zeros_(conv.weight)
        nn.init.zeros_(conv.bias)
        convs.append(conv)
    return convs
