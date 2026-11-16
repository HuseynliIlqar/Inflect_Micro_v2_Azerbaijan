"""Stage 5: the pitch part ("səs tonunu idarə edən hissə").

v2 has no explicit pitch: the melody hides in the latent z, so questions come out flat. v3 makes
pitch a quantity the model predicts and then obeys:

- PitchPredictor reads the text hidden states expanded to frames (each phoneme's state repeated
  for its duration) and predicts, per frame, the pitch in semitones from the voice's median and
  a voiced/unvoiced logit. It sees the hidden states detached, like v2's duration predictor, so
  it cannot disturb the text encoder. About 0.6M parameters, only convolutions (ONNX-friendly).
- PitchDiscriminator, training only, judges contours given the same frame features. L1 alone
  pulls every contour towards the average and narrows the range; the adversarial term pushes
  the predicted contours to look like real ones (the stage 4 idea, applied to pitch).
- F0PriorShift moves the frame-level prior mean by an amount learnt from the pitch. In training
  the true pitch goes in, so the flow learns latents that agree with it; at inference the
  predicted pitch goes in. Its last layer starts at zero, so v3 starts exactly as v2.

Pitch features everywhere are [semitones / 12 on voiced frames, voiced flag].
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from duration_discriminator import ChannelLayerNorm


def hz_to_semitones(f0_hz: torch.Tensor, ref_hz: float) -> tuple[torch.Tensor, torch.Tensor]:
    """f0 in Hz (0 = unvoiced) -> (semitones from ref_hz, 0 where unvoiced; voiced flag as float)."""
    voiced = (f0_hz > 0).to(f0_hz.dtype)
    semitones = 12.0 * torch.log2(torch.where(f0_hz > 0, f0_hz, torch.full_like(f0_hz, ref_hz)) / ref_hz)
    return semitones * voiced, voiced


def semitones_to_hz(semitones: torch.Tensor, voiced: torch.Tensor, ref_hz: float) -> torch.Tensor:
    return ref_hz * torch.pow(2.0, semitones / 12.0) * voiced


def pitch_features(semitones: torch.Tensor, voiced: torch.Tensor) -> torch.Tensor:
    """[b, 1, t] each -> [b, 2, t]."""
    return torch.cat([semitones * voiced / 12.0, voiced], dim=1)


def expand_to_frames(token_values: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
    """[b, c, t_x] and alignment [b, 1, t_y, t_x] -> [b, c, t_y]."""
    return torch.matmul(attn.squeeze(1), token_values.transpose(1, 2)).transpose(1, 2)


class _ResidualConvStack(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilations: tuple[int, ...], p_dropout: float) -> None:
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv1d(channels, channels, kernel_size, dilation=d,
                                             padding=d * (kernel_size - 1) // 2) for d in dilations)
        self.norms = nn.ModuleList(ChannelLayerNorm(channels) for _ in dilations)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for conv, norm in zip(self.convs, self.norms):
            x = x + self.drop(norm(torch.relu(conv(x * mask))))
        return x * mask


class PitchPredictor(nn.Module):
    def __init__(self, in_channels: int, channels: int = 192, kernel_size: int = 5,
                 dilations: tuple[int, ...] = (1, 2, 4, 8, 1), p_dropout: float = 0.1) -> None:
        super().__init__()
        self.pre = nn.Conv1d(in_channels, channels, 1)
        self.stack = _ResidualConvStack(channels, kernel_size, dilations, p_dropout)
        self.out = nn.Conv1d(channels, 2, 1)

    def forward(self, frames: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """frames [b, c, t] (detached here), mask [b, 1, t] -> (semitones, voicing logit), each [b, 1, t]."""
        x = self.stack(self.pre(frames.detach()) * mask, mask)
        semitones, logit = torch.split(self.out(x) * mask, 1, dim=1)
        return semitones, logit


class PitchDiscriminator(nn.Module):
    """Scores each frame of a contour as real or predicted, given the frame features. Training only."""

    def __init__(self, in_channels: int, channels: int = 128, kernel_size: int = 5,
                 dilations: tuple[int, ...] = (1, 2, 4, 8, 16), p_dropout: float = 0.1) -> None:
        super().__init__()
        self.pre_text = nn.Conv1d(in_channels, channels, 1)
        self.pre_pitch = nn.Conv1d(2, channels, 1)
        self.stack = _ResidualConvStack(channels, kernel_size, dilations, p_dropout)
        self.out = nn.Conv1d(channels, 1, 1)

    def forward(self, frames: torch.Tensor, mask: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        x = (self.pre_text(frames.detach()) + self.pre_pitch(features)) * mask
        return torch.sigmoid(self.out(self.stack(x, mask))) * mask


class F0PriorShift(nn.Module):
    """Pitch features -> a shift of the frame-level prior mean. Starts at exactly zero."""

    def __init__(self, out_channels: int, channels: int = 64, kernel_size: int = 5) -> None:
        super().__init__()
        self.conv = nn.Conv1d(2, channels, kernel_size, padding=kernel_size // 2)
        self.proj = nn.Conv1d(channels, out_channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.relu(self.conv(features * mask))) * mask


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def pitch_l1_loss(predicted: torch.Tensor, target: torch.Tensor, voiced: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean absolute error in semitones over voiced frames."""
    return _masked_mean((predicted - target).abs(), voiced * mask)


def voicing_loss(logit: torch.Tensor, voiced: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _masked_mean(F.binary_cross_entropy_with_logits(logit, voiced, reduction="none"), mask)


def contour_discriminator_loss(real: torch.Tensor, fake: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _masked_mean((1.0 - real) ** 2, mask) + _masked_mean(fake ** 2, mask)


def contour_generator_loss(fake: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _masked_mean((1.0 - fake) ** 2, mask)


def _voiced_spread(semitones: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Standard deviation over each utterance's voiced frames: [b, 1, t] -> [b]."""
    count = weight.sum(dim=(1, 2)).clamp_min(1.0)
    mean = (semitones * weight).sum(dim=(1, 2)) / count
    variance = (((semitones - mean[:, None, None]) ** 2) * weight).sum(dim=(1, 2)) / count
    return torch.sqrt(variance + 1e-6)


def range_loss(predicted: torch.Tensor, target: torch.Tensor, voiced: torch.Tensor, mask: torch.Tensor,
               min_frames: int = 10) -> torch.Tensor:
    """Mean absolute difference, in semitones, between the predicted and the true spread of pitch in each
    utterance. The trial run showed L1 plus the contour discriminator still narrowing the range by about 6%
    during joint training; this term targets the spread itself. Utterances with fewer than min_frames voiced
    frames are left out."""
    weight = voiced * mask
    usable = (weight.sum(dim=(1, 2)) >= min_frames).to(predicted.dtype)
    gap = (_voiced_spread(predicted, weight) - _voiced_spread(target, weight)).abs()
    return (gap * usable).sum() / usable.sum().clamp_min(1.0)
