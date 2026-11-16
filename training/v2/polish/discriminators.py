"""Discriminators for the Avaz decoder run: multi-period (MPD), multi-resolution (MRD) and
multi-scale sub-band CQT (CQTD).

Adapted from NVIDIA BigVGAN (discriminators.py, MIT License, Copyright (c) 2024 NVIDIA
CORPORATION), which in turn adapts HiFi-GAN, UnivNet, Vocos and descript-audio-codec (all
MIT). Module and parameter names follow BigVGAN so its released v2 24 kHz discriminator
weights (MPD and CQTD) load directly. See THIRD_PARTY_NOTICES.md next to this file.

Every family returns (real_scores, fake_scores, real_maps, fake_maps) as lists, one entry per
sub-discriminator, and runs in float32 even inside an autocast region.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm

LRELU_SLOPE = 0.1
LOG_FLOOR = 1.0e-4  # MRD magnitude floor before the log (about -80 dB below full scale)


def _padding(kernel: int, dilation: int = 1) -> int:
    return (kernel * dilation - dilation) // 2


def _pad_2d(kernel: tuple[int, int], dilation: tuple[int, int] = (1, 1)) -> tuple[int, int]:
    return ((kernel[0] - 1) * dilation[0]) // 2, ((kernel[1] - 1) * dilation[1]) // 2


class DiscriminatorP(nn.Module):
    """One period of the MPD: the waveform folded into [time / period, period]."""

    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3) -> None:
        super().__init__()
        self.period = period
        channels = (1, 32, 128, 512, 1024)
        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(cin, cout, (kernel_size, 1), (stride, 1), padding=(_padding(5, 1), 0)))
                for cin, cout in zip(channels[:-1], channels[1:])
            ]
            + [weight_norm(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0)))]
        )
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []
        b, c, t = x.shape
        if t % self.period:
            pad = self.period - t % self.period
            x = F.pad(x, (0, pad), "reflect")
            t += pad
        x = x.view(b, c, t // self.period, self.period)
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class DiscriminatorR(nn.Module):
    """One resolution of the MRD: convolutions over a linear magnitude spectrogram."""

    def __init__(self, resolution: Sequence[int]) -> None:
        super().__init__()
        self.resolution = tuple(resolution)
        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
                weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
            ]
        )
        self.conv_post = weight_norm(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        n_fft, hop, win = self.resolution
        x = F.pad(x, ((n_fft - hop) // 2, (n_fft - hop) // 2), mode="reflect").squeeze(1)
        window = torch.hann_window(win, device=x.device, dtype=x.dtype)
        spec = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win, window=window,
                          center=False, return_complex=True)
        # Log magnitude, unlike BigVGAN's linear one: our generated audio is already close to the target, and on a
        # linear scale a faint 6 kHz ring is invisible next to the low-frequency energy, so a fresh MRD collapsed to
        # scoring everything 0.5. On a log scale the ring stands out as a line.
        return torch.log(spec.abs().clamp_min(LOG_FLOOR))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []
        x = self.spectrogram(x).unsqueeze(1)
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class DiscriminatorCQT(nn.Module):
    """One scale of the CQTD: a constant-Q transform split into octaves (harmonic structure)."""

    def __init__(self, sample_rate: int, hop_length: int, n_octaves: int, bins_per_octave: int,
                 filters: int = 128, max_filters: int = 1024, dilations: Sequence[int] = (1, 2, 4)) -> None:
        super().__init__()
        from nnAudio import features  # imported here so the other families work without nnAudio
        import torchaudio

        self.n_octaves = n_octaves
        self.bins_per_octave = bins_per_octave
        kernel, stride = (3, 9), (1, 2)
        self.cqt_transform = features.cqt.CQT2010v2(
            sr=sample_rate * 2, hop_length=hop_length, n_bins=bins_per_octave * n_octaves,
            bins_per_octave=bins_per_octave, output_format="Complex", pad_mode="constant", verbose=False,
        )
        self.conv_pres = nn.ModuleList(
            [nn.Conv2d(2, 2, kernel_size=kernel, padding=_pad_2d(kernel)) for _ in range(n_octaves)]
        )
        convs = [nn.Conv2d(2, filters, kernel_size=kernel, padding=_pad_2d(kernel))]
        in_chs = min(filters, max_filters)
        for dilation in dilations:
            out_chs = min(filters, max_filters)
            convs.append(weight_norm(nn.Conv2d(in_chs, out_chs, kernel_size=kernel, stride=stride,
                                               dilation=(dilation, 1), padding=_pad_2d(kernel, (dilation, 1)))))
            in_chs = out_chs
        out_chs = min(filters, max_filters)
        convs.append(weight_norm(nn.Conv2d(in_chs, out_chs, kernel_size=(3, 3), padding=_pad_2d((3, 3)))))
        self.convs = nn.ModuleList(convs)
        self.conv_post = weight_norm(nn.Conv2d(out_chs, 1, kernel_size=(3, 3), padding=_pad_2d((3, 3))))
        self.activation = nn.LeakyReLU(negative_slope=LRELU_SLOPE)
        self.resample = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sample_rate * 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []
        z = self.cqt_transform(self.resample(x))
        z = torch.cat([z[:, :, :, 0].unsqueeze(1), z[:, :, :, 1].unsqueeze(1)], dim=1).permute(0, 1, 3, 2)
        width = self.bins_per_octave
        z = torch.cat([pre(z[:, :, :, i * width:(i + 1) * width]) for i, pre in enumerate(self.conv_pres)], dim=-1)
        for conv in self.convs:
            z = self.activation(conv(z))
            fmap.append(z)
        return self.conv_post(z), fmap


class _Family(nn.Module):
    """Runs every sub-discriminator on real and generated audio, in float32."""

    def forward(self, real: torch.Tensor, fake: torch.Tensor):
        real_scores, fake_scores, real_maps, fake_maps = [], [], [], []
        with torch.autocast(device_type=real.device.type, enabled=False):
            real, fake = real.float(), fake.float()
            for disc in self.discriminators:
                score_r, map_r = disc(real)
                score_f, map_f = disc(fake)
                real_scores.append(score_r)
                fake_scores.append(score_f)
                real_maps.append(map_r)
                fake_maps.append(map_f)
        return real_scores, fake_scores, real_maps, fake_maps


class MultiPeriodDiscriminator(_Family):
    def __init__(self, periods: Sequence[int] = (2, 3, 5, 7, 11)) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(p) for p in periods])


class MultiResolutionDiscriminator(_Family):
    def __init__(self, resolutions: Sequence[Sequence[int]] = ((1024, 120, 600), (2048, 240, 1200), (512, 50, 240))) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorR(r) for r in resolutions])


class MultiScaleSubbandCQTDiscriminator(_Family):
    def __init__(self, sample_rate: int = 24000, hop_lengths: Sequence[int] = (512, 256, 256),
                 n_octaves: Sequence[int] = (9, 9, 9), bins_per_octaves: Sequence[int] = (24, 36, 48)) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorCQT(sample_rate, hop, octaves, bins)
             for hop, octaves, bins in zip(hop_lengths, n_octaves, bins_per_octaves)]
        )


@dataclass(frozen=True)
class FamilyOutputs:
    real_scores: list[torch.Tensor]
    fake_scores: list[torch.Tensor]
    real_maps: list[list[torch.Tensor]]
    fake_maps: list[list[torch.Tensor]]


@dataclass(frozen=True)
class DiscriminatorSpec:
    families: tuple[str, ...] = ("mpd", "mrd", "cqtd")
    sample_rate: int = 24000
    mrd_resolutions: tuple[tuple[int, int, int], ...] = field(
        default=((1024, 120, 600), (2048, 240, 1200), (512, 50, 240)))


class AvazDiscriminator(nn.Module):
    """The three families side by side; forward returns {family: FamilyOutputs}."""

    def __init__(self, spec: DiscriminatorSpec = DiscriminatorSpec()) -> None:
        super().__init__()
        builders = {
            "mpd": lambda: MultiPeriodDiscriminator(),
            "mrd": lambda: MultiResolutionDiscriminator(spec.mrd_resolutions),
            "cqtd": lambda: MultiScaleSubbandCQTDiscriminator(spec.sample_rate),
        }
        unknown = set(spec.families) - builders.keys()
        if unknown:
            raise ValueError(f"Unknown discriminator families: {sorted(unknown)}")
        self.families = nn.ModuleDict({name: builders[name]() for name in spec.families})

    def forward(self, real: torch.Tensor, fake: torch.Tensor) -> dict[str, FamilyOutputs]:
        return {name: FamilyOutputs(*family(real, fake)) for name, family in self.families.items()}


def load_bigvgan_weights(model: AvazDiscriminator, payload: dict) -> dict[str, int]:
    """Copy BigVGAN-v2 discriminator weights: payload["mpd"] into MPD, payload["mrd"] (which
    holds the CQTD when the release uses CQTD instead of MRD) into CQTD. Learnable tensors must
    all match in name and shape; buffers such as the CQT kernels are rebuilt locally."""
    copied = {}
    for family, key in (("mpd", "mpd"), ("cqtd", "mrd")):
        if family not in model.families:
            continue
        target = model.families[family]
        learnable = {name for name, _ in target.named_parameters()}
        source = {name: tensor for name, tensor in payload[key].items() if name in learnable}
        missing = sorted(learnable - source.keys())
        if missing:
            raise RuntimeError(f"BigVGAN {key} weights lack {len(missing)} tensors, e.g. {missing[:3]}")
        state = target.state_dict()
        for name, tensor in source.items():
            if state[name].shape != tensor.shape:
                raise RuntimeError(f"{family}.{name}: shape {tuple(tensor.shape)} != {tuple(state[name].shape)}")
            state[name] = tensor
        target.load_state_dict(state)
        copied[family] = len(source)
    return copied
