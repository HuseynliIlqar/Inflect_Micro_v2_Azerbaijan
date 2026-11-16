"""VITS2's duration discriminator, used only in training, so the exported model is unchanged.

Kim et al., "VITS2" (2023), section 2.2: an adversarial loss on the duration predictor, so that
it learns the spread of real phoneme durations instead of their average. This is the form most
open ports use ("DurationDiscriminatorV2"): shared convolutions over the detached text hidden
states, then one head that scores (hidden state, log duration) pairs, token by token.

The losses are least-squares GAN losses as in VITS2, but averaged over real tokens only: the
original averages over padding too, where the real and the predicted inputs are both zero.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ChannelLayerNorm(nn.Module):
    """Layer norm over the channel axis of a [batch, channels, time] tensor (VITS' modules.LayerNorm)."""

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))
        self.channels, self.eps = channels, eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x.transpose(1, -1), (self.channels,), self.gamma, self.beta, self.eps).transpose(1, -1)


class DurationDiscriminator(nn.Module):
    """Scores each token's log duration as real (MAS alignment) or predicted, between 0 and 1."""

    def __init__(self, in_channels: int, filter_channels: int = 256, kernel_size: int = 3,
                 p_dropout: float = 0.1) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=pad)
        self.norm_1 = ChannelLayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=pad)
        self.norm_2 = ChannelLayerNorm(filter_channels)
        self.dur_proj = nn.Conv1d(1, filter_channels, 1)
        self.pre_out_conv_1 = nn.Conv1d(2 * filter_channels, filter_channels, kernel_size, padding=pad)
        self.pre_out_norm_1 = ChannelLayerNorm(filter_channels)
        self.pre_out_conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=pad)
        self.pre_out_norm_2 = ChannelLayerNorm(filter_channels)
        self.output = nn.Conv1d(filter_channels, 1, 1)

    def _block(self, conv: nn.Conv1d, norm: ChannelLayerNorm, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.drop(norm(torch.relu(conv(x * mask))))

    def _score(self, shared: torch.Tensor, mask: torch.Tensor, log_duration: torch.Tensor) -> torch.Tensor:
        x = torch.cat([shared, self.dur_proj(log_duration)], dim=1)
        x = self._block(self.pre_out_conv_1, self.pre_out_norm_1, x, mask)
        x = self._block(self.pre_out_conv_2, self.pre_out_norm_2, x, mask)
        return torch.sigmoid(self.output(x * mask)) * mask

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor, log_duration_real: torch.Tensor,
                log_duration_fake: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """hidden [b, c, t] (detached here), mask [b, 1, t], log durations [b, 1, t].

        Returns the scores for the real and the predicted durations, each [b, 1, t] and zero on padding.
        """
        shared = self._block(self.conv_1, self.norm_1, hidden.detach(), mask)
        shared = self._block(self.conv_2, self.norm_2, shared, mask)
        return self._score(shared, mask, log_duration_real), self._score(shared, mask, log_duration_fake)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def duration_discriminator_loss(real: torch.Tensor, fake: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The discriminator wants 1 for real durations and 0 for predicted ones."""
    return _masked_mean((1.0 - real) ** 2, mask) + _masked_mean(fake ** 2, mask)


def duration_generator_loss(fake: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The duration predictor wants its durations scored as real."""
    return _masked_mean((1.0 - fake) ** 2, mask)
