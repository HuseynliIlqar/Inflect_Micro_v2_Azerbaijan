"""The duration discriminator: shapes, padding, gradient paths and losses. CPU only, no model file."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from duration_discriminator import (DurationDiscriminator, duration_discriminator_loss,  # noqa: E402
                                    duration_generator_loss)

CHANNELS = 16


def _batch(lengths: list[int]):
    torch.manual_seed(0)
    t = max(lengths)
    mask = torch.zeros(len(lengths), 1, t)
    for i, n in enumerate(lengths):
        mask[i, :, :n] = 1.0
    hidden = torch.randn(len(lengths), CHANNELS, t) * mask
    real = torch.randn(len(lengths), 1, t) * mask
    fake = torch.randn(len(lengths), 1, t) * mask
    return hidden, mask, real, fake


def test_scores_are_probabilities_and_zero_on_padding() -> None:
    disc = DurationDiscriminator(CHANNELS, filter_channels=32).eval()
    hidden, mask, real, fake = _batch([7, 4])
    score_real, score_fake = disc(hidden, mask, real, fake)
    assert score_real.shape == score_fake.shape == (2, 1, 7)
    assert torch.all((score_real >= 0) & (score_real <= 1))
    assert torch.all(score_fake[1, :, 4:] == 0)


def test_padding_does_not_change_the_scores() -> None:
    disc = DurationDiscriminator(CHANNELS, filter_channels=32).eval()
    hidden, mask, real, fake = _batch([7, 4])
    alone = disc(hidden[1:, :, :4], mask[1:, :, :4], real[1:, :, :4], fake[1:, :, :4])
    batched = disc(hidden, mask, real, fake)
    for a, b in zip(alone, batched):
        assert torch.allclose(a[0], b[1, :, :4], atol=1e-6)


def test_only_the_predicted_durations_carry_gradient_to_the_generator() -> None:
    disc = DurationDiscriminator(CHANNELS, filter_channels=32)
    hidden, mask, real, fake = _batch([5, 5])
    hidden.requires_grad_(True)
    fake.requires_grad_(True)
    _, score_fake = disc(hidden, mask, real, fake)
    duration_generator_loss(score_fake, mask).backward()
    assert hidden.grad is None  # the text encoder is not trained by this loss
    assert fake.grad is not None and fake.grad.abs().sum() > 0


def test_losses_are_masked_least_squares() -> None:
    mask = torch.tensor([[[1.0, 1.0, 0.0]]])
    perfect_real = torch.tensor([[[1.0, 1.0, 0.0]]])
    perfect_fake = torch.zeros(1, 1, 3)
    assert duration_discriminator_loss(perfect_real, perfect_fake, mask).item() == 0.0
    assert duration_generator_loss(perfect_fake, mask).item() == 1.0
    half = torch.tensor([[[0.5, 0.5, 0.9]]])  # the padded 0.9 must not count
    assert abs(duration_generator_loss(half, mask).item() - 0.25) < 1e-7


def test_a_few_steps_teach_it_to_separate_real_from_predicted() -> None:
    torch.manual_seed(1)
    disc = DurationDiscriminator(CHANNELS, filter_channels=32)
    optimizer = torch.optim.Adam(disc.parameters(), lr=1e-2)
    hidden, mask, _, _ = _batch([6, 6, 6, 6])
    real = (torch.randn(4, 1, 6) * 0.8 + 1.0) * mask  # spread durations
    fake = torch.full((4, 1, 6), 1.0) * mask           # an over-smoothed predictor
    for _ in range(60):
        score_real, score_fake = disc(hidden, mask, real, fake)
        loss = duration_discriminator_loss(score_real, score_fake, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.2
