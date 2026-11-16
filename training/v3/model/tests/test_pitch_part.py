"""The pitch part on its own: unit conversions, sizes, gradient paths and losses. CPU only."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pitch_part import (F0PriorShift, PitchDiscriminator, PitchPredictor, contour_discriminator_loss,  # noqa: E402
                        contour_generator_loss, hz_to_semitones, pitch_l1_loss, semitones_to_hz, voicing_loss)

HIDDEN = 96  # v2's hidden size


def test_semitone_round_trip_and_unvoiced_zeros() -> None:
    f0 = torch.tensor([[0.0, 100.0, 200.0, 400.0]])
    st, voiced = hz_to_semitones(f0, 200.0)
    assert torch.allclose(st, torch.tensor([[0.0, -12.0, 0.0, 12.0]]))
    assert torch.equal(voiced, torch.tensor([[0.0, 1.0, 1.0, 1.0]]))
    assert torch.allclose(semitones_to_hz(st, voiced, 200.0), f0)


def test_sizes_fit_the_budget() -> None:
    count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert 300_000 <= count(PitchPredictor(HIDDEN)) <= 1_000_000
    assert count(F0PriorShift(192)) < 20_000


def test_predictor_does_not_train_the_text_side() -> None:
    frames = torch.randn(2, HIDDEN, 30, requires_grad=True)
    mask = torch.ones(2, 1, 30)
    st, logit = PitchPredictor(HIDDEN)(frames, mask)
    (st.sum() + logit.sum()).backward()
    assert frames.grad is None


def test_padding_does_not_change_predictions() -> None:
    predictor = PitchPredictor(HIDDEN).eval()
    torch.manual_seed(0)
    frames = torch.randn(1, HIDDEN, 40)
    mask = torch.ones(1, 1, 40)
    alone = predictor(frames[:, :, :25], mask[:, :, :25])
    padded_mask = mask.clone()
    padded_mask[:, :, 25:] = 0
    batched = predictor(frames * padded_mask, padded_mask)
    for a, b in zip(alone, batched):
        assert torch.allclose(a, b[:, :, :25], atol=1e-5)
        assert torch.all(b[:, :, 25:] == 0)


def test_prior_shift_starts_at_zero() -> None:
    shift = F0PriorShift(192)
    features = torch.randn(2, 2, 50)
    assert torch.all(shift(features, torch.ones(2, 1, 50)) == 0)


def test_losses_count_only_the_right_frames() -> None:
    mask = torch.tensor([[[1.0, 1.0, 1.0, 0.0]]])
    voiced = torch.tensor([[[1.0, 0.0, 1.0, 1.0]]])
    predicted = torch.tensor([[[1.0, 50.0, -1.0, 99.0]]])
    target = torch.zeros(1, 1, 4)
    assert pitch_l1_loss(predicted, target, voiced, mask).item() == 1.0  # frames 0 and 2 only
    confident = torch.tensor([[[20.0, -20.0, 20.0, -20.0]]])
    assert voicing_loss(confident, voiced, mask).item() < 1e-6
    assert contour_discriminator_loss(mask, torch.zeros(1, 1, 4), mask).item() == 0.0
    assert contour_generator_loss(torch.zeros(1, 1, 4), mask).item() == 1.0


def test_discriminator_learns_flat_from_lively_contours() -> None:
    torch.manual_seed(2)
    disc = PitchDiscriminator(HIDDEN, channels=32)
    optimizer = torch.optim.Adam(disc.parameters(), lr=3e-3)
    frames = torch.randn(4, HIDDEN, 60)
    mask = torch.ones(4, 1, 60)
    t = torch.linspace(0, 6.28, 60)
    lively = torch.cat([(4 * torch.sin(t)).expand(4, 1, 60) / 12, mask], dim=1)
    flat = torch.cat([torch.zeros(4, 1, 60), mask], dim=1)
    for _ in range(80):
        loss = contour_discriminator_loss(disc(frames, mask, lively), disc(frames, mask, flat), mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.1


def test_range_loss_sees_a_narrowed_contour_and_not_a_shifted_one() -> None:
    from pitch_part import range_loss
    t = torch.linspace(0, 6.28, 50)
    true = (6 * torch.sin(t)).view(1, 1, 50)
    voiced = torch.ones(1, 1, 50)
    mask = torch.ones(1, 1, 50)
    assert range_loss(true + 3.0, true, voiced, mask).item() < 1e-3   # a shift keeps the spread
    narrowed = range_loss(true * 0.8, true, voiced, mask).item()
    assert abs(narrowed - 0.2 * float(true.std(unbiased=False))) < 1e-2


def test_range_loss_pushes_a_flat_prediction_apart_and_skips_short_utterances() -> None:
    from pitch_part import range_loss
    target = torch.randn(2, 1, 40) * 4
    voiced = torch.ones(2, 1, 40)
    voiced[1, :, 5:] = 0  # only 5 voiced frames: left out
    mask = torch.ones(2, 1, 40)
    predicted = (torch.randn(2, 1, 40) * 0.1).requires_grad_(True)
    range_loss(predicted, target, voiced, mask).backward()
    assert predicted.grad[1].abs().sum() == 0
    spread_before = predicted[0].std()
    with torch.no_grad():
        stepped = predicted - 1.0 * predicted.grad
    assert stepped[0].std() > spread_before
