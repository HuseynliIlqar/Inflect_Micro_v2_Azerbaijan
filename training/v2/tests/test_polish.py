import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "polish"))
from discriminators import AvazDiscriminator, DiscriminatorSpec, load_bigvgan_weights  # noqa: E402
from losses import (  # noqa: E402
    discriminator_loss,
    feature_loss,
    generator_loss,
    linear_stft_loss,
    multi_scale_mel_loss,
)

SR = 24000
SEGMENT = 16384


@pytest.fixture(scope="module")
def disc():
    torch.manual_seed(0)
    return AvazDiscriminator(DiscriminatorSpec())


def _tone(freq, seconds=SEGMENT / SR, batch=2):
    t = torch.arange(int(seconds * SR)) / SR
    return (0.3 * torch.sin(2 * torch.pi * freq * t)).repeat(batch, 1, 1)


def test_every_family_scores_real_and_fake_audio(disc):
    real, fake = _tone(200.0), _tone(210.0)
    outputs = disc(real, fake)
    assert set(outputs) == {"mpd", "mrd", "cqtd"}
    counts = {name: len(o.real_scores) for name, o in outputs.items()}
    assert counts == {"mpd": 5, "mrd": 3, "cqtd": 3}
    for o in outputs.values():
        assert all(s.shape[0] == 2 and torch.isfinite(s).all() for s in o.real_scores + o.fake_scores)
        assert len(o.real_maps) == len(o.fake_maps)


def test_generator_gradient_reaches_the_waveform_through_every_family(disc):
    real = _tone(200.0)
    fake = _tone(230.0).clone().requires_grad_(True)
    for p in disc.parameters():
        p.requires_grad_(False)
    outputs = disc(real, fake)
    loss = sum(generator_loss(o.fake_scores) + feature_loss(o.real_maps, o.fake_maps) for o in outputs.values())
    loss.backward()
    for p in disc.parameters():
        p.requires_grad_(True)
    assert fake.grad is not None and fake.grad.abs().sum() > 0


def test_discriminator_loss_is_lowest_when_it_separates_real_from_fake():
    ones, zeros = [torch.ones(2, 5)], [torch.zeros(2, 5)]
    assert discriminator_loss(ones, zeros).item() == 0.0
    assert discriminator_loss(zeros, ones).item() == 2.0
    assert generator_loss(ones).item() == 0.0


def test_spectral_losses_are_zero_for_identical_audio_and_grow_with_a_difference():
    real = _tone(200.0)
    assert multi_scale_mel_loss(real, real, SR).item() == pytest.approx(0.0, abs=1e-6)
    assert linear_stft_loss(real, real).item() == pytest.approx(0.0, abs=1e-6)
    near, far = real + 0.01 * _tone(6000.0), real + 0.1 * _tone(6000.0)
    assert 0 < linear_stft_loss(near, real) < linear_stft_loss(far, real)
    assert 0 < multi_scale_mel_loss(near, real, SR) < multi_scale_mel_loss(far, real, SR)


def test_linear_stft_loss_weighs_a_6khz_ring_more_than_the_mel_loss_does():
    """The reason for the second loss: mel bins squeeze the 6 kHz region."""
    real = _tone(200.0)
    ringing = real + 0.02 * _tone(6000.0)
    low_hum = real + 0.02 * _tone(300.0)
    stft_ratio = linear_stft_loss(ringing, real) / linear_stft_loss(low_hum, real)
    mel_ratio = multi_scale_mel_loss(ringing, real, SR) / multi_scale_mel_loss(low_hum, real, SR)
    assert stft_ratio > mel_ratio


def test_bigvgan_weights_are_copied_into_mpd_and_cqtd(disc):
    torch.manual_seed(1)
    donor = AvazDiscriminator(DiscriminatorSpec())
    payload = {"mpd": donor.families["mpd"].state_dict(), "mrd": donor.families["cqtd"].state_dict()}
    copied = load_bigvgan_weights(disc, payload)
    assert copied["mpd"] == len(list(disc.families["mpd"].parameters()))
    assert copied["cqtd"] == len(list(disc.families["cqtd"].parameters()))
    for name, tensor in disc.families["cqtd"].named_parameters():
        assert torch.equal(tensor, dict(donor.families["cqtd"].named_parameters())[name])


def test_bigvgan_weights_with_a_missing_tensor_are_refused(disc):
    payload = {"mpd": dict(disc.families["mpd"].state_dict()), "mrd": dict(disc.families["cqtd"].state_dict())}
    payload["mpd"].pop(next(iter(dict(disc.families["mpd"].named_parameters()))))
    with pytest.raises(RuntimeError, match="lack"):
        load_bigvgan_weights(disc, payload)
