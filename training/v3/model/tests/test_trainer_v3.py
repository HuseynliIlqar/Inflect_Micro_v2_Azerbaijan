"""One v3 training step per phase on a tiny model: what learns, what stays frozen, and that the
losses are finite. Needs torchaudio (the polish run's losses), so it runs in the training venv:

    D:/ktts/avaz-train/.venv/Scripts/python.exe -m pytest training/v3/model/tests/test_trainer_v3.py -q -o addopts=""
"""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("torchaudio")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(REPO / "training" / "v2" / "polish"))

from discriminators import AvazDiscriminator, DiscriminatorSpec  # noqa: E402
from losses import discriminator_loss, feature_loss, generator_loss, linear_stft_loss, multi_scale_mel_loss  # noqa: E402
from monotonic_align import maximum_path  # noqa: E402
from runtime import load_runtime  # noqa: E402
from synthesizer_v3 import SynthesizerV3  # noqa: E402
from trainer_v3 import AudioLosses, StepV3, parameter_groups  # noqa: E402

TINY = {"inter_channels": 8, "hidden_channels": 16, "filter_channels": 32, "n_heads": 2, "n_layers": 2,
        "kernel_size": 3, "p_dropout": 0.1, "resblock": "1", "resblock_kernel_sizes": [3, 7],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5]], "upsample_rates": [8, 8, 2, 2],
        "upsample_initial_channel": 32, "upsample_kernel_sizes": [16, 16, 4, 4], "use_sdp": False}
N_VOCAB, SPEC, SEGMENT, HOP, FRAMES = 20, 33, 8, 256, 24
LOSSES = AudioLosses(multi_scale_mel_loss, linear_stft_loss, discriminator_loss, generator_loss, feature_loss)


@pytest.fixture(scope="module")
def runtime():
    return load_runtime(REPO / "model" / "runtime")


def _trainer(runtime) -> StepV3:
    torch.manual_seed(0)
    model = SynthesizerV3(runtime, N_VOCAB, SPEC, SEGMENT, copy.deepcopy(TINY), maximum_path)
    disc = AvazDiscriminator(DiscriminatorSpec(families=("mpd", "mrd")))
    return StepV3(model, disc, LOSSES, SEGMENT, HOP, 24000, lr=1e-3, disc_batch=2)


def _batch():
    torch.manual_seed(1)
    x = torch.randint(1, N_VOCAB, (2, 6))
    x_lengths = torch.tensor([6, 5])
    spec = torch.rand(2, SPEC, FRAMES)
    spec_lengths = torch.tensor([FRAMES, 20])
    wav = 0.1 * torch.randn(2, 1, FRAMES * HOP)
    f0 = 170.0 * torch.pow(2.0, torch.randn(2, FRAMES) / 6)
    f0[:, ::4] = 0.0
    f0[1, 20:] = 0.0
    return x, x_lengths, spec, spec_lengths, wav, torch.tensor([FRAMES * HOP, 20 * HOP]), f0


def _snapshot(module: torch.nn.Module) -> dict:
    return {n: p.detach().clone() for n, p in module.named_parameters()}


def _changed_parts(before: dict, module: torch.nn.Module) -> set[str]:
    return {n.split(".")[0] for n, p in module.named_parameters() if not torch.equal(before[n], p.detach())}


def test_every_parameter_is_in_exactly_one_group(runtime) -> None:
    trainer = _trainer(runtime)
    grouped = [id(p) for g in parameter_groups(trainer.model, {}) for p in g["params"]]
    assert len(grouped) == len(set(grouped)) == len(list(trainer.model.parameters()))


def test_prosody_phase_trains_only_the_rhythm_and_pitch_parts(runtime) -> None:
    trainer = _trainer(runtime)
    model_before, audio_before = _snapshot(trainer.model), _snapshot(trainer.audio_disc)
    dd_before, pd_before = _snapshot(trainer.duration_disc), _snapshot(trainer.pitch_disc)
    metrics = trainer.step(_batch(), "prosody", step=0)
    assert _changed_parts(model_before, trainer.model) == {"dp", "pitch", "prior_f0"}
    assert not _changed_parts(audio_before, trainer.audio_disc)
    assert _changed_parts(dd_before, trainer.duration_disc) and _changed_parts(pd_before, trainer.pitch_disc)
    for key in ("loss_duration", "loss_duration_adv", "loss_pitch_l1", "loss_voicing", "loss_pitch_adv", "loss_kl"):
        assert math.isfinite(metrics[key]), key
    assert "loss_d" not in metrics and "loss_mel_ms" not in metrics


def test_disc_warmup_trains_only_the_audio_discriminators(runtime) -> None:
    trainer = _trainer(runtime)
    model_before, audio_before = _snapshot(trainer.model), _snapshot(trainer.audio_disc)
    metrics = trainer.step(_batch(), "disc_warmup", step=0)
    assert not _changed_parts(model_before, trainer.model)
    assert _changed_parts(audio_before, trainer.audio_disc) == {"families"}
    assert math.isfinite(metrics["loss_d"]) and "loss_g" not in metrics


def test_joint_phase_trains_everything_including_the_source(runtime) -> None:
    trainer = _trainer(runtime)
    model_before = _snapshot(trainer.model)
    metrics = trainer.step(_batch(), "joint", step=0)
    changed = _changed_parts(model_before, trainer.model)
    assert {"enc_p", "enc_q", "flow", "dp", "dec", "pitch", "prior_f0", "source_convs"} <= changed
    for key in ("loss_g", "loss_d", "loss_mel_ms", "loss_stft", "loss_adv", "loss_fm", "loss_pitch_l1"):
        assert math.isfinite(metrics[key]), key


def test_learning_rate_decays_and_frozen_groups_get_zero(runtime) -> None:
    trainer = _trainer(runtime)
    trainer.step(_batch(), "prosody", step=1000)
    rates = {g["name"]: g["lr"] for g in trainer.opt_g.param_groups}
    assert rates["prosody"] == pytest.approx(1e-3 * 0.99999 ** 1000)
    assert rates["decoder"] == 0.0 and rates["posterior"] == 0.0
    assert rates["linguistic"] > 0  # holds dp
    trainer.step(_batch(), "joint", step=1000)
    rates = {g["name"]: g["lr"] for g in trainer.opt_g.param_groups}
    assert rates["decoder"] == pytest.approx(0.25 * rates["posterior"])


def test_state_round_trip_resumes_identically(runtime) -> None:
    trainer = _trainer(runtime)
    trainer.step(_batch(), "joint", step=0)
    state = copy.deepcopy(trainer.state_dict())
    other = _trainer(runtime)
    other.load_state_dict(state)
    for (name, a), b in zip(trainer.model.named_parameters(), other.model.parameters()):
        assert torch.equal(a, b), name
    assert other.opt_g.state_dict()["state"].keys() == trainer.opt_g.state_dict()["state"].keys()


def test_group_multiplier_survives_a_resume(runtime) -> None:
    trainer = _trainer(runtime)
    trainer.step(_batch(), "joint", step=0)
    other = _trainer(runtime)
    other.load_state_dict(copy.deepcopy(trainer.state_dict()))  # restores the saved multipliers
    other.set_group_multiplier("source", 3.0)
    other.step(_batch(), "joint", step=10)
    rates = {g["name"]: g["lr"] for g in other.opt_g.param_groups}
    assert rates["source"] == pytest.approx(3.0 * rates["posterior"])
    assert rates["decoder"] == pytest.approx(0.25 * rates["posterior"])
    with pytest.raises(KeyError):
        other.set_group_multiplier("no-such-group", 2.0)


@pytest.mark.parametrize("phase", ["prosody", "disc_warmup", "joint"])
def test_every_phase_runs_under_bf16_autocast(runtime, phase) -> None:
    """The pod trains with --bf16: the model's outputs come out in bfloat16, the discriminators
    and losses must still get float32 (the first trial run stopped on exactly this)."""
    trainer = _trainer(runtime)
    trainer.amp_dtype = torch.bfloat16
    metrics = trainer.step(_batch(), phase, step=0)
    assert all(math.isfinite(v) for v in metrics.values() if isinstance(v, float))


def test_phases_end_with_a_final_prosody_phase() -> None:
    from train_v3 import phase_for
    assert [phase_for(s, 10, 5, 20) for s in (0, 9, 10, 14, 15, 34, 35, 40)] == [
        "prosody", "prosody", "disc_warmup", "disc_warmup", "joint", "joint", "prosody", "prosody"]
    assert phase_for(10_000, 10, 5) == "joint"  # without a joint length: joint to the end, as before
