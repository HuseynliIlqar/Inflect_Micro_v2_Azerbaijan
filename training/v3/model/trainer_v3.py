"""One training step of Avaz v3, separate from data loading and files so it can be tested on the CPU.

Phases (stage 8 of training/v3/PLAN.md):
- "prosody": only the rhythm and pitch parts learn: the duration predictor with its
  discriminator (stage 4), the pitch predictor with its contour discriminator and the prior's
  pitch shift (stage 5). Everything they read is detached or frozen, so the decoder is not even
  run. Cheap: no audio losses.
- "disc_warmup": the generator is frozen; the audio discriminators adapt to the decoder.
- "joint": everything learns. The decoder gets the true pitch through the harmonic source
  (stage 6), the reconstruction and adversarial losses are the polish run's, and the
  prosody losses continue.

The audio losses and discriminators are the polish run's (training/v2/polish), passed in.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from duration_discriminator import DurationDiscriminator, duration_discriminator_loss, duration_generator_loss
from pitch_part import (PitchDiscriminator, contour_discriminator_loss, contour_generator_loss, pitch_features,
                        pitch_l1_loss, range_loss, voicing_loss)

PHASES = ("prosody", "disc_warmup", "joint")
PROSODY_PARTS = ("dp", "pitch", "prior_f0")
GROUPS = {"posterior": ("enc_q",), "linguistic": ("enc_p", "flow", "dp", "lang_emb"),
          "decoder": ("dec",), "source": ("source", "source_convs"), "prosody": ("pitch", "prior_f0")}


@dataclass(frozen=True)
class Weights:
    mel_ms: float = 9.534       # the polish run's calibration (45 x the old mel loss, 3/4 of it)
    stft: float = 3.885         # and 1/4 of it
    adv_scale: float = 6.0 / 11.0
    feature: float = 1.0
    kl: float = 1.0
    duration: float = 1.0
    duration_adv: float = 1.0   # VITS2 adds the duration generator loss with weight 1
    pitch_l1: float = 1.0
    voicing: float = 1.0
    pitch_adv: float = 2.0      # 1.0 in the trial run, which still narrowed the range by about 6%
    pitch_range: float = 1.0    # the per-utterance spread term (pitch_part.range_loss)


@dataclass(frozen=True)
class AudioLosses:
    """The polish run's functions, injected so this file needs neither torchaudio nor nnAudio."""
    multi_scale_mel: Callable
    linear_stft: Callable
    discriminator: Callable
    generator: Callable
    feature: Callable


def kl_loss(z_p, logs_q, m_p, logs_p, mask) -> torch.Tensor:
    """VITS' KL term, as the toolkit computes it."""
    value = logs_p.float() - logs_q.float() - 0.5
    value = value + 0.5 * ((z_p.float() - m_p.float()) ** 2) * torch.exp(-2.0 * logs_p.float())
    return torch.sum(value * mask.float()) / torch.sum(mask.float()).clamp_min(1.0)


def parameter_groups(model: nn.Module, multipliers: dict[str, float]) -> list[dict]:
    """AdamW groups by part; every parameter must belong to exactly one group."""
    named = dict(model.named_parameters())
    groups = []
    for group, parts in GROUPS.items():
        params = [p for n, p in named.items() if n.split(".")[0] in parts]
        if not params:
            raise RuntimeError(f"group {group} is empty")
        groups.append({"params": params, "name": group, "lr_multiplier": multipliers.get(group, 1.0)})
    covered = sum(len(g["params"]) for g in groups)
    if covered != len(named):
        unknown = [n for n in named if not any(n.split(".")[0] in parts for parts in GROUPS.values())]
        raise RuntimeError(f"parameters outside every group: {unknown[:5]}")
    return groups


def set_phase(model: nn.Module, phase: str) -> None:
    """Freeze what the phase does not train."""
    if phase not in PHASES:
        raise ValueError(phase)
    for name, param in model.named_parameters():
        part = name.split(".")[0]
        param.requires_grad_(phase == "joint" or (phase == "prosody" and part in PROSODY_PARTS))


def set_learning_rates(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """lr x each group's multiplier; zero for groups whose parameters are all frozen."""
    for group in optimizer.param_groups:
        trains = any(p.requires_grad for p in group["params"])
        group["lr"] = lr * group.get("lr_multiplier", 1.0) if trains else 0.0


class StepV3:
    """Holds the networks and optimisers; step(batch) does one optimisation step and returns metrics."""

    def __init__(self, model, audio_disc: nn.Module, losses: AudioLosses, segment_size: int, hop: int,
                 sample_rate: int, lr: float = 2e-4, lr_decay: float = 0.99999, betas=(0.8, 0.99),
                 eps: float = 1e-9, multipliers: dict[str, float] | None = None, weights: Weights = Weights(),
                 disc_batch: int = 12, max_grad_norm: float = 10.0, disc_clip: float = 500.0,
                 amp_dtype: torch.dtype | None = None) -> None:
        self.model, self.audio_disc, self.losses, self.weights = model, audio_disc, losses, weights
        self.segment_size, self.hop, self.sample_rate = segment_size, hop, sample_rate
        self.disc_batch, self.max_grad_norm, self.disc_clip = disc_batch, max_grad_norm, disc_clip
        self.lr, self.lr_decay, self.amp_dtype = lr, lr_decay, amp_dtype
        hidden = model.enc_p.hidden_channels
        self.duration_disc = DurationDiscriminator(hidden)
        self.pitch_disc = PitchDiscriminator(hidden)
        adam = dict(betas=betas, eps=eps)
        self.opt_g = torch.optim.AdamW(parameter_groups(model, multipliers or {"decoder": 0.25}), lr=lr, **adam)
        self.opt_d = torch.optim.AdamW(audio_disc.parameters(), lr=lr, **adam)
        self.opt_prosody_d = torch.optim.AdamW(list(self.duration_disc.parameters())
                                               + list(self.pitch_disc.parameters()), lr=lr, **adam)
        self.phase = None

    def to(self, device) -> "StepV3":
        for module in (self.model, self.audio_disc, self.duration_disc, self.pitch_disc):
            module.to(device)
        return self

    def enter(self, phase: str) -> None:
        if phase != self.phase:
            set_phase(self.model, phase)
            self.phase = phase

    def _autocast(self, device: torch.device):
        if self.amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=device.type, dtype=self.amp_dtype)

    def state_dict(self) -> dict:
        return {"model": self.model.state_dict(), "audio_disc": self.audio_disc.state_dict(),
                "duration_disc": self.duration_disc.state_dict(), "pitch_disc": self.pitch_disc.state_dict(),
                "opt_g": self.opt_g.state_dict(), "opt_d": self.opt_d.state_dict(),
                "opt_prosody_d": self.opt_prosody_d.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        self.audio_disc.load_state_dict(state["audio_disc"])
        self.duration_disc.load_state_dict(state["duration_disc"])
        self.pitch_disc.load_state_dict(state["pitch_disc"])
        self.opt_g.load_state_dict(state["opt_g"])
        self.opt_d.load_state_dict(state["opt_d"])
        self.opt_prosody_d.load_state_dict(state["opt_prosody_d"])

    def set_group_multiplier(self, name: str, multiplier: float) -> None:
        """Change one generator group's learning-rate multiplier, e.g. after a resume, which restores the saved ones."""
        groups = [g for g in self.opt_g.param_groups if g["name"] == name]
        if not groups:
            raise KeyError(name)
        for group in groups:
            group["lr_multiplier"] = multiplier

    def _prosody_losses(self, out: dict, train_disc: bool) -> tuple[torch.Tensor, dict]:
        """Duration and pitch losses for the generator; steps their two discriminators first."""
        x_mask, y_mask = out["x_mask"], out["y_mask"]
        real_pitch = pitch_features(out["semitones"], out["voiced"])
        fake_pitch = pitch_features(out["semitones_pred"], out["voiced"])  # judged on the true voicing
        if train_disc:
            dd_real, dd_fake = self.duration_disc(out["hidden"], x_mask, out["logw_target"].detach(), out["logw"].detach())
            pd_real = self.pitch_disc(out["frames"], y_mask, real_pitch)
            pd_fake = self.pitch_disc(out["frames"], y_mask, fake_pitch.detach())
            loss = (duration_discriminator_loss(dd_real, dd_fake, x_mask)
                    + contour_discriminator_loss(pd_real, pd_fake, y_mask))
            self.opt_prosody_d.zero_grad(set_to_none=True)
            loss.backward()
            self.opt_prosody_d.step()
        for p in list(self.duration_disc.parameters()) + list(self.pitch_disc.parameters()):
            p.requires_grad_(False)
        _, dd_fake = self.duration_disc(out["hidden"], x_mask, out["logw_target"], out["logw"])
        pd_fake = self.pitch_disc(out["frames"], y_mask, fake_pitch)
        for p in list(self.duration_disc.parameters()) + list(self.pitch_disc.parameters()):
            p.requires_grad_(True)
        w = self.weights
        parts = {"loss_duration": out["l_length"].float().sum(),
                 "loss_duration_adv": duration_generator_loss(dd_fake, x_mask),
                 "loss_pitch_l1": pitch_l1_loss(out["semitones_pred"], out["semitones"], out["voiced"], y_mask),
                 "loss_voicing": voicing_loss(out["voicing_logit"], out["voiced"], y_mask),
                 "loss_pitch_adv": contour_generator_loss(pd_fake, y_mask),
                 "loss_pitch_range": range_loss(out["semitones_pred"], out["semitones"], out["voiced"], y_mask),
                 "loss_kl": kl_loss(out["z_p"], out["logs_q"], out["m_p"], out["logs_p"], y_mask)}
        total = (w.duration * parts["loss_duration"] + w.duration_adv * parts["loss_duration_adv"]
                 + w.pitch_l1 * parts["loss_pitch_l1"] + w.voicing * parts["loss_voicing"]
                 + w.pitch_adv * parts["loss_pitch_adv"] + w.pitch_range * parts["loss_pitch_range"]
                 + w.kl * parts["loss_kl"])
        return total, parts

    def step(self, batch, phase: str, step: int) -> dict:
        """batch: (x, x_lengths, spec, spec_lengths, wav, wav_lengths, f0) on the model's device.
        step: the global step, for the learning-rate decay."""
        self.enter(phase)
        lr = self.lr * self.lr_decay ** step
        set_learning_rates(self.opt_g, lr)
        set_learning_rates(self.opt_d, lr)
        set_learning_rates(self.opt_prosody_d, lr)
        x, x_lengths, spec, spec_lengths, wav, _, f0 = batch
        metrics = {"phase": phase, "lr": lr}
        with torch.set_grad_enabled(phase != "disc_warmup"), self._autocast(x.device):
            out = self.model(x, x_lengths, spec, spec_lengths, f0=f0, decode=phase != "prosody")
        # Under bf16 autocast the outputs are bf16; the discriminators and losses run in float32.
        out = {k: v.float() if torch.is_tensor(v) and v.is_floating_point() else v for k, v in out.items()}
        if phase == "prosody":
            loss_g, parts = self._prosody_losses(out, train_disc=True)
            return self._generator_step(loss_g, parts, metrics)

        # Audio discriminators (disc_warmup and joint), on the first disc_batch clips.
        real = self.model.commons.slice_segments(wav, out["ids_slice"] * self.hop, self.segment_size * self.hop)
        real_d, fake_d = real[: self.disc_batch], out["audio"][: self.disc_batch]
        outputs = self.audio_disc(real_d, fake_d.detach())
        d_parts = {k: self.losses.discriminator(o.real_scores, o.fake_scores) for k, o in outputs.items()}
        loss_d = sum(d_parts.values())
        self.opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        norms = {k: float(torch.nn.utils.clip_grad_norm_(f.parameters(), self.disc_clip))
                 for k, f in self.audio_disc.families.items()}
        self.opt_d.step()
        metrics.update(loss_d=float(loss_d.detach()), **{f"d_{k}": float(v.detach()) for k, v in d_parts.items()},
                       **{f"gn_{k}": round(v, 4) for k, v in norms.items()})
        if phase == "disc_warmup":
            return metrics

        loss_g, parts = self._prosody_losses(out, train_disc=True)
        for p in self.audio_disc.parameters():
            p.requires_grad_(False)
        outputs = self.audio_disc(real_d, fake_d)
        for p in self.audio_disc.parameters():
            p.requires_grad_(True)
        w = self.weights
        parts["loss_adv"] = sum(self.losses.generator(o.fake_scores) for o in outputs.values())
        parts["loss_fm"] = sum(self.losses.feature(o.real_maps, o.fake_maps) for o in outputs.values())
        parts["loss_mel_ms"] = self.losses.multi_scale_mel(out["audio"], real, self.sample_rate)
        parts["loss_stft"] = self.losses.linear_stft(out["audio"], real)
        loss_g = (loss_g + w.adv_scale * (parts["loss_adv"] + w.feature * parts["loss_fm"])
                  + w.mel_ms * parts["loss_mel_ms"] + w.stft * parts["loss_stft"])
        return self._generator_step(loss_g, parts, metrics)

    def _generator_step(self, loss_g: torch.Tensor, parts: dict, metrics: dict) -> dict:
        self.opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        trainable = [p for g in self.opt_g.param_groups for p in g["params"] if p.requires_grad]
        metrics["gn_g"] = float(torch.nn.utils.clip_grad_norm_(trainable, self.max_grad_norm))
        self.opt_g.step()
        metrics.update(loss_g=float(loss_g.detach()), **{k: float(v.detach()) for k, v in parts.items()})
        return metrics
