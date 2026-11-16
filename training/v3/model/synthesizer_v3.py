"""Avaz v3 synthesiser: the v2 network plus the parts that stages 4-6 add.

The v2 parts keep their names and classes (enc_p, enc_q, flow, dp, dec), so a v2 training
checkpoint loads into them unchanged. Added so far:
- lang_emb: one vector per language, added to the phoneme embeddings. It starts at zero and v3
  uses only language 0, so it changes nothing until Turkish (v3.1) trains it.
- pitch, prior_f0: the pitch part (stage 5, pitch_part.py). prior_f0 starts at zero.
- source, source_convs: the harmonic source (stage 6, harmonic_source.py), injected into v2's
  decoder at every upsampling stage. The injections start at zero.

Alignment (MAS) still uses the per-phoneme prior, exactly as in v2; the pitch shift is added to
the prior only after it is expanded to frames, where the KL term sees it.

forward() returns a dict with everything the losses need: the text hidden states and both log
durations for the duration discriminator (stage 4), and, when the true pitch is passed, the
predicted and true pitch for the pitch losses.
"""
from __future__ import annotations

import math
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from harmonic_source import HarmonicSource, source_injections
from pitch_part import (F0PriorShift, PitchPredictor, expand_to_frames, hz_to_semitones, pitch_features,
                        semitones_to_hz)
from runtime import Runtime

V2_PARTS = ("enc_p", "enc_q", "flow", "dp", "dec")
VOICED_THRESHOLD = 0.5


class SynthesizerV3(nn.Module):
    def __init__(self, runtime: Runtime, n_vocab: int, spec_channels: int, segment_size: int, model: dict,
                 maximum_path: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], n_languages: int = 4,
                 f0_ref_hz: float = 178.0) -> None:  # the voice's median: 177.2 Hz in v1, 178.7 in v2
        super().__init__()
        self.f0_ref_hz = float(f0_ref_hz)
        if model.get("use_sdp", False):
            raise ValueError("v3 builds on v2's deterministic duration predictor (use_sdp false)")
        m = runtime.models
        self.commons = runtime.commons
        self.lrelu_slope = runtime.modules.LRELU_SLOPE
        self.maximum_path = maximum_path
        self.segment_size = segment_size
        inter, hidden = int(model["inter_channels"]), int(model["hidden_channels"])
        self.enc_p = m.TextEncoder(n_vocab, inter, hidden, int(model["filter_channels"]), int(model["n_heads"]),
                                   int(model["n_layers"]), int(model["kernel_size"]), float(model["p_dropout"]))
        self.dec = m.Generator(inter, model["resblock"], model["resblock_kernel_sizes"], model["resblock_dilation_sizes"],
                               model["upsample_rates"], int(model["upsample_initial_channel"]),
                               model["upsample_kernel_sizes"],
                               decoder_alias_free=model.get("decoder_alias_free", False),
                               decoder_alias_free_start_stage=model.get("decoder_alias_free_start_stage", 2),
                               decoder_snake_logscale=model.get("decoder_snake_logscale", True))
        self.enc_q = m.PosteriorEncoder(spec_channels, inter, hidden, 5, 1, 16)
        self.flow = m.ResidualCouplingBlock(inter, hidden, 5, 1, 4)
        self.dp = m.DurationPredictor(hidden, 256, 3, 0.5)
        self.lang_emb = nn.Embedding(n_languages, hidden)
        nn.init.zeros_(self.lang_emb.weight)
        self.pitch = PitchPredictor(hidden)
        self.prior_f0 = F0PriorShift(inter)
        self.source = HarmonicSource(hop=math.prod(model["upsample_rates"]))
        self.source_convs = source_injections(list(model["upsample_rates"]), int(model["upsample_initial_channel"]))

    def load_v2_state(self, state: dict) -> list[str]:
        """Load a v2 generator state dict. Returns the v3 tensors it did not cover (the new parts)."""
        unexpected = [key for key in state if key.split(".")[0] not in V2_PARTS]
        if unexpected:
            raise KeyError(f"not v2 generator tensors: {unexpected[:5]}")
        result = self.load_state_dict(state, strict=False)
        missing_v2 = [key for key in result.missing_keys if key.split(".")[0] in V2_PARTS]
        if missing_v2:
            raise KeyError(f"v2 tensors missing from the checkpoint: {missing_v2[:5]}")
        return list(result.missing_keys)

    def encode_text(self, x: torch.Tensor, x_lengths: torch.Tensor, lang: torch.Tensor | None = None):
        """v2's TextEncoder.forward with the language vector added to the phoneme embeddings."""
        enc = self.enc_p
        if lang is None:
            lang = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h = enc.emb(x) * math.sqrt(enc.hidden_channels) + self.lang_emb(lang)[:, None, :]
        h = torch.transpose(h, 1, -1)
        x_mask = torch.unsqueeze(self.commons.sequence_mask(x_lengths, h.size(2)), 1).to(h.dtype)
        h = enc.encoder(h * x_mask, x_mask)
        stats = enc.proj(h) * x_mask
        m_p, logs_p = torch.split(stats, enc.out_channels, dim=1)
        return h, m_p, logs_p, x_mask

    def decode(self, z: torch.Tensor, f0: torch.Tensor | None = None, source_noise: bool = True) -> torch.Tensor:
        """v2's Generator.forward, plus the harmonic source after each upsampling when f0 [b, 1, t] is given."""
        dec = self.dec
        source = self.source(f0, source_noise) if f0 is not None else None
        x = dec.conv_pre(z)
        for i in range(dec.num_upsamples):
            if dec.decoder_alias_free and i >= dec.decoder_alias_free_start_stage:
                x = dec.alias_free_pre_activations[i](x)
            else:
                x = F.leaky_relu(x, self.lrelu_slope)
            x = dec.ups[i](x)
            if source is not None:
                x = x + self.source_convs[i](source)
            xs = None
            for j in range(dec.num_kernels):
                block = dec.resblocks[i * dec.num_kernels + j](x)
                xs = block if xs is None else xs + block
            x = xs / dec.num_kernels
        x = dec.alias_free_post_activation(x) if dec.decoder_alias_free else F.leaky_relu(x)
        return torch.tanh(dec.conv_post(x))

    def _align(self, z_p, m_p, logs_p, x_mask, y_mask) -> torch.Tensor:
        with torch.no_grad():
            s_p_sq_r = torch.exp(-2 * logs_p)
            neg_cent = (torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)
                        + torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r)
                        + torch.matmul(z_p.transpose(1, 2), m_p * s_p_sq_r)
                        + torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True))
            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            return self.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()

    def forward(self, x, x_lengths, y, y_lengths, lang=None, f0=None, decode=True) -> dict:
        """f0: true pitch in Hz per spectrogram frame [b, t_y] (0 = unvoiced), or None for v2's path.
        decode=False skips the decoder (the prosody phase trains nothing behind it)."""
        hidden, m_p, logs_p, x_mask = self.encode_text(x, x_lengths, lang)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths)
        z_p = self.flow(z, y_mask)
        attn = self._align(z_p, m_p, logs_p, x_mask, y_mask)
        logw_target = torch.log(attn.sum(2) + 1e-6) * x_mask
        logw = self.dp(hidden, x_mask)
        l_length = torch.sum((logw - logw_target) ** 2, [1, 2]) / torch.sum(x_mask)
        m_p = expand_to_frames(m_p, attn)
        logs_p = expand_to_frames(logs_p, attn)
        out = {"l_length": l_length, "attn": attn, "x_mask": x_mask, "y_mask": y_mask, "z": z, "z_p": z_p,
               "m_q": m_q, "logs_q": logs_q, "hidden": hidden, "logw": logw, "logw_target": logw_target}
        if f0 is not None:
            semitones, voiced = hz_to_semitones(f0[:, None, : y_mask.size(2)], self.f0_ref_hz)
            semitones, voiced = semitones * y_mask, voiced * y_mask
            frames = expand_to_frames(hidden, attn)
            predicted, voicing_logit = self.pitch(frames, y_mask)
            m_p = m_p + self.prior_f0(pitch_features(semitones, voiced), y_mask)
            out.update(frames=frames, semitones=semitones, voiced=voiced, semitones_pred=predicted,
                       voicing_logit=voicing_logit)
        z_slice, ids_slice = self.commons.rand_slice_segments(z, y_lengths, self.segment_size)
        f0_slice = None
        if f0 is not None:
            f0_slice = self.commons.slice_segments(f0[:, None, : y_mask.size(2)] * y_mask, ids_slice, self.segment_size)
            out.update(f0_slice=f0_slice)
        out.update(ids_slice=ids_slice, m_p=m_p, logs_p=logs_p)
        if decode:
            out["audio"] = self.decode(z_slice, f0_slice)
        return out

    def predict_pitch(self, hidden, attn, y_mask, pitch_shift: float = 0.0, pitch_range: float = 1.0):
        """Predicted pitch for an alignment: (semitones, voiced flag, Hz), each [b, 1, t_y].

        pitch_range scales the contour around the voice's median; pitch_shift moves it, in semitones.
        """
        predicted, logit = self.pitch(expand_to_frames(hidden, attn), y_mask)
        voiced = (torch.sigmoid(logit) > VOICED_THRESHOLD).to(predicted.dtype) * y_mask
        semitones = (predicted * pitch_range + pitch_shift) * voiced
        return semitones, voiced, semitones_to_hz(semitones, voiced, self.f0_ref_hz)

    @torch.no_grad()
    def infer(self, x, x_lengths, lang=None, noise_scale=1.0, length_scale=1.0, max_len=None,
              use_pitch=True, pitch_shift=0.0, pitch_range=1.0, source_noise=True):
        """Returns (audio, alignment, frame mask, f0 in Hz per frame or None)."""
        hidden, m_p, logs_p, x_mask = self.encode_text(x, x_lengths, lang)
        w = torch.exp(self.dp(hidden, x_mask)) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(self.commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn = self.commons.generate_path(w_ceil, torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1))
        m_p = expand_to_frames(m_p, attn)
        logs_p = expand_to_frames(logs_p, attn)
        f0 = None
        if use_pitch:
            semitones, voiced, f0 = self.predict_pitch(hidden, attn, y_mask, pitch_shift, pitch_range)
            m_p = m_p + self.prior_f0(pitch_features(semitones, voiced), y_mask)
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, reverse=True)
        f0_dec = f0[:, :, :max_len] if f0 is not None else None
        return self.decode((z * y_mask)[:, :, :max_len], f0_dec, source_noise), attn, y_mask, f0
