"""Avaz v3 with the frame-rate decoder (frame_decoder.py) in place of v2's upsampling decoder.

The text side, flow, posterior encoder and the pitch part are SynthesizerV3's; only decode() changes.
The pitch is required: the decoder builds its sine bank from it (the predicted pitch at inference).
"""
from __future__ import annotations

import torch

from frame_decoder import FrameDecoder
from synthesizer_v3 import SynthesizerV3


class SynthesizerFD(SynthesizerV3):
    def __init__(self, *args, frame_decoder: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fdec = FrameDecoder(in_channels=self.enc_q.out_channels, f0_ref_hz=self.f0_ref_hz, **(frame_decoder or {}))

    def decode(self, z: torch.Tensor, f0: torch.Tensor | None = None, source_noise: bool = True) -> torch.Tensor:
        """z [b, c, t], f0 [b, 1, t] in Hz -> audio [b, 1, t * 256]."""
        if f0 is None:
            raise ValueError("the frame decoder needs the pitch (use_pitch=True)")
        noise = None if source_noise else torch.zeros(z.size(0), z.size(2) * 256, device=z.device)
        return self.fdec(z, f0, noise)[:, None]

    def inference_state(self) -> dict:
        """Weights needed to speak: everything but the posterior encoder and v2's unused decoder and source."""
        skip = ("enc_q.", "dec.", "source.", "source_convs.")
        return {k: v for k, v in self.state_dict().items() if not k.startswith(skip)}
