"""Batched monotonic alignment search, numerically identical to the toolkit's.

The public toolkit ships a dependency-free `maximum_path` whose dynamic program
is a pure-Python double loop over (audio frames x text tokens) for every item in
the batch. On this corpus that is roughly 28 million interpreted iterations per
training step, which measured out at ~8.7 s/step on an A40 with the GPU sitting
idle. Upstream VITS avoids it with a compiled kernel.

This module keeps the identical recurrence and tie-breaking but evaluates it with
NumPy across the whole batch at once: the loop over audio frames stays (it is
inherently sequential), while the text axis and the batch axis are vectorised.
Correctness is the whole point here, so `tests/test_fast_monotonic_align.py`
asserts bit-identical output against the reference implementation.
"""

from __future__ import annotations

import numpy as np
import torch

NEG_INF = -np.inf


def _validate(neg_cent: torch.Tensor, mask: torch.Tensor) -> None:
    if neg_cent.ndim != 3 or mask.ndim != 3 or neg_cent.shape != mask.shape:
        raise ValueError(
            "neg_cent and mask must have the same [batch, audio, text] shape."
        )


def maximum_path(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return the maximum-score monotonic path through ``[batch, audio, text]``.

    Each valid audio frame is assigned to one text token. The token index starts
    at zero, ends at the final valid token, and either stays fixed or advances by
    one at each frame.
    """

    _validate(neg_cent, mask)
    scores = neg_cent.detach().to(device="cpu", dtype=torch.float32).numpy()
    valid = mask.detach().to(device="cpu").numpy() > 0
    batch_size, audio_frames, text_tokens = scores.shape

    audio_lengths = np.zeros(batch_size, dtype=np.int64)
    text_lengths = np.zeros(batch_size, dtype=np.int64)
    for index in range(batch_size):
        audio_length = int(np.any(valid[index], axis=1).sum())
        text_length = int(np.any(valid[index], axis=0).sum())
        if audio_length == 0 or text_length == 0:
            raise ValueError("Monotonic alignment received an empty valid sequence.")
        if audio_length < text_length:
            raise ValueError(
                "Monotonic alignment requires at least one audio frame per text token; "
                f"received {audio_length} frames and {text_length} tokens."
            )
        expected = np.zeros_like(valid[index])
        expected[:audio_length, :text_length] = True
        if not np.array_equal(valid[index], expected):
            raise ValueError(
                "Monotonic alignment mask must be one top-left rectangular valid region."
            )
        audio_lengths[index] = audio_length
        text_lengths[index] = text_length

    token_index = np.arange(text_tokens, dtype=np.int64)[None, :]
    in_text = token_index < text_lengths[:, None]

    accumulated = np.full((batch_size, text_tokens), NEG_INF, dtype=np.float32)
    accumulated[:, 0] = scores[:, 0, 0]
    advanced = np.zeros((batch_size, audio_frames, text_tokens), dtype=np.bool_)

    for frame in range(1, audio_frames):
        # Only items still inside their own audio length take this frame.
        active = frame < audio_lengths
        if not active.any():
            break
        lower = np.maximum(0, text_lengths + frame - audio_lengths)[:, None]
        upper = np.minimum(text_lengths - 1, frame)[:, None]
        writable = (token_index >= lower) & (token_index <= upper) & in_text
        writable &= active[:, None]

        stay = accumulated
        move = np.empty_like(accumulated)
        move[:, 0] = NEG_INF
        move[:, 1:] = accumulated[:, :-1]

        # Ties keep the reference behaviour: advance only on a strict improvement,
        # except on the diagonal where advancing is forced.
        use_move = (token_index == frame) | (move > stay)
        candidate = scores[:, frame, :] + np.where(use_move, move, stay)

        accumulated = np.where(writable, candidate, accumulated)
        advanced[:, frame, :] = use_move & writable

    paths = np.zeros(scores.shape, dtype=np.float32)
    for index in range(batch_size):
        audio_length = int(audio_lengths[index])
        text_position = int(text_lengths[index]) - 1
        item_advanced = advanced[index]
        item_paths = paths[index]
        for frame in range(audio_length - 1, -1, -1):
            item_paths[frame, text_position] = 1.0
            if frame > 0 and item_advanced[frame, text_position]:
                text_position -= 1
        if text_position != 0:
            raise RuntimeError(
                "Monotonic alignment backtracking did not reach the first token."
            )

    return torch.from_numpy(paths).to(device=neg_cent.device, dtype=neg_cent.dtype)


__all__ = ["maximum_path"]
