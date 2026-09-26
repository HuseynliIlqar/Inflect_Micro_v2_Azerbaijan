"""Frozen copy of the toolkit's original reference alignment.

Kept verbatim so the optimised implementation can be checked against the
behaviour it replaced, even after the toolkit module itself is swapped out.
Do not optimise this file - being slow and obvious is its job.
"""

from __future__ import annotations

import numpy as np
import torch


def maximum_path(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return the maximum-score monotonic path through ``[batch, audio, text]``.

    Each valid audio frame is assigned to one text token. The token index starts
    at zero, ends at the final valid token, and either stays fixed or advances
    by one at each frame. This is a training-only operation; deployable
    inference checkpoints do not contain or call it.
    """

    if neg_cent.ndim != 3 or mask.ndim != 3 or neg_cent.shape != mask.shape:
        raise ValueError(
            "neg_cent and mask must have the same [batch, audio, text] shape."
        )
    scores = neg_cent.detach().to(device="cpu", dtype=torch.float32).numpy()
    valid = mask.detach().to(device="cpu").numpy() > 0
    paths = np.zeros(scores.shape, dtype=np.float32)

    for batch_index in range(scores.shape[0]):
        valid_audio = np.any(valid[batch_index], axis=1)
        valid_text = np.any(valid[batch_index], axis=0)
        audio_length = int(valid_audio.sum())
        text_length = int(valid_text.sum())
        if audio_length == 0 or text_length == 0:
            raise ValueError("Monotonic alignment received an empty valid sequence.")
        if audio_length < text_length:
            raise ValueError(
                "Monotonic alignment requires at least one audio frame per text token; "
                f"received {audio_length} frames and {text_length} tokens."
            )
        expected_mask = np.zeros_like(valid[batch_index])
        expected_mask[:audio_length, :text_length] = True
        if not np.array_equal(valid[batch_index], expected_mask):
            raise ValueError(
                "Monotonic alignment mask must be one top-left rectangular valid region."
            )

        values = scores[batch_index, :audio_length, :text_length]
        accumulated = np.full(values.shape, -np.inf, dtype=np.float32)
        advanced = np.zeros(values.shape, dtype=np.bool_)
        accumulated[0, 0] = values[0, 0]

        for audio_index in range(1, audio_length):
            minimum_text = max(0, text_length + audio_index - audio_length)
            maximum_text = min(text_length - 1, audio_index)
            for text_index in range(minimum_text, maximum_text + 1):
                stay = accumulated[audio_index - 1, text_index]
                move = (
                    accumulated[audio_index - 1, text_index - 1]
                    if text_index > 0
                    else -np.inf
                )
                use_move = text_index == audio_index or move > stay
                predecessor = move if use_move else stay
                accumulated[audio_index, text_index] = (
                    values[audio_index, text_index] + predecessor
                )
                advanced[audio_index, text_index] = use_move

        text_index = text_length - 1
        for audio_index in range(audio_length - 1, -1, -1):
            paths[batch_index, audio_index, text_index] = 1.0
            if audio_index > 0 and advanced[audio_index, text_index]:
                text_index -= 1
        if text_index != 0:
            raise RuntimeError("Monotonic alignment backtracking did not reach the first token.")

    return torch.from_numpy(paths).to(device=neg_cent.device, dtype=neg_cent.dtype)


__all__ = ["maximum_path"]
