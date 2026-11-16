"""Monotonic alignment search (VITS' maximum_path), in NumPy, vectorised over text tokens.

Same recurrence and tie-breaking as VITS' Cython core: each audio frame takes one token, the token
index starts at 0, ends at the last token and advances by at most one per frame. Training only.
"""
from __future__ import annotations

import numpy as np
import torch


def _path(value: np.ndarray) -> np.ndarray:
    """value [t_y, t_x] -> 0/1 path of the same shape."""
    t_y, t_x = value.shape
    acc = np.full((t_y, t_x), -np.inf, dtype=np.float64)
    columns = np.arange(t_x)
    for y in range(t_y):
        low, high = max(0, t_x + y - t_y), min(t_x, y + 1)
        if y == 0:
            acc[0, 0] = value[0, 0]
            continue
        stay = acc[y - 1].copy()
        stay[columns == y] = -np.inf  # x == y: the token cannot have been reached a frame earlier
        move = np.concatenate(([-np.inf], acc[y - 1, :-1]))
        best = np.maximum(stay, move)
        row = np.full(t_x, -np.inf)
        row[low:high] = value[y, low:high] + best[low:high]
        acc[y] = row
    path = np.zeros((t_y, t_x), dtype=np.float32)
    index = t_x - 1
    for y in range(t_y - 1, -1, -1):
        path[y, index] = 1.0
        if index != 0 and (index == y or acc[y - 1, index] < acc[y - 1, index - 1]):
            index -= 1
    return path


def maximum_path(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """neg_cent, mask [b, t_y, t_x] -> path [b, t_y, t_x] on neg_cent's device and dtype."""
    values = neg_cent.detach().float().cpu().numpy()
    valid = mask.detach().cpu().numpy() > 0
    paths = np.zeros(values.shape, dtype=np.float32)
    for b in range(values.shape[0]):
        t_y = int(valid[b].any(axis=1).sum())
        t_x = int(valid[b].any(axis=0).sum())
        if t_x == 0 or t_y < t_x:
            raise ValueError(f"alignment needs at least one frame per token: {t_y} frames, {t_x} tokens")
        paths[b, :t_y, :t_x] = _path(values[b, :t_y, :t_x].astype(np.float64))
    return torch.from_numpy(paths).to(device=neg_cent.device, dtype=neg_cent.dtype)
