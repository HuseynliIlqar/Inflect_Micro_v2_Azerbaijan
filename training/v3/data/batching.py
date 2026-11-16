"""Batches of similar-length clips whose padded size stays within a frame budget.

v2 trained on clips up to 14.5 s with 64 clips per batch. Dataset v2 adds 735 paragraphs of up to
24.4 s; a batch of 64 padded to such a length would need about 1.7 times v2's memory. So each
epoch the shuffled clips are taken in pools, sorted by length inside each pool, and cut into
batches of at most max_batch clips and at most frame_budget padded frames; the batch order is
then shuffled. Short clips still come 64 at a time, the longest about 37 at a time, and less
padding is wasted. Deterministic for a given seed and epoch.
"""
from __future__ import annotations

import random
from typing import Iterator, Sequence

import torch

POOL_BATCHES = 50


class FrameBudgetBatches(torch.utils.data.Sampler):
    def __init__(self, frames: Sequence[int], max_batch: int, frame_budget: int, seed: int = 0) -> None:
        if max(frames) > frame_budget:
            raise ValueError(f"a clip of {max(frames)} frames exceeds the budget of {frame_budget}")
        self.frames, self.max_batch, self.frame_budget, self.seed = list(frames), max_batch, frame_budget, seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def batches(self) -> list[list[int]]:
        rng = random.Random(self.seed * 1_000_003 + self.epoch)
        order = list(range(len(self.frames)))
        rng.shuffle(order)
        pool_size = POOL_BATCHES * self.max_batch
        batches = []
        for start in range(0, len(order), pool_size):
            pool = sorted(order[start: start + pool_size], key=lambda i: self.frames[i])
            current: list[int] = []
            for index in pool:  # ascending, so the newest clip is the batch's longest
                if current and (len(current) + 1 > self.max_batch
                                or (len(current) + 1) * self.frames[index] > self.frame_budget):
                    batches.append(current)
                    current = []
                current.append(index)
            if current:
                batches.append(current)
        rng.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        batches = self.batches()
        self.epoch += 1
        return iter(batches)

    def __len__(self) -> int:
        return len(self.batches())
