"""Frame-budget batches: coverage, limits, length grouping and epoch shuffling."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from batching import FrameBudgetBatches  # noqa: E402


def _lengths(n: int = 5000) -> list[int]:
    rng = random.Random(0)
    return [rng.randint(94, 1360) if rng.random() > 0.05 else rng.randint(1360, 2300) for _ in range(n)]


def test_each_clip_once_per_epoch_within_both_limits() -> None:
    frames = _lengths()
    sampler = FrameBudgetBatches(frames, max_batch=64, frame_budget=64 * 1360, seed=3)
    batches = sampler.batches()
    seen = sorted(i for b in batches for i in b)
    assert seen == list(range(len(frames)))
    for batch in batches:
        assert len(batch) <= 64
        assert len(batch) * max(frames[i] for i in batch) <= 64 * 1360


def test_short_clips_fill_whole_batches_and_long_ones_smaller() -> None:
    frames = _lengths()
    batches = FrameBudgetBatches(frames, max_batch=64, frame_budget=64 * 1360, seed=3).batches()
    short = [b for b in batches if max(frames[i] for i in b) <= 1360]
    long = [b for b in batches if max(frames[i] for i in b) > 2000]
    assert max(len(b) for b in short) == 64
    assert 30 <= max(len(b) for b in long) <= (64 * 1360) // 2000  # a pool's leftovers may be fewer


def test_batches_group_similar_lengths() -> None:
    frames = _lengths()
    batches = FrameBudgetBatches(frames, max_batch=64, frame_budget=64 * 1360, seed=3).batches()
    padding = sum(len(b) * max(frames[i] for i in b) - sum(frames[i] for i in b) for b in batches)
    assert padding / sum(frames) < 0.05


def test_epochs_differ_but_repeat_for_the_same_seed() -> None:
    frames = _lengths(800)
    a = FrameBudgetBatches(frames, 64, 64 * 1360, seed=1)
    b = FrameBudgetBatches(frames, 64, 64 * 1360, seed=1)
    first, second = list(iter(a)), list(iter(a))
    assert first != second
    assert first == list(iter(b))


def test_a_clip_over_the_budget_is_refused() -> None:
    with pytest.raises(ValueError):
        FrameBudgetBatches([100, 5000], 64, 4000)
