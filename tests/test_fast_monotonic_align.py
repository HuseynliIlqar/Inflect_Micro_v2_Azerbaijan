"""The fast alignment must match the reference bit for bit.

Alignment decides which audio frame belongs to which phoneme, so a divergence
here would silently corrupt every duration target in training. These tests
compare against the toolkit's own reference implementation on randomised shapes,
including the tie cases where the recurrence has to pick between staying and
advancing.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.reference_monotonic_align import maximum_path as reference_path
from fast_monotonic_align import maximum_path as fast_path


def _batch(
    lengths: list[tuple[int, int]], *, seed: int = 0, scale: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a padded [batch, audio, text] score/mask pair."""
    generator = np.random.default_rng(seed)
    audio_max = max(audio for audio, _ in lengths)
    text_max = max(text for _, text in lengths)
    scores = generator.normal(0.0, scale, (len(lengths), audio_max, text_max))
    mask = np.zeros_like(scores, dtype=bool)
    for index, (audio, text) in enumerate(lengths):
        mask[index, :audio, :text] = True
    return (
        torch.from_numpy(scores.astype(np.float32)),
        torch.from_numpy(mask.astype(np.float32)),
    )


def _assert_same(lengths: list[tuple[int, int]], **kwargs) -> torch.Tensor:
    scores, mask = _batch(lengths, **kwargs)
    expected = reference_path(scores, mask)
    actual = fast_path(scores, mask)
    assert torch.equal(actual, expected)
    return actual


def test_single_item() -> None:
    _assert_same([(20, 7)], seed=1)


def test_square_alignment_has_only_one_legal_path() -> None:
    # audio == text forces every frame to advance.
    path = _assert_same([(9, 9)], seed=2)
    assert torch.equal(path[0], torch.eye(9))


def test_ragged_batch() -> None:
    _assert_same([(40, 11), (25, 9), (33, 4), (18, 18)], seed=3)


def test_batch_with_identical_lengths() -> None:
    _assert_same([(30, 10)] * 6, seed=4)


@pytest.mark.parametrize("seed", range(12))
def test_randomised_batches_match(seed: int) -> None:
    generator = np.random.default_rng(1000 + seed)
    lengths = []
    for _ in range(int(generator.integers(1, 6))):
        text = int(generator.integers(2, 25))
        audio = int(generator.integers(text, text + 40))
        lengths.append((audio, text))
    _assert_same(lengths, seed=seed)


def test_flat_scores_exercise_the_tie_rule() -> None:
    # Every score equal: the choice between staying and advancing is decided
    # purely by the tie-breaking rule, so any mismatch shows up here.
    lengths = [(24, 8), (17, 5)]
    scores, mask = _batch(lengths, seed=5, scale=0.0)
    assert torch.equal(fast_path(scores, mask), reference_path(scores, mask))


def test_quantised_scores_produce_many_ties() -> None:
    scores, mask = _batch([(30, 9), (22, 7)], seed=6, scale=1.0)
    scores = torch.round(scores)  # collisions on purpose
    assert torch.equal(fast_path(scores, mask), reference_path(scores, mask))


def test_every_frame_is_assigned_exactly_one_token() -> None:
    path = _assert_same([(35, 12), (28, 6)], seed=7)
    for index, (audio, _) in enumerate([(35, 12), (28, 6)]):
        assigned = path[index, :audio].sum(dim=1)
        assert torch.equal(assigned, torch.ones(audio))


def test_path_is_monotonic_and_spans_all_tokens() -> None:
    lengths = [(45, 13)]
    path = _assert_same(lengths, seed=8)
    positions = path[0, : lengths[0][0]].argmax(dim=1)
    steps = positions[1:] - positions[:-1]
    assert positions[0] == 0
    assert positions[-1] == lengths[0][1] - 1
    assert bool(((steps == 0) | (steps == 1)).all())


def test_padding_stays_zero() -> None:
    lengths = [(30, 10), (12, 4)]
    path = _assert_same(lengths, seed=9)
    assert float(path[1, 12:].abs().sum()) == 0.0
    assert float(path[1, :, 4:].abs().sum()) == 0.0


def test_mismatched_shapes_are_rejected() -> None:
    scores, mask = _batch([(10, 4)], seed=10)
    with pytest.raises(ValueError, match="same"):
        fast_path(scores, mask[:, :5])


def test_more_tokens_than_frames_is_rejected() -> None:
    scores, mask = _batch([(6, 6)], seed=11)
    mask[0, 4:, :] = 0.0  # 4 frames left for 6 tokens
    with pytest.raises(ValueError, match="at least one audio frame"):
        fast_path(scores, mask)


def test_non_rectangular_mask_is_rejected() -> None:
    scores, mask = _batch([(12, 5)], seed=12)
    mask[0, 3, 2] = 0.0
    with pytest.raises(ValueError, match="rectangular"):
        fast_path(scores, mask)


def test_dtype_and_device_follow_the_input() -> None:
    scores, mask = _batch([(14, 5)], seed=13)
    path = fast_path(scores.double(), mask)
    assert path.dtype == torch.float64
