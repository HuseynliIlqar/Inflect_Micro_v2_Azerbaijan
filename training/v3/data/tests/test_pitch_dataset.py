"""Pitch lookup by prepared clip name and the collate that carries it. No toolkit, no audio."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pitch_dataset import PitchTable, WithPitch, collate_with_pitch, slug_of  # noqa: E402


def test_slug_is_the_source_stem() -> None:
    assert slug_of("0000123-000123-0a1b2c3d4e.wav") == "000123"
    assert slug_of("0013900-v2_A1_00001-ffffffffff.wav") == "v2_A1_00001"
    with pytest.raises(ValueError):
        slug_of("000123.wav")


@pytest.fixture()
def table(tmp_path: Path) -> PitchTable:
    np.savez(tmp_path / "a.npz", stems=np.array(["000001", "000002"]), offsets=np.array([0, 3, 7]),
             f0=np.array([100, 0, 110, 0, 0, 120, 130], dtype=np.float32))
    np.savez(tmp_path / "b.npz", stems=np.array(["v2_C_00001"]), offsets=np.array([0, 2]),
             f0=np.array([200, 210], dtype=np.float32))
    return PitchTable([tmp_path / "a.npz", tmp_path / "b.npz"])


def test_tracks_are_found_and_lengths_checked(table: PitchTable) -> None:
    assert len(table) == 3
    assert torch.equal(table.track("0000005-000002-aaaaaaaaaa.wav", 4), torch.tensor([0.0, 0.0, 120.0, 130.0]))
    with pytest.raises(ValueError, match="frames"):
        table.track("0000005-000002-aaaaaaaaaa.wav", 5)


class _Fake(torch.utils.data.Dataset):
    """Like the toolkit's dataset; with_speaker mimics the pod's patched version (4 items)."""

    def __init__(self, names_frames, with_speaker=False):
        self.records = [(Path("/data") / name, "phonemes", 0) for name, _ in names_frames]
        self.frames = [frames for _, frames in names_frames]
        self.with_speaker = with_speaker

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        n = self.frames[i]
        item = (torch.arange(1, 2 * n + 1), torch.ones(5, n), torch.ones(1, n * 256))
        return item + (torch.tensor(0),) if self.with_speaker else item


def test_dataset_and_collate_carry_pitch_in_spectrogram_order(table: PitchTable) -> None:
    data = WithPitch(_Fake([("0000001-v2_C_00001-bbbbbbbbbb.wav", 2), ("0000002-000002-cccccccccc.wav", 4)]), table)
    text, text_lengths, spec, spec_lengths, wav, wav_lengths, f0 = collate_with_pitch([data[0], data[1]])
    assert spec_lengths.tolist() == [4, 2]  # longest first, as the toolkit sorts
    assert torch.equal(f0, torch.tensor([[0.0, 0.0, 120.0, 130.0], [200.0, 210.0, 0.0, 0.0]]))
    assert f0.shape[1] == spec.shape[2]


def test_a_clip_without_pitch_is_refused(table: PitchTable) -> None:
    with pytest.raises(KeyError):
        WithPitch(_Fake([("0000001-999999-bbbbbbbbbb.wav", 2)]), table)


def test_the_patched_toolkits_speaker_id_is_dropped(table: PitchTable) -> None:
    data = WithPitch(_Fake([("0000002-000002-cccccccccc.wav", 4)], with_speaker=True), table)
    tokens, spec, wav, f0 = data[0]
    assert f0.shape == (4,) and spec.shape == (5, 4)
