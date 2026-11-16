"""Add the per-frame pitch (extract_pitch.py) to the toolkit's prepared dataset.

The toolkit names prepared clips "{index:07d}-{slug}-{hash10}.wav", where the slug is the source
file's stem ("000123" for v1, "v2_A1_00001" for v2), so the slug finds the clip's pitch track.
The track has one value per spectrogram frame; a length mismatch is an error, not something to
pad over, because it would mean the audio changed after the pitch was tracked.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

INDEX_WIDTH, HASH_WIDTH = 7, 10


def slug_of(prepared_name: str) -> str:
    stem = Path(prepared_name).stem
    if len(stem) <= INDEX_WIDTH + HASH_WIDTH + 2 or stem[INDEX_WIDTH] != "-" or stem[-HASH_WIDTH - 1] != "-":
        raise ValueError(f"not a prepared clip name: {prepared_name!r}")
    return stem[INDEX_WIDTH + 1: -HASH_WIDTH - 1]


class PitchTable:
    """Pitch tracks from one or more extract_pitch.py files, looked up by clip stem."""

    def __init__(self, paths: Sequence[Path]) -> None:
        self._tracks: dict[str, np.ndarray] = {}
        for path in paths:
            data = np.load(path)
            f0, offsets = data["f0"], data["offsets"]
            for i, stem in enumerate(data["stems"].tolist()):
                if stem in self._tracks:
                    raise ValueError(f"clip {stem!r} appears in two pitch files")
                self._tracks[stem] = f0[offsets[i]: offsets[i + 1]]

    def __len__(self) -> int:
        return len(self._tracks)

    def track(self, prepared_name: str, frames: int) -> torch.Tensor:
        values = self._tracks[slug_of(prepared_name)]
        if len(values) != frames:
            raise ValueError(f"{prepared_name}: pitch has {len(values)} frames, the spectrogram {frames}")
        return torch.from_numpy(np.asarray(values, dtype=np.float32).copy())


class WithPitch(torch.utils.data.Dataset):
    """Wraps a dataset of (tokens, spec, wav, ...) whose .records[i][0] is the clip's path.

    The pod's patched toolkit returns a fourth item, the speaker id; anything after the first
    three is dropped (v3 is single-speaker)."""

    def __init__(self, dataset, table: PitchTable) -> None:
        self.dataset, self.table = dataset, table
        missing = [str(path) for path, *_ in dataset.records if slug_of(Path(path).name) not in table._tracks]
        if missing:
            raise KeyError(f"{len(missing)} clips have no pitch track, e.g. {missing[:3]}")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        tokens, spec, wav = self.dataset[index][:3]
        return tokens, spec, wav, self.table.track(Path(self.dataset.records[index][0]).name, spec.shape[-1])


def collate_with_pitch(rows: Iterable[tuple]) -> tuple[torch.Tensor, ...]:
    """The toolkit's TTSCollate (sorted by spectrogram length, longest first) plus f0 [b, t_y]."""
    rows = sorted(rows, key=lambda row: row[1].shape[-1], reverse=True)
    text_lengths = torch.tensor([r[0].numel() for r in rows], dtype=torch.long)
    spec_lengths = torch.tensor([r[1].shape[-1] for r in rows], dtype=torch.long)
    wav_lengths = torch.tensor([r[2].shape[-1] for r in rows], dtype=torch.long)
    text = torch.zeros(len(rows), int(text_lengths.max()), dtype=torch.long)
    spec = torch.zeros(len(rows), rows[0][1].shape[0], int(spec_lengths.max()))
    wav = torch.zeros(len(rows), 1, int(wav_lengths.max()))
    f0 = torch.zeros(len(rows), int(spec_lengths.max()))
    for i, (tokens, spectrum, waveform, pitch) in enumerate(rows):
        text[i, : tokens.numel()] = tokens
        spec[i, :, : spectrum.shape[-1]] = spectrum
        wav[i, :, : waveform.shape[-1]] = waveform
        f0[i, : pitch.numel()] = pitch
    return text, text_lengths, spec, spec_lengths, wav, wav_lengths, f0
