"""Dataset rows from a selection, without audio."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finalize import SPEAKER, dataset_rows  # noqa: E402

ROWS = {"A1_00001": {"id": "A1_00001", "cat": "A1", "text": "Gəlirsənmi?"},
        "D1_00002": {"id": "D1_00002", "cat": "D1", "text": "Yağış yağır."}}
SELECTION = [
    {"id": "D1_00002", "cat": "D1", "take": "A", "margin": 1.2, "end_drop_st": None, "cer": 0.0},
    {"id": "A1_00001", "cat": "A1", "take": "C", "margin": 0.4, "end_drop_st": 9.1, "cer": 0.0},
    {"id": "B_00003", "cat": "B", "dropped": {"A": "misread"}},
]


def test_rows_follow_v1_format_and_skip_dropped() -> None:
    metadata, clip_map = dataset_rows(SELECTION, ROWS)
    assert [m["audio"] for m in metadata] == ["audio/az-01-main/v2_A1_00001.wav", "audio/az-01-main/v2_D1_00002.wav"]
    assert metadata[0] == {"audio": "audio/az-01-main/v2_A1_00001.wav", "text": "Gəlirsənmi?", "speaker": SPEAKER}
    assert clip_map[0]["take"] == "C" and clip_map[0]["source"] == "takes/C/A1_00001.wav"
    assert all(set(m) == {"audio", "text", "speaker"} for m in metadata)
