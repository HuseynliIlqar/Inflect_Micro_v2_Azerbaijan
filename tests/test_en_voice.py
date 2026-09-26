"""The English checkpoint's presence check.

Nothing here loads a model: `checkpoint_status` only looks at the file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aztts import DEFAULT_EN_MODEL_DIR, EnVoice, ModelNotFoundError
from aztts.en_voice import checkpoint_status

POINTER = (
    b"version https://git-lfs.github.com/spec/v1\n"
    b"oid sha256:3eede065c9ccfa88ade0a5a9a5c23de34afcbbb32213e59aad44d5cf100fdee8\n"
    b"size 37529995\n"
)


def test_the_english_weights_ship_with_the_repository() -> None:
    assert DEFAULT_EN_MODEL_DIR.name == "model-en"
    assert (DEFAULT_EN_MODEL_DIR / "model.pth").is_file()
    assert (DEFAULT_EN_MODEL_DIR / "config.json").is_file()


def test_the_shipped_checkpoint_is_the_real_file() -> None:
    assert checkpoint_status() == "ok"


def test_an_empty_directory_is_missing(tmp_path: Path) -> None:
    assert checkpoint_status(tmp_path) == "missing"


def test_a_git_lfs_pointer_is_recognised(tmp_path: Path) -> None:
    (tmp_path / "model.pth").write_bytes(POINTER)
    assert checkpoint_status(tmp_path) == "pointer"


def test_a_real_checkpoint_is_not_mistaken_for_a_pointer(tmp_path: Path) -> None:
    (tmp_path / "model.pth").write_bytes(b"PK\x03\x04" + b"\x00" * 8192)
    assert checkpoint_status(tmp_path) == "ok"


def test_a_missing_directory_says_where_to_look(tmp_path: Path) -> None:
    with pytest.raises(ModelNotFoundError) as error:
        EnVoice(tmp_path)
    assert "model-en" in str(error.value)


def test_a_pointer_says_to_run_git_lfs(tmp_path: Path) -> None:
    (tmp_path / "model.pth").write_bytes(POINTER)
    with pytest.raises(ModelNotFoundError) as error:
        EnVoice(tmp_path)
    assert "git lfs pull" in str(error.value)
