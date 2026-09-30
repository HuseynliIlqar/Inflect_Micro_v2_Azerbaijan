"""The browser runtime fetcher: integrity is checked, and only the needed files land.

No network here -- a small tarball is built in memory and handed to the same
functions the tool runs on the real npm package.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "fetch_web_runtime", Path(__file__).resolve().parent.parent / "tools" / "fetch_web_runtime.py"
)
fetch_web_runtime = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fetch_web_runtime)

FILES = fetch_web_runtime.FILES
IntegrityError = fetch_web_runtime.IntegrityError
extract = fetch_web_runtime.extract
integrity_of = fetch_web_runtime.integrity_of
verify = fetch_web_runtime.verify


def _tarball(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def _package() -> bytes:
    members = {f"package/dist/{name}": name.encode() for name in FILES}
    members["package/dist/ort.all.min.mjs"] = b"not needed"
    members["package/package.json"] = b"{}"
    return _tarball(members)


def test_integrity_is_npm_style_sha512() -> None:
    data = b"abc"
    expected = "sha512-" + base64.b64encode(hashlib.sha512(data).digest()).decode()
    assert integrity_of(data) == expected


def test_a_matching_tarball_passes() -> None:
    data = _package()
    verify(data, integrity_of(data))


def test_a_tampered_tarball_is_refused() -> None:
    data = _package()
    with pytest.raises(IntegrityError):
        verify(data + b"x", integrity_of(data))


def test_only_the_runtime_files_are_extracted(tmp_path: Path) -> None:
    written = extract(_package(), tmp_path)
    assert sorted(path.name for path in written) == sorted(FILES)
    assert not (tmp_path / "ort.all.min.mjs").exists()
    assert (tmp_path / "ort.webgpu.min.mjs").read_bytes() == b"ort.webgpu.min.mjs"


def test_a_missing_file_is_an_error(tmp_path: Path) -> None:
    incomplete = _tarball({"package/dist/ort.webgpu.min.mjs": b"x"})
    with pytest.raises(FileNotFoundError):
        extract(incomplete, tmp_path)


def test_a_path_outside_the_target_is_never_written(tmp_path: Path) -> None:
    hostile = _tarball({"../../evil.mjs": b"x", **{f"package/dist/{n}": b"x" for n in FILES}})
    extract(hostile, tmp_path / "out")
    assert not (tmp_path / "evil.mjs").exists()
