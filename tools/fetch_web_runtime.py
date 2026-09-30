"""Put the browser playground's ONNX Runtime next to the page, in web/vendor/.

    python tools/fetch_web_runtime.py            # -> web/vendor/onnxruntime-web/

The playground used to import onnxruntime-web from a CDN. WebKit -- the engine
of every browser on an iPhone or iPad -- checks each module a worker imports
against the page's Cross-Origin-Embedder-Policy and refuses a CDN's, whatever
its CORS and CORP headers say. The worker never started there, so synthesis
failed on every iPhone. Served from the page's own origin, the files carry the
Space's headers and load.

The files are not in the repository -- the .wasm is 27 MB, the same reason
web/onnx/ is not -- so this fetches them from the npm registry, checks the
tarball against the integrity npm publishes for this exact version, and
extracts only the three files the WebGPU bundle loads. Standard library only.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import sys
import tarfile
import urllib.request
from io import BytesIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aztts.console import use_utf8  # noqa: E402

use_utf8()

VERSION = "1.30.0"
TARBALL_URL = f"https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-{VERSION}.tgz"
# `npm view onnxruntime-web@1.30.0 dist.integrity`
INTEGRITY = (
    "sha512-q0y+JrrtukXSzsBWEMccVfqX25LRmosXHF+CaRJmg8pZClzcV7svNc4rKY3jL02Vb7QmRMDs1SigqR4CXAfKYQ=="
)
# What ort.webgpu.min.mjs loads: itself, and the asyncify build it imports.
FILES: tuple[str, ...] = (
    "ort.webgpu.min.mjs",
    "ort-wasm-simd-threaded.asyncify.mjs",
    "ort-wasm-simd-threaded.asyncify.wasm",
)
DEFAULT_OUT = PROJECT_ROOT / "web" / "vendor" / "onnxruntime-web"

# The package ships no licence file of its own; package.json says MIT, and this
# is the notice from the Microsoft/onnxruntime repository it is built from.
LICENSE_TEXT = """MIT License

Copyright (c) Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


class IntegrityError(ValueError):
    """The downloaded tarball is not the one npm published."""


def integrity_of(data: bytes) -> str:
    """An npm-style Subresource Integrity string: sha512 of the bytes, base64."""
    return "sha512-" + base64.b64encode(hashlib.sha512(data).digest()).decode("ascii")


def verify(data: bytes, expected: str = INTEGRITY) -> None:
    """Raise unless the bytes hash to the published integrity."""
    actual = integrity_of(data)
    if actual != expected:
        raise IntegrityError(f"tarball integrity mismatch:\n  expected {expected}\n  got      {actual}")


def extract(data: bytes, out_dir: Path) -> list[Path]:
    """Write the `FILES` from the package's dist/ into `out_dir`; nothing else.

    Members are looked up by their exact name, never extracted by the paths the
    archive claims, so an entry such as `../x` cannot write outside `out_dir`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with tarfile.open(fileobj=BytesIO(data), mode="r:gz") as archive:
        names = set(archive.getnames())
        for name in FILES:
            member = f"package/dist/{name}"
            if member not in names:
                raise FileNotFoundError(f"{member} is not in the onnxruntime-web {VERSION} package")
            source = archive.extractfile(member)
            if source is None:
                raise FileNotFoundError(f"{member} is not a regular file")
            destination = out_dir / name
            destination.write_bytes(source.read())
            written.append(destination)
    return written


def download(url: str = TARBALL_URL, timeout: float = 300.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 -- fixed https URL
        return response.read()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Where the runtime files go.")
    args = parser.parse_args(argv)

    print(f"downloading onnxruntime-web {VERSION} from the npm registry ...")
    try:
        data = download()
        verify(data)
    except (OSError, IntegrityError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"integrity ok ({len(data) / 1e6:.1f} MB tarball)")
    try:
        written = extract(data, args.out)
        (args.out / "LICENSE").write_text(LICENSE_TEXT, encoding="utf-8")
    except (OSError, tarfile.TarError) as error:
        print(f"error: could not write the runtime into {args.out}: {error}", file=sys.stderr)
        return 1
    for path in written:
        shown = path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path
        print(f"  {shown}  {path.stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
