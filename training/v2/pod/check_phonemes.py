"""Compare the toolkit's prepared phonemes with the reference made on Windows.

    python check_phonemes.py PREPARED_DIR REFERENCE_JSON MANIFEST_JSONL

Prepared rows name their audio after the source clip, so each row is matched by
the clip number in its file name. Exits non-zero on any difference, and unless
every manifest row was prepared.
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path


def canonical(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def clip_id(prepared_audio: str, known: set[str]) -> str:
    # The toolkit names prepared audio "{row index:07d}-{source stem}-{hash}.wav".
    parts = Path(prepared_audio).stem.split("-")
    if len(parts) != 3 or parts[1] not in known:
        raise ValueError(f"cannot tell which clip {prepared_audio!r} came from")
    return parts[1]


def main(prepared: Path, reference_path: Path, manifest: Path) -> None:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    known = set(reference)
    expected = sum(1 for line in manifest.read_text(encoding="utf-8").splitlines() if line)
    rows = []
    for split in ("train.jsonl", "validation.jsonl"):
        rows += [json.loads(line) for line in (prepared / split).read_text(encoding="utf-8").splitlines() if line]
    differences = []
    for row in rows:
        key = clip_id(row["audio"], known)
        if canonical(row["phonemes"]) != reference[key]:
            differences.append((key, reference[key], row["phonemes"]))
    print(f"compared {len(rows)} prepared rows (manifest has {expected}), {len(differences)} differ")
    for key, windows, pod in differences[:5]:
        print(f"  {key}\n    windows: {windows}\n    pod:     {pod}")
    if differences or len(rows) != expected:
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
