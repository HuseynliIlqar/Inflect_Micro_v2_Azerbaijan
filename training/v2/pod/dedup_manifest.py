"""Keep the first clip of each transcript and drop exact repeats.

    python dedup_manifest.py METADATA_JSONL

The teacher TTS was run deterministically, so a sentence that occurs twice in the corpus
gave two bit-identical clips, which the toolkit refuses (the same clip could land
in both the training and the validation split). The file is rewritten in place;
the original is kept next to it as .with-duplicates.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def main(path: Path) -> None:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    seen: set[str] = set()
    kept, dropped = [], []
    for line in lines:
        row = json.loads(line)
        (dropped if row["text"] in seen else kept).append(line)
        seen.add(row["text"])
    backup = path.with_name(path.name + ".with-duplicates")
    if not backup.exists():
        shutil.copyfile(path, backup)
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(f"kept {len(kept)} rows, dropped {len(dropped)} repeats")
    for line in dropped:
        print("  dropped", json.loads(line)["audio"])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
