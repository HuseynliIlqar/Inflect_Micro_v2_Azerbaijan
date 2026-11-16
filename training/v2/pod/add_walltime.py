"""Add a wall-clock time to every metrics.jsonl row, so the monitor can show speed and time left.

    python add_walltime.py /workspace/Inflect/finetune/inflect_finetune/training.py

Run after multispeaker.patch. Refuses to edit unless each anchor occurs exactly once.
"""
from __future__ import annotations

import sys
from pathlib import Path

IMPORT_OLD = "import random\n"
IMPORT_NEW = "import random\nimport time\n"
ROW_OLD = '            metrics = {\n                "step": state.step,\n'
ROW_NEW = ROW_OLD + '                "wall_time": round(time.time(), 3),\n'


def main(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    if '"wall_time"' in source:
        print("wall_time already present")
        return
    for anchor in (IMPORT_OLD, ROW_OLD):
        count = source.count(anchor)
        if count != 1:
            sys.exit(f"anchor found {count} times, expected once: {anchor!r}")
    patched = source.replace(IMPORT_OLD, IMPORT_NEW).replace(ROW_OLD, ROW_NEW)
    path.write_text(patched, encoding="utf-8")
    print("wall_time added")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
