"""Are two takes of the same sentences identical? Decides whether extra takes need another
source of variety than the seed.

    python determinism.py /workspace/rec/takes/A /workspace/rec/takes/A_seed1
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def main() -> None:
    first, second = Path(sys.argv[1]), Path(sys.argv[2])
    same = different = 0
    for path in sorted(second.glob("*.wav")):
        other = first / path.name
        if not other.exists():
            continue
        a, _ = sf.read(other, dtype="int16")
        b, _ = sf.read(path, dtype="int16")
        if a.shape == b.shape and np.array_equal(a, b):
            same += 1
        else:
            different += 1
    report = {"compared": same + different, "identical": same, "different": different}
    (second.parent / "determinism.json").write_text(json.dumps(report))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
