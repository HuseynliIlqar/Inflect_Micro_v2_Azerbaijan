"""Download the Inflect-Micro-v2 weights and runtime into ./model."""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "owensong/Inflect-Micro-v2"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "base-model"

# Benchmark artefacts and card images are not needed for inference.
IGNORE_PATTERNS = ["evaluation/*", "assets/*", "third_party/*"]


def main() -> int:
    path = snapshot_download(
        REPO_ID,
        local_dir=str(MODEL_DIR),
        ignore_patterns=IGNORE_PATTERNS,
    )
    checkpoint = Path(path) / "model.pth"
    if not checkpoint.is_file():
        print(f"error: {checkpoint} is missing after download", file=sys.stderr)
        return 1
    print(f"model ready: {path}")
    print(f"checkpoint:  {checkpoint.stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
