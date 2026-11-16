"""Decode the uploaded FLAC clips back to 16-bit WAV and check every sample.

    python decode_verify.py UPLOAD_DIR DATASET_DIR

UPLOAD_DIR holds flac/, metadata.jsonl and pcm_checksums.json from pack_flac.py.
DATASET_DIR receives metadata.jsonl and audio/az-01-main/NNNNNN.wav, the layout
the manifest refers to. Any mismatch stops with a non-zero exit.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 24_000


def decode_one(job: tuple[str, str, int, str]) -> str | None:
    source, destination, frames, expected = job
    samples, rate = sf.read(source, dtype="int16")
    digest = hashlib.sha256(np.ascontiguousarray(samples, dtype="<i2").tobytes()).hexdigest()
    if rate != SAMPLE_RATE or samples.shape[0] != frames or digest != expected:
        return f"{source}: rate {rate}, frames {samples.shape[0]} (expected {frames}), checksum differs: {digest != expected}"
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, samples, rate, format="WAV", subtype="PCM_16")
    return None


def main(upload: Path, dataset: Path) -> None:
    rows = [json.loads(line) for line in (upload / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if line]
    checks = {item["file"]: item for item in json.loads((upload / "pcm_checksums.json").read_text(encoding="utf-8"))}
    if len(rows) != len(checks):
        sys.exit(f"metadata has {len(rows)} rows but the checksum list has {len(checks)}")
    jobs = []
    for row in rows:
        name = Path(row["audio"]).stem + ".flac"
        check = checks[name]
        jobs.append((str(upload / "flac" / name), str(dataset / row["audio"]), check["frames"], check["pcm_sha256"]))
    with ProcessPoolExecutor() as pool:
        problems = [problem for problem in pool.map(decode_one, jobs, chunksize=64) if problem]
    if problems:
        sys.exit(f"{len(problems)} clips failed the check, first: {problems[0]}")
    shutil.copyfile(upload / "metadata.jsonl", dataset / "metadata.jsonl")
    hours = sum(check["frames"] for check in checks.values()) / SAMPLE_RATE / 3600
    print(f"decoded and verified {len(jobs)} clips, {hours:.2f} h")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
