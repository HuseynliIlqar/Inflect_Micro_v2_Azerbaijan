"""Convert the Hugging Face parquet TTS dataset into a toolkit manifest.

`inflect-adapt prepare` aborts the entire run on the first clip it rejects, so
every filter that would reject a row has to be applied here instead. Rows that
fall outside the accepted window are dropped and counted, never passed on.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

DEFAULT_REPO_ID = "ughurabbasov/azerbaijani-tts-dataset"

#: Speaker labels become ids by ALPHABETICAL rank: `prepare` does
#: ``sorted({row.speaker ...})`` and a label's position in that sorted list is
#: the id baked into ``emb_g``. A label that sorts before an existing one
#: therefore renumbers every voice and invalidates an already-trained embedding
#: table. The zero-padded enrolment number keeps sort order equal to enrolment
#: order, so a voice added later always takes the next id instead of stealing one.
SPEAKER_LABEL_PATTERN = r"^[a-z]{2}-[0-9]{2}-[a-z0-9-]+$"
DEFAULT_SPEAKER = "az-01-main"

TARGET_SAMPLE_RATE = 24_000
TARGET_SUBTYPE = "PCM_16"

#: Clips outside this window are dropped; see docs/DATASET.md for the reasoning.
MIN_SECONDS = 1.0
MAX_SECONDS = 14.0
#: Fraction of samples at full scale above which a clip is treated as clipped.
MAX_CLIPPED_FRACTION = 0.001

#: Brackets are never spoken, and eSpeak passes them through as phoneme
#: symbols the base checkpoint has no embedding row for. Two transcripts in
#: the corpus carry them ("kukurd (S) elementi"), which is far too few to
#: teach a fresh row, so they are removed and the text inside is kept.
SILENT_BRACKETS = "()[]{}"


@dataclass
class ConversionStats:
    seen: int = 0
    written: int = 0
    total_seconds: float = 0.0
    dropped_short: int = 0
    dropped_long: int = 0
    dropped_clipped: int = 0
    dropped_unreadable: int = 0
    dropped_empty_text: int = 0
    dropped_duplicate_text: int = 0
    recoded: int = 0
    with_digits: int = 0
    examples: list[str] = field(default_factory=list)

    @property
    def hours(self) -> float:
        return self.total_seconds / 3600.0

    @property
    def dropped(self) -> int:
        return (
            self.dropped_short
            + self.dropped_long
            + self.dropped_clipped
            + self.dropped_unreadable
            + self.dropped_empty_text
            + self.dropped_duplicate_text
        )

    def render(self) -> str:
        lines = [
            f"rows seen           : {self.seen}",
            f"clips written       : {self.written}",
            f"audio hours         : {self.hours:.2f}",
            f"dropped (total)     : {self.dropped}",
            f"  too short (<{MIN_SECONDS}s)  : {self.dropped_short}",
            f"  too long (>{MAX_SECONDS}s)  : {self.dropped_long}",
            f"  clipped            : {self.dropped_clipped}",
            f"  unreadable         : {self.dropped_unreadable}",
            f"  empty transcript   : {self.dropped_empty_text}",
            f"  duplicate text     : {self.dropped_duplicate_text}",
            f"re-encoded          : {self.recoded}",
            f"transcripts w/ digits: {self.with_digits}",
        ]
        return "\n".join(lines)


def shard_names(count: int = 10) -> tuple[str, ...]:
    return tuple(f"data/train-{index:05d}-of-{count:05d}.parquet" for index in range(count))


def iter_rows(shard_paths: list[Path], batch_size: int = 32) -> Iterator[tuple[bytes, str]]:
    """Yield (wav_bytes, transcript) one row at a time, never loading a full shard."""
    import pyarrow.parquet as pq

    for path in shard_paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=["audio", "text"]):
            for row in batch.to_pylist():
                audio = row.get("audio") or {}
                payload = audio.get("bytes") if isinstance(audio, dict) else None
                if payload:
                    yield payload, str(row.get("text") or "")


def _strip_brackets(text: str) -> str:
    """Drop bracket characters, keeping whatever they enclosed."""
    return text.translate({ord(char): " " for char in SILENT_BRACKETS})


def _write_clip(payload: bytes, destination: Path) -> tuple[float, bool, float]:
    """Persist one clip. Returns (seconds, was_recoded, clipped_fraction)."""
    import numpy as np
    import soundfile as sf

    handle = io.BytesIO(payload)
    info = sf.info(handle)
    handle.seek(0)
    samples, rate = sf.read(handle, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    clipped_fraction = float(np.mean(np.abs(samples) >= 0.999)) if samples.size else 1.0
    seconds = samples.size / float(rate or TARGET_SAMPLE_RATE)

    already_canonical = (
        rate == TARGET_SAMPLE_RATE and info.channels == 1 and info.subtype == TARGET_SUBTYPE
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if already_canonical:
        # Byte-identical copy: no decode/encode round trip, no quality loss.
        destination.write_bytes(payload)
        return seconds, False, clipped_fraction

    sf.write(destination, samples, TARGET_SAMPLE_RATE, format="WAV", subtype=TARGET_SUBTYPE)
    return seconds, True, clipped_fraction


def validate_speaker_label(speaker: str) -> str:
    """Reject labels whose sort position is not stable as voices are added."""
    if not re.match(SPEAKER_LABEL_PATTERN, speaker):
        raise ValueError(
            f"Speaker label {speaker!r} does not match {SPEAKER_LABEL_PATTERN}. "
            "Use a zero-padded enrolment number, e.g. 'az-01-main', 'az-02-aysel'. "
            "Ids follow alphabetical order, so an unnumbered label added later can "
            "renumber the voices an existing checkpoint was trained with."
        )
    return speaker


def existing_speakers(output_dir: Path) -> dict[str, int]:
    """Voices already present in ``output_dir`` and how many clips each has."""
    manifest_path = Path(output_dir) / "metadata.jsonl"
    counts: dict[str, int] = {}
    if not manifest_path.is_file():
        return counts
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        speaker = json.loads(line).get("speaker")
        if speaker:
            counts[speaker] = counts.get(speaker, 0) + 1
    return counts


def convert(
    shard_paths: list[Path],
    output_dir: Path,
    *,
    speaker: str = DEFAULT_SPEAKER,
    min_seconds: float = MIN_SECONDS,
    max_seconds: float = MAX_SECONDS,
    max_clipped_fraction: float = MAX_CLIPPED_FRACTION,
    limit: int | None = None,
    append: bool = False,
) -> ConversionStats:
    """Write ``audio/<speaker>/*.wav`` plus ``metadata.jsonl`` under ``output_dir``.

    Clips live under a per-speaker directory and are numbered within that
    speaker, so enrolling another voice with ``append=True`` can never collide
    with or renumber the clips already written.
    """
    if min_seconds >= max_seconds:
        raise ValueError("min_seconds must be below max_seconds")
    validate_speaker_label(speaker)
    output_dir = Path(output_dir)
    audio_dir = output_dir / "audio" / speaker
    audio_dir.mkdir(parents=True, exist_ok=True)

    present = existing_speakers(output_dir)
    if present and not append:
        raise FileExistsError(
            f"{output_dir / 'metadata.jsonl'} already holds {sum(present.values())} "
            f"row(s) for {sorted(present)}. Pass append=True to enrol another voice, "
            "or choose a new output directory."
        )
    stats = ConversionStats()
    seen_text: set[str] = set()
    # Numbering continues after whatever this speaker already has on disk.
    index = present.get(speaker, 0)
    manifest_path = output_dir / "metadata.jsonl"
    with manifest_path.open("a" if append else "w", encoding="utf-8") as manifest:
        for payload, text in iter_rows(shard_paths):
            stats.seen += 1
            if limit is not None and stats.written >= limit:
                break
            cleaned = " ".join(_strip_brackets(text).split())
            if not cleaned:
                stats.dropped_empty_text += 1
                continue
            key = cleaned.lower()
            if key in seen_text:
                stats.dropped_duplicate_text += 1
                continue

            name = f"{index:06d}.wav"
            destination = audio_dir / name
            try:
                seconds, recoded, clipped_fraction = _write_clip(payload, destination)
            except Exception:  # noqa: BLE001 - a bad clip must not stop the sweep
                destination.unlink(missing_ok=True)
                stats.dropped_unreadable += 1
                continue

            reject = None
            if seconds < min_seconds:
                reject, stats.dropped_short = "short", stats.dropped_short + 1
            elif seconds > max_seconds:
                reject, stats.dropped_long = "long", stats.dropped_long + 1
            elif clipped_fraction > max_clipped_fraction:
                reject, stats.dropped_clipped = "clipped", stats.dropped_clipped + 1
            if reject:
                destination.unlink(missing_ok=True)
                continue

            seen_text.add(key)
            manifest.write(
                json.dumps(
                    {
                        "audio": f"audio/{speaker}/{name}",
                        "text": cleaned,
                        "speaker": speaker,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            index += 1
            stats.written += 1
            stats.total_seconds += seconds
            stats.recoded += int(recoded)
            if any(char.isdigit() for char in cleaned):
                stats.with_digits += 1
                if len(stats.examples) < 5:
                    stats.examples.append(cleaned)
    return stats
