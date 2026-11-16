"""Read what a training run leaves on disk, for the monitor page.

- ``metrics.jsonl``: one row per step (losses, stage, VRAM, ``wall_time``), read incrementally;
- ``validation/``: one ``step-XXXXXXXX.wav`` per validation (a single voice) with a ``step-XXXXXXXX.json``, or
  (v3) a ``step-XXXXXXXX/`` folder of ``NN.wav`` clips with ``summary.json`` (predicted and true pitch range);
- ``listen/``: ``base/`` and ``step-XXXXXXXX/`` folders of ``NN.wav``, the same sentences read by the old Avaz
  and by each checkpoint (``pod/listen.py``), with ``sentences.json``.
"""

from __future__ import annotations

import json
import re
import statistics
import threading
from pathlib import Path
from typing import Any

LOSSES = ("loss_mel", "loss_kl", "loss_duration", "loss_g", "loss_d")
# Logged only by the decoder run (polish/train_polish.py): per-discriminator losses and the new reconstruction terms.
OPTIONAL = ("d_mpd", "d_mrd", "d_cqtd", "g_mpd", "g_mrd", "g_cqtd", "loss_mel_ms", "loss_stft", "loss_adv", "loss_fm",
            # v3 (training/v3/model/trainer_v3.py): the pitch part and the two prosody discriminators
            "loss_pitch_l1", "loss_voicing", "loss_pitch_adv", "loss_duration_adv", "loss_pitch_range")
VOICE = "Avaz"
_CLIP = re.compile(r"^step-(\d{8})\.wav$")
_STEP_DIR = re.compile(r"^step-(\d{8})$")
_V3_CLIP = re.compile(r"^(\d{2})\.wav$")
_LISTEN_STEP = re.compile(r"^step-(\d{8})$")
_LISTEN_CLIP = re.compile(r"^(\d{2})\.wav$")


class MetricsReader:
    """Reads ``metrics.jsonl`` from where it stopped last time; starts over if the file shrank (a new run).

    The page asks for status and history at the same moment and the server answers in threads, so a lock keeps two
    reads from both advancing the offset (which then points into the middle of a line). On the pod's network disk a
    freshly appended tail can briefly read as NUL bytes: reading stops there and picks it up next time. A line that
    is complete but not JSON is skipped rather than blocking every later read.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._rows: list[dict[str, Any]] = []
        self._offset = 0
        self._lock = threading.Lock()

    def rows(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        with self._lock:
            if self.path.stat().st_size < self._offset:
                self._rows, self._offset = [], 0
            with self.path.open("rb") as handle:
                handle.seek(self._offset)
                chunk = handle.read()
            new_rows, consumed = _parse_lines(chunk)
            self._rows = self._rows + new_rows  # one copy per read, not per line: the file reaches 100k rows
            self._offset += consumed
            return list(self._rows)


def _parse_lines(chunk: bytes) -> tuple[list[dict[str, Any]], int]:
    """The JSON rows in ``chunk`` and how many bytes they cover: complete lines only, up to any NUL-filled line."""
    rows: list[dict[str, Any]] = []
    consumed = 0
    for line in chunk[: chunk.rfind(b"\n") + 1].splitlines(keepends=True):
        if b"\x00" in line:
            break
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        consumed += len(line)
    return rows, consumed


def downsample(rows: list[dict[str, Any]], points: int, from_step: int = 0) -> list[dict[str, Any]]:
    """At most ``points`` points over the steps from ``from_step`` on: per group of consecutive rows the mean of each
    loss with its min and max (the spread the chart draws as a band), the peak VRAM, and the seconds per step."""
    rows = [r for r in rows if r["step"] >= from_step]
    if not rows:
        return []
    size = max(1, -(-len(rows) // points))
    out = []
    for start in range(0, len(rows), size):
        group = rows[start : start + size]
        point = {"step": group[-1]["step"], "stage": group[-1]["stage"], "epoch": group[-1].get("epoch"),
                 "peak_reserved_gb": max(r.get("peak_reserved_gb", 0.0) for r in group)}
        for key in LOSSES + OPTIONAL:
            values = [r[key] for r in group if key in r]
            if not values:
                continue
            point[key] = round(statistics.fmean(values), 4)
            point[f"{key}_min"] = round(min(values), 4)
            point[f"{key}_max"] = round(max(values), 4)
        timed = [r for r in group if "wall_time" in r]
        if len(timed) >= 2 and timed[-1]["step"] > timed[0]["step"]:
            point["seconds_per_step"] = round(
                (timed[-1]["wall_time"] - timed[0]["wall_time"]) / (timed[-1]["step"] - timed[0]["step"]), 3)
        out.append(point)
    return out


def steps_per_epoch(rows: list[dict[str, Any]]) -> int | None:
    """Steps in one pass over the data, from where the logged epoch number changes (None before the second pass)."""
    starts: dict[int, int] = {}
    for r in rows:
        starts.setdefault(r.get("epoch", 0), r["step"])
    firsts = [starts[e] for e in sorted(starts)]
    if len(firsts) >= 3:  # whole epochs only: the first one may have started before the log did (a resume)
        gaps = [b - a for a, b in zip(firsts[1:], firsts[2:])]
    elif len(firsts) == 2 and firsts[0] <= 1:  # the run started at step 1, so its first epoch is whole too
        gaps = [firsts[1] - firsts[0]]
    else:
        return None
    return int(statistics.median(gaps))


def progress(rows: list[dict[str, Any]], max_steps: int, price_per_hour: float, window: int = 200) -> dict[str, Any]:
    """Where the run is: step, epoch, stage, median seconds per step over the last ``window`` steps, ETA and the GPU
    cost of the time the trainer has run (from the first to the last logged step)."""
    if not rows:
        return {"step": 0, "max_steps": max_steps, "percent": 0.0, "stage": None, "seconds_per_step": None,
                "eta_seconds": None, "cost_so_far": 0.0, "latest": None, "epoch": 0, "steps_per_epoch": None,
                "total_epochs": None}
    per_epoch = steps_per_epoch(rows)
    recent = rows[-window:]
    gaps = [b["wall_time"] - a["wall_time"] for a, b in zip(recent, recent[1:])
            if "wall_time" in a and "wall_time" in b and b["wall_time"] > a["wall_time"]]
    per_step = round(statistics.median(gaps), 4) if gaps else None
    step = rows[-1]["step"]
    elapsed = rows[-1].get("wall_time", 0.0) - rows[0].get("wall_time", 0.0)
    return {
        "step": step, "max_steps": max_steps, "percent": round(100 * step / max_steps, 2), "stage": rows[-1]["stage"],
        "epoch": rows[-1].get("epoch", 0), "steps_per_epoch": per_epoch,
        "total_epochs": round(max_steps / per_epoch, 1) if per_epoch else None,
        "seconds_per_step": per_step,
        "eta_seconds": round((max_steps - step) * per_step) if per_step else None,
        "cost_so_far": round(elapsed / 3600 * price_per_hour, 3),
        "latest": _latest(rows[-50:], LOSSES + ("loss_pitch_l1",))
                  | {"peak_reserved_gb": rows[-1].get("peak_reserved_gb", rows[-1].get("peak_memory_gb"))},
    }


def _latest(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    """Mean of each key over the rows that log it (v3's phases log different losses), None if none do."""
    out = {}
    for key in keys:
        values = [r[key] for r in rows if key in r]
        out[key] = round(statistics.fmean(values), 4) if values else None
    return out


def validation_steps(folder: Path, limit: int = 3) -> list[dict[str, Any]]:
    """The newest ``limit`` validation steps with their clips (label, file, seconds; v3 adds the folder and the
    predicted and true pitch range of each clip)."""
    if not folder.is_dir():
        return []
    steps: dict[int, list[dict[str, Any]]] = {}
    for path in folder.glob("step-*.wav"):
        if m := _CLIP.match(path.name):
            info_path = folder / f"step-{int(m.group(1)):08d}.json"
            info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
            steps[int(m.group(1))] = [{"speaker": VOICE, "file": path.name, "seconds": info.get("seconds")}]
    for path in folder.iterdir():
        if path.is_dir() and (m := _STEP_DIR.match(path.name)):
            summary_path = path / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else []
            by_clip = {row.get("clip"): row for row in summary}
            clips = []
            for clip in sorted(p for p in path.glob("*.wav") if _V3_CLIP.match(p.name)):
                row = by_clip.get(int(clip.stem), {})
                clips.append({"speaker": f"Cümlə {int(clip.stem) + 1}", "file": clip.name, "folder": path.name,
                              "seconds": row.get("seconds"), "range_pred": row.get("range_st_pred"),
                              "range_true": row.get("range_st_true")})
            if clips:
                steps[int(m.group(1))] = clips
    return [{"step": step, "clips": steps[step]} for step in sorted(steps, reverse=True)[:limit]]


def validation_extras(folder: Path) -> dict[str, Any]:
    """The validation sentence and the clips to compare against: the old Avaz reading it ("before") and the
    dataset recording ("target"), when ``pod/validation_before.py`` has written them."""
    sentence_path = folder / "sentence.json"
    sentence = json.loads(sentence_path.read_text(encoding="utf-8")) if sentence_path.is_file() else {}
    return {"text": sentence.get("text"),
            "before": "before.wav" if (folder / "before.wav").is_file() else None,
            "target": "target.wav" if (folder / "target.wav").is_file() else None}


def listen_versions(folder: Path) -> dict[str, Any]:
    """The listening sentences and every complete version that read them: newest checkpoint first, the old
    Avaz (``base``) last. Folders still being written end in ``.partial`` and are not listed."""
    sentences_path = folder / "sentences.json"
    sentences = json.loads(sentences_path.read_text(encoding="utf-8")) if sentences_path.is_file() else []
    if not folder.is_dir():
        return {"sentences": sentences, "versions": [], "reference": None}

    def clips(version: Path) -> list[str]:
        return sorted(path.name for path in version.glob("*.wav") if _LISTEN_CLIP.match(path.name))

    steps = sorted(((int(m.group(1)), path) for path in folder.iterdir()
                    if path.is_dir() and (m := _LISTEN_STEP.match(path.name))), reverse=True)
    def measured(path: Path) -> dict[str, Any] | None:
        metrics = path / "metrics.json"
        return json.loads(metrics.read_text(encoding="utf-8")) if metrics.is_file() else None

    versions = [{"name": path.name, "step": step, "files": clips(path), "metrics": measured(path)} for step, path in steps]
    if (folder / "base").is_dir():
        versions.append({"name": "base", "step": None, "files": clips(folder / "base"), "metrics": measured(folder / "base")})
    reference_path = folder / "dataset_metrics.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8")) if reference_path.is_file() else None
    return {"sentences": sentences, "versions": versions, "reference": reference}
