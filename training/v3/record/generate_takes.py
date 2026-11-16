"""Read dataset-v2 sentences with the teacher TTS in the voice of ref_data, one folder per take.

    python generate_takes.py --take A --out /workspace/rec/takes [--ids ids.txt]
                             [--config '{"class_temperature": 0.5}'] [--seed 1]

The teacher is named by exported environment variables, not by this file
(setup.sh installs it from TEACHER_PIP):
  TEACHER_PACKAGE       the Python module that provides it
  TEACHER_CLASS         its model class (from_pretrained, create_voice_clone_prompt, generate)
  TEACHER_CONFIG_CLASS  its generation-config class
  TEACHER_MODEL         the hub id of the Azerbaijani checkpoint

Take A uses v1's exact settings (D:/ktts/teacher/kit_corpus/generate_corpus.py).
Other takes override them with --config. Finished clips are skipped, so a rerun resumes.
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf
import torch

KIT = Path(__file__).resolve().parent
TEACHER_VARIABLES = ("TEACHER_PACKAGE", "TEACHER_CLASS", "TEACHER_CONFIG_CLASS", "TEACHER_MODEL")
SAMPLE_RATE = 24000
SPEED = 0.9
V1_SETTINGS = {
    "num_step": 32, "guidance_scale": 2.0, "position_temperature": 0.0, "class_temperature": 0.0,
    "denoise": True, "preprocess_prompt": True, "postprocess_output": False,
}
MAX_FAILED_SHARE = 0.01
LOG = logging.getLogger("generate_takes")


def teacher_from_environment() -> tuple[type, type, str]:
    """The teacher's model class, generation-config class and checkpoint id, as the environment names them."""
    missing = [name for name in TEACHER_VARIABLES if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"set {', '.join(missing)} first (see this file's docstring)")
    module = importlib.import_module(os.environ["TEACHER_PACKAGE"])
    return (getattr(module, os.environ["TEACHER_CLASS"]), getattr(module, os.environ["TEACHER_CONFIG_CLASS"]),
            os.environ["TEACHER_MODEL"])


def read_rows(ids_file: Path | None) -> list[dict]:
    rows = [json.loads(line) for line in (KIT / "sentences_v2.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if ids_file is None:
        return rows
    wanted = set(ids_file.read_text(encoding="utf-8").split())
    return [row for row in rows if row["id"] in wanted]


def batch_limit(characters: int) -> int:
    return 24 if characters <= 100 else 12 if characters <= 250 else 6


def batches(rows: list[dict]) -> Iterator[list[dict]]:
    """Batches of similar length, so short sentences do not wait for paragraphs."""
    current: list[dict] = []
    for row in sorted(rows, key=lambda r: len(r["text"])):
        if current and len(current) >= batch_limit(len(row["text"])):
            yield current
            current = []
        current.append(row)
    if current:
        yield current


def write_clip(path: Path, audio: np.ndarray) -> float:
    temporary = path.with_name(path.stem + ".part.wav")
    sf.write(temporary, audio, SAMPLE_RATE, subtype="PCM_16")
    temporary.replace(path)
    return len(audio) / SAMPLE_RATE


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--take", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ids", type=Path)
    parser.add_argument("--config", default="{}", help="JSON overrides of the v1 generation settings.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    teacher_class, config_class, model_id = teacher_from_environment()

    folder = args.out / args.take
    folder.mkdir(parents=True, exist_ok=True)
    settings = {**V1_SETTINGS, **json.loads(args.config)}
    (folder / "settings.json").write_text(json.dumps({"settings": settings, "seed": args.seed, "speed": SPEED}, indent=1))
    rows = [row for row in read_rows(args.ids) if not (folder / f"{row['id']}.wav").exists()]
    LOG.info("take %s: %d clips to do, settings %s", args.take, len(rows), settings)

    model = teacher_class.from_pretrained(model_id, device_map="cuda", dtype=torch.float16)
    reference = json.loads((KIT / "refs" / "refs.json").read_text(encoding="utf-8"))
    prompt = model.create_voice_clone_prompt(ref_audio=str(KIT / "refs" / "ref_data.wav"), ref_text=reference["ref_data"],
                                             preprocess_prompt=True)
    config = config_class(**settings)
    done = failed = 0
    audio_seconds = busy = 0.0

    def run(batch: list[dict]) -> None:
        nonlocal done, failed, audio_seconds, busy
        started = time.perf_counter()
        try:
            audios = model.generate(text=[r["text"] for r in batch], language="az", voice_clone_prompt=prompt,
                                    speed=SPEED, generation_config=config)
        except Exception as error:  # noqa: BLE001 - one bad batch must not stop a paid run
            torch.cuda.empty_cache()
            if len(batch) == 1:
                failed += 1
                LOG.error("failed %s: %s", batch[0]["id"], error)
                return
            LOG.warning("batch of %d failed (%s); retrying one by one", len(batch), error)
            for row in batch:
                run([row])
            return
        busy += time.perf_counter() - started
        for row, audio in zip(batch, audios, strict=True):
            audio_seconds += write_clip(folder / f"{row['id']}.wav", np.asarray(audio, dtype=np.float32))
            done += 1

    for number, batch in enumerate(batches(rows), start=1):
        torch.manual_seed(args.seed * 1_000_003 + number)
        run(batch)
        if number % 20 == 1:
            LOG.info("take %s batch %d: %d done, %d failed, audio %.0f s, rtf %.3f", args.take, number, done, failed,
                     audio_seconds, busy / max(audio_seconds, 1e-9))
        (folder / "progress.json").write_text(json.dumps({"done": done, "todo": len(rows), "failed": failed,
                                                          "audio_seconds": round(audio_seconds, 1)}))
    if failed > MAX_FAILED_SHARE * max(len(rows), 1):
        raise SystemExit(f"{failed} of {len(rows)} clips failed")
    LOG.info("take %s finished: %d clips, %.0f s audio, rtf %.3f", args.take, done, audio_seconds, busy / max(audio_seconds, 1e-9))


if __name__ == "__main__":
    main()
