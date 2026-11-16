"""Read the same sentences with every new checkpoint, for the monitor's "Dinlə" section.

    python listen.py --run /workspace/runs/main [--base /workspace/base] [--once]

Writes <run>/listen/base/NN.wav once (the old Avaz with its own eSpeak frontend, the
"before"), then <run>/listen/step-XXXXXXXX/NN.wav for every exports/model-step-*.pth
(the new model with the stress frontend). The sentences are in listen_sentences.json
and none of them is in the training data. Runs on the CPU so training keeps the GPU.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

KIT = Path(__file__).resolve().parent
SEED = 0
VARIATION = 0.667
_EXPORT = re.compile(r"^model-step-(\d{8})\.pth$")


def load_engine(base: Path):
    sys.path.insert(0, str(base))
    from inference import InflectTTS  # the package's own runtime and frontend

    return InflectTTS(base, device="cpu")


def load_stress_frontend():
    spec = importlib.util.spec_from_file_location("az_stress_frontend_listen", KIT / "az_stress_frontend.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_frontend("az")


def load_weights(engine, checkpoint: Path) -> None:
    """Strict: only the training-only posterior encoder may be absent from an export."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    missing, unexpected = engine.model.load_state_dict(state, strict=False)
    stray = [key for key in missing if not key.startswith("enc_q.")]
    if unexpected or stray:
        raise RuntimeError(f"{checkpoint.name}: unexpected {unexpected[:3]}, missing {stray[:3]}")


def render(engine, folder: Path, sentences: list[str], phonemes: list[str] | None) -> None:
    staging = folder.with_name(folder.name + ".partial")
    staging.mkdir(parents=True, exist_ok=True)
    for index, text in enumerate(sentences):
        if phonemes is None:
            rate, wave = engine.synthesize(text, variation=VARIATION, seed=SEED)
        else:
            rate, wave = engine.synthesize(text, phonemes=phonemes[index], variation=VARIATION, seed=SEED)
        sf.write(staging / f"{index:02d}.wav", wave, rate)
    staging.rename(folder)  # the page only sees complete folders


def pending(run: Path, done: Path) -> list[tuple[str, Path]]:
    found = []
    for path in sorted((run / "exports").glob("model-step-*.pth")):
        match = _EXPORT.match(path.name)
        if match and not (done / f"step-{match.group(1)}").is_dir():
            found.append((f"step-{match.group(1)}", path))
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=Path("/workspace/base"))
    parser.add_argument("--prepared", type=Path, default=Path("/workspace/prepared"))
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)  # leave the CPU to the training data loaders

    sentences = json.loads((KIT / "listen_sentences.json").read_text(encoding="utf-8"))
    out = args.run / "listen"
    out.mkdir(parents=True, exist_ok=True)
    (out / "sentences.json").write_text(json.dumps(sentences, ensure_ascii=False), encoding="utf-8")
    engine = load_engine(args.base)
    if not (out / "base").is_dir():
        render(engine, out / "base", sentences, None)
        print("rendered the old Avaz", flush=True)
    frontend = load_stress_frontend()
    phonemes = [frontend.phonemize(frontend.normalize(text)) for text in sentences]
    measure_dataset(args.prepared, out)
    while True:
        for name, checkpoint in pending(args.run, out):
            time.sleep(5)  # let the trainer finish writing the file
            load_weights(engine, checkpoint)
            render(engine, out / name, sentences, phonemes)
            print(f"rendered {name}", flush=True)
        measure_versions(out)
        if args.once:
            return
        time.sleep(60)


def _summary(paths: list[Path]) -> dict:
    """Median buzz (2-4 kHz harmonicity, higher = cleaner) and ring over the given clips."""
    import statistics

    import buzz_measure as bm  # the same measurement as the earlier Avaz / teacher comparisons

    rows = [bm.measure(str(path)) for path in paths]
    finite = lambda key: [r[key] for r in rows if r[key] == r[key]]  # noqa: E731 - drop NaN
    return {"h2_4k": statistics.median(finite("h2-4k")), "h1_2k": statistics.median(finite("h1-2k")),
            "ring_share": statistics.median(finite("ring_share")), "ring_hz": statistics.median(finite("ring_hz")),
            "clips": len(rows)}


def measure_versions(out: Path) -> None:
    for folder in sorted(out.iterdir()):
        if folder.is_dir() and not folder.name.endswith(".partial") and not (folder / "metrics.json").exists():
            clips = sorted(folder.glob("[0-9][0-9].wav"))
            if clips:
                (folder / "metrics.json").write_text(json.dumps(_summary(clips)), encoding="utf-8")
                print(f"measured {folder.name}", flush=True)


def measure_dataset(prepared: Path, out: Path, count: int = 40) -> None:
    """The target: the same measurement on a fixed random sample of the training recordings."""
    import random

    target = out / "dataset_metrics.json"
    if target.exists() or not (prepared / "audio").is_dir():
        return
    clips = sorted((prepared / "audio").glob("*.wav"))
    sample = random.Random(0).sample(clips, min(count, len(clips)))
    target.write_text(json.dumps(_summary(sample)), encoding="utf-8")


if __name__ == "__main__":
    main()
