"""End-to-end multi-speaker smoke test on synthetic Azerbaijani data.

Builds a tiny two-voice corpus, prepares it, trains a handful of CPU steps,
exports a PyTorch package, and synthesises with each speaker id. This proves the
patched pipeline wires speaker ids all the way from the manifest to deployment;
it says nothing about voice quality.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
WORK = PROJECT_ROOT / "out" / "smoke_multispeaker"
SAMPLE_RATE = 24_000

# Distinct text per voice: the splitter refuses a corpus whose rows are all
# connected by a shared normalized transcript.
SENTENCES = {
    "voice-a": [
        "Salam, bu gun hava cox gozeldir.",
        "Evvelce penchereni bagla, sonra isigi sondur.",
        "Bakidan Genceye qeder yol iki yuz otuz kilometrdir.",
        "Seherin merkezinde yerlesen kitabxana seher doqquzda acilir.",
        "Qirmizi jaketli qiz oten il Naxcivanda yasayirdi.",
        "Usaqlar heyetde futbol oynayirdilar ve qonsu qadin baxirdi.",
    ],
    "voice-b": [
        "Professor universitetde fizika ve riyaziyyat dersi deyir.",
        "Yagis yagirdi, kuleyin sureti saatda otuz kilometr idi.",
        "Qapini ortdu, acari cibine qoydu ve pillekenle asagi dusdu.",
        "Onun sesi titreyirdi, cunki cox heyecanli idi.",
        "Sabah axsam teyyare ile Istanbula ucacagiq.",
        "Muellim lovhede yeni movzunun basligini yazdi.",
    ],
}

# Two synthetic "voices" separated by fundamental frequency.
VOICES = {"voice-a": 110.0, "voice-b": 190.0}


def synth_clip(path: Path, f0: float, seconds: float, seed: int) -> None:
    """Write a harmonic-plus-noise tone: cheap, finite, and never silent."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / SAMPLE_RATE
    signal = np.zeros_like(t)
    for harmonic in range(1, 6):
        signal += np.sin(2 * np.pi * f0 * harmonic * t) / (harmonic + 1)
    envelope = 0.5 * (1 - np.cos(2 * np.pi * np.clip(t / seconds, 0, 1)))
    signal = signal * envelope + rng.normal(0, 0.005, t.shape).astype(np.float32)
    peak = float(np.abs(signal).max()) or 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, (signal / peak * 0.7).astype(np.float32), SAMPLE_RATE)


def build_corpus(root: Path) -> Path:
    rows = []
    for voice_index, (speaker, f0) in enumerate(VOICES.items()):
        for index, text in enumerate(SENTENCES[speaker]):
            name = f"{speaker}/{index:03d}.wav"
            synth_clip(root / name, f0, 2.0 + 0.25 * index, seed=voice_index * 100 + index)
            rows.append({"audio": name, "text": text, "speaker": speaker})
    manifest = root / "metadata.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return manifest


def run(stage: str, *args: str) -> None:
    command = [str(VENV_PYTHON), "-m", "inflect_finetune", stage, *args]
    print(f"\n$ inflect-adapt {stage} ...")
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"{stage} failed with exit code {result.returncode}")
    tail = (result.stdout or "").strip().splitlines()[-3:]
    for line in tail:
        print(f"  {line}")


def main() -> int:
    import shutil

    if WORK.exists():
        shutil.rmtree(WORK)
    manifest = build_corpus(WORK / "source")
    prepared, run_dir, export_dir = WORK / "prepared", WORK / "run", WORK / "export"

    run("prepare", "--manifest", str(manifest), "--audio-root", str(WORK / "source"),
        "--language", "az", "--frontend", "espeak", "--output", str(prepared),
        "--validation-fraction", "0.2", "--min-duration-seconds", "0.5")

    dataset = json.loads((prepared / "dataset.json").read_text(encoding="utf-8"))
    print(f"  speakers in dataset.json: {dataset['speakers']}")
    assert dataset["speakers"] == sorted(VOICES), dataset["speakers"]

    run("train", "--base", str(PROJECT_ROOT / "training" / "base-model"), "--dataset", str(prepared),
        "--output", str(run_dir), "--device", "cpu", "--max-steps", "6",
        "--batch-size", "2", "--gradient-accumulation-steps", "1",
        "--checkpoint-interval", "6", "--validation-interval", "3", "--no-amp")

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    print(f"  trained config: n_speakers={config['model'].get('n_speakers')} "
          f"gin_channels={config['model'].get('gin_channels')}")

    validation = sorted((run_dir / "validation").glob("*speaker*.wav"))
    print(f"  per-speaker validation clips: {[p.name for p in validation]}")
    assert len(validation) >= 2, "expected one validation clip per speaker"

    run("export", "--checkpoint", str(run_dir / "checkpoints" / "adaptation-final.pth"),
        "--prepared-dataset", str(prepared), "--format", "pytorch",
        "--package-template", str(PROJECT_ROOT / "training" / "base-model"),
        "--output", str(export_dir))

    sys.path.insert(0, str(export_dir))
    from inference import InflectTTS  # type: ignore[import-not-found]

    tts = InflectTTS(str(export_dir), device="cpu")
    print(f"  exported package reports n_speakers={tts.n_speakers}")
    lengths = []
    for speaker in range(tts.n_speakers):
        destination = WORK / f"exported-speaker{speaker}.wav"
        tts.save("Salam dunya.", destination, seed=7, speaker=speaker)
        data, _ = sf.read(destination)
        lengths.append(len(data))
        print(f"  speaker={speaker} -> {destination.name} ({len(data) / SAMPLE_RATE:.2f}s)")

    try:
        tts.synthesize("Salam.", speaker=99)
    except ValueError as error:
        print(f"  out-of-range speaker rejected: {error}")
    else:
        raise SystemExit("speaker bounds were not enforced")

    print("\nOK: speaker ids flow manifest -> prepare -> train -> export -> inference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
