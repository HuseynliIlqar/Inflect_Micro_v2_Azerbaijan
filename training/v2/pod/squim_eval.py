"""Reference-free quality estimates (torchaudio SQUIM objective: STOI, PESQ, SI-SDR) for the
validation clips of every version. SQUIM was trained on noisy English speech, so read it as a
noise/distortion meter, not a naturalness score.

    TORCH_HOME=/workspace/hf/torch /workspace/venv/bin/python squim_eval.py [--out /workspace/asr_eval]
Writes <out>/squim_rows.json and <out>/squim_summary.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from torchaudio.pipelines import SQUIM_OBJECTIVE

METRICS = ("stoi", "pesq", "si_sdr")


@torch.inference_mode()
def score(model, path: str, device: str) -> dict:
    wave, rate = sf.read(path, dtype="float32")
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    tensor = torchaudio.functional.resample(torch.from_numpy(wave)[None], rate, SQUIM_OBJECTIVE.sample_rate)
    values = model(tensor.to(device))
    return {name: float(value[0]) for name, value in zip(METRICS, values)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/workspace/asr_eval"))
    parser.add_argument("--versions", nargs="+", default=["base", "step-00025000", "step-00030000"])
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SQUIM_OBJECTIVE.get_model().to(device).eval()
    rows = json.loads((args.out / "rows.json").read_text(encoding="utf-8"))
    paths = {"dataset": [row["target"] for row in rows]}
    for version in args.versions:
        paths[version] = [str(args.out / version / f"{i:04d}.wav") for i in range(len(rows))]
    scored = {version: [score(model, path, device) for path in files] for version, files in paths.items()}
    summary = {version: {name: round(statistics.median(r[name] for r in values), 3) for name in METRICS}
               for version, values in scored.items()}
    (args.out / "squim_rows.json").write_text(json.dumps(scored), encoding="utf-8")
    (args.out / "squim_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
