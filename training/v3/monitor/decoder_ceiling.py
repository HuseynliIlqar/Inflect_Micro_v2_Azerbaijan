"""Where is the quality lost: in the decoder, or in what the text side hands it?

    python decoder_ceiling.py --checkpoint /workspace/keep_ckpt/v3-step-00020204.pth \
        --fd-export /workspace/runs/v4_fd/exports/model-step-00065000.pth --base /workspace/base \
        --dataset /workspace/prepared --pitch /workspace/pitch/pitch_v1.npz /workspace/pitch/pitch_v2.npz \
        --count 60 --out /workspace/ceiling

For the first --count validation clips (never trained on), with their true durations and true pitch:
- real: the recording itself, the target;
- resynth-old / resynth-fd: v3's upsampling decoder and the frame decoder fed the posterior latent of the
  recording (what both decoders were trained on): the decoder's own ceiling;
- gta-old / gta-fd: the same decoders fed the latent the text side and flow produce for the true alignment
  and pitch (the prior, sampled at --noise like inference): what speaking adds on top, except for the
  predicted durations and pitch.
If resynth-fd is close to real and gta-fd is not, the decoder should be trained on the gta latent.
Writes <out>/summary.json and <out>/<condition>/*.wav.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "model"))
sys.path.insert(0, str(HERE.parent / "data"))

from eval_candidates import measure, squim  # noqa: E402
from pitch_dataset import PitchTable, WithPitch, collate_with_pitch  # noqa: E402
from synthesizer_v3 import SynthesizerV3  # noqa: E402
from train_fdec import build_model  # noqa: E402

CONDITIONS = ("real", "resynth-old", "resynth-fd", "gta-old", "gta-fd")


@torch.no_grad()
def render(model, dataset, count: int, noise: float, out: Path, device) -> dict[str, list[str]]:
    files: dict[str, list[str]] = {c: [] for c in CONDITIONS}
    for c in CONDITIONS:
        (out / c).mkdir(parents=True, exist_ok=True)
    for i in range(min(count, len(dataset))):
        x, x_lengths, spec, spec_lengths, wav, wav_lengths, f0 = [t.to(device) for t in collate_with_pitch([dataset[i]])]
        o = model(x, x_lengths, spec, spec_lengths, f0=f0, decode=False)
        y_mask = o["y_mask"]
        f0b = f0[:, None, : y_mask.size(2)] * y_mask
        torch.manual_seed(i)
        z_p = o["m_p"] + torch.randn_like(o["m_p"]) * torch.exp(o["logs_p"]) * noise
        z_gta = model.flow(z_p, y_mask, reverse=True) * y_mask
        z_post = o["z"] * y_mask
        audio = {
            "real": wav[0, 0, : int(wav_lengths[0])],
            "resynth-old": SynthesizerV3.decode(model, z_post, f0b)[0, 0],
            "resynth-fd": model.fdec(z_post, f0b)[0],
            "gta-old": SynthesizerV3.decode(model, z_gta, f0b)[0, 0],
            "gta-fd": model.fdec(z_gta, f0b)[0],
        }
        for c, a in audio.items():
            path = out / c / f"{i:03d}.wav"
            sf.write(path, a.float().clamp(-1, 1).cpu().numpy(), 24000, subtype="PCM_16")
            files[c].append(str(path))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "fd_export", "base", "dataset", "out"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--pitch", type=Path, nargs="+", required=True)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--noise", type=float, default=0.3, help="prior sampling scale, as when speaking")
    parser.add_argument("--f0-ref", type=float, default=178.0)
    args = parser.parse_args()
    args.init = None
    sys.stdout.reconfigure(encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from inflect_finetune.modeling import load_symbols
    from inflect_finetune.training_data import AudioConfig, PreparedTTSDataset

    data = json.loads((args.base / "config.json").read_text(encoding="utf-8"))["data"]
    audio = AudioConfig(sampling_rate=int(data["sampling_rate"]), filter_length=int(data["filter_length"]),
                        hop_length=int(data["hop_length"]), win_length=int(data["win_length"]),
                        add_blank=bool(data.get("add_blank", True)))
    symbols = load_symbols(args.dataset / "symbols.json")
    dataset = WithPitch(PreparedTTSDataset(args.dataset, "validation", symbols, audio), PitchTable(args.pitch))

    model = build_model(args, symbols, device)
    state = torch.load(args.fd_export, map_location="cpu", weights_only=False)["model"]
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys or any(k.startswith("fdec.") for k in result.missing_keys):
        raise RuntimeError(f"fd export mismatch: {result.unexpected_keys[:3]} {result.missing_keys[:3]}")
    model.eval()

    files = render(model, dataset, args.count, args.noise, args.out, device)
    print("rendered", {c: len(p) for c, p in files.items()}, flush=True)
    summary = {}
    with ProcessPoolExecutor(6) as pool:
        for c, paths in files.items():
            rows = list(pool.map(measure, paths, chunksize=8))
            summary[c] = {k: round(float(np.nanmean([r[k] for r in rows])), 3)
                          for k in ("buzz", "low", "ring", "line_db", "range")}
            summary[c].update({k: round(v, 3) for k, v in squim(paths, device).items()})
            print(c, json.dumps(summary[c]), flush=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
