"""Where a joint training step spends its GPU time (torch.profiler), on a small batch beside a running job.

    python profile_v3.py --init ... --base ... --dataset ... --pitch ... --batch-size 16

Prints the operators with the most CUDA time and wall-clock timings of the step's parts.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "data"))
sys.path.insert(0, str(HERE.parents[1] / "v2" / "polish"))

from batching import FrameBudgetBatches  # noqa: E402
from discriminators import AvazDiscriminator, DiscriminatorSpec  # noqa: E402
from losses import discriminator_loss, feature_loss, generator_loss, linear_stft_loss, multi_scale_mel_loss  # noqa: E402
from pitch_dataset import PitchTable, WithPitch, collate_with_pitch  # noqa: E402
from runtime import load_runtime  # noqa: E402
from synthesizer_v3 import SynthesizerV3  # noqa: E402
from train_v3 import _alignment  # noqa: E402
from trainer_v3 import AudioLosses, StepV3  # noqa: E402


def timed(label: str, fn, results: dict):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    results[label] = results.get(label, 0.0) + time.perf_counter() - start
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--pitch", type=Path, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    from inflect_finetune.modeling import load_symbols
    from inflect_finetune.training_data import AudioConfig, PreparedTTSDataset

    device = torch.device("cuda")
    config = json.loads((args.base / "config.json").read_text(encoding="utf-8"))
    data = config["data"]
    hop = int(data["hop_length"])
    segment = int(config["train"]["segment_size"]) // hop
    symbols = load_symbols(args.dataset / "symbols.json")
    audio = AudioConfig(sampling_rate=24000, filter_length=int(data["filter_length"]), hop_length=hop,
                        win_length=int(data["win_length"]), add_blank=bool(data.get("add_blank", True)))
    train_set = WithPitch(PreparedTTSDataset(args.dataset, "train", symbols, audio), PitchTable(args.pitch))
    frames = [int(float(r["duration_seconds"]) * 24000) // hop for r in train_set.dataset.rows]
    budget = args.batch_size * 1360
    loader = torch.utils.data.DataLoader(train_set, batch_sampler=FrameBudgetBatches(frames, args.batch_size, budget),
                                         num_workers=4, collate_fn=collate_with_pitch)
    options = {k: v for k, v in config["model"].items() if k != "inference_only"}
    model = SynthesizerV3(load_runtime(args.base / "runtime"), len(symbols), int(data["filter_length"]) // 2 + 1,
                          segment, options, _alignment())
    init = torch.load(args.init, map_location="cpu", weights_only=False)
    model.load_v2_state(init["generator"])
    disc = AvazDiscriminator(DiscriminatorSpec(families=("mpd", "mrd", "cqtd")))
    disc.load_state_dict(init["discriminator"])
    losses = AudioLosses(multi_scale_mel_loss, linear_stft_loss, discriminator_loss, generator_loss, feature_loss)
    trainer = StepV3(model, disc, losses, segment, hop, 24000, disc_batch=12, amp_dtype=torch.bfloat16).to(device)
    batches = []
    for batch in loader:
        batches.append(tuple(t.to(device) for t in batch))
        if len(batches) == 5:
            break

    parts: dict[str, float] = {}
    for b in batches[:2]:
        trainer.step(b, "joint", 0)
    for b in batches[2:]:
        timed("whole joint step", lambda: trainer.step(b, "joint", 0), parts)
        x, xl, spec, sl, wav, _, f0 = b
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = timed("model forward (all)", lambda: model(x, xl, spec, sl, f0=f0), parts)
            timed("model forward without decoder", lambda: model(x, xl, spec, sl, f0=f0, decode=False), parts)
            z = out["z"][:, :, :segment].detach()
            f0s = out["f0_slice"]
            timed("decoder alone", lambda: model.decode(z, f0s), parts)
            timed("decoder without source", lambda: model.decode(z), parts)
            timed("harmonic source alone", lambda: model.source(f0s), parts)
        real = torch.randn(12, 1, segment * hop, device=device)
        timed("audio discriminators fwd", lambda: disc(real, real), parts)
    print(json.dumps({k: round(v / 3, 3) for k, v in parts.items()}, indent=1))

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]) as prof:
        trainer.step(batches[2], "joint", 0)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=60))


if __name__ == "__main__":
    main()
