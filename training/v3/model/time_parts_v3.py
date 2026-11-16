"""Wall-clock time of each part of a full-size joint step, with CUDA synchronised around every part.

    python time_parts_v3.py --init ... --base ... --dataset ... --pitch ... [--batch-size 64] [--amp bf16|fp16|none]

Forward hooks time every top-level part of the model and the discriminators; the alignment search,
the losses' backward passes and the optimiser steps are timed by wrapping them. Run on a free GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
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

TIMES: dict[str, float] = defaultdict(float)


def _wrap(label: str, fn):
    def inner(*args, **kwargs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = fn(*args, **kwargs)
        torch.cuda.synchronize()
        TIMES[label] += time.perf_counter() - start
        return out
    return inner


def _hook(module: torch.nn.Module, label: str) -> None:
    starts = {}

    def pre(*_):
        torch.cuda.synchronize()
        starts["t"] = time.perf_counter()

    def post(*_):
        torch.cuda.synchronize()
        TIMES[f"fwd {label}"] += time.perf_counter() - starts["t"]

    module.register_forward_pre_hook(pre)
    module.register_forward_hook(post)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--pitch", type=Path, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--amp", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--benchmark", action="store_true", help="torch.backends.cudnn.benchmark = True")
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = args.benchmark
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
    sampler = FrameBudgetBatches(frames, args.batch_size, args.batch_size * 1360, seed=5)
    loader = torch.utils.data.DataLoader(train_set, batch_sampler=sampler, num_workers=8, collate_fn=collate_with_pitch)
    options = {k: v for k, v in config["model"].items() if k != "inference_only"}
    model = SynthesizerV3(load_runtime(args.base / "runtime"), len(symbols), int(data["filter_length"]) // 2 + 1,
                          segment, options, _wrap("alignment (MAS, CPU)", _alignment()))
    init = torch.load(args.init, map_location="cpu", weights_only=False)
    model.load_v2_state(init["generator"])
    disc = AvazDiscriminator(DiscriminatorSpec(families=("mpd", "mrd", "cqtd")))
    disc.load_state_dict(init["discriminator"])
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[args.amp]
    losses = AudioLosses(_wrap("loss multi-scale mel", multi_scale_mel_loss), _wrap("loss linear stft", linear_stft_loss),
                         discriminator_loss, generator_loss, feature_loss)
    trainer = StepV3(model, disc, losses, segment, hop, 24000, disc_batch=12, amp_dtype=amp).to(device)
    for name in ("enc_q", "flow", "dp", "pitch", "prior_f0", "source"):
        _hook(getattr(model, name), name)
    model.encode_text = _wrap("fwd text encoder", model.encode_text)
    model.decode = _wrap("fwd decode (decoder + source)", model.decode)
    for name, family in disc.families.items():
        _hook(family, f"audio disc {name}")
    _hook(trainer.duration_disc, "duration disc")
    _hook(trainer.pitch_disc, "pitch disc")
    trainer._generator_step = _wrap("generator backward + step", trainer._generator_step)
    trainer._prosody_losses = _wrap("prosody losses (incl. their D step)", trainer._prosody_losses)

    shapes, step_times = [], []
    it = iter(loader)
    for i in range(args.steps + 2):
        batch = tuple(t.to(device) for t in next(it))
        if i == 2:
            TIMES.clear()
        torch.cuda.synchronize()
        start = time.perf_counter()
        trainer.step(batch, "joint", 0)
        torch.cuda.synchronize()
        if i >= 2:
            step_times.append(time.perf_counter() - start)
            shapes.append((batch[0].size(0), batch[2].size(2)))
    n = len(step_times)
    print(json.dumps({"amp": args.amp, "benchmark": args.benchmark, "batches (clips, frames)": shapes,
                      "step_s": [round(t, 2) for t in step_times], "mean_step_s": round(sum(step_times) / n, 2),
                      "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
                      "parts_s_per_step": {k: round(v / n, 3) for k, v in sorted(TIMES.items(), key=lambda kv: -kv[1])}},
                     indent=1))


if __name__ == "__main__":
    main()
