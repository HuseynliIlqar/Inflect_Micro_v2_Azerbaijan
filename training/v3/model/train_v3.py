"""Train Avaz v3 on the pod: prosody phase, discriminator warm-up, then joint training.

    python train_v3.py --init /workspace/keep/polish-step-00030000.pth --base /workspace/base \
        --dataset /workspace/prepared --pitch /workspace/pitch/pitch_v1.npz /workspace/pitch/pitch_v2.npz \
        --output /workspace/runs/v3 --prosody-steps 15000 --warmup-steps 1000 --joint-steps 40000

--init is the polish run's checkpoint (v2's final weights and its trained discriminators). Writes
metrics.jsonl (the monitor's format: step, stage, loss_d, loss_g, loss_mel), validation clips with
their predicted pitch every --validation-interval, resumable checkpoints (the last three) and
inference weights every --checkpoint-interval. Rerunning the same command resumes from latest.pth.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
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
from trainer_v3 import AudioLosses, StepV3  # noqa: E402

LOGGER = logging.getLogger("avaz_v3")
KEEP = 3
PAUSED_EXIT = 3  # SIGTERM: a checkpoint was written and the run can resume; run_full.sh reports "paused"
VALIDATION_CLIPS = 8


def _alignment():
    """The toolkit's vectorised alignment when installed (the pod), else the NumPy one here."""
    try:
        from inflect_finetune.monotonic_align import maximum_path
    except ImportError:
        from monotonic_align import maximum_path
    return maximum_path


def phase_for(step: int, prosody: int, warmup: int, joint: int | None = None) -> str:
    """prosody, disc_warmup, joint, then (from prosody + warmup + joint on) a final prosody phase that refits
    the rhythm and pitch parts to the text side as joint training left it."""
    if step < prosody:
        return "prosody"
    if step < prosody + warmup:
        return "disc_warmup"
    if joint is None or step < prosody + warmup + joint:
        return "joint"
    return "prosody"


def _loader(dataset, sampler: FrameBudgetBatches, workers: int):
    return torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=workers,
                                       collate_fn=collate_with_pitch, pin_memory=torch.cuda.is_available(),
                                       persistent_workers=workers > 0)


@torch.no_grad()
def validate(model: SynthesizerV3, dataset, device, out_dir: Path, step: int) -> dict:
    """Synthesise the first validation clips at variation 0.3 and compare their pitch with the truth."""
    model.eval()
    folder = out_dir / "validation" / f"step-{step:08d}"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(min(VALIDATION_CLIPS, len(dataset))):
        tokens, _, _, f0_true = dataset[index]
        x = tokens[None].to(device)
        audio, _, _, f0 = model.infer(x, torch.tensor([x.size(1)], device=device), noise_scale=0.3)
        sf.write(folder / f"{index:02d}.wav", audio[0, 0].float().cpu().numpy(), 24000, subtype="PCM_16")
        predicted = f0[0, 0].float().cpu().numpy()
        voiced = predicted[predicted > 0]
        truth = f0_true.numpy()[f0_true.numpy() > 0]
        spread = lambda v: float(12 * np.log2(np.percentile(v, 95) / np.percentile(v, 5))) if len(v) > 10 else None  # noqa: E731
        rows.append({"clip": index, "seconds": round(audio.size(-1) / 24000, 2), "range_st_pred": spread(voiced),
                     "range_st_true": spread(truth)})
        np.save(folder / f"{index:02d}_f0.npy", predicted)
    model.train()
    (folder / "summary.json").write_text(json.dumps(rows, indent=1))
    return {"validation_range_st_pred": float(np.nanmean([r["range_st_pred"] or np.nan for r in rows])),
            "validation_range_st_true": float(np.nanmean([r["range_st_true"] or np.nan for r in rows]))}


def _drop_rows_after(path: Path, step: int) -> None:
    """On resume, forget the logged steps after the checkpoint, so the charts do not jump back."""
    if not path.is_file():
        return
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = [line for line in lines if json.loads(line)["step"] <= step]
    if len(kept) != len(lines):
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")
        tmp.replace(path)


def _save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True, help="model package with config.json and runtime/")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--pitch", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prosody-steps", type=int, default=15000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--joint-steps", type=int, default=40000)
    parser.add_argument("--final-prosody-steps", type=int, default=2000,
                        help="a last prosody phase after joint training (0 to skip)")
    parser.add_argument("--batch-size", type=int, default=64, help="most clips per batch (short clips)")
    parser.add_argument("--frame-budget", type=int, default=64 * 1360,
                        help="most padded frames per batch: v2's 64 clips of up to 14.5 s")
    parser.add_argument("--disc-batch", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1.5e-4, help="v2's rate where the polish run ended")
    parser.add_argument("--f0-ref", type=float, default=178.0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--workers", type=int, default=5,
                        help="data loader processes; RunPod's A40 pods have a quota of about 7.6 CPUs")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads of the main process")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda", help="cpu for a local smoke test")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--validation-interval", type=int, default=2500)
    parser.add_argument("--checkpoint-interval", type=int, default=5000)
    parser.add_argument("--source-lr-multiplier", type=float, default=None,
                        help="learning-rate multiplier for the harmonic source (default 1.0); applied after a resume too")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)

    from inflect_finetune.modeling import load_symbols
    from inflect_finetune.training_data import AudioConfig, PreparedTTSDataset

    config = json.loads((args.base / "config.json").read_text(encoding="utf-8"))
    data = config["data"]
    hop = int(data["hop_length"])
    segment = int(config["train"]["segment_size"]) // hop
    symbols = load_symbols(args.dataset / "symbols.json")
    audio = AudioConfig(sampling_rate=int(data["sampling_rate"]), filter_length=int(data["filter_length"]),
                        hop_length=hop, win_length=int(data["win_length"]), add_blank=bool(data.get("add_blank", True)))
    table = PitchTable(args.pitch)
    train_set = WithPitch(PreparedTTSDataset(args.dataset, "train", symbols, audio), table)
    validation_set = WithPitch(PreparedTTSDataset(args.dataset, "validation", symbols, audio), table)
    frames = [int(float(row["duration_seconds"]) * audio.sampling_rate) // hop for row in train_set.dataset.rows]
    sampler = FrameBudgetBatches(frames, args.batch_size, args.frame_budget, seed=args.seed)
    loader = _loader(train_set, sampler, args.workers)
    LOGGER.info("train %d clips, validation %d, pitch tracks %d", len(train_set), len(validation_set), len(table))

    runtime = load_runtime(args.base / "runtime")
    model_options = {k: v for k, v in config["model"].items() if k != "inference_only"}
    model = SynthesizerV3(runtime, len(symbols), int(data["filter_length"]) // 2 + 1, segment, model_options,
                          _alignment(), f0_ref_hz=args.f0_ref)
    disc = AvazDiscriminator(DiscriminatorSpec(families=("mpd", "mrd", "cqtd")))
    losses = AudioLosses(multi_scale_mel_loss, linear_stft_loss, discriminator_loss, generator_loss, feature_loss)
    trainer = StepV3(model, disc, losses, segment, hop, audio.sampling_rate, lr=args.lr, disc_batch=args.disc_batch,
                     amp_dtype=torch.bfloat16 if args.bf16 else None).to(device)

    out = args.output
    latest = out / "checkpoints" / "latest.pth"
    if latest.is_file():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        trainer.load_state_dict(state["trainer"])
        step = int(state["step"])
        sampler.set_epoch(int(state.get("epoch", 0)))
        _drop_rows_after(out / "metrics.jsonl", step)
        LOGGER.info("resumed at step %d", step)
    else:
        init = torch.load(args.init, map_location="cpu", weights_only=False)
        new = model.load_v2_state(init["generator"])
        disc.load_state_dict(init["discriminator"])
        step = 0
        LOGGER.info("v2 weights and discriminators loaded; new tensors: %d", len(new))
    if args.source_lr_multiplier is not None:
        trainer.set_group_multiplier("source", args.source_lr_multiplier)
        LOGGER.info("harmonic source learning-rate multiplier %.2f", args.source_lr_multiplier)
    total = args.prosody_steps + args.warmup_steps + args.joint_steps + args.final_prosody_steps
    out.mkdir(parents=True, exist_ok=True)
    (out.parent / f"{out.name}.pid").write_text(str(os.getpid()))  # the monitor shows the run as alive while this lives
    (out / "training-options.json").write_text(json.dumps({**vars(args), "max_steps": total}, default=str, indent=2))
    model.train()
    stop_requested = []
    signal.signal(signal.SIGTERM, lambda *_: stop_requested.append(True))  # e.g. the balance watcher

    def save_checkpoint() -> None:
        payload = {"trainer": trainer.state_dict(), "step": step, "epoch": sampler.epoch}
        _save(out / "checkpoints" / f"v3-step-{step:08d}.pth", payload)
        _save(latest, payload)
        for old in sorted((out / "checkpoints").glob("v3-step-*.pth"))[:-KEEP]:
            old.unlink()

    while step < total:
        for batch in loader:
            phase = phase_for(step, args.prosody_steps, args.warmup_steps, args.joint_steps)
            batch = tuple(t.to(device, non_blocking=True) for t in batch)
            metrics = trainer.step(batch, phase, step)
            step += 1
            row = {"step": step, "wall_time": round(time.time(), 3), "stage": phase,
                   "loss_mel": metrics.get("loss_mel_ms", 0.0), "loss_g": metrics.get("loss_g", 0.0),
                   "loss_d": metrics.get("loss_d", 0.0), **metrics,
                   "peak_memory_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 3) if device.type == "cuda" else 0.0}
            with (out / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            if step % args.log_interval == 0:
                LOGGER.info("step=%d %s g=%.3f d=%.3f pitch_l1=%.3f", step, phase, row["loss_g"], row["loss_d"],
                            metrics.get("loss_pitch_l1", float("nan")))
            if step % args.validation_interval == 0 or step == total:
                info = validate(model, validation_set, device, out, step)
                LOGGER.info("validation at %d: %s", step, info)
            if step % args.checkpoint_interval == 0 or step == total:
                save_checkpoint()
                weights = {k: v for k, v in model.state_dict().items() if not k.startswith("enc_q.")}
                _save(out / "exports" / f"model-step-{step:08d}.pth", {"model": weights, "f0_ref_hz": args.f0_ref,
                                                                        "step": step})
            if stop_requested:
                save_checkpoint()
                LOGGER.info("stop requested: checkpoint written at step %d, exiting to resume later", step)
                sys.exit(PAUSED_EXIT)
            if step >= total:
                break
    LOGGER.info("finished at step %d", step)


if __name__ == "__main__":
    main()
