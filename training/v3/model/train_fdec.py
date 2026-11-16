"""Train the frame-rate decoder (frame_decoder.py) on the pod, with the rest of Avaz v3 frozen.

    python train_fdec.py --checkpoint /workspace/keep_ckpt/v3-step-00020204.pth --base /workspace/base \
        --dataset /workspace/prepared --pitch /workspace/pitch/pitch_v1.npz /workspace/pitch/pitch_v2.npz \
        --kit /workspace/kit --listen-from /workspace/runs/v3_full/listen --output /workspace/runs/v4_fd

Stage one ("spectral"): log-mel + multi-resolution STFT, harmonicity x2, modulation x5 (Paradee's recipe).
Stage two ("gan"): the spectral losses x3, LS-GAN + feature matching x2 against compact critics, the two
statistic losses kept. Each stage starts a fresh optimiser with 500 warm-up steps and a cosine decay.

The decoder sees the posterior latent of the true audio and the true pitch (the Praat tracks), one 64-frame
segment per clip; the text side, flow, posterior encoder and pitch part stay as in the v3 checkpoint
(--init takes v2's polish weights instead, for a local smoke test). Every --eval-interval it renders the 12
listening sentences through the whole model (predicted durations and pitch) into <out>/listen/step-*/ with
buzz and ring, the monitor's format, and writes inference weights to <out>/exports/. SIGTERM writes a
checkpoint and exits 3; rerunning the same command resumes.

--latent gta (with --init-fdec, a finished run's checkpoint) fine-tunes the decoder on what it gets when
speaking instead: the text side's prior for the true alignment and true pitch, sampled at --gta-noise and
taken through the flow in reverse, against the true audio (as vocoders are fine-tuned on ground-truth-aligned
features). Use --spectral-steps 0 to go straight to the adversarial stage.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import signal
import statistics
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "data"))
sys.path.insert(0, str(HERE.parent / "monitor"))

from batching import FrameBudgetBatches  # noqa: E402
from frame_losses import Discriminators, SpectralLosses, d_loss, g_loss, harmonicity_loss, modulation_loss  # noqa: E402
from pitch_dataset import PitchTable, WithPitch  # noqa: E402
from runtime import load_runtime  # noqa: E402
from synthesizer_fd import SynthesizerFD  # noqa: E402
from train_v3 import _alignment, _drop_rows_after, _loader, _save  # noqa: E402

LOGGER = logging.getLogger("avaz_fd")
KEEP = 3
PAUSED_EXIT = 3
SEGMENT = 64  # frames, 0.68 s
WARMUP = 500
WEIGHTS = {"harm": 2.0, "mod": 5.0, "spectral_gan": 3.0, "fm": 2.0}


def schedule(step: int, total: int) -> float:
    return min(1.0, step / WARMUP) * 0.5 * (1 + math.cos(math.pi * min(step, total) / total))


def build_model(args, symbols, device) -> SynthesizerFD:
    config = json.loads((args.base / "config.json").read_text(encoding="utf-8"))
    data = config["data"]
    options = {k: v for k, v in config["model"].items() if k != "inference_only"}
    model = SynthesizerFD(load_runtime(args.base / "runtime"), len(symbols), int(data["filter_length"]) // 2 + 1,
                          int(config["train"]["segment_size"]) // int(data["hop_length"]), options, _alignment(),
                          f0_ref_hz=args.f0_ref)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["trainer"]["model"]
        result = model.load_state_dict(state, strict=False)
        stray = [k for k in result.missing_keys if not k.startswith("fdec.")]
        if stray or result.unexpected_keys:
            raise RuntimeError(f"checkpoint mismatch: missing {stray[:3]}, unexpected {result.unexpected_keys[:3]}")
        LOGGER.info("v3 weights from %s", args.checkpoint)
    else:
        model.load_v2_state(torch.load(args.init, map_location="cpu", weights_only=False)["generator"])
        LOGGER.info("v2 weights from %s (smoke test)", args.init)
    model.requires_grad_(False)
    model.fdec.requires_grad_(True)
    return model.to(device)


class Listener:
    """Renders the monitor's 12 listening sentences through the whole model and measures buzz and ring."""

    def __init__(self, args, symbols, device):
        from listen_v3 import load_frontend, summary, tokens_for

        self.summary, self.device = summary, device
        config = json.loads((args.base / "config.json").read_text(encoding="utf-8"))
        add_blank = bool(config["data"].get("add_blank", True))
        sentences = json.loads((HERE.parent / "monitor" / "listen_sentences_v3.json").read_text(encoding="utf-8"))
        frontend = load_frontend(args.kit)
        self.tokens = [tokens_for(frontend.phonemize(frontend.normalize(t)), symbols, add_blank) for t in sentences]
        self.out = args.output / "listen"
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "sentences.json").write_text(json.dumps(sentences, ensure_ascii=False), encoding="utf-8")
        if args.listen_from and args.listen_from.is_dir():  # v2's renders and the dataset's numbers, for comparison
            for name in ("base", "dataset_metrics.json"):
                src, dst = args.listen_from / name, self.out / name
                if src.exists() and not dst.exists():
                    (shutil.copytree if src.is_dir() else shutil.copyfile)(src, dst)

    @torch.no_grad()
    def __call__(self, model: SynthesizerFD, step: int) -> dict:
        model.eval()
        folder = self.out / f"step-{step:08d}"
        staging = folder.with_name(folder.name + ".partial")
        staging.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, x in enumerate(self.tokens):
            torch.manual_seed(0)
            x = x.to(self.device)
            audio, _, _, _ = model.infer(x, torch.tensor([x.size(1)], device=self.device), noise_scale=0.3)
            path = staging / f"{i:02d}.wav"
            sf.write(path, audio[0, 0].float().clamp(-1, 1).cpu().numpy(), 24000, subtype="PCM_16")
            paths.append(path)
        try:
            metrics = self.summary(paths)
        except statistics.StatisticsError:  # no voiced frames found yet (an untrained decoder makes noise)
            metrics = {"h2_4k": None, "h1_2k": None, "ring_share": None, "ring_hz": None, "clips": len(paths)}
        (staging / "metrics.json").write_text(json.dumps(metrics))
        if folder.exists():
            shutil.rmtree(folder)
        staging.rename(folder)
        return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="a v3 trainer checkpoint (v3-step-*.pth)")
    source.add_argument("--init", type=Path, help="v2's polish weights instead (smoke test)")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--pitch", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kit", type=Path, help="folder with az_stress_frontend.py; without it no listening renders")
    parser.add_argument("--listen-from", type=Path, help="a v3 run's listen/ folder: copies v2's renders")
    parser.add_argument("--latent", choices=("posterior", "gta"), default="posterior",
                        help="decoder input: the recording's posterior latent, or the text side's (see above)")
    parser.add_argument("--gta-noise", type=float, default=0.3, help="prior sampling scale for --latent gta")
    parser.add_argument("--init-fdec", type=Path, help="start the decoder and critics from this fd checkpoint")
    parser.add_argument("--spectral-steps", type=int, default=25000)
    parser.add_argument("--gan-steps", type=int, default=15000)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--frame-budget", type=int, default=48 * 1000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--f0-ref", type=float, default=178.0)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=2500)
    parser.add_argument("--checkpoint-interval", type=int, default=2500)
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
    symbols = load_symbols(args.dataset / "symbols.json")
    audio = AudioConfig(sampling_rate=int(data["sampling_rate"]), filter_length=int(data["filter_length"]),
                        hop_length=hop, win_length=int(data["win_length"]), add_blank=bool(data.get("add_blank", True)))
    train_set = WithPitch(PreparedTTSDataset(args.dataset, "train", symbols, audio), PitchTable(args.pitch))
    frames = [int(float(row["duration_seconds"]) * audio.sampling_rate) // hop for row in train_set.dataset.rows]
    sampler = FrameBudgetBatches(frames, args.batch_size, args.frame_budget, seed=args.seed)
    loader = _loader(train_set, sampler, args.workers)
    LOGGER.info("train %d clips", len(train_set))

    model = build_model(args, symbols, device)
    model.eval()
    model.fdec.train()
    spectral = SpectralLosses().to(device)
    critics = Discriminators().to(device)
    listener = Listener(args, symbols, device) if args.kit else None
    total = args.spectral_steps + args.gan_steps
    adam = dict(lr=args.lr, betas=(0.8, 0.99), weight_decay=0.01)

    def fresh_optimisers():
        return torch.optim.AdamW(model.fdec.parameters(), **adam), torch.optim.AdamW(critics.parameters(), **adam)

    opt, opt_d = fresh_optimisers()
    opt_stage = "spectral"  # the stage the optimisers belong to
    out = args.output
    latest = out / "checkpoints" / "latest.pth"
    step = 0
    if latest.is_file():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        model.fdec.load_state_dict(state["fdec"])
        critics.load_state_dict(state["critics"])
        opt.load_state_dict(state["opt"])
        opt_d.load_state_dict(state["opt_d"])
        opt_stage = state.get("opt_stage", "spectral")
        step = int(state["step"])
        sampler.set_epoch(int(state.get("epoch", 0)))
        _drop_rows_after(out / "metrics.jsonl", step)
        LOGGER.info("resumed at step %d", step)
    elif args.init_fdec:
        state = torch.load(args.init_fdec, map_location="cpu", weights_only=False)
        model.fdec.load_state_dict(state["fdec"])
        critics.load_state_dict(state["critics"])
        LOGGER.info("decoder and critics from %s (step %s)", args.init_fdec, state.get("step"))
    out.mkdir(parents=True, exist_ok=True)
    (out.parent / f"{out.name}.pid").write_text(str(os.getpid()))
    (out / "training-options.json").write_text(json.dumps({**vars(args), "max_steps": total}, default=str, indent=2))
    LOGGER.info("frame decoder %.2fM parameters", sum(p.numel() for p in model.fdec.parameters()) / 1e6)
    stop_requested = []
    signal.signal(signal.SIGTERM, lambda *_: stop_requested.append(True))

    def save_checkpoint() -> None:
        payload = {"fdec": model.fdec.state_dict(), "critics": critics.state_dict(), "opt": opt.state_dict(),
                   "opt_d": opt_d.state_dict(), "opt_stage": opt_stage, "step": step, "epoch": sampler.epoch}
        _save(out / "checkpoints" / f"fd-step-{step:08d}.pth", payload)
        _save(latest, payload)
        for old in sorted((out / "checkpoints").glob("fd-step-*.pth"))[:-KEEP]:
            old.unlink()

    def evaluate() -> None:
        _save(out / "exports" / f"model-step-{step:08d}.pth", {"model": model.inference_state(), "step": step})
        if listener is not None:
            metrics = listener(model, step)
            LOGGER.info("step %d listening: %s", step, json.dumps(metrics))
        model.fdec.train()

    metrics_file = (out / "metrics.jsonl").open("a", encoding="utf-8")
    started, done_at_start = time.time(), step
    while step < total:
        for batch in loader:
            stage = "spectral" if step < args.spectral_steps else "gan"
            if stage != opt_stage:  # stage two starts fresh optimisers, as Paradee's --init-from did
                opt, opt_d = fresh_optimisers()
                opt_stage = stage
            local = step if stage == "spectral" else step - args.spectral_steps
            span = args.spectral_steps if stage == "spectral" else args.gan_steps
            for group in (*opt.param_groups, *opt_d.param_groups):
                group["lr"] = args.lr * schedule(local, span)
            x, x_lengths, spec, spec_lengths, wav, _, f0 = [t.to(device, non_blocking=True) for t in batch]
            with torch.no_grad():
                if args.latent == "gta":
                    prior = model(x, x_lengths, spec, spec_lengths, f0=f0, decode=False)
                    y_mask = prior["y_mask"]
                    z_p = prior["m_p"] + torch.randn_like(prior["m_p"]) * torch.exp(prior["logs_p"]) * args.gta_noise
                    z = model.flow(z_p, y_mask, reverse=True) * y_mask
                else:
                    z, _, _, y_mask = model.enc_q(spec, spec_lengths)
                z_slice, ids = model.commons.rand_slice_segments(z, spec_lengths, SEGMENT)
                f0_slice = model.commons.slice_segments(f0[:, None, : y_mask.size(2)] * y_mask, ids, SEGMENT)[:, 0]
                target = model.commons.slice_segments(wav, ids * hop, SEGMENT * hop)[:, 0]
            y = model.fdec(z_slice, f0_slice)
            head = model.fdec.head.last
            lm, ls = spectral(y, target)
            lh = harmonicity_loss(y, target, f0_slice)
            lmod = modulation_loss(y, target, head["cycles"], head["voiced"])
            row = {"loss_mel": lm.item(), "loss_stft": ls.item(), "loss_harm": lh.item(), "loss_mod": lmod.item()}
            if stage == "gan":
                ld = d_loss(critics, target, y)
                opt_d.zero_grad(set_to_none=True)
                ld.backward()
                opt_d.step()
                adv, fm = g_loss(critics, target, y)
                loss = WEIGHTS["spectral_gan"] * (lm + ls) + adv + WEIGHTS["fm"] * fm
                row.update(loss_d=ld.item(), loss_adv=adv.item(), loss_fm=fm.item())
            else:
                loss = lm + ls
            loss = loss + WEIGHTS["harm"] * lh + WEIGHTS["mod"] * lmod
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.fdec.parameters(), 5.0)
            opt.step()
            step += 1
            if not math.isfinite(float(loss)):
                raise RuntimeError(f"non-finite loss at step {step}: {row}")
            if step % 10 == 0:
                metrics_file.write(json.dumps({"step": step, "wall_time": time.time(), "stage": stage,
                                               "loss_g": float(loss), **row}) + "\n")
                metrics_file.flush()
            if step % args.log_interval == 0:
                rate = (time.time() - started) / max(1, step - done_at_start)
                LOGGER.info("step=%d %s loss=%.3f mel=%.3f stft=%.3f harm=%.3f mod=%.3f%s %.2fs/step", step, stage,
                            float(loss), row["loss_mel"], row["loss_stft"], row["loss_harm"], row["loss_mod"],
                            f" adv={row['loss_adv']:.3f} fm={row['loss_fm']:.3f} d={row['loss_d']:.3f}"
                            if stage == "gan" else "", rate)
            if step % args.checkpoint_interval == 0 or step == total:
                save_checkpoint()
            if step % args.eval_interval == 0 or step == total:
                evaluate()
            if stop_requested:
                save_checkpoint()
                LOGGER.info("stop requested: checkpoint written at step %d, exiting to resume later", step)
                sys.exit(PAUSED_EXIT)
            if step >= total:
                break
    LOGGER.info("finished at step %d", step)


if __name__ == "__main__":
    main()
