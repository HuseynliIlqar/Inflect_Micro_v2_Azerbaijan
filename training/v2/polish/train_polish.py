"""Decoder run for Avaz with stronger discriminators and spectral losses.

Starts from a checkpoint of the main run (default: step 20,000, before the decoder unfroze),
then:
  1. disc_warmup: the generator is frozen and only the new discriminators learn, so they can
     already tell the old decoder's artefacts (6 kHz ring, buzz) from the dataset audio;
  2. decoder_polish: everything trains as in the toolkit's decoder stage, with
     MPD + MRD + CQTD instead of the release MPD, and multi-scale mel + linear STFT losses
     instead of the single mel loss.

The reconstruction weights are calibrated once on the first batch so that, together, the new
losses weigh what 45 x mel weighed in the main run (3/4 mel, 1/4 linear STFT); the adversarial
and feature terms are scaled by 6/11 so the eleven sub-discriminators push as hard as the six
of the main run. Both choices are written to the run options and kept on resume.

Writes metrics.jsonl (the monitor's format, with the old mel loss still logged for comparison),
validation clips every 2,500 steps, exports/model-step-*.pth every 5,000 (read by listen.py)
and resumable checkpoints.

    python train_polish.py --init /workspace/keep/adaptation-step-00020000.pth \
        --bigvgan /workspace/bigvgan/bigvgan_discriminator_optimizer.pt --output /workspace/runs/polish
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
import torch.nn.functional as F

from inflect_finetune.checkpoint import save_inference_checkpoint
from inflect_finetune.modeling import build_training_models, load_symbols
from inflect_finetune.training import (
    STAGE_DECODER,
    TrainingOptions,
    _apply_stage,
    _autocast,
    _generator_groups,
    _grad_scaler,
    _kl_loss,
    _mel_from_spec,
    _mel_from_waveform,
    _peak_memory_gb,
    _reserved_memory_gb,
    _seed_everything,
    _validate,
)
from inflect_finetune.training_data import AudioConfig, PreparedTTSDataset, create_dataloader

from discriminators import AvazDiscriminator, DiscriminatorSpec, load_bigvgan_weights
from losses import discriminator_loss, feature_loss, generator_loss, linear_stft_loss, multi_scale_mel_loss

LOGGER = logging.getLogger("avaz_polish")
STAGE_WARMUP = "disc_warmup"
OLD_MEL_WEIGHT = 45.0
MEL_SHARE, STFT_SHARE = 0.75, 0.25
ADV_SCALE = 6.0 / 11.0
KEEP_CHECKPOINTS = 3
DISC_CLIP = 500.0  # per discriminator family, as clip_grad_norm in BigVGAN-v2's config
D_SPIKE = 30.0  # total discriminator loss above which the generator skips the batch (normal is about 4-5)


@dataclass(frozen=True)
class PolishSettings:
    start_step: int
    warmup_steps: int
    end_step: int
    mel_ms_weight: float = 0.0  # set by calibration
    stft_weight: float = 0.0
    adv_scale: float = ADV_SCALE
    families: tuple[str, ...] = ("mpd", "mrd", "cqtd")


def _stage(settings: PolishSettings, step: int) -> str:
    return STAGE_WARMUP if step < settings.start_step + settings.warmup_steps else STAGE_DECODER


def _set_generator_trainable(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                             options: TrainingOptions, stage: str) -> None:
    if stage == STAGE_WARMUP:
        for group in optimizer.param_groups:
            group["lr"] = 0.0
            for parameter in group["params"]:
                parameter.requires_grad_(False)
                parameter.grad = None
    else:
        _apply_stage(model, optimizer, options, STAGE_DECODER)  # resets learning rates, as the toolkit does


def _segment(bundle, audio: AudioConfig, batch, device):
    x, x_lengths, spec, spec_lengths, waveform, _, _ = (t.to(device, non_blocking=True) for t in batch)
    return x, x_lengths, spec, spec_lengths, waveform


def _calibrate(bundle, audio, loader, device, amp) -> tuple[float, float, dict]:
    """Weights that make the new reconstruction losses weigh what 45 x mel did, on one batch."""
    segment = int(bundle.config["train"]["segment_size"])
    batch = next(iter(loader))
    x, x_lengths, spec, spec_lengths, waveform = _segment(bundle, audio, batch, device)
    with torch.no_grad(), _autocast(device, amp):
        generated, _, _, ids_slice, _, _, _ = bundle.generator(x, x_lengths, spec, spec_lengths)
        real = bundle.components.commons.slice_segments(waveform, ids_slice * audio.hop_length, segment)
        target_mel = bundle.components.commons.slice_segments(_mel_from_spec(spec, bundle), ids_slice, segment // audio.hop_length)
        old = F.l1_loss(target_mel.float(), _mel_from_waveform(generated.squeeze(1), bundle).float()).item()
        mel_ms = multi_scale_mel_loss(generated, real, audio.sampling_rate).item()
        stft = linear_stft_loss(generated, real).item()
    total = OLD_MEL_WEIGHT * old
    weights = (MEL_SHARE * total / mel_ms, STFT_SHARE * total / stft)
    return weights[0], weights[1], {"old_mel": old, "mel_ms": mel_ms, "stft": stft, "old_weighted": total}


def _save(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _prune(folder: Path) -> None:
    steps = sorted(folder.glob("polish-step-*.pth"))
    for old in steps[:-KEEP_CHECKPOINTS]:
        old.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", type=Path, required=True, help="main-run training checkpoint to start from")
    parser.add_argument("--bigvgan", type=Path, help="BigVGAN-v2 bigvgan_discriminator_optimizer.pt")
    parser.add_argument("--base", type=Path, default=Path("/workspace/base"))
    parser.add_argument("--dataset", type=Path, default=Path("/workspace/prepared"))
    parser.add_argument("--preset", type=Path, default=Path("/workspace/pod/az-continue.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-steps", type=int, default=2500)
    parser.add_argument("--steps", type=int, default=30000, help="steps after the start checkpoint, warm-up included")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--accumulation", type=int)
    parser.add_argument("--families", default="mpd,mrd,cqtd")
    parser.add_argument("--disc-batch", type=int, default=12,
                        help="clips per step the discriminators see (memory); reconstruction losses use the whole batch")
    parser.add_argument("--memory-fraction", type=float, help="cap this process's GPU memory (tests beside another run)")
    parser.add_argument("--d-lr", type=float, help="discriminator learning rate (also applied on resume); "
                        "the BigVGAN-v2 discriminators ended their own training near 1.4e-5")
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--validation-interval", type=int)
    parser.add_argument("--log-interval", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    overrides = {k: v for k, v in (("batch_size", args.batch_size), ("gradient_accumulation_steps", args.accumulation),
                                   ("checkpoint_interval", args.checkpoint_interval),
                                   ("validation_interval", args.validation_interval),
                                   ("log_interval", args.log_interval)) if v}
    options = TrainingOptions.from_preset(args.preset, base_model=args.base, prepared_dir=args.dataset,
                                          output_dir=args.output, **overrides).resolved()
    _seed_everything(options.seed)
    device = torch.device("cuda")
    if args.memory_fraction:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    amp = bool(options.amp)
    out = args.output
    latest = out / "checkpoints" / "latest.pth"

    symbols = load_symbols(args.dataset / "symbols.json")
    bundle = build_training_models(args.base, symbols, seed=options.seed)
    disc = AvazDiscriminator(DiscriminatorSpec(families=tuple(args.families.split(","))))
    bundle.generator.to(device)
    disc.to(device)

    optimizer_g = torch.optim.AdamW(_generator_groups(bundle.generator, options), lr=options.learning_rate_g,
                                    betas=options.adam_betas, eps=options.adam_eps, weight_decay=options.weight_decay)
    optimizer_d = torch.optim.AdamW(disc.parameters(), lr=args.d_lr or options.learning_rate_d, betas=options.adam_betas,
                                    eps=options.adam_eps, weight_decay=options.weight_decay)
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=options.lr_decay)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=options.lr_decay)
    scaler = _grad_scaler(amp)

    audio = AudioConfig(sampling_rate=int(bundle.config["data"]["sampling_rate"]),
                        filter_length=int(bundle.config["data"]["filter_length"]),
                        hop_length=int(bundle.config["data"]["hop_length"]),
                        win_length=int(bundle.config["data"]["win_length"]),
                        add_blank=bool(bundle.config["data"].get("add_blank", True)))
    train_loader = create_dataloader(PreparedTTSDataset(args.dataset, "train", symbols, audio), batch_size=options.batch_size,
                                     shuffle=True, num_workers=options.num_workers, seed=options.seed, pin_memory=True)
    validation_loader = create_dataloader(PreparedTTSDataset(args.dataset, "validation", symbols, audio), batch_size=1,
                                          shuffle=False, num_workers=1, seed=options.seed, pin_memory=True)

    if latest.is_file():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        bundle.generator.load_state_dict(state["generator"])
        disc.load_state_dict(state["discriminator"])
        optimizer_g.load_state_dict(state["optimizer_g"])
        optimizer_d.load_state_dict(state["optimizer_d"])
        scheduler_g.load_state_dict(state["scheduler_g"])
        scheduler_d.load_state_dict(state["scheduler_d"])
        scaler.load_state_dict(state["scaler"])
        settings = PolishSettings(**{**state["settings"], "families": tuple(state["settings"]["families"])})
        step, epoch, stage = state["step"], state["epoch"], state["stage"]
        _set_generator_trainable(bundle.generator, optimizer_g, options, stage)
        if stage == STAGE_DECODER:
            for group, saved in zip(optimizer_g.param_groups, state["optimizer_g"]["param_groups"]):
                group["lr"] = saved["lr"]  # keep the decayed rates, not the stage's starting ones
        if args.d_lr:
            for group in optimizer_d.param_groups:
                group["lr"] = args.d_lr  # decays from here with the scheduler
        LOGGER.info("resumed at step %d (%s), discriminator lr %.2e", step, stage, optimizer_d.param_groups[0]["lr"])
    else:
        init = torch.load(args.init, map_location="cpu", weights_only=False)
        bundle.generator.load_state_dict(init["generator"])
        optimizer_g.load_state_dict(init["optimizer_g"])
        step, epoch = int(init["step"]), int(init["epoch"])
        if args.bigvgan:
            copied = load_bigvgan_weights(disc, torch.load(args.bigvgan, map_location="cpu", weights_only=False))
            LOGGER.info("BigVGAN-v2 discriminator tensors copied: %s", copied)
        settings = PolishSettings(start_step=step, warmup_steps=args.warmup_steps, end_step=step + args.steps,
                                  families=tuple(args.families.split(",")))
        bundle.generator.train()
        mel_w, stft_w, seen = _calibrate(bundle, audio, train_loader, device, amp)
        settings = replace(settings, mel_ms_weight=mel_w, stft_weight=stft_w)
        LOGGER.info("calibration on one batch: %s -> mel_ms weight %.3f, stft weight %.3f", seen, mel_w, stft_w)
        out.mkdir(parents=True, exist_ok=True)
        (out / "calibration.json").write_text(json.dumps({**seen, **asdict(settings)}, indent=2))
        stage = _stage(settings, step)
        _set_generator_trainable(bundle.generator, optimizer_g, options, stage)

    (out / "training-options.json").write_text(json.dumps({"max_steps": settings.end_step, **asdict(settings),
                                                           "batch_size": options.batch_size, "disc_batch": args.disc_batch,
                                                           "gradient_accumulation_steps": options.gradient_accumulation_steps}, indent=2))
    segment = int(bundle.config["train"]["segment_size"])
    log_path = out / "metrics.jsonl"
    accumulate = options.gradient_accumulation_steps
    bundle.generator.train()
    disc.train()
    micro = 0
    skipped_g = 0

    while step < settings.end_step:
        epoch += 1
        for batch in train_loader:
            wanted = _stage(settings, step)
            if wanted != stage:
                stage = wanted
                _set_generator_trainable(bundle.generator, optimizer_g, options, stage)
                micro = 0
                LOGGER.info("stage %s from step %d", stage, step)
            x, x_lengths, spec, spec_lengths, waveform = _segment(bundle, audio, batch, device)
            warm = stage == STAGE_WARMUP
            with torch.set_grad_enabled(not warm), _autocast(device, amp):
                generated, duration, _, ids_slice, _, z_mask, latent = bundle.generator(x, x_lengths, spec, spec_lengths)
                z, z_p, m_p, logs_p, _, logs_q = latent
                real = bundle.components.commons.slice_segments(waveform, ids_slice * audio.hop_length, segment)

            # The discriminators see the first disc_batch clips (the loader shuffles, so they vary).
            real_d, generated_d = real[: args.disc_batch], generated[: args.disc_batch]
            for p in disc.parameters():
                p.requires_grad_(True)
            outputs = disc(real_d, generated_d.detach())
            d_parts = {name: discriminator_loss(o.real_scores, o.fake_scores) for name, o in outputs.items()}
            loss_d = sum(d_parts.values())
            scaler.scale(loss_d / accumulate).backward()
            # A discriminator that has blown up (a CQTD burst reached 22,000) sends the generator a huge, meaningless
            # gradient. The discriminator still steps so it can recover; the generator sits this batch out.
            skip_g = not warm and (not torch.isfinite(loss_d) or float(loss_d) > D_SPIKE)
            if skip_g:
                skipped_g += 1
                LOGGER.warning("step %d: discriminator loss %.1f, generator update skipped (%d so far)",
                               step + 1, float(loss_d), skipped_g)

            g_parts, loss_g = {}, None
            with torch.no_grad():
                target_mel = bundle.components.commons.slice_segments(_mel_from_spec(spec, bundle), ids_slice, segment // audio.hop_length)
                old_mel = F.l1_loss(target_mel.float(), _mel_from_waveform(generated.detach().squeeze(1), bundle).float())
            if not warm and not skip_g:
                for p in disc.parameters():
                    p.requires_grad_(False)
                outputs_g = disc(real_d, generated_d)
                g_parts = {name: generator_loss(o.fake_scores) for name, o in outputs_g.items()}
                loss_fm = sum(feature_loss(o.real_maps, o.fake_maps) for o in outputs_g.values())
                loss_adv = sum(g_parts.values())
                loss_mel_ms = multi_scale_mel_loss(generated, real, audio.sampling_rate)
                loss_stft = linear_stft_loss(generated, real)
                loss_duration = duration.float().sum()
                loss_kl = _kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
                loss_g = (settings.adv_scale * (loss_adv + options.feature_loss_weight * loss_fm)
                          + settings.mel_ms_weight * loss_mel_ms + settings.stft_weight * loss_stft
                          + options.duration_loss_weight * loss_duration + options.kl_loss_weight * loss_kl)
                scaler.scale(loss_g / accumulate).backward()
                for p in disc.parameters():
                    p.requires_grad_(True)
            micro += 1
            if micro % accumulate:
                continue

            scaler.unscale_(optimizer_d)
            # Each family is clipped on its own, with BigVGAN's limit: one global norm over all of them let the
            # large pretrained MPD and CQTD use up the budget, so the freshly initialised MRD never learned.
            grad_norms = {name: float(torch.nn.utils.clip_grad_norm_(family.parameters(), DISC_CLIP))
                          for name, family in disc.families.items()}
            scaler.step(optimizer_d)
            if not warm and not skip_g:
                scaler.unscale_(optimizer_g)
                torch.nn.utils.clip_grad_norm_([p for g in optimizer_g.param_groups for p in g["params"] if p.requires_grad],
                                               options.max_grad_norm)
                scaler.step(optimizer_g)
                scheduler_g.step()
            scaler.update()
            optimizer_g.zero_grad(set_to_none=True)
            optimizer_d.zero_grad(set_to_none=True)
            scheduler_d.step()
            step += 1

            row = {"step": step, "wall_time": round(time.time(), 3), "epoch": epoch, "stage": stage,
                   "loss_d": float(loss_d.detach()), "loss_mel": float(old_mel),
                   **{f"d_{k}": float(v.detach()) for k, v in d_parts.items()},
                   **{f"gn_{k}": round(v, 4) for k, v in grad_norms.items()},
                   "peak_memory_gb": round(_peak_memory_gb(device), 3), "peak_reserved_gb": round(_reserved_memory_gb(device), 3)}
            if warm or skip_g:
                row.update({"loss_g": 0.0, "loss_duration": 0.0, "loss_kl": 0.0, "g_skipped": int(skip_g)})
            else:
                row.update({"loss_g": float(loss_g.detach()), "loss_duration": float(loss_duration.detach()),
                            "loss_kl": float(loss_kl.detach()), "loss_mel_ms": float(loss_mel_ms.detach()),
                            "loss_stft": float(loss_stft.detach()), "loss_adv": float(loss_adv.detach()),
                            "loss_fm": float(loss_fm.detach()), **{f"g_{k}": float(v.detach()) for k, v in g_parts.items()}})
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            if step % options.log_interval == 0:
                LOGGER.info("step=%d stage=%s d=%.3f g=%.3f mel=%.4f", step, stage, row["loss_d"], row["loss_g"], row["loss_mel"])
            if step % options.validation_interval == 0:
                info = _validate(bundle, validation_loader, device, out, step, options.validation_seed)
                (out / "validation" / f"step-{step:08d}.json").write_text(json.dumps(info, indent=2))
            if step % options.checkpoint_interval == 0 or step >= settings.end_step:
                payload = dict(generator=bundle.generator.state_dict(), discriminator=disc.state_dict(),
                               optimizer_g=optimizer_g.state_dict(), optimizer_d=optimizer_d.state_dict(),
                               scheduler_g=scheduler_g.state_dict(), scheduler_d=scheduler_d.state_dict(),
                               scaler=scaler.state_dict(), step=step, epoch=epoch, stage=stage,
                               settings=asdict(settings))
                _save(out / "checkpoints" / f"polish-step-{step:08d}.pth", **payload)
                _save(latest, **payload)
                _prune(out / "checkpoints")
                save_inference_checkpoint(out / "exports" / f"model-step-{step:08d}.pth", generator=bundle.generator,
                                          iteration=step, learning_rate=optimizer_g.param_groups[0]["lr"])
            if step >= settings.end_step:
                break
    LOGGER.info("finished at step %d", step)


if __name__ == "__main__":
    main()
