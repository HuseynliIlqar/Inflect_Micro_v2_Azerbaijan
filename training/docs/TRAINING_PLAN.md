# Azerbaijani training plan

This document describes the configuration and the reason behind each choice.
**Training has not started yet** — what is here is only preparation.

> **Dataset: single speaker.** `ughurabbasov/azerbaijani-tts-dataset` is one voice
> across all shards (F0 195–201 Hz, std 11–15) — see [DATASET.md](DATASET.md). So
> `n_speakers` stays at 0 and the **official, tested single-voice path** is used;
> the multi-speaker patch ([MULTISPEAKER.md](MULTISPEAKER.md)) is installed but is
> not activated in this run. The presets are identical in either case.
>
> The labelling is in multi-speaker form regardless: every row carries a `speaker`
> field and the `az-01-main` convention keeps the id order stable — see
> [DATASET.md](DATASET.md#labelling--ready-for-future-voices).

## Configuration

There are three presets in the `configs/` directory. A preset JSON can set **any**
field of `TrainingOptions` — knobs that are not on the CLI (lr multipliers, loss
weights, `lr_decay`) are controlled only from there.

| | `az-smoke` | `az-multispeaker-24gb` | `az-multispeaker-quality` | `az-large-gpu` |
|---|---:|---:|---:|---:|
| **Minimum VRAM** | 4 GB | **8 GB** | **16 GB** | **32 GB** |
| `batch_size` | 8 | 16 | 32 | 64 |
| `gradient_accumulation_steps` | 1 | 4 | 2 | 1 |
| **Effective batch** | 8 | **64** | **64** | **64** |
| `max_steps` | 400 | 200,000 | 200,000 | 200,000 |
| `learning_rate_g/d` | 2e-4 | 2e-4 | 2e-4 | 2e-4 |
| `posterior_warmup_steps` | 50 | 5,000 | 5,000 | 5,000 |
| `decoder_unfreeze_step` | 200 | 30,000 | 30,000 | 30,000 |
| `checkpoint_interval` | 200 | 5,000 | 5,000 | 5,000 |
| `checkpoint_retention` | 2 | 5 | 5 | 5 |

The three full presets give the **same effective batch (64)** — they differ only by
the memory of the card, so their results can be compared directly. The VRAM figures
were measured; see the section below.

## Rationale for the choices

**Effective batch 64** — the batch size of the original VITS paper. The stability of
GAN training is sensitive to batch size; a small batch makes the discriminator
fluctuate.

**`learning_rate` 2e-4** — the from-scratch learning rate of VITS. The toolkit
default of 8e-5 is conservative and is meant for a *single voice change*. Here the
language **and** the voices change, meaning the linguistic path has to move a lot.

**LR multipliers** — the actual LR per stage:

| Group | Multiplier | Starting LR | Reason |
|---|---:|---:|---|
| `posterior` | 1.0 | 2.0e-4 | `enc_q` is entirely random, it needs full speed |
| `linguistic` | 1.0 | 2.0e-4 | `enc_p`/`dp`/`flow` have to learn the rhythm of the new language |
| `speaker` | 1.0 | 2.0e-4 | `emb_g` + `cond` start random — they must not be choked |
| `decoder` | **0.25** | 5.0e-5 | The most valuable part of the warm start; fast change gives buzz/metallic audio |

The toolkit default for the decoder is 0.1. I raised it to 0.25, because **timbre
lives in the decoder** — it has to move for new voices. This is a trade-off: too low
= the voices sound alike, too high = artifacts. Listen to the validation clips and
adjust.

**Stage boundaries** — the defaults are for 20k steps (2.5% and 15%). I scaled them
to 200k with the same ratios:

```
0        →  5,000   posterior_warmup      only enc_q + speaker
5,000    → 30,000   linguistic_adaptation + enc_p, dp, flow
30,000   → 200,000  decoder_polish        + dec (0.25× LR)
```

**`lr_decay` 0.99999** (per step) — at the end of 200k steps the LR is **13.5%** of
the starting value (2.7e-5). That is appropriate for this scale; for a shorter run
the decay has to be increased.

**Loss weights** — `mel=45.0`, `kl=1.0`, `feature=1.0`, `duration=1.0`. These are the
VITS standard, I did not change them.

**`max_steps` 200,000 is my judgement**, there is no figure in the documentation.
Context: VITS is trained from scratch for ~800k steps on LJSpeech (24 hours). Here we
have a warm start, so fewer are enough — but the toolkit's 20k default is clearly too
few for a change of this scale. **Trust the checkpoint selection, not the number**:
40 checkpoints are written, stop when validation stops improving.

## How much VRAM is needed

I did not estimate — I measured it for real on a local **RTX 4050 Laptop (6.44 GB)**.
The trainer now writes `peak_memory_gb` and `peak_reserved_gb` at every step
(`metrics.jsonl`).

The measurement was done with a **worst case** corpus (12–14 s clips, 12.88 s mean),
because the dataloader does not group by length — every batch is padded to its own
longest clip, meaning a single 14 s clip makes the whole batch expensive.

| batch | allocated | reserved |
|---:|---:|---:|
| 1 | 1.44 GB | 1.62 GB |
| 2 | 1.70 GB | 1.87 GB |
| 4 | 2.27 GB | 2.39 GB |
| 8 | 3.35 GB | 3.65 GB |
| 12 | 4.44 GB | 4.93 GB |
| 16 | 5.43 GB | 6.32 GB |

The scaling is perfectly linear:

```
reserved GB = 1.215 + 0.3137 x batch      (R2 = 0.998)
```

**OOM is determined by `reserved`**, not `allocated` — the allocator holds on to free
blocks and that is the number the driver sees. On top of that the CUDA context
(~0.5 GB) and a fragmentation margin are needed.

### Recommendation (keeping effective batch 64)

| Preset | batch × accum | measured/computed | **minimum card** | comfortable |
|---|---|---:|---:|---:|
| `az-multispeaker-24gb` | 16 × 4 | 6.2 GB → 7.7 GB | **8 GB** | 12 GB |
| `az-multispeaker-quality` | 32 × 2 | 11.3 GB → 13.5 GB | **16 GB** | 24 GB |
| `az-large-gpu` | 64 × 1 | 21.3 GB → 25.1 GB | **32 GB** | 48 GB |

(The "→" column: reserve + context + 15% fragmentation margin.)

**Short answer: 16 GB is the minimum, 24 GB is recommended.** On RunPod an
RTX 4090 / L4 / A10 (24 GB) is enough for this job and is cheap; an A100 80 GB is not
needed.

`az-large-gpu` (batch 64, accum 1) is **the fastest** — one pass without accumulation
means less forward/backward overhead. If you have a 32 GB+ card, choose this; the
effective batch is the same, so the results are comparable.

### Warnings

- **The batch 32 and 64 figures are extrapolation**, not measurement — they cannot be
  measured on a 6.44 GB card. The regression is very clean (R²=0.998), but
  **confirm it with a smoke run**: run `bash scripts/train_az.sh smoke` and look at
  `peak_reserved_gb` in `metrics.jsonl`.
- On the 4050, batch 16 (6.32 GB) only passed on a 6.44 GB card thanks to Windows
  shared memory. **On Linux it would have been an OOM** — that is the real limit.
- If you hit an OOM: halve `--batch-size`, double
  `--gradient-accumulation-steps`. The effective batch stays fixed and the run stays
  comparable.

### One optimization opportunity

The dataloader does **not** bucket by length (only `shuffle` in `training_data.py`).
A sampler that groups by length would reduce both memory and time noticeably, because
a 9.33 s mean clip is padded up to 14 s. This is not implemented at present — the
official behaviour was kept.

## How many epochs there will be

The real size of the dataset was measured: **9,674 clips, 25.07 hours, mean clip
9.33 s** (see [DATASET.md](DATASET.md)). After splitting off 2% validation,
**9,481 training clips**.

| Preset | effective batch | steps/epoch | max_steps | **epochs** |
|---|---:|---:|---:|---:|
| `az-smoke` | 8 | 1185 | 400 | **0.34** |
| `az-multispeaker-quality` | 64 | 148 | 200,000 | **1,350** |
| `az-multispeaker-24gb` | 64 | 148 | 200,000 | **1,350** |

The two quality presets use the same effective batch, so the epoch count is the same
too — only the wall clock differs.

### "Epoch" is a partial measure here

`segment_size = 16384` samples = **0.683 s**. Only the model's **waveform decoder and
discriminator** see a random window of that size from the clip (`models.py:531`):

```python
z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
o = self.dec(z_slice, g=g)
```

The text encoder, duration predictor, flow and posterior encoder see the **whole
sentence**. Since the mean clip is 9.33 s, there are ~13.7 such windows per clip:

| | over 1,350 epochs |
|---|---:|
| How many times each clip is shown (text/duration/flow) | **1,350** |
| How many times each 0.683 s region reaches the decoder | **~99** |

So the linguistic path is well saturated, while the decoder sees considerably fewer
repetitions — this is a further reason for keeping the decoder in the
`decoder_polish` stage and at a low LR.

### Comparison

VITS on LJSpeech (13,100 clips, batch 64 → 205 steps/epoch) is typically trained for
~800k steps, that is **~3,900 epochs** — from scratch. Our 1,350 epochs is roughly a
third of that, but there is a warm start: 410/410 tensors are transferred and the
phoneme inventory overlaps completely. The number is reasonable, **but it is not
proof** — checkpoint selection has to be done by listening to validation.

## Checkpoint system

At every `checkpoint_interval` (5,000 steps) **two** files are written:

| File | Contents | Size | What it is for |
|---|---|---:|---|
| `checkpoints/adaptation-step-*.pth` | model + discriminator + optimizer + RNG | **~690 MB** | Only for continuing training |
| `exports/model-step-*.pth` | speaking model only | 37.6 MB | **Selecting the final model** |

Most of the heavy file is the discriminator (46.7M parameters, while the generator is
11.2M), plus AdamW keeps two moments per parameter.

Writing is atomic (`.tmp` → `os.replace`), so if the pod dies mid-write no partial
file is left behind.

The checkpoint grows with the stage: ~621 MB during `posterior_warmup` (there is no
optimizer state yet for the frozen groups), ~696 MB after `decoder_polish`.

### Retention — old heavy files are deleted

`checkpoint_retention: 5` is set in the presets: only the **last 5** heavy
checkpoints are kept, older ones are deleted after every save.

- `latest.pth` and `adaptation-final.pth` are **never** deleted (they do not match
  the name pattern)
- `exports/` is **never** touched — that is exactly where you will pick the final
  model from
- `checkpoint_retention: 0` keeps everything (the official behaviour)

Disk impact at 200,000 steps: **27.6 GB → 3.5 GB**.

It can also be given from the CLI: `--checkpoint-retention N`.

### Automatic resume

`train_az.sh` checks whether the file `checkpoints/latest.pth` exists:

- **present** → continues automatically with `--resume`
- **absent** → starts from scratch

After the pod stops you type **the same command**, that is all:

```bash
bash scripts/train_az.sh large
```

If you pass `FRESH=1` and there is an existing run, the script **refuses** — this
prevents accidentally overwriting it.

**Important limitation:** a resume is **rejected** if the configuration changes.
`validate_run_identity` compares all options (batch_size, max_steps, learning rate,
preset values...) as canonical JSON. So you cannot change `--max-steps` midway and
continue — you have to start a new run with a new `--output` directory.

A resume starts **from the last save point**, not from the exact step where it
stopped: if it dies at 52,300 and the last save was 50,000, 2,300 steps are repeated.
`checkpoint_interval` is kept at 5,000 for that reason — so the loss stays small.

### Verified

Retention: a 12-step run, interval 2, retention 3 → 2/4/6 were deleted, 8/10/12
remained, all 7 files in `exports/` were in place.

Resume: the run was killed at step 10, the same command was launched again → it
continued from step 11 and went up to 60.

## Tooling that has been prepared

### `inflect-preflight` — before renting a GPU

```powershell
inflect-preflight --manifest data\metadata.jsonl --audio-root data\audio --language az
```

It checks: manifest format, that every file can be read, **hours/clip share per voice
and the speaker id map**, clips that are too short/too long, stereo/sample rate,
digits in the transcript, duplicate text, **eSpeak phonemization + match against the
base inventory**, and the epoch count for the chosen effective batch.

If there is a blocker it returns exit code 1. Example output:

```
speaker                    clips    hours   share   median     p95
aysel                       4210    12.40   47.7%     9.80   14.20   -> speaker=0
resad                       4655    13.60   52.3%    10.10   13.90   -> speaker=1

unknown to base model  : 0
   every phoneme already exists in the base inventory (full warm start)
```

### `scripts/runpod_setup.sh` — pod preparation

Clones the pinned commit, applies the multi-speaker patch, installs CUDA torch,
downloads the base model, runs the toolkit tests and reports the GPU.

### `scripts/train_az.sh` — launcher

```bash
bash scripts/train_az.sh smoke      # 400 steps, speed measurement
bash scripts/train_az.sh quality    # full run
bash scripts/train_az.sh quality --resume runs/az-quality/checkpoints/latest.pth
```

## RunPod environment

The **RunPod PyTorch 2.4.0** template is suitable. The only thing needed is **SSH
access** (the template has it, port 22). Jupyter and NGINX are not needed for this
job, but they do no harm.

### Three issues

**1. The template's torch is old.** The template brings 2.4.0, while the base model's
`requirements.txt` asks for `torch>=2.6`. I did not find any 2.6-specific API in the
code (`weights_only` is passed explicitly everywhere), so 2.4 would most likely work
— but `runpod_setup.sh` installs a new torch anyway, so the issue does not arise.

**2. CUDA version.** When creating the pod, choose **CUDA 12.8** — by default the
script installs `cu128` wheels. If you chose 12.4:

```bash
CUDA_TAG=cu124 bash scripts/runpod_setup.sh
```

**3. Training dies when SSH drops.** This is the most important one. 200,000 steps
take hours; if you start it from an ordinary SSH session or a Jupyter terminal, the
process gets `SIGHUP` and stops as soon as the connection drops. On RunPod this also
happens when the browser is refreshed.

For that reason **`train_bg.sh`** was added — it launches training detached:

```bash
bash scripts/train_bg.sh large          # start, the terminal may be closed
bash scripts/train_bg.sh large tail     # follow the log
```

It uses `tmux` if available, otherwise `setsid nohup` — nothing has to be installed
in advance. If you type the same command again while a run is in progress, it warns
you and does not start a second copy.

After reconnecting:

```bash
bash scripts/train_bg.sh large          # if latest.pth exists, continues from there
```

### Splitting the storage (when there is no network volume)

There are two disks on RunPod and they behave differently:

| Disk | Mount | When the pod is stopped |
|---|---|---|
| Container disk | `/`, `/root` | **deleted** |
| Volume disk | `/workspace` | kept (until the pod is deleted) |

The scripts' default `WORKDIR` is `/workspace`, meaning they **write everything to
the volume**. The whole project takes ~27 GB — it will not fit in a 20 GB volume. So
it has to be split.

**Recommendation: Container disk 50 GB, Volume 20 GB.**

| Volume `/workspace` (kept) | | Container disk (deleted) | |
|---|---:|---|---:|
| `prepared/az` | 4.3 GB | venv + torch/CUDA | 7.0 GB |
| `runs/` | 6.5 GB | HF parquet cache | 4.6 GB |
| | | `data/az` raw WAV | 4.3 GB |
| | | pip temp + reserve | 3.0 GB |
| **TOTAL** | **10.8 GB** | **TOTAL** | **18.9 GB** + image |

Only the things that are expensive to recreate stay on the volume. If the pod stops,
the venv and the raw WAVs are lost — but they can be rebuilt in 15 minutes, while the
checkpoints stay in place.

```bash
# 1. Setup - on the container disk
export WORK=/root/work
export HF_HOME=$WORK/hf-cache
bash scripts/runpod_setup.sh $WORK

# 2. Dataset - raw WAV on the container, prepared on the volume
source $WORK/Inflect/.venv/bin/activate
python -m pip install pyarrow -e /workspace/project     # path of this project
python scripts/build_az_dataset.py --output $WORK/data/az
inflect-preflight --manifest $WORK/data/az/metadata.jsonl \
    --audio-root $WORK/data/az --language az
inflect-adapt prepare --manifest $WORK/data/az/metadata.jsonl \
    --audio-root $WORK/data/az --language az --frontend espeak \
    --output /workspace/prepared/az \
    --min-duration-seconds 1.0 --max-duration-seconds 14.0 --validation-fraction 0.02

# 3. Train - result on the volume
export BASE=$WORK/base
export DATASET=/workspace/prepared/az
export OUTPUT=/workspace/runs/az-large
bash scripts/train_bg.sh large
```

`train_bg.sh` passes these variables explicitly into the tmux session — because an
already running tmux server does not inherit the environment.

**Warning:** the volume is "tied to the life of the pod" — if you **stop** the pod it
stays, but if you **terminate** it, it is deleted. So download good checkpoints to a
local machine from time to time. The `exports/model-step-*.pth` files are only
37.6 MB:

```bash
runpodctl send /workspace/runs/az-large/exports/model-step-00100000.pth
```

### vCPU

The presets have `num_workers: 8`. If the pod has 9+ vCPUs there is no problem. If it
has fewer, change it:

```bash
bash scripts/train_az.sh large --num-workers 4
```

The model is small (11.2M) and the GPU steps are fast — the dataloader (WAV reading +
STFT) can be a bottleneck on the CPU. The smoke run shows this: if GPU utilization is
low and steps/second are below expectation, the bottleneck is the CPU.

## The order when training starts

**0. Build the dataset** — `python scripts/build_az_dataset.py --output data/az`.
It downloads the parquet shards, filters them and writes the manifest. Filtering has
to happen here: `prepare` stops the whole run at the first clip it rejects.

**1. Preflight** (local, no GPU) — until no blocker remains.

**2. Prepare** — change the defaults for quality:

```bash
inflect-adapt prepare --manifest data/metadata.jsonl --audio-root data/audio \
  --language az --frontend espeak --output prepared/az \
  --min-duration-seconds 1.0 --max-duration-seconds 14.0 --validation-fraction 0.02
```

The toolkit default is `min-duration 0.05s` and the maximum is unlimited — both are
harmful for quality. At 26 hours, 2% validation ≈ 30 minutes, which is enough.

**3. Audit** — `inflect-adapt audit --dataset prepared/az`.
No unknown symbol should remain.

**4. Smoke run** — `bash scripts/train_az.sh smoke`. Two purposes:
   - The pipeline works and there is no OOM
   - **steps/second is measured** → the real time for 200k steps is computed

If you hit an OOM: halve `--batch-size`, double `--gradient-accumulation-steps`. The
effective batch stays fixed and the run stays comparable.

**5. Full run** — `bash scripts/train_bg.sh large` (detached; it continues even if
SSH drops). Log: `bash scripts/train_bg.sh large tail`.

**6. Checkpoint selection.** The toolkit deliberately does not create a `best.pth`.
Every 2,500 steps `runs/az-quality/validation/step-XXXXXXXX-speakerNN.wav` is written
— the same text, the same seed, for every voice. Declare the selection rule **before
you look at the test set**. While listening, look for: clarity, buzz/metallic
resonance, cut-off endings, whether the voices are distinguishable from each other,
sibilant instability.

**7. Export** (not ONNX, it does not work for multi-speaker):

```bash
inflect-adapt export --checkpoint runs/az-quality/checkpoints/adaptation-step-XXXXXXXX.pth \
  --prepared-dataset prepared/az --package-template ./base \
  --format pytorch --output exports/az-quality
```

## What is not ready

- **The full dataset has not been extracted locally** — all 10 shards were measured
  (9,674 clips), but the WAVs will be written on RunPod with a single command.
- **A real time estimate** — it will be known after the smoke run. I do not want to
  state a number in advance.
- **`decoder_lr_multiplier` 0.25 has not been tested** — this is my judgement, it has
  to be adjusted after the first validation clips.
- **Emotion** — it is not in this plan; as decided, it is left for later.
