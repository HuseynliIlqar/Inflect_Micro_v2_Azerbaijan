# Training pipeline

This folder records **how the model was made**. Nothing here is needed to run
the model — the [README](../README.md) at the project root is enough for that.

There are two kinds of file here, and the difference matters:

| | |
| --- | --- |
| **Working code** | `fast_monotonic_align.py` — still runs today, together with its 25 tests (`python -m pytest`) |
| **Archive** | `scripts/`, `configs/`, `patches/`, `results/` — used on a RunPod A40 pod. The pod is gone; these are kept as documentation |

---

## The change that mattered most: fast monotonic alignment

The official toolkit's `maximum_path` function was a double loop in pure
Python — roughly **28 million iterations** per step. The GPU sat idle, because
all the time went into the CPU.

`fast_monotonic_align.py` rewrites the same algorithm with vectorised torch
operations — dynamic programming that advances diagonal by diagonal, entirely
on the GPU.

| | Before | After |
| --- | ---: | ---: |
| Step time | 8.7 s | **1.395 s** |
| GPU load | ~0% | **86–100%** |
| 200,000 steps | ~20 days | **3.2 days** |
| Cost (A40 rental) | ~$213 | **~$34** |

**6.2× faster, $179 saved.**

### How correctness was proved

Being fast is not enough — the result has to be **bit-identical**, or the
model learns something else. The original pure-Python implementation is frozen
in `tests/reference_monotonic_align.py`, and 25 tests compare the output of the
two: at random sizes, on edge cases (single row, single column), across mask
variants, and for determinism.

```bash
python -m pytest tests/test_fast_monotonic_align.py -v
```

These tests still run today — no GPU or dataset required.

---

## Other changes

**Multi-speaker support** (`patches/multispeaker.patch`, 939 lines) — unlocks
the toolkit's `emb_g`/`cond` layers and carries the `sid` (speaker id) value
from the manifest all the way to the deployment package. This dataset is
single-voice, so it is **not active**, but the code is ready. Details:
[docs/MULTISPEAKER.md](docs/MULTISPEAKER.md).

**Checkpoint retention** — only the last 5 heavy checkpoints are kept. Disk
usage drops from 27.6 GB to 3.5 GB; `exports/` is left untouched.

**VRAM reporting** — `peak_memory_gb` and `peak_reserved_gb` are written to
`metrics.jsonl` on every step. The formula that came out of it:

```
reserved GB = 1.215 + 0.3137 × batch        (R² = 0.998, measured on an RTX 4050)
```

This formula made it possible to work out in advance which preset would fit,
before renting a GPU.

The patch applies to the official toolkit at the commit in
`patches/BASE_COMMIT.txt`.

---

## Results (`results/`)

| File | What it is |
| --- | --- |
| `training-summary.json` | Final state: 200,000 steps, 1,343 epochs, stage `decoder_polish` |
| `compatibility-report.json` | Warm start report — 409 of 410 tensors copied bit-identical from the base |
| `run-identity.json` | Base checkpoint SHA-256, optimizer layout |
| `training-options.json` | Every parameter that was used |
| `config.json` | The run's full configuration |
| `loss-curve.csv` | Loss history over 200,000 steps, one row every 500 steps (401 rows) |

The path the loss took:

| Step | `loss_mel` | `loss_kl` | Stage |
| ---: | ---: | ---: | --- |
| 1 | 1.881 | 182.98 | `posterior_warmup` |
| 500 | 0.916 | 8.73 | `posterior_warmup` |
| 200,000 | 0.415 | 1.98 | `decoder_polish` |

**Note:** `best.pth` is deliberately absent. No loss value identifies the
best-*sounding* model — the checkpoint was chosen by ear. Why:
[../docs/MODEL.md](../docs/MODEL.md).

---

## Presets (`configs/`)

| Preset | batch × accum | Min VRAM | Steps |
| --- | ---: | ---: | ---: |
| `az-smoke.json` | 8 × 1 | 4 GB | 400 |
| `az-multispeaker-24gb.json` | 16 × 4 | 8 GB | 200,000 |
| `az-multispeaker-quality.json` | 32 × 2 | 16 GB | 200,000 |
| **`az-large-gpu.json`** ← the one used | 64 × 1 | 32 GB | 200,000 |

All four give the **same effective batch (64)**, so their results are
comparable. The only difference is how they fit into memory.

---

## English base model (`base-model/`)

The starting point of the adaptation:
[`owensong/Inflect-Micro-v2`](https://huggingface.co/owensong/Inflect-Micro-v2)
— the unmodified upstream package (38 MB). The Azerbaijani weights are built
on top of it.

Why it is kept: to hear what the adaptation changed, or to check the warm start
claim in `compatibility-report.json` yourself.

Making it speak English (normalisation and splitting are turned off — those are
for Azerbaijani):

```bash
python say.py --model training/base-model --raw --max-words 0 \
  -o out/base "Hello, this is the original English base model."
```

If it needs downloading again: `python training/scripts/download_model.py`.

---

## Archived scripts (`scripts/`)

These were run on the RunPod pod. The pod is gone and they are kept **as they
were** — to read as documentation, not to run directly. Most of them expect
either the `inflect-adapt` toolkit (not included in this folder) or the pod's
`/root/work` paths.

| File | What it did |
| --- | --- |
| `runpod_setup.sh` | venv, CUDA wheels and toolkit installation on the pod |
| `train_az.sh`, `train_bg.sh` | Starts training; `train_bg.sh` runs detached (an SSH drop does not stop it) |
| `status.sh` | Checks progress |
| `build_az_dataset.py` | Builds the dataset from HuggingFace (uses `hf_dataset.py`) |
| `smoke_multispeaker.py` | End-to-end CPU check of the multi-speaker patch |
| `download_model.py` | Downloads the English base from HF → `base-model/` |
| `compare_checkpoints.py` | Renders the same sentences from several checkpoints |

Full steps for building from scratch:
[docs/TRAINING_PLAN.md](docs/TRAINING_PLAN.md).

**When copying files from Windows to the pod:** CRLF line endings break the
bash scripts (`set: pipefail: invalid option name`). The fix:
`sed -i 's/\r$//' scripts/*.sh`.

---

## Continuing training

It is possible, but the resume checkpoint (`adaptation-final.pth`, 665 MB) is
not included in this folder — because of its size. It is in the original
project if it is needed.

What you need to know:

- **26 GB of VRAM** is required (not possible on an ordinary laptop card);
- Resume **refuses** if the configuration changed — `configs/az-large-gpu.json`
  has to stay as it is;
- The dataset must have been built **together with its audio** (metadata alone
  is not enough).
