# Multi-speaker Azerbaijani adaptation

The official `inflect-finetune` toolkit supports only a single voice.
`patches/multispeaker.patch` makes it multi-speaker: speaker ids are carried from the
manifest all the way through to deployment.

Base commit: `patches/BASE_COMMIT.txt`

## What changed

| File | Change |
|---|---|
| `prepare.py` | A multi-speaker manifest is accepted; a sorted `speakers` list is written into `dataset.json`. A row without a label, or a voice that does not appear in training, is rejected. |
| `training_data.py` | The dataset and the collate function carry a `sid` tensor for every row. |
| `modeling.py` | `gin_channels=256` is turned on when `n_speakers >= 2`; the choice is written into `config`. `n_speakers=1` is rejected explicitly (it crashes in stock VITS). |
| `checkpoint.py` | The warm start allows new `emb_g` / `cond` tensors (only in multi-speaker mode). |
| `training.py` | `sid` is passed into the forward; the conditioning parameters are in a separate **`speaker`** group and are open **in all stages**; validation writes a separate clip for every voice; `config.json` is saved into the run directory. |
| `exporting.py` | Deployment accepts `InflectTTS.synthesize(..., speaker=N)` and checks the bound; ONNX export gives a clear error for a multi-speaker checkpoint. |

**Note:** the conditioning parameters are learned from the `posterior` stage onwards.
Otherwise the voices could not be separated until the decoder is unfrozen.

## Workflow on RunPod

```bash
git clone https://github.com/owenawsong/Inflect.git
cd Inflect && git checkout $(cat BASE_COMMIT.txt)
git apply /workspace/patches/multispeaker.patch

python -m venv .venv && . .venv/bin/activate
pip install -e ./finetune
pip install huggingface_hub && hf download owensong/Inflect-Micro-v2 --local-dir base
```

### 1. Manifest (JSONL) — one clip per row

```json
{"audio":"aysel/000001.wav","text":"Salam, bu gün hava çox gözəldir.","speaker":"aysel"}
{"audio":"resad/000001.wav","text":"Qapını örtdü və çıxdı.","speaker":"resad"}
```

`speaker` must be present **on every row** and the label must be in the `az-01-main`
format: ids are assigned in alphabetical order, so an unnumbered label can later
renumber all the voices — see
[DATASET.md](DATASET.md#one-trap-ids-are-assigned-in-alphabetical-order). Give the
same `group_id` to clips cut from the same source recording so that they do not leak
between training and validation.

### 2. Prepare

```bash
inflect-adapt prepare --manifest data/metadata.jsonl --audio-root data/audio \
  --language az --frontend espeak --output prepared/az
```

Check the `speakers` list in `dataset.json` — the order is the id (`speakers[0]` → `speaker=0`).

### 3. Audit (no GPU needed)

```bash
inflect-adapt audit --dataset prepared/az
```

There should be no unknown symbols. The Azerbaijani eSpeak output fits entirely
inside the base's 178-symbol IPA inventory, so no new symbol is expected here.

### 4. Training

```bash
inflect-adapt train --base ./base --dataset prepared/az \
  --preset balanced --output runs/az-micro --device cuda
```

`runs/az-micro/validation/step-XXXXXXXX-speakerNN.wav` — a clip with the same text and
the same seed for every voice. This is the main tool for comparing the voices.

### 5. Export and use

```bash
inflect-adapt export --checkpoint runs/az-micro/checkpoints/adaptation-final.pth \
  --prepared-dataset prepared/az --package-template ./base \
  --format pytorch --output exports/az-micro
```

```python
from inference import InflectTTS
tts = InflectTTS("exports/az-micro", device="cuda")
tts.save("Salam dünya.", "aysel.wav", speaker=0)
tts.save("Salam dünya.", "resad.wav", speaker=1)
```

ONNX export **does not work** for a multi-speaker checkpoint — the graphs have no
speaker input.

## Size impact

| | Parameters |
|---|---:|
| Base (single voice) | 9,356,513 |
| Multi-speaker (`gin_channels=256`) | 10,258,049 (**+9.6%**) |

**410/410 tensors** of the released checkpoint are transferred as they are; only 17
new conditioning tensors start from scratch.

## Local check

```powershell
.\.venv\Scripts\python.exe scripts\smoke_multispeaker.py
```

It builds a synthetic 2-voice corpus and checks the prepare → train → export →
inference chain end to end on the CPU. It says nothing about audio quality, it only
tests the connections.

## Limitation

9.36M parameters is small for multi-speaker (for comparison: VCTK VITS ~83M). The
capacity is divided between the voices. If there are not enough hours for each voice,
each of them will come out worse than the single-voice variant.
