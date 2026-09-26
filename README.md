# Azerbaijani TTS

A **fully offline** text-to-speech model that speaks Azerbaijani. 9.36M
parameters, 24 kHz mono, **2-4x faster than real time** on an ordinary laptop
CPU. No server, no API key, no internet.

> Azərbaycanca oxumaq üçün: [README.az.md](README.az.md)

```bash
python say.py "Salam, necəsiniz?"
```

The audio lands in `out/01.wav`. That is the whole thing.

---

## Install

Python 3.11 or newer. The repository stores the model weights with
[Git LFS](https://git-lfs.com), so install it before cloning:

```bash
git lfs install
git clone https://github.com/<user>/azerbaycan-tts
cd azerbaycan-tts
```

```powershell
# Windows
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe say.py "Salam dünya."
```

```bash
# Linux / macOS
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python say.py "Salam dünya."
```

PyTorch is large. If you are only going to run on the CPU, take the CPU wheel --
it is considerably smaller:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The model weights (`model/`, 37 MB) are in the repository -- there is nothing
extra to download.

## Usage

### Command line

```bash
python say.py "One sentence."                     # -> out/01.wav
python say.py "First." "Second."                  # -> out/01.wav, out/02.wav
python say.py -f book.txt --one-file -o out/book  # whole file into one WAV
python say.py --show-text "II Dünya, 25% artım"   # show what reaches the model
python say.py --speed 0.85 --seed 42 "Daha yavaş."
```

| Flag | Default | What it does |
| --- | ---: | --- |
| `--speed` | `1.0` | 0.5-2.0. Lower is slower |
| `--variation` | `0.667` | 0.0-1.0. Lower is steadier, higher is livelier |
| `--seed` | `7` | The same seed gives the same voice. Change it for a sentence you do not like |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--out` | `out/` | Output directory |
| `--raw` | -- | Turn normalisation off (see below) |
| `--max-words` | `15` | Chunk length. `0` leaves splitting to the model |
| `--prosody` | `off` | `safe` / `wide` -- thins out stress marks (experimental) |
| `--show-text` | -- | Print the normalised text and the chunk boundaries |

### From Python

```python
from aztts import AzTTS

tts = AzTTS()                       # build once, reuse
tts.save("Mətn burada.", "out/a.wav")

waveform = tts.synthesize("Xam massiv lazımdırsa.", speed=1.1, seed=3)
# -> float32 numpy array, mono, [-1, 1], 24 000 Hz
```

The text helpers do not depend on the model and can be used on their own:

```python
from aztts import normalize_az, chunk_text

normalize_az("II qrupda 25% artım oldu.")
# 'İkinci qrupda iyirmi beş faiz artım oldu.'

chunk_text("Uzun bir cümlə...", max_words=15)
# ('Uzun bir cümlə...',)
```

## What happens to the text before it reaches the model

The model was trained on ordinary prose, so digits and abbreviations come out
wrong if they reach it raw -- `II` is spoken as "ı ı", `25%` loses the percent
sign entirely. Two steps prevent that:

**1. Normalisation** (`aztts/az_text.py`) -- digits, Roman numerals, dates,
percentages, units and abbreviations become spoken words:

| Input | What reaches the model |
| --- | --- |
| `II Dünya müharibəsi` | `İkinci Dünya müharibəsi` |
| `01/09/1939` | `birinci sentyabr min doqquz yüz otuz doqquz` |
| `25%` | `iyirmi beş faiz` |
| `5 kq` | `beş kiloqram` |

**2. Chunking** (`aztts/az_chunk.py`) -- sentences are cut into chunks of about
fifteen words. The model is small, and towards the end of a long sentence the
intonation flattens and word endings get clipped. A pause proportional to the
punctuation is inserted between chunks, so the result does not sound choppy.

Pass `--show-text` to see what is happening. Pass `--raw` to turn it off.

## Samples

`samples/` holds five ready-made WAVs -- listen to them to hear how the model
sounds. To regenerate them:

```bash
python say.py --out samples
```

## Quality tools

```bash
# Render one line under 10 different seeds and pick the best
python tools/seed_sweep.py --seeds 10 "Xoş gəlmisiniz."

# Find the vocoder's metallic resonance, then clean it with ffmpeg
python tools/audio_postprocess.py analyze samples
python tools/audio_postprocess.py clean out --lowpass 11000
```

For lines an application repeats (a greeting, an error prompt) it is worth
picking a seed once and hard-coding it -- the reading improves noticeably.

## Tests

```bash
python -m pytest
```

141 tests:

| | |
| --- | ---: |
| Text normalisation, chunking, the stress layer | 116 |
| Fast monotonic alignment (a training optimisation) | 25 |

The tests do not load the model -- they finish in three seconds.

## Layout

```
say.py            Command line -- the main entry point
aztts/
  engine.py       AzTTS -- model loading, normalisation, chunking, synthesis
  az_text.py      Digits / Roman numerals / abbreviations -> words
  az_chunk.py     Cuts sentences to a length the model handles
  az_prosody.py   Stress layer (optional, --prosody)
  console.py      Switches the Windows console to UTF-8
model/            Azerbaijani weights + runtime (37 MB) -- do not edit
tools/            seed_sweep, audio_postprocess -- quality tools
training/         How the model was made (the base model is downloaded)
packaging/        Publishing to GitHub / Hugging Face / Kaggle
samples/          Example audio
tests/            141 tests
out/              The WAV files you generate
```

Publishing steps for all three platforms are in
[packaging/PUBLISHING.md](packaging/PUBLISHING.md). If you are working on the
code -- with or without an AI assistant -- start from [CLAUDE.md](CLAUDE.md):
it lists the setup, the verification commands and the traps that are not
obvious from reading the source. Cursor picks the same rules up automatically
from `.cursor/rules/`.

`model/` is the upstream export package and brings everything it needs with it
(weights, phoneme frontend, VITS runtime). All of our own code lives under
`aztts/`.

## About the model

| | |
| --- | --- |
| Base | [`owensong/Inflect-Micro-v2`](https://huggingface.co/owensong/Inflect-Micro-v2) |
| Parameters | 9.36M |
| Voice | 24 kHz mono, single speaker |
| Training | 200,000 steps = 1,343 epochs, 25 hours of Azerbaijani audio |
| Speed | 2-4x real time on CPU |

For the details -- the training process, the dataset, measurements and
limitations -- see [docs/MODEL.md](docs/MODEL.md).

## How the model was made

[`training/`](training/README.md) holds the whole process: the patch applied to
the official toolkit, the presets, the training results and the loss history.

The most important part: the toolkit's `maximum_path` function was a pure Python
loop that ate 8.7 seconds of every step. A vectorised torch version brought that
down to **1.395 seconds**: 20 days became 3.2 days, ~$213 became ~$34. The
result is proven **bit-identical** to the original by 25 tests, and those tests
still run.

The English base model (the starting point for the adaptation) is not stored in
this repository -- it is a 38 MB upstream snapshot. Fetch it when you need it:

```bash
pip install huggingface_hub
python training/scripts/download_model.py
python say.py --model training/base-model --raw --max-words 0 -o out/base "Hello, this is the English base model."
```

## Limitations

- **One voice.** There is no voice cloning and no multi-speaker support.
- **No emotion control.** The reading is calm and neutral.
- The model is small (9.36M) -- the result is clear and intelligible, but not
  fully natural. The vocoder sometimes leaves a metallic resonance.
- Rare names, foreign words and unusual spellings are at the mercy of the
  phoneme frontend.
- Normalisation runs automatically inside `say.py` and `AzTTS`. If you call the
  `model/` package directly, you have to call `normalize_az` yourself.

## Credits

This model exists because two people published their work openly.

**Base model --- [`owensong/Inflect-Micro-v2`](https://huggingface.co/owensong/Inflect-Micro-v2)**
by Owen Song, Apache-2.0. Every weight here started as one of his: 409 of the
410 tensors were carried over bit-identically, and the Azerbaijani phonemes were
already in his symbol set. Without that checkpoint, this would have been months
of training instead of days.

**Training data --- [`ughurabbasov/azerbaijani-tts-dataset`](https://huggingface.co/datasets/ughurabbasov/azerbaijani-tts-dataset)**
by `ughurabbasov`. 25.07 hours of single-speaker Azerbaijani audio. The
repository declares no licence file; asked directly in the dataset's Hugging
Face community tab, the author confirmed that anyone may use it and asked to be
credited. **Attribution is the condition of use, so if you build on this model,
credit the dataset too.**

If you use this project, please cite it and both sources --- see
[CITATION.cff](CITATION.cff):

```bibtex
@software{azerbaijani_tts,
  title  = {Azerbaijani TTS: an offline 9.36M-parameter VITS model},
  author = {Huseynli, Ilqar},
  year   = {2026},
  url    = {https://github.com/<user>/azerbaycan-tts},
  note   = {Adapted from owensong/Inflect-Micro-v2 (Apache-2.0);
            trained on ughurabbasov/azerbaijani-tts-dataset,
            used with the author's permission}
}
```

## Licence

Our own code is Apache-2.0 -- [LICENSE](LICENSE).

**Note on the runtime dependencies.** Phonemisation goes through
[`phonemizer`](https://github.com/bootphon/phonemizer) (**GPL-3.0-or-later**) and
[eSpeak NG](https://github.com/espeak-ng/espeak-ng) (**GPL-3.0-or-later**), whose
shared library `espeakng-loader` bundles into the wheel. Installing and running
this project is unaffected, but if you **redistribute** a combined work -- a
bundled application, a container image, a binary -- the GPL-3.0 terms apply to
those components. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the
full picture and for the ways to avoid it.

Third-party components (VITS, BigVGAN, alias-free-torch) and their licences are
listed in the same file.

**Training data.** See [Credits](#credits) above for the dataset and the terms
it is used under. The terms of the system that most likely produced the audio
remain unknown -- see [docs/MODEL.md](docs/MODEL.md).

Do not use this voice to impersonate a real person or to produce deceptive
content.
