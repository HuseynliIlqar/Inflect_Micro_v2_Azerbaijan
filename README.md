# Azerbaijani TTS

A **fully offline** text-to-speech model that speaks Azerbaijani. 9.36M
parameters, 24 kHz mono, **2-4x faster than real time** on an ordinary laptop
CPU. No server, no API key, no internet.

It is a fine-tune of
**[`owensong/Inflect-Micro-v2`](https://huggingface.co/owensong/Inflect-Micro-v2)**
by Owen Song (Apache-2.0) --- 200,000 steps on 25.07 hours of Azerbaijani
speech, 409 of its 410 tensors carried over bit-identically. That checkpoint
also ships here as a second voice, unchanged; see [Credits](#credits).

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
git clone https://github.com/HuseynliIlqar/Inflect_Micro_v2_Azerbaijan
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
| `--voice` | `az` | `en` speaks with the English base model (see below) |

### Browser interface

A page for trying the model out without the command line: type a sentence,
adjust the parameters, listen, download. It needs Gradio, which the core
library deliberately does not depend on -- install it only if you want the
page.

```bash
pip install -r requirements-app.txt     # or: pip install -e ".[app]"
python app.py                           # http://127.0.0.1:7860
```

The interface switches between Azerbaijani and English, exposes every flag the
CLI has, and shows two things the command line hides: the normalised text that
actually reaches the model, and the `say.py` command matching the settings you
picked -- so the page doubles as a way to learn the CLI.

`Clean up the audio` filters the vocoder's metallic resonance out of the finished
clip. It is the single-clip version of `tools/audio_postprocess.py`, needs no
ffmpeg, and reports which frequencies it removed.

### The English voice

The page and the CLI can also speak with `owensong/Inflect-Micro-v2`, the
English checkpoint this model was adapted from. It ships in `model-en/`, so a
clone speaks both languages with nothing to download:

```bash
python say.py --voice en "Hello there."
```

It is a different voice by another author, Apache-2.0, and `model-en/README.md`
says how to cite it. The weights go through Git LFS like the Azerbaijani ones --
a clone made without `git lfs install` gets a pointer file, and both the CLI and
the page say so plainly instead of failing inside torch.

The Azerbaijani text layers do not apply to it -- `normalize_az` rewrites
numbers into Azerbaijani words and the stress layer is tuned to Azerbaijani, so
both are switched off for English and greyed out in the interface. The English
text is phonemised with eSpeak's `en-us` voice and handed to the runtime as
phonemes.

This project's own model does not speak English. It was adapted over 200,000
steps on a single Azerbaijani voice; what English text produces through it is
that voice reading unfamiliar phonemes.

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

Ready-made WAVs, one set per voice. Listen to them before installing anything.

**Azerbaijani** -- this project's model, `samples/`:

| File | Sentence |
| --- | --- |
| [`01-salam.wav`](samples/01-salam.wav) | `Salam, bu model tamamilə yerli maşında işləyir.` |
| [`02-payiz.wav`](samples/02-payiz.wav) | `Payız gəlmişdi və şəhərin küçələri saralmış yarpaqlarla örtülmüşdü.` |
| [`03-sual.wav`](samples/03-sual.wav) | `Sən bu kitabı oxumusan? Mənə çox maraqlı gəldi.` |
| [`04-reqem.wav`](samples/04-reqem.wav) | `II Dünya müharibəsi 01/09/1939 tarixində başladı və 25% artım oldu.` |
| [`05-uzun.wav`](samples/05-uzun.wav) | A long sentence, to hear where the intonation flattens |

**English** -- the base model, `samples/en/`:

| File | Sentence |
| --- | --- |
| [`01-hello.wav`](samples/en/01-hello.wav) | `Hello, this model runs completely offline on your machine.` |
| [`02-autumn.wav`](samples/en/02-autumn.wav) | `Autumn had come, and the streets were covered with yellow leaves.` |
| [`03-question.wav`](samples/en/03-question.wav) | `Have you read this book? I found it very interesting.` |

The two are different voices by different authors. The Azerbaijani one is this
project's; the English one is `owensong/Inflect-Micro-v2`, unchanged.

To regenerate them:

```bash
python say.py --out samples                 # Azerbaijani
python say.py --voice en --out samples/en   # English
```

Both sets go through Git LFS, so only regenerate them when a model actually
changed.

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

227 tests:

| | |
| --- | ---: |
| Text normalisation, chunking, the stress layer | 116 |
| Fast monotonic alignment (a training optimisation) | 25 |
| Interface labels, settings, the resonance filter | 86 |

The tests do not load the model -- they finish in three seconds.

## Layout

```
say.py            Command line -- the main entry point
app.py            Browser interface (Gradio) -- optional
aztts/
  engine.py       AzTTS -- model loading, normalisation, chunking, synthesis
  az_text.py      Digits / Roman numerals / abbreviations -> words
  az_chunk.py     Cuts sentences to a length the model handles
  az_prosody.py   Stress layer (optional, --prosody)
  en_voice.py     The English base model, through the same runtime
  console.py      Switches the Windows console to UTF-8
webui/
  i18n.py         Interface labels, Azerbaijani and English
  runner.py       Settings, the CLI equivalent, one synthesis
  cleanup.py      Single-clip resonance filter (no ffmpeg)
model/            Azerbaijani weights + runtime (37 MB) -- do not edit
model-en/         English base-model weights (37 MB) -- do not edit
tools/            seed_sweep, audio_postprocess -- quality tools
training/         How the model was made (the base model is downloaded)
packaging/        Publishing to GitHub / Hugging Face / Kaggle
samples/          Example audio
tests/            227 tests
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
  url    = {https://github.com/HuseynliIlqar/Inflect_Micro_v2_Azerbaijan},
  note   = {Adapted from owensong/Inflect-Micro-v2 (Apache-2.0);
            trained on ughurabbasov/azerbaijani-tts-dataset,
            used with the author's permission}
}
```

## Commercial use

Short answer: **yes for running it, with one real condition if you redistribute
it.** The condition is not the model -- it is the phonemiser.

| Part | Licence | Commercial use |
| --- | --- | --- |
| Our code (`aztts/`, `say.py`, `app.py`, `webui/`) | Apache-2.0 | Yes |
| Azerbaijani weights (`model/`) | Apache-2.0 | Yes |
| English weights (`model-en/`) | Apache-2.0 | Yes |
| Runtime pieces (VITS, BigVGAN, alias-free-torch) | MIT / Apache-2.0 | Yes |
| **`phonemizer` + eSpeak NG** | **GPL-3.0-or-later** | **Running: yes. Redistributing: see below** |

### The one thing to check before shipping

Phonemisation calls [`phonemizer`](https://github.com/bootphon/phonemizer) and
[eSpeak NG](https://github.com/espeak-ng/espeak-ng) **in-process**, and both are
GPL-3.0-or-later. Using them inside your company changes nothing. Shipping a
combined work -- a desktop app, a container image, a binary you hand to
customers -- brings GPL-3.0 obligations for those components, including offering
corresponding source.

The model itself does not need eSpeak. `model/config.json` declares
`accepts_prephonemized_input: true`, so a product that must stay away from the
GPL can phonemise with its own tool and hand the phonemes in.
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) sets out both routes.

### The two voices

**Azerbaijani.** Trained on
[`ughurabbasov/azerbaijani-tts-dataset`](https://huggingface.co/datasets/ughurabbasov/azerbaijani-tts-dataset).
The repository declares no licence file; asked directly in the HuggingFace
community tab, the author confirmed free use and asked to be credited.
Attribution is the condition. The audio is most likely synthetic, produced by
another TTS system whose terms are not known -- see
[docs/MODEL.md](docs/MODEL.md).

**English.** `owensong/Inflect-Micro-v2`, Apache-2.0, redistributed here
unmodified. Three facts from its model card matter for a commercial decision:

- the voice is **synthetic**. The package does not redistribute a real-speaker
  corpus and does not claim the voice as any real person's identity, so there is
  no personality or likeness right to clear;
- the release is **open-weight, not open-data**: the corpus-generation pipeline
  and the filtering infrastructure are private, so the training data cannot be
  audited from public material. If your compliance process requires that audit,
  the author invites deployment inquiries by email;
- its responsible-use statement asks that synthetic speech be disclosed where
  the context could otherwise mislead, and that the voice not be used to
  impersonate anyone.

### What is not permitted, in either voice

Impersonating a real person, producing deceptive content, or presenting
synthetic speech as a genuine recording. That is this project's position as well
as the upstream one.

This is a reading of the licences, not legal advice. If a product depends on it,
have someone qualified confirm it.

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
