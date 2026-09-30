# Azerbaijani TTS

A **fully offline** text-to-speech model that speaks Azerbaijani. 9.36M
parameters, 24 kHz mono, **2-4x faster than real time** on an ordinary laptop
CPU. No server, no API key, no internet.

It is a fine-tune of
**[`owensong/Inflect-Micro-v2`](https://huggingface.co/owensong/Inflect-Micro-v2)**
by Owen Song (Apache-2.0) --- 200,000 steps on 25.07 hours of Azerbaijani
speech, 409 of its 410 tensors carried over bit-identically.

> Azərbaycanca oxumaq üçün: **[README.az.md](README.az.md)**

### Three ways to use it

| | |
| --- | --- |
| **Hear it now** | [The playground](https://huggingface.co/spaces/ilqarrrr/Inflect_Micro_v2_Azerbaijan) --- runs in your browser, nothing installed |
| **Run it locally** | `git clone` this repository, `pip install`, `python say.py "Salam."` |
| **Use the weights** | [`ilqarrrr/Inflect_Micro_v2_Azerbaijan`](https://huggingface.co/ilqarrrr/Inflect_Micro_v2_Azerbaijan) on the Hub --- PyTorch and ONNX |

---

## Contents

[Quick start](#quick-start) · [Hear it](#hear-it) ·
[Command line](#command-line) · [From Python](#from-python) ·
[The browser playground](#the-browser-playground) ·
[The local interface](#the-local-interface) · [The two voices](#the-two-voices) ·
[What the text layer does](#what-the-text-layer-does) ·
[Working on the code](#working-on-the-code) · [About the model](#about-the-model) ·
[Limitations](#limitations) ·
[Licence and commercial use](#licence-and-commercial-use) ·
[Credits](#credits)

---

## Quick start

Python 3.11 or newer. The weights are stored with
[Git LFS](https://git-lfs.com), so install that **before** cloning:

```bash
git lfs install
git clone https://github.com/HuseynliIlqar/Inflect_Micro_v2_Azerbaijan
cd Inflect_Micro_v2_Azerbaijan
```

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
./.venv/bin/pip install -r requirements.txt                   # Linux / macOS
```

```bash
python say.py "Salam, necəsiniz?"
```

The audio lands in `out/01.wav`. That is the whole thing.

Nothing is downloaded at install time: both checkpoints (37 MB each) are in the
repository. If you will only ever run on the CPU, take the smaller PyTorch
wheel:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Hear it

Ready-made WAVs, one set per voice --- listen before installing anything.

**Azerbaijani**, this project's model:

| File | Sentence |
| --- | --- |
| [`01-salam.wav`](samples/01-salam.wav) | `Salam, bu model tamamilə yerli maşında işləyir.` |
| [`02-payiz.wav`](samples/02-payiz.wav) | `Payız gəlmişdi və şəhərin küçələri saralmış yarpaqlarla örtülmüşdü.` |
| [`03-sual.wav`](samples/03-sual.wav) | `Sən bu kitabı oxumusan? Mənə çox maraqlı gəldi.` |
| [`04-reqem.wav`](samples/04-reqem.wav) | `II Dünya müharibəsi 01/09/1939 tarixində başladı və 25% artım oldu.` |
| [`05-uzun.wav`](samples/05-uzun.wav) | A long sentence, to hear where the intonation flattens |

**English**, the base model:

| File | Sentence |
| --- | --- |
| [`01-hello.wav`](samples/en/01-hello.wav) | `Hello, this model runs completely offline on your machine.` |
| [`02-autumn.wav`](samples/en/02-autumn.wav) | `Autumn had come, and the streets were covered with yellow leaves.` |
| [`03-question.wav`](samples/en/03-question.wav) | `Have you read this book? I found it very interesting.` |

Regenerate them only when a model actually changed --- both sets go through
Git LFS:

```bash
python say.py --out samples                 # Azerbaijani
python say.py --voice en --out samples/en   # English
```

## Command line

```bash
python say.py "One sentence."                     # -> out/01.wav
python say.py "First." "Second."                  # -> out/01.wav, out/02.wav
python say.py -f book.txt --one-file -o out/book  # whole file into one WAV
python say.py --show-text "II Dünya, 25% artım"   # show what reaches the model
python say.py --speed 0.85 --seed 42 "Daha yavaş."
python say.py --voice en "Hello there."           # the English base model
```

| Flag | Default | What it does |
| --- | ---: | --- |
| `--speed` | `1.0` | 0.5-2.0. Lower is slower |
| `--variation` | `0.667` | 0.0-1.0. Lower is steadier, higher is livelier |
| `--seed` | `7` | The same seed gives the same voice. Change it for a sentence you do not like |
| `--voice` | `az` | `en` speaks with the English base model |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--out` | `out/` | Output directory |
| `--raw` | -- | Turn normalisation off |
| `--max-words` | `15` | Chunk length. `0` leaves splitting to the model |
| `--prosody` | `off` | `safe` / `wide` -- thins out stress marks (experimental) |
| `--show-text` | -- | Print the normalised text and the chunk boundaries |
| `--allow-profanity` | -- | Speak obscenities as written instead of bleeping them |

## From Python

```python
from aztts import AzTTS

tts = AzTTS()                       # build once, reuse
tts.save("Mətn burada.", "out/a.wav")

waveform = tts.synthesize("Xam massiv lazımdırsa.", speed=1.1, seed=3)
# -> float32 numpy array, mono, [-1, 1], 24 000 Hz
```

The weights ship with this repository, so `AzTTS()` downloads nothing. To use
the copy on the Hub instead:

```python
from huggingface_hub import snapshot_download

tts = AzTTS(snapshot_download("ilqarrrr/Inflect_Micro_v2_Azerbaijan"))
```

The text helpers do not depend on the model and can be used on their own:

```python
from aztts import normalize_az, chunk_text

normalize_az("II qrupda 25% artım oldu.")
# 'İkinci qrupda iyirmi beş faiz artım oldu.'

chunk_text("Uzun bir cümlə...", max_words=15)
# ('Uzun bir cümlə...',)
```

## The browser playground

**<https://huggingface.co/spaces/ilqarrrr/Inflect_Micro_v2_Azerbaijan>**

The model also runs **entirely in a browser**, both voices, with nothing sent
anywhere. Once the page has loaded it works with the network switched off.

- ONNX Runtime Web on WebGPU --- roughly 8x real time on a laptop
- eSpeak NG compiled to WebAssembly for phonemisation
- the Azerbaijani text layer ported to JavaScript
- synthesis in a Web Worker, so the page never freezes, with a progress bar

**If the playground feels slow, it is the browser, not the model.** Phones
always run it on the CPU --- their browsers' WebGPU computed wrong durations and
was no faster --- and so does any browser without WebGPU, such as Firefox. The
CPU path uses an int8 decoder that is about 1.5x faster and half the download;
its change to the sound measured smaller than the model's own variation
between two takes, and the page says when it is in use (`?precision=fp32` turns
it off). On the
[direct link](https://ilqarrrr-inflect-micro-v2-azerbaijan.static.hf.space/index.html)
the page is cross-origin isolated and uses up to 4 threads, about twice as fast
as one. Inside the Hugging Face page it cannot be isolated and gets a single
thread, where a sentence can take up to a minute on a phone. The page says which
case applies and keeps showing progress. The text box takes up to 500
characters. The same model runs at 2--4x real time from the command line. The
first visit downloads 37 MB; after that the graphs
stay on the device.

The port is not trusted on faith: it is checked against the Python original
over golden files generated from it --- 24,022 number-word comparisons, 655
normalisation cases, 655 chunkings --- and the ONNX path is checked against
onnxruntime in Python (same sample count, same RMS to seven decimals).
[web/README.md](web/README.md) has the details.

## The local interface

The same thing without the Hub, and with every CLI flag exposed except
`--allow-profanity`:

```bash
pip install -r requirements-app.txt     # or: pip install -e ".[app]"
python app.py                           # http://127.0.0.1:7860
```

Gradio is deliberately **not** a core dependency --- someone importing `aztts`
as a library should not pay for a web stack.

The page shows two things the command line hides: the normalised text that
actually reaches the model, and the `say.py` command matching the settings you
picked, so it doubles as a way to learn the CLI. `Clean up the audio` filters
the vocoder's metallic resonance out of the finished clip, needs no ffmpeg, and
reports which frequencies it removed.

## The two voices

| | Azerbaijani | English |
| --- | --- | --- |
| Model | this project's | [`owensong/Inflect-Micro-v2`](https://huggingface.co/owensong/Inflect-Micro-v2), unchanged |
| Where | `model/` | `model-en/` |
| Command | `python say.py "Salam."` | `python say.py --voice en "Hello."` |
| Text layer | censoring, normalisation, chunking, stress | censoring, chunking |

Both ship with the repository, so a clone speaks both languages with nothing to
download. The Azerbaijani text layers do not apply to English: `normalize_az`
would rewrite numbers into Azerbaijani words, so it and the stress layer are
switched off there and greyed out in the interface. English text is phonemised
with eSpeak's `en-us` voice and handed to the runtime as phonemes.

**This project's own model does not speak English.** It was adapted over
200,000 steps on a single Azerbaijani voice; English text pushed through it is
that voice reading phonemes it never heard. `model-en/README.md` says where the
English weights came from and how to cite them.

## What the text layer does

The model was trained on ordinary prose, so digits and abbreviations come out
wrong if they reach it raw --- `II` is spoken as "ı ı", `25%` loses the percent
sign entirely. Two steps prevent that.

**1. Normalisation** (`aztts/az_text.py`) --- digits, Roman numerals, dates,
percentages, units and abbreviations become spoken words:

| Input | What reaches the model |
| --- | --- |
| `II Dünya müharibəsi` | `İkinci Dünya müharibəsi` |
| `01/09/1939` | `bir sentyabr min doqquz yüz otuz doqquzuncu il` |
| `25%` | `iyirmi beş faiz` |
| `5 kq` | `beş kiloqram` |
| `19,99 AZN` | `on doqquz manat doxsan doqquz qəpik` |
| `3,14` | `üç tam yüzdə on dörd` |

**2. Chunking** (`aztts/az_chunk.py`) --- sentences are cut into chunks of about
fifteen words, because towards the end of a long sentence this model's
intonation flattens and word endings get clipped. A pause proportional to the
punctuation is inserted between chunks, so the result does not sound choppy.

`--show-text` shows what is happening; `--raw` turns it off.

**Before both, censoring** (`aztts/az_profanity.py`) --- obscene words are
replaced by a 1 kHz bleep, the way television does it: `Sən qəhbəsən, bildin?`
is spoken as "Sən *(bip)*, bildin?". Inflected forms, spellings without
Azerbaijani letters (`qehbe`) and look-alike digits (`s1kdir`) are caught;
clean words that share the letters (`şikayət`, `sikkə`, `götürmək`) are not.
Mild insults such as `axmaq` or `eşşək` are read as written.

It is always on in the browser playground and in the hosted interface, with no
switch to turn it off. In a clone it is on by default too, and turned off only
on purpose: `python say.py --allow-profanity`, `python app.py
--allow-profanity`, or `AzTTS().synthesize(text, censor=False)`. `--raw` does
not turn it off, and it covers the English voice too, since Azerbaijani can be
typed into either. It is a filter for ordinary text, not a guarantee: a word
spelled out letter by letter (`s i k`) still gets through.

## Working on the code

```bash
python -m pytest        # 443 tests, three seconds, never loads the model
```

<details>
<summary>What the tests cover</summary>

| | |
| --- | ---: |
| Text normalisation, chunking, the stress layer | 192 |
| The profanity censor | 122 |
| Fast monotonic alignment (a training optimisation) | 25 |
| Interface labels, settings, the resonance filter | 85 |
| The English voice, the onnxruntime-web fetch, the int8 decoder tool | 19 |

The browser port has its own, run with Node:

```bash
for f in web/tests/*.mjs; do node "$f" || break; done   # all 16 suites
node web/tests/test_num_az.mjs      # 24,022 checks against num2words
node web/tests/test_az_text.mjs     # 655 cases against normalize_az
node web/tests/test_az_chunk.mjs    # the same 655 against chunk_text
node web/tests/test_progress.mjs    # the progress bar and the cached model download
```

</details>

<details>
<summary>Repository layout</summary>

```
say.py            Command line -- the main entry point
app.py            Browser interface (Gradio) -- optional
aztts/
  engine.py       AzTTS -- model loading, normalisation, chunking, synthesis
  az_text.py      normalize_az: Roman numerals, abbreviations, acronyms
  az_dates.py     Dates, times, phone numbers -> words
  az_amounts.py   Money, percentages, units, fractions, ranges -> words
  az_words.py     Casing, vowel harmony, numbers as words
  az_tables.py    The word tables (months, units, currencies, ...)
  az_chunk.py     Cuts sentences to a length the model handles
  az_prosody.py   Stress layer (optional, --prosody)
  az_profanity.py Obscenities -> a bleep (on unless --allow-profanity)
  en_voice.py     The English base model, through the same runtime
  console.py      Switches the Windows console to UTF-8
webui/
  i18n.py         Interface labels, Azerbaijani and English
  runner.py       Settings, the CLI equivalent, one synthesis
  cleanup.py      Single-clip resonance filter (no ffmpeg)
web/              The browser playground (a Hugging Face static Space)
model/            Azerbaijani weights + runtime (37 MB) -- do not edit
model-en/         English base-model weights (37 MB) -- do not edit
tools/            seed_sweep, audio_postprocess -- quality tools;
                  fetch_web_runtime, quantize_decoder -- the playground
training/         How the model was made (the base model is downloaded)
packaging/        Publishing to GitHub / Hugging Face / Kaggle
samples/          Example audio
tests/            443 tests
out/              The WAV files you generate
```

`model/` is the upstream export package and brings everything it needs with it
(weights, phoneme frontend, VITS runtime). All of our own code lives under
`aztts/`, `webui/` and `web/`.

</details>

<details>
<summary>Quality tools</summary>

```bash
# Render one line under 10 different seeds and pick the best
python tools/seed_sweep.py --seeds 10 "Xoş gəlmisiniz."

# Find the vocoder's metallic resonance, then clean it with ffmpeg
python tools/audio_postprocess.py analyze samples
python tools/audio_postprocess.py clean out --lowpass 11000
```

For lines an application repeats (a greeting, an error prompt) it is worth
picking a seed once and hard-coding it --- the reading improves noticeably.

</details>

Before changing anything, read [CLAUDE.md](CLAUDE.md): the setup, the commands
that verify a change, and the traps that are not obvious from the source.
Cursor picks the same rules up from `.cursor/rules/`. Publishing steps are in
[packaging/PUBLISHING.md](packaging/PUBLISHING.md).

## About the model

| | |
| --- | --- |
| Base | [`owensong/Inflect-Micro-v2`](https://huggingface.co/owensong/Inflect-Micro-v2) |
| Parameters | 9.36M |
| Voice | 24 kHz mono, single speaker |
| Training | 200,000 steps = 1,343 epochs, 25.07 hours of Azerbaijani audio |
| Speed | 2-4x real time on CPU |

[docs/MODEL.md](docs/MODEL.md) has the measurements, the dataset evidence and
the limitations in full. [`training/`](training/README.md) holds the process:
the patch applied to the official toolkit, the presets, the results and the
loss history.

One part of that is worth repeating here. The toolkit's `maximum_path` was a
pure Python loop eating 8.7 seconds of every step; a vectorised torch version
does it in **1.395 seconds**. Twenty days became 3.2 days, ~$213 became ~$34,
and 25 tests prove the result **bit-identical** to the original.

## Limitations

- **One voice.** No voice cloning, no multi-speaker support.
- **No emotion control.** The reading is calm and neutral.
- At 9.36M parameters the speech is clear and intelligible but not fully
  natural, and the vocoder sometimes leaves a metallic resonance.
- Rare names, foreign words and unusual spellings are at the mercy of the
  phoneme frontend.
- Normalisation runs automatically inside `say.py` and `AzTTS`. Calling the
  `model/` package directly means calling `normalize_az` yourself.

## Licence and commercial use

Our code and both checkpoints are **Apache-2.0** ([LICENSE](LICENSE)).
Commercial use is allowed. There is exactly one condition, and it comes from
the phonemiser rather than the model.

| Part | Licence | Commercial use |
| --- | --- | --- |
| Our code (`aztts/`, `say.py`, `app.py`, `webui/`, `web/`) | Apache-2.0 | Yes |
| Azerbaijani weights (`model/`) | Apache-2.0 | Yes |
| English weights (`model-en/`) | Apache-2.0 | Yes |
| Runtime pieces (VITS, BigVGAN, alias-free-torch) | MIT / Apache-2.0 | Yes |
| **`phonemizer` + eSpeak NG** | **GPL-3.0-or-later** | **Running: yes. Redistributing: see below** |

### The one thing to check before shipping

Phonemisation calls [`phonemizer`](https://github.com/bootphon/phonemizer) and
[eSpeak NG](https://github.com/espeak-ng/espeak-ng) **in-process**, and both are
GPL-3.0-or-later. Using them inside your company changes nothing. Shipping a
combined work --- a desktop app, a container image, a binary you hand to
customers --- brings GPL-3.0 obligations for those components, including
offering corresponding source.

The model itself does not need eSpeak: `model/config.json` declares
`accepts_prephonemized_input: true`, so a product that must stay away from the
GPL can phonemise with its own tool and hand the phonemes in.
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) sets out both routes and lists
every third-party component.

### Where the voices came from

**Azerbaijani.** Trained on
[`ughurabbasov/azerbaijani-tts-dataset`](https://huggingface.co/datasets/ughurabbasov/azerbaijani-tts-dataset).
The repository declares no licence file; asked directly in the Hugging Face
community tab, the author confirmed free use and asked to be credited.
**Attribution is the condition of use.** The audio is most likely synthetic,
produced by another TTS system whose terms are not known ---
[docs/MODEL.md](docs/MODEL.md).

**English.** `owensong/Inflect-Micro-v2`, redistributed unmodified. Three facts
from its model card matter for a commercial decision:

- the voice is **synthetic**. The package does not redistribute a real-speaker
  corpus and does not claim the voice as any real person's identity, so there is
  no personality or likeness right to clear;
- the release is **open-weight, not open-data**: the corpus pipeline and the
  filtering infrastructure are private, so the training data cannot be audited
  from public material. If your compliance process requires that audit, the
  author invites deployment inquiries by email;
- its responsible-use statement asks that synthetic speech be disclosed where
  the context could otherwise mislead, and that the voice not be used to
  impersonate anyone.

### Not permitted, in either voice

Impersonating a real person, producing deceptive content, or presenting
synthetic speech as a genuine recording. That is this project's position as
well as the upstream one.

This is a reading of the licences, not legal advice. If a product depends on it,
have someone qualified confirm it.

## Credits

This model exists because two people published their work openly.

**[`owensong/Inflect-Micro-v2`](https://huggingface.co/owensong/Inflect-Micro-v2)**
by Owen Song, Apache-2.0. Every weight here started as one of his: 409 of the
410 tensors were carried over bit-identically, and the Azerbaijani phonemes were
already in his symbol set. Without that checkpoint this would have been months
of training instead of days.

**[`ughurabbasov/azerbaijani-tts-dataset`](https://huggingface.co/datasets/ughurabbasov/azerbaijani-tts-dataset)**
by `ughurabbasov`. 25.07 hours of single-speaker Azerbaijani audio, used with
the author's permission on condition of attribution. **If you build on this
model, credit the dataset too.**

Please cite the project and both sources --- see [CITATION.cff](CITATION.cff):

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
