# Publishing

Everything needed to put this project on GitHub, Hugging Face and Kaggle.
Nothing here runs automatically -- these are the steps and the files they use.

The GitHub repository already exists:
<https://github.com/HuseynliIlqar/Inflect_Micro_v2_Azerbaijan>. The Hugging Face
and Kaggle accounts do not, so `<user>` still stands for them -- replace it
before publishing there.

---

## Before the first push, anywhere

```bash
git lfs install          # required: both model.pth files and the samples
python -m pytest         # 303 tests, three seconds, no model load
```

The repository carries two checkpoints -- `model/` (Azerbaijani, 37 MB) and
`model-en/` (the English base model, 37 MB) -- so a clone pulls about 75 MB of
LFS objects. GitHub's free tier gives 1 GB of LFS storage and 1 GB of bandwidth
a month, which is roughly thirteen clones. If the project gets popular enough
for that to bite, the English weights are the part to drop: they are a copy of
`owensong/Inflect-Micro-v2`, still downloadable with
`python training/scripts/download_model.py`.

Check that nothing large slipped in:

```bash
git ls-files -s | wc -l
du -sh .git 2>/dev/null
```

`training/base-model/` is deliberately not tracked -- it is the complete 38 MB
upstream package, needed only for training work. Fetch it locally with
`python training/scripts/download_model.py`.

---

## GitHub

The code lives here. This is the canonical repository; the other two point back
to it.

```bash
git init
git add .
git commit -m "feat: offline Azerbaijani TTS"
git branch -M main
git remote add origin https://github.com/HuseynliIlqar/Inflect_Micro_v2_Azerbaijan
git push -u origin main
```

Then, in the repository settings:

- **Topics:** `text-to-speech`, `tts`, `azerbaijani`, `azerbaycan`, `vits`,
  `speech-synthesis`, `offline`, `on-device`, `low-resource-languages`
- **Description:** *Offline Azerbaijani text-to-speech. 9.36M parameters,
  24 kHz, 2-4x real time on CPU.*
- **Website:** the Hugging Face model URL, once it exists.

Files used: `.gitignore`, `.gitattributes` (LFS + `linguist-vendored`),
`LICENSE`, `THIRD_PARTY_NOTICES.md`.

---

## Hugging Face

Two repositories, both live:

| | |
| --- | --- |
| Model | <https://huggingface.co/ilqarrrr/Inflect_Micro_v2_Azerbaijan> |
| Space | <https://huggingface.co/spaces/ilqarrrr/Inflect_Micro_v2_Azerbaijan> |

### The model repository

It carries the PyTorch export **at the root** -- `model.pth`, `config.json`,
the frontend and the VITS runtime, exactly the contents of `model/` -- so a
snapshot can be handed straight to the engine:

```python
from huggingface_hub import snapshot_download
from aztts import AzTTS

tts = AzTTS(snapshot_download("ilqarrrr/Inflect_Micro_v2_Azerbaijan"))
```

Plus `onnx/` (the two graphs the playground runs), `samples/`, `CITATION.cff`
and `NOTICES.md`. The English base model is **not** copied there: it is one
click away at `owensong/Inflect-Micro-v2`, and the card links it.

To update it:

```bash
git clone https://huggingface.co/ilqarrrr/Inflect_Micro_v2_Azerbaijan hf-model
cd hf-model
cp ../model/* .                       # the export goes at the root
cp ../packaging/huggingface/MODEL_CARD.md README.md
git add -A && git commit -m "Update the weights" && git push
```

Two things that cost a push each when they were missed:

1. **The YAML front matter is what makes the model findable.** Without
   `pipeline_tag: text-to-speech` and `language: az` the model does not appear
   in the filters that matter. It is already in `MODEL_CARD.md`.
2. **Every binary needs LFS, including the samples.** The Hub rejects a push
   that contains plain binary files, naming them. `*.wav` was missing from the
   repository's `.gitattributes`; `*.pth` and `*.onnx` were already there.

### The static Space (live)

The playground at
<https://huggingface.co/spaces/ilqarrrr/Inflect_Micro_v2_Azerbaijan> is a
**static** Space, named after the checkpoint it descends from, the same as the
GitHub repository:
`web/` plus the 38 MB ONNX export, no server. Static Spaces are free for
everyone; Gradio and Docker Spaces need a Pro subscription, which is why this
one is static.

```bash
git clone https://huggingface.co/spaces/ilqarrrr/Inflect_Micro_v2_Azerbaijan hf-space
cd hf-space
cp ../web/index.html ../web/*.css ../web/app.js ../web/worker.js .
cp -r ../web/js .
mkdir -p onnx && cp ../web/onnx/*.onnx onnx/
# onnxruntime-web from the page's own origin -- WebKit (every iPhone browser)
# refuses a CDN's modules in a worker. Without it the Space ships a page whose
# worker cannot start, so stop here if the fetch fails.
(cd .. && python tools/fetch_web_runtime.py) || exit 1
mkdir -p vendor && cp -r ../web/vendor/onnxruntime-web vendor/
# The 27 MB .wasm must go through LFS (the Space's *.wasm rule); check it is listed:
git lfs ls-files | grep -q "ort-wasm-simd-threaded.asyncify.wasm" || git add --renormalize vendor/
# README.md needs `sdk: static`, `app_file: index.html` and the `custom_headers`
# block from web/README.md in its YAML header (the headers give wasm threads).
git add -A && git commit -m "Update the playground" && git push
```

The ONNX export is not in this repository (`web/onnx/` is ignored): the
repository already carries two checkpoints, and the Space is where the graphs
belong. Copy them in from the training workspace when working on the page.

`onnx/decode.int8.onnx` is the CPU step of the fallback chain. Rebuild it
whenever `decode.onnx` changes, then bump `CACHE_NAME` in `web/js/fetch-model.js`:

```bash
pip install onnx                       # the tool's only extra need
python tools/quantize_decoder.py       # web/onnx/decode.onnx -> web/onnx/decode.int8.onnx
```

It caps its own memory (3 GiB by default, `--memory-gib`) and calibrates one
sample at a time. Without that, onnxruntime's calibrator holds ~24 GiB of
activations and froze a 16 GB machine.

#### Regenerating the golden files

`web/` is a port of the Python text layer, so its tests compare against output
generated from the Python side. After changing `az_text.py`, `az_chunk.py`,
`az_profanity.py` or the number words, regenerate:

```bash
python - <<'PY'
import json, random, re, sys
sys.path.insert(0, '.')
from aztts import normalize_az, chunk_text
from num2words import num2words as n
# numbers.json: every integer to 10,000 plus a sample above it
vals = list(range(10001)); random.seed(7)
vals += [random.randint(10001, 10**12) for _ in range(2000)]
json.dump({str(v): [n(v, lang='az'), n(v, lang='az', to='ordinal')] for v in vals},
          open('web/tests/golden/numbers.json', 'w', encoding='utf-8'), ensure_ascii=False)
# normalise.json / chunks.json: keep the existing inputs, refresh the outputs
old = json.load(open('web/tests/golden/normalise.json', encoding='utf-8'))
json.dump({t: normalize_az(t) for t in old},
          open('web/tests/golden/normalise.json', 'w', encoding='utf-8'), ensure_ascii=False)
json.dump({t: list(chunk_text(normalize_az(t))) for t in old if normalize_az(t)},
          open('web/tests/golden/chunks.json', 'w', encoding='utf-8'), ensure_ascii=False)
# censor.json: keep the existing inputs, refresh the bleeped text and segments
from aztts.az_profanity import censor_az, bleep_segments
old = json.load(open('web/tests/golden/censor.json', encoding='utf-8'))
json.dump({t: [censor_az(t), list(bleep_segments(censor_az(t)))] for t in old},
          open('web/tests/golden/censor.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
PY
node web/tests/test_num_az.mjs
node web/tests/test_az_text.mjs
node web/tests/test_az_chunk.mjs
node web/tests/test_progress.mjs
node web/tests/test_theme.mjs
node web/tests/test_backend_plan.mjs
node web/tests/test_engine_fallback.mjs
node web/tests/test_mode_banner.mjs
node web/tests/test_device_check.mjs
node web/tests/test_az_censor.mjs
node web/tests/test_cancel.mjs
node web/tests/test_waveform.mjs
node web/tests/test_reveal.mjs
node web/tests/test_phone_copy.mjs
```

### The Gradio Space (needs Pro)

`app.py` in the repository root is the demo, and it runs on a free CPU Space --
the model is 2-4x faster than real time there. A page people can click is worth
more for adoption than any amount of documentation.

A Space is its own git repository and needs three things from this one:
`app.py`, `webui/`, `aztts/`, `say.py` (the examples come from it) and `model/`,
plus a `requirements.txt` of its own and a README with the Space's YAML header.

```bash
hf repo create <user>/azerbaijani-tts-demo --repo-type space --space_sdk gradio
git clone https://huggingface.co/spaces/<user>/azerbaijani-tts-demo hf-space
cd hf-space
cp ../.gitattributes .
cp ../app.py ../say.py .
cp -r ../webui ../aztts ../model ../model-en .
cp ../packaging/huggingface/space-requirements.txt requirements.txt
```

The Space's `README.md` is `packaging/huggingface/SPACE_README.md`, copied over:

```bash
cp ../packaging/huggingface/SPACE_README.md README.md
```

Its YAML header is what makes the Space work -- `sdk: gradio`, `sdk_version`
and `app_file: app.py`. Keep `sdk_version` in step with the Gradio the
interface was tested against.

One thing to check before pushing: **both models go through LFS.**
`.gitattributes` is copied for that reason; without it each `model.pth` is
pushed as a 37 MB blob and the Space is rejected. Leave `model-en/` out if you
would rather the Space offered the Azerbaijani voice alone -- the interface
handles a missing English checkpoint and says so in the voice selector.

`space-requirements.txt` exists because a Space reads only a file named
`requirements.txt`, so the repository's `requirements-app.txt` -- whose first
line is `-r requirements.txt` -- would refer to itself. Keep the two in step
when a dependency changes.

### Also worth doing: safetensors

`model.pth` is a pickle, and Hugging Face flags pickles with a security warning
on the model page. Publishing a `.safetensors` copy alongside it removes the
warning. The inference runtime would need a small change to prefer it.

---

## Kaggle

Two artefacts, published separately.

### The model

```bash
pip install kaggle
mkdir -p kaggle-upload && cp -r model samples aztts say.py kaggle-upload/
cp packaging/kaggle/model-metadata.json kaggle-upload/
# edit ownerSlug first
kaggle models create -p kaggle-upload
```

Then the instance, using `model-instance-metadata.json` in the same directory:

```bash
kaggle models instances create -p kaggle-upload
```

### The demo notebook

Not written here. When you write it, keep three things in mind:

- **Write it in English.** An Azerbaijani-language notebook will not find an
  audience on Kaggle.
- **Lead with the offline claim.** Kaggle notebooks often run with networking
  disabled, and a TTS model that works in that state is unusual. That is the
  strongest thing this project has to say on that platform.
- **Verify `espeakng-loader` in the Kaggle image first.** If the bundled eSpeak
  NG library does not load there, the notebook needs
  `apt-get install espeak-ng`, which needs networking -- and that contradicts
  the offline pitch. Test it before writing the notebook around the claim.

Its `kernel-metadata.json` goes in the notebook's own directory and must list
the model as a `modelDataSources` entry.

---

## What to say in the announcement

The model itself is a small one and the README is honest about that. The two
things genuinely worth leading with are:

1. **`fast_monotonic_align`** -- the toolkit's `maximum_path` was a pure Python
   loop eating 8.7 s per step; the vectorised torch version does it in 1.395 s,
   bit-identically, proven by 25 tests. 20 days became 3.2 days, ~$213 became
   ~$34. That is an independent contribution and would be welcome upstream as a
   pull request.
2. **`aztts/az_text.py` and `az_chunk.py`** -- the Azerbaijani text
   normalisation does not depend on the model and is reusable by any Azerbaijani
   TTS or NLP project. Splitting it out as its own PyPI package would probably
   outlive the model.
