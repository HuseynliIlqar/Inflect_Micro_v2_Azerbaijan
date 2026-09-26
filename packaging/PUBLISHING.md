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
python -m pytest         # 227 tests, three seconds, no model load
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

The weights live here, and this is where people searching for an Azerbaijani TTS
model will find the project.

```bash
pip install -U "huggingface_hub[cli]"
hf auth login
hf repo create <user>/azerbaijani-tts --repo-type model
```

```bash
git clone https://huggingface.co/<user>/azerbaijani-tts hf-model
cd hf-model
cp ../packaging/huggingface/MODEL_CARD.md README.md   # the YAML header matters
cp ../.gitattributes .
cp -r ../model .
cp -r ../samples .
cp ../LICENSE ../THIRD_PARTY_NOTICES.md .
git add . && git commit -m "Add Azerbaijani TTS 9.36M" && git push
```

Two things that are easy to get wrong:

1. **The YAML front matter is what makes the model findable.** Without
   `pipeline_tag: text-to-speech` and `language: az` the model does not appear
   in the filters that matter. It is already in `MODEL_CARD.md`.
2. **Fix the sample URLs.** `MODEL_CARD.md` embeds `<audio>` tags pointing at
   `resolve/main/samples/...`; replace `<user>` in them or the players stay
   silent.

Once the repository exists, add the model URL to the GitHub repository's
Website field and link back from `README.md`.

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
cp ../web/index.html ../web/styles.css ../web/app.js .
cp -r ../web/js .
mkdir -p onnx && cp ../web/onnx/*.onnx onnx/
# README.md needs `sdk: static` and `app_file: index.html` in its YAML header.
git add -A && git commit -m "Update the playground" && git push
```

The ONNX export is not in this repository (`web/onnx/` is ignored): the
repository already carries two checkpoints, and the Space is where the graphs
belong. Copy them in from the training workspace when working on the page.

#### Regenerating the golden files

`web/` is a port of the Python text layer, so its tests compare against output
generated from the Python side. After changing `az_text.py`, `az_chunk.py` or
the number words, regenerate:

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
PY
node web/tests/test_num_az.mjs
node web/tests/test_az_text.mjs
node web/tests/test_az_chunk.mjs
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
