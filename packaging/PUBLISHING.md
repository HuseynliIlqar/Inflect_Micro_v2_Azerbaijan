# Publishing

Everything needed to put this project on GitHub, Hugging Face and Kaggle.
Nothing here runs automatically -- these are the steps and the files they use.

Replace `<user>` with your account name everywhere before publishing.

---

## Before the first push, anywhere

```bash
git lfs install          # required: model.pth and the samples go through LFS
python -m pytest         # 141 tests, three seconds, no model load
```

Check that nothing large slipped in:

```bash
git ls-files -s | wc -l
du -sh .git 2>/dev/null
```

`training/base-model/` is deliberately not tracked -- it is a 38 MB upstream
snapshot. Fetch it locally with `python training/scripts/download_model.py`.

---

## GitHub

The code lives here. This is the canonical repository; the other two point back
to it.

```bash
git init
git add .
git commit -m "feat: offline Azerbaijani TTS"
git branch -M main
git remote add origin https://github.com/<user>/azerbaycan-tts
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

### Worth doing next: a Gradio Space

The model runs at 2-4x real time on CPU, which means it works on a free CPU
Space. A demo people can click is worth more for adoption than any amount of
documentation. Not built here -- it is a separate `app.py` plus a Space README
with `sdk: gradio` in its YAML header.

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
