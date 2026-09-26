---
language:
- az
license: apache-2.0
license_name: apache-2.0
pipeline_tag: text-to-speech
library_name: pytorch
base_model: owensong/Inflect-Micro-v2
datasets:
- ughurabbasov/azerbaijani-tts-dataset
tags:
- tts
- text-to-speech
- azerbaijani
- azerbaycan
- vits
- offline
- on-device
- low-resource
model-index:
- name: azerbaijani-tts
  results: []
---

# Azerbaijani TTS — 9.36M, 24 kHz, offline

A text-to-speech model that speaks Azerbaijani **entirely offline**: no server,
no API key, no network at inference. 9.36M parameters, 24 kHz mono, single
speaker, **2-4x faster than real time on a laptop CPU**.

A fine-tune of [`owensong/Inflect-Micro-v2`](https://huggingface.co/owensong/Inflect-Micro-v2)
by Owen Song — 200,000 steps on 25.07 hours of Azerbaijani speech, 409 of its
410 tensors carried over bit-identically.

| | |
| --- | --- |
| **Hear it** | [The playground](https://huggingface.co/spaces/ilqarrrr/Inflect_Micro_v2_Azerbaijan) — runs in your browser, nothing installed |
| **Code** | [github.com/HuseynliIlqar/Inflect_Micro_v2_Azerbaijan](https://github.com/HuseynliIlqar/Inflect_Micro_v2_Azerbaijan) |
| **In this repo** | PyTorch at the root, ONNX in `onnx/`, WAVs in `samples/` |

## Samples

| Text | Audio |
| --- | --- |
| `Salam, bu model tamamilə yerli maşında işləyir.` | <audio controls src="https://huggingface.co/ilqarrrr/Inflect_Micro_v2_Azerbaijan/resolve/main/samples/01-salam.wav"></audio> |
| `Payız gəlmişdi və şəhərin küçələri saralmış yarpaqlarla örtülmüşdü.` | <audio controls src="https://huggingface.co/ilqarrrr/Inflect_Micro_v2_Azerbaijan/resolve/main/samples/02-payiz.wav"></audio> |
| `Sən bu kitabı oxumusan? Mənə çox maraqlı gəldi.` | <audio controls src="https://huggingface.co/ilqarrrr/Inflect_Micro_v2_Azerbaijan/resolve/main/samples/03-sual.wav"></audio> |
| `II Dünya müharibəsi 01/09/1939 tarixində başladı və 25% artım oldu.` | <audio controls src="https://huggingface.co/ilqarrrr/Inflect_Micro_v2_Azerbaijan/resolve/main/samples/04-reqem.wav"></audio> |

The last one shows why the text layer matters: `II` becomes `İkinci`,
`01/09/1939` becomes `birinci sentyabr min doqquz yüz otuz doqquz`, `25%`
becomes `iyirmi beş faiz`.

## Usage

The weights are useless without the Azerbaijani text layer, which lives in the
GitHub repository together with the CLI, the tests and the training pipeline:

```bash
git lfs install
git clone https://github.com/HuseynliIlqar/Inflect_Micro_v2_Azerbaijan
cd Inflect_Micro_v2_Azerbaijan
pip install -r requirements.txt
python say.py "Salam, necəsiniz?"
```

That clone already contains these weights. To use the copy in this repository
instead:

```python
from huggingface_hub import snapshot_download
from aztts import AzTTS

tts = AzTTS(snapshot_download("ilqarrrr/Inflect_Micro_v2_Azerbaijan"))
tts.save("Salam, necəsiniz?", "out/salam.wav")
```

Normalisation and chunking run automatically. Calling the `model/` package
directly means calling `normalize_az` yourself.

### Without PyTorch

`onnx/` holds the same model as two graphs, verified against the PyTorch module
at a waveform correlation of 0.9999999999916:

| Graph | Inputs | Outputs |
| --- | --- | --- |
| `onnx/duration.onnx` | `tokens`, `lengths`, `length_scale` | `m_p_exp`, `logs_p_exp`, `y_mask` |
| `onnx/decode.onnx` | `m_p_exp`, `logs_p_exp`, `y_mask`, `zp_noise`, `noise_scale` | `waveform` |

This is what the browser playground runs, through ONNX Runtime Web. `web/` in
the GitHub repository is a working implementation, including the Azerbaijani
text layer ported to JavaScript.

## Model details

| | |
| --- | --- |
| Architecture | VITS (compact), 9,356,513 parameters |
| Audio | 24 kHz, mono, single speaker |
| Frontend | eSpeak NG (`az`) through `phonemizer`, with stress |
| Training | 200,000 steps = 1,343 epochs, ~3.7 days on one A40, ~$39 |
| Data | 25.07 hours, 9,674 clips |
| Speed | 2-4x real time on CPU; ~0.3 s to load |

The text layer ahead of the model does two things the checkpoint cannot:
rewrites digits, Roman numerals, dates, units and abbreviations into spoken
words, and cuts sentences into ~15-word chunks, because this model's intonation
flattens towards the end of a long sentence.

## Limitations

- **One voice.** No voice cloning, no multi-speaker support, no emotion control.
- At 9.36M parameters the speech is clear and intelligible but not fully
  natural; the vocoder sometimes leaves a metallic resonance.
- Rare names, foreign words and unusual spellings are at the mercy of the
  phoneme frontend.
- The training audio is most likely synthetic, so the quality ceiling is the
  system that produced it. The evidence is in `docs/MODEL.md` in the repository.

## Licence and commercial use

Apache-2.0 for this project's code and for these weights, inherited from the
base model. **Commercial use is allowed**, with one condition that comes from
the phonemiser rather than the model.

Phonemisation calls [`phonemizer`](https://github.com/bootphon/phonemizer) and
[eSpeak NG](https://github.com/espeak-ng/espeak-ng) **in-process**, and both are
**GPL-3.0-or-later**. Running them changes nothing. Shipping a combined work —
a desktop app, a container image, a binary handed to customers — brings GPL-3.0
obligations for those components.

The checkpoint itself does not need eSpeak: `config.json` declares
`accepts_prephonemized_input: true`, so phonemes can come from any frontend.
`THIRD_PARTY_NOTICES.md` in the GitHub repository sets out both routes.

**Training data.** [`ughurabbasov/azerbaijani-tts-dataset`](https://huggingface.co/datasets/ughurabbasov/azerbaijani-tts-dataset)
declares no licence file; asked in its Hugging Face community tab, the author
confirmed free use and asked to be credited. **Attribution is the condition of
use** — if you build on this model, credit the dataset too.

**Files.** The PyTorch export sits at the root (`model.pth`, `config.json`, the
frontend and the VITS runtime); `checksums.sha256` verifies 24 of them.
`NOTICES.md` is this project's full third-party notice, while `LICENSE` and
`THIRD_PARTY_NOTICES.md` are the upstream package's own, unchanged.

## Ethical use

Do not use this voice to impersonate a real person or to produce deceptive
content. Disclose synthetic speech where the context could otherwise mislead.

## Credits and citation

Built on two openly published works: **`owensong/Inflect-Micro-v2`** by Owen
Song (the checkpoint) and **`ughurabbasov/azerbaijani-tts-dataset`** by
`ughurabbasov` (the audio). Please cite all three:

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
