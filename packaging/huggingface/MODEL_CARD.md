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

# Azerbaijani TTS (9.36M, 24 kHz, offline)

A fully offline text-to-speech model that speaks Azerbaijani. 9.36M parameters,
24 kHz mono, **2-4x faster than real time on a laptop CPU**. No server, no API
key, no internet.

Code, tests and the full training pipeline:
**https://github.com/<user>/azerbaycan-tts**

## Samples

| Text | Audio |
| --- | --- |
| `Salam, bu model tamamilə yerli maşında işləyir.` | <audio controls src="https://huggingface.co/<user>/azerbaijani-tts/resolve/main/samples/01-salam.wav"></audio> |
| `Payız gəlmişdi və şəhərin küçələri saralmış yarpaqlarla örtülmüşdü.` | <audio controls src="https://huggingface.co/<user>/azerbaijani-tts/resolve/main/samples/02-payiz.wav"></audio> |
| `Sən bu kitabı oxumusan? Mənə çox maraqlı gəldi.` | <audio controls src="https://huggingface.co/<user>/azerbaijani-tts/resolve/main/samples/03-sual.wav"></audio> |
| `II Dünya müharibəsi 01/09/1939 tarixində başladı və 25% artım oldu.` | <audio controls src="https://huggingface.co/<user>/azerbaijani-tts/resolve/main/samples/04-reqem.wav"></audio> |

## Usage

```bash
git clone https://github.com/<user>/azerbaycan-tts
cd azerbaycan-tts
pip install -r requirements.txt
python say.py "Salam, necəsiniz?"
```

```python
from aztts import AzTTS

tts = AzTTS()
tts.save("Salam, necəsiniz?", "out/salam.wav")
```

Text normalisation runs automatically: `II` becomes `İkinci`, `25%` becomes
`iyirmi beş faiz`, `01/09/1939` becomes `birinci sentyabr min doqquz yüz otuz
doqquz`. Long sentences are cut into roughly fifteen-word chunks, because the
model's intonation flattens beyond that.

## Model details

| | |
| --- | --- |
| Architecture | VITS (compact), alias-free vocoder |
| Parameters | 9,356,513 |
| Sample rate | 24 000 Hz, mono |
| Voice | Single speaker (F0 195-201 Hz) |
| Phoneme frontend | eSpeak NG, `az`, with stress marks |
| Base model | [`owensong/Inflect-Micro-v2`](https://huggingface.co/owensong/Inflect-Micro-v2) |
| Training | 200,000 steps = 1,343 epochs, 25.07 hours of audio |
| Hardware | NVIDIA A40, ~3.7 days, roughly $39 |

The warm start was complete: 409 of 410 tensors were copied bit-identically from
the base model, and every Azerbaijani phoneme was already in the base symbol set.

## Speed

| Context | Result |
| --- | --- |
| Short sentence | 2-4x real time (CPU) |
| Two minutes of text | 122 s of audio / 43 s of compute |
| Model load | 0.3-2 s |
| 1 thread (phone-like) | 2.5x real time |

## Limitations

- **One voice, no emotion control.** The reading is neutral and cannot be
  changed. There is no voice cloning and no multi-speaker support.
- **The model is small.** 9.36M parameters is enough for clear speech, not for
  full naturalness. The vocoder sometimes leaves a metallic resonance; an 11 kHz
  low-pass reduces it.
- **Rare names, foreign words and unusual spellings** are at the mercy of the
  phoneme frontend.
- **The training audio is most likely synthetic** -- constant-length silences, a
  noise floor of exactly zero, vocoder traces in the spectrum. The quality
  ceiling is therefore the system that produced it.

## Credits

This model exists because two people published their work openly. If you build
on it, credit both of them as well.

**Base model --- [`owensong/Inflect-Micro-v2`](https://huggingface.co/owensong/Inflect-Micro-v2)**
by Owen Song, Apache-2.0. Every weight here started as one of his: 409 of the
410 tensors were carried over bit-identically, and every Azerbaijani phoneme was
already in his symbol set.

**Training data --- [`ughurabbasov/azerbaijani-tts-dataset`](https://huggingface.co/datasets/ughurabbasov/azerbaijani-tts-dataset)**
by `ughurabbasov`. 9,674 clips used out of 10,088, 25.07 hours, single speaker.
The repository declares no licence file; asked directly in its Hugging Face
community tab, the author confirmed that anyone may use the dataset and asked to
be credited. **Attribution is the condition of use.** The terms of the upstream
TTS system that most likely produced the audio are a separate question and
remain unknown.

## Licence

The weights and this project's own code are **Apache-2.0**, inherited from the
base model.

**Runtime note.** Phonemisation goes through
[`phonemizer`](https://github.com/bootphon/phonemizer) and
[eSpeak NG](https://github.com/espeak-ng/espeak-ng), both **GPL-3.0-or-later**.
Installing and running is unaffected, but redistributing a combined work -- a
bundled application, a container image, a binary -- brings the GPL-3.0 terms with
it for those components. The checkpoint itself does not depend on eSpeak: the
config declares `accepts_prephonemized_input: true`, so phonemes can come from
any frontend. See `THIRD_PARTY_NOTICES.md` in the GitHub repository.

## Ethical use

Do not use this voice to impersonate a real person or to produce deceptive
content.

## Citation

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
