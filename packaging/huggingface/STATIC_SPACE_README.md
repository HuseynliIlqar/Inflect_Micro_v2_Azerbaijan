---
title: Avaz — On-device Text-to-Speech
emoji: 🗣️
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
custom_headers:
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
short_description: On-device text-to-speech, in the browser and on phones
models:
  - owensong/Inflect-Micro-v2
datasets:
  - ughurabbasov/azerbaijani-tts-dataset
tags:
  - text-to-speech
  - tts
  - on-device
  - mobile
  - offline
  - azerbaijani
  - english
  - vits
  - onnx
  - webgpu
---

# Avaz — text-to-speech that runs on your device

**Avaz** is open-source text-to-speech that runs on the device itself -- in the
browser, on a CPU, with no server. We are building it for natural intonation
on phones. Today it speaks Azerbaijani and English; Turkish and Russian are
next (planned, no dates yet). This page is its playground.

The Azerbaijani voice is the Inflect-Micro-v2 Azerbaijani model, a
9.36M-parameter VITS network and a fine-tune of
[`owensong/Inflect-Micro-v2`](https://huggingface.co/owensong/Inflect-Micro-v2)
by Owen Song: 200,000 steps on 25.07 hours of Azerbaijani speech, 409 of its
410 tensors carried over bit-identically. The English voice is that base model,
unmodified. Nothing you type is sent anywhere: the model and the phonemiser are
downloaded once, cached by the browser, and run there. Once the page has
loaded, it works with the network switched off, with no sign-up.

**We are not there yet.** The current model is small and fast, but its
intonation is still flat, especially over long sentences, and phone browsers
run it slowly. The weak spots are listed under *Speed* and *Limitations* below,
and we fix them release by release.

- **Two ONNX graphs per voice** (38 MB) through ONNX Runtime Web, on WebGPU where the
  browser has it -- about 3-5x faster than real time, the same as the Python
  command line. Browsers without WebGPU fall back to WebAssembly and are
  slower -- see *Speed* below.
- **eSpeak NG compiled to WebAssembly** for phonemisation, verified to return
  byte-identical phonemes to the Python frontend the model was trained with.
- **The Azerbaijani text layer ported to JavaScript** -- `II` becomes `İkinci`,
  `25%` becomes `iyirmi beş faiz`, `01/09/1939` becomes `bir sentyabr min doqquz
  yüz otuz doqquzuncu il`. The port is checked against the Python original over
  a golden file: 550 normalisation cases and 24,022 number-word checks.

The panel on the right shows the normalised text that actually reaches the
model, which is the part of this project worth seeing.

## Speed

The model runs on your device, so the speed depends on the browser.

- **With WebGPU** synthesis is faster than real time.
- **Without WebGPU** (many phones, Firefox) it runs on the CPU. On the
  [direct link](https://ilqarrrr-inflect-micro-v2-azerbaijan.static.hf.space/index.html)
  the page is cross-origin isolated and uses up to 4 threads. Inside this
  Hugging Face page it cannot be, so it gets one thread and runs about half as
  fast; a sentence can then take up to a minute on a phone.

Without WebGPU the page uses an int8 decoder on the CPU: about 1.5x faster than
the full model and half the download, and its change to the sound measured
smaller than the model's own difference between two takes. A banner above the
player says when it is in use, with a button to speak in full quality instead
(`?precision=fp32` turns it off for the whole visit).

The page checks the device as soon as it opens, before anything downloads: a
card in the top bar shows a three-bar speed meter and the verdict, and a popup
says what to expect -- including the direct link when this Hugging Face page
limits the model to one thread.

Synthesis runs in a Web Worker, so the page never freezes, and a progress bar
shows how far it has got. The statistics under the player name the backend
that was used. The first visit downloads 37 MB; after that the graphs stay on
the device.

## Limitations

One voice, no emotion control, no voice cloning. At 9.36M parameters the speech
is clear but not fully natural, and the vocoder leaves a metallic resonance on
some sentences. Intonation flattens over long sentences, which is why the text
is cut into roughly fifteen-word chunks. eSpeak NG puts a stress on 92.2% of
words, so the reading is flat. The Azerbaijani training audio is most likely
synthetic, which caps the quality.

**Two voices.** The Azerbaijani one is this project's model. The English one is
[`owensong/Inflect-Micro-v2`](https://huggingface.co/owensong/Inflect-Micro-v2)
itself, from its official ONNX release -- a different voice by another author,
included so you can hear where this model started. Its 38 MB is downloaded only
if you pick it, and none of the Azerbaijani text layers apply to it: no number
rewriting, no stress layer.

## Licence

Apache-2.0 for the code and the model. Phonemisation uses `piper-phonemize`,
a WebAssembly build of eSpeak NG, which is **GPL-3.0-or-later** and is loaded
from a CDN by this page.

Do not use this voice to impersonate a real person or to produce deceptive
content.

The goal, the languages, the weak spots, how it works and a FAQ (including
commercial use), as plain pages:
[in English](https://ilqarrrr-inflect-micro-v2-azerbaijan.static.hf.space/about-en.html) ·
[in Azerbaijani](https://ilqarrrr-inflect-micro-v2-azerbaijan.static.hf.space/about.html).

Code, tests and the training pipeline:
<https://github.com/HuseynliIlqar/Inflect_Micro_v2_Azerbaijan>
