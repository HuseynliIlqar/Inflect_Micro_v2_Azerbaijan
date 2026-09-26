---
title: Azerbaijani TTS
emoji: 🗣️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Offline Azerbaijani text-to-speech, 9.36M parameters, on CPU
models:
  - owensong/Inflect-Micro-v2
datasets:
  - ughurabbasov/azerbaijani-tts-dataset
---

# Azerbaijani TTS

A fully offline text-to-speech model that speaks Azerbaijani. 9.36M parameters,
24 kHz mono, 2-4x faster than real time on a CPU. Type a sentence, adjust the
parameters, listen.

The panel on the right exposes every setting the command line has, and the page
shows two things a demo usually hides: the normalised text that actually reaches
the model (`25%` becomes `iyirmi beş faiz`), and the `say.py` command matching
the settings you picked.

**Two voices.** The Azerbaijani one is this project's model. The English one is
[`owensong/Inflect-Micro-v2`](https://huggingface.co/owensong/Inflect-Micro-v2)
by Owen Song, the checkpoint this model was adapted from, included unchanged --
a different voice by another author.

This Space runs on free CPU hardware and sleeps after a period without visitors,
so the first request after a quiet spell takes a few seconds longer while the
model loads.

Code, tests and the full training pipeline:
<https://github.com/HuseynliIlqar/Inflect_Micro_v2_Azerbaijan>

## Licence

Apache-2.0 for the code and both checkpoints. Phonemisation goes through
`phonemizer` and eSpeak NG, both GPL-3.0-or-later; this Space redistributes
them, and its source is public here. See `THIRD_PARTY_NOTICES.md` in the GitHub
repository for what that means if you redistribute a bundle of your own.

Do not use either voice to impersonate a real person or to produce deceptive
content.
