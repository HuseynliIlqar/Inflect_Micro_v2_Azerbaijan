# Third-Party Notices

This project's own code -- everything under `aztts/`, `tools/`, `tests/`,
`training/` and `say.py` -- is Apache-2.0. The components below are not ours,
and their own licences apply to them.

Two of them are copyleft. If you only install and run this project, nothing here
asks anything of you. If you **redistribute** a combined work -- a packaged
application, a container image, a frozen binary, a hosted service that ships the
code -- read the "Phonemisation" section first.

---

## Phonemisation: GPL-3.0-or-later

Text becomes phonemes through two GPL components. They are runtime dependencies,
pulled in by `requirements.txt`, and are not vendored into this repository.

### phonemizer

- Project: [`bootphon/phonemizer`](https://github.com/bootphon/phonemizer)
- Copyright: Copyright (c) 2015-2021 Mathieu Bernard
- Licence: **GNU General Public License v3.0 or later**

`model/deployment_frontend.py` imports `phonemizer` and calls it in-process. That
is linking, not a subprocess call.

### eSpeak NG

- Project: [`espeak-ng/espeak-ng`](https://github.com/espeak-ng/espeak-ng)
- Licence: **GNU General Public License v3.0 or later**

eSpeak NG provides the Azerbaijani (`az`) phoneme rules. It reaches the process
as a shared library: `espeakng-loader` ships pre-compiled eSpeak NG binaries
inside its wheel, and `deployment_frontend.py` loads that library with `ctypes`
via `phonemizer`. `espeakng-loader` itself declares no licence of its own on
PyPI, so the binaries it distributes are governed by eSpeak NG's GPL-3.0.

### What that means in practice

| You are | What applies |
| --- | --- |
| Installing and running this project | Nothing extra. Use it as you like. |
| Modifying it for your own use | Nothing extra. |
| **Redistributing a combined work** | GPL-3.0 applies to the phonemiser components you ship, including the obligation to offer corresponding source. |

The safe reading is that a distributed bundle containing `phonemizer` and the
eSpeak NG library must satisfy GPL-3.0 for those parts. Our Apache-2.0 grant on
our own code is unaffected; the two licences are compatible in that direction
(Apache-2.0 code may be combined into a GPL-3.0 work, not the reverse).

### If you need to avoid the GPL

The checkpoint itself is Apache-2.0 and does not depend on eSpeak. Only the
frontend does. Two routes out, neither implemented here:

1. **Pre-phonemise elsewhere.** `model/config.json` declares
   `accepts_prephonemized_input: true`, so phonemes can be produced by any tool
   you like and handed to the model directly.
2. **Call `espeak-ng` as a subprocess** instead of linking `phonemizer`. Invoking
   a separate program is a materially weaker link than loading a library into
   your process. This changes the frontend, so the phoneme output must be
   verified against the current pipeline before trusting it.

Neither route is legal advice. If redistribution matters to you, have someone
qualified look at it.

---

## Model architecture and runtime

The following are vendored inside `model/` (and inside `training/base-model/`
once it is downloaded), as part of the upstream Inflect v2 export. Their licences
apply to the corresponding portions; the rest of that package is Apache-2.0.

### VITS

- Project: [VITS](https://github.com/jaywalnut310/vits) (`jaywalnut310/vits`)
- Copyright: Copyright (c) 2021 Jaehyeon Kim
- Licence: MIT

The compact model architecture and several inference runtime modules derive from
VITS. The text frontend also retains its original Keith Ito MIT licence at
`model/runtime/text/LICENSE`.

### BigVGAN

- Project: [BigVGAN](https://github.com/NVIDIA/BigVGAN) (`NVIDIA/BigVGAN`)
- Copyright: Copyright (c) 2024 NVIDIA CORPORATION
- Licence: MIT

The lightweight alias-free waveform activation implementation derives from
BigVGAN's alias-free design and was adapted for this compact runtime.

### alias-free-torch

- Project: [`junjun3518/alias-free-torch`](https://github.com/junjun3518/alias-free-torch)
- Licence: Apache License 2.0

The anti-aliased activation resampling design used by the compact waveform
runtime includes concepts and adapted implementation structure from
`alias-free-torch`.

### Base checkpoint

- Model: [`owensong/Inflect-Micro-v2`](https://huggingface.co/owensong/Inflect-Micro-v2)
- Author: Owen Song
- Licence: Apache-2.0

The weights in `model/` are an adaptation of this checkpoint, so the Apache-2.0
grant carries through to them.

---

## Training data

- Dataset: [`ughurabbasov/azerbaijani-tts-dataset`](https://huggingface.co/datasets/ughurabbasov/azerbaijani-tts-dataset)

The dataset repository declares no licence file. Asked directly in the
HuggingFace community tab, the author confirmed that anyone may use the dataset
and asked to be credited. Attribution is therefore the condition, and this
project credits the dataset here, in `README.md` and in `docs/MODEL.md`.

The audio is most likely synthetic, produced by another TTS system (see
`docs/MODEL.md` for the evidence). The terms of that upstream system are not
known, which is a separate question from the dataset author's permission.

---

## Note on this package

The `third_party/*.txt` files referenced by the upstream export were not
included in it. The full licence texts can be read here:

- VITS (MIT) -- <https://github.com/jaywalnut310/vits/blob/main/LICENSE>
- BigVGAN (MIT) -- <https://github.com/NVIDIA/BigVGAN/blob/main/LICENSE>
- alias-free-torch (Apache-2.0) -- <https://github.com/junjun3518/alias-free-torch/blob/main/LICENSE>
- phonemizer (GPL-3.0+) -- <https://github.com/bootphon/phonemizer/blob/master/LICENSE.txt>
- eSpeak NG (GPL-3.0+) -- <https://github.com/espeak-ng/espeak-ng/blob/master/COPYING>

Keith Ito's text frontend licence ships inside the package:
`model/runtime/text/LICENSE`.
