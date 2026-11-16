# Third-party code in this folder

`discriminators.py` and `losses.py` adapt code from:

- **NVIDIA BigVGAN** — https://github.com/NVIDIA/BigVGAN (commit 7d2b454), MIT License,
  Copyright (c) 2024 NVIDIA CORPORATION. Discriminator architectures (MPD, MRD, CQTD) and the
  multi-scale mel loss. The released `nvidia/bigvgan_v2_24khz_100band_256x` discriminator weights
  (MIT) are loaded at the start of training.
- BigVGAN itself credits, all under the MIT License: HiFi-GAN (jik876), UnivNet (maum-ai),
  Vocos (gemelo-ai), descript-audio-codec (descriptinc) and audiotools (descriptinc).
- The multi-resolution STFT loss follows Parallel WaveGAN (Yamamoto et al., 2020).

MIT License text: see BigVGAN's LICENSE file; the notice above must be kept with copies of this code.
