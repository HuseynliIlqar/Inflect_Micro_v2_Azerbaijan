# Model card

## What it is

[`owensong/Inflect-Micro-v2`](https://huggingface.co/owensong/Inflect-Micro-v2)
-- a 9.36M-parameter, 24 kHz, single-voice English VITS variant -- adapted to
Azerbaijani.

| | |
| --- | --- |
| Architecture | VITS (compact), alias-free vocoder |
| Parameters | 9,356,513 |
| Sample rate | 24 000 Hz, mono |
| Voice | Single speaker (F0 195-201 Hz) |
| Phoneme frontend | eSpeak NG, `az`, with stress marks |
| Weight size | 37 MB (`model/model.pth`) |
| Licence | Apache-2.0 |

## Training

| | |
| --- | --- |
| Dataset | 9,674 clips, **25.07 hours**, single speaker |
| Phoneme coverage | **1.0** -- no new symbols, a complete warm start |
| Steps | **200,000** = **1,343 epochs** |
| Effective batch | 64 (batch 64 x accumulation 1) |
| Optimiser | AdamW, lr 2e-4, betas (0.8, 0.99) |
| Hardware | NVIDIA A40, 45 GB (peak VRAM 26.2 GB) |
| Duration | ~3.7 days (1.6 s/step) |
| Cost | roughly $39 |

The warm start was complete: 409 of 410 tensors were copied bit-identically from
the base, and only `enc_q.` (the posterior encoder, 1.86M parameters used during
training) started from scratch. 178 embedding rows were migrated and none were
created anew -- every Azerbaijani phoneme was already in the base model's symbol
set.

### Why there is no `best.pth`

No loss value identifies the best-*sounding* model. The vocoder loss can keep
falling while the reading grows more metallic. The checkpoint was chosen by ear,
on clarity, buzz/resonance, clipped word endings and sibilant stability. The
weights in this package are from step 200,000.

## Measured speed

On a local laptop CPU:

| Context | Result |
| --- | --- |
| Short sentence | **2-4x real time** |
| Two minutes of text | 122 s of audio / 43 s of compute |
| Model load | 0.3-2 s |
| 1 thread (phone-like) | 2.5x real time |

So an average laptop produces a minute of speech in 15-30 seconds.

## An honest note about the dataset

The training data is built on
[`ughurabbasov/azerbaijani-tts-dataset`](https://huggingface.co/datasets/ughurabbasov/azerbaijani-tts-dataset).

The audio is **most likely synthetic** -- the silences are of constant length,
the noise floor is exactly zero, and the spectrum carries vocoder traces. What
follows from that:

- **The quality ceiling is the source system.** This model cannot sound better
  than whatever produced its training data.
- **Licence.** The dataset repository declares no licence file. Asked directly
  in the HuggingFace community tab, the author confirmed that anyone may use the
  dataset and asked to be credited, so attribution is the condition and this
  project credits it here and in `README.md`. The terms of the upstream TTS
  system that produced the audio are a separate question and remain unknown.
- The voice does not belong to a real person (or has not been confirmed to), but
  do not rely on that to imitate anyone.

## Limitations

- **One voice, no emotion control.** The reading is neutral and cannot be
  changed.
- **The model is small.** 9.36M parameters is enough for clear speech, not for
  full naturalness. Intonation flattens over long sentences -- which is why
  `az_chunk` cuts text into roughly fifteen-word chunks.
- **Vocoder resonance.** Some sentences carry a metallic ring in the high
  frequencies. An 11 kHz low-pass (`ffmpeg -af lowpass=f=11000`) reduces it.
- **Sensitivity to the phoneme frontend.** eSpeak NG is good for Azerbaijani,
  but rare names, foreign words and unusual spellings can be misread.
- **eSpeak stresses nearly every word** (92.2% of the words in the training data
  carry a primary stress). That is why the reading is flat. `--prosody safe`
  thins the stress marks out, but it is experimental and does not improve every
  text.

## Practical advice for better audio

1. **Change the seed.** For a sentence you do not like, try `--seed` from 1 to
   20 -- the difference is sometimes large.
2. **Punctuate.** Commas and full stops give both pauses and intonation.
   Unpunctuated text sounds flat.
3. **Lower `--variation`** (0.4-0.5) for a steady, formal reading; raise it
   (0.8) for a livelier but less predictable voice.
4. **Write long numbers as words** if normalisation does not read them the way
   you want -- check what is going in with `--show-text`.
5. **Filter the metallic ring:**
   `ffmpeg -i input.wav -af lowpass=f=11000 output.wav`.

## The English base model

The repository also ships `model-en/`: the English checkpoint this model was
adapted from, `owensong/Inflect-Micro-v2`, redistributed unmodified under
Apache-2.0 so that a clone speaks both languages. It is a different voice by
another author, synthetic, and none of the Azerbaijani text layers apply to it
except the profanity bleep (`censor_az`).
`python say.py --voice en "Hello."`, or the voice selector in the interface.
Samples for it are in `samples/en/`; see `model-en/README.md` for provenance and
`README.md` for what it means for commercial use.

**This model does not speak English.** It was adapted over 200,000 steps on a
single Azerbaijani voice; English text pushed through it is that voice reading
phonemes it was never trained on.

## Sources

- Base model: [`owensong/Inflect-Micro-v2`](https://huggingface.co/owensong/Inflect-Micro-v2) (Apache-2.0)
- Architecture: [VITS](https://github.com/jaywalnut310/vits) (MIT)
- Vocoder ideas: [BigVGAN](https://github.com/NVIDIA/BigVGAN) (MIT),
  [alias-free-torch](https://github.com/junjun3518/alias-free-torch) (Apache-2.0)
- Phonemisation: [eSpeak NG](https://github.com/espeak-ng/espeak-ng) and
  [phonemizer](https://github.com/bootphon/phonemizer), both GPL-3.0-or-later --
  see [../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for what that means
  if you redistribute.
