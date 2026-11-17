# Model card

## What it is

[`owensong/Inflect-Micro-v2`](https://huggingface.co/owensong/Inflect-Micro-v2)
-- a 9.36M-parameter, 24 kHz, single-voice English VITS variant -- adapted to
Azerbaijani. This card describes the checkpoint now in `model/`, **v2.1**, the
second Azerbaijani checkpoint.

| | |
| --- | --- |
| Architecture | VITS (compact), alias-free vocoder |
| Parameters | 9,356,513 |
| Sample rate | 24 000 Hz, mono |
| Voice | Single speaker |
| Phoneme frontend | eSpeak NG, `az`, with stress marks, then the stress frontend (`model/frontend_hook.py`) |
| Weight size | 37 MB (`model/model.pth`) |
| Licence | Apache-2.0 |

## How it was made

**The first checkpoint (v1)** was adapted from the base model on
[`ughurabbasov/azerbaijani-tts-dataset`](https://huggingface.co/datasets/ughurabbasov/azerbaijani-tts-dataset).
The warm start was complete: 409 of 410 tensors were copied bit-identically
from the base, and every Azerbaijani phoneme was already in the base model's
symbol set. [`training/README.md`](../training/README.md) archives that round.

**v2.1** continues from v1. The full record, with the test battery, is
[`training/v2/README.md`](../training/v2/README.md).

| | |
| --- | --- |
| Starting point | v1, all 410 tensors |
| Toolkit | Inflect at commit `6dad538`, with vectorised monotonic alignment |
| Data | 23,088 training clips and 471 validation clips, distilled from a teacher TTS; 1.0-14.5 s each; 3 duplicate transcripts dropped |
| Phonemes | prepared with the stress frontend |
| Main run | steps 0-5k posterior warm-up, 5k-20k linguistic adaptation, decoder unfrozen from 20k |
| Polish run | restarted from the main run's step 20k: 2.5k steps of discriminator warm-up with the generator frozen, then decoder training with stronger losses and discriminators (MPD, MRD, CQTD) |
| Exported | polish step 30,000, as PyTorch and ONNX (ONNX parity error 7e-9) |

The main run was stopped at step 41.7k: once the decoder unfroze, a fixed
6 kHz (fs/4) ring rose from 20% to 55% of frames. The polish run guarded
against that with a gradient clip of 500 per discriminator family, and the
generator skipped any batch whose discriminator loss exceeded 30.

### The stress frontend

eSpeak NG puts a stress mark on nearly every Azerbaijani word, and often on the
wrong syllable. The stress frontend phonemises with eSpeak NG and then moves the
stress marks with rules checked by a native speaker: negative verbs, tense and
person endings, unstressed particles, place names, postpositions and loans. A
rule only fires when it is sure; otherwise eSpeak's stress is kept.

v2.1 was trained on its output. The package ships it as
`model/frontend_hook.py`, and `AzTTS` runs it automatically; its source is
`training/stress_frontend/`. The phonemes a model is trained on and the
phonemes it is given must come from the same frontend, so phonemes handed in
directly should follow the same stress rules.

### Choosing the checkpoint

No loss value identifies the best-*sounding* model. During the second round,
every checkpoint read eight fixed sentences next to v1, with buzz and ring
measured for each. The polish run was stopped at 30.45k, after the
measurements had stopped improving past 25k, and step 30,000 was exported.
There is no `best.pth`.

## Measurements

All figures come from the 471 validation sentences, which were never trained
on. "Teacher" is the original recording. `variation` is the sampling noise
(`--variation`); 0.3 is the default.

| Measure | Teacher | v1 | v2.1, variation 0.667 | v2.1, variation 0.3 |
| --- | ---: | ---: | ---: | ---: |
| Whisper word error, 416 number-free sentences | 2.07% | 4.84% | 3.65% | ~2.65% (0.333) |
| Word error, 1-5 word sentences | 0.8% | 9.2% | 5.7% | 3.2% |
| SQUIM PESQ | 4.20 | 3.47 | 4.18 | -- |
| Harmonic clarity 2-4 kHz (buzz score, higher is better) | 10.6 | 6.4 | 8.8 | 8.9 |
| Fixed ring, share of frames | 13% | 20% (6 kHz) | 15% | 15% |
| Pitch range (semitones) | 16.8 | 13.0 | 15.3 | 15.0 |
| Yes/no question ending (semitones) | +2.9 | -3.0 | -1.2 | -1.8 |
| Sentences that fail with some of 5 seeds | -- | -- | 40% | 19% |

What the figures show:

- **The sound is at teacher level** by SQUIM, by the band spectrum (within
  1 dB) and by jitter, shimmer and HNR. Only harmonic clarity in 2-4 kHz lags.
  The fixed 6 kHz ring of v1 is gone.
- **Most pronunciation errors come from sampling noise.** Lowering `variation`
  cuts them by about 30% at almost no cost to pitch range. That is why the
  default is 0.3.
- **What remains** is listed under *Limitations* below.

The JSON behind each number is listed in
[`training/v2/README.md`](../training/v2/README.md).

## Measured speed

| Context | Result |
| --- | --- |
| The Python package on a laptop CPU | **2-4x real time** |
| ONNX, 1 thread, on the training pod's CPU | real-time factor 0.25 |
| ONNX, 4 threads, on the training pod's CPU | real-time factor 0.08 |

So an average laptop produces a minute of speech in 15-30 seconds.

## An honest note about the data

**v2.1 learnt from synthetic audio.** Its training clips were distilled from a
teacher TTS, so the quality ceiling is the teacher: this model cannot sound
better than the recordings it learnt from.

**The first checkpoint's dataset.** v1 was trained on
[`ughurabbasov/azerbaijani-tts-dataset`](https://huggingface.co/datasets/ughurabbasov/azerbaijani-tts-dataset),
and v2.1 starts from v1's weights. That audio is **most likely synthetic** too
-- the silences are of constant length, the noise floor is exactly zero, and
the spectrum carries vocoder traces.

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
- **Yes/no questions end falling.** The teacher's questions rise at the end;
  v2.1's fall (see the table). Only 677 training clips are such questions, and
  VITS has no explicit pitch model.
- **Consonant place confusions.** m->n, b->d and q->k, most often on a
  sentence's first and last word. A lower `variation` makes them rarer.
- **Stress is carried by length only.** The stress frontend's corrections are
  realised through length, but not yet through pitch.
- **Harmonic clarity at 2-4 kHz** is below the teacher's. `tools/audio_postprocess.py`
  still applies to any leftover resonance.
- **Sensitivity to the phoneme frontend.** Rare names and unusual spellings can
  be misread. A bare -ma/-mə order before a comma (`Bu kitabı alma, o birini
  al`) gets the stress of a verbal noun; the fix is in the stress frontend's
  source and reaches `model/` with the next export.

## Practical advice for better audio

1. **Change the seed.** For a sentence you do not like, try `--seed` from 1 to
   20 -- the difference is sometimes large.
2. **Punctuate.** Commas and full stops give both pauses and intonation.
   Unpunctuated text sounds flat.
3. **Keep `--variation` low.** The default, 0.3, gives the fewest
   pronunciation errors; raise it for a livelier but less predictable voice.
4. **Write long numbers as words** if normalisation does not read them the way
   you want -- check what is going in with `--show-text`.
5. **Clean leftover resonance** in saved audio:
   `python tools/audio_postprocess.py analyze out`, then `clean`.

## The English base model

The repository also ships `model-en/`: the English checkpoint this model was
adapted from, `owensong/Inflect-Micro-v2`, redistributed unmodified under
Apache-2.0 so that a clone speaks both languages. It is a different voice by
another author, synthetic, and none of the Azerbaijani text layers apply to it
except the profanity bleep (`censor_az`).
`python say.py --voice en "Hello."`, or the voice selector in the interface.
Samples for it are in `samples/en/`; see `model-en/README.md` for provenance and
`README.md` for what it means for commercial use.

**This model does not speak English.** It was trained only on a single
Azerbaijani voice; English text pushed through it is that voice reading
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
- Stress lexicon: [`LocalDoc/azerbaijani_spelling_dictionary_2021`](https://huggingface.co/datasets/LocalDoc/azerbaijani_spelling_dictionary_2021)
  (MIT); the stress frontend's word lists are derived from it and embedded in
  `model/frontend_hook.py`.
