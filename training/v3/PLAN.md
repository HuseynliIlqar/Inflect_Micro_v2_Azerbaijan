# Avaz v3 plan

**Starting point:** the v2 checkpoint in `model/` ([training/v2](../v2/README.md)). It has a backup outside the repository.

**Goal:** fix the weaknesses measured in v2 at their root, then teach the model Turkish (v3.1). The v2 weaknesses are:
- yes/no questions end falling;
- m→n, b→d and q→k confusions;
- stress carried only by length;
- a narrow pitch range;
- the 2–4 kHz harmonic gap.

## Overview

| # | Stage | Where | Time | Cost |
| --- | --- | --- | --- | --- |
| 1 | `variation` default 0.3 | PC | done | 0 |
| 2 | Text layer: the remaining gaps | PC | done | 0 |
| 3 | Record dataset v2 with careful take selection (done, 3.6) | pod | ~1 day | ~$15–25 |
| 4 | Rhythm part: VITS2 duration discriminator | PC | ½ day | 0 |
| 5 | Pitch part: a predictor that controls the voice's pitch | PC | 1–2 days | 0 |
| 6 | New decoder that takes the predicted pitch | PC | 2–3 days | 0 |
| 7 | Short trial run | pod | a few hours | ~$3 |
| 8 | Full run | pod | ~2–2.5 days | ~$25–40 |
| 9 | Test against v2 | pod + PC | ½ day | ~$1 |
| 10 | v3.1: Turkish, with VoxCPM2 as teacher | PC + pod | ~4–6 days | ~$30–50 |
| W | Port the stress frontend to `web/` | PC | ~1–2 days | 0 |

**Ground rules:**
- Stage 3 runs on the pod while stages 4–6 are coded on the PC.
- The text layer (2) and the data (3) are final before training starts.
- No code reaches the full run (8) without passing the trial (7).
- Every pod is deleted once its results are downloaded and verified.

**Turkish waits until v3 is measured.** v3 already brings three big changes, and a fourth would hide which one caused what. Two small preparations go into v3 itself (stage 2 and stages 4–6), so v3.1 needs no architecture change.

**Stage W is independent of training.** The playground phonemises in the browser with plain eSpeak NG (`web/js/engine/phonemize.js`), and `web/onnx/` still holds v1 graphs. A v2 or v3 graph only belongs there once the stress frontend runs in JavaScript too. Until then the playground stays on v1.

---

## 1. `variation` default 0.3 (done)

- `aztts/engine.py`, `aztts/en_voice.py`, `say.py`, `webui/runner.py`, the labels in `webui/i18n.py` and both READMEs now default to 0.3.
- The user listened to 0.667, 0.3 and 0.2 on the same texts and chose 0.3.
- The static playground (`web/`) keeps its own default, because it still runs v1.
- English was not measured at 0.3. Listen to it before release.

## 2. Text layer: remaining gaps

**Status (2026-11-12): done.**

| Item | Result |
| --- | --- |
| Foreign words | `FOREIGN_WORDS` and `_replace_foreign_words` in `aztts/`, mirrored in `web/js/text/`. 15 tests; 14 new golden cases; JS and Python agree on all 669 cases; no existing output changed. |
| Negative imperatives before a comma | The stress frontend's source now lives in `training/stress_frontend/`. Its new `_clause_imperative` rule marks 210 of ~260 orders in dataset v2. On v1's 23,559 sentences it changes 8 words, all real orders. 344 tests pass. |
| Turkish symbols | 60 phonologically varied Turkish sentences through eSpeak `tr` use 43 symbols, all already in the 178-symbol inventory. **No symbol needs adding.** |

**For stage 3 onward:** prepare the v3 data with the rebuilt hook (`training/stress_frontend/dist/az_stress_frontend.py`), not the v2 copy in `model/`.

The original description of this stage follows.

`normalize_az` already reads ordinals, Roman numerals, units, `Dr.`, `və s.`, minus signs, times and percentages correctly. Two gaps remain:
- **A small lexicon for foreign words in Latin spelling** (Microsoft, Google, iPhone, …). It goes in `aztts/az_tables.py` and its mirror `web/js/text/az-tables.js`, with tests, and the golden files are regenerated.
- **Context stress for negative imperatives.** In "kitabı alma" the stress is initial, but in "alma" meaning apple it is final. This belongs to the stress frontend's rules (`model/frontend_hook.py` is vendored, so the change is made in the frontend's source, `D:\ktts\avaz-stress`, and re-exported). It can also be a pre-pass in `aztts/`.

**Turkish preparation:** phonemise ~2,000 Turkish sentences with eSpeak `tr` and compare them with the 178-symbol inventory. Any missing symbol is added before v3 trains; v3 never uses it, so it is harmless there.

**Acceptance:**
- all tests pass;
- every phoneme is in the inventory;
- **the phonemes of v1's digit-free sentences do not change.**

## 3. Record dataset v2 (careful selection)

**Input:** `training/v2/dataset_v2/sentences_v2.jsonl`, 14,853 lines. The user reads a sample before recording starts.

### 3.1 Match v1 exactly

| Setting | Value |
| --- | --- |
| Teacher | the same teacher TTS, version and checkpoint as v1 (named on the pod by the `TEACHER_*` variables, see `record/generate_takes.py`) |
| Reference | `kit_corpus/refs/ref_data.wav` (sha256 `f2e3df2a…c4fc`) with its transcript in `refs/refs.json` |
| Generation | `num_step=32`, `guidance_scale=2.0`, `position_temperature=0.0`, `class_temperature=0.0`, `denoise=True`, `preprocess_prompt=True`, `postprocess_output=False`, `speed=0.9`, `language="az"` |
| Output | 24 kHz PCM_16, no loudness normalisation |
| Manifest | `{"audio": "audio/az-01-main/NNNNNN.wav", "text": …, "speaker": "az-01-main"}` |

### 3.2 Pilot (~600 sentences, ~40 per category)

1. **Determinism test.** Read the same sentence twice with temperatures at 0 and different seeds. If the two takes are identical, extra takes need another source of variety.
2. **Takes:**
   - **A** uses the v1 settings exactly.
   - **B** and **C** sample the timing: `position_temperature` 5.0 (seeds 2 and 3), everything else as v1.
   - Yes/no questions (A1–A3) get two more sampled takes, **D** and **E** (seeds 613 and 214).
3. **Measure** Whisper error, buzz, speaker similarity and the question ending for each take, and listen to ~5 clips per category.
4. **Calibrate** the gates against v1's distribution. Settings and thresholds are written down and approved before the full run.

### 3.3 Hard gates (each take)

| Check | Threshold | Catches |
| --- | --- | --- |
| Duration | 1.0–25 s. 14.5 s maximum except for paragraphs and long sentences | truncation, run-on |
| Pace | 2.6–8.5 syllables/s (v1 rule) | skipped or invented speech |
| Peak / clipping | peak ≥ 0.02, clipped ≤ 0.1% | silence, distortion |
| Whisper character error | ≤ 2%; J ≤ 5%; paragraphs ≤ 3% | misreading |
| Numbers (I) | pace and speaker checks only, plus 30 clips by ear (Whisper writes digits) | number misreading |
| Speaker similarity (ECAPA) to the reference | ≥ v1's 2nd percentile | timbre drift |
| Buzz score | ≥ v1's 10th percentile | buzzy take |
| Ring share | ≤ v1's 90th percentile | ringing take |
| Pitch range | ≥ v1's 2nd percentile (12.8 st) | monotone take |
| Exclamations (C) | pitch range ≥ v1's 25th percentile (16.1 st) | flat exclamation |

Yes/no questions have **no hard pitch gate**. The pilot's ear test (3.8) showed that a gate on a final rise keeps the wrong takes.

### 3.4 Choosing a take

Among the takes that pass, choose by:
1. lowest Whisper error;
2. for yes/no questions, the largest rise-then-fall (`end_drop_st`: the held peak of the last 0.8 s minus the pitch at the very end);
3. highest buzz score;
4. highest speaker similarity.

Yes/no questions choose only among the sampled takes B–E (`select_takes.py --question-takes B C D E`). On a tie, the earlier take wins. If no take passes, record up to 2 more; if those fail too, drop the sentence and log why. If more than 20% of a category is dropped, stop and investigate.

### 3.5 Guarding the selection

- Compare the chosen takes with all takes on pace, pitch range, duration and loudness. The selection must not drift towards the fastest, flattest or shortest takes.
- Compare the new set with v1 on loudness, pace and edge silence.
- Keep every score of every take in `scores.jsonl`.

### 3.6 Post-processing

- Match edge silence to v1 (median 0.23 s before and 0.32 s after).
  - **Pilot check:** measured the same way (10 ms frames, 40 dB below the loudest), v1's 400 sample clips have 0.23 s before and 0.31 s after; the pilot's chosen takes have 0.28 s and 0.31 s. The 90th percentiles also match (0.95 vs 0.86 s before). So `finalize.py` copies the takes unchanged and reports the medians in `qc_summary.json`.
- **Extract the pitch track of every v1 and v2 clip** at the model's frame step (256 samples at 24 kHz). This is the input for stage 5. It runs on the PC at the start of stage 5, so the tracker can be chosen together with the pitch part.

`finalize.py` writes the set in v1's layout: `audio/az-01-main/v2_<id>.wav`, `metadata.jsonl`, `clip_map.jsonl` (take and scores per clip), `qc_summary.json` and `checksums.sha256`. The "v2_" prefix keeps the names apart from v1's, so both sets can train together.

**Full run (2026-11-12 11:36 UTC to 2026-11-13 05:10 UTC, pod `6o7zqflbvdwcjw`):** `full_generate.sh` and `full_score.sh` ran side by side; `finalize.py` ran after `full.done`.
- **Generation:** 50,335 takes (A, B, C on all 14,853 sentences; D, E on the 2,798 yes/no questions), no failures. The teacher sets the length from the text, so every take of a sentence has the same duration.
- **Scoring was the bottleneck.** Sharing the GPU with generation, it scored 30–55 clips a minute against 150 alone, so it finished 3 h after generation. Next time, score in parallel processes once generation ends.
- **Result:** 13,557 of 14,853 sentences kept (91.3%), 19.33 h. Questions A1–A3 kept 2,582 (v1 had 677). Lowest category: C, exclamations, 75.7% (148 flat). Drop reasons: misread 690, buzz 428, ring 412, monotone 226, flat exclamation 148, voice 39, too long 4. Chosen takes: A 3,787, B 4,190, C 4,314, D 616, E 650. Edge silence 0.24 / 0.31 s. Bias guard: the chosen takes' median buzz is 11.1 against 10.3 for all takes, range 17.1 against 16.9 st.
- **Delivery:** the dataset (sha256 verified on the PC), every take as FLAC with PCM checksums (all 50,335 verified bit-exact), scores, selection and the listening page are in `D:\ktts\avaz-train\record_full\`. The pod was deleted on 2026-11-13 at 08:11 UTC. The pod cost ~$14 in all, counting the pilot and the English comparison ($21.61 before, $7.53 after).

### 3.7 Delivery

- A listening page with 25 random and 25 borderline clips per category, plus the distribution of question endings.
- A report of counts, drop reasons and hours.
- Download, verify with sha256, then **delete the pod**.

**Acceptance:**
- ≤ 10% dropped;
- A1–A3 takes are chosen by the rule in 3.4, and the user spot-checks them by ear;
- speaker similarity at v1 level;
- the user approves the listening page.

### 3.8 Pilot results (2026-11-12)

**Run:** 600 sentences (40 per category) × takes A, B, C on an A40, plus D, E, F on the 120 questions. RTF 0.164.

**Determinism:** 27 of 60 repeated A clips were identical; the rest differ because batches mix sentences of different lengths. With temperatures at 0, variety needs `position_temperature`.

**Ear test of yes/no questions.** The user rated all 6 takes of 24 questions (144 takes) as a proper question or not.

| Take | Sounds like a proper question |
| --- | --- |
| A (deterministic, v1 settings) | 8 / 24 |
| B–F (sampled, same settings, different seeds) | 81 / 120 |

What the ratings showed:
- Good takes **rise on the syllable before "-mı" and fall on the particle**. Their final movement is lower than that of bad takes (median −2.0 vs −1.2 st), so the planned "ending ≥ +1 st" gate picked the wrong takes: it kept 9 of the 24 rated questions, and only 3 of those were good.
- No single pitch measure separates good from bad well (best balanced accuracy 0.67, for `end_drop_st`). Pitch range does not either. Whisper writes "?" whenever the text has "-mı", so it cannot judge intonation.
- Choosing the take with the largest `end_drop_st` among k sampled takes gives a good question in 72% (k=2), 76% (k=3), 78% (k=4) and 79% (k=5) of sentences, against 68% for a random sampled take and 33% for A. These figures are measured on the same ratings the rule came from, so expect a little less on new sentences.

**Gate changes from the pilot:**
- the final-rise gate is removed;
- the pitch-range gate moves from v1's 10th to its 2nd percentile;
- the exclamation gate moves from v1's median to its 25th percentile;
- questions use 4 sampled takes (B–E) and the rise-then-fall preference.

**Effect on the pilot:** kept 466 → 554 of 600 (78% → 92%). A1–A3 go from 38–43% to 88–98% kept. The rated questions: 24 of 24 kept, 18 judged good by ear. The bias guard is unchanged (chosen vs all: duration 3.50 vs 3.36 s, range 17.1 vs 16.7 st, buzz 11.0 vs 10.4).

**Tools:** `question_contours.py` (pitch shape per question take), `build_question_page.py` (the rating page), `pitch.py` (the cleaned pitch track shared with `score.py`), `common.rise_fall_st` (the shape measure, pure Python and tested).

## 4–6. Shared rule: a language-tag input

Add a small language embedding, zero-initialised, to the text encoder, the pitch part and the rhythm part. v3 has one language, so the tag changes nothing. v3.1 plugs Turkish in through it. A CPU test proves the output is the same with and without it.

## 4. Rhythm part (VITS2 duration discriminator)

A discriminator compares predicted durations with real ones. It exists only at training time, so export carries no extra tensors.

**Status (2026-11-13): code done, not yet in a training loop.** The v3 code lives in `training/v3/model/`; its tests run on the CPU without the model file:

```bash
.venv/Scripts/python.exe -m pytest training/v3/model/tests training/v3/data/tests/test_pitch_dataset.py -q -o addopts=""
```

- `duration_discriminator.py`: VITS2's discriminator in the "V2" form most ports use. The least-squares losses are averaged over real tokens only.
- `synthesizer_v3.py`: the v3 network. The v2 parts keep their names, so a v2 training checkpoint loads into them. The tests check that it reproduces v2 exactly in training and in inference while the new parts are at their starting values. Its `forward` also returns the hidden states and both log durations the discriminator needs.
- `runtime.py`: imports `model/runtime` read-only without leaking its module names.
- `monotonic_align.py`: alignment search, checked step for step against VITS' Cython recurrence.
- The discriminator goes into `train_v3.py`, which is written together with stage 6.

## 5. Pitch part

- It predicts each phoneme's pitch from the text encoder, plus voiced/unvoiced.
- It works on speaker-normalised semitones, so human pitch contours can train it later.
- It uses a stochastic or adversarial loss; plain L2 compresses the range.
- Training uses the true pitch; inference uses the prediction.
- Size: ~0.3–1M parameters.
- It must be ONNX-exportable.

**Status (2026-11-13): code and pitch tracks done.**

**Pitch tracks** (`training/v3/data/`, outputs in `D:\ktts\avaz-train\pitch\`):
- `pitch_track.py` gives one value per spectrogram frame, at the frame centres (sample 256 i + 128). It uses the take selection's two-pass Praat tracker.
- `extract_pitch.py` processed v1 (23,562 clips) and v2 (13,557) in 5 minutes on the PC.
- Both sets have a median of 177–179 Hz and 52% voiced frames. The 5th–95th percentile range is −4.7 to +9.3 semitones.
- Octave-sized jumps (over 9 semitones between neighbouring frames) occur in 0.014–0.019% of voiced frame pairs.
- The toolkit's preparation does not change audio length, so the PC tracks line up with the pod's frames. `pitch_dataset.py` finds each prepared clip's track by its file name, refuses length mismatches and adds `f0` to the batch.

**Design, as built in `training/v3/model/pitch_part.py`:**
- **Frame level, not phoneme level.** The predictor reads the text hidden states expanded to frames and predicts semitones from the voice's median (178 Hz) plus a voiced/unvoiced logit. Its input is detached, like v2's duration predictor. It has 0.94M parameters, all convolutions with a ~65-frame (0.7 s) field; the hidden states already carry sentence context.
- **Losses.** L1 on voiced frames, voicing cross-entropy, and a contour discriminator against range compression (the stage 4 idea applied to pitch).
- **F0PriorShift (zero-initialised).** It shifts the frame-level prior mean by a learnt function of the pitch. Training feeds the true pitch, so the flow learns latents that agree with it; inference feeds the prediction. Alignment still uses the per-phoneme prior, unchanged from v2.
- **Controls.** `predict_pitch` has `pitch_range` and `pitch_shift` knobs for later.

## 6. New decoder

- **Design:** a harmonic source built from the pitch (sine harmonics up to at least 4–5 kHz) plus a noise source and an iSTFT output layer. The iSTFT is written as a convolution, so ONNX can export it.
- **Reused from the v2 checkpoint:** the text encoder, the flow and the trained discriminators (MPD, MRD, CQTD).
- **Acceptance:**
  - ≤ ~12M parameters in total;
  - ONNX matches PyTorch;
  - ONNX real-time factor on 1 thread ≤ 0.25 (v2: 0.249).

**Changed 2026-11-13, with the user's approval: keep v2's decoder and add a harmonic source (NSF), instead of a new iSTFT decoder.**

- **Why.** v2 already sounds at teacher level (PESQ 4.18 vs 4.20), so a decoder trained from scratch risks losing that. Keeping v2's decoder keeps everything it learnt. The harmonic source supplies what it lacks, clear harmonics in 2–4 kHz, and the decoder run is shorter.
- **As built (`training/v3/model/harmonic_source.py`).**
  - From the pitch per frame, 48 sine harmonics are built at the sample rate, each kept only below 6 kHz. Unvoiced samples get noise. One learnt linear layer mixes them.
  - One strided convolution per upsampling stage adds the source at that stage's rate (NSF-HiFiGAN, as in DiffSinger and so-vits-svc). These convolutions start at zero, so v3 starts as v2.
  - Phase is summed per frame and then within the frame, so float32 stays accurate to ~1e-4 cycles over a long sentence.
  - `SynthesizerV3.decode` is v2's `Generator.forward` with the injections. In training the decoder gets the true pitch, sliced with the same frames as the latent; at inference it gets the predicted pitch.
- **Acceptance, checked on the PC (`model/onnx_check.py`, full size, random weights):**
  - 10.32M inference parameters (v2 9.36M; the pitch part 0.94M, the source 11k);
  - ONNX matches PyTorch to 8e-6;
  - ONNX RTF on 1 thread 0.242 against 0.237 for v2's path on the same PC (+2%). The pod's CPU measured v2 at 0.249, so v3 may land just over 0.25 there. Measure on the pod in stage 9.
  - The file is 41.5 MB instead of 37.6 MB (fp32).

**Changed again 2026-11-14: the NSF route did not remove the ring, so the frame-rate decoder was built after all.**

- **Why.** At v3 step 20,000 (60 held-out test sentences plus the 12 listening sentences, two renders each) the intonation range was 16.7 semitones (real recordings 17.3, v2 15.0). Buzz and ring were not better than v2's: buzz 8.73 against 8.95, ring 17.9% against 15.0% (real recordings 10.50 and 12.7%). Raising the source's learning rate ×3 made the ring audibly worse, so the joint run was stopped at step 22,889.
- **As built.** `frame_decoder.py` ports Paradee's frame-rate source-filter head ("v2", Apache-2.0) to 256-sample frames. A ConvNeXt trunk reads z and the pitch. The head shapes a phase-locked sine bank and three noise paths in the STFT domain, with no upsampling.
  - Size: 4.66M parameters, against 5.77M for v2's decoder.
  - Speed: decoder RTF 0.021 on one CPU thread.
  - Wrapper: `synthesizer_fd.py` keeps v3's text side, flow and pitch part.
- **Training.** `train_fdec.py`, with v3 frozen at step 20,204 (before the ×3 change).
  - Stage one: 40k steps with log-mel + multi-resolution STFT loss, harmonicity ×2 and modulation ×5.
  - Stage two: 25k LS-GAN steps.
  - After 2k trial steps the ring was already at the dataset's level (12.1%).
- **Result at 65k (12 listening sentences).** Buzz 9.37, ring 14.3%. At v3 step 20,000, with v2's decoder, buzz was 8.88 and ring 17.6%.
- **Where the rest is lost.** `monitor/decoder_ceiling.py` ran on 60 held-out clips, with the true durations and pitch:

  | Condition | Buzz | Ring | SQUIM PESQ |
  |---|---|---|---|
  | Real recordings | 10.12 | 12.9% | 4.20 |
  | v2 decoder fed the recording's latent | 8.50 | 17.4% | 4.20 |
  | Frame decoder fed the recording's latent | 9.87 | 13.5% | 3.06 |
  | v2 decoder fed the text side's latent | 9.26 | 18.4% | 4.21 |
  | Frame decoder fed the text side's latent | 9.28 | 13.6% | 2.87 |

  - Fed the recording's latent, the frame decoder nearly reaches the recordings. Fed the text side's latent, it loses about 0.6 dB of buzz.
  - Its log-spectral distance to the recordings is lower than v2's decoder in every band. There are no clicks, and its noise floor is lower.
  - Its low SQUIM PESQ is not explained yet, and needs a listening check.
- **Next: `--latent gta` (run v5_gta).** The decoder is fine-tuned from 65k for 15k adversarial steps at lr 1e-4. Its input is the text side's prior for the true alignment and true pitch, sampled at 0.3 as when speaking.

## 7. Trial run

3–5k steps on a subset. It checks that:
- the losses fall;
- the pitch error falls;
- the audio is recognisable;
- memory and speed are within budget.

Any visible or audible problem means fixing the code before the full run.

**Ready on the PC (2026-11-13).**

The training code (`training/v3/model/`):
- `trainer_v3.py`: one step per phase (`prosody`, `disc_warmup`, `joint`), separate from files so it can be tested.
- `train_v3.py`: the pod script. It handles data, validation clips with their predicted pitch, checkpoints, resuming and inference weights.

`test_trainer_v3.py` checks on a tiny model that each phase trains only its own parts.

Commands for the local checks:
- All v3 tests: `D:/ktts/avaz-train/.venv/Scripts/python.exe -m pytest training/v3/model/tests training/v3/data/tests -q -o addopts=""` (49 pass, 1 skipped there: the Praat test needs `D:/ktts/eval/.venv`).
- End-to-end smoke test on the PC's CPU: `D:\ktts\avaz-train\smoke\`. It used the toolkit at the pinned commit with the pod's patch, 40 real clips prepared with the rebuilt stress hook, the polish checkpoint and the pitch files. It ran 3 + 2 + 4 steps.
  - All losses are finite. The audio discriminators' loss is 4–5 in warm-up, as in the polish run.
  - The multi-scale mel loss is 1.36.
  - Validation clips and checkpoints are written, and a rerun resumes from `latest.pth`.

What the smoke test caught, now fixed and tested:
- **The pod's patched toolkit returns four items per clip (a speaker id).** `WithPitch` takes the first three.
- **735 dataset-v2 clips (4.0 h, almost all paragraphs, up to 24.4 s) are over the old 14.5 s limit.** With that limit `inflect-adapt prepare` stops with an error, it does not drop them.
  - Prepare v3 with `--max-duration-seconds 25`.
  - `data/batching.py` builds batches of similar length within a frame budget of 64 × 1360 frames (v2's 64 clips of up to 14.5 s). Short clips still come 64 at a time; the longest come about 37 at a time.

**Upload:**
- v1's FLAC (3.03 GB, `D:\ktts\avaz-train\upload`);
- dataset v2 as FLAC;
- `D:\ktts\avaz-train\pitch\pitch_v1.npz` and `pitch_v2.npz`;
- the polish checkpoint's generator and discriminators;
- the code.

**Trial run settings:** `--prosody-steps 2000 --warmup-steps 300 --joint-steps 2500`, with bf16.

**Trial run (2026-11-13, pod `kowm342jy5uyjg`, A40, EU-SE-1).**

Data: 9,905 clips, 20.5 h. That is the first 6,000 v1 clips, every dataset-v2 paragraph and a quarter of the other v2 categories. The run went 2,000 prosody + 300 warm-up + 2,500 joint steps. It cost $1.62, and the pod was deleted after the download.

Results (`D:\ktts\avaz-train\v3_trial_results\`; serve them with the monitor, `MONITOR_RUNS=<that>/runs`):

| | Start | End of prosody (2k) | End of joint (4.8k) |
| --- | --- | --- | --- |
| Pitch L1, semitones on voiced frames | 2.94 | 2.41 | ~2.3 |
| Voicing cross-entropy | 0.20 | 0.128 | 0.125 |
| KL | 1.353 | 1.325 | 1.31 |
| Multi-scale mel | — | — | 1.40 (flat) |
| Validation pitch range, predicted / true (8 clips) | — | 13.2 / 12.4 | 12.2, 10.6, 12.5, 12.0, 11.1, 12.5 / 12.4 |

On the 12 listening sentences (5 of them questions):
- buzz score 8.41 for v2 → 8.66–9.49 for the trial checkpoints (8.85 at the end), against 10.98 for the dataset;
- ring 15–17%, against v2's 15% and the dataset's 12%.

The trial is too short to judge the sound. It confirmed that every part trains and that nothing regresses.

**Found by the trial, fixed and tested:**
- **bf16 autocast left the model's outputs in bf16**, and the float32 prosody discriminators stopped at the first step. `StepV3.step` now casts the outputs to float32. A test runs each phase under bf16.
- **RunPod's A40 pod has a CPU quota of 7.65 CPUs although `nproc` says 96.** Eight data loaders, the alignment search and the listening renderer starved the GPU, giving 5.2 s per joint step. Now: `--workers 5 --threads 2`, `OMP/MKL/OPENBLAS_NUM_THREADS=2`, and the renderer runs with one thread at `nice 19`.
  - The joint step is now 2.04 s (the polish run took 2.25 s).
  - `time_parts_v3.py` times each part on a free GPU: the generator backward 0.65 s, the alignment search 0.47 s on the CPU, CQTD 0.19 s.
- **The predicted pitch range narrowed by about 6% on average during joint training** (11.7 vs 12.4 semitones; it was 13.2 after the prosody phase). For the full run:
  - `range_loss`: per utterance, the L1 between the predicted and the true spread of voiced pitch, weight 1;
  - the contour discriminator's weight is raised from 1 to 2;
  - `--final-prosody-steps 2000` refits the rhythm and pitch parts after joint training.
- **Operations.**
  - `train_v3.py` writes `<run>.pid`, so the monitor shows the run as alive.
  - On resume it drops the logged steps after the checkpoint, so the charts never jump back.
  - The monitor (`training/v3/monitor`, with pitch charts, per-clip pitch ranges and 12 listening sentences) needs the pod created with `--ports "8000/http,22/tcp"`. Adding the port later restarts the container.
  - `pkill -f` must not match the SSH command itself; use the PID files.

**Full run settings:** the whole of v1 and v2 (~54 h); `--prosody-steps 15000 --warmup-steps 1000 --joint-steps 40000` (extend towards 60000 if the measures still improve) `--final-prosody-steps 2000 --bf16 --workers 5 --threads 2`. Expect ~29–41 pod hours and ~$18–25.

## 8. Full run

1. Pitch and rhythm parts on the trained text side (~10–20k steps).
2. Decoder: discriminator warm-up, then ~60–100k steps.
3. Joint fine-tuning on all ~52 h.

**Monitoring:** avazmon, plus a listen-and-measure every 5k steps, with the pod balance watched. When the run ends, download, then delete the pod.

**Stop rule:** if the measures do not improve for 10k steps, stop and evaluate.

## 9. Test against v2

Run the whole battery from `training/v2/pod`:
- the 471 validation sentences, 3 long texts and 52 hard sentences;
- 5 seeds;
- the stress test;
- SQUIM, Praat and buzz.

| Measure | v2 | Target | Teacher |
| --- | --- | --- | --- |
| Whisper error, number-free, default variation | 3.65% | ≤ 2.5% | 2.07% |
| Yes/no questions that sound like questions (blind ear test, 24 questions × 1 render) | most sound like statements | ≥ 65% | 68% (sampled takes, 3.8) |
| Pitch range | 15.3 st | ≥ 16 st | 16.8 st |
| Buzz score | 8.8 | ≥ 9.5 | 10.6 |
| Corrected stress carried by pitch | tie (34% vs 35%) | new position wins | — |
| SQUIM PESQ | 4.18 | ≥ 4.15 | 4.20 |
| ONNX 1-thread RTF | 0.249 | ≤ 0.25 | — |

If v3 loses to v2 on most targets, keep v2. Back v3 up before starting v3.1.

---

## 10. v3.1: Turkish (teacher: VoxCPM2)

**The teacher.** `openbmb/VoxCPM2`:
- Apache-2.0; impersonation, fraud and disinformation are forbidden;
- 2B parameters, ~8 GB VRAM;
- 30 languages, Turkish included (its own Turkish WER is 1.65%), **but not Azerbaijani**, so Azerbaijani keeps the v1 teacher;
- 48 kHz output, downsampled to 24 kHz;
- two cloning modes: reference only, and reference plus transcript.

**Main risk:** cloning the Azerbaijani reference into Turkish may shift the timbre or carry the accent over.

### 10.1 Teacher pilot (~100 sentences)

- Compare both cloning modes.
- Measure Turkish Whisper error, speaker similarity against the Azerbaijani data (ECAPA), buzz, pace and the question rise.
- Listen for accent.
- **Pass:** similarity close to the Azerbaijani distribution, Turkish error ≤ 3%, and an acceptable accent. Otherwise the user decides between Turkish as a separate voice (separated by the language tag) and a separate Turkish model.

### 10.2 Turkish frontend

- eSpeak `tr` plus stress exceptions: place names (Ánkara), the unstressed `-mi` particle, the pre-stressing negative.
- Turkish number and abbreviation normalisation.
- Tests and a Turkish hard-text set.

### 10.3 Turkish sentences

- The same method as dataset v2: categories, agents, the checker. Aim for ~20–25 h.
- Reuse texts and data from the earlier Turkish distillation project if they fit and the timbre matches.

### 10.4 Recording and selection

The stage 3 rules apply, with these changes:
- Turkish Whisper;
- speaker similarity measured against the Azerbaijani data;
- 48 to 24 kHz downsampling;
- `"lang": "tr"` in the manifest.

### 10.5 Training

- Fine-tune v3 with the language tag.
- Mix about 60% Azerbaijani and 40% Turkish in every batch, so Azerbaijani is not forgotten.
- About 30–50k steps, ~1–1.5 days, ~$15–25.

### 10.6 Testing

**Azerbaijani: the full battery must not regress against v3.**
- Whisper error may rise by at most +0.3 points.
- Buzz may drop by at most 0.2.
- The question rise must hold.

**Turkish:** Whisper error, the question rise, hard texts and stress.

**Language bleed:**
- Do ə, x and q survive in Azerbaijani?
- Is Turkish ğ lengthened, and is the Azerbaijani accent absent?

**Speed:** ONNX real-time factor.

**Decision:** keep the bilingual model if Azerbaijani holds; otherwise build a separate Turkish model and keep v3 for Azerbaijani.

---

## 11. v3.2: English (after Turkish)

**Measured starting point, 2026-11-12.** 16 English sentences read by Avaz's English voice (the untouched Inflect-Micro-v2 base model in `model-en/`) and by Paradee 1.2 (7.88M parameters, distilled from Kokoro):

| | Avaz EN (0.3 / 0.667) | Paradee 1.2 (fp32 / int8) |
| --- | --- | --- |
| Real Whisper misreadings, 16 sentences | 1 word | 1 word |
| Buzz score, 2–4 kHz (higher is better) | 6.1 / 6.4 | 9.7 / 9.0 |
| Fixed ring | 38% (about 9.6 kHz) | 16% |
| SQUIM PESQ | 4.28 / 4.30 | 2.65 / 2.71 |
| CPU real-time factor, 1 thread | 0.59 / 0.52 (PyTorch) | 0.031 / 0.036 |

- **Intelligibility is equal.** The English voice's sound has v1's weakness: a fixed ring and weak harmonics.
- **SQUIM's lower score for Paradee follows from its brighter top end** (9 dB more energy above 8 kHz), not from noise: both systems have the same floor in pauses, -76 dB.
- Clips, scores and a listening page are in `D:\ktts\avaz-train\english_compare\`.

**Approach.** Same as v3.1: language tag, a targeted sentence set, careful take selection, and fine-tuning with Azerbaijani (and Turkish) data mixed in. The v2 decoder work (stronger discriminators, harmonic decoder) applies here unchanged. On Azerbaijani it removed the 6 kHz ring, and the same is expected for English.

**Teacher.** Two candidates, to be piloted side by side before deciding:
- **VoxCPM2** clones the Avaz voice into English, so all three languages share one voice.
- **Kokoro** (Apache-2.0, the teacher behind Paradee) gives a cleaner but different voice.
