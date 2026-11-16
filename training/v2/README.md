# Training round 2: distilled data, stress frontend, polished decoder

This folder records how the second Azerbaijani checkpoint now in `model/` was made, and holds the tools used to test it. Nothing here is needed to run the model.

Like `training/` v1, this is an archive. The RunPod pod is gone. Large artefacts stay outside the repository.

## Pipeline

1. **Toolkit.** Inflect at commit `6dad538` (`pod/BASE_COMMIT.txt`), with these additions:
   - `pod/multispeaker.patch`: vectorised monotonic alignment and VRAM metrics;
   - `pod/add_walltime.py`: wall-clock time in the metrics rows.

   `pod/setup.sh` builds the environment.
2. **Data.** `pod/prepare.sh` does four things:
   - decodes the uploaded FLAC;
   - drops 3 duplicate transcripts (`dedup_manifest.py`);
   - prepares the set with the stress frontend (1.0–14.5 s, 2% validation);
   - checks that the pod's phonemes equal the ones made on Windows (`check_phonemes.py`).

   Result: 23,088 training clips and 471 validation clips.
3. **Main run** (`pod/train.sh`, `pod/az-continue.json`). Warm start from v1 (410 tensors).
   - Steps 0–5k: posterior warm-up.
   - Steps 5k–20k: linguistic adaptation.
   - From 20k: the decoder unfreezes.

   The decoder stage raised a fixed 6 kHz (fs/4) ring from 20% to 55% of frames, so the run was stopped at 41.7k.
4. **Polish run** (`polish/train_polish.py`, `pod/polish.sh`). It restarts from the 20k checkpoint.
   - Steps 1–2.5k: discriminator warm-up with the generator frozen.
   - Then decoder training with the stronger losses and discriminators listed above.
   - Safeguards: a gradient clip of 500 per discriminator family, and the generator skips any batch whose discriminator loss exceeds 30.

   Stopped at 30.45k, after measurements had stopped improving past 25k. **Step 30k was exported** (PyTorch + ONNX; ONNX parity error 7e-9).
5. **Monitor.** `avazmon/` (`pod/monitor.sh`) is a small FastAPI page, usable on desktop and on a phone. It shows:
   - losses and the discriminators' scores;
   - every checkpoint reading eight fixed sentences next to v1, with buzz and ring measured for each (`pod/listen.py`);
   - validation clips next to the v1 "before" and the target recording.

   Its labels are in Azerbaijani because it is the operator's own interface.

## Results

All figures come from the 471 validation sentences, which were never trained on. "Teacher" is the original recording.

| Measure | Teacher | v1 | v2, variation 0.667 | v2, variation 0.3 |
| --- | ---: | ---: | ---: | ---: |
| Whisper word error, 416 number-free sentences | 2.07% | 4.84% | 3.65% | ~2.65% (0.333) |
| Word error, 1–5 word sentences | 0.8% | 9.2% | 5.7% | 3.2% |
| SQUIM PESQ | 4.20 | 3.47 | 4.18 | — |
| Harmonic clarity 2–4 kHz (buzz score, higher is better) | 10.6 | 6.4 | 8.8 | 8.9 |
| Fixed ring, share of frames | 13% | 20% (6 kHz) | 15% | 15% |
| Pitch range (semitones) | 16.8 | 13.0 | 15.3 | 15.0 |
| Yes/no question ending (semitones) | +2.9 | −3.0 | −1.2 | −1.8 |
| Sentences that fail with some of 5 seeds | — | — | 40% | 19% |

What the figures show:

- **Sound is at teacher level** by SQUIM, by the band spectrum (within 1 dB) and by jitter, shimmer and HNR. Only harmonic clarity in 2–4 kHz lags.
- **Most pronunciation errors come from sampling noise.** Lowering `variation` cuts them by about 30% at almost no cost to pitch range. That is why the default is now 0.3.
- **Remaining weaknesses.**
  - Consonant place confusions: m→n, b→d, q→k. They concentrate on the first and last word of a sentence.
  - Yes/no questions end falling where the teacher rises. The training set has only 677 such clips, and VITS has no explicit pitch model.
  - Stress corrections are realised through length, but not yet through pitch.
- CPU real-time factor on the pod's CPU: ONNX 0.25 on 1 thread and 0.08 on 4 threads. Model size is 9.36M parameters.

The JSON behind each number is in `results/`:

| File | What it holds |
| --- | --- |
| `asr_eval/asr_summary.json`, `asr_analysis.json` | Whisper error rates, the per-letter breakdown and long texts |
| `voice_summary.json`, `voice_summary_variants.json` | pitch, tempo, jitter, shimmer, HNR, and the variation sweep |
| `squim_summary.json` | SQUIM scores |
| `stress_check.json` | stress realisation |
| `seed_stability.json` | the 5-seed stability test |
| `edge_experiment.json` | the variation and lead-pause experiments |
| `gap_analysis.json` | error position in the sentence, edge silence, training-set make-up |
| `rtf.json` | CPU speed |
| `hardtext/hardtext.json` | 52 hard sentences through the raw package |
| `variation_metrics.json` | buzz and ring at each variation value |

The hard-sentence test fed text **straight to the package**, so its number and abbreviation failures do not apply to `say.py`. `normalize_az` already reads `2024-cü`, `XX`, `km`, `Dr.`, `və s.` and `-5` correctly. Two gaps remain in the project's text path:
- words in foreign Latin spelling, such as Microsoft and Google;
- the negative imperative "alma", which needs initial stress.

## The test battery (`pod/`)

The scripts run on a GPU pod against `/workspace` paths. Together they are the regression suite for the next model.

| Script | What it does |
| --- | --- |
| `render_eval.py` + `eval_asr.py` | read the validation set with each version, then transcribe it with an Azerbaijani Whisper (`hasan0v/whisper-turbo-az-ct2`, faster-whisper 1.1.0, PyAV < 14) |
| `asr_analysis.py`, `gap_analysis.py` | break the errors down by letter, word, position, sentence type and length |
| `analyze_voice.py` | Praat measures: pitch range and ending, tempo, pauses, jitter, shimmer, HNR, band spectrum |
| `squim_eval.py` | reference-free quality from torchaudio SQUIM |
| `stress_check.py` | whether each stressed vowel is the longest, loudest and highest in its word, using the model's own alignment |
| `seed_stability.py` | 5 seeds per sentence, to separate random errors from systematic ones |
| `edge_experiment.py` | inference-time variants: variation values and a lead pause |
| `hardtext.py` + `hardtext.json` | 52 hard sentences through a whole package |
| `longtest.py` | three long texts |
| `rtf.py` | CPU speed of PyTorch and ONNX |
| `run_deep.sh` | runs the battery detached, so a dropped SSH session does not stop it |

`analysis/` holds the PC-side pieces:
- local breakdowns;
- the report builder;
- the variation listening test.

`tests/` holds 30 tests for the monitor and the discriminators. They need torch, nnAudio and FastAPI, so the root suite does not collect them. Run them from this folder:

```bash
python -m pytest tests -o addopts=""
```

To upload the kit, put the stress frontend at `pod/az_stress_frontend.py`. For a re-run of v2 that is a copy of `model/frontend_hook.py`; for new data it is the hook rebuilt from `training/stress_frontend/`. `pod/reference_phonemes.json` (2.5 MB) is not kept. Rebuild it by phonemising the manifest with the hook on Windows.

## Dataset v2 texts (`dataset_v2/`)

These are 14,853 sentences and paragraphs, about 18 h once read. They were written to target the weaknesses above:
- yes/no questions, including questions marked only by intonation;
- other questions and exclamations;
- -m/-n contrasts and consonants at sentence edges;
- words with non-final stress;
- paragraphs and long sentences;
- new vocabulary and numbers written as words.

