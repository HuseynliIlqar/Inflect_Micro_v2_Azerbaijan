# CLAUDE.md

Offline Azerbaijani text-to-speech. A 9.36M-parameter VITS model plus the
Azerbaijani text layer it needs. Runs on CPU at 2-4x real time, no network.
Ships with a browser interface and, as a guest voice, the English base model it
was adapted from.

Read `README.md` for what the project does. This file is about how to work on it
without breaking it.

This is the single source of truth for agent instructions. `AGENTS.md` and
`.cursor/rules/` point here on purpose -- keep `AGENTS.md` a pointer, never a
copy, or the two will drift apart.

## Setup

```bash
git lfs install                      # required: model.pth is an LFS object
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
./.venv/bin/pip install -r requirements.txt                   # Linux / macOS
pip install pytest                   # tests are not in requirements.txt
pip install -r requirements-app.txt  # only if you want `python app.py`
```

Python 3.11+. The 37 MB `model/` directory is in the repository -- there is
nothing to download for normal work.

## Verify before saying anything is done

```bash
python -m pytest                     # ~3 seconds, never loads the model
for f in web/tests/*.mjs; do node "$f" || break; done   # every web suite; a failure stops
python -m compileall -q say.py app.py aztts webui tools training
cd model && sha256sum -c checksums.sha256       # 24 files, all must say OK
cd model-en && sha256sum -c checksums.sha256    # 6 files, all must say OK
python say.py --show-text "II Dünya, 25% artım"   # prints the normalised text
python say.py "Salam."               # end-to-end, writes out/01.wav
```

`--show-text` prints the chunks that reach the model *and* still synthesises --
the flag adds the printout, it does not replace the synthesis. To check a text
change without touching the model, call the text layer directly:

```bash
python -c "from aztts.console import use_utf8; use_utf8(); from aztts import normalize_az; print(normalize_az('II Dünya, 25%'))"
```

`use_utf8()` first, or printing `ə` on a Windows console raises
`UnicodeEncodeError` -- the same reason every CLI entry point calls it.

## Hard rules

**Never edit `model/`, `model-en/` or `training/base-model/`.** They are
vendored upstream exports, not our code. `model-en/` holds the English weights
this repository ships (`model-en/checksums.sha256` verifies its six vendored
files; `README.md` there is ours). `aztts/en_voice.py` reads them but never
imports the upstream package's Python: both packages ship modules named `inference`,
`models` and `text`, so importing the second one in a live process silently
returns the first. The English weights are loaded into `model/`'s runtime
instead, and the English text is phonemised here and passed as `phonemes=`. `model/checksums.sha256` verifies 24 of those files and
any edit breaks it. If the model needs different behaviour, wrap it in `aztts/`
instead.

**The `ə`, `ɣ`, `ʃ` characters inside `model/` are IPA phoneme symbols, not
Azerbaijani text.** A grep for Azerbaijani letters matches them. Do not
"translate" anything there.

**All prose is English** -- comments, docstrings, `print()` output, argparse
help, Markdown. Azerbaijani appears only as *language data*:

- TTS example sentences (`say.py` `DEMO`, test inputs, doctests)
- `LETTER_NAMES`, `MONTHS`, `ABBREVIATIONS`, `UNITS`, `CURRENCIES`,
  `CURRENCY_SUBUNITS`, `SYMBOLS`, `ACRONYMS`, `ROMAN_NOUNS`, `ORDINAL_SUFFIXES` in
  `az_tables.py` (and `web/js/text/az-tables.js`)
- `_CONJUNCTIONS` in `az_chunk.py`
- `CLITICS`, `WEAK_HEADS`, `_WIDE_WEAK`, `_PARTICIPLE_WORDS` in `az_prosody.py`
- `PROFANE_STEMS`, `CLEAN_PREFIXES`, `PROFANE_WORDS` in `az_profanity.py`
  (and their copies in `web/js/text/az-censor.js`)
- the `_AZ` table in `webui/i18n.py` -- the interface's own labels; the page has
  to speak Azerbaijani, and this is the only module where it does
- phoneme strings anywhere

Translating any of those changes behaviour. `README.az.md` is the one deliberate
exception -- it is the Azerbaijani mirror of `README.md` and must be kept in
sync when README content changes.

**Never call `.lower()` or `.upper()` on Azerbaijani text.** Azerbaijani has
dotted and dotless i: `I` lowercases to `ı`, `İ` lowercases to `i`. Use
`az_lower()` and `az_capitalise()` from `aztts/az_words.py`. Python's built-ins
silently produce the wrong letter.

**`fast_monotonic_align.py` must stay bit-identical** to the reference
implementation frozen in `tests/reference_monotonic_align.py`. That equivalence
is the whole reason the file exists (it cut training from 20 days to 3.2 days).
`tests/test_fast_monotonic_align.py` proves it across random shapes, edge cases,
mask variants and determinism. Never change that file without those tests
passing.

**Tests must not load the model.** The suite runs in about three seconds and
that is deliberate -- it keeps CI free and fast. Test the text layers, not
synthesis.

**Do not deepen the GPL surface.** `phonemizer` and eSpeak NG are
GPL-3.0-or-later and are loaded in-process by `model/deployment_frontend.py`.
That is documented and bounded. Do not add more GPL dependencies, and do not
relicense our Apache-2.0 code. See `THIRD_PARTY_NOTICES.md`.

**Both READMEs answer the commercial-use question, and must keep saying the
same thing.** Running it is unrestricted; redistributing a combined work brings
GPL-3.0 obligations for `phonemizer` and eSpeak NG; `accepts_prephonemized_input`
is the way out. The English voice is synthetic and Apache-2.0, but the release is
open-weight, not open-data. The Azerbaijani training data is used with the
dataset author's permission, on condition of attribution. Do not compress any of
that into "free for commercial use".

## Conventions

- **Pure functions over mutation.** Every text helper takes a string and returns
  a new one. Nothing is mutated in place.
- **Every CLI entry point calls `use_utf8()` first** (`aztts/console.py`).
  Without it, printing `ə` on a Windows console raises `UnicodeEncodeError`.
  Currently: `say.py`, `app.py`, `tools/seed_sweep.py`,
  `tools/audio_postprocess.py`, `tools/fetch_web_runtime.py`,
  `tools/quantize_decoder.py`, `training/scripts/compare_checkpoints.py`.
- **Step order in `normalize_az()` is load-bearing.** Abbreviations run first so
  their full stops are not mistaken for sentence ends; acronyms run last so
  Roman numerals win the `II`-style ambiguity. Adding a step means choosing its
  position deliberately, with a test.
- **Numbers are read the way a person reads them aloud.** A date's day is a
  cardinal and its year an ordinal plus `il` (`01/09/1939` -> `bir sentyabr ...
  otuz doqquzuncu il`, never `birinci sentyabr`); a decimal names the
  denominator first (`onda beş`, not `beş onda`); two decimals on money are
  coins (`19,99 AZN` -> `... manat doxsan doqquz qəpik`). A suffix written
  against a symbol (`AZN-dən`, `%-ə`) is re-harmonised onto the spoken word by
  `harmonise()`. `tests/test_az_text_edges.py` holds these cases.
- Type hints everywhere, `from __future__ import annotations` at the top.
- Files stay small and single-purpose. `normalize_az()` lives in `az_text.py`,
  but its rules are split: dates, times and phones in `az_dates.py`,
  money/percent/units/fractions/ranges in `az_amounts.py`, the word tables in
  `az_tables.py`, shared primitives (casing, harmony, number words) in
  `az_words.py`. `web/js/text/` mirrors the same split. See "Long files and
  messy folders" below.
- No new runtime dependencies without a reason stated in the PR.
- **Censoring is on unless someone with a clone turns it off.** `censor_az`
  runs before normalisation, survives `--raw` and covers the English voice
  too -- it only removes words, so it is the one Azerbaijani layer `EnVoice`
  may use. The static playground's `Engine` censors by default and `speak()`
  reads no option that lifts it; the Gradio page has no widget for it and only
  `python app.py --allow-profanity` does. Do not add a switch to either page:
  both are hosted publicly. A stem that collides with a clean word needs a
  `CLEAN_PREFIXES` entry and a test in `tests/test_az_profanity.py`; then
  regenerate `web/tests/golden/censor.json` (see `packaging/PUBLISHING.md`).
- **The interface's dependency stays optional.** Gradio is declared under
  `[project.optional-dependencies] app` and in `requirements-app.txt`, never in
  `requirements.txt`: someone importing `aztts` as a library should not pay for
  a web stack. `app.py` is the only module allowed to import it.
- **The English voice is a guest, not a second product.** `EnVoice` exists so a
  visitor can hear where this model started. None of the Azerbaijani text
  layers apply to it: no `normalize_az`, no `restress`. Keep it that way --
  Azerbaijani number words in an English sentence is the failure mode. The one
  exception is `censor_az`, which writes nothing, only bleeps.
- **`web/` is a port, and a port drifts.** The JavaScript in `web/js/text/` mirrors
  `az_text.py` (with `az_dates.py`, `az_amounts.py`, `az_tables.py`,
  `az_words.py`), `az_chunk.py` and `num2words`. When any of those change,
  regenerate the golden files and run every web suite
  (`for f in web/tests/*.mjs; do node "$f" || break; done`); the snippet that
  writes them is in `packaging/PUBLISHING.md`. A change to the text layer that
  does not reach `web/` makes the page and the CLI say different things.
- **Interface logic belongs in `webui/`, not in `app.py`.** That is what keeps
  `tests/test_webui_runner.py` able to run against a stand-in engine instead of
  loading the model.

## Long files and messy folders

Do not write long scripts, and do not leave a folder flat and mixed. Both have
already cost this project: `web/` once had an 800-line `app.js` and two
stylesheets of 640-800 lines loose at its top level, next to the page.

- **A file does one job and stays under ~400 lines** -- Python, JavaScript,
  CSS and HTML alike; 200-300 is the norm here. When a change would take a
  file past that, split it first, in its own change, then make the change.
  Golden files, vendored code and generated label tables are exempt.
- **Split by responsibility, not by line count.** A module named for what it
  does (`az_dates.py`, `worker-client.js`, `dialog.css`), never `utils2` or
  `part-b`. Keep public names importable from where they were: re-export from
  the old module rather than breaking callers.
- **An entry point is wiring only.** `say.py`, `app.py`, `web/js/app.js` and
  `web/js/worker.js` import and connect; logic goes in a module they import.
- **Group a folder by role once it holds more than a handful of files.** No
  loose scripts at a folder's top level beside its entry point. `web/` is the
  model: `index.html`, `README.md` and `package.json` at the top, then `css/`,
  `js/{page,engine,text,ui}/` and `tests/`.
- **Stylesheets split in cascade order.** A CSS split cuts the file into
  consecutive pieces and `web/index.html` links them in the same order, so the
  cascade cannot change. A new stylesheet goes where its rules belong in that
  order.
- **A split is a pure move.** Behaviour must be identical: run the Python
  suite and every `web/tests/*.mjs` before and after, and for `web/` open the
  page and speak a sentence. Update every path that named the old file --
  imports, `index.html`, `web/README.md`, `packaging/PUBLISHING.md`, this file,
  `.cursor/rules/`.

## Where to change what

| You want to | Go to |
| --- | --- |
| Fix how a number, date, unit or acronym is read | `aztts/az_text.py` (pipeline), `az_dates.py`, `az_amounts.py`, `az_tables.py` |
| Change where sentences are cut | `aztts/az_chunk.py` |
| Change stress placement (`--prosody`) | `aztts/az_prosody.py` |
| Change which words are bleeped | `aztts/az_profanity.py`, then `web/js/text/az-censor.js` |
| Change synthesis, chunk pauses, model loading | `aztts/engine.py` |
| Add or change a CLI flag | `say.py` |
| Change the browser interface, its labels or parameters | `app.py`, `webui/` |
| Change the browser playground (the static Space) | `web/` (its layout is in `web/README.md`) |
| Change how the English base model is spoken | `aztts/en_voice.py` |
| Audio cleanup, seed selection | `tools/` |
| Rebuild the int8 decoder the playground runs on the CPU | `tools/quantize_decoder.py` |
| Change the self-hosted onnxruntime-web version | `tools/fetch_web_runtime.py` (`VERSION` + integrity) |
| Anything about how the model was trained | `training/` (archive; the pod is gone) |
| Publish to GitHub / HF / Kaggle | `packaging/PUBLISHING.md` |

`aztts/` is the library (about 1,700 lines with `en_voice.py`), `say.py` is the
CLI, and `app.py` plus `webui/` are the browser interface.

## Generated and ignored

- `out/` is where synthesis writes. Never commit its contents; `out/.gitkeep` is
  the only tracked file in it.
- `samples/` is tracked and goes through Git LFS: five Azerbaijani WAVs and
  three English ones in `samples/en/`. Regenerating them
  (`python say.py --out samples`, `python say.py --voice en --out samples/en`)
  rewrites eight binaries -- only do that when a model actually changed. Both
  READMEs list the sentences, so a new sample means editing both.
- `model-en/` **is** tracked, through Git LFS: 37 MB of English weights so a
  clone speaks both languages. It is the reason the repository is ~75 MB.
- `web/onnx/` (the graphs, 38 MB plus the int8 decoder) and `web/vendor/`
  (onnxruntime-web, 27 MB, fetched by `python tools/fetch_web_runtime.py`) are
  git-ignored; see `web/README.md`.
- `training/base-model/` is not tracked. It is the complete 38 MB upstream
  package, only needed for training work; fetch it with
  `python training/scripts/download_model.py`.

## Known limitations, so you do not chase them

These are properties of a 9.36M model, not bugs:

- One voice, no emotion control, no cloning.
- The vocoder leaves a metallic resonance on some sentences. Mitigate with
  `tools/audio_postprocess.py`, do not try to fix it in the text layer.
- eSpeak stresses 92.2% of words, which is why the reading is flat. `--prosody`
  is the experimental attempt at this and does not improve every text.
- Intonation flattens over long sentences. That is why chunking exists at ~15
  words; the number is empirical, not arbitrary.
- The training audio is most likely synthetic, so the quality ceiling is the
  system that produced it. See `docs/MODEL.md`.
