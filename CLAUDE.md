# CLAUDE.md

Offline Azerbaijani text-to-speech. A 9.36M-parameter VITS model plus the
Azerbaijani text layer it needs. Runs on CPU at 2-4x real time, no network.

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
```

Python 3.11+. The 37 MB `model/` directory is in the repository -- there is
nothing to download for normal work.

## Verify before saying anything is done

```bash
python -m pytest                     # ~3 seconds, never loads the model
python -m compileall -q say.py aztts tools training
cd model && sha256sum -c checksums.sha256    # 24 files, all must say OK
python say.py --show-text "II Dünya, 25% artım"   # normalisation, no synthesis
python say.py "Salam."               # end-to-end, writes out/01.wav
```

`--show-text` is the fastest way to check a text-layer change: it prints what
would reach the model and synthesises nothing.

## Hard rules

**Never edit `model/` or `training/base-model/`.** They are vendored upstream
exports, not our code. `model/checksums.sha256` verifies 24 of those files and
any edit breaks it. If the model needs different behaviour, wrap it in `aztts/`
instead.

**The `ə`, `ɣ`, `ʃ` characters inside `model/` are IPA phoneme symbols, not
Azerbaijani text.** A grep for Azerbaijani letters matches them. Do not
"translate" anything there.

**All prose is English** -- comments, docstrings, `print()` output, argparse
help, Markdown. Azerbaijani appears only as *language data*:

- TTS example sentences (`say.py` `DEMO`, test inputs, doctests)
- `LETTER_NAMES`, `MONTHS`, `ABBREVIATIONS`, `UNITS`, `CURRENCIES`, `ACRONYMS`,
  `ORDINAL_SUFFIXES` in `az_text.py`
- `_CONJUNCTIONS` in `az_chunk.py`
- `CLITICS`, `WEAK_HEADS`, `_WIDE_WEAK`, `_PARTICIPLE_WORDS` in `az_prosody.py`
- phoneme strings anywhere

Translating any of those changes behaviour. `README.az.md` is the one deliberate
exception -- it is the Azerbaijani mirror of `README.md` and must be kept in
sync when README content changes.

**Never call `.lower()` or `.upper()` on Azerbaijani text.** Azerbaijani has
dotted and dotless i: `I` lowercases to `ı`, `İ` lowercases to `i`. Use
`az_lower()` and `az_capitalise()` from `aztts/az_text.py`. Python's built-ins
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

## Conventions

- **Pure functions over mutation.** Every text helper takes a string and returns
  a new one. Nothing is mutated in place.
- **Every CLI entry point calls `use_utf8()` first** (`aztts/console.py`).
  Without it, printing `ə` on a Windows console raises `UnicodeEncodeError`.
  Currently: `say.py`, `tools/seed_sweep.py`, `tools/audio_postprocess.py`,
  `training/scripts/compare_checkpoints.py`.
- **Step order in `normalize_az()` is load-bearing.** Abbreviations run first so
  their full stops are not mistaken for sentence ends; acronyms run last so
  Roman numerals win the `II`-style ambiguity. Adding a step means choosing its
  position deliberately, with a test.
- Type hints everywhere, `from __future__ import annotations` at the top.
- Files stay small and single-purpose. The largest is `az_text.py` at ~390
  lines.
- No new runtime dependencies without a reason stated in the PR.

## Where to change what

| You want to | Go to |
| --- | --- |
| Fix how a number, date, unit or acronym is read | `aztts/az_text.py` |
| Change where sentences are cut | `aztts/az_chunk.py` |
| Change stress placement (`--prosody`) | `aztts/az_prosody.py` |
| Change synthesis, chunk pauses, model loading | `aztts/engine.py` |
| Add or change a CLI flag | `say.py` |
| Audio cleanup, seed selection | `tools/` |
| Anything about how the model was trained | `training/` (archive; the pod is gone) |
| Publish to GitHub / HF / Kaggle | `packaging/PUBLISHING.md` |

`aztts/` is all of our library code, about 900 lines. `say.py` is the CLI.

## Generated and ignored

- `out/` is where synthesis writes. Never commit its contents; `out/.gitkeep` is
  the only tracked file in it.
- `samples/` is tracked and goes through Git LFS. Regenerating it
  (`python say.py --out samples`) rewrites five binaries -- only do that when the
  model actually changed.
- `training/base-model/` is not tracked. It is a 38 MB upstream snapshot; fetch
  it with `python training/scripts/download_model.py` when you need it.

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
