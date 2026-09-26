# AGENTS.md

Offline Azerbaijani text-to-speech. Python 3.11+, CPU-only, no network at
runtime.

**Read [CLAUDE.md](CLAUDE.md) before changing anything.** It is the single source
of truth for this project: setup, the commands that verify a change, and the
traps that are not obvious from reading the source. This file exists only so
that agents which do not look for `CLAUDE.md` still find their way there.

The three rules worth repeating here, because breaking them is silent:

1. **Never edit `model/`, `model-en/` or `training/base-model/`.** They are
   vendored upstream exports; `model/checksums.sha256` verifies 24 files and
   `model-en/checksums.sha256` verifies 6, and any edit breaks them. The `ə` and
   `ʃ` characters in there are IPA phoneme symbols, not Azerbaijani text.
2. **All prose is English.** Azerbaijani appears only as language data (example
   sentences, letter names, abbreviation tables, phoneme strings, and the
   interface labels in `webui/i18n.py`). Translating that data changes
   behaviour. `README.az.md` is the one deliberate exception.
3. **Never call `.lower()` / `.upper()` on Azerbaijani text.** `I` → `ı` and
   `İ` → `i`; Python's built-ins give the wrong letter. Use `az_lower()` and
   `az_capitalise()` from `aztts/az_text.py`.

Verify before saying a change is done:

```bash
python -m pytest                                  # ~3s, never loads the model
python say.py --show-text "II Dünya, 25% artım"   # prints the chunks, still speaks
python say.py "Salam."                            # end to end -> out/01.wav
python say.py --voice en "Hello."                 # the English base model
```

Cursor users: the same rules are in `.cursor/rules/`, picked up automatically.
