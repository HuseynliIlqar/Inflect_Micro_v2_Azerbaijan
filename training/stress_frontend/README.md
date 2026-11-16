# Stress frontend

eSpeak NG puts a stress mark on nearly every Azerbaijani word, and often on the wrong syllable. For example, it stresses the syllable before a final -ə: ildə, ölkə, şəkildə. This frontend phonemises with eSpeak NG and then moves the stress marks with rules checked by a native speaker.

The v2 checkpoint was trained on its output. The package ships it as `model/frontend_hook.py`, and `AzTTS` runs it automatically.

This folder is the **source** of that file. `model/` is vendored, so never edit the copy there. Change the rules here, rebuild, and use the new file at the next `prepare` and export. The phonemes a model is trained on and the phonemes it is given at runtime must come from the same frontend.

## Files

| File | What it holds |
| --- | --- |
| `az_stress.py` | The rules: negative verbs, tense and person endings, unstressed particles, place names, postpositions, loans, the final -ə exceptions, and bare -ma/-mə orders (see below). |
| `frontend_part.py` | The custom-frontend class for the toolkit (`create_frontend`). |
| `make_hook.py` | Joins the two files with the base symbol list and the compressed lexicon into `dist/az_stress_frontend.py`. `dist/` is git-ignored. |
| `build_lexicon.py` | Rebuilds `lexicon/{verbs,lemmas,forms}.txt` from `spelling_dictionary.txt` and the corpus counts in `vocab.tsv`. |
| `lexicon/` | Word lists from `LocalDoc/azerbaijani_spelling_dictionary_2021` (MIT; see `THIRD_PARTY_NOTICES.md`). |
| `native_review/` | Native-speaker judgements from two review rounds. Each is encoded as a test. |
| `tests/` | 344 tests; no eSpeak needed. |

## Build and test

```bash
cd training/stress_frontend
python -m pytest tests -o addopts=""
python make_hook.py ../../model/symbols.json     # -> dist/az_stress_frontend.py
```

With Python 3.12 (standard zlib), the rebuilt file is byte-identical to the shipped `model/frontend_hook.py`, as of the v2 rules. Python 3.14 builds with zlib-ng: the lexicon inside is the same, but the compressed bytes differ.

## Changes since the v2 export

**A bare -ma/-mə form before a comma.** The v2 rule treated a bare -ma/-mə verb form as an order only at the end of a sentence ("Sabah bizə GƏLmə."). Before a comma it was always read as a verbal noun. Dataset v2 has about 260 orders in that position ("Tələsmə, qızım, ...", "Bu kitabı alma, o birini al"), so they would have been labelled with the wrong stress.

`_clause_imperative` in `az_stress.py` now decides each case in two steps.

**A list keeps the verbal noun.** These cases stay nouns:
- a comma or "və" on either side, or "və" within the next three words ("ştamplama, oyma, tökmə"; "etimad və anlaşma,"; "proqramlaşdırma, robot texnikası və ...");
- a single closing noun after the comma ("Qidalanma, idman.").

**Otherwise, positive evidence makes an order.** Any one of these is enough:
- a form of address set off by a comma ("Tələsmə, qızım", "Ata, buraxma", "Ay qızım, bu qədər oxuma");
- the start of a sentence;
- "sən", "siz" or "heç" in the clause;
- a case-marked argument right before the verb ("kitabı alma", "buzun üstünə çıxma");
- a sentence ending in "!";
- another order later in the sentence.

With no evidence, the old reading stands. "xırda oğurluq güllələnmə, ..." stays a noun, as the native review requires.

| Measured on | Effect |
| --- | --- |
| v1's 23,559 sentences | 8 words change; all 8 are real orders. Nothing else changes, because only bare -ma/-mə forms before a comma can be affected. |
| Dataset v2 | 210 of about 260 orders are now marked. The other ~50 keep the old reading. Only 3 matches fall in the informative categories: 2 real orders and 1 noisy mined line. |
