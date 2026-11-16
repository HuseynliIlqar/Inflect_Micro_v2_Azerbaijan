# Dataset v2: sentences to record

`sentences_v2.jsonl` holds one object per line: `{"id": "A1_00001", "cat": "A1", "text": "..."}`. There are 14,853 lines, about 18 h of speech, estimated from the teacher's pace of 5.77 syllables per second plus 0.5 s of edge silence. The per-category counts are in `summary.json`.

Every line passed these checks:
- `check_sentences.py`: alphabet, digits, length, and duplicates against the v1 dataset and the 471 test sentences;
- `phoneme_check.py`: phonemised with the stress frontend; every symbol is in the model's 178-symbol inventory.

| Code | Content | Lines | Targets |
| --- | --- | ---: | --- |
| A1, A2 | Short yes/no questions, about 25% marked only by intonation | 1,998 | question ending falls instead of rising |
| A3 | Long yes/no questions | 800 | same |
| B | Question-word, alternative ("yoxsa") and rhetorical questions | 1,500 | intonation variety |
| C | Exclamations | 1,199 | narrow pitch range; exclamations are 1.4% of v1 data |
| D1, D2 | -m/-n contrasts in verbs, possessives and names | 2,498 | m→n confusion, which changes the person |
| E | Consonants at sentence edges, short sentences, consonant clusters | 1,500 | first and last word errors; b→d, q→k |
| F | Words with non-final stress | 1,197 | stress corrections half learned |
| G1, G2 | Narrative and informative paragraphs (25–60 words) | 800 | no paragraph prosody |
| L | Long sentences (16–35 words) | 1,200 | long-sentence rhythm |
| J | Specialised vocabulary; 33% of its words are new to the model | 1,500 | unseen words |
| I | Numbers written as words | 300 | number reading |
| H | Mined 6–10 word sentences | 361 | mid-length sentences |

`BRIEF.md` is the instruction set the sentences were written from. The tools are:

| Tool | What it does |
| --- | --- |
| `check_sentences.py` | format, length and duplicate checks |
| `phoneme_check.py` | symbol-inventory check through the stress frontend |
| `merge.py` | merges the per-category files and assigns ids |
| `mine.py`, `mine_count.py` | mine the local text sources |
| `train_vocab.txt` | word counts from the v1 training set |

The scripts use absolute paths from the training PC.

## Recording rules

These are in the v3 plan, stage 3.

**Match v1 exactly**: same teacher model, same reference clip, same generation settings, 24 kHz PCM_16, no loudness normalisation.

**Run a pilot first.** With both temperatures at 0 the teacher may be deterministic, which would make several takes per sentence identical. The pilot checks this before anything else.

**Record 2–3 takes per sentence and keep the best one that passes every gate:**
- Whisper character error ≤ 2%; ≤ 5% for J and ≤ 3% for paragraphs;
- pace 2.6–8.5 syllables per second;
- speaker similarity to the reference;
- buzz and ring within the v1 distribution;
- pitch range above v1's 10th percentile;
- **in A1–A3, the sentence must end ≥ +1 semitone higher.**

**Paragraphs and long sentences reach 25 s.** Raise `--max-duration-seconds` from 14.5 to 25 when preparing the set.
