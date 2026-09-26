# `ughurabbasov/azerbaijani-tts-dataset` — investigation and preparation

Hugging Face: [ughurabbasov/azerbaijani-tts-dataset](https://huggingface.co/datasets/ughurabbasov/azerbaijani-tts-dataset)

The dataset card is **empty** — there is no information about provenance, licence or
speaker. Everything below is the result of downloading 3 shards (0, 5, 9) and
measuring them myself.

## Format — a perfect match for the target

| | |
|---|---|
| Structure | 10 parquet shards, `audio: struct<bytes, path>` + `text: string` |
| Row count | 10,088 |
| Size | 4.58 GB |
| Audio | **WAV / PCM_16 / 24000 Hz / mono** |
| Duration | median 8.6–13.6s (varies by shard), range 5.3–14.4s |

24 kHz mono PCM_16 — **identical** to Inflect's target format. No resampling or
re-encoding is needed; the converter copies the bytes as they are (`re-encoded: 0`).

## Single speaker — confirmed

I measured F0 separately in three shards:

| Shard | F0 median | F0 std |
|---|---:|---:|
| 0 | 200.8 Hz | 15.2 |
| 5 | 195.1 Hz | 11.2 |
| 9 | 200.8 Hz | 13.1 |

**One voice** across the whole dataset (female range). This means **the multi-speaker
patch is not needed for this dataset** — `n_speakers` stays at 0 and the official
single-voice path is used. The patch will be needed in the future when other voices
are added.

## Text/audio match — good

The most destructive problem in TTS training is a transcript that does not match the
audio. I measured the relationship between phoneme count and clip duration:

| Shard | correlation | phonemes/s | CV |
|---|---:|---:|---:|
| 0 | **0.911** | 14.39 | 6.7% |
| 9 | **0.922** | 14.76 | 10.1% |

The speaking rate is very stable. This check is now part of `inflect-preflight` and
raises a **blocker** if the correlation drops below 0.80.

### Digits are not a problem

In the later shards 19–27% of the transcripts contain digits. eSpeak `az` expands
them correctly:

```
1798-ci  →  mˈin jeddˈijˈyz doxsˌansæçcˈizdʒi   ("min yeddi yüz doxsan səkkizinci")
641-ci   →  altˈɯjˈyz ɡˌɯrxbˈirdʒi              ("altı yüz qırx birinci")
```

I measured it: the speaking rate of rows with digits differs from those without by
**only 1.9%** — meaning eSpeak's expansion matches the audio. Manual normalization is
not required.

**Exception:** Roman numerals are read incorrectly — `kral II Nektanebo` → `ˌɯˈɯ`
("ı ı" instead of "ikinci"). It is rare, but worth knowing.

## Phoneme coverage — full warm start

**0 unknown symbols** in the real dataset text. Everything eSpeak `az` outputs exists
in Inflect's 178-symbol IPA inventory, meaning the embedding table is transferred in
full and no row starts random.

## This is most likely synthetic audio

I cannot prove it, but the evidence is strong:

| Sign | Measurement |
|---|---|
| Leading silence is far too consistent | p25=0.227s, med=0.247s, p75=0.264s |
| Noise floor is practically zero | p5 median 0.0003 (a real recording has room tone) |
| RMS is very narrow | 0.081–0.171, med 0.119 |
| A bump at 8–10 kHz in the spectrum, then a sharp cut-off at ~11 kHz | neural vocoder signature |
| The texts are in an encyclopedic/LLM style | "Müasir..." 72 times at the start of a sentence |
| The dataset card is empty | no provenance |

**Consequence:** if you train on it, you will be distilling the voice of another TTS
system. The quality ceiling is the quality of the source system and the prosody will
be flat. The upside: the audio is very clean and there is no speaker drift — that
makes training easier.

**Licence.** The dataset repository declares no licence file. The dataset author was
asked directly in the Hugging Face community tab and confirmed that the dataset may
be used freely by anyone, with attribution requested. So attribution is the
condition. The terms of the source TTS system are still not known.

## Filtering — it has to happen BEFORE prepare

Critical: `inflect-adapt prepare` **stops the whole run** at the first clip it does
not accept (`prepare.py:306` → `PreparationError`), it does not skip the row. So the
filtering is done in the converter and prepare never sees a row that would be
rejected.

I measured all 10 shards (`min=1.0s`, `max=14.0s`, `clipped>0.1%`):

| Shard | rows | kept | long | hours |
|---|---:|---:|---:|---:|
| 0 | 1009 | 1008 | 1 | 2.50 |
| 1 | 1009 | 988 | 20 | 2.86 |
| 2 | 1009 | 1006 | 2 | 2.38 |
| 3 | 1009 | 1007 | 2 | 2.37 |
| 4 | 1009 | 956 | 53 | 2.47 |
| 5 | 1009 | 969 | 40 | 2.47 |
| 6 | 1009 | 975 | 34 | 2.50 |
| 7 | 1009 | 954 | 54 | 2.48 |
| 8 | 1008 | 916 | 92 | 2.64 |
| 9 | 1008 | 895 | 113 | 2.40 |

**Result: 9,674 clips out of 10,088 rows, 25.07 hours, mean clip 9.33 s.**

Dropped: 411 too long (>14s), 1 clipped, 2 duplicate/empty, 0 too short. The clips get
longer as the shard number increases — in the last shard 11% exceed the limit.
**Practically nothing is dropped because of clipping**: in some clips the peak touches
1.0, but the share of full-scale samples is below 0.1%.

## Usage

```powershell
# 1. Download and convert to a manifest (the shards are cached)
python scripts\build_az_dataset.py --output data\az

# for a quick look:
python scripts\build_az_dataset.py --output data\az_probe --shards 1 --limit 120

# 2. Check
inflect-preflight --manifest data\az\metadata.jsonl --audio-root data\az --language az

# 3. Prepare (the limits must be the same as in the converter)
inflect-adapt prepare --manifest data/az/metadata.jsonl --audio-root data/az `
  --language az --frontend espeak --output prepared/az `
  --min-duration-seconds 1.0 --max-duration-seconds 14.0 --validation-fraction 0.02
```

The converter writes `data/az/audio/NNNNNN.wav`, `data/az/metadata.jsonl` and
`data/az/conversion-summary.json`.

## Labelling — ready for future voices

The dataset is single-voice, but the manifest schema is multi-speaker: **every row
carries a `speaker` field** and it is exactly the field the multi-speaker patch reads.
The schema does not change when a second voice is added.

### One trap: ids are assigned in alphabetical order

This is what `prepare` does (`prepare.py:263`):

```python
speakers = sorted({row.speaker for row in rows if row.speaker})
```

So **the speaker id is the position in alphabetical order**, and that id is the row
number of the `emb_g` table. If you later add a label that sorts before an existing
one, all the voices are renumbered and the already trained embedding table becomes
invalid:

```
sorted(["az-01-main", "anar"])  ->  ["anar", "az-01-main"]
                                     id=0     id=1  (was 0 before!)
```

### The fix: a numbered label convention

```
az-01-main      -> speaker=0
az-02-aysel     -> speaker=1
az-03-resad     -> speaker=2
```

Format: `^[a-z]{2}-[0-9]{2}-[a-z0-9-]+$`. The zero-padded registration number **makes
alphabetical order equal to registration order**, so a voice added later always gets
the next id and does not steal an existing one. The converter **rejects** a label that
does not match this format.

### File layout and adding a voice

The clips are separated by voice and each voice is numbered within itself:

```
data/az/
  metadata.jsonl
  audio/
    az-01-main/000000.wav ...
    az-02-aysel/000000.wav ...
```

```powershell
python scripts\build_az_dataset.py --output data\az --speaker az-01-main
python scripts\build_az_dataset.py --output data\az --speaker az-02-aysel --append
```

Writing into an existing directory without `--append` is **rejected** — this prevents
accidentally overwriting the manifest.

### Verified with the real dataset

I tried a two-voice enrolment (60 clips each), then ran the real `prepare`:

```
manifest: {'az-01-main': 60, 'az-02-second': 60}
dataset.json speakers = ['az-01-main', 'az-02-second']
  speaker=0  ->  az-01-main
  speaker=1  ->  az-02-second
```

The registration order and the id order line up.

## Open risks

- **Synthetic source** — the quality ceiling is unknown, and so are the terms of the
  source TTS system.
- **The prosody may be flat** — there is little natural variation in TTS output.
- **Roman numerals** are read incorrectly.
- **Clip length is not evenly distributed** — the later shards are systematically
  longer, meaning the 14s filter cuts more from the end of the dataset.
