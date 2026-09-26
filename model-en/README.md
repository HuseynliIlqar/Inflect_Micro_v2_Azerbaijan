# The English voice

This is **Inflect-Micro-v2** by Owen Song, the English checkpoint this project's
Azerbaijani model was adapted from. It is included here so that a clone can
speak both languages without downloading anything.

- Source: <https://huggingface.co/owensong/Inflect-Micro-v2>
- Licence: Apache-2.0 (`LICENSE`), the same as this repository
- 9,356,513 parameters, 410 tensors, 24 kHz, single speaker

**Do not edit anything in this directory.** It is a vendored upstream export,
not our code; `checksums.sha256` verifies all six files. If the English voice
needs to behave differently, change `aztts/en_voice.py` instead.

```bash
cd model-en && sha256sum -c checksums.sha256
```

## What is here, and what is not

Only what is needed to speak: the weights and their configuration, plus the
licence, the notices and the citation file that must travel with them.

| File | |
| --- | --- |
| `model.pth` | The weights (37 MB, Git LFS) |
| `model.pth.json` | The upstream export report, including the weights' SHA-256 |
| `config.json` | Architecture and audio settings |
| `LICENSE`, `THIRD_PARTY_NOTICES.md`, `CITATION.cff` | Upstream, unchanged |

The full upstream package also ships its own `inference.py`, runtime and
frontend. Those are deliberately left out: they declare modules with the same
names as the ones in `model/`, and importing both in one process returns
whichever was imported first. `aztts/en_voice.py` therefore loads these weights
into `model/`'s runtime and phonemises English text itself.

The complete package is still available for training work, and
`training/scripts/download_model.py` fetches it into `training/base-model/`,
which this repository does not track.

## Citing it

The English voice is someone else's work. If you use it, cite it as
`CITATION.cff` asks:

> Song, Owen. *Inflect-Micro-v2*, version 2.0.0, 2026.
> <https://huggingface.co/owensong/Inflect-Micro-v2>
