# Training

This folder records **how the checkpoint in `model/` was made**, and holds the
work on the next one. Nothing here is needed to run the model -- the
[README](../README.md) at the project root is enough for that.

| | What it is |
| --- | --- |
| [`v2/`](v2/README.md) | The round that produced **v2.1**, the checkpoint in `model/`: pipeline, results, the test battery a new checkpoint is measured with (`v2/pod/`), and the dataset-v2 sentences (`v2/dataset_v2/`). An archive: the pod is gone. |
| [`stress_frontend/`](stress_frontend/README.md) | The source of `model/frontend_hook.py`, the stress rules v2.1 was trained with, and their native-speaker review. The playground runs a JavaScript port of the hook (`web/js/engine/stress*.js`). |
| [`v3/`](v3/PLAN.md) | The next model: the plan, the recording kit, the model code and the training monitor. |
| `fast_monotonic_align.py` | Working code, still used: the vectorised alignment, held bit-identical to the reference by 25 tests. |

## The first round (v1)

The first Azerbaijani checkpoint was adapted from
[`owensong/Inflect-Micro-v2`](https://huggingface.co/owensong/Inflect-Micro-v2)
on [`ughurabbasov/azerbaijani-tts-dataset`](https://huggingface.co/datasets/ughurabbasov/azerbaijani-tts-dataset):
200,000 steps (1,343 epochs) on 25.07 hours of audio, on a RunPod A40. The warm
start copied 409 of 410 tensors bit-identically from the base, and the
checkpoint was chosen by ear. v2.1 continues from it
([`docs/MODEL.md`](../docs/MODEL.md)).

v2.1 replaced it, so that round's scripts, presets, multi-speaker patch and
results are no longer in the tree. They are in the history, as of commit
`f823ee1`:

```bash
git show f823ee1:training/README.md            # the full write-up of that round
git checkout f823ee1 -- training/scripts       # or configs, patches, results, docs
```

## Fast monotonic alignment

The official toolkit's `maximum_path` was a double loop in pure Python --
roughly **28 million iterations** per step, with the GPU idle.
`fast_monotonic_align.py` runs the same dynamic programme with vectorised torch
operations, diagonal by diagonal, on the GPU.

| | Before | After |
| --- | ---: | ---: |
| Step time | 8.7 s | **1.395 s** |
| GPU load | ~0% | **86-100%** |
| 200,000 steps | ~20 days | **3.2 days** |

Fast is not enough: the result has to be **bit-identical**, or the model learns
something else. The original implementation is frozen in
`tests/reference_monotonic_align.py`, and 25 tests compare the two at random
sizes, on edge cases, across mask variants and for determinism. They need no
GPU or dataset:

```bash
python -m pytest tests/test_fast_monotonic_align.py -v
```
