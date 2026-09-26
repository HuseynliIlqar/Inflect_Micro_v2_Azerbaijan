# The browser playground

`index.html` is the whole application: the model runs in the visitor's browser,
not on a server. It is deployed as a Hugging Face **static** Space, which costs
nothing and never sleeps.

<https://huggingface.co/spaces/ilqarrrr/Inflect_Micro_v2_Azerbaijan>

## How it works

| Piece | What it does |
| --- | --- |
| `js/phonemize.js` | eSpeak NG compiled to WebAssembly, the same phonemiser the Python frontend calls |
| `js/az-text.js` | `normalize_az` ported from `aztts/az_text.py` |
| `js/num-az.js` | Azerbaijani number words, standing in for `num2words` |
| `js/az-chunk.js` | `chunk_text` ported from `aztts/az_chunk.py` |
| `js/tts.js` | The ONNX graphs, the pauses between chunks and WAV encoding |
| `js/i18n.js` | Interface labels, generated from `webui/i18n.py` |

Synthesis runs through ONNX Runtime Web on WebGPU where the browser has it
(about 8x faster than real time on a laptop), falling back to WebAssembly
otherwise. The WASM path is single-threaded on a static host -- no
cross-origin isolation -- and is several times slower.

## The two voices

`onnx/` holds this project's Azerbaijani graphs; `onnx/en/` holds
[`owensong/Inflect-Micro-v2-ONNX`](https://huggingface.co/owensong/Inflect-Micro-v2-ONNX),
the base model's official export. Both were exported by the same toolkit and
take the same inputs, so one `Engine` class drives either -- it is constructed
with the graph paths and a phonemiser, and the page builds one per voice.

Each engine loads lazily: a visitor who never picks English never downloads its
38 MB. English skips `normalizeAz` and the stress layer entirely, exactly as
`aztts/en_voice.py` does on the command line, and the page disables that control
rather than ignoring it silently.

## Why a port and not the Python code

The page cannot call Python, and the parts that matter are small and pure. What
matters is that the port does not drift, so both halves are checked against the
original rather than against expectations:

```bash
node tests/test_num_az.mjs      # 24,022 checks against num2words
node tests/test_az_text.mjs     # 550 sentences against normalize_az
node tests/test_az_chunk.mjs    # the same 550 against chunk_text
```

The golden files under `tests/golden/` are generated from the Python side; the
snippet that writes them is in `packaging/PUBLISHING.md`.

`tests/golden/onnx.json` holds a waveform fingerprint produced by onnxruntime in
Python with the noise input zeroed. Running the same thing in the browser gives
the same sample count and the same RMS to seven decimals, which is how the ONNX
path was verified end to end.

## Running it locally

`web/onnx/` is not tracked -- the exports are 76 MB and the repository already
carries two checkpoints. Put the Azerbaijani graphs in `web/onnx/` and fetch the
English ones into `web/onnx/en/`:

```bash
python -c "
from huggingface_hub import hf_hub_download; import shutil, pathlib
pathlib.Path('web/onnx/en').mkdir(parents=True, exist_ok=True)
for name in ('duration', 'decode'):
    shutil.copy(hf_hub_download('owensong/Inflect-Micro-v2-ONNX', f'onnx/{name}.onnx'),
                f'web/onnx/en/{name}.onnx')
"
```

Then serve the directory:

```bash
python -m http.server 8123 --directory web
```

Opening `index.html` from the file system will not work: the modules and the
WebAssembly are fetched, so they need an HTTP origin.
