# The browser playground

`index.html` is the whole application: the model runs in the visitor's browser,
not on a server. It is deployed as a Hugging Face **static** Space, which costs
nothing and never sleeps.

<https://huggingface.co/spaces/ilqarrrr/azerbaijani-tts>

## How it works

| Piece | What it does |
| --- | --- |
| `js/phonemize.js` | eSpeak NG compiled to WebAssembly, the same phonemiser the Python frontend calls |
| `js/az-text.js` | `normalize_az` ported from `aztts/az_text.py` |
| `js/num-az.js` | Azerbaijani number words, standing in for `num2words` |
| `js/az-chunk.js` | `chunk_text` ported from `aztts/az_chunk.py` |
| `js/tts.js` | The two ONNX graphs, the pauses between chunks and WAV encoding |
| `js/i18n.js` | Interface labels, generated from `webui/i18n.py` |

Synthesis runs through ONNX Runtime Web on WebGPU where the browser has it
(about 8x faster than real time on a laptop), falling back to WebAssembly
otherwise. The WASM path is single-threaded on a static host -- no
cross-origin isolation -- and is several times slower.

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

`web/onnx/` is not tracked -- the export is 38 MB and the repository already
carries two checkpoints. Copy `duration.onnx` and `decode.onnx` in, then serve
the directory:

```bash
python -m http.server 8123 --directory web
```

Opening `index.html` from the file system will not work: the modules and the
WebAssembly are fetched, so they need an HTTP origin.
