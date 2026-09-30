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
| `js/tts.js` | The ONNX graphs, the pauses between chunks, the bleep and WAV encoding |
| `js/az-censor.js` | `censor_az` ported from `aztts/az_profanity.py`; always on here |
| `js/waveform.js` | The result's waveform: bar heights, drawing, click-to-seek |
| `js/wave-view.js` | That waveform wired to the page: redraws on play, resize and theme |
| `js/reveal.js` | When a finished clip is off screen, scroll to it (phones lost the result) |
| `js/backend-plan.js` | The fallback chain: which backend and precision to try, in what order |
| `js/mode-banner.js` | What the page says about it: the mode banner, the badge, the diagnostics line |
| `js/device-check.js` | The device verdict on page load, and which next step fits an error |
| `js/device-view.js`, `js/notify.js` | The device card, its dialog, and toasts |
| `worker.js` | Runs the phonemiser and the graphs off the page's main thread |
| `vendor/onnxruntime-web/` | onnxruntime-web 1.30.0, served from this origin; not in git -- `python tools/fetch_web_runtime.py` |
| `js/fetch-model.js` | Downloads the graphs with byte progress and keeps them in Cache Storage |
| `js/progress.js` | The progress bar's arithmetic |
| `js/theme.js` | Light, dark or automatic; the choice is remembered on the device |
| `js/i18n.js` | Interface labels, generated from `webui/i18n.py` |
| `tokens.css`, `styles.css`, `components.css` | Colour tokens (light and both dark blocks), the page, then the banner, device card, dialog and toasts -- loaded in that order |

Synthesis runs through ONNX Runtime Web on WebGPU where the browser has it
(about 8x faster than real time on a laptop), falling back to WebAssembly
otherwise. The WASM path is single-threaded on a static host -- no
cross-origin isolation -- and is several times slower.

## Why it can be slow, and why the page no longer freezes

onnxruntime-web runs a graph synchronously on the thread that calls it. When
that was the page's own thread, a phone without WebGPU froze solid for the
whole synthesis -- 57 seconds for a four-second sentence in a phone-class CPU
emulation -- and Android offered to kill the tab. Everything now runs in
`worker.js`; the page only draws the status, the elapsed seconds and the bar.

The WASM path is slower than WebGPU, and how much depends on threads. Threads
need cross-origin isolation, so the Space's README sets it:

```yaml
custom_headers:
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
```

On the direct `*.static.hf.space` link the worker then runs on up to 4 threads:
1.8-2.3 s for a 5.25 s sentence on a desktop CPU, against 3.9 s on one thread.
Inside the huggingface.co page the Space is an iframe without the
`cross-origin-isolated` permission, so it stays on one thread; there the note
under the button links to the direct URL. Everything cross-origin the page
loads -- jsdelivr (CORP `cross-origin`) and the Hub CDN (CORS `*`) -- passes
`require-corp`; anything new must too, or it will be blocked. The statistics
name the backend and the thread count. The CLI runs the same model at 2-4x
real time.

`?backend=wasm` on the page URL forces the CPU path. It is for telling a GPU
driver problem on a particular phone apart from a model problem: if the audio
is wrong with WebGPU and right with `?backend=wasm`, the driver is at fault.

Phones and tablets start on the CPU and never ask for a WebGPU adapter. On a
Redmi Note 10 Pro (Adreno 618, Chrome 154) the GPU step was no faster than
int8 on four threads (0.3x real time either way, plus 6-20 s of GPU compile),
computed wrong durations -- the same sentence and seed came out 4.03 s and
4.12 s long where the CPU gives 3.79 s every time -- and lost the WebGPU
instance after each use. `?backend=webgpu` still tries the GPU on a phone, for
testing; the device card says *Phone -- running on the processor* and the
diagnostics line carries `phone=yes | gpu=skipped-on-phone`. A phone visitor is told
this in plain words -- the device popup, a note under *Speak* and the mode
banner say the speech is made in the phone's browser, which -- not the phone:
the same model as an installed app runs at normal speed -- makes it slower and
simplified, and that a computer with Chrome or Edge gives full speed and
quality. The popup's *Show the technical reason* lists why: WebAssembly instead
of native code, the threads the browser allows, WebGPU unreliable on phones,
the int8 model, the first download, and the page pausing with the screen off;
`web/tests/test_phone_copy.mjs` keeps those texts free of jargon. When a clip
is ready the result gets a *Ready* chip and a pulse, and the page scrolls to it
if it is off screen -- not while the visitor is typing or the device dialog is
open, and without smooth scrolling under reduced motion. An iPad in its
default desktop mode reports a Mac and keeps WebGPU.

## The fallback chain

`js/backend-plan.js` decides the order and `Engine` walks it, moving down one
step whenever the current one fails to load or fails while running:

| Step | Backend | Decoder | When |
| --- | --- | --- | --- |
| 1 | WebGPU | fp32 `decode.onnx` | a WebGPU adapter exists, and not a phone (unless `?backend=webgpu`) |
| 2 | CPU (wasm) | int8 `decode.int8.onnx` | the voice ships one (Azerbaijani does) |
| 3 | CPU (wasm) | fp32 `decode.onnx` | always, last |

Every failed step is kept with its reason (`engine.failures`, sent to the page
as `fallbacks`), so a tester's report says what was tried and why it failed.

**The int8 decoder** (14.8 MB, built by `tools/quantize_decoder.py`) is static
QDQ int8 with the flow, the vocoder's first and last convolutions and its last
upsampling stage kept in fp32. Measured against fp32 on the five demo sentences
with the same noise: a mean log-spectral distance of 0.53, below the 0.73
between two fp32 seeds -- the change is smaller than the model's own variation
between takes -- and about 1.5x faster in onnxruntime-web's wasm backend on
1 and 4 threads. Dynamic quantisation was slower than fp32 on this model;
quantising the last stage cost quality without gaining speed.

Tester switches: `?precision=fp32` skips the int8 step, and a request with
`options.fullQuality` does the same for one synthesis. English has no int8
decoder and goes straight to step 3 without WebGPU.

**What a tester sees.** Above the player, a banner names the step that is
speaking whenever it is not WebGPU: *Fast mode* (int8, with a *Speak in full
quality* button), *Full quality* (fp32 on the CPU by choice, with the way
back), *Fallback mode* (int8 failed on this device), and WebGPU's own error
when it failed. Beside the download button a badge reads `GPU · fp32` or
`CPU · int8 · 4 threads`, so a screenshot says which model was heard, and
*Copy diagnostics* puts one line on the clipboard with the step, threads,
isolation, every failure and its reason, the speed and the browser.

**Before anything is downloaded**, the page checks the device and says what to
expect, in two places at once. A card beside the header stays for the whole
visit: a three-bar speed meter (3 WebGPU, 2 CPU on 4 threads, 1 CPU on one
thread), coloured green, amber or red, with the verdict in words. And a popup:
a modal dialog when the device is degraded -- inside the Hugging Face page on
one thread it offers the direct link -- or a toast when WebGPU is there.
*Don't show this again* is remembered per verdict, so a device that changes
class is told again; *Details* on the card reopens the dialog. If WebGPU was
expected and the worker falls back, the card turns and a toast says why.

Errors come as toasts that name the next step (network, worker, memory, no
backend), and stay until closed; empty text is flagged under the field itself.

## Caching the graphs

The Space answers `onnx/decode.onnx` with a `no-store` redirect to a freshly
signed CDN URL, so the browser's HTTP cache never hits and every visit
downloaded 37 MB again. `js/fetch-model.js` keeps the bytes in Cache Storage
under the page's own path instead. **When the graphs on the Space change, bump
`CACHE_NAME`** in that file; older caches are deleted on the next visit.

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
node tests/test_progress.mjs    # the progress bar and the cached download
node tests/test_backend_plan.mjs    # the order of the fallback chain
node tests/test_engine_fallback.mjs # Engine walking it, against a stand-in onnxruntime
node tests/test_mode_banner.mjs    # the banner, the badge and the diagnostics line
node tests/test_device_check.mjs   # the device verdict, its words, and error advice
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

The page also needs onnxruntime-web next to it, in `web/vendor/` (git-ignored,
27 MB): WebKit refuses a CDN's modules in a worker, so it is served from the
page's own origin. Fetch it once:

```bash
python tools/fetch_web_runtime.py
```

Then serve the directory:

```bash
python -m http.server 8123 --directory web
```

Opening `index.html` from the file system will not work: the modules and the
WebAssembly are fetched, so they need an HTTP origin.
