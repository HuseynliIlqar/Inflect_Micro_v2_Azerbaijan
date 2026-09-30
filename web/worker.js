/**
 * Synthesis off the page's main thread.
 *
 * onnxruntime-web runs a graph synchronously on whatever thread calls it. On
 * the page itself, a phone without WebGPU froze for the whole synthesis --
 * about a minute per sentence, long enough for the browser to offer to kill
 * the tab. Here the page stays responsive and draws the progress bar.
 *
 * Protocol, one request at a time:
 *   in:  { id, voice, text, options }
 *   out: { id, type: "progress", stage, ...detail }
 *        { id, type: "result", waveform, chunks, backend, threads }  (waveform transferred)
 *        { id, type: "error", message }
 */

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs";
import { phonemize, loadPhonemizer } from "./js/phonemize.js";
import { Engine } from "./js/tts.js";
import { fetchModels } from "./js/fetch-model.js";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
// Threads need cross-origin isolation, which a static Space does not send;
// asking for them anyway only prints a warning and falls back to one.
// The Space sends COOP/COEP (`custom_headers` in its README), so the direct
// *.static.hf.space link is isolated; inside the huggingface.co iframe it is not.
const threads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
ort.env.wasm.numThreads = threads;

// `?backend=wasm` on the page forces the CPU path, to tell a GPU driver
// problem apart from a model problem on a given phone.
const preferWasm = new URL(self.location.href).searchParams.get("backend") === "wasm";

// Both lazy: a visitor who never picks English never downloads its 38 MB.
const ENGINES = {
  az: new Engine(ort, {
    durationPath: "./onnx/duration.onnx",
    decodePath: "./onnx/decode.onnx",
    phonemize: (text) => phonemize(text, "az"),
    fetchModels,
    preferWasm,
  }),
  en: new Engine(ort, {
    durationPath: "./onnx/en/duration.onnx",
    decodePath: "./onnx/en/decode.onnx",
    phonemize: (text) => phonemize(text, "en-us"),
    fetchModels,
    preferWasm,
  }),
};

async function handle({ id, voice, text, options }) {
  const post = (message, transfer) => self.postMessage({ id, ...message }, transfer ?? []);
  const engine = ENGINES[voice] ?? ENGINES.az;

  post({ type: "progress", stage: "phonemes" });
  await loadPhonemizer();

  if (!engine.decode) {
    await engine.load((stage, detail) => post({ type: "progress", stage, ...detail }));
  }
  post({ type: "progress", stage: "backend", backend: engine.backend, threads });

  const { waveform, chunks } = await engine.speak(text, {
    ...options,
    onChunk: (done, total) => post({ type: "progress", stage: "chunk", done, total }),
    onBackend: (backend) => post({ type: "progress", stage: "backend", backend, threads }),
  });
  post({ type: "result", waveform, chunks, backend: engine.backend, threads }, [waveform.buffer]);
}

// One request at a time: two overlapping requests would load the same engine
// twice and run one onnxruntime session concurrently.
let queue = Promise.resolve();

self.addEventListener("message", (event) => {
  queue = queue.then(async () => {
    try {
      await handle(event.data);
    } catch (error) {
      console.error("[aztts worker]", error);
      self.postMessage({
        id: event.data?.id,
        type: "error",
        message: error?.message ?? String(error),
      });
    }
  });
});
