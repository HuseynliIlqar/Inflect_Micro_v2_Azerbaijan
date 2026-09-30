/**
 * Synthesis off the page's main thread.
 *
 * onnxruntime-web runs a graph synchronously on whatever thread calls it. On
 * the page itself, a phone without WebGPU froze for the whole synthesis --
 * about a minute per sentence, long enough for the browser to offer to kill
 * the tab. Here the page stays responsive and draws the progress bar.
 *
 * Protocol, one request at a time:
 *   in:  { id, voice, text, options } -- text is checked by js/text/text-check.js;
 *        a refused one comes back as an error whose message is the reason
 *   out: { id, type: "progress", stage, ...detail }
 *        { id, type: "result", waveform, chunks, backend, precision, fallbacks, threads }
 *          (waveform transferred; precision is "fp32" or "int8"; fallbacks lists
 *          every step that failed)
 *   options.fullQuality skips the int8 step of the fallback chain.
 *        { id, type: "error", message }
 *   in:  { type: "cancel", id } -- handled at once, not queued; the request
 *        stops before its next chunk (or before it starts, if still queued)
 *   out: { id, type: "cancelled" }
 */

// onnxruntime-web 1.30.0, served from this origin rather than a CDN: WebKit --
// every browser on an iPhone -- checks each module a worker imports against the
// page's Cross-Origin-Embedder-Policy and refuses a CDN's, whatever its CORS
// and CORP headers say, so the worker never started there. The files come from
// `python tools/fetch_web_runtime.py`; they are not in the repository.
import * as ort from "../vendor/onnxruntime-web/ort.webgpu.min.mjs";
import { phonemize, loadPhonemizer } from "./engine/phonemize.js";
import { CancelledError, Engine } from "./engine/engine.js";
import { fetchModels } from "./engine/fetch-model.js";
import { planOptionsFromQuery, thisIsPhone } from "./engine/backend-plan.js";
import { checkText } from "./text/text-check.js";

ort.env.wasm.wasmPaths = new URL("../vendor/onnxruntime-web/", import.meta.url).href;
// Threads need cross-origin isolation, which a static Space does not send;
// asking for them anyway only prints a warning and falls back to one.
// The Space sends COOP/COEP (`custom_headers` in its README), so the direct
// *.static.hf.space link is isolated; inside the huggingface.co iframe it is not.
const threads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
ort.env.wasm.numThreads = threads;

// Tester switches on the page URL: `?backend=wasm` forces the CPU, to tell a
// GPU driver problem apart from a model problem on a given phone, and
// `?precision=fp32` starts with the full-quality decoder instead of int8.
const { preferWasm, fullQuality, forceGpu } = planOptionsFromQuery(self.location.search);
const phone = thisIsPhone(self.navigator);

// Both lazy: a visitor who never picks English never downloads its 38 MB.
const ENGINES = {
  az: new Engine(ort, {
    durationPath: "../onnx/duration.onnx",
    decodePath: "../onnx/decode.onnx",
    // The CPU step of the fallback chain; see js/engine/backend-plan.js.
    int8DecodePath: "../onnx/decode.int8.onnx",
    phonemize: (text) => phonemize(text, "az"),
    // On for both voices (the Engine default), and no request option can turn
    // it off: the playground is public. Only a clone, with say.py, can.
    censor: true,
    fetchModels,
    preferWasm,
    fullQuality,
    phone,
    forceGpu,
  }),
  en: new Engine(ort, {
    durationPath: "../onnx/en/duration.onnx",
    decodePath: "../onnx/en/decode.onnx",
    phonemize: (text) => phonemize(text, "en-us"),
    fetchModels,
    preferWasm,
    fullQuality,
    phone,
    forceGpu,
  }),
};

// Ids the page has cancelled. Filled by the listener the moment a cancel
// arrives, read by the request it names at its next checkpoint.
const cancelled = new Set();

async function handle({ id, voice, text, options }) {
  const post = (message, transfer) => self.postMessage({ id, ...message }, transfer ?? []);
  const isCancelled = () => cancelled.has(id);
  const checkpoint = () => {
    if (isCancelled()) throw new CancelledError();
  };
  checkpoint();
  // The page checks first; this catches a request posted around it.
  const verdict = checkText(text, { voice });
  if (verdict.error) throw new Error(verdict.error);
  const engine = ENGINES[voice] ?? ENGINES.az;
  // The page's "full quality" choice, per request; the engine reloads only if
  // that changes what runs.
  await engine.setFullQuality(Boolean(options?.fullQuality) || fullQuality);
  const state = () => ({
    backend: engine.backend,
    precision: engine.precision,
    fallbacks: engine.failures,
    threads,
  });

  post({ type: "progress", stage: "phonemes" });
  await loadPhonemizer();
  checkpoint();

  if (!engine.decode) {
    await engine.load((stage, detail) => post({ type: "progress", stage, ...detail }));
  }
  post({ type: "progress", stage: "backend", ...state() });
  checkpoint();

  const { waveform, chunks } = await engine.speak(verdict.text, {
    ...options,
    isCancelled,
    onChunk: (done, total) => post({ type: "progress", stage: "chunk", done, total }),
    onBackend: () => post({ type: "progress", stage: "backend", ...state() }),
  });
  post(
    { type: "result", waveform, chunks, ...state() },
    [waveform.buffer],
  );
}

// One request at a time: two overlapping requests would load the same engine
// twice and run one onnxruntime session concurrently.
let queue = Promise.resolve();

self.addEventListener("message", (event) => {
  if (event.data?.type === "cancel") {
    cancelled.add(event.data.id);
    return;
  }
  const id = event.data?.id;
  queue = queue.then(async () => {
    try {
      await handle(event.data);
    } catch (error) {
      if (error instanceof CancelledError) {
        self.postMessage({ id, type: "cancelled" });
        return;
      }
      console.error("[aztts worker]", error);
      self.postMessage({
        id,
        type: "error",
        message: error?.message ?? String(error),
      });
    } finally {
      cancelled.delete(id);
    }
  });
});
