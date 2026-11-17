/**
 * Text -> IPA phonemes in the browser, through eSpeak NG compiled to WebAssembly.
 *
 * The build is `@diffusionstudio/piper-wasm`, the same eSpeak NG the Python
 * frontend calls. With the punctuation handled as phonemizer handles it
 * (./punctuation.js), it returns byte-identical phonemes to the Python frontend:
 * checked in Chrome on all 1,871 chunks of tests/golden/stress.json, before and
 * after the stress frontend (./stress.js), and on the English demo sentences.
 * That is what makes a browser playground possible at all -- the model only
 * ever heard eSpeak's output.
 *
 * eSpeak NG is GPL-3.0-or-later. The browser fetches it from a CDN; this page
 * does not serve it, and whoever self-hosts it redistributes it under the GPL.
 * See THIRD_PARTY_NOTICES.md.
 */

import { withPunctuation } from "./punctuation.js";

const CDN = "https://cdn.jsdelivr.net/npm/@diffusionstudio/piper-wasm@1.0.0/build/piper_phonemize";

// Every `callMain` runs eSpeak's main again, and this build does not recover
// what each run leaves behind: after roughly 120-150 calls (fewer with longer
// text) the module aborts with "RuntimeError: null function" -- a long text,
// or a long session, stopped speaking. So an instance is retired after
// MAX_CALLS calls, and a call that fails anyway is retried once on a fresh one.
// A new instance reuses the cached script, wasm and data.
const MAX_CALLS = 64;

let modulePromise = null;
let calls = 0;

/**
 * The build is a classic script that declares one global. The page runs this
 * module in a module worker, which has neither `<script>` nor a working
 * `importScripts()`, so the source is fetched and evaluated for that global.
 */
async function loadScript(url) {
  if (globalThis.createPiperPhonemize) return;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`could not load ${url} (HTTP ${response.status})`);
  const source = await response.text();
  globalThis.createPiperPhonemize = new Function(`${source}\nreturn createPiperPhonemize;`)();
}

/** Start eSpeak NG once; later calls reuse the same instance. */
export function loadPhonemizer() {
  if (!modulePromise) {
    calls = 0;
    modulePromise = (async () => {
      await loadScript(`${CDN}.js`);
      const lines = [];
      const module = await globalThis.createPiperPhonemize({
        print: (line) => lines.push(line),
        printErr: () => {},
        locateFile: (path) =>
          path.endsWith(".wasm")
            ? `${CDN}.wasm`
            : path.endsWith(".data")
              ? `${CDN}.data`
              : path,
      });
      return { module, lines };
    })().catch((error) => {
      // A dropped connection must not poison every later attempt.
      modulePromise = null;
      throw error;
    });
  }
  return modulePromise;
}

/**
 * One chunk of text as a phoneme string, punctuation handled as the Python
 * frontends handle it (./punctuation.js): eSpeak sees the pieces between the
 * marks, and the marks are put back.
 */
export function phonemize(text, voice = "az") {
  return withPunctuation(text, (piece) => phonemizePiece(piece, voice));
}

async function phonemizePiece(text, voice) {
  try {
    return await phonemizeOnce(text, voice);
  } catch {
    modulePromise = null;
    return phonemizeOnce(text, voice);
  }
}

/** The CLI prints one JSON line per sentence; the lines are joined with a space. */
async function phonemizeOnce(text, voice) {
  if (calls >= MAX_CALLS) modulePromise = null;
  const { module, lines } = await loadPhonemizer();
  calls += 1;
  lines.length = 0;
  module.callMain([
    "-l",
    voice,
    "--input",
    JSON.stringify([{ text }]),
    "--espeak_data",
    "/espeak-ng-data",
  ]);
  return lines
    .map((line) => JSON.parse(line).phonemes.join(""))
    .join(" ")
    .trim();
}
