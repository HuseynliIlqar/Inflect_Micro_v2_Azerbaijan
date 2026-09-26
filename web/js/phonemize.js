/**
 * Text -> IPA phonemes in the browser, through eSpeak NG compiled to WebAssembly.
 *
 * The build is `@diffusionstudio/piper-wasm`, the same eSpeak NG the Python
 * frontend calls. Verified: for Azerbaijani it returns byte-identical phonemes
 * to `model/deployment_frontend.py`, which is what makes a browser playground
 * possible at all -- the model only ever heard eSpeak's output.
 *
 * eSpeak NG is GPL-3.0-or-later. It is loaded from a CDN here and redistributed
 * by whoever hosts this page; see THIRD_PARTY_NOTICES.md.
 */

const CDN = "https://cdn.jsdelivr.net/npm/@diffusionstudio/piper-wasm@1.0.0/build/piper_phonemize";

let modulePromise = null;

function loadScript(url) {
  return new Promise((resolve, reject) => {
    if (globalThis.createPiperPhonemize) return resolve();
    const element = document.createElement("script");
    element.src = url;
    element.onload = () => resolve();
    element.onerror = () => reject(new Error(`could not load ${url}`));
    document.head.appendChild(element);
  });
}

/** Start eSpeak NG once; later calls reuse the same instance. */
export function loadPhonemizer() {
  if (!modulePromise) {
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
    })();
  }
  return modulePromise;
}

/**
 * One chunk of text as a phoneme string.
 *
 * The CLI prints one JSON line per sentence; the pieces are joined with a
 * space, which is what the Python frontend hands the model as well.
 */
export async function phonemize(text, voice = "az") {
  const { module, lines } = await loadPhonemizer();
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
