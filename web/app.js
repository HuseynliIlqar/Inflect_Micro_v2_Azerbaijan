/**
 * The playground: wiring only. Everything with logic in it lives in js/.
 *
 * Nothing is sent anywhere. The phonemiser and the model are downloaded once,
 * cached by the browser, and run on the visitor's machine -- which is the same
 * claim the command line version makes, kept honest on a web page.
 */

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs";
import { DEFAULT_LANGUAGE, label, chunkCount, LABELS } from "./js/i18n.js";
import { phonemize, loadPhonemizer } from "./js/phonemize.js";
import { Engine, toWav, SAMPLE_RATE } from "./js/tts.js";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 1);

const EXAMPLES = [
  "Salam, bu model tamamilə yerli maşında işləyir.",
  "Payız gəlmişdi və şəhərin küçələri saralmış yarpaqlarla örtülmüşdü.",
  "Sən bu kitabı oxumusan? Mənə çox maraqlı gəldi.",
  "II Dünya müharibəsi 01/09/1939 tarixində başladı və 25% artım oldu.",
];

const $ = (id) => document.getElementById(id);
const els = {
  title: $("title"), subtitle: $("subtitle"), languageLabel: $("language-label"),
  text: $("text"), textLabel: $("text-label"), speak: $("speak"), status: $("status"),
  examplesLabel: $("examples-label"), examples: $("examples"),
  audio: $("audio"), audioLabel: $("audio-label"), download: $("download"), stats: $("stats"),
  parameters: $("parameters-heading"),
  speed: $("speed"), speedValue: $("speed-value"), speedLabel: $("speed-label"), speedInfo: $("speed-info"),
  variation: $("variation"), variationValue: $("variation-value"),
  variationLabel: $("variation-label"), variationInfo: $("variation-info"),
  seed: $("seed"), seedLabel: $("seed-label"), seedInfo: $("seed-info"), shuffle: $("shuffle"),
  normalise: $("normalise"), normaliseLabel: $("normalise-label"), normaliseInfo: $("normalise-info"),
  advancedLabel: $("advanced-label"),
  maxWords: $("max-words"), maxWordsValue: $("max-words-value"),
  maxWordsLabel: $("max-words-label"), maxWordsInfo: $("max-words-info"),
  chunks: $("chunks"), chunksLabel: $("chunks-label"), chunksInfo: $("chunks-info"),
  offlineNote: $("offline-note"), footer: $("footer"), enNote: $("en-note"),
};

let language = DEFAULT_LANGUAGE;
let lastResult = null;
let busy = false;

const engine = new Engine(ort, {
  durationPath: "./onnx/duration.onnx",
  decodePath: "./onnx/decode.onnx",
  phonemize: (text) => phonemize(text, "az"),
});

// -- labels -----------------------------------------------------------------

function applyLanguage(next) {
  language = LABELS[next] ? next : DEFAULT_LANGUAGE;
  const t = (key) => label(language, key);

  document.documentElement.lang = language;
  document.title = t("title");
  els.title.textContent = t("title");
  els.subtitle.textContent = t("subtitle");
  els.languageLabel.textContent = t("language_label");
  els.textLabel.textContent = t("text_label");
  els.text.placeholder = t("text_placeholder");
  els.speak.textContent = t("speak");
  els.examplesLabel.textContent = t("examples_label");
  els.audioLabel.textContent = t("audio_label");
  els.download.textContent = t("download");
  els.parameters.textContent = t("parameters");
  els.speedLabel.textContent = t("speed_label");
  els.speedInfo.textContent = t("speed_info");
  els.variationLabel.textContent = t("variation_label");
  els.variationInfo.textContent = t("variation_info");
  els.seedLabel.textContent = t("seed_label");
  els.seedInfo.textContent = t("seed_info");
  els.shuffle.textContent = t("random_seed");
  els.normaliseLabel.textContent = t("normalise_label");
  els.normaliseInfo.textContent = t("normalise_info");
  els.advancedLabel.textContent = t("advanced");
  els.maxWordsLabel.textContent = t("max_words_label");
  els.maxWordsInfo.textContent = t("max_words_info");
  els.chunksLabel.textContent = t("chunks_label");
  els.chunksInfo.textContent = t("chunks_info");
  els.offlineNote.textContent = t("offline_note");
  els.enNote.textContent = t("en_note");
  els.footer.textContent = t("footer");

  for (const button of document.querySelectorAll(".segment")) {
    const active = button.dataset.language === language;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  }
  if (lastResult) showStats(lastResult);
}

function showStats(result) {
  els.stats.textContent = label(language, "stats", {
    seconds: result.seconds.toFixed(2),
    elapsed: result.elapsed.toFixed(2),
    realtime: result.realtime.toFixed(1),
    chunks: chunkCount(language, result.chunks),
    rate: SAMPLE_RATE,
  });
}

// -- synthesis ---------------------------------------------------------------

async function speak() {
  if (busy) return;
  const text = els.text.value.trim();
  if (!text) {
    els.status.textContent = label(language, "error_empty");
    return;
  }

  busy = true;
  els.speak.disabled = true;
  els.stats.textContent = "";
  try {
    els.status.textContent = label(language, "loading_phonemes");
    await loadPhonemizer();
    els.status.textContent = label(language, "loading_model");
    await engine.load();
    els.status.textContent = label(language, "synthesising");

    const began = performance.now();
    const { waveform, chunks } = await engine.speak(text, {
      speed: Number(els.speed.value),
      variation: Number(els.variation.value),
      seed: Number(els.seed.value) || 0,
      normalize: els.normalise.checked,
      maxWords: Number(els.maxWords.value),
      onChunk: (index, total) => {
        els.status.textContent = `${label(language, "synthesising")} ${index + 1}/${total}`;
      },
    });
    const elapsed = (performance.now() - began) / 1000;

    const blob = toWav(waveform);
    if (els.audio.src) URL.revokeObjectURL(els.audio.src);
    const url = URL.createObjectURL(blob);
    els.audio.src = url;
    els.download.href = url;
    els.download.hidden = false;

    els.chunks.textContent = chunks
      .map((chunk, index) => `[${index + 1}] ${chunk}`)
      .join("\n");

    const seconds = waveform.length / SAMPLE_RATE;
    lastResult = {
      seconds,
      elapsed,
      realtime: elapsed > 0 ? seconds / elapsed : 0,
      chunks: chunks.length,
    };
    showStats(lastResult);
    els.status.textContent = "";
  } catch (error) {
    els.status.textContent =
      error?.message === "empty" ? label(language, "error_empty") : String(error?.message ?? error);
  } finally {
    busy = false;
    els.speak.disabled = false;
  }
}

// -- wiring ------------------------------------------------------------------

for (const button of document.querySelectorAll(".segment")) {
  button.addEventListener("click", () => applyLanguage(button.dataset.language));
}

for (const sentence of EXAMPLES) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "example";
  button.textContent = sentence;
  button.addEventListener("click", () => {
    els.text.value = sentence;
    els.text.focus();
  });
  els.examples.appendChild(button);
}

els.speed.addEventListener("input", () => {
  els.speedValue.textContent = Number(els.speed.value).toFixed(2);
});
els.variation.addEventListener("input", () => {
  els.variationValue.textContent = Number(els.variation.value).toFixed(3);
});
els.maxWords.addEventListener("input", () => {
  els.maxWordsValue.textContent = els.maxWords.value;
});
els.shuffle.addEventListener("click", () => {
  els.seed.value = String(Math.floor(Math.random() * 2 ** 31));
});
els.speak.addEventListener("click", speak);
els.text.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) speak();
});

applyLanguage(DEFAULT_LANGUAGE);
els.status.textContent = label(language, "loading");
