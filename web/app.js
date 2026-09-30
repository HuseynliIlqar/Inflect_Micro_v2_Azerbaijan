/**
 * The playground: wiring only. Everything with logic in it lives in js/, and
 * the synthesis itself runs in worker.js so the page never freezes.
 *
 * Nothing is sent anywhere. The phonemiser and the model are downloaded once,
 * cached on the device, and run on the visitor's machine -- which is the same
 * claim the command line version makes, kept honest on a web page.
 */

import { DEFAULT_LANGUAGE, label, chunkCount, LABELS } from "./js/i18n.js";
import { toWav, SAMPLE_RATE } from "./js/tts.js";
import { downloadFraction, chunkFraction, megabytes } from "./js/progress.js";

const EXAMPLES = {
  az: [
    "Salam, bu model tamamilə yerli maşında işləyir.",
    "Payız gəlmişdi və şəhərin küçələri saralmış yarpaqlarla örtülmüşdü.",
    "Sən bu kitabı oxumusan? Mənə çox maraqlı gəldi.",
    "II Dünya müharibəsi 01/09/1939 tarixində başladı və 25% artım oldu.",
  ],
  en: [
    "Hello, this model runs completely offline on your machine.",
    "Autumn had come, and the streets were covered with yellow leaves.",
    "Have you read this book? I found it very interesting.",
  ],
};

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
  voiceLabel: $("voice-label"), voiceInfo: $("voice-info"),
  offlineNote: $("offline-note"), speedNote: $("speed-note"),
  footer: $("footer"), enNote: $("en-note"),
  progress: $("progress"), elapsed: $("elapsed"), slowNote: $("slow-note"),
};

// The Azerbaijani model is this project's; the English one is
// owensong/Inflect-Micro-v2, the checkpoint it was adapted from. Each loads
// lazily inside the worker.
const VOICES = ["az", "en"];

let language = DEFAULT_LANGUAGE;
let voice = "az";
let lastResult = null;
let busy = false;
let backend = null;
let threads = 1;

// The page shipped on the Space's direct host; inside the huggingface.co
// iframe it cannot be cross-origin isolated, so it gets no wasm threads.
const DIRECT_URL = "https://ilqarrrr-inflect-micro-v2-azerbaijan.static.hf.space/index.html";
const IN_FRAME = window.top !== window.self;

// -- the worker ----------------------------------------------------------------

let pending = null;
let nextId = 0;
let worker = null;

// A worker killed for memory sends nothing at all. If none of its messages
// arrives for this long, it is treated as dead rather than waited on forever.
// A single long chunk on a slow phone's CPU stays well inside it.
const WORKER_SILENCE_MS = 5 * 60 * 1000;
let watchdog = null;

function armWatchdog() {
  clearTimeout(watchdog);
  watchdog = setTimeout(() => failWorker("silent"), WORKER_SILENCE_MS);
}

function failPending(error) {
  clearTimeout(watchdog);
  pending?.reject(error);
  pending = null;
}

/** Drop a broken worker; the next request starts a fresh one. */
function failWorker(reason) {
  console.error("[aztts] worker failed", reason);
  worker?.terminate();
  worker = null;
  failPending(new Error(label(language, "error_worker")));
}

function startWorker() {
  try {
    const url = new URL("./worker.js", import.meta.url);
    url.search = location.search; // passes ?backend=wasm through
    const created = new Worker(url, { type: "module" });
    created.addEventListener("message", ({ data }) => {
      if (!pending || data.id !== pending.id) return;
      armWatchdog();
      if (data.type === "progress") pending.onProgress(data);
      else if (data.type === "result") {
        pending.resolve(data);
        pending = null;
      } else if (data.type === "error") failPending(new Error(data.message));
    });
    // Fires when the module fails to load (a CDN import on a dropped
    // connection, a browser without module workers) or throws at top level.
    created.addEventListener("error", (event) => {
      event.preventDefault();
      failWorker(event.message || "error");
    });
    created.addEventListener("messageerror", () => failWorker("messageerror"));
    return created;
  } catch (error) {
    console.error("[aztts] worker could not start", error);
    return null;
  }
}

/** One request to the worker; progress arrives through `onProgress`. */
function synthesise(request, onProgress) {
  worker ??= startWorker();
  if (!worker) return Promise.reject(new Error(label(language, "error_worker")));
  return new Promise((resolve, reject) => {
    nextId += 1;
    pending = { id: nextId, resolve, reject, onProgress };
    armWatchdog();
    worker.postMessage({ id: nextId, ...request });
  });
}

// Started at load so the imports download while the visitor is typing.
worker = startWorker();

// -- progress ------------------------------------------------------------------

/** `undefined` hides the bar, `null` makes it indeterminate, 0..1 fills it. */
function setProgress(fraction, text = els.status.textContent) {
  const bar = els.progress;
  const fill = bar.firstElementChild;
  if (fraction === undefined) {
    bar.hidden = true;
    return;
  }
  bar.hidden = false;
  bar.setAttribute("aria-label", label(language, "progress_label"));
  if (text) bar.setAttribute("aria-valuetext", text);
  else bar.removeAttribute("aria-valuetext");
  bar.classList.toggle("is-indeterminate", fraction === null);
  if (fraction === null) {
    bar.removeAttribute("aria-valuenow");
    fill.style.width = "";
  } else {
    const percent = Math.round(fraction * 100);
    bar.setAttribute("aria-valuenow", String(percent));
    fill.style.width = `${percent}%`;
  }
}

/** The CPU-path note: how slow to expect, and the faster link when framed. */
function renderSlowNote() {
  els.slowNote.hidden = backend !== "wasm";
  if (els.slowNote.hidden) return;
  const t = (key, values) => label(language, key, values);
  if (threads > 1) {
    els.slowNote.textContent = t("slow_note_threads", { threads });
    return;
  }
  els.slowNote.textContent = t("slow_note");
  if (IN_FRAME) {
    const link = document.createElement("a");
    link.href = DIRECT_URL;
    link.target = "_blank";
    link.rel = "noopener";
    link.textContent = DIRECT_URL.replace("https://", "").replace("/index.html", "");
    els.slowNote.append(" ", t("open_direct"), " ", link);
  }
}

function backendName(name, threads = 1) {
  if (!name) return "";
  if (name === "wasm" && threads > 1) return label(language, "backend_wasm_threads", { threads });
  return label(language, `backend_${name}`);
}

let lastDownloadStep = null;

/** Status line and bar for one progress message from the worker. */
function showProgress(message, requestedVoice) {
  const t = (key, values) => label(language, key, values);
  switch (message.stage) {
    case "phonemes":
      lastDownloadStep = null;
      els.status.textContent = t("loading_phonemes");
      setProgress(null);
      break;
    case "download": {
      const fraction = downloadFraction(message.loaded, message.total);
      const text =
        fraction === null
          ? t(requestedVoice === "en" ? "en_loading" : "loading_model")
          : t("download_progress", {
              loaded: megabytes(message.loaded),
              total: megabytes(message.total),
            });
      // The status is a live region: rewrite it in 10% steps, not per read,
      // or a screen reader queues dozens of byte counts.
      const step = fraction === null ? -1 : Math.floor(fraction * 10);
      if (step !== lastDownloadStep) {
        lastDownloadStep = step;
        els.status.textContent = text;
      }
      setProgress(fraction, text);
      break;
    }
    case "compile":
      els.status.textContent = t("compiling");
      setProgress(null);
      break;
    case "backend":
      backend = message.backend;
      threads = message.threads ?? 1;
      renderSlowNote();
      break;
    case "chunk":
      els.status.textContent =
        message.total > 1
          ? `${t("synthesising")} ${message.done + 1}/${message.total}`
          : t("synthesising");
      setProgress(chunkFraction(message.done, message.total));
      break;
    default:
      break;
  }
}

/** Seconds since `began`, beside the status line, once a second. */
function startClock(began) {
  const tick = () => {
    els.elapsed.textContent = `${Math.floor((performance.now() - began) / 1000)} s`;
  };
  tick();
  const timer = setInterval(tick, 1000);
  return () => {
    clearInterval(timer);
    els.elapsed.textContent = "";
  };
}

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
  els.voiceLabel.textContent = t("voice_label");
  els.voiceInfo.textContent = t("voice_info");
  for (const button of document.querySelectorAll("[data-voice]")) {
    button.textContent = t(`voice_${button.dataset.voice}`);
  }
  els.chunksLabel.textContent = t("chunks_label");
  els.chunksInfo.textContent = t("chunks_info");
  els.offlineNote.textContent = t("offline_note");
  els.speedNote.textContent = t("speed_note");
  renderSlowNote();
  els.progress.setAttribute("aria-label", t("progress_label"));
  els.enNote.textContent = t("en_note");
  els.footer.textContent = t("footer");

  for (const button of document.querySelectorAll(".segment")) {
    const active = button.dataset.language === language;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  }
  applyVoice(voice);
  if (lastResult) showStats(lastResult);
}

/**
 * The Azerbaijani text layer does not apply to the English checkpoint --
 * `normalizeAz` would rewrite numbers into Azerbaijani words -- so its control
 * is disabled rather than quietly ignored.
 */
function applyVoice(next) {
  voice = VOICES.includes(next) ? next : "az";
  const azerbaijani = voice === "az";

  for (const button of document.querySelectorAll("[data-voice]")) {
    const active = button.dataset.voice === voice;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  }

  els.normalise.disabled = !azerbaijani;
  els.normaliseInfo.textContent =
    label(language, "normalise_info") +
    (azerbaijani ? "" : " " + label(language, "az_only"));

  els.examples.replaceChildren();
  for (const sentence of EXAMPLES[voice]) {
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
}

function showStats(result) {
  els.stats.textContent = label(language, "stats", {
    seconds: result.seconds.toFixed(2),
    elapsed: result.elapsed.toFixed(2),
    realtime: result.realtime.toFixed(1),
    chunks: chunkCount(language, result.chunks),
    rate: SAMPLE_RATE,
  }) + (result.backend ? ` · ${backendName(result.backend, result.threads)}` : "");
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
  const requestedVoice = voice;
  const began = performance.now();
  const stopClock = startClock(began);
  // The statistics time synthesis alone, not the one-off download and setup.
  let synthesisBegan = null;
  try {
    const { waveform, chunks, backend: used, threads } = await synthesise(
      {
        voice: requestedVoice,
        text,
        options: {
          speed: Number(els.speed.value),
          variation: Number(els.variation.value),
          seed: Number(els.seed.value) || 0,
          normalize: requestedVoice === "az" && els.normalise.checked,
          maxWords: Number(els.maxWords.value),
        },
      },
      (message) => {
        if (message.stage === "chunk" && synthesisBegan === null) synthesisBegan = performance.now();
        showProgress(message, requestedVoice);
      },
    );
    const elapsed = (performance.now() - (synthesisBegan ?? began)) / 1000;
    setProgress(1);

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
      backend: used,
      threads,
    };
    showStats(lastResult);
    els.status.textContent = "";
  } catch (error) {
    console.error("[aztts] synthesis failed", error);
    els.status.textContent =
      error?.message === "empty" ? label(language, "error_empty") : String(error?.message ?? error);
  } finally {
    stopClock();
    setProgress(undefined);
    busy = false;
    els.speak.disabled = false;
  }
}

// -- wiring ------------------------------------------------------------------

for (const button of document.querySelectorAll(".segment")) {
  button.addEventListener("click", () => applyLanguage(button.dataset.language));
}

for (const button of document.querySelectorAll("[data-voice]")) {
  button.addEventListener("click", () => applyVoice(button.dataset.voice));
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
