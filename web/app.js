/**
 * The playground: wiring only. Everything with logic in it lives in js/, and
 * the synthesis itself runs in worker.js so the page never freezes.
 *
 * Nothing is sent anywhere. The phonemiser and the model are downloaded once,
 * cached on the device, and run on the visitor's machine -- which is the same
 * claim the command line version makes, kept honest on a web page.
 */

import { DEFAULT_LANGUAGE, label, chunkCount, LABELS } from "./js/i18n.js";
import { toWav, SAMPLE_RATE, hasWebGpu, CancelledError } from "./js/tts.js";
import { downloadFraction, chunkFraction, megabytes } from "./js/progress.js";
import { applyTheme, initialTheme, readStoredTheme } from "./js/theme.js";
import { planOptionsFromQuery, thisIsPhone } from "./js/backend-plan.js";
import { bannerState, badgeParts, diagnosticsLine, bannerCopyKeys } from "./js/mode-banner.js";
import { deviceVerdict, errorKind, adviceKey, isAppleMobile } from "./js/device-check.js";
import { createDeviceView } from "./js/device-view.js";
import { toast } from "./js/notify.js";
import { createWaveView } from "./js/wave-view.js";
import { needsReveal } from "./js/reveal.js";

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
  themeLabel: $("theme-label"),
  text: $("text"), textLabel: $("text-label"), speak: $("speak"), cancel: $("cancel"),
  status: $("status"),
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
  footer: $("footer"), censorNote: $("censor-note"), aboutLabel: $("about-label"),
  skipLink: document.querySelector(".skip-link"),
  wave: $("wave"), waveBox: document.querySelector(".wave"), waveEmpty: $("wave-empty"),
  progress: $("progress"), elapsed: $("elapsed"), slowNote: $("slow-note"),
  phoneNote: $("phone-note"), phoneNoteText: $("phone-note-text"), phoneWhy: $("phone-why"),
  result: $("result"), resultReady: $("result-ready"), resultReadyText: $("result-ready-text"),
  modeBanner: $("mode-banner"), modeTitle: $("mode-title"), modeText: $("mode-text"),
  modeGpu: $("mode-gpu"), modeAction: $("mode-action"), modeBadge: $("mode-badge"),
  copyDiagnostics: $("copy-diagnostics"), copyStatus: $("copy-status"), diagnostics: $("diagnostics"),
  deviceCard: $("device-card"), deviceMeter: document.querySelector("#device-card .meter"),
  deviceLabel: $("device-label"), deviceTitle: $("device-title"), deviceDetails: $("device-details"),
  dialog: $("device-dialog"), dialogMeter: document.querySelector("#device-dialog .meter"),
  dialogTitle: $("dialog-title"), dialogText: $("dialog-text"), dialogNote: $("dialog-note"),
  dialogDontShow: $("dialog-dont-show"), dialogDontShowLabel: $("dialog-dont-show-label"),
  dialogDirect: $("dialog-direct"), dialogOk: $("dialog-ok"),
  toasts: $("toasts"), textError: $("text-error"),
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
// The fallback chain's state as the worker last reported it (js/backend-plan.js).
let precision = null;
let fallbacks = [];
let modeVoice = voice;
// "Full quality" skips the int8 step; `?precision=fp32` starts with it on.
let fullQuality = planOptionsFromQuery(location.search).fullQuality;

// The page shipped on the Space's direct host; inside the huggingface.co
// iframe it cannot be cross-origin isolated, so it gets no wasm threads.
const DIRECT_URL = "https://ilqarrrr-inflect-micro-v2-azerbaijan.static.hf.space/index.html";
const IN_FRAME = window.top !== window.self;

// -- the device check ------------------------------------------------------------

const device = createDeviceView(els, { directUrl: DIRECT_URL });
const { preferWasm, forceGpu } = planOptionsFromQuery(location.search);
// The worker applies the same rule; see js/backend-plan.js.
const PHONE = thisIsPhone();
// Every browser on an iPhone or iPad is WebKit: advice never names another one.
const APPLE_MOBILE = isAppleMobile({ userAgent: navigator.userAgent, maxTouchPoints: navigator.maxTouchPoints });
// The same thread count the worker will use (worker.js).
const expectedThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

/** On page load, before any model is fetched: what will this device do? */
async function checkDevice() {
  // A phone does not ask for an adapter it will not use.
  const gpu = PHONE && !forceGpu ? false : await hasWebGpu();
  device.announce(deviceVerdict({
    gpu, preferWasm, phone: PHONE, forceGpu, appleMobile: APPLE_MOBILE,
    threads: expectedThreads, framed: IN_FRAME,
  }));
}

/** WebGPU was expected and the worker fell back: say so, once. */
function reviseDevice() {
  if (device.verdict?.level !== "gpu" || backend !== "wasm") return;
  const reason = fallbacks.find((f) => f.backend === "webgpu")?.message ?? "";
  device.revise(deviceVerdict({ gpu: true, gpuFailed: true, appleMobile: APPLE_MOBILE, threads, framed: IN_FRAME }), reason);
}

/** Empty text: said next to the field, not in a popup. */
function showTextError(show, { focus = true } = {}) {
  els.textError.hidden = !show;
  if (show) {
    els.textError.textContent = `${label(language, "error_empty")} ${label(language, "error_empty_hint")}`;
    els.text.setAttribute("aria-invalid", "true");
    els.text.setAttribute("aria-describedby", "text-error");
    if (focus) els.text.focus();
  } else {
    els.text.removeAttribute("aria-invalid");
    els.text.removeAttribute("aria-describedby");
  }
}

/** An error as a toast that names the next step, not just the problem. */
function showError(message) {
  toast(els.toasts, {
    tone: "bad",
    title: label(language, "error_title"),
    text: label(language, adviceKey(errorKind(message), { appleMobile: APPLE_MOBILE })),
    closeLabel: label(language, "toast_close"),
  });
}

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
      // The cancelled request has stopped (or finished first): the worker is
      // free, so it need not be terminated.
      if (data.id === cancelling?.id && data.type !== "progress") {
        settleCancel();
        return;
      }
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

// After Cancel the worker stops at its next chunk and confirms. One chunk on a
// slow phone's CPU can run for a minute, so a worker that has not confirmed
// within this long is terminated; the next request starts a fresh one, with
// the model from the browser's cache.
const CANCEL_GRACE_MS = 2000;
let cancelling = null; // { id, timer } until the worker confirms

function settleCancel() {
  clearTimeout(cancelling?.timer);
  cancelling = null;
}

function dropUnresponsiveWorker() {
  settleCancel();
  worker?.terminate();
  worker = null;
}

/** Stop the running request. The page is free at once; the worker follows. */
function cancelSynthesis() {
  if (!pending) return;
  const { id } = pending;
  worker?.postMessage({ type: "cancel", id });
  settleCancel();
  cancelling = { id, timer: setTimeout(dropUnresponsiveWorker, CANCEL_GRACE_MS) };
  failPending(new CancelledError());
}

/** One request to the worker; progress arrives through `onProgress`. */
function synthesise(request, onProgress) {
  // Still inside a cancelled chunk: a new request would queue behind it.
  if (cancelling) dropUnresponsiveWorker();
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
  // On a phone the phone note already says this, in plainer words.
  els.slowNote.hidden = backend !== "wasm" || PHONE;
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

/** The banner above the player: which step of the fallback chain is speaking. */
function renderModeBanner() {
  const t = (key, values) => label(language, key, values);
  const state = bannerState({ voice: modeVoice, backend, precision, fallbacks, fullQuality });
  els.modeBanner.hidden = state.kind === "none";
  if (els.modeBanner.hidden) return;

  els.modeBanner.dataset.kind = state.kind;
  const copy = bannerCopyKeys(state, { phone: PHONE && !forceGpu });
  els.modeTitle.textContent = t(copy.title);
  els.modeText.textContent = copy.text ? t(copy.text) : "";
  els.modeText.hidden = !copy.text;
  els.modeGpu.hidden = !state.gpuFailure;
  if (state.gpuFailure) els.modeGpu.textContent = t("mode_gpu_failed", { reason: state.gpuFailure });
  els.modeAction.hidden = !state.action;
  if (state.action) {
    els.modeAction.textContent = t(`action_${state.action}`);
    els.modeAction.dataset.action = state.action;
  }
}

/** "CPU · int8 · 4 threads" beside the download button. */
function renderBadge(result) {
  const parts = result ? badgeParts(result) : [];
  els.modeBadge.hidden = parts.length === 0;
  els.modeBadge.dataset.precision = result?.precision ?? "";
  els.modeBadge.textContent = parts
    .map((part) => (typeof part === "string" ? part : label(language, "badge_threads", part)))
    .join(" · ");
}

function diagnostics() {
  return diagnosticsLine({
    voice: lastResult?.voice ?? modeVoice,
    backend,
    precision,
    threads,
    isolated: self.crossOriginIsolated,
    framed: IN_FRAME,
    fullQuality,
    phone: PHONE,
    forceGpu,
    fallbacks,
    seconds: lastResult?.seconds,
    elapsed: lastResult?.elapsed,
    chunks: lastResult?.chunks,
    userAgent: navigator.userAgent,
  });
}

/** Copy the diagnostics line; where the clipboard is refused, show it to select. */
async function copyDiagnostics() {
  const line = diagnostics();
  els.diagnostics.hidden = true;
  try {
    await navigator.clipboard.writeText(line);
    els.copyStatus.textContent = label(language, "copied");
  } catch (error) {
    console.warn("[aztts] clipboard refused", error);
    els.copyStatus.textContent = label(language, "copy_failed");
    els.diagnostics.textContent = line;
    els.diagnostics.hidden = false;
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
      precision = message.precision ?? null;
      fallbacks = message.fallbacks ?? [];
      modeVoice = requestedVoice;
      renderSlowNote();
      renderModeBanner();
      reviseDevice();
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
  // A status line that still holds a fixed message (the loading line, or
  // "cancelled") follows the switch instead of staying in the old language.
  const statusKey = ["loading", "status_cancelled", "result_ready"].find((key) =>
    Object.keys(LABELS).some((code) => els.status.textContent === label(code, key)));
  language = LABELS[next] ? next : DEFAULT_LANGUAGE;
  const t = (key) => label(language, key);
  if (statusKey) els.status.textContent = t(statusKey);

  document.documentElement.lang = language;
  document.title = t("title");
  els.title.textContent = t("title");
  els.subtitle.textContent = t("subtitle");
  els.languageLabel.textContent = t("language_label");
  els.themeLabel.textContent = t("theme_label");
  // The theme buttons are icons: the word is their accessible name and tooltip.
  for (const button of document.querySelectorAll("[data-theme-choice]")) {
    const name = t(`theme_${button.dataset.themeChoice}`);
    button.setAttribute("aria-label", name);
    button.title = name;
  }
  els.textLabel.textContent = t("text_label");
  els.text.placeholder = t("text_placeholder");
  els.speak.textContent = t("speak");
  els.cancel.textContent = t("cancel");
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
  els.copyDiagnostics.textContent = t("copy_diagnostics");
  els.copyStatus.textContent = "";
  renderModeBanner();
  if (lastResult) renderBadge(lastResult);
  device.setLanguage((key, values) => label(language, key, values));
  if (!els.textError.hidden) showTextError(true, { focus: false });
  els.censorNote.textContent = t("censor_note");
  els.phoneNote.hidden = !(PHONE && !forceGpu);
  els.phoneNoteText.textContent = t("phone_note");
  els.phoneWhy.textContent = t("phone_why");
  els.resultReadyText.textContent = t("result_ready_badge");
  els.skipLink.textContent = t("skip_to_text");
  els.waveEmpty.textContent = t("wave_empty");
  els.aboutLabel.textContent = t("about_label");
  els.footer.textContent = t("footer");

  for (const button of document.querySelectorAll("[data-language]")) {
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

// -- the waveform: js/wave-view.js ---------------------------------------------

const wave = createWaveView({ canvas: els.wave, box: els.waveBox, audio: els.audio });

// -- the finished clip: show where it is ------------------------------------------

// On a phone the result sits a screen or two below Speak, and people did not
// find it. When a clip is ready: a "Ready" chip, a pulse round the waveform,
// and a scroll to the result if it is off screen -- unless the visitor has
// started typing again, or the device dialog is open.
let typedSinceSpeak = false;
els.text.addEventListener("input", () => {
  if (busy) typedSinceSpeak = true;
});

function revealResult() {
  els.resultReady.hidden = false;
  els.waveBox.classList.remove("is-new");
  void els.waveBox.offsetWidth; // restarts the pulse for a second clip
  els.waveBox.classList.add("is-new");
  if (typedSinceSpeak || els.dialog.open) return;
  if (!needsReveal(els.result.getBoundingClientRect(), window.innerHeight)) return;
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  els.result.scrollIntoView({ behavior: still ? "auto" : "smooth", block: "start" });
}

els.waveBox.addEventListener("animationend", () => els.waveBox.classList.remove("is-new"));
els.phoneWhy.addEventListener("click", () => els.deviceDetails.click());

// -- synthesis ---------------------------------------------------------------

async function speak() {
  if (busy) return;
  const text = els.text.value.trim();
  if (!text) {
    showTextError(true);
    return;
  }
  showTextError(false);

  busy = true;
  els.speak.disabled = true;
  els.cancel.hidden = false;
  els.stats.textContent = "";
  els.resultReady.hidden = true;
  typedSinceSpeak = false;
  const requestedVoice = voice;
  const began = performance.now();
  const stopClock = startClock(began);
  // The statistics time synthesis alone, not the one-off download and setup.
  let synthesisBegan = null;
  let ready = false;
  try {
    const result = await synthesise(
      {
        voice: requestedVoice,
        text,
        options: {
          speed: Number(els.speed.value),
          variation: Number(els.variation.value),
          seed: Number(els.seed.value) || 0,
          normalize: requestedVoice === "az" && els.normalise.checked,
          maxWords: Number(els.maxWords.value),
          fullQuality,
        },
      },
      (message) => {
        if (message.stage === "chunk" && synthesisBegan === null) synthesisBegan = performance.now();
        showProgress(message, requestedVoice);
      },
    );
    const { waveform, chunks } = result;
    const elapsed = (performance.now() - (synthesisBegan ?? began)) / 1000;
    setProgress(1);
    backend = result.backend;
    threads = result.threads ?? 1;
    precision = result.precision ?? null;
    fallbacks = result.fallbacks ?? [];
    modeVoice = requestedVoice;

    const blob = toWav(waveform);
    if (els.audio.src) URL.revokeObjectURL(els.audio.src);
    const url = URL.createObjectURL(blob);
    els.audio.src = url;
    els.audio.hidden = false;
    wave.show(waveform);
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
      backend,
      precision,
      threads,
      voice: requestedVoice,
    };
    showStats(lastResult);
    renderModeBanner();
    renderBadge(lastResult);
    els.copyDiagnostics.hidden = false;
    els.copyStatus.textContent = "";
    // Said in the live status line, not by moving focus: a screen-reader user
    // stays by Speak and Cancel.
    els.status.textContent = label(language, "result_ready");
    // After the finally block: hiding Cancel and the bar there moves the page,
    // which would leave a scroll started now short of the result.
    ready = true;
  } catch (error) {
    if (error instanceof CancelledError) {
      els.status.textContent = label(language, "status_cancelled");
    } else if (error?.message === "empty") {
      console.error("[aztts] synthesis failed", error);
      showTextError(true);
      els.status.textContent = "";
    } else {
      console.error("[aztts] synthesis failed", error);
      els.status.textContent = String(error?.message ?? error);
      showError(error?.message);
    }
  } finally {
    stopClock();
    setProgress(undefined);
    busy = false;
    els.speak.disabled = false;
    // A hidden button cannot hold focus; hand it to the one that replaces it.
    const hadFocus = document.activeElement === els.cancel;
    els.cancel.hidden = true;
    if (hadFocus) els.speak.focus();
    if (ready) revealResult();
  }
}

// -- wiring ------------------------------------------------------------------

for (const button of document.querySelectorAll("[data-language]")) {
  button.addEventListener("click", () => applyLanguage(button.dataset.language));
}

for (const button of document.querySelectorAll("[data-voice]")) {
  button.addEventListener("click", () => applyVoice(button.dataset.voice));
}

for (const button of document.querySelectorAll("[data-theme-choice]")) {
  button.addEventListener("click", () => applyTheme(button.dataset.themeChoice));
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
els.cancel.addEventListener("click", cancelSynthesis);
document.addEventListener("keydown", (event) => {
  // Escape belongs to the device dialog while it is open.
  if (event.key !== "Escape" || !busy || document.getElementById("device-dialog")?.open) return;
  cancelSynthesis();
});
els.modeAction.addEventListener("click", () => {
  fullQuality = els.modeAction.dataset.action === "full";
  speak();
});
els.copyDiagnostics.addEventListener("click", copyDiagnostics);
els.text.addEventListener("input", () => {
  if (!els.textError.hidden && els.text.value.trim()) showTextError(false);
});
els.text.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) speak();
});

applyTheme(initialTheme(readStoredTheme(), location.search), { remember: false });
applyLanguage(DEFAULT_LANGUAGE);
els.status.textContent = label(language, "loading");
checkDevice();
