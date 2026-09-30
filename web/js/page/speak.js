/** Speak: check the text, ask the worker, and show the clip where it can be found. */

import { CancelledError } from "../engine/engine.js";
import { toWav, SAMPLE_RATE } from "../engine/audio.js";
import { ERROR_KEYS } from "../text/text-check.js";
import { adviceKey, errorKind } from "../ui/device-check.js";
import { toast } from "../ui/notify.js";
import { createWaveView } from "../ui/wave-view.js";
import { needsReveal } from "../ui/reveal.js";
import { els } from "./dom.js";
import { APPLE_MOBILE, state, t } from "./context.js";
import { createWorkerClient } from "./worker-client.js";
import { renderTextCheck } from "./text-view.js";
import { setProgress, showProgress, startClock } from "./progress-view.js";
import { renderBadge, renderModeBanner, showStats } from "./mode-view.js";

const client = createWorkerClient({
  url: new URL("../worker.js", import.meta.url),
  workerError: () => new Error(t("error_worker")),
});

const wave = createWaveView({ canvas: els.wave, box: els.waveBox, audio: els.audio });

/** Stop the running request; see js/page/worker-client.js. */
export const cancelSynthesis = () => client.cancel();

/** An error as a toast that names the next step, not just the problem. */
function showError(message) {
  toast(els.toasts, {
    tone: "bad",
    title: t("error_title"),
    text: t(adviceKey(errorKind(message), { appleMobile: APPLE_MOBILE })),
    closeLabel: t("toast_close"),
  });
}

// On a phone the result sits a screen or two below Speak and people lost it: a
// Ready chip, a pulse, and a scroll to it -- unless typing, or the dialog is open.
let typedSinceSpeak = false;
els.text.addEventListener("input", () => {
  if (state.busy) typedSinceSpeak = true;
});
els.waveBox.addEventListener("animationend", () => els.waveBox.classList.remove("is-new"));

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

function readOptions(requestedVoice) {
  return {
    speed: Number(els.speed.value),
    variation: Number(els.variation.value),
    seed: Number(els.seed.value) || 0,
    normalize: requestedVoice === "az" && els.normalise.checked,
    maxWords: Number(els.maxWords.value),
    fullQuality: state.fullQuality,
  };
}

/** Put a finished clip on the page: player, waveform, chunks, statistics. */
function showResult(result, requestedVoice, elapsed) {
  const { waveform, chunks } = result;
  state.backend = result.backend;
  state.threads = result.threads ?? 1;
  state.precision = result.precision ?? null;
  state.fallbacks = result.fallbacks ?? [];
  state.modeVoice = requestedVoice;

  const blob = toWav(waveform);
  if (els.audio.src) URL.revokeObjectURL(els.audio.src);
  const url = URL.createObjectURL(blob);
  els.audio.src = url;
  els.audio.hidden = false;
  wave.show(waveform);
  els.download.href = url;
  els.download.hidden = false;
  els.chunks.textContent = chunks.map((chunk, index) => `[${index + 1}] ${chunk}`).join("\n");

  const seconds = waveform.length / SAMPLE_RATE;
  state.lastResult = {
    seconds,
    elapsed,
    realtime: elapsed > 0 ? seconds / elapsed : 0,
    chunks: chunks.length,
    backend: state.backend,
    precision: state.precision,
    threads: state.threads,
    voice: requestedVoice,
  };
  showStats(state.lastResult);
  renderModeBanner();
  renderBadge(state.lastResult);
  els.copyDiagnostics.hidden = false;
  els.copyStatus.textContent = "";
  // Said in the live status line, not by moving focus: readers stay by Speak.
  els.status.textContent = t("result_ready");
}

function showFailure(error) {
  if (error instanceof CancelledError) {
    els.status.textContent = t("status_cancelled");
  } else if (Object.hasOwn(ERROR_KEYS, error?.message)) {
    // The worker's own check, or a text that normalises to nothing.
    console.error("[aztts] synthesis refused", error);
    renderTextCheck({ error: error.message });
    els.status.textContent = "";
  } else {
    console.error("[aztts] synthesis failed", error);
    els.status.textContent = String(error?.message ?? error);
    showError(error?.message);
  }
}

export async function speak() {
  if (state.busy) return;
  const check = renderTextCheck({ insist: true });
  if (check.error) {
    els.text.focus();
    return;
  }

  state.busy = true;
  els.speak.disabled = true;
  els.cancel.hidden = false;
  els.stats.textContent = "";
  els.resultReady.hidden = true;
  typedSinceSpeak = false;
  const requestedVoice = state.voice;
  const began = performance.now();
  const stopClock = startClock(began);
  // The statistics time synthesis alone, not the one-off download and setup.
  let synthesisBegan = null;
  let ready = false;
  try {
    const result = await client.synthesise(
      { voice: requestedVoice, text: check.text, options: readOptions(requestedVoice) },
      (message) => {
        if (message.stage === "chunk" && synthesisBegan === null) synthesisBegan = performance.now();
        showProgress(message, requestedVoice);
      },
    );
    const elapsed = (performance.now() - (synthesisBegan ?? began)) / 1000;
    setProgress(1);
    showResult(result, requestedVoice, elapsed);
    // After the finally block: hiding Cancel and the bar there moves the page,
    // which would leave a scroll started now short of the result.
    ready = true;
  } catch (error) {
    showFailure(error);
  } finally {
    stopClock();
    setProgress(undefined);
    state.busy = false;
    els.speak.disabled = false;
    // A hidden button cannot hold focus; hand it to the one that replaces it.
    const hadFocus = document.activeElement === els.cancel;
    els.cancel.hidden = true;
    if (hadFocus) els.speak.focus();
    if (ready) revealResult();
  }
}
