/** Speak: check the text, ask the worker, and put the clip where it can be heard. */

import { CancelledError } from "../engine/engine.js";
import { toWav, SAMPLE_RATE } from "../engine/audio.js";
import { ERROR_KEYS } from "../text/text-check.js";
import { adviceKey, errorKind } from "../ui/device-check.js";
import { envelope } from "../ui/envelope.js";
import { toast } from "../ui/notify.js";
import { needsReveal } from "../ui/reveal.js";
import { els } from "./dom.js";
import { APPLE_MOBILE, state, t } from "./context.js";
import { client } from "./client.js";
import { renderTextCheck } from "./text-view.js";
import { setProgress, showProgress, startClock } from "./progress-view.js";
import { renderBadge, renderModeBanner, showStats } from "./mode-view.js";
import { flashDone, renderOrb, renderSpeakBusy, renderSpeakIdle } from "./busy-view.js";
import { player } from "./player-view.js";
import { isRemembered, remember } from "./history-view.js";
import { readOptions, signature } from "./request.js";

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

/**
 * Show a clip: in the player, behind the download button, in the statistics.
 * Used for a fresh clip and for one picked again from this session's list.
 */
export function presentClip(clip, { autoplay = true } = {}) {
  const previous = player.clip;
  player.load(clip, { autoplay });
  // The clip it replaced gives its memory back, unless the session list keeps it.
  if (previous && previous.url !== clip.url && !isRemembered(previous.url)) {
    URL.revokeObjectURL(previous.url);
  }
  els.download.href = clip.url;
  els.download.hidden = false;
  els.regen.disabled = false;
  state.lastResult = clip.stats;
  showStats(clip.stats);
  renderBadge(clip.stats);
  els.copyDiagnostics.hidden = false;
  els.copyStatus.textContent = "";
  els.result.hidden = false;
  renderOrb();
}

/** A finished request as a clip: the audio, its loudness, where each chunk sits. */
function makeClip(result, { key, request, text, voice, elapsed }) {
  const { waveform, chunks } = result;
  state.backend = result.backend;
  state.threads = result.threads ?? 1;
  state.precision = result.precision ?? null;
  state.fallbacks = result.fallbacks ?? [];
  state.modeVoice = voice;
  const seconds = waveform.length / SAMPLE_RATE;
  return {
    id: Date.now(),
    key,
    text,
    voice,
    url: URL.createObjectURL(toWav(waveform)),
    samples: waveform,
    envelope: envelope(waveform, SAMPLE_RATE),
    spans: result.chunkSpans ?? [],
    chunks,
    seconds,
    // The settings it was made with, so a pick from this session restores them.
    options: request,
    stats: {
      seconds,
      elapsed,
      realtime: elapsed > 0 ? seconds / elapsed : 0,
      chunks: chunks.length,
      backend: state.backend,
      precision: state.precision,
      threads: state.threads,
      voice,
    },
  };
}

function showFailure(error) {
  if (error instanceof CancelledError) {
    els.status.textContent = t("status_cancelled");
  } else if (Object.hasOwn(ERROR_KEYS, error?.message)) {
    // The worker's own check, or a text that normalises to nothing.
    console.error("[avaz] synthesis refused", error);
    renderTextCheck({ error: error.message });
    els.status.textContent = "";
  } else {
    console.error("[avaz] synthesis failed", error);
    els.status.textContent = String(error?.message ?? error);
    showError(error?.message);
  }
}

function setBusy(on) {
  state.busy = on;
  els.speak.disabled = on;
  els.regen.disabled = on || !player.clip;
  // The voice of a running request is fixed; the switch waits for it.
  for (const option of document.querySelectorAll("[data-voice]")) option.disabled = on;
  // A hidden button cannot hold focus; hand it to the one that replaces it.
  const hadFocus = !on && document.activeElement === els.cancel;
  els.cancel.hidden = !on;
  if (hadFocus) els.speak.focus();
  renderOrb();
}

export async function speak() {
  if (state.busy || state.checking) return;
  const check = renderTextCheck({ insist: true });
  if (check.error) {
    els.text.focus();
    return;
  }

  const text = els.text.value;
  const voice = state.voice;
  // Taken now: the knobs may move while the request runs.
  const key = signature(text, voice);
  const request = readOptions(voice);
  setBusy(true);
  renderSpeakBusy(null);
  els.stats.textContent = "";
  els.resultReady.hidden = true;
  els.result.hidden = false;
  typedSinceSpeak = false;
  const began = performance.now();
  const stopClock = startClock(began);
  // The statistics time synthesis alone, not the one-off download and setup.
  let synthesisBegan = null;
  let ready = false;
  try {
    const result = await client.synthesise(
      { voice, text: check.text, options: request },
      (message) => {
        if (message.stage === "chunk" && synthesisBegan === null) synthesisBegan = performance.now();
        showProgress(message, voice);
      },
    );
    const elapsed = (performance.now() - (synthesisBegan ?? began)) / 1000;
    setProgress(1);
    const clip = makeClip(result, { key, request, text, voice, elapsed });
    renderModeBanner();
    remember(clip);
    // Said in the live status line, not by moving focus: readers stay by Speak.
    els.status.textContent = t("result_ready");
    presentClip(clip);
    // After the finally block: hiding Cancel and the bar there moves the page,
    // which would leave a scroll started now short of the result.
    ready = true;
  } catch (error) {
    showFailure(error);
  } finally {
    stopClock();
    setProgress(undefined);
    setBusy(false);
    if (ready) {
      flashDone();
      revealResult();
    } else {
      renderSpeakIdle();
    }
  }
}

/** The clip's sentence again with the next seed: another take, kept beside it. */
export function speakAgain() {
  if (state.busy || state.checking || !player.clip) return undefined;
  els.seed.value = String((Number(els.seed.value) || 0) + 1);
  els.seed.dispatchEvent(new Event("input"));
  if (els.text.value !== player.clip.text) {
    els.text.value = player.clip.text;
    els.text.dispatchEvent(new Event("input"));
  }
  return speak();
}
