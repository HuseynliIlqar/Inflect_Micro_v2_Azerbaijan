/**
 * The result's player: the round Play button, the waveform as a slider, the
 * clock, and the chunk being heard lit in the list below. The hidden <audio>
 * element does the playing; the orb follows its loudness.
 */

import { SAMPLE_RATE } from "../engine/audio.js";
import { createWaveView } from "../ui/wave-view.js";
import { chunkAt, formatTime, seekTime } from "../ui/playback.js";
import { levelAt } from "../ui/envelope.js";
import { els } from "./dom.js";
import { t } from "./context.js";
import { orb } from "./orb-view.js";

function createPlayer() {
  const audio = els.audio;
  const wave = createWaveView({ canvas: els.wave, box: els.waveBox, audio });
  const listeners = new Set();
  // { key, text, voice, url, samples, envelope, spans, chunks, seconds, stats }
  let clip = null;
  let lit = -2;
  // The slider's spoken value changes once a second, not on every timeupdate,
  // or a screen reader on the focused waveform chatters.
  let spokenSecond = -1;
  let readyTimer = 0;
  // Long enough for a polite live region to read "Ready" before it goes.
  const READY_HOLD_MS = 1500;

  const isPlaying = () => !audio.paused && !audio.ended;
  const duration = () => (audio.duration > 0 ? audio.duration : clip?.seconds ?? 0);
  const notify = () => listeners.forEach((listener) => listener());

  function renderChunks() {
    lit = -2;
    els.chunks.replaceChildren(...(clip?.chunks ?? []).map((chunk) => {
      const item = document.createElement("li");
      item.textContent = chunk;
      return item;
    }));
  }

  function highlight() {
    const heard = isPlaying() || audio.currentTime > 0;
    const index = clip && heard ? chunkAt(clip.spans, Math.floor(audio.currentTime * SAMPLE_RATE)) : -1;
    if (index === lit) return;
    lit = index;
    els.chunks.classList.toggle("is-following", index >= 0);
    [...els.chunks.children].forEach((item, i) => item.classList.toggle("is-current", i === index));
  }

  function renderClock() {
    const total = duration();
    const at = Math.min(audio.currentTime || 0, total);
    const clock = `${formatTime(at)} / ${formatTime(total)}`;
    els.time.textContent = clock;
    if (Math.floor(at) !== spokenSecond) {
      spokenSecond = Math.floor(at);
      els.wave.setAttribute("aria-valuemax", String(Math.round(total)));
      els.wave.setAttribute("aria-valuenow", String(spokenSecond));
      els.wave.setAttribute("aria-valuetext", clock);
    }
    const midway = clip && at > 0 && at < total - 0.05;
    els.orbHint.textContent = isPlaying() ? t("orb_playing", { time: clock })
      : midway ? t("orb_paused", { time: clock }) : t("orb_hint");
    highlight();
  }

  function renderButton() {
    const playing = isPlaying();
    els.play.dataset.state = playing ? "playing" : "paused";
    els.play.setAttribute("aria-label", t(playing ? "pause" : "play"));
  }

  const refresh = () => {
    renderButton();
    renderClock();
    notify();
  };

  for (const event of ["play", "pause", "ended", "seeked", "loadedmetadata"]) audio.addEventListener(event, refresh);
  audio.addEventListener("timeupdate", renderClock);
  audio.addEventListener("play", () => {
    // Once "Ready" has been read out, the line under the orb shows the clock.
    clearTimeout(readyTimer);
    readyTimer = setTimeout(() => {
      if (els.status.textContent === t("result_ready")) els.status.textContent = "";
    }, READY_HOLD_MS);
  });

  orb.setLevelSource(() => (clip && isPlaying() ? levelAt(clip.envelope, audio.currentTime) : 0));

  els.play.addEventListener("click", () => toggle());
  els.wave.addEventListener("keydown", (event) => {
    const next = seekTime(event.key, audio.currentTime, duration());
    if (next === null) return;
    event.preventDefault();
    audio.currentTime = next;
    renderClock();
  });

  function play(from) {
    if (!clip) return Promise.resolve();
    if (from !== undefined) audio.currentTime = from;
    else if (audio.ended || audio.currentTime >= duration() - 0.05) audio.currentTime = 0;
    // A browser may refuse to start sound without a fresh tap (iOS); the
    // Play button is still there, so a refusal is not an error.
    return audio.play().catch((error) => console.warn("[avaz] playback waits for a tap", error));
  }

  function toggle() {
    if (isPlaying()) audio.pause();
    else play();
  }

  return {
    get clip() {
      return clip;
    },
    get playing() {
      return isPlaying();
    },
    /** True when the loaded clip was made from exactly this request (js/page/request.js). */
    matches(key) {
      return Boolean(clip) && clip.key === key;
    },
    load(next, { autoplay = false } = {}) {
      audio.pause();
      clip = next;
      audio.src = next.url;
      audio.currentTime = 0;
      wave.show(next.samples);
      els.wave.tabIndex = 0;
      els.wave.removeAttribute("aria-disabled");
      spokenSecond = -1;
      els.play.disabled = false;
      renderChunks();
      refresh();
      if (autoplay) play(0);
    },
    toggle,
    relabel: refresh,
    onChange(listener) {
      listeners.add(listener);
    },
  };
}

export const player = createPlayer();
