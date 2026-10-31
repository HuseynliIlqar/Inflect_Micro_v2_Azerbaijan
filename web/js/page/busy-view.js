/**
 * The Speak button and the orb: what they say and show before, during and
 * just after a request -- "Speak", "Preparing 42%", then "Ready" for a moment.
 */

import { els } from "./dom.js";
import { state, t } from "./context.js";
import { orb } from "./orb-view.js";
import { player } from "./player-view.js";
import { signature } from "./request.js";

const DONE_MS = 1500;
const APPLE = /Mac|iPhone|iPad/u.test(navigator.userAgent);
let doneTimer = 0;

/** The orb's icon, name and motion, from the state of the page and the player. */
export function renderOrb() {
  const playing = player.playing;
  els.orb.dataset.icon = state.busy ? "stop" : playing ? "pause" : "play";
  const action = state.busy ? "cancel" : playing ? "pause"
    : player.matches(signature()) ? "play" : "speak";
  els.orb.setAttribute("aria-label", t(action));
  els.orb.disabled = state.checking;
  orb.setMode(state.checking ? "check" : state.busy ? "busy" : playing ? "playing" : "idle");
}

/** At rest: "Speak", and the shortcut for a keyboard. */
export function renderSpeakIdle() {
  els.speakKbd.textContent = APPLE ? "⌘ ↵" : "Ctrl ↵";
  if (!state.busy && !els.speak.classList.contains("is-done")) {
    els.speak.classList.remove("is-busy");
    els.speak.style.removeProperty("--fill");
    els.speakLabel.textContent = t("speak");
  }
  renderOrb();
}

/** During a request: how far it has got, or that it is waiting on something unsized. */
export function renderSpeakBusy(fraction) {
  els.speak.classList.add("is-busy");
  els.speak.classList.remove("is-done");
  const known = typeof fraction === "number";
  els.speakLabel.textContent = known
    ? t("speak_preparing", { percent: Math.round(fraction * 100) })
    : t("speak_wait");
  els.speak.style.setProperty("--fill", known ? `${Math.round(fraction * 100)}%` : "0%");
}

/** A clip landed: "Ready" on the button for a moment, then back to Speak. */
export function flashDone() {
  clearTimeout(doneTimer);
  els.speak.classList.remove("is-busy");
  els.speak.classList.add("is-done");
  els.speak.style.removeProperty("--fill");
  els.speakLabel.textContent = t("speak_done");
  doneTimer = setTimeout(() => {
    els.speak.classList.remove("is-done");
    renderSpeakIdle();
  }, DONE_MS);
}

player.onChange(renderOrb);
