/**
 * What a Speak request asks for, read from the page's controls -- and its
 * signature, so a clip knows whether it still says what the controls say.
 * The orb plays a clip only while the signatures match; change a knob and it
 * offers to speak again instead of replaying a stale take.
 */

import { els } from "./dom.js";
import { state } from "./context.js";

export function readOptions(voice = state.voice) {
  return {
    speed: Number(els.speed.value),
    variation: Number(els.variation.value),
    seed: Number(els.seed.value) || 0,
    normalize: voice === "az" && els.normalise.checked,
    maxWords: Number(els.maxWords.value),
    fullQuality: state.fullQuality,
    // The device check's order of backends; the worker guesses without it.
    plan: state.calibration?.plan ?? null,
  };
}

/** Text, voice and every setting that changes the sound, as one string. */
export function signature(text = els.text.value, voice = state.voice) {
  const { speed, variation, seed, normalize, maxWords, fullQuality } = readOptions(voice);
  return JSON.stringify([voice, text, speed, variation, seed, normalize, maxWords, fullQuality]);
}
