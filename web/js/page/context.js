/**
 * What every part of the page shares: facts about this visit that never
 * change, and the one object that holds what does.
 */

import { DEFAULT_LANGUAGE, label } from "../ui/i18n.js";
import { planOptionsFromQuery, thisIsPhone } from "../engine/backend-plan.js";
import { isAppleMobile } from "../ui/device-check.js";

// The page shipped on the Space's direct host; inside the huggingface.co
// iframe it cannot be cross-origin isolated, so it gets no wasm threads.
export const DIRECT_URL = "https://ilqarrrr-inflect-micro-v2-azerbaijan.static.hf.space/index.html";
export const IN_FRAME = window.top !== window.self;

// Tester switches on the page URL; see js/engine/backend-plan.js.
export const QUERY = planOptionsFromQuery(location.search);
// The worker applies the same rule; see js/engine/backend-plan.js.
export const PHONE = thisIsPhone();
// Every browser on an iPhone or iPad is WebKit: advice never names another one.
export const APPLE_MOBILE = isAppleMobile({
  userAgent: navigator.userAgent,
  maxTouchPoints: navigator.maxTouchPoints,
});
// The same thread count the worker will use (js/worker.js).
export const EXPECTED_THREADS = self.crossOriginIsolated
  ? Math.min(4, navigator.hardwareConcurrency || 1)
  : 1;

// The Azerbaijani model is this project's; the English one is
// owensong/Inflect-Micro-v2, the checkpoint it was adapted from. Each loads
// lazily inside the worker.
export const VOICES = ["az", "en"];

/**
 * The page's changing state. One object, so every module reads the same
 * values; only the modules in js/page/ write to it.
 */
export const state = {
  language: DEFAULT_LANGUAGE,
  voice: "az",
  lastResult: null,
  busy: false,
  backend: null,
  threads: 1,
  // The fallback chain's state as the worker last reported it.
  precision: null,
  fallbacks: [],
  modeVoice: "az",
  // "Full quality" skips the int8 step; `?precision=fp32` starts with it on.
  fullQuality: QUERY.fullQuality,
};

/** A label in the page's current language. */
export const t = (key, values) => label(state.language, key, values);
