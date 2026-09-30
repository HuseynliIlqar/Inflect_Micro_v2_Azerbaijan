/** Every label on the page in the chosen language, and the voice switch. */

import { DEFAULT_LANGUAGE, label, LABELS } from "../ui/i18n.js";
import { els } from "./dom.js";
import { PHONE, QUERY, VOICES, state, t } from "./context.js";
import { device } from "./device.js";
import { renderBadge, renderModeBanner, renderSlowNote, showStats } from "./mode-view.js";
import { renderTextCheck } from "./text-view.js";

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

// Element -> label key, for every element whose whole text is one label.
const TEXT_LABELS = [
  ["title", "title"], ["subtitle", "subtitle"], ["languageLabel", "language_label"],
  ["themeLabel", "theme_label"], ["textLabel", "text_label"], ["speak", "speak"],
  ["cancel", "cancel"], ["examplesLabel", "examples_label"], ["audioLabel", "audio_label"],
  ["download", "download"], ["parameters", "parameters"], ["speedLabel", "speed_label"],
  ["speedInfo", "speed_info"], ["variationLabel", "variation_label"],
  ["variationInfo", "variation_info"], ["seedLabel", "seed_label"], ["seedInfo", "seed_info"],
  ["shuffle", "random_seed"], ["normaliseLabel", "normalise_label"],
  ["normaliseInfo", "normalise_info"], ["advancedLabel", "advanced"],
  ["maxWordsLabel", "max_words_label"], ["maxWordsInfo", "max_words_info"],
  ["voiceLabel", "voice_label"], ["voiceInfo", "voice_info"], ["chunksLabel", "chunks_label"],
  ["chunksInfo", "chunks_info"], ["offlineNote", "offline_note"], ["speedNote", "speed_note"],
  ["copyDiagnostics", "copy_diagnostics"], ["censorNote", "censor_note"],
  ["phoneNoteText", "phone_note"], ["phoneWhy", "phone_why"],
  ["resultReadyText", "result_ready_badge"], ["skipLink", "skip_to_text"],
  ["waveEmpty", "wave_empty"], ["aboutLabel", "about_label"], ["footer", "footer"],
  ["feedbackLink", "feedback_link"],
];

function markPressed(selector, dataKey, value) {
  for (const button of document.querySelectorAll(selector)) {
    const active = button.dataset[dataKey] === value;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  }
}

function renderExamples() {
  els.examples.replaceChildren();
  for (const sentence of EXAMPLES[state.voice]) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "example";
    button.textContent = sentence;
    button.addEventListener("click", () => {
      els.text.value = sentence;
      renderTextCheck();
      els.text.focus();
    });
    els.examples.appendChild(button);
  }
}

/**
 * The Azerbaijani text layer does not apply to the English checkpoint --
 * `normalizeAz` would rewrite numbers into Azerbaijani words -- so its control
 * is disabled rather than quietly ignored.
 */
export function applyVoice(next) {
  state.voice = VOICES.includes(next) ? next : "az";
  const azerbaijani = state.voice === "az";
  markPressed("[data-voice]", "voice", state.voice);

  els.normalise.disabled = !azerbaijani;
  // Whether the text suits the voice is part of the check.
  renderTextCheck();
  els.normaliseInfo.textContent = t("normalise_info") + (azerbaijani ? "" : " " + t("az_only"));
  renderExamples();
}

export function applyLanguage(next) {
  // A status line that still holds a fixed message (the loading line, or
  // "cancelled") follows the switch instead of staying in the old language.
  const statusKey = ["loading", "status_cancelled", "result_ready"].find((key) =>
    Object.keys(LABELS).some((code) => els.status.textContent === label(code, key)));
  state.language = LABELS[next] ? next : DEFAULT_LANGUAGE;
  if (statusKey) els.status.textContent = t(statusKey);

  document.documentElement.lang = state.language;
  document.title = t("title");
  for (const [element, key] of TEXT_LABELS) els[element].textContent = t(key);
  els.text.placeholder = t("text_placeholder");
  // The theme buttons are icons: the word is their accessible name and tooltip.
  for (const button of document.querySelectorAll("[data-theme-choice]")) {
    const name = t(`theme_${button.dataset.themeChoice}`);
    button.setAttribute("aria-label", name);
    button.title = name;
  }
  for (const button of document.querySelectorAll("[data-voice]")) {
    button.textContent = t(`voice_${button.dataset.voice}`);
  }
  renderSlowNote();
  els.progress.setAttribute("aria-label", t("progress_label"));
  els.copyStatus.textContent = "";
  renderModeBanner();
  if (state.lastResult) renderBadge(state.lastResult);
  device.setLanguage(t);
  renderTextCheck();
  els.phoneNote.hidden = !(PHONE && !QUERY.forceGpu);

  markPressed("[data-language]", "language", state.language);
  applyVoice(state.voice);
  if (state.lastResult) showStats(state.lastResult);
}
