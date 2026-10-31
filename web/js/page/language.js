/** Every label on the page in the chosen language, and the voice switch. */

import { DEFAULT_LANGUAGE, label, LABELS } from "../ui/i18n.js";
import { els } from "./dom.js";
import { VOICES, state, t } from "./context.js";
import { device } from "./device.js";
import { renderBadge, renderModeBanner, renderPhoneNote, renderSlowNote, showStats } from "./mode-view.js";
import { renderTextCheck } from "./text-view.js";
import { renderPreview } from "./preview-view.js";
import { renderThemeToggle } from "./theme-view.js";
import { renderSpeakIdle } from "./busy-view.js";
import { player } from "./player-view.js";

// Short labels on the chips; the sentence goes into the text box. Language
// data, like the examples in say.py -- see CLAUDE.md.
const EXAMPLES = {
  az: [
    { label: "Salam", text: "Salam, bu model tamamilə yerli maşında işləyir." },
    { label: "Payız", text: "Payız gəlmişdi və şəhərin küçələri saralmış yarpaqlarla örtülmüşdü." },
    { label: "Sual", text: "Sən bu kitabı oxumusan? Mənə çox maraqlı gəldi." },
    { label: "Rəqəmlər", text: "II Dünya müharibəsi 01/09/1939 tarixində başladı və 25% artım oldu." },
  ],
  en: [
    { label: "Hello", text: "Hello, this model runs completely offline on your machine." },
    { label: "Autumn", text: "Autumn had come, and the streets were covered with yellow leaves." },
    { label: "Question", text: "Have you read this book? I found it very interesting." },
  ],
};

/** The sentence the page opens with: the one that shows the text layer at work. */
export const FIRST_TEXT = EXAMPLES.az[3].text;

// The About page is static HTML, one file per language, so a crawler that runs
// no script still reads it; its links follow the interface language.
const ABOUT_PAGES = { az: "./about.html", en: "./about-en.html" };

const isExample = (text) => Object.values(EXAMPLES).some((list) => list.some((e) => e.text === text));

function markPressed(selector, dataKey, value) {
  for (const button of document.querySelectorAll(selector)) {
    const active = button.dataset[dataKey] === value;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  }
}

function renderExamples() {
  els.examples.replaceChildren();
  for (const example of EXAMPLES[state.voice]) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "example";
    button.textContent = example.label;
    button.title = example.text;
    button.setAttribute("aria-pressed", String(els.text.value === example.text));
    button.addEventListener("click", () => {
      els.text.value = example.text;
      els.text.dispatchEvent(new Event("input"));
      els.text.focus();
    });
    els.examples.appendChild(button);
  }
}

/** Mark the example chip whose sentence is in the box, if any. */
export function markExample() {
  for (const button of els.examples.children) {
    button.setAttribute("aria-pressed", String(els.text.value === button.title));
  }
}

/**
 * The Azerbaijani text layer does not apply to the English checkpoint --
 * `normalizeAz` would rewrite numbers into Azerbaijani words -- so its control
 * is disabled rather than quietly ignored.
 */
export function applyVoice(next) {
  const previous = state.voice;
  state.voice = VOICES.includes(next) ? next : "az";
  const azerbaijani = state.voice === "az";
  markPressed("[data-voice]", "voice", state.voice);
  // The sliding pill behind the two options reads this (css/page/composer.css).
  document.querySelector(".voice-switch").dataset.voice = state.voice;
  // An untouched example follows the voice; the visitor's own text stays.
  if (previous !== state.voice && (!els.text.value.trim() || isExample(els.text.value))) {
    els.text.value = EXAMPLES[state.voice][0].text;
  }

  els.normalise.disabled = !azerbaijani;
  els.normaliseInfo.textContent = t("normalise_info") + (azerbaijani ? "" : " " + t("az_only"));
  els.voiceInfo.hidden = azerbaijani;
  els.voiceInfo.textContent = azerbaijani ? "" : t("voice_info");
  // Whether the text suits the voice is part of the check.
  renderTextCheck();
  renderPreview();
  renderExamples();
}

/** Text of every `data-l10n` element, and the attributes marked the same way. */
function applyLabels() {
  for (const element of document.querySelectorAll("[data-l10n]")) element.textContent = t(element.dataset.l10n);
  for (const element of document.querySelectorAll("[data-l10n-aria]")) {
    element.setAttribute("aria-label", t(element.dataset.l10nAria));
  }
}

export function applyLanguage(next) {
  // A status line that still holds a fixed message (the loading line, or
  // "cancelled") follows the switch instead of staying in the old language.
  const statusKey = ["loading", "status_cancelled", "result_ready"].find((key) =>
    Object.keys(LABELS).some((code) => els.status.textContent === label(code, key)));
  state.language = LABELS[next] ? next : DEFAULT_LANGUAGE;
  if (statusKey) els.status.textContent = t(statusKey);

  document.documentElement.lang = state.language;
  // The tab and the search result say what the page does; the <h1> is the
  // hero line, a label of its own.
  document.title = t("page_title");
  applyLabels();
  els.text.placeholder = t("text_placeholder");
  for (const link of els.aboutLinks) link.href = ABOUT_PAGES[state.language] ?? ABOUT_PAGES.az;
  for (const button of document.querySelectorAll("[data-voice]")) {
    button.textContent = t(`voice_option_${button.dataset.voice}`);
  }
  els.waveEmpty.textContent = t("wave_empty");
  renderThemeToggle();
  renderSpeakIdle();
  renderSlowNote();
  els.progress.setAttribute("aria-label", t("progress_label"));
  els.copyDiagnostics.textContent = t("copy_diagnostics");
  els.copyStatus.textContent = "";
  renderModeBanner();
  if (state.lastResult) renderBadge(state.lastResult);
  device.setLanguage(t);
  renderPhoneNote();
  player.relabel();

  markPressed("[data-language]", "language", state.language);
  applyVoice(state.voice);
  if (state.lastResult) showStats(state.lastResult);
}
