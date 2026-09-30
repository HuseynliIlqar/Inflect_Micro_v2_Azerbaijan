/**
 * The playground: wiring only. What each part of the page does lives in
 * js/page/, the logic it draws on in js/text/, js/engine/ and js/ui/, and the
 * synthesis itself runs in js/worker.js so the page never freezes.
 *
 * Nothing is sent anywhere. The phonemiser and the model are downloaded once,
 * cached on the device, and run on the visitor's machine -- which is the same
 * claim the command line version makes, kept honest on a web page.
 */

import { DEFAULT_LANGUAGE } from "./ui/i18n.js";
import { applyTheme, initialTheme, readStoredTheme } from "./ui/theme.js";
import { els } from "./page/dom.js";
import { state, t } from "./page/context.js";
import { checkDevice } from "./page/device.js";
import { applyLanguage, applyVoice } from "./page/language.js";
import { copyDiagnostics } from "./page/mode-view.js";
import { renderTextCheck } from "./page/text-view.js";
import { cancelSynthesis, speak } from "./page/speak.js";

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
  if (event.key !== "Escape" || !state.busy || els.dialog?.open) return;
  cancelSynthesis();
});
els.modeAction.addEventListener("click", () => {
  state.fullQuality = els.modeAction.dataset.action === "full";
  speak();
});
els.copyDiagnostics.addEventListener("click", copyDiagnostics);
els.phoneWhy.addEventListener("click", () => els.deviceDetails.click());
els.text.addEventListener("input", () => renderTextCheck());
els.text.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) speak();
});

applyTheme(initialTheme(readStoredTheme(), location.search), { remember: false });
applyLanguage(DEFAULT_LANGUAGE);
els.status.textContent = t("loading");
checkDevice();
