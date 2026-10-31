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
import { remeasure, startDeviceCheck } from "./page/calibration.js";
import { FIRST_TEXT, applyLanguage, applyVoice, markExample } from "./page/language.js";
import { copyDiagnostics } from "./page/mode-view.js";
import { renderTextCheck } from "./page/text-view.js";
import { renderPreview } from "./page/preview-view.js";
import { initControls } from "./page/controls-view.js";
import { renderOrb } from "./page/busy-view.js";
import { toggleTheme } from "./page/theme-view.js";
import { onPick } from "./page/history-view.js";
import { orb } from "./page/orb-view.js";
import { cancelSynthesis, speak } from "./page/speak.js";
import { onKeydown, onOrb, onPickClip, onRegen } from "./page/actions.js";

for (const button of document.querySelectorAll("[data-language]")) {
  button.addEventListener("click", () => applyLanguage(button.dataset.language));
}
for (const button of document.querySelectorAll("[data-voice]")) {
  button.addEventListener("click", () => {
    applyVoice(button.dataset.voice);
    renderOrb();
  });
}
els.themeToggle.addEventListener("click", toggleTheme);
initControls();

els.orb.addEventListener("click", onOrb);
els.speak.addEventListener("click", speak);
els.cancel.addEventListener("click", cancelSynthesis);
els.regen.addEventListener("click", onRegen);
onPick(onPickClip);
document.addEventListener("keydown", onKeydown);

els.text.addEventListener("input", () => {
  renderTextCheck();
  renderPreview();
  markExample();
  renderOrb();
});
els.normalise.addEventListener("change", () => {
  renderPreview();
  renderOrb();
});
els.clearText.addEventListener("click", () => {
  els.text.value = "";
  els.text.dispatchEvent(new Event("input"));
  els.text.focus();
});
els.modeAction.addEventListener("click", () => {
  state.fullQuality = els.modeAction.dataset.action === "full";
  speak();
});
els.copyDiagnostics.addEventListener("click", copyDiagnostics);
els.dialogRemeasure.addEventListener("click", () => {
  els.dialog.close();
  remeasure();
});
els.phoneWhy.addEventListener("click", () => els.deviceDetails.click());

applyTheme(initialTheme(readStoredTheme(), location.search), { remember: false });
els.text.value = FIRST_TEXT;
applyLanguage(DEFAULT_LANGUAGE);
renderPreview();
orb.start();
els.status.textContent = t("loading");
startDeviceCheck();
