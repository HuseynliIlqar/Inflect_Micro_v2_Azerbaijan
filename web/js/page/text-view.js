/** The text: its limit and what is wrong with it (js/text/text-check.js). */

import { checkText, ERROR_KEYS } from "../text/text-check.js";
import { els } from "./dom.js";
import { state, t } from "./context.js";

// The error beside the field, if one is shown: a key of ERROR_KEYS.
let shownTextError = null;

/** "123 / 500" under the field, with a spoken form for screen readers. */
function renderCounter(check) {
  const visible = document.createElement("span");
  visible.setAttribute("aria-hidden", "true");
  visible.textContent = t("char_count", { count: check.count, max: check.max });
  const spoken = document.createElement("span");
  spoken.className = "visually-hidden";
  spoken.textContent = t("char_count_label", { count: check.count, max: check.max });
  els.textCount.replaceChildren(visible, spoken);
  const over = check.error === "too_long";
  els.textCount.classList.toggle("is-over", over);
  els.textCount.classList.toggle("is-near", check.near && !over);
}

/**
 * Check the text and say what is wrong next to the field, not in a popup.
 * Too long is said as soon as it happens; empty and nothing-to-say only once
 * Speak is pressed (`insist`), so an untouched field is not shouted at.
 * An error already shown stays until the text no longer has it.
 */
export function renderTextCheck({ insist = false, error: forced = null } = {}) {
  const check = checkText(els.text.value, { voice: state.voice });
  const error = forced ?? check.error;
  renderCounter(check);

  if (error && (insist || forced || error === "too_long" || error === shownTextError)) {
    shownTextError = error;
  } else {
    shownTextError = null;
  }
  els.textError.hidden = !shownTextError;
  if (shownTextError) {
    els.textError.textContent = ERROR_KEYS[shownTextError]
      .map((key) => t(key, { count: check.count, max: check.max }))
      .join(" ");
  }

  const warnings = shownTextError ? [] : check.warnings;
  els.textWarning.hidden = warnings.length === 0;
  els.textWarning.textContent = warnings.map((key) => t(key)).join(" ");

  const described = ["text-count", shownTextError && "text-error", warnings.length && "text-warning"];
  els.text.setAttribute("aria-describedby", described.filter(Boolean).join(" "));
  if (shownTextError) els.text.setAttribute("aria-invalid", "true");
  else els.text.removeAttribute("aria-invalid");
  return { ...check, error };
}
