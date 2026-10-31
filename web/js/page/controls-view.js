/**
 * The composer's chips -- speed, liveliness, seed, more -- and the one small
 * panel each opens below the text. One panel at a time; Escape closes it.
 */

import { els } from "./dom.js";
import { renderOrb } from "./busy-view.js";

const SEED_RANGE = 10000;
const panelOf = (chip) => document.getElementById(chip.getAttribute("aria-controls"));

/** The values shown on the chips, from the controls inside the panels. */
export function renderChipValues() {
  els.speedValue.textContent = `${Number(els.speed.value).toFixed(2)}×`;
  els.variationValue.textContent = Number(els.variation.value).toFixed(2);
  els.seedValue.textContent = String(Number(els.seed.value) || 0);
  els.maxWordsValue.textContent = els.maxWords.value;
}

function setOpen(open) {
  for (const chip of els.chips) {
    const isOpen = chip === open;
    chip.setAttribute("aria-expanded", String(isOpen));
    chip.classList.toggle("is-open", isOpen);
    panelOf(chip).hidden = !isOpen;
  }
}

/** Close whichever panel is open; true if one was. Focus inside it returns to its chip. */
export function closePanels() {
  const open = els.chips.find((chip) => chip.getAttribute("aria-expanded") === "true");
  const focusInside = open && panelOf(open).contains(document.activeElement);
  setOpen(null);
  if (focusInside) open.focus();
  return Boolean(open);
}

/** Put a clip's settings back on the controls (a pick from this session). */
export function applyOptions(options) {
  if (!options) return;
  els.speed.value = String(options.speed);
  els.variation.value = String(options.variation);
  els.seed.value = String(options.seed);
  els.maxWords.value = String(options.maxWords);
  if (!els.normalise.disabled) els.normalise.checked = options.normalize;
  renderChipValues();
}

export function initControls() {
  for (const chip of els.chips) {
    chip.addEventListener("click", (event) => {
      const opening = chip.getAttribute("aria-expanded") !== "true";
      setOpen(opening ? chip : null);
      // From the keyboard (detail 0), move into the panel; a tap does not, so
      // a phone does not raise its keyboard for the seed field.
      if (opening && event.detail === 0) panelOf(chip).querySelector("input, button")?.focus();
    });
  }
  for (const input of [els.speed, els.variation, els.seed, els.maxWords]) {
    input.addEventListener("input", () => {
      renderChipValues();
      // A new setting is a new request: the orb offers Speak, not the old clip.
      renderOrb();
    });
  }
  els.shuffle.addEventListener("click", () => {
    els.seed.value = String(Math.floor(Math.random() * SEED_RANGE));
    els.seed.dispatchEvent(new Event("input"));
  });
  renderChipValues();
}
