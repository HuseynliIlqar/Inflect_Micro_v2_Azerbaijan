/**
 * The one theme button in the top bar: it shows where a click goes (a moon on
 * a light page, a sun on a dark one) and says so to a screen reader.
 */

import { applyTheme, toggledTheme } from "../ui/theme.js";
import { els } from "./dom.js";
import { t } from "./context.js";

const darkDevice = matchMedia("(prefers-color-scheme: dark)");
const chosen = () => document.documentElement.getAttribute("data-theme") ?? "auto";

export function renderThemeToggle() {
  const next = toggledTheme(chosen(), darkDevice.matches);
  els.themeToggle.dataset.next = next;
  els.themeToggle.setAttribute("aria-label", t(next === "dark" ? "theme_to_dark" : "theme_to_light"));
}

export function toggleTheme() {
  applyTheme(toggledTheme(chosen(), darkDevice.matches));
  renderThemeToggle();
}

// In "auto" the device decides; follow it when it flips.
darkDevice.addEventListener("change", renderThemeToggle);
