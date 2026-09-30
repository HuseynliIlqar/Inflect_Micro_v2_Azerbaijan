/**
 * Light, dark, or whatever the device prefers.
 *
 * The choice lives on `<html data-theme>`; `styles.css` reads it. "auto" is the
 * absence of the attribute, so the stylesheet's `prefers-color-scheme` rule
 * decides. An inline script in `index.html` applies the saved choice before the
 * first paint, so a dark page never flashes white while this module loads.
 */

export const THEMES = ["auto", "light", "dark"];
export const STORAGE_KEY = "theme";

// The browser chrome colour for each explicit theme; mirrors `--paper`.
const THEME_COLORS = { light: "#fbfbfa", dark: "#121417" };

/**
 * The theme to start with: a saved choice wins, then Hugging Face's
 * `?__theme=` hint (the Hub passes its own light/dark setting to Spaces),
 * then "auto".
 */
export function initialTheme(stored, search = "") {
  if (THEMES.includes(stored)) return stored;
  const hinted = new URLSearchParams(search).get("__theme");
  if (hinted === "light" || hinted === "dark") return hinted;
  return "auto";
}

export function readStoredTheme() {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch {
    return null; // Storage blocked (private window, sandboxed iframe).
  }
}

function storeTheme(theme) {
  try {
    if (theme === "auto") localStorage.removeItem(STORAGE_KEY);
    else localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    // Not remembered across visits; the page still switches.
  }
}

export function applyTheme(theme, { remember = true } = {}) {
  const next = THEMES.includes(theme) ? theme : "auto";
  const root = document.documentElement;
  if (next === "auto") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", next);

  for (const meta of document.querySelectorAll('meta[name="theme-color"]')) {
    // Each tag keeps its own media query in "auto"; an explicit theme overrides both.
    meta.content = next === "auto" ? meta.dataset.color : THEME_COLORS[next];
  }
  for (const button of document.querySelectorAll("[data-theme-choice]")) {
    const active = button.dataset.themeChoice === next;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  }
  if (remember) storeTheme(next);
  return next;
}
