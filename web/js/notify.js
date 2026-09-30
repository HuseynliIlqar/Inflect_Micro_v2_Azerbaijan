/**
 * Toasts: short messages at the edge of the screen, for things that happen
 * while the visitor is looking elsewhere -- the device check, a fallback in
 * the middle of synthesis, an error with its next step.
 *
 * Errors stay until closed. Everything else leaves after a while, but not
 * while the pointer or keyboard focus is on it.
 */

const TOAST_MS = 7000;
const MAX_TOASTS = 3;

/**
 * @param {HTMLElement} region  the `aria-live` container
 * @param {{ tone: "good"|"warn"|"bad", title: string, text?: string, closeLabel: string }} message
 * @returns {() => void} dismisses the toast
 */
export function toast(region, { tone, title, text = "", closeLabel }) {
  const item = document.createElement("div");
  item.className = "toast";
  item.dataset.tone = tone;
  item.setAttribute("role", tone === "bad" ? "alert" : "status");

  const body = document.createElement("div");
  body.className = "toast-body";
  const heading = document.createElement("p");
  heading.className = "toast-title";
  heading.textContent = title;
  body.append(heading);
  if (text) {
    const detail = document.createElement("p");
    detail.className = "toast-text";
    detail.textContent = text;
    body.append(detail);
  }

  const close = document.createElement("button");
  close.type = "button";
  close.className = "toast-close";
  close.setAttribute("aria-label", closeLabel);
  close.textContent = "×";

  let timer = null;
  const dismiss = () => {
    clearTimeout(timer);
    item.remove();
  };
  const arm = () => {
    if (tone !== "bad") timer = setTimeout(dismiss, TOAST_MS);
  };
  close.addEventListener("click", dismiss);
  for (const event of ["mouseenter", "focusin"]) item.addEventListener(event, () => clearTimeout(timer));
  for (const event of ["mouseleave", "focusout"]) item.addEventListener(event, arm);

  item.append(body, close);
  region.append(item);
  while (region.children.length > MAX_TOASTS) region.firstElementChild.remove();
  arm();
  return dismiss;
}
