/**
 * "This session": the clips made in this visit, newest first, one tap to hear
 * one again. Nothing is stored: closing the tab forgets them, and a clip that
 * falls off the list gives its memory back (its object URL is revoked).
 */

import { addToHistory } from "../ui/history.js";
import { formatTime } from "../ui/playback.js";
import { els } from "./dom.js";
import { player } from "./player-view.js";

let clips = [];
let pick = () => {};

function render() {
  els.history.hidden = clips.length < 2;
  els.historyList.replaceChildren(...clips.map((clip) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "history-item";
    button.dataset.url = clip.url;
    const text = document.createElement("span");
    text.className = "history-text";
    text.textContent = clip.text;
    const meta = document.createElement("span");
    meta.className = "history-meta";
    meta.translate = false;
    meta.textContent = `${clip.voice.toUpperCase()} · ${formatTime(clip.seconds)}`;
    button.append(text, meta);
    button.addEventListener("click", () => pick(clip));
    const item = document.createElement("li");
    item.append(button);
    return item;
  }));
  markCurrent();
}

/**
 * The clip in the player is marked, for the eye and for a screen reader --
 * on the buttons already there, so a focused one keeps its focus.
 */
function markCurrent() {
  for (const button of els.historyList.querySelectorAll(".history-item")) {
    if (button.dataset.url === player.clip?.url) button.setAttribute("aria-current", "true");
    else button.removeAttribute("aria-current");
  }
}

/** Add a finished clip; whatever falls off the list is released. */
export function remember(clip) {
  const { list, dropped } = addToHistory(clips, clip);
  clips = list;
  for (const old of dropped) {
    if (old.url !== player.clip?.url) URL.revokeObjectURL(old.url);
  }
  render();
}

/** Whether the session list still holds the clip behind this object URL. */
export function isRemembered(url) {
  return clips.some((clip) => clip.url === url);
}

// A clip picked again from the list becomes the current one.
player.onChange(markCurrent);

/** What a tap on an entry does: the page wiring decides (js/page/actions.js). */
export function onPick(handler) {
  pick = handler;
}
