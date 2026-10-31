/**
 * What the page's controls do when used: the orb, Another take, a pick from
 * this session's list, the keyboard. js/app.js only connects them.
 */

import { els } from "./dom.js";
import { state } from "./context.js";
import { applyVoice } from "./language.js";
import { applyOptions, closePanels } from "./controls-view.js";
import { player } from "./player-view.js";
import { signature } from "./request.js";
import { cancelSynthesis, presentClip, speak, speakAgain } from "./speak.js";

/** The orb: stop a request, play or pause the clip of exactly this request, or speak it. */
export function onOrb() {
  if (state.busy) cancelSynthesis();
  else if (player.matches(signature())) player.toggle();
  else speak();
}

/** Another take of the clip in the player, in its own voice, with the next seed. */
export function onRegen() {
  if (state.busy || !player.clip) return;
  if (player.clip.voice !== state.voice) applyVoice(player.clip.voice);
  speakAgain();
}

/** A clip from this session: its voice, text and settings back, then heard. */
export function onPickClip(clip) {
  if (state.busy) return;
  if (clip.voice !== state.voice) applyVoice(clip.voice);
  applyOptions(clip.options);
  els.text.value = clip.text;
  els.text.dispatchEvent(new Event("input"));
  presentClip(clip);
}

/** Escape stops a request or closes a panel; Ctrl/Cmd+Enter speaks. */
export function onKeydown(event) {
  // Escape belongs to the dialogs while one is open.
  if (els.dialog?.open || els.cal?.open) return;
  if (event.key === "Escape") {
    if (state.busy) cancelSynthesis();
    else closePanels();
  } else if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    event.preventDefault();
    speak();
  }
}
