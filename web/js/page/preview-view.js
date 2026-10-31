/**
 * "The model will read": the text as the engine will hand it to the model --
 * censored, then normalised -- with every rewritten word marked. Shown only
 * for the Azerbaijani voice with normalisation on, and only when something
 * actually changes.
 */

import { censorAz } from "../text/az-censor.js";
import { normalizeAz } from "../text/az-text.js";
import { diffWords } from "../text/diff-words.js";
import { els } from "./dom.js";
import { state } from "./context.js";

export function renderPreview() {
  const text = els.text.value;
  const applies = state.voice === "az" && els.normalise.checked && text.trim().length > 0;
  const runs = applies ? diffWords(text, normalizeAz(censorAz(text))) : [];
  els.preview.hidden = !runs.some((run) => run.changed);
  if (els.preview.hidden) return;
  const nodes = runs.flatMap((run, index) => {
    const node = run.changed ? document.createElement("mark") : document.createTextNode(run.text);
    if (run.changed) node.textContent = run.text;
    return index ? [" ", node] : [node];
  });
  els.previewText.replaceChildren(...nodes);
}
