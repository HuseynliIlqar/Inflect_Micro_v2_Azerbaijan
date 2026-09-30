/** The status line, the progress bar and the seconds counter during a request. */

import { downloadFraction, chunkFraction, megabytes } from "../ui/progress.js";
import { els } from "./dom.js";
import { state, t } from "./context.js";
import { renderModeBanner, renderSlowNote } from "./mode-view.js";
import { reviseDevice } from "./device.js";

/** `undefined` hides the bar, `null` makes it indeterminate, 0..1 fills it. */
export function setProgress(fraction, text = els.status.textContent) {
  const bar = els.progress;
  const fill = bar.firstElementChild;
  if (fraction === undefined) {
    bar.hidden = true;
    return;
  }
  bar.hidden = false;
  bar.setAttribute("aria-label", t("progress_label"));
  if (text) bar.setAttribute("aria-valuetext", text);
  else bar.removeAttribute("aria-valuetext");
  bar.classList.toggle("is-indeterminate", fraction === null);
  if (fraction === null) {
    bar.removeAttribute("aria-valuenow");
    fill.style.width = "";
  } else {
    const percent = Math.round(fraction * 100);
    bar.setAttribute("aria-valuenow", String(percent));
    fill.style.width = `${percent}%`;
  }
}

let lastDownloadStep = null;

function showDownload(message, requestedVoice) {
  const fraction = downloadFraction(message.loaded, message.total);
  const text =
    fraction === null
      ? t(requestedVoice === "en" ? "en_loading" : "loading_model")
      : t("download_progress", {
          loaded: megabytes(message.loaded),
          total: megabytes(message.total),
        });
  // The status is a live region: rewrite it in 10% steps, not per read,
  // or a screen reader queues dozens of byte counts.
  const step = fraction === null ? -1 : Math.floor(fraction * 10);
  if (step !== lastDownloadStep) {
    lastDownloadStep = step;
    els.status.textContent = text;
  }
  setProgress(fraction, text);
}

/** Status line and bar for one progress message from the worker. */
export function showProgress(message, requestedVoice) {
  switch (message.stage) {
    case "phonemes":
      lastDownloadStep = null;
      els.status.textContent = t("loading_phonemes");
      setProgress(null);
      break;
    case "download":
      showDownload(message, requestedVoice);
      break;
    case "compile":
      els.status.textContent = t("compiling");
      setProgress(null);
      break;
    case "backend":
      state.backend = message.backend;
      state.threads = message.threads ?? 1;
      state.precision = message.precision ?? null;
      state.fallbacks = message.fallbacks ?? [];
      state.modeVoice = requestedVoice;
      renderSlowNote();
      renderModeBanner();
      reviseDevice();
      break;
    case "chunk":
      els.status.textContent =
        message.total > 1
          ? `${t("synthesising")} ${message.done + 1}/${message.total}`
          : t("synthesising");
      setProgress(chunkFraction(message.done, message.total));
      break;
    default:
      break;
  }
}

/** Seconds since `began`, beside the status line, once a second. */
export function startClock(began) {
  const tick = () => {
    els.elapsed.textContent = `${Math.floor((performance.now() - began) / 1000)} s`;
  };
  tick();
  const timer = setInterval(tick, 1000);
  return () => {
    clearInterval(timer);
    els.elapsed.textContent = "";
  };
}
