/**
 * The device check's screen: a modal that blocks the page while the device is
 * measured, then shows the verdict until the visitor presses Start.
 *
 * It draws what js/ui/calibration-progress.js and js/ui/calibration-copy.js
 * decide; nothing here chooses anything. Escape does not close it while the
 * check runs -- the page is not usable yet -- and means Start once it is done.
 */

import { els } from "./dom.js";

const say = (t, item) => t(item.key, item.values);

function renderBar(fraction, text) {
  const percent = Math.round(fraction * 100);
  els.calProgress.setAttribute("aria-valuenow", String(percent));
  els.calProgress.setAttribute("aria-valuetext", text);
  els.calProgress.firstElementChild.style.width = `${percent}%`;
}

function stepRow(t, row) {
  const item = document.createElement("li");
  item.dataset.status = row.status;
  const dot = document.createElement("span");
  dot.className = "calibration-dot";
  dot.setAttribute("aria-hidden", "true");
  const name = document.createElement("span");
  name.textContent = t(row.label);
  const detail = document.createElement("span");
  detail.className = "calibration-detail";
  // The dot is colour alone; a screen reader hears the status in words.
  detail.textContent = row.detail ? say(t, row.detail) : t(`cal_status_${row.status}`);
  item.append(dot, name, detail);
  return item;
}

export function createCalibrationView() {
  let t = (key) => key;
  let finished = null; // resolves the promise a result or an error screen returns
  let measuring = false;

  els.cal.addEventListener("cancel", (event) => {
    event.preventDefault();
    if (finished && !els.calContinue.hidden) els.calContinue.click();
  });
  // Chrome closes a dialog on a second Escape even when the first was
  // prevented. Mid-check the page is not usable yet, so it opens again; on a
  // result or an error screen, closing counts as the way forward.
  els.cal.addEventListener("close", () => {
    if (measuring) els.cal.showModal();
    else settle(els.calContinue.hidden ? "skip" : "continue");
  });

  function settle(value) {
    const resolve = finished;
    finished = null;
    resolve?.(value);
  }
  els.calContinue.addEventListener("click", () => settle("continue"));
  els.calRetry.addEventListener("click", () => settle("retry"));
  els.calSkip.addEventListener("click", () => settle("skip"));

  function showButtons({ go = false, retry = false, skip = false }) {
    els.calContinue.hidden = !go;
    els.calRetry.hidden = !retry;
    els.calSkip.hidden = !skip;
    els.calContinue.textContent = t("cal_continue");
    els.calRetry.textContent = t("cal_retry");
    els.calSkip.textContent = t("cal_skip");
  }

  return {
    /** Open on the measuring screen. False when the browser cannot show a modal. */
    open(translate) {
      t = translate;
      if (typeof els.cal.showModal !== "function") return false;
      els.cal.dataset.tone = "pending";
      els.cal.setAttribute("aria-labelledby", "cal-heading");
      els.calMeasuring.hidden = false;
      els.calResult.hidden = true;
      els.calError.hidden = true;
      els.calHeading.textContent = t("cal_heading");
      els.calIntro.textContent = t("cal_intro");
      els.calSteps.setAttribute("aria-label", t("cal_steps_label"));
      els.calProgress.setAttribute("aria-label", t("cal_progress_label"));
      showButtons({});
      measuring = true;
      if (!els.cal.open) els.cal.showModal();
      // No button yet: the dialog itself holds focus, so a reader starts at its title.
      els.cal.focus();
      return true;
    },

    /** One state from js/ui/calibration-progress.js. */
    render(progress) {
      const { stage } = progress;
      const values = stage.backendKey ? { ...stage.values, backend: t(stage.backendKey) } : stage.values;
      const text = t(stage.key, values);
      if (els.calStage.textContent !== text) els.calStage.textContent = text;
      renderBar(progress.fraction, text);
      els.calSteps.replaceChildren(...progress.rows.map((row) => stepRow(t, row)));
    },

    /** The verdict, from calibrationCopy. Resolves when the visitor presses Start. */
    showResult(copy) {
      measuring = false;
      els.cal.dataset.tone = copy.tone;
      els.cal.setAttribute("aria-labelledby", "cal-result-title");
      els.calMeasuring.hidden = true;
      els.calResult.hidden = false;
      els.calMeter.dataset.bars = String(copy.bars);
      els.calResultTitle.textContent = t(copy.title);
      els.calSummary.textContent = say(t, copy.summary);
      els.calRunsOn.textContent = copy.runsOn ? say(t, copy.runsOn) : "";
      els.calRunsOn.hidden = !copy.runsOn;
      els.calWhy.textContent = t("cal_why");
      els.calReasons.replaceChildren(...copy.reasons.map((reason) => {
        const item = document.createElement("li");
        item.textContent = say(t, reason);
        return item;
      }));
      els.calNoteOnce.textContent = t("cal_note_once");
      showButtons({ go: true });
      els.calContinue.focus();
      return new Promise((resolve) => {
        finished = resolve;
      });
    },

    /** The check itself failed. Resolves "retry" or "skip". */
    showError(message) {
      measuring = false;
      els.cal.dataset.tone = "bad";
      els.calError.textContent = message;
      els.calError.hidden = false;
      showButtons({ retry: true, skip: true });
      els.calRetry.focus();
      return new Promise((resolve) => {
        finished = resolve;
      });
    },

    close() {
      measuring = false;
      finished = null;
      if (els.cal.open) els.cal.close();
    },
  };
}
