/**
 * The drawn half of the device check: the card beside the header and the
 * dialog. The verdict itself comes from js/device-check.js.
 *
 * On load a degraded device gets the dialog (once per verdict, unless the
 * visitor asked not to see it again); a device with WebGPU gets a toast.
 * The card stays, and its Details button reopens the dialog.
 */

import { toast } from "./notify.js";

const DISMISS_KEY = "aztts-device-dialog-dismissed";

function readDismissed() {
  try {
    return localStorage.getItem(DISMISS_KEY);
  } catch {
    return null;
  }
}

function writeDismissed(level) {
  try {
    localStorage.setItem(DISMISS_KEY, level);
  } catch {
    // Private windows refuse storage; the dialog then shows on each visit.
  }
}

/** The title and text a verdict reads as, in the current language. */
export function deviceCopy(verdict, t) {
  const text = {
    gpu: () => t("device_text_gpu"),
    phone: () => t("device_text_phone", { threads: verdict.threads }),
    threads: () => t("device_text_threads", { threads: verdict.threads }),
    "single-framed": () => t("device_text_single_framed"),
    single: () => t(verdict.appleMobile ? "device_text_single_ios" : "device_text_single"),
  }[verdict.level]();
  const title = verdict.gpuFailed
    ? t("device_gpu_failed_title")
    : t(verdict.level === "gpu" ? "device_title_gpu"
      : verdict.level === "phone" ? "device_title_phone"
      : verdict.level === "threads" ? "device_title_threads" : "device_title_single");
  const note = verdict.forced ? t("device_note_forced")
    : verdict.level === "phone" ? t("device_note_phone") : "";
  return { title, text, note };
}

export function createDeviceView(els, { directUrl }) {
  let verdict = null;
  let t = (key) => key;

  function renderCard() {
    els.deviceLabel.textContent = t("device_card_label");
    els.deviceDetails.textContent = t("device_details");
    if (!verdict) {
      els.deviceCard.dataset.tone = "pending";
      els.deviceTitle.textContent = t("device_checking");
      return;
    }
    const { title } = deviceCopy(verdict, t);
    els.deviceCard.dataset.tone = verdict.tone;
    els.deviceMeter.dataset.bars = String(verdict.bars);
    els.deviceTitle.textContent = title;
    els.deviceDetails.hidden = false;
  }

  function fillDialog() {
    const { title, text, note } = deviceCopy(verdict, t);
    els.dialog.dataset.tone = verdict.tone;
    els.dialogMeter.dataset.bars = String(verdict.bars);
    els.dialogTitle.textContent = title;
    els.dialogText.textContent = text;
    els.dialogNote.textContent = note;
    els.dialogNote.hidden = !note;
    els.dialogDontShowLabel.textContent = t("dialog_dont_show");
    els.dialogOk.textContent = t("dialog_ok");
    els.dialogDirect.hidden = verdict.action !== "open-direct";
    els.dialogDirect.href = directUrl;
    els.dialogDirect.textContent = t("dialog_open_direct");
  }

  function openDialog() {
    if (!verdict) return;
    fillDialog();
    els.dialogDontShow.checked = false;
    if (typeof els.dialog.showModal === "function") els.dialog.showModal();
    else announceAsToast();
  }

  function announceAsToast() {
    const { title, text } = deviceCopy(verdict, t);
    toast(els.toasts, { tone: verdict.tone, title, text, closeLabel: t("toast_close") });
  }

  els.dialog.addEventListener("close", () => {
    if (els.dialogDontShow.checked) writeDismissed(verdict.level);
  });
  els.deviceDetails.addEventListener("click", openDialog);

  return {
    /** New labels: redraw whatever is showing. */
    setLanguage(translate) {
      t = translate;
      renderCard();
      if (els.dialog.open && verdict) fillDialog();
    },
    /** The first verdict, on page load: a dialog when degraded, else a toast. */
    announce(next) {
      verdict = next;
      renderCard();
      if (verdict.dialog && readDismissed() !== verdict.level) openDialog();
      else announceAsToast();
    },
    /** A revised verdict mid-session (WebGPU failed): card and a toast. */
    revise(next, reason) {
      verdict = next;
      renderCard();
      const { title } = deviceCopy(verdict, t);
      toast(els.toasts, {
        tone: verdict.tone,
        title,
        text: reason ? t("mode_gpu_failed", { reason }) : deviceCopy(verdict, t).text,
        closeLabel: t("toast_close"),
      });
    },
    get verdict() {
      return verdict;
    },
  };
}
