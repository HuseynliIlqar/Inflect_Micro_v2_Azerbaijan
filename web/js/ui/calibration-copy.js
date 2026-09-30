/**
 * What the visitor is told after the device check: a verdict they can act on
 * ("this will work comfortably here" / "slowly"), the measured speed in
 * seconds per sentence, and the reasons behind it.
 *
 * Pure: label keys and values out, js/ui/i18n-calibration.js holds the words.
 * The title and summary stay plain; the reasons may name WebGPU, because
 * whether the browser has it is the question the visitor came with.
 */

const LOOK = {
  fast: { tone: "good", bars: 3 },
  usable: { tone: "warn", bars: 2 },
  slow: { tone: "bad", bars: 1 },
  failed: { tone: "bad", bars: 1 },
};

const GPU_KEYS = {
  chosen: "cal_gpu_chosen",
  slower: "cal_gpu_slower",
  unsupported: "cal_gpu_unsupported",
  "no-adapter": "cal_gpu_no_adapter",
  software: "cal_gpu_software",
  "device-failed": "cal_gpu_device_failed",
  error: "cal_gpu_error",
  "wrong-output": "cal_gpu_wrong_output",
  unstable: "cal_gpu_unstable",
  off: "cal_gpu_off",
};

const item = (key, values = {}) => ({ key, values });

function runsOn(chosen) {
  if (!chosen) return null;
  if (chosen.backend === "webgpu") return item("cal_runs_gpu");
  return item(chosen.precision === "int8" ? "cal_runs_cpu_fast" : "cal_runs_cpu_full");
}

function threadReason({ chosen, threads, framed }) {
  if (chosen?.backend !== "wasm") return null;
  if (threads > 1) return item("cal_threads_many", { threads });
  return item(framed ? "cal_threads_framed" : "cal_threads_single");
}

function advice(calibration, { phone, appleMobile }) {
  const { tier, chosen } = calibration;
  if (tier === "fast") return null;
  if (appleMobile) return item("cal_advice_ios");
  if (phone) return item("cal_advice_phone");
  return chosen?.backend === "webgpu" ? null : item("cal_advice_browser");
}

/**
 * @param {object} calibration  the verdict from js/engine/calibration-verdict.js
 * @param {{ phone?: boolean, appleMobile?: boolean }} device
 * @returns {{ tone: string, bars: number, title: string,
 *             summary: {key: string, values: object},
 *             runsOn: {key: string, values: object}|null,
 *             reasons: {key: string, values: object}[] }}
 */
export function calibrationCopy(calibration, { phone = false, appleMobile = false } = {}) {
  const { tier, rtf, sentenceSeconds, gpu, gpuMessage, int8Failed } = calibration;
  const look = LOOK[tier] ?? LOOK.failed;
  const summary = tier === "failed"
    ? item("cal_summary_failed")
    : item("cal_summary", { rtf: rtf.toFixed(2), sentence: sentenceSeconds.toFixed(1) });
  const gpuKey = GPU_KEYS[gpu] ?? "cal_gpu_error";
  const reasons = [
    item(gpuKey, { message: gpuMessage || "-" }),
    threadReason(calibration),
    int8Failed ? item("cal_int8_failed") : null,
    advice(calibration, { phone, appleMobile }),
  ].filter(Boolean);
  return { ...look, title: `cal_title_${tier in LOOK ? tier : "failed"}`, summary, runsOn: runsOn(calibration.chosen), reasons };
}

/**
 * The device card's verdict (js/ui/device-view.js) for a measured device: no
 * dialog on load -- the check's own screen already said it.
 */
export function measuredDeviceVerdict(calibration, device = {}) {
  const copy = calibrationCopy(calibration, device);
  const framedSingle = calibration.framed && calibration.threads === 1 && calibration.chosen?.backend !== "webgpu";
  return {
    level: "measured",
    tone: copy.tone,
    bars: copy.bars,
    dialog: false,
    action: framedSingle ? "open-direct" : null,
    threads: calibration.threads,
    calibration,
    copy,
  };
}

/** One line for the diagnostics a tester pastes: what was measured and chosen. */
export function calibrationLine(calibration) {
  if (!calibration) return "check=none";
  const chosen = calibration.chosen ? `${calibration.chosen.backend}/${calibration.chosen.precision}` : "none";
  const rtf = calibration.rtf === null ? "-" : calibration.rtf.toFixed(2);
  const trials = calibration.trials
    .map((t) => `${t.backend}/${t.precision}:${t.ok ? "ok" : t.reason}${t.rtf === null ? "" : `@${t.rtf.toFixed(2)}`}`)
    .join(",");
  return `check=${chosen}@${rtf} gpu:${calibration.gpu} [${trials}]`;
}
