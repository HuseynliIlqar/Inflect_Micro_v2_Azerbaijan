/**
 * What the page tells a visitor about this device: checked the moment the page
 * opens, shown in the device card and in a popup, and revised when synthesis
 * finds out more (a WebGPU adapter that then fails).
 *
 * Pure, so every verdict is tested without a browser.
 */

/**
 * @param {{ gpu?: boolean, gpuFailed?: boolean, preferWasm?: boolean,
 *           threads?: number, framed?: boolean }} facts
 * @returns {{ level: "gpu"|"threads"|"single-framed"|"single", bars: 1|2|3,
 *             tone: "good"|"warn"|"bad", dialog: boolean,
 *             action: "open-direct"|null, threads: number,
 *             gpuFailed: boolean, forced: boolean }}
 *   bars: the speed meter; dialog: a modal on load (degraded), else a toast;
 *   action: the extra button the popup offers.
 */
export function deviceVerdict({ gpu = false, gpuFailed = false, preferWasm = false, threads = 1, framed = false }) {
  const base = { threads, gpuFailed, forced: preferWasm && gpu };
  if (gpu && !gpuFailed && !preferWasm) {
    return { ...base, level: "gpu", bars: 3, tone: "good", dialog: false, action: null };
  }
  if (threads > 1) {
    return { ...base, level: "threads", bars: 2, tone: "warn", dialog: true, action: null };
  }
  return framed
    ? { ...base, level: "single-framed", bars: 1, tone: "bad", dialog: true, action: "open-direct" }
    : { ...base, level: "single", bars: 1, tone: "bad", dialog: true, action: null };
}

/**
 * Which advice fits an error message: the page names the next step, not just
 * the problem. "network" | "worker" | "backend" | "memory" | "generic".
 */
export function errorKind(message) {
  const text = String(message ?? "").toLowerCase();
  if (/could not (download|load https?:)|failed to fetch|networkerror|http \d{3}/u.test(text)) return "network";
  if (/background worker|web worker/u.test(text)) return "worker";
  if (/no backend could load/u.test(text)) return "backend";
  if (/allocation failed|out of memory|bad_alloc/u.test(text)) return "memory";
  return "generic";
}
