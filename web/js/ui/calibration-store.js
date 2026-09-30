/**
 * The device check's result, kept in this browser so it runs once per device.
 *
 * Reused only while nothing it measured has changed: the graphs (the
 * reference's version), the browser (its user agent), the threads it allows
 * and the GPU it offers. After a month it is measured again anyway -- drivers
 * update without the user agent saying so.
 *
 * Storage is passed in, so this is tested without a browser; every read and
 * write survives a storage that throws (private windows, blocked site data).
 */

const STORAGE_KEY = "aztts-device-check";
export const MAX_AGE_MS = 30 * 24 * 60 * 60 * 1000;
const KEY_FIELDS = ["version", "userAgent", "threads", "adapter"];

/**
 * @param {Storage|undefined} storage
 * @param {{ version: string, userAgent: string, threads: number, adapter: string }} key
 * @param {number} now  milliseconds since the epoch
 * @returns {object|null} the stored verdict, or null when it must be measured
 */
export function readStored(storage, key, now) {
  let entry;
  try {
    entry = JSON.parse(storage?.getItem(STORAGE_KEY) ?? "null");
  } catch {
    return null;
  }
  const verdict = entry?.verdict;
  if (!isVerdict(verdict) || verdict.version !== key.version) return null;
  if (!KEY_FIELDS.every((field) => entry.key?.[field] === key[field])) return null;
  const age = now - verdict.measuredAt;
  return age >= 0 && age <= MAX_AGE_MS ? verdict : null;
}

const isStep = (step) => typeof step?.backend === "string" && typeof step?.precision === "string";

/** The shape js/engine/calibration-verdict.js returns -- storage is not trusted. */
function isVerdict(verdict) {
  return Boolean(verdict)
    && typeof verdict.measuredAt === "number"
    && Array.isArray(verdict.plan) && verdict.plan.length > 0 && verdict.plan.every(isStep)
    && Array.isArray(verdict.trials)
    && (verdict.chosen === null || isStep(verdict.chosen))
    && (verdict.rtf === null || typeof verdict.rtf === "number")
    && (verdict.sentenceSeconds === null || typeof verdict.sentenceSeconds === "number")
    && ["fast", "usable", "slow", "failed"].includes(verdict.tier)
    && typeof verdict.gpu === "string";
}

export function writeStored(storage, key, verdict) {
  try {
    storage?.setItem(STORAGE_KEY, JSON.stringify({ key, verdict }));
  } catch {
    // Refused storage: the check simply runs again next visit.
  }
}

/** A verdict that turned out wrong (its backend failed later): measure again next visit. */
export function forgetStored(storage) {
  try {
    storage?.removeItem(STORAGE_KEY);
  } catch {
    // Nothing stored that could be read either.
  }
}

/** The page's localStorage, or undefined where even touching it throws. */
export function browserStorage() {
  try {
    return globalThis.localStorage;
  } catch {
    return undefined;
  }
}
