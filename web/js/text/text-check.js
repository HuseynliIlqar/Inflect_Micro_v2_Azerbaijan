/**
 * What the playground accepts before a request reaches the worker.
 *
 * The page is public and runs on the visitor's own device: a phone's browser
 * takes 10-20 s per sentence, so an unbounded paste would hold it for many
 * minutes (and can run it out of memory). The limit keeps one request to a
 * few paragraphs. The worker checks it again, so the page cannot be bypassed.
 *
 * Errors block Speak; warnings only explain what the model will do.
 */

/** Characters (code points) one request may carry, after trimming. */
export const MAX_CHARS = 500;

/** From this share of the limit on, the counter is shown as a caution. */
export const NEAR_SHARE = 0.9;

// Control characters other than tab and newline: invisible, never spoken, and
// a paste from a PDF or a terminal brings them in.
const CONTROL = /[\p{Cc}\p{Cf}]/gu;
const KEEP = new Set(["\t", "\n", "\r"]);

// Letters an English sentence does not use; their presence means the text is
// Azerbaijani and the English voice would read it with English sounds.
const AZ_ONLY_LETTERS = /[əƏıİğĞ]/u;
// A letter outside the Latin alphabet (Cyrillic, Arabic, Greek, ...).
const NON_LATIN_LETTER = /(?=\p{L})\P{Script=Latin}/u;
const SPEAKABLE = /[\p{L}\p{N}]/u;

/**
 * The text with invisible control characters removed and the ends trimmed.
 * @param {string} text
 * @returns {string}
 */
export function cleanText(text) {
  return String(text ?? "")
    .replace(CONTROL, (ch) => (KEEP.has(ch) ? ch : ""))
    .trim();
}

/**
 * Length as a person counts it: code points, so `ə` or an emoji is one.
 * @param {string} text
 * @returns {number}
 */
export function charCount(text) {
  return Array.from(cleanText(text)).length;
}

/**
 * @typedef {{
 *   text: string,
 *   count: number,
 *   max: number,
 *   error: null | "empty" | "too_long" | "nothing_to_say",
 *   warnings: string[],
 *   near: boolean,
 * }} TextCheck
 */

/**
 * Whether `text` may be spoken by `voice`, and what to tell the visitor.
 * @param {string} text
 * @param {{ voice?: string, max?: number }} [options]
 * @returns {TextCheck}
 */
export function checkText(text, { voice = "az", max = MAX_CHARS } = {}) {
  const clean = cleanText(text);
  const count = Array.from(clean).length;
  const near = count >= Math.ceil(max * NEAR_SHARE);
  const result = (error, warnings = []) => ({ text: clean, count, max, error, warnings, near });

  if (!clean) return result("empty");
  if (count > max) return result("too_long");
  if (!SPEAKABLE.test(clean)) return result("nothing_to_say");

  const warnings = [];
  if (NON_LATIN_LETTER.test(clean)) warnings.push("warn_script");
  if (voice === "en" && AZ_ONLY_LETTERS.test(clean)) warnings.push("warn_az_in_en");
  return result(null, warnings);
}

/** i18n keys for an error: the problem, then what to do about it. */
export const ERROR_KEYS = {
  empty: ["error_empty", "error_empty_hint"],
  too_long: ["error_too_long", "error_too_long_hint"],
  nothing_to_say: ["error_nothing_to_say", "error_nothing_to_say_hint"],
};
