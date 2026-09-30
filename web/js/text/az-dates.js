/**
 * Normalisation steps for things read by their parts: phone numbers, dates,
 * dotted version numbers and clock times. Called in order by `normalizeAz`.
 */

import { numberToWords, ordinalToWords } from "./num-az.js";
import { MONTHS, ORDINAL_SUFFIXES } from "./az-tables.js";
import {
  B_END,
  B_START,
  SUFFIX,
  attachSuffix,
  azLower,
  harmonise,
  speakDigits,
} from "./az-words.js";

/** `iyun` -> `[iİ]yun`, the same pattern the Python side builds. */
function eitherCase(word) {
  const head = word[0];
  const upper = head === "i" ? "İ" : head.toUpperCase();
  return `[${head}${upper}]${word.slice(1)}`;
}

const MONTH_RE = MONTHS.map(eitherCase).join("|");
// "il" right after a date already names the year: "01.09.1939 ildə".
const YEAR_WORD_RE = new RegExp(`^\\s+il(?:in|də|dən|i|ə)?${B_END}`, "u");

/** Split an unbroken run the way numbers are dictated: `... 123 45 67`. */
function phoneGroups(digits) {
  if (digits.length <= 3) return digits ? [digits] : [];
  if (digits.length === 4) return [digits.slice(0, 2), digits.slice(2)];
  if (digits.length >= 7) {
    return [
      ...phoneGroups(digits.slice(0, -7)),
      digits.slice(-7, -4), digits.slice(-4, -2), digits.slice(-2),
    ];
  }
  return [...phoneGroups(digits.slice(0, -2)), digits.slice(-2)];
}

/** Read a phone number group by group, zeros included. */
export function replacePhoneNumbers(text) {
  const speak = (match) =>
    match
      .match(/\d+/gu)
      .flatMap(phoneGroups)
      .map(speakDigits)
      .join(" ");
  text = text.replace(/\+\d[\d\s\-()]{7,}\d/gu, speak);
  return text.replace(
    /(?<![\d+])\(?0\d{2}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\d)/gu,
    speak,
  );
}

/**
 * `1, 9, 1939` -> `bir sentyabr min doqquz yüz otuz doqquzuncu il`: the day a
 * cardinal, the year an ordinal followed by `il`.
 */
function speakDate(day, month, year, suffix, yearWordFollows) {
  if (!(day >= 1 && day <= 31 && month >= 1 && month <= 12)) return null;
  const head = `${numberToWords(day)} ${MONTHS[month - 1]}`;
  if (year === null) return harmonise(head, suffix);
  const spokenYear = ordinalToWords(year);
  if (suffix && ORDINAL_SUFFIXES.has(azLower(suffix))) return `${head} ${spokenYear}`;
  if (suffix) return `${head} ${spokenYear} ${harmonise("il", suffix)}`;
  if (yearWordFollows) return `${head} ${spokenYear}`;
  return `${head} ${spokenYear} il`;
}

export function replaceDates(text) {
  // `offset` and `whole` are the last two arguments String.replace passes.
  const speak = (match, offset, whole, day, month, year, suffix) => {
    const follows = YEAR_WORD_RE.test(whole.slice(offset + match.length));
    const spoken = speakDate(day, month, year, suffix, follows);
    return spoken === null ? match : spoken;
  };

  text = text.replace(
    new RegExp(`(?<![\\d.,])(\\d{4})-(\\d{2})-(\\d{2})(?!\\d)${SUFFIX}${B_END}`, "gu"),
    (match, y, m, d, suffix, offset, whole) =>
      speak(match, offset, whole, Number(d), Number(m), Number(y), suffix),
  );
  text = text.replace(
    new RegExp(
      `(?<![\\d.,/])(\\d{1,2})([./-])(\\d{1,2})\\2(\\d{4}|\\d{2})(?!\\d|[./]\\d)${SUFFIX}${B_END}`,
      "gu",
    ),
    (match, d, separator, m, y, suffix, offset, whole) => {
      if (separator === "-" && y.length !== 4) return match;
      return speak(match, offset, whole, Number(d), Number(m), Number(y), suffix);
    },
  );
  // "15/08": a zero-padded second part is a month, never a denominator.
  text = text.replace(
    new RegExp(`(?<![\\d.,/])(\\d{1,2})/(0[1-9])(?![\\d/])${SUFFIX}`, "gu"),
    (match, d, m, suffix, offset, whole) =>
      speak(match, offset, whole, Number(d), Number(m), null, suffix),
  );
  // "01 may": the zero is padding, not a digit to read.
  return text.replace(
    new RegExp(`(?<![\\d.,])0(\\d)(?=\\s+(?:${MONTH_RE}))`, "gu"),
    "$1",
  );
}

/** Versions and addresses, or a date that failed validation: `1.2.3`. */
export function replaceDottedChains(text) {
  return text.replace(/(?<![\d.,])\d+(?:\.\d+){2,}(?!\d)/gu, (match) =>
    match
      .split(".")
      .map((part) => numberToWords(BigInt(part)))
      .join(" nöqtə "),
  );
}

export function replaceTimes(text) {
  const pattern = new RegExp(
    `${B_START}(\\d{1,2}):(\\d{2})${SUFFIX}${B_END}`,
    "gu",
  );
  return text.replace(pattern, (match, h, m, suffix) => {
    const hour = Number(h);
    const minute = Number(m);
    if (minute > 59 || hour > 24 || (hour === 24 && minute)) return match;
    if (minute === 0) {
      return attachSuffix(hour === 0 ? "sıfır sıfır" : numberToWords(hour), suffix);
    }
    const tail = attachSuffix(numberToWords(minute), suffix);
    return minute < 10
      ? `${numberToWords(hour)} sıfır ${tail}`
      : `${numberToWords(hour)} ${tail}`;
  });
}
