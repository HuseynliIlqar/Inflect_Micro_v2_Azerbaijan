/**
 * The building blocks every normalisation step shares: Unicode-aware word
 * boundaries, Azerbaijani casing, vowel harmony, and numbers read as words.
 */

import { numberToWords } from "./num-az.js";
import { LETTER_NAMES } from "./az-tables.js";

// Word boundaries, Unicode-aware, matching Python's `\b`.
export const WORD = "\\p{L}\\p{N}_";
export const B_START = `(?<![${WORD}])`;
export const B_END = `(?![${WORD}])`;

const ROMAN_VALUES = { I: 1, V: 5, X: 10, L: 50 };
export const LONE_ROMAN = { I: 1, V: 5, X: 10 };
// Deliberately limited to I/V/X/L so ordinary upper-case words (DVD, MIX, CD)
// are never mistaken for numerals.
const ROMAN_RE = /^(?=[IVXL]{2,})(XL|L?X{0,4})(IX|IV|V?I{0,3})$/;
const ROMAN_MAX = 50;

// Optional Azerbaijani suffix written after a dash: "2024-cü", "ATM-də".
export const SUFFIX = `(?:-(\\p{L}[${WORD}]*))?`;

// A number, and a number or a range of them: "2,5", "10-15".
export const NUMBER = "\\d+(?:[.,]\\d+)?";
export const AMOUNT = `${NUMBER}(?:\\s?[-–]\\s?${NUMBER})?`;
// A scale word written between an amount and its currency: "5 mln AZN".
export const SCALE = "(?:\\s?(?:mln|mlrd|trln|milyon|milyard|trilyon|min)\\.?)?";

const VOWELS = "aıoueəiöü";
const BACK_VOWELS = "aıou";
// The four-way vowel (ı/i/u/ü) that follows each vowel.
const FOUR_WAY = { a: "ı", ı: "ı", o: "u", u: "u", e: "i", ə: "i", i: "i", ö: "ü", ü: "ü" };

export function escapeRegExp(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Lower-case with the Azerbaijani dotted/dotless `i` rules. */
export function azLower(text) {
  return text.replace(/I/g, "ı").replace(/İ/g, "i").toLowerCase();
}

/** Capitalise the first letter; `i` becomes `İ`, not `I`. */
export function azCapitalise(text) {
  if (!text) return text;
  const head = text[0] === "i" ? "İ" : text[0].toUpperCase();
  return head + text.slice(1);
}

/** The value of a Roman numeral, or null if it is not one. */
export function romanToInt(token) {
  if (!ROMAN_RE.test(token)) return null;
  let total = 0;
  let previous = 0;
  for (let index = token.length - 1; index >= 0; index -= 1) {
    const value = ROMAN_VALUES[token[index]];
    total = value < previous ? total - value : total + value;
    previous = Math.max(previous, value);
  }
  return total > 0 && total <= ROMAN_MAX ? total : null;
}

/** Read an acronym one letter at a time: `ATM` -> `a te em`. */
export function spellAcronym(token) {
  return [...azLower(token)]
    .map((letter) => LETTER_NAMES[letter] ?? letter)
    .join(" ");
}

/** Glue a written suffix onto a spoken word, dropping the dash. */
export function attachSuffix(word, suffix) {
  return suffix ? `${word}${suffix}` : word;
}

/**
 * Re-attach a written suffix to a *different* spoken word: `10 AZN-dən` is
 * said `manatdan`. Vowels follow the new word and the buffer consonant is
 * added or dropped.
 */
export function harmonise(word, suffix) {
  if (!suffix) return word;
  const base = azLower(word);
  let ending = azLower(suffix);
  if (VOWELS.includes(base.at(-1)) && VOWELS.includes(ending[0])) {
    ending = ("aə".includes(ending[0]) ? "y" : "n") + ending;
  } else if (
    !VOWELS.includes(base.at(-1)) &&
    ending.length > 1 &&
    "yn".includes(ending[0]) &&
    VOWELS.includes(ending[1])
  ) {
    ending = ending.slice(1);
  }
  let last = [...base].reverse().find((c) => VOWELS.includes(c)) ?? "a";
  let harmonised = "";
  for (let character of ending) {
    if ("aə".includes(character)) {
      character = BACK_VOWELS.includes(last) ? "a" : "ə";
    } else if ("ıiuü".includes(character)) {
      character = FOUR_WAY[last];
    }
    if (VOWELS.includes(character)) last = character;
    harmonised += character;
  }
  return word + harmonised;
}

/** `1,5` -> `bir tam onda beş`; falls back to digits when very long. */
function decimalToWords(whole, fraction) {
  const places = { 1: "onda", 2: "yüzdə", 3: "mində" }[fraction.length];
  const head = numberToWords(BigInt(whole));
  if (places === undefined) {
    const digits = [...fraction].map((d) => numberToWords(BigInt(d))).join(" ");
    return `${head} tam ${digits}`;
  }
  return `${head} tam ${places} ${numberToWords(BigInt(fraction))}`;
}

/** An integer, or a decimal written with `,` or `.`. */
export function speakNumber(raw) {
  if (raw.includes(",") || raw.includes(".")) {
    const [whole, fraction] = raw.split(/[.,]/u, 2);
    return decimalToWords(whole, fraction);
  }
  return numberToWords(BigInt(raw));
}

/** `007` -> `sıfır sıfır yeddi`: leading zeros are read, not dropped. */
export function speakDigits(raw) {
  const rest = raw.replace(/^0+/u, "");
  const zeros = Array(raw.length - rest.length).fill("sıfır");
  return [...zeros, ...(rest ? [numberToWords(BigInt(rest))] : [])].join(" ");
}

export function atSentenceStart(text, index) {
  const before = text.slice(0, index).trimEnd();
  return !before || ".!?".includes(before.at(-1));
}

export const isDigits = (text) => /^\d+$/u.test(text);
