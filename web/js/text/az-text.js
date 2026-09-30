/**
 * Azerbaijani text normalisation -- a port of `aztts/az_text.py`.
 *
 * The model was trained on prose, so digits, Roman numerals, abbreviations and
 * symbols have to become the words a reader would say before they reach it.
 * This runs in the browser; the Python module runs in the CLI. They must agree
 * exactly, because the model hears whatever comes out of here.
 * `web/tests/test_az_text.mjs` checks them against each other over a golden
 * file generated from the Python side.
 *
 * Two things differ between the languages and are handled deliberately:
 *
 * - Python's `\w` and `\b` are Unicode-aware; JavaScript's are ASCII. A `\b`
 *   after "cü" would behave differently in the two, so every boundary here is
 *   written as an explicit lookaround over `\p{L}\p{N}_`.
 * - Python's `re.IGNORECASE` folds case its own way; where that matters the
 *   text is lower-cased with `azLower` first.
 */

import { ordinalToWords } from "./num-az.js";
import { ABBREVIATIONS, ACRONYMS, ROMAN_NOUNS, SYMBOLS } from "./az-tables.js";
import {
  B_END,
  B_START,
  LONE_ROMAN,
  SUFFIX,
  WORD,
  atSentenceStart,
  attachSuffix,
  azCapitalise,
  azLower,
  escapeRegExp,
  romanToInt,
  spellAcronym,
} from "./az-words.js";
import {
  replaceDates,
  replaceDottedChains,
  replacePhoneNumbers,
  replaceTimes,
} from "./az-dates.js";
import {
  replaceBareNumbers,
  replaceCurrency,
  replaceFractions,
  replaceOperators,
  replacePercent,
  replaceRanges,
  replaceSuffixedNumbers,
  replaceUnits,
  stripGroupSeparators,
} from "./az-amounts.js";

// The helpers other modules import from here, as before the split.
export { azCapitalise, azLower, harmonise, romanToInt, spellAcronym } from "./az-words.js";

function replaceAbbreviations(text) {
  for (const [written, spoken] of Object.entries(ABBREVIATIONS)) {
    const pattern = new RegExp(
      `(?<![${WORD}])${escapeRegExp(written)}`,
      "giu",
    );
    text = text.replace(pattern, spoken);
  }
  return text;
}

/** `II` -> `ikinci`, capitalised only where a sentence starts. */
function replaceRoman(text) {
  const wordAt = (whole, offset, value) => {
    const word = ordinalToWords(value);
    return atSentenceStart(whole, offset) ? azCapitalise(word) : word;
  };
  const value = (token) => romanToInt(token) ?? LONE_ROMAN[token] ?? null;
  const ordinal = "(?:-(?:cı|ci|cu|cü))?";

  text = text.replace(
    new RegExp(`${B_START}([IVXL]+)\\s?[-–]\\s?([IVXL]+)${B_END}${ordinal}`, "gu"),
    (match, a, b, offset, whole) => {
      const first = value(a);
      const second = value(b);
      if (first === null || second === null) return match;
      return `${wordAt(whole, offset, first)} ilə ${ordinalToWords(second)}`;
    },
  );
  text = text.replace(
    new RegExp(`${B_START}([IVXL]{2,})${B_END}${ordinal}`, "gu"),
    (match, token, offset, whole) => {
      const number = romanToInt(token);
      return number === null ? match : wordAt(whole, offset, number);
    },
  );
  return text.replace(
    new RegExp(
      `(?<![${WORD}.])([IVX])${B_END}(-(?:cı|ci|cu|cü))?(?=(?:\\s+(\\p{L}+))?)`,
      "gu",
    ),
    (match, token, suffix, following, offset, whole) => {
      const next = following ?? "";
      const isNumeral =
        Boolean(suffix) ||
        ROMAN_NOUNS.some((noun) => azLower(next).startsWith(noun)) ||
        (token === "I" && /^\p{Lu}/u.test(next));
      return isNumeral ? wordAt(whole, offset, LONE_ROMAN[token]) : match;
    },
  );
}

function replaceAcronyms(text) {
  return text.replace(
    new RegExp(`${B_START}([A-ZƏÇĞİÖŞÜ]{2,5})${SUFFIX}${B_END}`, "gu"),
    (_match, token, suffix) =>
      attachSuffix(ACRONYMS[token] ?? spellAcronym(token), suffix),
  );
}

function replaceSymbols(text) {
  for (const [symbol, spoken] of Object.entries(SYMBOLS)) {
    text = text.split(symbol).join(spoken);
  }
  return text;
}

/** Straighten quotes and dashes, then collapse whitespace. */
function tidy(text) {
  const pairs = [
    ["‘", "'"], ["’", "'"], ["“", '"'], ["”", '"'],
    ["–", "-"], ["—", "-"], ["…", "."], [" ", " "],
  ];
  for (const [source, target] of pairs) {
    text = text.split(source).join(target);
  }
  text = text.replace(/\s+([,.;:!?])/gu, "$1");
  return text.split(/\s+/u).filter(Boolean).join(" ");
}

const STEPS = [
  tidy,
  replaceAbbreviations,
  replacePhoneNumbers,
  stripGroupSeparators,
  replaceDates,
  replaceDottedChains,
  replaceTimes,
  replaceOperators,
  replaceCurrency,
  replacePercent,
  replaceUnits,
  replaceRoman,
  replaceFractions,
  replaceRanges,
  replaceSuffixedNumbers,
  replaceBareNumbers,
  replaceAcronyms,
  replaceSymbols,
  tidy,
];

/**
 * Rewrite digits, numerals, units and abbreviations as spoken Azerbaijani.
 *
 * The order matters: abbreviations first so their full stops never look like
 * sentence ends, then the number forms from most specific (dates, times) to
 * least (bare digits), and acronyms last so Roman numerals win the ambiguity.
 */
export function normalizeAz(text) {
  if (!text || !text.trim()) return "";
  for (const step of STEPS) text = step(text);
  return text;
}
