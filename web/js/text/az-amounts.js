/**
 * Normalisation steps for quantities: digit grouping, operators, money,
 * percentages, units, fractions, ranges and bare numbers. Called in order by
 * `normalizeAz`.
 */

import { numberToWords, ordinalToWords } from "./num-az.js";
import {
  CURRENCIES,
  CURRENCY_SUBUNITS,
  ORDINAL_SUFFIXES,
  UNITS_TABLE,
} from "./az-tables.js";
import {
  AMOUNT,
  B_END,
  B_START,
  NUMBER,
  SCALE,
  SUFFIX,
  WORD,
  attachSuffix,
  azLower,
  escapeRegExp,
  harmonise,
  isDigits,
  speakDigits,
  speakNumber,
} from "./az-words.js";

/**
 * `1 000 000`, `1.000.000` and `1,000,000` become plain digit runs. A leading
 * `0` is never a thousands group, and a single `,000` stays a decimal comma.
 */
export function stripGroupSeparators(text) {
  const join = (match) => match.replace(/[ .,]/gu, "");
  text = text.replace(/(?<![\d.,])[1-9]\d{0,2}(?: \d{3})+(?!\d)/gu, join);
  text = text.replace(/(?<![\d.,])[1-9]\d{0,2}(?:\.\d{3})+(?!\d|\.\d)/gu, join);
  return text.replace(/(?<![\d.,])[1-9]\d{0,2}(?:,\d{3}){2,}(?!\d|,\d)/gu, join);
}

/** A score or ratio (`3:2`) is two numbers; `2x3` is a product. */
export function replaceOperators(text) {
  text = text.replace(/(?<=\d):(?=\d)/gu, " ");
  return text.replace(/(?<=\d)\s?[xх×]\s?(?=\d)/gu, " × ");
}

/** `19,99` manat -> `19 manat 99 qəpik`; digits stay digits for later steps. */
function money(amount, currency, suffix) {
  const coins = /^(\d+)[.,](\d{2})$/u.exec(amount);
  if (!coins || !(currency in CURRENCY_SUBUNITS)) {
    return `${amount} ${harmonise(currency, suffix)}`;
  }
  const whole = coins[1];
  const cents = Number(coins[2]);
  const coin = `${cents} ${harmonise(CURRENCY_SUBUNITS[currency], suffix)}`;
  if (!cents) return `${whole} ${harmonise(currency, suffix)}`;
  if (!Number(whole)) return coin;
  return `${whole} ${currency} ${coin}`;
}

/** Put the currency after the amount as a word. */
export function replaceCurrency(text) {
  const symbols = escapeRegExp(
    Object.keys(CURRENCIES).filter((key) => !/^\p{L}+$/u.test(key)).join(""),
  );
  const codes = Object.keys(CURRENCIES)
    .filter((key) => /^\p{L}+$/u.test(key))
    .join("|");

  text = text.replace(
    new RegExp(`([${symbols}])\\s?(${AMOUNT}${SCALE})`, "gu"),
    (_match, symbol, amount) => money(amount, CURRENCIES[symbol], undefined),
  );
  text = text.replace(
    new RegExp(
      `(?<![${WORD}.,])(${AMOUNT}${SCALE})\\s?(${codes}|[${symbols}])${SUFFIX}(?![${WORD}])`,
      "giu",
    ),
    (_match, amount, unit, suffix) =>
      money(amount, CURRENCIES[azLower(unit)], suffix),
  );
  // A code on its own: "AZN ilə ödəniş".
  return text.replace(
    new RegExp(`${B_START}(AZN|USD|EUR|GBP|RUB)${SUFFIX}${B_END}`, "gu"),
    (_match, code, suffix) => harmonise(CURRENCIES[azLower(code)], suffix),
  );
}

export function replacePercent(text) {
  text = text.replace(new RegExp(`%\\s?(${AMOUNT})`, "gu"), "$1 faiz");
  return text.replace(
    new RegExp(`(${AMOUNT})\\s?%${SUFFIX}`, "gu"),
    (_match, amount, suffix) => `${amount} ${harmonise("faiz", suffix)}`,
  );
}

export function replaceUnits(text) {
  const keys = Object.keys(UNITS_TABLE).sort((a, b) => b.length - a.length);
  const pattern = keys.map(escapeRegExp).join("|");
  return text.replace(
    new RegExp(`(\\d)\\s?(${pattern})${SUFFIX}(?![${WORD}])`, "giu"),
    (_match, digit, unit, suffix) =>
      `${digit} ${harmonise(UNITS_TABLE[azLower(unit)], suffix)}`,
  );
}

/** `3/4` -> `dörddə üç`; otherwise a division: `24/7` -> `... bölü yeddi`. */
export function replaceFractions(text) {
  return text.replace(
    new RegExp(`(?<![\\d/.,])(\\d+)/(\\d+)(?![\\d/]|[.,]\\d)${SUFFIX}`, "gu"),
    (_match, a, b, suffix) => {
      const top = BigInt(a);
      const bottom = BigInt(b);
      if (top > 0n && top < bottom) {
        const head = harmonise(numberToWords(bottom), "da");
        return `${head} ${attachSuffix(numberToWords(top), suffix)}`;
      }
      return `${numberToWords(top)} bölü ${attachSuffix(numberToWords(bottom), suffix)}`;
    },
  );
}

/** `10-15` -> `on ilə on beş`; a suffix belongs to the second number. */
export function replaceRanges(text) {
  return text.replace(
    new RegExp(`(?<![${WORD}.,])(${NUMBER})\\s?[-–]\\s?(${NUMBER})${SUFFIX}${B_END}`, "gu"),
    (_match, first, second, suffix) => {
      const tail =
        suffix && ORDINAL_SUFFIXES.has(azLower(suffix)) && isDigits(second)
          ? ordinalToWords(BigInt(second))
          : attachSuffix(speakNumber(second), suffix);
      return `${speakNumber(first)} ilə ${tail}`;
    },
  );
}

/** `5-ci` -> `beşinci`; `10-da` -> `onda` (the suffix already harmonises). */
export function replaceSuffixedNumbers(text) {
  return text.replace(
    new RegExp(`(?<![${WORD}.,])(${NUMBER})-(\\p{L}[${WORD}]*)`, "gu"),
    (_match, value, suffix) =>
      ORDINAL_SUFFIXES.has(azLower(suffix)) && isDigits(value)
        ? ordinalToWords(BigInt(value))
        : `${speakNumber(value)}${suffix}`,
  );
}

export function replaceBareNumbers(text) {
  // "Su-27", "COVID-19": that dash joins a name to a number; it is no minus.
  text = text.replace(/(?<=\p{L})-(?=\d)/gu, " ");
  text = text.replace(new RegExp(`(?<![${WORD}+])\\+(?=\\d)`, "gu"), "üstəgəl ");
  return text.replace(
    new RegExp(`(?:(?<![${WORD}])(-))?(${NUMBER})`, "gu"),
    (_match, sign, raw) => {
      const spoken =
        isDigits(raw) && raw.length > 1 && raw.startsWith("0")
          ? speakDigits(raw)
          : speakNumber(raw);
      return sign ? `mənfi ${spoken}` : spoken;
    },
  );
}
