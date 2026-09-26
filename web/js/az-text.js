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

import { numberToWords, ordinalToWords } from "./num-az.js";

// Word boundaries, Unicode-aware, matching Python's `\b`.
const WORD = "\\p{L}\\p{N}_";
const B_START = `(?<![${WORD}])`;
const B_END = `(?![${WORD}])`;

const LETTER_NAMES = {
  a: "a", b: "be", c: "ce", ç: "çe", d: "de", e: "e",
  ə: "ə", f: "fe", g: "ge", ğ: "ğe", h: "he", x: "xe",
  ı: "ı", i: "i", j: "je", k: "ka", q: "qe", l: "el",
  m: "em", n: "en", o: "o", ö: "ö", p: "pe", r: "er",
  s: "se", ş: "şe", t: "te", u: "u", ü: "ü", v: "ve",
  y: "ye", z: "ze",
};

const MONTHS = [
  "yanvar", "fevral", "mart", "aprel", "may", "iyun",
  "iyul", "avqust", "sentyabr", "oktyabr", "noyabr", "dekabr",
];

// Written abbreviation -> spoken form. Applied before anything else so their
// full stops never look like sentence boundaries to the chunker.
const ABBREVIATIONS = {
  "və s.": "və sairə",
  "və b.": "və başqaları",
  "və i.a.": "və iləaxır",
  "məs.": "məsələn",
  "e.ə.": "eramızdan əvvəl",
  "b.e.ə.": "bizim eradan əvvəl",
  "prof.": "professor",
  "dos.": "dosent",
  "akad.": "akademik",
  "dr.": "doktor",
  "küç.": "küçə",
  "şəh.": "şəhər",
  "r-nu": "rayonu",
  "mkr.": "mikrorayon",
  "səh.": "səhifə",
  "bax:": "bax,",
};

const UNITS_TABLE = {
  kq: "kiloqram", q: "qram", t: "ton", mq: "milliqram",
  km: "kilometr", m: "metr", sm: "santimetr", mm: "millimetr",
  l: "litr", ml: "millilitr",
  san: "saniyə", dəq: "dəqiqə", st: "ədəd",
  "kv.m": "kvadrat metr", "kub.m": "kub metr", ha: "hektar",
  kvt: "kilovatt", vt: "vatt", hz: "herts",
  gb: "giqabayt", mb: "meqabayt", kb: "kilobayt", tb: "terabayt",
  "°c": "dərəcə selsi", "°f": "dərəcə farenheyt",
};

const CURRENCIES = {
  azn: "manat", "₼": "manat",
  usd: "dollar", $: "dollar",
  eur: "avro", "€": "avro",
  gbp: "funt sterlinq", "£": "funt sterlinq",
  rub: "rubl", "₽": "rubl",
  try: "Türkiyə lirəsi", "₺": "Türkiyə lirəsi",
};

const SYMBOLS = {
  "&": " və ", "№": " nömrə ", "§": " paraqraf ",
  "×": " vurulsun ", "÷": " bölünsün ", "±": " artı mənfi ",
  "≈": " təxminən ", "≤": " kiçik və ya bərabər ",
  "≥": " böyük və ya bərabər ", "=": " bərabərdir ",
};

// Acronyms with a settled spoken form; anything else short and upper-case is
// spelled out letter by letter.
const ACRONYMS = {
  AMEA: "a em e a",
  ADA: "ada",
  ASAN: "asan",
  NATO: "nato",
  UNESCO: "yunesko",
  UNICEF: "yunisef",
  COVID: "kovid",
  PIN: "pin",
  SIM: "sim",
};

const ORDINAL_SUFFIXES = new Set([
  "cı", "ci", "cu", "cü",
  "ncı", "nci", "ncu", "ncü",
  "ıncı", "inci", "uncu", "üncü",
]);

const ROMAN_VALUES = { I: 1, V: 5, X: 10, L: 50 };
// Deliberately limited to I/V/X/L so ordinary upper-case words (DVD, MIX, CD)
// are never mistaken for numerals.
const ROMAN_RE = /^(?=[IVXL]{2,})(XL|L?X{0,4})(IX|IV|V?I{0,3})$/;
const ROMAN_MAX = 50;

// Optional Azerbaijani suffix written after a dash: "2024-cü", "ATM-də".
const SUFFIX = `(?:-(\\p{L}[${WORD}]*))?`;

function escapeRegExp(text) {
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
function attachSuffix(word, suffix) {
  return suffix ? `${word}${suffix}` : word;
}

/** `1,5` -> `bir tam beş onda`; falls back to digits when very long. */
function decimalToWords(whole, fraction) {
  const places = { 1: "onda", 2: "yüzdə", 3: "mində" }[fraction.length];
  const head = numberToWords(BigInt(whole));
  if (places === undefined) {
    const digits = [...fraction].map((d) => numberToWords(BigInt(d))).join(" ");
    return `${head} tam ${digits}`;
  }
  return `${head} tam ${numberToWords(BigInt(fraction))} ${places}`;
}

/** `1 000 000` and `1.000.000` become plain digit runs. */
function stripGroupSeparators(text) {
  text = text.replace(
    new RegExp(`(?<=\\d)[  ](?=\\d{3}${B_END})`, "gu"),
    "",
  );
  return text.replace(new RegExp(`(?<=\\d)\\.(?=\\d{3}${B_END})`, "gu"), "");
}

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

function replacePhoneNumbers(text) {
  return text.replace(/\+\d[\d\s\-()]{7,}\d/gu, (match) =>
    [...match.replace(/\D/gu, "")]
      .map((digit) => numberToWords(BigInt(digit)))
      .join(" "),
  );
}

function replaceDates(text) {
  const pattern = new RegExp(
    `${B_START}(\\d{1,2})[./](\\d{1,2})[./](\\d{4})${SUFFIX}${B_END}`,
    "gu",
  );
  return text.replace(pattern, (match, d, m, y, suffix) => {
    const day = Number(d);
    const month = Number(m);
    if (!(day >= 1 && day <= 31 && month >= 1 && month <= 12)) return match;
    const year =
      suffix && ORDINAL_SUFFIXES.has(azLower(suffix))
        ? ordinalToWords(BigInt(y))
        : attachSuffix(numberToWords(BigInt(y)), suffix);
    return `${ordinalToWords(day)} ${MONTHS[month - 1]} ${year}`;
  });
}

function replaceTimes(text) {
  const pattern = new RegExp(
    `${B_START}(\\d{1,2}):(\\d{2})${SUFFIX}${B_END}`,
    "gu",
  );
  return text.replace(pattern, (match, h, m, suffix) => {
    const hour = Number(h);
    const minute = Number(m);
    if (hour > 23 || minute > 59) return match;
    if (minute === 0) return attachSuffix(numberToWords(hour), suffix);
    const tail = attachSuffix(numberToWords(minute), suffix);
    return minute < 10
      ? `${numberToWords(hour)} sıfır ${tail}`
      : `${numberToWords(hour)} ${tail}`;
  });
}

function speakAmount(raw) {
  if (raw.includes(",") || raw.includes(".")) {
    const [whole, fraction] = raw.split(/[.,]/u, 2);
    return decimalToWords(whole, fraction);
  }
  return numberToWords(BigInt(raw));
}

function replaceCurrency(text) {
  const symbols = Object.keys(CURRENCIES)
    .filter((key) => !/^\p{L}+$/u.test(key))
    .join("");
  const codes = Object.keys(CURRENCIES)
    .filter((key) => /^\p{L}+$/u.test(key))
    .join("|");
  const amount = "\\d+(?:[.,]\\d+)?";

  text = text.replace(
    new RegExp(`([${escapeRegExp(symbols)}])\\s?(${amount})`, "gu"),
    (_match, symbol, value) => `${speakAmount(value)} ${CURRENCIES[symbol]}`,
  );
  return text.replace(
    new RegExp(
      `${B_START}(${amount})\\s?(${codes}|[${escapeRegExp(symbols)}])${B_END}`,
      "giu",
    ),
    (_match, value, unit) =>
      `${speakAmount(value)} ${CURRENCIES[azLower(unit)]}`,
  );
}

function replacePercent(text) {
  text = text.replace(
    /%\s?(\d+)/gu,
    (_match, value) => `${numberToWords(BigInt(value))} faiz`,
  );
  return text.replace(
    /(\d+)\s?%/gu,
    (_match, value) => `${numberToWords(BigInt(value))} faiz`,
  );
}

function replaceUnits(text) {
  const keys = Object.keys(UNITS_TABLE).sort((a, b) => b.length - a.length);
  const pattern = keys.map(escapeRegExp).join("|");
  return text.replace(
    new RegExp(`(\\d)\\s?(${pattern})${B_END}`, "giu"),
    (_match, digit, unit) => `${digit} ${UNITS_TABLE[azLower(unit)]}`,
  );
}

function replaceRoman(text) {
  return text.replace(
    /(?<![\p{L}\p{N}_])([IVXL]{2,})(?![\p{L}\p{N}_])(?:-(?:cı|ci|cu|cü))?/gu,
    (match, token) => {
      const value = romanToInt(token);
      if (value === null) return match;
      const word = ordinalToWords(value);
      return match[0] === match[0].toUpperCase() && /\p{Lu}/u.test(match[0])
        ? azCapitalise(word)
        : word;
    },
  );
}

function replaceSuffixedNumbers(text) {
  return text.replace(
    new RegExp(`${B_START}(\\d+)-(\\p{L}[${WORD}]*)`, "gu"),
    (_match, value, suffix) =>
      ORDINAL_SUFFIXES.has(azLower(suffix))
        ? ordinalToWords(BigInt(value))
        : `${numberToWords(BigInt(value))}${suffix}`,
  );
}

function replaceRanges(text) {
  return text.replace(
    new RegExp(`${B_START}(\\d+)\\s?[-–]\\s?(\\d+)${B_END}`, "gu"),
    (_match, from, to) =>
      `${numberToWords(BigInt(from))} ilə ${numberToWords(BigInt(to))}`,
  );
}

function replaceBareNumbers(text) {
  return text.replace(/-?\d+(?:[.,]\d+)?/gu, (raw) => {
    if (raw.includes(",") || raw.includes(".")) {
      const negative = raw.startsWith("-");
      const body = negative ? raw.slice(1) : raw;
      const [whole, fraction] = body.split(/[.,]/u, 2);
      const spoken = decimalToWords(whole, fraction);
      return negative ? `mənfi ${spoken}` : spoken;
    }
    return numberToWords(BigInt(raw));
  });
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
    ["–", "-"], ["—", "-"], ["…", "."], [" ", " "],
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
  replaceTimes,
  replaceCurrency,
  replacePercent,
  replaceUnits,
  replaceRoman,
  replaceSuffixedNumbers,
  replaceRanges,
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
