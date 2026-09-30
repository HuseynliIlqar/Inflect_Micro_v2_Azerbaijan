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
  "km/saat": "kilometr saatda", "km/s": "kilometr saniyədə",
  "m/s": "metr saniyədə",
  mln: "milyon", mlrd: "milyard", trln: "trilyon",
};

const CURRENCIES = {
  azn: "manat", "₼": "manat",
  usd: "dollar", $: "dollar",
  eur: "avro", "€": "avro",
  gbp: "funt sterlinq", "£": "funt sterlinq",
  rub: "rubl", "₽": "rubl",
  try: "Türkiyə lirəsi", "₺": "Türkiyə lirəsi",
};

// The coin an amount's two decimal places count: "19,99 AZN" is said
// "on doqquz manat doxsan doqquz qəpik".
const CURRENCY_SUBUNITS = {
  manat: "qəpik", dollar: "sent", avro: "sent", rubl: "qəpik",
  "funt sterlinq": "pens", "Türkiyə lirəsi": "quruş",
};

const SYMBOLS = {
  "&": " və ", "№": " nömrə ", "§": " paraqraf ",
  "×": " vurulsun ", "÷": " bölünsün ", "±": " artı mənfi ",
  "≈": " təxminən ", "≤": " kiçik və ya bərabər ",
  "≥": " böyük və ya bərabər ", "=": " bərabərdir ",
  "°": " dərəcə ",
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

// Nouns after which a lone I, V or X is a numeral: "V əsr", "X sinif". A lone
// letter is otherwise left alone -- it is far more often just a letter.
const ROMAN_NOUNS = [
  "əsr", "sinif", "kurs", "hissə", "fəsil", "bölmə", "cild", "maddə",
  "dərəcə", "qrup", "rüb", "yarımil", "mərtəbə",
];

const ORDINAL_SUFFIXES = new Set([
  "cı", "ci", "cu", "cü",
  "ncı", "nci", "ncu", "ncü",
  "ıncı", "inci", "uncu", "üncü",
]);

const ROMAN_VALUES = { I: 1, V: 5, X: 10, L: 50 };
const LONE_ROMAN = { I: 1, V: 5, X: 10 };
// Deliberately limited to I/V/X/L so ordinary upper-case words (DVD, MIX, CD)
// are never mistaken for numerals.
const ROMAN_RE = /^(?=[IVXL]{2,})(XL|L?X{0,4})(IX|IV|V?I{0,3})$/;
const ROMAN_MAX = 50;

// Optional Azerbaijani suffix written after a dash: "2024-cü", "ATM-də".
const SUFFIX = `(?:-(\\p{L}[${WORD}]*))?`;

// A number, and a number or a range of them: "2,5", "10-15".
const NUMBER = "\\d+(?:[.,]\\d+)?";
const AMOUNT = `${NUMBER}(?:\\s?[-–]\\s?${NUMBER})?`;
// A scale word written between an amount and its currency: "5 mln AZN".
const SCALE = "(?:\\s?(?:mln|mlrd|trln|milyon|milyard|trilyon|min)\\.?)?";

const VOWELS = "aıoueəiöü";
const BACK_VOWELS = "aıou";
// The four-way vowel (ı/i/u/ü) that follows each vowel.
const FOUR_WAY = { a: "ı", ı: "ı", o: "u", u: "u", e: "i", ə: "i", i: "i", ö: "ü", ü: "ü" };

/** `iyun` -> `[iİ]yun`, the same pattern the Python side builds. */
function eitherCase(word) {
  const head = word[0];
  const upper = head === "i" ? "İ" : head.toUpperCase();
  return `[${head}${upper}]${word.slice(1)}`;
}

const MONTH_RE = MONTHS.map(eitherCase).join("|");
// "il" right after a date already names the year: "01.09.1939 ildə".
const YEAR_WORD_RE = new RegExp(`^\\s+il(?:in|də|dən|i|ə)?${B_END}`, "u");

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
function speakNumber(raw) {
  if (raw.includes(",") || raw.includes(".")) {
    const [whole, fraction] = raw.split(/[.,]/u, 2);
    return decimalToWords(whole, fraction);
  }
  return numberToWords(BigInt(raw));
}

/** `007` -> `sıfır sıfır yeddi`: leading zeros are read, not dropped. */
function speakDigits(raw) {
  const rest = raw.replace(/^0+/u, "");
  const zeros = Array(raw.length - rest.length).fill("sıfır");
  return [...zeros, ...(rest ? [numberToWords(BigInt(rest))] : [])].join(" ");
}

function atSentenceStart(text, index) {
  const before = text.slice(0, index).trimEnd();
  return !before || ".!?".includes(before.at(-1));
}

const isDigits = (text) => /^\d+$/u.test(text);

/**
 * `1 000 000`, `1.000.000` and `1,000,000` become plain digit runs. A leading
 * `0` is never a thousands group, and a single `,000` stays a decimal comma.
 */
function stripGroupSeparators(text) {
  const join = (match) => match.replace(/[ .,]/gu, "");
  text = text.replace(/(?<![\d.,])[1-9]\d{0,2}(?: \d{3})+(?!\d)/gu, join);
  text = text.replace(/(?<![\d.,])[1-9]\d{0,2}(?:\.\d{3})+(?!\d|\.\d)/gu, join);
  return text.replace(/(?<![\d.,])[1-9]\d{0,2}(?:,\d{3}){2,}(?!\d|,\d)/gu, join);
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
function replacePhoneNumbers(text) {
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

function replaceDates(text) {
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
function replaceDottedChains(text) {
  return text.replace(/(?<![\d.,])\d+(?:\.\d+){2,}(?!\d)/gu, (match) =>
    match
      .split(".")
      .map((part) => numberToWords(BigInt(part)))
      .join(" nöqtə "),
  );
}

function replaceTimes(text) {
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

/** A score or ratio (`3:2`) is two numbers; `2x3` is a product. */
function replaceOperators(text) {
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
function replaceCurrency(text) {
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

function replacePercent(text) {
  text = text.replace(new RegExp(`%\\s?(${AMOUNT})`, "gu"), "$1 faiz");
  return text.replace(
    new RegExp(`(${AMOUNT})\\s?%${SUFFIX}`, "gu"),
    (_match, amount, suffix) => `${amount} ${harmonise("faiz", suffix)}`,
  );
}

function replaceUnits(text) {
  const keys = Object.keys(UNITS_TABLE).sort((a, b) => b.length - a.length);
  const pattern = keys.map(escapeRegExp).join("|");
  return text.replace(
    new RegExp(`(\\d)\\s?(${pattern})${SUFFIX}(?![${WORD}])`, "giu"),
    (_match, digit, unit, suffix) =>
      `${digit} ${harmonise(UNITS_TABLE[azLower(unit)], suffix)}`,
  );
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

/** `3/4` -> `dörddə üç`; otherwise a division: `24/7` -> `... bölü yeddi`. */
function replaceFractions(text) {
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
function replaceRanges(text) {
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
function replaceSuffixedNumbers(text) {
  return text.replace(
    new RegExp(`(?<![${WORD}.,])(${NUMBER})-(\\p{L}[${WORD}]*)`, "gu"),
    (_match, value, suffix) =>
      ORDINAL_SUFFIXES.has(azLower(suffix)) && isDigits(value)
        ? ordinalToWords(BigInt(value))
        : `${speakNumber(value)}${suffix}`,
  );
}

function replaceBareNumbers(text) {
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
