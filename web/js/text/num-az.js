/**
 * Azerbaijani number words, matching `num2words(lang="az")` exactly.
 *
 * The Python side calls num2words; there is no equivalent in JavaScript, so
 * this reproduces it. "Matching exactly" includes the quirks -- 11000 is
 * "on min", not "on bir min", because a trailing "bir" is dropped inside the
 * thousands group. tests/test_num_az.mjs checks every integer up to 10,000 and
 * a wide sample above it against the Python output.
 */

const UNITS = [
  "sıfır", "bir", "iki", "üç", "dörd",
  "beş", "altı", "yeddi", "səkkiz", "doqquz",
];

const TENS = [
  "", "on", "iyirmi", "otuz", "qırx",
  "əlli", "altmış", "yetmiş", "səksən", "doxsan",
];

// Scale words, smallest first. Each is 1000x the previous one.
const SCALES = ["", "min", "milyon", "milyard", "trilyon", "katrilyon"];

const NEGATIVE = "mənfi";

/** Front and back vowels, for the four-way harmony an ordinal suffix follows. */
const VOWELS = "aıoueəiöü";
const SUFFIX_BY_VOWEL = {
  a: "ıncı", ı: "ıncı",
  e: "inci", ə: "inci", i: "inci",
  o: "uncu", u: "uncu",
  ö: "üncü", ü: "üncü",
};

/** 0-999 as words. Returns "" for 0 so groups can be skipped. */
function groupToWords(value) {
  const parts = [];
  const hundreds = Math.floor(value / 100);
  const tens = Math.floor((value % 100) / 10);
  const units = value % 10;
  if (hundreds) {
    // 100 is "yüz", not "bir yüz".
    if (hundreds > 1) parts.push(UNITS[hundreds]);
    parts.push("yüz");
  }
  if (tens) parts.push(TENS[tens]);
  if (units) parts.push(UNITS[units]);
  return parts.join(" ");
}

/**
 * A whole number as Azerbaijani words.
 *
 * Accepts a Number or a BigInt; anything past Number.MAX_SAFE_INTEGER should
 * come in as a BigInt so the digits survive.
 */
export function numberToWords(value) {
  let n = typeof value === "bigint" ? value : BigInt(Math.trunc(value));
  if (n === 0n) return UNITS[0];

  const negative = n < 0n;
  if (negative) n = -n;

  // Split into 1000-groups, least significant first.
  const groups = [];
  while (n > 0n) {
    groups.push(Number(n % 1000n));
    n /= 1000n;
  }
  if (groups.length > SCALES.length) {
    // Beyond katrilyon num2words would keep going; we do not, and saying the
    // digits is better than saying something wrong.
    return String(value);
  }

  const parts = [];
  for (let index = groups.length - 1; index >= 0; index -= 1) {
    const group = groups[index];
    if (!group) continue;
    let words = groupToWords(group);
    if (index === 1 && group % 10 === 1) {
      // The thousands quirk: 1000 is "min", 11000 is "on min".
      words = words.slice(0, words.length - UNITS[1].length).trim();
    }
    if (words) parts.push(words);
    if (index > 0) parts.push(SCALES[index]);
  }

  const spoken = parts.join(" ");
  return negative ? `${NEGATIVE} ${spoken}` : spoken;
}

/** The ordinal suffix a word takes, from its last vowel. */
function ordinalSuffix(word) {
  let vowel = "";
  for (const character of word) {
    if (VOWELS.includes(character)) vowel = character;
  }
  const suffix = SUFFIX_BY_VOWEL[vowel] || "ıncı";
  // A word ending in a vowel loses the suffix's own leading vowel:
  // iki -> ikinci, not ikiinci.
  return VOWELS.includes(word[word.length - 1]) ? suffix.slice(1) : suffix;
}

/** A whole number as an Azerbaijani ordinal: 5 -> "beşinci". */
export function ordinalToWords(value) {
  const words = numberToWords(value);
  const cut = words.lastIndexOf(" ");
  const head = cut === -1 ? "" : words.slice(0, cut + 1);
  const last = cut === -1 ? words : words.slice(cut + 1);
  return head + last + ordinalSuffix(last);
}
