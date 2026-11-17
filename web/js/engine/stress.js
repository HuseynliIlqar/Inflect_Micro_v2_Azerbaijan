/**
 * The checkpoint's frontend in the browser: eSpeak NG, then the stress
 * corrections it was trained with (model/frontend_hook.py).
 *
 * The model only ever heard phonemes that went through this rewrite, so the
 * playground must apply it too: plain eSpeak puts the stress where the model
 * did not learn it. The word rules live in ./stress-rules.js; this module
 * pairs words with eSpeak's tokens, moves the stress marks, and repeats the
 * package's own text and phoneme clean-up around it.
 */

import { DROP_ACCENT, Lexicon, WEAK_ACCENT, azLower, stressRule, vowelCount } from "./stress-rules.js";
import { SYMBOLS } from "./symbols.js";

const PRIMARY = "ˈ";
const SECONDARY = "ˌ";
const IPA_VOWELS = new Set("aæeiɯoœuy");
const PUNCTUATION = new Set(",.;:!?…\"«»“”()—-");
const CLOSING = new Set("\"”»)");
const SENTENCE_END = new Set(".!?…");
const ALLOWED = new Set(SYMBOLS);
// More than one word in this many with a different vowel count means the alignment is off.
const MAX_MISMATCH_SHARE = 10;

/** Python's str.strip(chars) / rstrip(chars). */
function strip(text, characters) {
  let start = 0;
  let end = text.length;
  while (start < end && characters.has(text[start])) start += 1;
  while (end > start && characters.has(text[end - 1])) end -= 1;
  return text.slice(start, end);
}

function rstrip(text, characters) {
  let end = text.length;
  while (end > 0 && characters.has(text[end - 1])) end -= 1;
  return text.slice(0, end);
}

const words = (text) => text.split(/\s+/u).filter(Boolean);

function splitPunctuation(token) {
  const core = strip(token, PUNCTUATION);
  if (!core) return [token, "", ""];
  const start = token.indexOf(core);
  return [token.slice(0, start), core, token.slice(start + core.length)];
}

/** Move the primary stress of an eSpeak token to the given syllable, just before its vowel. */
export function placeStress(token, syllable) {
  const [before, core, after] = splitPunctuation(token);
  const plain = core.replaceAll(PRIMARY, "").replaceAll(SECONDARY, "");
  const vowels = [];
  for (let index = 0; index < plain.length; index += 1) if (IPA_VOWELS.has(plain[index])) vowels.push(index);
  if (!(syllable >= 0 && syllable < vowels.length)) return null;
  const at = vowels[syllable];
  return before + plain.slice(0, at) + PRIMARY + plain.slice(at) + after;
}

function ipaVowelCount(token) {
  let count = 0;
  for (const character of token) if (IPA_VOWELS.has(character)) count += 1;
  return count;
}

const weaken = (token, keepSecondary) => token.replaceAll(PRIMARY, keepSecondary ? SECONDARY : "");

/** For each word, whether the sentence it belongs to ends with a question mark. */
function questionFlags(items) {
  const flags = [];
  let pending = 0;
  for (const word of items) {
    pending += 1;
    const stripped = rstrip(word, CLOSING);
    const last = stripped.at(-1);
    if (last === "?" || last === "." || last === "!" || last === "…") {
      for (let index = 0; index < pending; index += 1) flags.push(last === "?");
      pending = 0;
    }
  }
  for (let index = 0; index < pending; index += 1) flags.push(false);
  return flags;
}

/** question / sentenceFinal for each word of a text (punctuation-only words included). */
function contexts(items) {
  const flags = questionFlags(items);
  return items.map((word, index) => {
    const later = items.slice(index + 1).some((other) => strip(other, PUNCTUATION));
    const end = rstrip(word, CLOSING).at(-1);
    return { question: flags[index], sentenceFinal: SENTENCE_END.has(end) || !later };
  });
}

function restressWord(word, token, lexicon, context) {
  const bare = azLower(strip(word, PUNCTUATION));
  if (!bare) return token;
  if (DROP_ACCENT.has(bare)) return weaken(token, false);
  if (WEAK_ACCENT.has(bare)) return weaken(token, true);
  const [syllable] = stressRule(bare, lexicon, context);
  if (syllable === null || vowelCount(bare) !== ipaVowelCount(token)) return token;
  return placeStress(token, syllable) ?? token;
}

/** Rewrite eSpeak's stress marks in phonemes, using text for the words. Misaligned input is returned as is. */
export function restressSentence(text, phonemes, lexicon) {
  const every = words(text);
  const everyContext = contexts(every);
  // Punctuation-only items are left out on both sides: eSpeak drops some and keeps others.
  const kept = every
    .map((word, index) => [word, everyContext[index]])
    .filter(([word]) => strip(word, PUNCTUATION));
  const tokens = words(phonemes);
  const spoken = tokens.flatMap((token, index) => (strip(token, PUNCTUATION) ? [index] : []));
  if (!spoken.length || kept.length !== spoken.length) return phonemes;
  const pairs = kept.map(([word, context], position) => [word, context, tokens[spoken[position]]]);
  const mismatched = pairs.filter(
    ([word, , token]) => vowelCount(azLower(strip(word, PUNCTUATION))) !== ipaVowelCount(token),
  ).length;
  if (mismatched > Math.max(1, Math.floor(pairs.length / MAX_MISMATCH_SHARE))) return phonemes;
  const rebuilt = [...tokens];
  pairs.forEach(([word, context, token], position) => {
    rebuilt[spoken[position]] = restressWord(word, token, lexicon, context);
  });
  return rebuilt.join(" ");
}

const collapse = (text) => text.replace(/\s+/gu, " ").trim();

/** The package's text clean-up: NFKC, line breaks to spaces, no control characters. */
export function frontendNormalize(text) {
  const value = text.normalize("NFKC").replace(/[\r\n\t]/gu, " ").replace(/\p{C}/gu, "");
  return collapse(value);
}

/** Brackets become commas (a short pause); any symbol the model does not know is dropped. */
function declaredOnly(phonemes) {
  const kept = [...phonemes.replace(/[()]/gu, ",")].filter((character) => ALLOWED.has(character)).join("");
  return collapse(kept.replace(/,(?:\s*,)+/gu, ","));
}

/** eSpeak's raw phonemes for normalized text -> what the checkpoint is fed. */
export function finishPhonemes(normalized, raw, lexicon) {
  const clean = collapse(raw.normalize("NFC"));
  return collapse(declaredOnly(restressSentence(normalized, clean, lexicon)).normalize("NFC"));
}

/** zlib bytes -> text, with the browser's (and Node's) own decompressor. */
async function inflate(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate"));
  return new Response(stream).text();
}

let lexiconPromise = null;

/** The word lists, decoded once; the 600 kB module is fetched only when first needed. */
export function loadLexicon() {
  if (!lexiconPromise) {
    lexiconPromise = (async () => {
      const { LEXICON_ZLIB_BASE64 } = await import("./stress-lexicon.js");
      const bytes = Uint8Array.from(atob(LEXICON_ZLIB_BASE64), (character) => character.charCodeAt(0));
      return new Lexicon(JSON.parse(await inflate(bytes)));
    })().catch((error) => {
      lexiconPromise = null;
      throw error;
    });
  }
  return lexiconPromise;
}

/**
 * Wrap an eSpeak phonemiser (text -> raw phonemes) into the checkpoint's
 * frontend: text -> the phonemes the model was trained on.
 */
export function stressPhonemizer(phonemize) {
  return async (text) => {
    const normalized = frontendNormalize(text);
    if (!normalized) return "";
    const [raw, lexicon] = await Promise.all([phonemize(normalized), loadLexicon()]);
    return finishPhonemes(normalized, raw, lexicon);
  };
}
