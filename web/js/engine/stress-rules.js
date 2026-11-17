/**
 * Which syllable of an Azerbaijani word carries the stress -- a port of the
 * word rules in model/frontend_hook.py, the stress frontend the checkpoint was
 * trained with.
 *
 * eSpeak NG stresses the last syllable of almost every word. Azerbaijani does
 * too, but the negative -ma/-mə, person endings after a tense, the question
 * particle, -dır, -kən, -la, several adverbs, place names, -iya/-ika loans and
 * intensive adjectives are exceptions. A rule only fires when it is sure;
 * otherwise eSpeak's stress is kept (`null`).
 *
 * Kept line for line with the Python, so tests/golden/stress.json (written by
 * tools/build_web_stress.py) can hold the two equal. The word lists below are
 * language data, copied as they are.
 */

const A = "[aə]";
const I = "[ıiuü]";

export const VOWELS = new Set("aıoueəiöü");

// Words that never carry an accent of their own.
export const DROP_ACCENT = new Set(["və", "bu", "o", "da", "də", "ya", "həm", "biz", "siz", "mən", "sən", "ki", "hər", "mı", "mi", "mu", "mü", "idi", "imiş"]);
// Words that keep only a weak accent.
export const WEAK_ACCENT = new Set(["ilə", "isə", "tam", "onun", "onların", "onlar", "mənim", "sənin", "sizin", "bizim"]);

// Words stressed on a syllable other than the last, as syllable index from the start.
const EXCEPTIONS = new Map(Object.entries({
  sonra: 0, indi: 0, amma: 0, ancaq: 0, lakin: 0, "çünki": 0, "hətta": 0, "dünən": 0,
  "yalnız": 0, "axı": 0, "bəzən": 0, sanki: 0, yoxsa: 0, deyil: 0, daha: 0, "əgər": 0, "bəli": 0,
  ona: 0,
}));
const POSTPOSITION_BASES = [
  "nəticəsin", "sayəsin", "ərzin", "üzərin", "için", "haqqın", "barəsin", "arasın", "altın", "üstün",
  "yanın", "qarşısın", "ətrafın", "daxilin", "xaricin", "əsasın", "tərəfin", "müddətin", "çərçivəsin",
  "ortasın", "içərisin", "tərkibin", "nəzdin",
];
const POSTPOSITION_CASES = ["da", "də", "dan", "dən", "a", "ə"];
const POSTPOSITIONS = new Set(POSTPOSITION_BASES.flatMap((base) => POSTPOSITION_CASES.map((end) => base + end)));
const PLACE_EXCEPTIONS = new Map(Object.entries({
  "şəki": 0, quba: 0, "şuşa": 0, ankara: 0, "türkiyə": 0, "isveçrə": 1, bura: 0, ora: 0, hara: 0,
}));
const KEEP_NONFINAL = new Set([
  "necə", "neçə", "niyə", "belə", "elə", "hələ", "görə", "mənə", "sənə", "bizə", "sizə",
  "bəlkə", "həmişə", "əlbəttə", "yenə", "üzrə", "nəsə", "kimsə", "öncə", "eləcə",
  "üstə", "kəsə", "birgə", "əslində", "nəticədə", "barədə", "özgə", "ilə",
]);
const UNSTRESSED_TAILS = new Set([
  "mı", "mi", "mu", "mü", "dır", "dir", "dur", "dür", "dı", "di", "du", "dü", "sa", "sə", "kən",
  "mış", "miş", "muş", "müş", "dırmı", "dirmi", "durmu", "dürmü",
]);
const CASE_ENDINGS = new Set([
  "", "da", "də", "dan", "dən", "ya", "yə", "a", "ə", "nın", "nin", "nun", "nün", "nı", "ni", "nu", "nü",
  "dakı", "dəki", "ın", "in", "ı", "i",
]);

const VERBAL_NOUN = new RegExp(`^m${A}(?:[qk]|ğa$|yə$|ğı$|yi$|s[ıi](?!n$))`, "u");
const VERB_CHAIN = new RegExp(
  `^(?<neg>m${A}(?=[yzdsmrlqkğ])|m${I}(?=r))?`
  + `(?<tense>(?<=m[ıiuü])r|y?${I}r|y?${A}c${A}[qkğy]|m${I}ş|m${I}(?=s${A}n|s${I}n${I}z)`
  + `|(?<=m[aə])z|y?${A}r|m${A}l${I}|d${I}|y?${I}b|y?${A}r${A}k|d${A}n|y?${A}nd${A}|s${A})?`
  + `(?<cop>y?d${I}(?!r)|y?m${I}ş|y?s${A}(?=[mnqk]|$))?`
  + `(?<pers>y?${A}m|s${A}n|y?${I}[qk]|s${I}n${I}z|l${A}r|d${I}r(?:l${A}r)?`
  + `|m|n|q|k|n${I}z|y?${I}n|s${I}n(?:l${A}r)?)?`
  + `(?<q>m${I})?(?<ken>y?k${A}n)?$`,
  "du",
);
const NO_PRE_STRESS_TENSES = new RegExp(`^(?:d${I}|s${A})$`, "u");
const ADVERBIAL_CA = new RegExp(`c${A}$`, "du");
const NOUN_ENDINGS = [
  new RegExp(`d${I}r(?:l${A}r)?$`, "du"),
  new RegExp(`(?:y?${A}m|s${A}n|y?${I}[qk]|s${I}n${I}z)$`, "du"),
  new RegExp(`y?l${A}$`, "du"),
  new RegExp(`y?k${A}n$`, "du"),
  new RegExp(`y?d${I}$`, "du"),
];
const LOAN = /^(?<head>.*[aıoueəiöü].*?)(?<loan>iya|ika)(?<rest>[a-zəğıöüşç]*)$/u;
const INTENSIVE = /^(?<onset>[^aıoueəiöü]+)(?<vowel>[aıoueəiöü])(?<joint>[pms])\k<onset>\k<vowel>/du;
const QUESTION_TAIL = new RegExp(`m${I}$`, "u");
const VOWEL_INTENSIVES = new Set(["apağ", "upuzun", "əpəyri", "tərtəmiz"]);
const MIN_INTENSIVE_REST = 3;
const MIN_STEM_LETTERS = 3;

/** verbs: verb stems; lemmas: dictionary headwords; forms: word forms seen in running text. */
export class Lexicon {
  constructor({ verbs, lemmas, forms }) {
    this.verbs = new Set(verbs);
    this.lemmas = new Set(lemmas);
    this.forms = new Set(forms);
  }

  known(word) {
    return this.lemmas.has(word) || this.forms.has(word);
  }
}

/** Azerbaijani lower case: I -> ı and İ -> i first, then the usual rule. */
export function azLower(text) {
  return text.replaceAll("I", "ı").replaceAll("İ", "i").toLowerCase();
}

export function vowelCount(word) {
  let count = 0;
  for (const character of word) if (VOWELS.has(character)) count += 1;
  return count;
}

const lastVowelIndex = (word, end) => vowelCount(word.slice(0, end)) - 1;

function isVerbStem(stem, rest, lexicon) {
  if (lexicon.verbs.has(stem)) return true;
  return stem.endsWith("d") && lexicon.verbs.has(`${stem.slice(0, -1)}t`) && rest.length > 0 && VOWELS.has(rest[0]);
}

/** [parsed as a verb, stressed syllable or null for the default]. */
function verbStress(word, lexicon, question, sentenceFinal) {
  for (let cut = word.length; cut > 1; cut -= 1) {
    const stem = word.slice(0, cut);
    const rest = word.slice(cut);
    if (!isVerbStem(stem, rest, lexicon)) continue;
    if (/^m[aə]$/u.test(rest)) return [true, sentenceFinal ? vowelCount(stem) - 1 : null];
    if (VERBAL_NOUN.test(rest)) return [true, null];
    const match = VERB_CHAIN.exec(rest);
    if (match === null) continue;
    if (!rest) return [true, null];
    const { groups, indices } = match;
    if (groups.neg && indices.groups.neg[1] < rest.length) return [true, vowelCount(stem) - 1];
    let unstressedAfter = Boolean(groups.cop || groups.ken || (groups.q && question));
    const tense = groups.tense;
    if (groups.pers && tense && !NO_PRE_STRESS_TENSES.test(tense)) unstressedAfter = true;
    if (tense && unstressedAfter && !NO_PRE_STRESS_TENSES.test(tense)) {
      return [true, lastVowelIndex(word, cut + indices.groups.tense[1])];
    }
    if (groups.q && question) return [true, lastVowelIndex(word, cut + indices.groups.q[0])];
    return [true, null];
  }
  return [false, null];
}

function exceptionStress(word) {
  if (EXCEPTIONS.has(word)) return EXCEPTIONS.get(word);
  if (POSTPOSITIONS.has(word)) return vowelCount(word) - 2;
  if (VOWEL_INTENSIVES.has(word)) return 0;
  for (const [base, syllable] of EXCEPTIONS) {
    if (word.startsWith(base) && UNSTRESSED_TAILS.has(word.slice(base.length))) return syllable;
  }
  for (const [place, syllable] of PLACE_EXCEPTIONS) {
    if (word.startsWith(place) && CASE_ENDINGS.has(word.slice(place.length))) return syllable;
  }
  return null;
}

function loanStress(word) {
  const match = LOAN.exec(word);
  if (match === null || vowelCount(word) < 3 || !CASE_ENDINGS.has(match.groups.rest)) return null;
  return vowelCount(match.groups.head) - 1;
}

function intensiveStress(word, lexicon) {
  const match = INTENSIVE.exec(word);
  if (match === null) return null;
  const rest = word.slice(match.indices.groups.joint[1]);
  return lexicon.lemmas.has(rest) && rest.length >= MIN_INTENSIVE_REST ? 0 : null;
}

function stemStress(stem, lexicon) {
  const inner = lexicalStress(stem, lexicon);
  return inner ?? vowelCount(stem) - 1;
}

function nounStress(word, lexicon, question) {
  const adverbial = ADVERBIAL_CA.exec(word);
  if (adverbial && !KEEP_NONFINAL.has(word)) {
    const stem = word.slice(0, adverbial.index);
    if (stem.length >= MIN_STEM_LETTERS && lexicon.known(stem) && !lexicon.verbs.has(stem)) {
      return stemStress(stem, lexicon);
    }
  }
  if (question && QUESTION_TAIL.test(word) && !lexicon.known(word.slice(0, -1)) && !lexicon.known(word)) {
    const stem = word.slice(0, -2);
    if (stem.length >= MIN_STEM_LETTERS) return stemStress(stem, lexicon);
  }
  if (lexicon.lemmas.has(word)) return null;
  for (const ending of NOUN_ENDINGS) {
    const match = ending.exec(word);
    if (match === null) continue;
    const stem = word.slice(0, match.index);
    if (stem.length < MIN_STEM_LETTERS || !lexicon.known(stem) || lexicon.verbs.has(stem)) continue;
    return stemStress(stem, lexicon);
  }
  return null;
}

/** Words ending in -ə take the final stress, except the listed ones and the unstressed -lə. */
function finalSchwa(word, lexicon) {
  if (!word.endsWith("ə") || KEEP_NONFINAL.has(word)) return null;
  if (word.endsWith("lə")) {
    // -lə is unstressed after a known word, also as -ilə (tamamilə = tamam + ilə).
    const before = word.slice(-3, -2);
    const stems = [word.slice(0, -2), "ıiuü".includes(before) ? word.slice(0, -3) : ""];
    if (stems.some((stem) => stem.length >= MIN_STEM_LETTERS && lexicon.known(stem))) return null;
  }
  return vowelCount(word) - 1;
}

/**
 * [stressed syllable or null, name of the rule that decided].
 * question: the sentence ends with "?"; sentenceFinal: the word ends its sentence.
 */
export function stressRule(input, lexicon, { question = false, sentenceFinal = false } = {}) {
  const word = azLower(input);
  if (vowelCount(word) < 2) return [null, "one syllable"];
  let found = exceptionStress(word);
  if (found !== null) return [found, "exception"];
  const [isVerb, verbSyllable] = verbStress(word, lexicon, question, sentenceFinal);
  if (isVerb) {
    const schwa = verbSyllable === null ? finalSchwa(word, lexicon) : null;
    if (schwa !== null) return [schwa, "final ə"];
    return [verbSyllable, "verb"];
  }
  found = loanStress(word);
  if (found !== null) return [found, "loan"];
  found = intensiveStress(word, lexicon);
  if (found !== null) return [found, "intensive"];
  found = nounStress(word, lexicon, question);
  if (found !== null) return [found, "noun ending"];
  found = finalSchwa(word, lexicon);
  return [found, found !== null ? "final ə" : "default"];
}

/** The stressed syllable of word (0 = first), or null to keep eSpeak's choice. */
export function lexicalStress(word, lexicon, { question = false } = {}) {
  return stressRule(word, lexicon, { question })[0];
}
