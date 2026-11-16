"""Correct the word stress that eSpeak NG writes for Azerbaijani.

eSpeak puts the stress on the last syllable of almost every word. Azerbaijani does that too,
but several common word types are exceptions, and eSpeak gets them wrong:

* the negative suffix -ma/-mə: the syllable before it is stressed (gəlmədi = GƏLmədi);
* person and copula endings after a tense marker are unstressed (gəlirəm = gəLİrəm);
* the question particle -mı, the predicative -dır, -kən and the comitative -la are unstressed
  on nouns too (müəllimdir = müəlLİMdir, maşınla = maŞINla);
* adverbs and conjunctions such as sonra, indi, amma, and some place names (Şəki, Quba);
* Russian-style loans in -iya and -ika (Rusiya = RUsiya, Amerika = aMErika);
* intensive adjectives (qapqara = QAPqara).

A rule only fires when it is sure; otherwise eSpeak's stress is kept. Clitics such as "və"
and "ki" lose their accent. The functions here are pure: they take strings and return strings.
"""
from __future__ import annotations

import re

PRIMARY = "ˈ"
SECONDARY = "ˌ"
VOWELS = frozenset("aıoueəiöü")
IPA_VOWELS = frozenset("aæeiɯoœuy")
PUNCTUATION = ",.;:!?…\"«»“”()—-"
_LOWER = str.maketrans({"I": "ı", "İ": "i"})

A = "[aə]"
I = "[ıiuü]"

# Words that never carry an accent of their own.
DROP_ACCENT = frozenset({"və", "bu", "o", "da", "də", "ya", "həm", "biz", "siz", "mən", "sən", "ki", "hər", "mı", "mi", "mu", "mü", "idi", "imiş"})
# Words that keep only a weak accent.
WEAK_ACCENT = frozenset({"ilə", "isə", "tam", "onun", "onların", "onlar", "mənim", "sənin", "sizin", "bizim"})

# Forms of address. After a comma they mark the word before as an order: "Tələsmə, qızım".
VOCATIVES = frozenset({
    "ana", "ata", "anam", "atam", "bala", "balam", "qızım", "oğlum", "canım", "əzizim", "gözüm",
    "qardaşım", "bacım", "dostum", "əmi", "dayı", "xala", "bibi", "nənə", "baba", "nənəm", "babam",
    "müəllim", "uşaqlar", "xanım", "can",
})

# Words stressed on a syllable other than the last, as syllable index from the start.
EXCEPTIONS = {
    "sonra": 0, "indi": 0, "amma": 0, "ancaq": 0, "lakin": 0, "çünki": 0, "hətta": 0, "dünən": 0,
    "yalnız": 0, "axı": 0, "bəzən": 0, "sanki": 0, "yoxsa": 0, "deyil": 0, "daha": 0, "əgər": 0, "bəli": 0,
    "ona": 0,
}
# Place names keep their stress when a case ending follows.
# Postpositions built from a noun, a possessive and a case ending stress the syllable before the
# case ending. Confirmed by the native speaker: nəticəsində, sayəsində, ərzində, üzərində, içində,
# haqqında. The rest follow the same pattern.
_POSTPOSITION_BASES = (
    "nəticəsin", "sayəsin", "ərzin", "üzərin", "için", "haqqın", "barəsin", "arasın", "altın", "üstün",
    "yanın", "qarşısın", "ətrafın", "daxilin", "xaricin", "əsasın", "tərəfin", "müddətin", "çərçivəsin",
    "ortasın", "içərisin", "tərkibin", "nəzdin",
)
_POSTPOSITION_CASES = ("da", "də", "dan", "dən", "a", "ə")
# Both harmony variants are listed; the one that breaks vowel harmony never occurs in text.
POSTPOSITIONS = frozenset(base + case for base in _POSTPOSITION_BASES for case in _POSTPOSITION_CASES)
# Place names and place words that keep their stress when a case ending follows (BUrada, TÜRkiyədə).
PLACE_EXCEPTIONS = {"şəki": 0, "quba": 0, "şuşa": 0, "ankara": 0, "türkiyə": 0, "isveçrə": 1, "bura": 0, "ora": 0, "hara": 0}
# eSpeak stresses the syllable before a final -ə in almost every word, which a native speaker
# rejected in 30 of 42 cases (ildə, ölkə, şəkildə are final). These words really are stressed
# earlier (confirmed or closely related to confirmed ones), so eSpeak's choice is kept for them.
KEEP_NONFINAL = frozenset({
    "necə", "neçə", "niyə", "belə", "elə", "hələ", "görə", "mənə", "sənə", "bizə", "sizə",
    "bəlkə", "həmişə", "əlbəttə", "yenə", "üzrə", "nəsə", "kimsə", "öncə", "eləcə",
    "üstə", "kəsə", "birgə", "əslində", "nəticədə", "barədə", "özgə", "ilə",
})
# Unstressed endings that keep the stress of an exception word where it was (deyilmi = DEyilmi).
UNSTRESSED_TAILS = frozenset({
    "mı", "mi", "mu", "mü", "dır", "dir", "dur", "dür", "dı", "di", "du", "dü", "sa", "sə", "kən",
    "mış", "miş", "muş", "müş", "dırmı", "dirmi", "durmu", "dürmü",
})
CASE_ENDINGS = frozenset({
    "", "da", "də", "dan", "dən", "ya", "yə", "a", "ə", "nın", "nin", "nun", "nün", "nı", "ni", "nu", "nü",
    "dakı", "dəki", "ın", "in", "ı", "i",
})

# Infinitive and verbal noun endings (gəlmək, gəlməyə, gəlməsi), which keep the final stress.
# "gəlməsin" is the negative imperative, so -sın at the very end is not a verbal noun.
_VERBAL_NOUN = re.compile(rf"^m{A}(?:[qk]|ğa$|yə$|ğı$|yi$|s[ıi](?!n$))")
_VERB_CHAIN = re.compile(
    rf"""^
    (?P<neg>m{A}(?=[yzdsmrlqkğ])|m{I}(?=r))?
    (?P<tense>
        (?<=m[ıiuü])r
      | y?{I}r
      | y?{A}c{A}[qkğy]
      | m{I}ş | m{I}(?=s{A}n|s{I}n{I}z)
      | (?<=m[aə])z | y?{A}r
      | m{A}l{I}
      | d{I}
      | y?{I}b | y?{A}r{A}k | d{A}n | y?{A}nd{A}
      | s{A}
    )?
    (?P<cop>y?d{I}(?!r) | y?m{I}ş | y?s{A}(?=[mnqk]|$))?
    (?P<pers>
        y?{A}m | s{A}n | y?{I}[qk] | s{I}n{I}z | l{A}r | d{I}r(?:l{A}r)?
      | m | n | q | k | n{I}z | y?{I}n | s{I}n(?:l{A}r)?
    )?
    (?P<q>m{I})?
    (?P<ken>y?k{A}n)?
    $""",
    re.VERBOSE,
)
_NO_PRE_STRESS_TENSES = re.compile(rf"^(?:d{I}|s{A})$")
# The adverbial -ca/-cə is unstressed even on dictionary words (sadəcə, təkcə, sizcə).
_ADVERBIAL_CA = re.compile(rf"c{A}$")
_NOUN_ENDINGS = (
    re.compile(rf"d{I}r(?:l{A}r)?$"),
    re.compile(rf"(?:y?{A}m|s{A}n|y?{I}[qk]|s{I}n{I}z)$"),
    re.compile(rf"y?l{A}$"),
    re.compile(rf"y?k{A}n$"),
    re.compile(rf"y?d{I}$"),
)
_LOAN = re.compile(rf"^(?P<head>.*[aıoueəiöü].*?)(?P<loan>iya|ika)(?P<rest>[a-zəğıöüşç]*)$")
# Intensive adjectives repeat the first syllable with p, m or s: qap-qara, yam-yaşıl, büs-bütün.
_INTENSIVE = re.compile(r"^(?P<onset>[^aıoueəiöü]+)(?P<vowel>[aıoueəiöü])(?P<joint>[pms])(?P=onset)(?P=vowel)")
# Vowel-initial intensives are few and look like ordinary words (aparan, aralıq), so they are listed.
VOWEL_INTENSIVES = frozenset({"apağ", "upuzun", "əpəyri", "tərtəmiz"})
MIN_INTENSIVE_REST = 3
MIN_STEM_LETTERS = 3
# More than one word in this many with a different vowel count means the alignment is off.
MAX_MISMATCH_SHARE = 10


class Lexicon:
    """verbs: verb stems; lemmas: dictionary headwords; forms: word forms seen in running text.

    A plain class rather than a dataclass: the model package loads this file without registering
    it in sys.modules, and dataclasses need that.
    """

    __slots__ = ("verbs", "lemmas", "forms")

    def __init__(self, *, verbs: frozenset[str], lemmas: frozenset[str], forms: frozenset[str]) -> None:
        self.verbs, self.lemmas, self.forms = frozenset(verbs), frozenset(lemmas), frozenset(forms)

    def known(self, word: str) -> bool:
        return word in self.lemmas or word in self.forms


def az_lower(text: str) -> str:
    return text.translate(_LOWER).lower()


def vowel_count(word: str) -> int:
    return sum(character in VOWELS for character in word)


def _last_vowel_index(word: str, end: int) -> int:
    """Index of the last vowel in word[:end], counted among all vowels of the word."""
    return vowel_count(word[:end]) - 1


def _is_verb_stem(stem: str, rest: str, lexicon: Lexicon) -> bool:
    """True for a verb stem, including the t/d alternation before a vowel (et: edir, get: gedəcək)."""
    if stem in lexicon.verbs:
        return True
    return stem.endswith("d") and stem[:-1] + "t" in lexicon.verbs and rest[:1] in VOWELS


def _verb_stress(word: str, lexicon: Lexicon, question: bool, sentence_final: bool = False) -> tuple[bool, int | None]:
    """(parsed as a verb, stressed syllable or None for the default)."""
    for cut in range(len(word), 1, -1):
        stem, rest = word[:cut], word[cut:]
        if not _is_verb_stem(stem, rest, lexicon):
            continue
        if re.fullmatch(rf"m{A}", rest):
            # Bare -ma/-mə: a negative imperative at the end of a sentence ("tərpətmə.", "aşma!"),
            # otherwise a verbal noun, as in lists ("qidalanma, idman və yuxu").
            return True, (vowel_count(stem) - 1 if sentence_final else None)
        if _VERBAL_NOUN.match(rest):
            return True, None
        match = _VERB_CHAIN.match(rest)
        if match is None:
            continue
        if not rest:
            return True, None
        if match.group("neg") and match.end("neg") < len(rest):
            return True, vowel_count(stem) - 1
        unstressed_after = any(match.group(name) for name in ("cop", "ken")) or (match.group("q") and question)
        tense = match.group("tense")
        if match.group("pers") and tense and not _NO_PRE_STRESS_TENSES.match(tense):
            unstressed_after = True
        if tense and unstressed_after and not _NO_PRE_STRESS_TENSES.match(tense):
            return True, _last_vowel_index(word, cut + match.end("tense"))
        if match.group("q") and question:
            return True, _last_vowel_index(word, cut + match.start("q"))
        return True, None
    return False, None


def _exception_stress(word: str) -> int | None:
    if word in EXCEPTIONS:
        return EXCEPTIONS[word]
    if word in POSTPOSITIONS:
        return vowel_count(word) - 2
    if word in VOWEL_INTENSIVES:
        return 0
    for base, syllable in EXCEPTIONS.items():
        if word.startswith(base) and word[len(base):] in UNSTRESSED_TAILS:
            return syllable
    for place, syllable in PLACE_EXCEPTIONS.items():
        if word.startswith(place) and word[len(place):] in CASE_ENDINGS:
            return syllable
    return None


def _loan_stress(word: str) -> int | None:
    match = _LOAN.match(word)
    if match is None or vowel_count(word) < 3 or match.group("rest") not in CASE_ENDINGS:
        return None
    return vowel_count(match.group("head")) - 1


def _intensive_stress(word: str, lexicon: Lexicon) -> int | None:
    match = _INTENSIVE.match(word)
    if match is None:
        return None
    rest = word[match.end("joint"):]
    return 0 if rest in lexicon.lemmas and len(rest) >= MIN_INTENSIVE_REST else None


def _stem_stress(stem: str, lexicon: Lexicon) -> int:
    inner = lexical_stress(stem, lexicon, question=False)
    return inner if inner is not None else vowel_count(stem) - 1


def _noun_stress(word: str, lexicon: Lexicon, question: bool) -> int | None:
    adverbial = _ADVERBIAL_CA.search(word)
    if adverbial and word not in KEEP_NONFINAL:
        stem = word[: adverbial.start()]
        if len(stem) >= MIN_STEM_LETTERS and lexicon.known(stem) and stem not in lexicon.verbs:
            return _stem_stress(stem, lexicon)
    if question and re.search(rf"m{I}$", word) and not lexicon.known(word[:-1]) and not lexicon.known(word):
        stem = word[:-2]
        if len(stem) >= MIN_STEM_LETTERS:
            inner = lexical_stress(stem, lexicon, question=False)
            return inner if inner is not None else vowel_count(stem) - 1
    if word in lexicon.lemmas:
        return None
    for ending in _NOUN_ENDINGS:
        match = ending.search(word)
        if match is None:
            continue
        stem = word[: match.start()]
        if len(stem) < MIN_STEM_LETTERS or not lexicon.known(stem) or stem in lexicon.verbs:
            continue
        inner = lexical_stress(stem, lexicon, question=False)
        return inner if inner is not None else vowel_count(stem) - 1
    return None


def _final_schwa(word: str, lexicon: Lexicon) -> int | None:
    """Words ending in -ə take the final stress, except the listed ones and the unstressed -lə."""
    if not word.endswith("ə") or word in KEEP_NONFINAL:
        return None
    if word.endswith("lə"):
        # -lə is unstressed after a known word, also as -ilə (tamamilə = tamam + ilə).
        stems = (word[:-2], word[:-3] if word[-3:-2] in "ıiuü" else "")
        if any(len(stem) >= MIN_STEM_LETTERS and lexicon.known(stem) for stem in stems):
            return None
    return vowel_count(word) - 1


def stress_rule(
    word: str, lexicon: Lexicon, *, question: bool = False, sentence_final: bool = False, clause_imperative: bool = False
) -> tuple[int | None, str]:
    """(stressed syllable or None, name of the rule that decided).

    question: the sentence ends with "?"; sentence_final: the word ends its sentence;
    clause_imperative: the word ends a clause before a comma outside a list, where a bare
    -ma/-mə form is an order ("Bu kitabı alma, o birini al").
    """
    word = az_lower(word)
    if vowel_count(word) < 2:
        return None, "one syllable"
    found = _exception_stress(word)
    if found is not None:
        return found, "exception"
    is_verb, verb_syllable = _verb_stress(word, lexicon, question, sentence_final or clause_imperative)
    if is_verb:
        schwa = _final_schwa(word, lexicon) if verb_syllable is None else None
        if schwa is not None:
            return schwa, "final ə"
        return verb_syllable, "verb"
    found = _loan_stress(word)
    if found is not None:
        return found, "loan"
    found = _intensive_stress(word, lexicon)
    if found is not None:
        return found, "intensive"
    found = _noun_stress(word, lexicon, question)
    if found is not None:
        return found, "noun ending"
    found = _final_schwa(word, lexicon)
    return found, ("final ə" if found is not None else "default")


def lexical_stress(word: str, lexicon: Lexicon, *, question: bool = False) -> int | None:
    """The stressed syllable of word (0 = first), or None to keep eSpeak's choice."""
    return stress_rule(word, lexicon, question=question)[0]


def _split_punctuation(token: str) -> tuple[str, str, str]:
    core = token.strip(PUNCTUATION)
    if not core:
        return token, "", ""
    start = token.index(core)
    return token[:start], core, token[start + len(core):]


def place_stress(token: str, syllable: int) -> str | None:
    """Move the primary stress of an eSpeak token to the given syllable, just before its vowel."""
    before, core, after = _split_punctuation(token)
    plain = core.replace(PRIMARY, "").replace(SECONDARY, "")
    vowels = [index for index, character in enumerate(plain) if character in IPA_VOWELS]
    if not 0 <= syllable < len(vowels):
        return None
    at = vowels[syllable]
    return before + plain[:at] + PRIMARY + plain[at:] + after


def _ipa_vowel_count(token: str) -> int:
    return sum(character in IPA_VOWELS for character in token)


def _weaken(token: str, *, keep_secondary: bool) -> str:
    return token.replace(PRIMARY, SECONDARY if keep_secondary else "")


def _question_flags(words: list[str]) -> list[bool]:
    """For each word, whether the sentence it belongs to ends with a question mark."""
    flags: list[bool] = []
    pending = 0
    for word in words:
        pending += 1
        stripped = word.rstrip("\"”»)")
        if stripped.endswith(("?", ".", "!", "…")):
            flags += [stripped.endswith("?")] * pending
            pending = 0
    return flags + [False] * pending


_SENTENCE_END = tuple(".!?…")
_CLOSING = "\"”»)"


def _is_vocative(token: str) -> bool:
    """A form of address ("qızım,") or a name closing its phrase ("Səlim.")."""
    bare = token.strip(PUNCTUATION)
    return az_lower(bare) in VOCATIVES or (bare[:1].isupper() and token.rstrip(_CLOSING)[-1:] in ",.!?")


# Bare stems that are far more often nouns or particles than orders.
_AMBIGUOUS_STEMS = frozenset({"at", "üz", "yaş", "qal", "dur", "bil", "ol", "de", "ye", "al", "gəl", "bax", "sür", "çək"})
# Words before an order that a verbal noun rarely has: "sən tələsmə", "heç yerə getmə".
_ORDER_MARKERS = frozenset({"sən", "siz", "heç"})


def _bare(token: str) -> str:
    return az_lower(token.strip(PUNCTUATION))


def _is_order_word(token: str, lexicon: Lexicon | None) -> bool:
    """A bare verb stem used as an order: "Dur,", "Gözlə,"."""
    bare = _bare(token)
    return lexicon is not None and bare in lexicon.verbs and bare not in _AMBIGUOUS_STEMS


def _is_verb_form(token: str, lexicon: Lexicon | None) -> bool:
    """A word the verb parser recognises: "sınar", "yıxılaram"."""
    return lexicon is not None and _verb_stress(_bare(token), lexicon, False)[0]


def _in_a_list(words: list[str], index: int, lexicon: Lexicon | None = None) -> bool:
    """A comma or "və" on either side, or a single closing noun after the comma."""
    previous = words[index - 1] if index else ""
    following = words[index + 1 : index + 4]  # "proqramlaşdırma, robot texnikası və ..."
    if _bare(previous) == "və" or (previous.endswith(",") and _bare(previous) not in VOCATIVES):
        return True
    if not following:
        return False
    if following[0].rstrip(_CLOSING).endswith(",") or "və" in [_bare(item) for item in following]:
        return True
    # "Qidalanma, idman." is a two-item list: one closing noun. A closing exclamation, verb or
    # predicate is not ("Buzun üstünə çıxma, sınar!", "Pəncərədən boylanma, təhlükəlidir!").
    closing = following[0].rstrip(_CLOSING)
    if not closing.endswith(_SENTENCE_END) or closing.endswith("!"):
        return False
    return not (_is_verb_form(following[0], lexicon) or re.search(rf"d{I}r$", _bare(following[0])))


def _sentence_start(words: list[str], index: int) -> bool:
    if index == 0:
        return True
    previous = words[index - 1].rstrip(_CLOSING)
    return previous.endswith(_SENTENCE_END + (":",))


# Case endings of an argument right before a verb: accusative (kitabı), dative (üstünə),
# locative (suda), ablative (evdən) and comitative (kibritlə).
_ARGUMENT_ENDINGS = (
    "nı", "ni", "nu", "nü", "ı", "i", "u", "ü",
    "na", "nə", "ya", "yə", "a", "ə",
    "da", "də", "ta", "tə", "dan", "dən", "tan", "tən",
    "la", "lə", "yla", "ylə",
)


def _argument_before(token: str, lexicon: Lexicon | None) -> bool:
    """A case-marked argument right before the verb: "kitabı alma", "buzun üstünə çıxma".
    Needs a known stem, since plenty of words merely end in a vowel (qırmızı, bəzi)."""
    if lexicon is None:
        return False
    word = _bare(token)
    for ending in _ARGUMENT_ENDINGS:
        if word.endswith(ending) and len(word) > len(ending) + 1 and lexicon.known(word[: -len(ending)]):
            return True
    return False


def _clause_before(words: list[str], index: int) -> list[str]:
    """The bare words from the start of the clause (after a comma, colon or sentence end) up to index."""
    start = index
    while start > 0 and not words[start - 1].rstrip(_CLOSING).endswith((",", ";", ":") + _SENTENCE_END):
        start -= 1
    return [_bare(item) for item in words[start:index]]


def _address_in_sentence(words: list[str], index: int) -> bool:
    """A form of address anywhere in the same sentence: "Ay qızım, bu qədər oxuma, ...".

    An address is set off by a comma or ends the sentence, or follows "ay"; "Babam hər dəfə
    gələndə ..." is a subject, not an address.
    """
    start = index
    while start > 0 and not words[start - 1].rstrip(_CLOSING).endswith(_SENTENCE_END):
        start -= 1
    for position in range(start, len(words)):
        token = words[position].rstrip(_CLOSING)
        set_off = token.endswith((",",) + _SENTENCE_END) or (position > 0 and _bare(words[position - 1]) == "ay")
        if _bare(token) in VOCATIVES and set_off:
            return True
        if token.endswith(_SENTENCE_END):
            break
    return False


def _sentence_ends_with_bang(words: list[str], index: int) -> bool:
    for token in words[index:]:
        stripped = token.rstrip(_CLOSING)
        if stripped.endswith(_SENTENCE_END):
            return stripped.endswith("!")
    return False


def _later_order(words: list[str], index: int, lexicon: Lexicon | None) -> bool:
    """A bare verb stem later in the same sentence, i.e. another order: "..., o birini al"."""
    if lexicon is None:
        return False
    for token in words[index + 1 :]:
        bare = _bare(token)
        if bare in lexicon.verbs and bare not in _AMBIGUOUS_STEMS:
            return True
        if token.rstrip(_CLOSING).endswith(_SENTENCE_END):
            return False
    return False


def _clause_imperative(words: list[str], index: int, lexicon: Lexicon | None = None) -> bool:
    """The word ends a clause before a comma and is an order, not a verbal noun.

    Only positive evidence makes an order, and a list always wins over it:
    "ştamplama, oyma, tökmə", "etimad və anlaşma,", "Qidalanma, idman." stay nouns.
    Evidence for an order:
    - a form of address next to it ("Tələsmə, qızım", "Ata, buraxma");
    - the start of a sentence ("Tələsmə, vaxtımız çoxdur"), unless the word is a dictionary
      noun ("Birləşmə, hərbi hissə ...");
    - "sən/siz/heç" in the same clause, or a case-marked argument right before it
      ("kitabı alma", "buzun üstünə çıxma");
    - a sentence that ends with "!" ("Kibritlə oynama, ev yanar!");
    - another order later in the sentence ("..., o birini al").
    Without evidence the old behaviour stands: "xırda oğurluq güllələnmə, ..." is a noun.
    """
    if not words[index].rstrip(_CLOSING).endswith((",", ";")):
        return False
    following = words[index + 1 : index + 2]
    previous = words[index - 1] if index else ""
    if following and _is_vocative(following[0]):
        return True
    if previous.endswith(",") and _bare(previous) in VOCATIVES:  # "Ata, buraxma, yıxılaram!"
        return True
    if previous.endswith(",") and _is_order_word(previous, lexicon):  # "Dur, qaçma, sözüm var!"
        return True
    if _in_a_list(words, index, lexicon):
        return False
    return (
        _sentence_start(words, index)
        or any(item in _ORDER_MARKERS for item in _clause_before(words, index))
        or _argument_before(previous, lexicon)
        or _sentence_ends_with_bang(words, index)
        or _address_in_sentence(words, index)
        or _later_order(words, index, lexicon)
    )


def _contexts(words: list[str], lexicon: Lexicon | None = None) -> list[dict[str, bool]]:
    """question / sentence_final / clause_imperative for each word of a text (punctuation-only words included)."""
    flags = _question_flags(words)
    contexts = []
    for index, (word, question) in enumerate(zip(words, flags)):
        later = [w for w in words[index + 1 :] if w.strip(PUNCTUATION)]
        contexts.append({
            "question": question,
            "sentence_final": word.rstrip(_CLOSING).endswith(_SENTENCE_END) or not later,
            "clause_imperative": _clause_imperative(words, index, lexicon),
        })
    return contexts


def word_context(sentence: str, word: str, lexicon: Lexicon | None = None) -> dict[str, bool]:
    """The context flags of the first occurrence of word (lowercase, no punctuation) in sentence."""
    words = sentence.split()
    for item, context in zip(words, _contexts(words, lexicon)):
        if az_lower(item.strip(PUNCTUATION)) == word:
            return context
    return {"question": sentence.rstrip().endswith("?"), "sentence_final": False, "clause_imperative": False}


def restress_word(
    word: str, token: str, lexicon: Lexicon, *, question: bool, sentence_final: bool = False, clause_imperative: bool = False
) -> str:
    bare = az_lower(word.strip(PUNCTUATION))
    if not bare:
        return token
    if bare in DROP_ACCENT:
        return _weaken(token, keep_secondary=False)
    if bare in WEAK_ACCENT:
        return _weaken(token, keep_secondary=True)
    syllable = stress_rule(
        bare, lexicon, question=question, sentence_final=sentence_final, clause_imperative=clause_imperative
    )[0]
    if syllable is None or vowel_count(bare) != _ipa_vowel_count(token):
        return token
    return place_stress(token, syllable) or token


def restress_sentence(text: str, phonemes: str, lexicon: Lexicon) -> str:
    """Rewrite eSpeak's stress marks in phonemes, using text for the words. Misaligned input is returned as is."""
    every_word = text.split()
    # Punctuation-only items are left out on both sides: eSpeak drops some (a free-standing dash)
    # and keeps others (a comma after a space), so they cannot be paired by position.
    kept = [(word, context) for word, context in zip(every_word, _contexts(every_word, lexicon)) if word.strip(PUNCTUATION)]
    tokens = phonemes.split()
    spoken = [index for index, token in enumerate(tokens) if token.strip(PUNCTUATION)]
    if not spoken or len(kept) != len(spoken):
        return phonemes
    pairs = [(word, context, tokens[index]) for (word, context), index in zip(kept, spoken)]
    mismatched = sum(vowel_count(az_lower(word.strip(PUNCTUATION))) != _ipa_vowel_count(token) for word, _, token in pairs)
    if mismatched > max(1, len(pairs) // MAX_MISMATCH_SHARE):
        return phonemes  # the counts match by accident, so the words are probably misaligned
    rebuilt = list(tokens)
    for (word, context, token), index in zip(pairs, spoken):
        rebuilt[index] = restress_word(word, token, lexicon, **context)
    return " ".join(rebuilt)
