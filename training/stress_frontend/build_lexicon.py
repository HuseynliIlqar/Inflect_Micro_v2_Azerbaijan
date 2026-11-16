"""Build the lexicon files: dictionary headwords, verb stems and word forms seen in running text.

Sources: the official 2021 spelling dictionary (LocalDoc/azerbaijani_spelling_dictionary_2021, MIT)
and the word counts in lexicon/vocab.tsv.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from az_stress import az_lower

HERE = Path(__file__).resolve().parent / "lexicon"
MIN_FORM_COUNT = 3
SHORT_VERBS = frozenset({"et", "ol", "al", "de", "ye", "iç", "öl", "aç", "at", "as", "uç", "üz", "aş", "ək", "ər", "oy", "öp", "ov", "an", "en", "ud", "az", "öt", "əy", "ax", "yu"})
# Nouns that end in -maq/-mək and are not infinitives.
NOT_INFINITIVES = frozenset({"kömək", "barmaq", "qaymaq", "mərcimək", "damaq", "toxmaq", "əmək", "çaxmaq", "qarmaq", "dırmaq"})
INFINITIVE = re.compile(r"^(?P<stem>.+)m[aə][qk]$")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    lemmas = {az_lower(line.strip()) for line in (HERE / "spelling_dictionary.txt").read_text(encoding="utf-8").splitlines() if line.strip()}
    verbs = set(SHORT_VERBS)
    for word in lemmas:
        match = INFINITIVE.match(word)
        if match and word not in NOT_INFINITIVES and len(match.group("stem")) >= 3 and " " not in word:
            verbs.add(match.group("stem"))
    forms = set()
    for line in (HERE / "vocab.tsv").read_text(encoding="utf-8").splitlines():
        word, count = line.split("\t")
        if int(count) >= MIN_FORM_COUNT:
            forms.add(word)
    for name, words in (("lemmas", lemmas), ("verbs", verbs), ("forms", forms)):
        (HERE / f"{name}.txt").write_text("\n".join(sorted(w for w in words if " " not in w)) + "\n", encoding="utf-8")
        print(f"{name}: {len(words)}")


if __name__ == "__main__":
    main()
