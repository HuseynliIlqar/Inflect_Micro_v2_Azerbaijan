"""Bring the checkpoint's frontend to the browser playground.

    python tools/build_web_frontend.py

The checkpoint in ``model/`` was trained on eSpeak NG phonemes, produced by
``phonemizer`` (punctuation preserved) and rewritten by
``model/frontend_hook.py``. The playground runs eSpeak NG in the browser, so it
needs both steps; ``web/js/engine/punctuation.js`` and ``stress*.js`` are those
steps, ported. This writes the files the port depends on, from the vendored
hook and the installed phonemizer, so neither is copied by hand:

- ``web/js/engine/stress-lexicon.js``: the hook's embedded word lists, as
  base64 of the same zlib bytes;
- ``web/tests/golden/stress.json``: what the hook does to real text -- eSpeak's
  raw phonemes and the hook's output for each chunk, plus the stress rule for
  each word -- which ``web/tests/test_stress.mjs`` holds the port to;
- ``web/tests/golden/punctuation.json``: how phonemizer cuts each line at its
  punctuation and puts the marks back, for ``web/tests/test_punctuation.mjs``.

Rerun it whenever ``model/frontend_hook.py`` or phonemizer changes. Needs the
model's own frontend dependencies (phonemizer, espeakng-loader), not weights.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import re
import sys
import unicodedata
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aztts.console import use_utf8  # noqa: E402

use_utf8()

from aztts import chunk_text, normalize_az  # noqa: E402

HOOK = PROJECT_ROOT / "model" / "frontend_hook.py"
LEXICON_JS = PROJECT_ROOT / "web" / "js" / "engine" / "stress-lexicon.js"
GOLDEN = PROJECT_ROOT / "web" / "tests" / "golden" / "stress.json"
PUNCTUATION_GOLDEN = PROJECT_ROOT / "web" / "tests" / "golden" / "punctuation.json"
# Lines only the punctuation test needs: English, decimals, marks alone or doubled.
PUNCTUATION_EXTRA = (
    'He said "hello" (quietly) and left.', "Wait... what?", "It costs 3.50, or 3,5 euros.",
    "Well — maybe; maybe not!", "[Note] {braces} too", "¿Qué? ¡Sí!", "a , b", "?!", "  , x ,  ",
)
# One sentence in this many from dataset v2: varied text without a huge golden file.
DATASET_STRIDE = 20
# Inputs that exercise the hook's punctuation, bracket and question handling.
EDGE_CASES = (
    "Gəlirsən? Yoxsa gəlmirsən?",
    "Sən də gəlirsənmi?",
    "Tərpətmə! Aşma.",
    "Qidalanma, idman və yuxu vacibdir.",
    "Bu (mötərizədəki) söz — tire ilə ayrılıb.",
    "«Sitat» və “başqa sitat” burada.",
    "Rusiyada, Amerikada və Türkiyədə.",
    "Qapqara, yamyaşıl, büsbütün.",
    "Müəllimdir, maşınla gəldi, gəlmədi.",
    "— Salam!",
    "...",
)


def load_hook() -> ModuleType:
    """The vendored hook, loaded the way the model package loads it: from its file."""
    spec = importlib.util.spec_from_file_location("az_frontend_hook", HOOK)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {HOOK}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_texts() -> list[str]:
    """Sentences the playground will meet: demo, listening, stress-review and dataset text."""
    texts = list(EDGE_CASES)
    import say  # the demo sentences

    texts += [text for _, text in say.DEMO]
    for path in ("training/v2/pod/listen_sentences.json", "training/v3/monitor/listen_sentences_v3.json"):
        data = json.loads((PROJECT_ROOT / path).read_text(encoding="utf-8"))
        texts += [item if isinstance(item, str) else item.get("text", "") for item in data]
    for path in ("training/stress_frontend/tests/stress_items.json",
                 "training/stress_frontend/native_review/review2/stress_items.json"):
        texts += [item["sentence"] for item in json.loads((PROJECT_ROOT / path).read_text(encoding="utf-8"))]
    lines = (PROJECT_ROOT / "training/v2/dataset_v2/sentences_v2.jsonl").read_text(encoding="utf-8").splitlines()
    texts += [json.loads(line)["text"] for line in lines[::DATASET_STRIDE] if line.strip()]
    texts += list(json.loads((PROJECT_ROOT / "web/tests/golden/normalise.json").read_text(encoding="utf-8")))
    return [text for text in texts if text]


def chunks_of(texts: list[str]) -> list[str]:
    """The chunks the engine hands the frontend, each once, in first-seen order."""
    seen: dict[str, None] = {}
    for text in texts:
        normalized = normalize_az(text)
        for chunk in (chunk_text(normalized) if normalized else []):
            seen.setdefault(chunk, None)
    return list(seen)


def case(hook: ModuleType, frontend: object, chunk: str) -> dict[str, str] | None:
    normalized = hook._toolkit_normalize(chunk)
    if not normalized:
        return None
    raw = frontend._backend.phonemize([normalized], separator=frontend._separator, strip=True, njobs=1)[0]
    raw = re.sub(r"\s+", " ", unicodedata.normalize("NFC", raw)).strip()
    return {"text": chunk, "raw": raw, "out": frontend.phonemize(normalized)}


def word_rules(hook: ModuleType, lexicon: object, cases: list[dict[str, str]]) -> dict[str, list]:
    """stress_rule for every distinct word, plain and in a question."""
    words = sorted({hook.az_lower(token.strip(hook.PUNCTUATION))
                    for item in cases for token in item["text"].split()} - {""})
    return {word: [list(hook.stress_rule(word, lexicon)), list(hook.stress_rule(word, lexicon, question=True)),
                   list(hook.stress_rule(word, lexicon, sentence_final=True))] for word in words}


def punctuation_cases(lines: list[str]) -> list[list]:
    """[line, pieces, [[mark, position]...], restored] with a stand-in phonemiser that brackets each piece."""
    from phonemizer.punctuation import Punctuation
    from phonemizer.separator import Separator

    punctuation = Punctuation()
    separator = Separator(phone="", word=" ", syllable="")
    cases = []
    for line in lines:
        pieces, marks = punctuation.preserve([line])
        restored = Punctuation.restore([f"<{piece}>" for piece in pieces], marks, separator, strip=True)
        cases.append([line, pieces, [[mark.mark, mark.position] for mark in marks], restored[0] if restored else ""])
    return cases


def write_lexicon(hook: ModuleType) -> dict[str, int]:
    blob = hook._LEXICON_BLOB
    zlib_bytes = base64.b85decode(blob)
    LEXICON_JS.write_text(
        "// Generated by tools/build_web_frontend.py from model/frontend_hook.py -- do not edit.\n"
        "// zlib-compressed JSON {verbs, lemmas, forms}, base64; the package's lexicon_sha256 is\n"
        f"// {hashlib.sha256(blob.encode('ascii')).hexdigest()}\n"
        f'export const LEXICON_ZLIB_BASE64 = "{base64.b64encode(zlib_bytes).decode("ascii")}";\n',
        encoding="utf-8",
    )
    lexicon = hook._embedded_lexicon()
    return {"verbs": len(lexicon.verbs), "lemmas": len(lexicon.lemmas), "forms": len(lexicon.forms)}


def main() -> None:
    hook = load_hook()
    counts = write_lexicon(hook)
    frontend = hook.create_frontend("az")
    cases = [item for chunk in chunks_of(source_texts()) if (item := case(hook, frontend, chunk))]
    golden = {"lexicon": counts, "cases": cases, "words": word_rules(hook, frontend._lexicon, cases)}
    GOLDEN.write_text(json.dumps(golden, ensure_ascii=False, indent=0), encoding="utf-8")
    lines = [hook._toolkit_normalize(item["text"]) for item in cases] + list(PUNCTUATION_EXTRA)
    marks = punctuation_cases(list(dict.fromkeys(lines)))
    PUNCTUATION_GOLDEN.write_text(json.dumps(marks, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"{LEXICON_JS.relative_to(PROJECT_ROOT)}: {LEXICON_JS.stat().st_size / 1e3:.0f} kB")
    print(f"{GOLDEN.relative_to(PROJECT_ROOT)}: {len(cases)} chunks, {len(golden['words'])} words, lexicon {counts}")
    print(f"{PUNCTUATION_GOLDEN.relative_to(PROJECT_ROOT)}: {len(marks)} lines")


if __name__ == "__main__":
    main()
