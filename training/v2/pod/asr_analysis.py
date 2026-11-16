"""Where the pronunciation errors are: Whisper results from eval_asr.py broken down by sentence
type and length, per letter (which sounds the model loses more often than the original
recordings do) and per word, plus Whisper on the long test texts.

    LD_LIBRARY_PATH=... /workspace/venv_asr/bin/python asr_analysis.py [--out /workspace/asr_eval]
Writes <out>/asr_analysis.json.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from eval_asr import edit_distance, transcribe, word_list

MODEL_VERSION = "step-00030000"
_DIGIT = re.compile(r"\d")


def sentence_type(text: str) -> str:
    return {"?": "question", "!": "exclamation"}.get(text.rstrip()[-1:], "statement")


def length_bin(text: str) -> str:
    words = len(text.split())
    return "1-5" if words <= 5 else "6-10" if words <= 10 else "11-15" if words <= 15 else "16+"


def align(ref: list, hyp: list) -> list[tuple[str, object, object]]:
    """Levenshtein alignment as (op, ref_item, hyp_item) with op in ok/sub/del/ins."""
    n, m = len(ref), len(hyp)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        cost[i][0] = i
    for j in range(m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(cost[i - 1][j] + 1, cost[i][j - 1] + 1, cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("ok" if ref[i - 1] == hyp[j - 1] else "sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, hyp[j - 1]))
            j -= 1
    return ops[::-1]


def wer_of(rows: list[dict], version: str) -> dict:
    words = [(word_list(r["text"]), word_list(r[version])) for r in rows]
    total = sum(len(ref) for ref, _ in words)
    return {"clips": len(rows), "wer": round(sum(edit_distance(a, b) for a, b in words) / max(total, 1), 4)}


def breakdown(rows: list[dict], versions: list[str], key) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[key(row["text"])].append(row)
    return {name: {v: wer_of(group, v) for v in versions} for name, group in sorted(groups.items())}


def letter_errors(rows: list[dict], version: str) -> tuple[dict, Counter]:
    seen, wrong, swaps = Counter(), Counter(), Counter()
    for row in rows:
        ref, hyp = " ".join(word_list(row["text"])), " ".join(word_list(row[version]))
        for op, r, h in align(list(ref), list(hyp)):
            if r is None or r == " ":
                continue
            seen[r] += 1
            if op != "ok":
                wrong[r] += 1
                swaps[f"{r}>{h if h else '∅'}"] += 1
    return {ch: (seen[ch], wrong[ch] / seen[ch]) for ch in seen}, swaps


def model_only_words(rows: list[dict], version: str) -> list[dict]:
    """Words the original recording got through Whisper but the model did not."""
    missed, examples = Counter(), {}
    for row in rows:
        ref = word_list(row["text"])
        data_ok = {r for op, r, _ in align(ref, word_list(row["dataset"])) if op == "ok"}
        for op, r, h in align(ref, word_list(row[version])):
            if r is not None and op != "ok" and r in data_ok:
                missed[r] += 1
                examples.setdefault(r, h or "∅")
    return [{"word": w, "count": c, "heard_as": examples[w]} for w, c in missed.most_common(40)]


def long_texts(out: Path, root: Path) -> dict:
    from faster_whisper import WhisperModel

    texts = json.loads((root / "texts.json").read_text(encoding="utf-8"))
    engine = WhisperModel("hasan0v/whisper-turbo-az-ct2", device="cuda", compute_type="float16")
    result = {}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        result[folder.name] = []
        for number, text in enumerate(texts):
            ref = " ".join(text["paragraphs"])
            hyp = transcribe(engine, str(folder / f"{number:02d}.wav"))
            ref_words, hyp_words = word_list(ref), word_list(hyp)
            result[folder.name].append({"title": text["title"], "hyp": hyp,
                                        "wer": round(edit_distance(ref_words, hyp_words) / len(ref_words), 4)})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/workspace/asr_eval"))
    parser.add_argument("--longtest", type=Path, default=Path("/workspace/longtest"))
    args = parser.parse_args()
    rows = json.loads((args.out / "asr_rows.json").read_text(encoding="utf-8"))
    versions = [k for k in rows[0] if k != "text"]
    # Whisper writes many numbers as digits while the reference spells them out; those rows
    # measure Whisper's formatting, not pronunciation, so the main figures leave them out.
    clean = [r for r in rows if not any(_DIGIT.search(r[v]) for v in versions)]

    model_letters, model_swaps = letter_errors(clean, MODEL_VERSION)
    data_letters, data_swaps = letter_errors(clean, "dataset")
    base_letters, _ = letter_errors(clean, "base")
    letters = sorted(
        ({"letter": ch, "count": n, "model": round(rate, 4), "dataset": round(data_letters.get(ch, (0, 0.0))[1], 4),
          "base": round(base_letters.get(ch, (0, 0.0))[1], 4),
          "excess": round(rate - data_letters.get(ch, (0, 0.0))[1], 4)}
         for ch, (n, rate) in model_letters.items() if n >= 30),
        key=lambda item: -item["excess"])

    report = {
        "rows": len(rows), "clean_rows": len(clean),
        "overall_clean": {v: wer_of(clean, v) for v in versions},
        "by_type": breakdown(clean, versions, sentence_type),
        "by_length": breakdown(clean, versions, length_bin),
        "letters": letters,
        "model_swaps": model_swaps.most_common(25),
        "dataset_swaps": data_swaps.most_common(15),
        "model_only_words": model_only_words(clean, MODEL_VERSION),
        "long_texts": long_texts(args.out, args.longtest),
    }
    (args.out / "asr_analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("clean_rows", "overall_clean", "by_type", "by_length")}, indent=1))
    print("letters", json.dumps(letters[:12], ensure_ascii=False))
    print("swaps", model_swaps.most_common(12))
    print("model-only words", json.dumps(report["model_only_words"][:15], ensure_ascii=False))
    print("long", json.dumps({k: [x["wer"] for x in v] for k, v in report["long_texts"].items()}))


if __name__ == "__main__":
    main()
