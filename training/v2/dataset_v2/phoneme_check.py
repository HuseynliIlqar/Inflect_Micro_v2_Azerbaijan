"""Run dataset-v2 lines through the stress frontend and check every phoneme against the model's
178-symbol inventory, so nothing reaches the teacher that the model cannot read.

    <azerbaycan-tts venv python> phoneme_check.py out/*.jsonl

Reports lines whose phonemisation fails or yields symbols outside the inventory, and the
share of words whose stress the frontend puts before the last syllable.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from glob import glob
from pathlib import Path

HOOK = Path("D:/ktts/avaz-train/pod/az_stress_frontend.py")
SYMBOLS = Path("D:/ktts/avaz-train/results/final/avaz-v2/symbols.json")
VOWELS = set("aæeiɯoœuyɑ")


def load_frontend():
    spec = importlib.util.spec_from_file_location("az_stress_frontend_check", HOOK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_frontend("az")


def early_stress(phonemes: str) -> tuple[int, int]:
    early = total = 0
    for word in phonemes.split():
        vowels = [i for i, ch in enumerate(word) if ch in VOWELS]
        mark = word.find("ˈ")
        if len(vowels) < 2 or mark < 0:
            continue
        total += 1
        stressed = next((i for i in vowels if i > mark), None)
        early += stressed is not None and stressed != vowels[-1]
    return early, total


def main() -> None:
    frontend = load_frontend()
    symbols = set(json.loads(SYMBOLS.read_text(encoding="utf-8"))["symbols"])
    files = [Path(p) for arg in sys.argv[1:] for p in sorted(glob(arg))]
    bad = early = words = lines = 0
    for path in files:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            text = json.loads(line)["text"]
            lines += 1
            try:
                phonemes = frontend.phonemize(frontend.normalize(text))
            except Exception as error:  # report and continue
                bad += 1
                print(f"  {path.name}:{number}: frontend error {error!r} | {text[:80]}")
                continue
            unknown = sorted(set(phonemes) - symbols)
            if unknown:
                bad += 1
                print(f"  {path.name}:{number}: symbols outside the inventory {unknown} | {text[:80]}")
            e, t = early_stress(phonemes)
            early, words = early + e, words + t
    print(f"TOTAL: {lines} lines, {bad} with problems, early-stress words {early}/{words} ({early / max(words, 1):.1%})")


if __name__ == "__main__":
    main()
