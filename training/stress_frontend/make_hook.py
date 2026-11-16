"""Write dist/az_stress_frontend.py: the stress rules, the frontend class and the embedded lexicon in one file.

Usage: python make_hook.py BASE_SYMBOLS_JSON
"""
from __future__ import annotations

import base64
import json
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "dist" / "az_stress_frontend.py"


def main(argv: list[str]) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    symbols = json.loads(Path(argv[0]).read_text(encoding="utf-8"))["symbols"]
    lexicon = {name: sorted((HERE / "lexicon" / f"{name}.txt").read_text(encoding="utf-8").split()) for name in ("verbs", "lemmas", "forms")}
    lexicon["forms"] = sorted(set(lexicon["forms"]) - set(lexicon["lemmas"]))
    blob = base64.b85encode(zlib.compress(json.dumps(lexicon, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), 9)).decode("ascii")
    source = (HERE / "az_stress.py").read_text(encoding="utf-8") + (HERE / "frontend_part.py").read_text(encoding="utf-8")
    source += f"\n\n_BASE_SYMBOLS = {json.dumps(symbols, ensure_ascii=False)}\n"
    source += f'_LEXICON_BLOB = "{blob}"\n'
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(source, encoding="utf-8")
    print(f"{OUT}: {OUT.stat().st_size / 1024:.0f} KB, symbols {len(symbols)}, verbs {len(lexicon['verbs'])}, lemmas {len(lexicon['lemmas'])}, extra forms {len(lexicon['forms'])}")


if __name__ == "__main__":
    main(sys.argv[1:])
