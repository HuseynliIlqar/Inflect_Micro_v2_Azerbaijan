"""Render one line under many seeds so the best reading can be pinned down.

    python tools/seed_sweep.py "Xoş gəlmisiniz."
    python tools/seed_sweep.py --seeds 20 --variation 0.5 0.667 0.8 "Xoş gəlmisiniz."

The seed steers the model's sampling, so the same text comes out slightly
differently each time.  For a line an application repeats -- a greeting, an
error prompt -- it is worth listening to a dozen readings and hard-coding the
seed that sounds best.  This does nothing for arbitrary user text.

`variation` (the model's noise scale) trades stability against liveliness:
lower is steadier and flatter, higher is livelier and riskier.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aztts import AzTTS, ModelNotFoundError  # noqa: E402
from aztts.az_text import normalize_az  # noqa: E402
from aztts.console import use_utf8  # noqa: E402

use_utf8()

DEFAULT_PACKAGE = PROJECT_ROOT / "model"
DEFAULT_OUT = PROJECT_ROOT / "out" / "seeds"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="The line to sweep.")
    parser.add_argument("--seeds", type=int, default=10, help="Try seeds 1..N.")
    parser.add_argument(
        "--variation",
        type=float,
        nargs="+",
        default=[0.667],
        help="One or more noise scales to try (default 0.667).",
    )
    parser.add_argument("--package", "-p", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--out", "-o", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", "-d", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--raw", action="store_true", help="Skip normalisation.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    package = args.package.resolve()
    if args.seeds < 1:
        print("error: --seeds must be at least 1", file=sys.stderr)
        return 1
    for variation in args.variation:
        if not 0.0 <= variation <= 1.0:
            print(f"error: variation must be within 0..1: {variation}", file=sys.stderr)
            return 1

    text = args.text if args.raw else normalize_az(args.text)
    if not text:
        print("error: the text is empty", file=sys.stderr)
        return 1

    try:
        engine = AzTTS(package, device=args.device)
    except ModelNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"model : {package.name}")
    print(f"text  : {text}")
    print()

    for variation in args.variation:
        for seed in range(1, args.seeds + 1):
            name = (
                f"seed-{seed:02d}.wav"
                if len(args.variation) == 1
                else f"var-{variation:.3f}-seed-{seed:02d}.wav"
            )
            engine.save(
                text,
                args.out / name,
                speed=args.speed,
                variation=variation,
                seed=seed,
                normalize=False,  # already normalised once, above
            )
            print(f"  {args.out / name}")

    print()
    print("Listen, pick the best one, and hard-code that seed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
