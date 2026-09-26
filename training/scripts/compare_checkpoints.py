"""Render the same sentences from several packages, for side-by-side listening.

    python scripts/compare_checkpoints.py
    python scripts/compare_checkpoints.py --packages exports/az-95k exports/az-final
    python scripts/compare_checkpoints.py --text-file cumlelər.txt --seed 3

Output is grouped by sentence, not by package, so one folder holds every
candidate for the same line:

    out/compare/01-salam-bu-model/az-95k.wav
    out/compare/01-salam-bu-model/az-170k.wav
    out/compare/01-salam-bu-model/az-final.wav

The final checkpoint is not automatically the best one.  After 1343 epochs a
model can sound over-polished; some intermediate export often reads as livelier.
No loss value settles this -- only listening does.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aztts.az_text import normalize_az  # noqa: E402
from aztts.console import use_utf8  # noqa: E402

use_utf8()

DEFAULT_OUT = PROJECT_ROOT / "out" / "compare"

# Lines chosen to expose the failure modes small models have: sibilants, long
# clauses, questions, and the number/numeral forms the normaliser rewrites.
PROBES = (
    "Salam, bu model tamamilə yerli maşında işləyir.",
    "Sən bu kitabı oxumusan? Mənə çox maraqlı gəldi.",
    "Payız gəlmişdi və şəhərin küçələri saralmış yarpaqlarla örtülmüşdü.",
    "Şuşa şəhərinin səssiz küçələrində şaxtalı bir səhər idi.",
    "İkinci qrupda iyirmi beş faiz artım qeydə alındı.",
)


def slugify(text: str, limit: int = 28) -> str:
    lowered = text.replace("I", "ı").replace("İ", "i").lower()
    ascii_ish = (
        lowered.replace("ə", "e").replace("ö", "o").replace("ü", "u")
        .replace("ı", "i").replace("ç", "c").replace("ş", "s").replace("ğ", "g")
    )
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_ish).strip("-")
    return slug[:limit].rstrip("-") or "metn"


def discover_packages(root: Path) -> tuple[Path, ...]:
    """Every exports/* directory holding PyTorch weights worth comparing.

    An ONNX export ships the same ``model.pth`` as its PyTorch sibling, so
    including it would render the identical audio twice.
    """
    if not root.is_dir():
        return ()
    return tuple(
        sorted(
            child
            for child in root.iterdir()
            if (child / "model.pth").is_file() and not (child / "onnx").is_dir()
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packages",
        nargs="+",
        type=Path,
        help="Packages to compare (default: every package under exports/).",
    )
    parser.add_argument("--text-file", "-f", type=Path, help="One sentence per line.")
    parser.add_argument("--out", "-o", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", "-d", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--variation", type=float, default=0.667)
    parser.add_argument("--raw", action="store_true", help="Skip normalisation.")
    return parser


def load_sentences(path: Path | None) -> tuple[str, ...]:
    if path is None:
        return PROBES
    lines = path.read_text(encoding="utf-8").splitlines()
    return tuple(line.strip() for line in lines if line.strip())


def synthesize_with(package: Path, sentences: tuple[str, ...], args) -> None:
    """Load one package and write every sentence under its own folder."""
    sys.path.insert(0, str(package))
    for module in ("inference", "models", "commons", "utils", "text"):
        sys.modules.pop(module, None)
    from inference import InflectTTS  # type: ignore[import-not-found]

    engine = InflectTTS(str(package), device=args.device)
    for index, sentence in enumerate(sentences, start=1):
        text = sentence if args.raw else normalize_az(sentence)
        folder = args.out / f"{index:02d}-{slugify(sentence)}"
        folder.mkdir(parents=True, exist_ok=True)
        engine.save(
            text,
            folder / f"{package.name}.wav",
            speed=args.speed,
            variation=args.variation,
            seed=args.seed,
        )
    sys.path.remove(str(package))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    packages = tuple(args.packages or discover_packages(PROJECT_ROOT / "exports"))
    missing = [p for p in packages if not (p / "model.pth").is_file()]
    if missing:
        print(f"error: package without a model: {', '.join(str(p) for p in missing)}", file=sys.stderr)
        return 1
    if not packages:
        print("error: no package found to compare", file=sys.stderr)
        return 1

    sentences = load_sentences(args.text_file)
    if not sentences:
        print("error: the sentence list is empty", file=sys.stderr)
        return 1

    print(f"packages  : {', '.join(p.name for p in packages)}")
    print(f"sentences : {len(sentences)}")
    print()
    for package in packages:
        print(f"-> {package.name}")
        synthesize_with(package, sentences, args)

    print()
    print(f"done: {args.out}")
    print("Each folder holds every variant of the same sentence side by side.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
