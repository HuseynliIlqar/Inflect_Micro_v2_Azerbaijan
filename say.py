"""Speak Azerbaijani -- the main entry point.

    python say.py "Salam, necəsiniz?"
    python say.py --text-file text.txt --one-file --out out/book
    python say.py --show-text "II Dünya müharibəsi, 25% artım"
    python say.py --voice en "Hello there."     # the English base model
    python say.py --allow-profanity "..."        # speak obscenities as written
    python say.py                       # renders the demo sentences

Audio lands in `out/` unless you say otherwise.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from aztts import (  # noqa: E402
    DEFAULT_MAX_WORDS,
    DEFAULT_MODEL_DIR,
    AzTTS,
    EnVoice,
    ModelNotFoundError,
)
from aztts.console import use_utf8  # noqa: E402

use_utf8()

DEFAULT_OUT = PROJECT_ROOT / "out"

DEMO: tuple[tuple[str, str], ...] = (
    ("01-salam", "Salam, bu model tamamilə yerli maşında işləyir."),
    ("02-payiz", "Payız gəlmişdi və şəhərin küçələri saralmış yarpaqlarla örtülmüşdü."),
    ("03-sual", "Sən bu kitabı oxumusan? Mənə çox maraqlı gəldi."),
    ("04-reqem", "II Dünya müharibəsi 01/09/1939 tarixində başladı və 25% artım oldu."),
    ("05-uzun",
     "Səhər tezdən qalxıb pəncərəni açdı, həyətdəki ağacların arasından keçən "
     "sərin külək otağı doldurdu və o, uzun müddət heç nə düşünmədən dayandı."),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("text", nargs="*", help="Text to speak; leave it empty to render the demo sentences.")
    parser.add_argument(
        "--text-file", "-f", type=Path,
        help="A UTF-8 text file. Blank lines are paragraph boundaries and each "
             "paragraph becomes its own file; pass --one-file to join them.",
    )
    parser.add_argument("--one-file", action="store_true", help="With --text-file: write everything into a single WAV.")
    parser.add_argument("--out", "-o", type=Path, default=DEFAULT_OUT, help=f"Output directory (default: {DEFAULT_OUT.name}/).")
    parser.add_argument("--model", "-m", type=Path, default=DEFAULT_MODEL_DIR, help="Path to the model package.")
    parser.add_argument(
        "--voice", choices=("az", "en"), default="az",
        help="az is this project's Azerbaijani model. en is the English base "
             "model it was adapted from -- a different voice by another "
             "author, shipped in model-en/.",
    )
    parser.add_argument("--device", "-d", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--speed", type=float, default=1.0, help="0.5-2.0; lower is slower.")
    parser.add_argument("--variation", type=float, default=0.667, help="0.0-1.0; lower is steadier, higher is livelier.")
    parser.add_argument("--seed", type=int, default=7, help="The same seed gives the same voice.")
    parser.add_argument("--raw", action="store_true", help="Turn normalisation off and pass the text through as written.")
    parser.add_argument(
        "--max-words", type=int, default=DEFAULT_MAX_WORDS,
        help=f"Cut sentences into chunks of this many words (default "
             f"{DEFAULT_MAX_WORDS}; 0 leaves splitting to the package's own "
             "280-character rule).",
    )
    parser.add_argument(
        "--prosody", choices=("off", "safe", "wide"), default="off",
        help="Re-place the stress marks. eSpeak stresses nearly every word, "
             "which is what flattens the reading. \"safe\" only unstresses words "
             "the training data already shows without an accent; \"wide\" goes "
             "further (experimental).",
    )
    parser.add_argument("--prosody-drop", action="store_true", help="With --prosody: remove the accent outright instead of demoting it.")
    parser.add_argument("--show-text", action="store_true", help="Print the normalised text and the chunk boundaries.")
    parser.add_argument(
        "--allow-profanity", action="store_true",
        help="Speak obscenities as written. By default each one is replaced by "
             "a bleep, in either voice and even with --raw.",
    )
    return parser


EN_DEMO: tuple[tuple[str, str], ...] = (
    ("01-hello", "Hello, this model runs completely offline on your machine."),
    ("02-autumn", "Autumn had come, and the streets were covered with yellow leaves."),
    ("03-question", "Have you read this book? I found it very interesting."),
)


def collect_items(args: argparse.Namespace) -> list[tuple[str, str]] | None:
    """Turn the CLI arguments into (name, text) pairs; None on a read error."""
    if args.text_file:
        try:
            raw = args.text_file.read_text(encoding="utf-8")
        except OSError as error:
            print(f"error: could not read {args.text_file}: {error}", file=sys.stderr)
            return None
        if args.one_file:
            return [("full", " ".join(raw.split()))]
        paragraphs = [" ".join(p.split()) for p in raw.split("\n\n") if p.strip()]
        return [(f"{i:02d}", p) for i, p in enumerate(paragraphs, start=1)]
    if args.text:
        return [(f"{i:02d}", t) for i, t in enumerate(args.text, start=1)]
    return list(EN_DEMO if args.voice == "en" else DEMO)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    english = args.voice == "en"
    started = time.perf_counter()
    try:
        tts = (
            EnVoice(device=args.device)
            if english
            else AzTTS(args.model, device=args.device)
        )
    except ModelNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"model : {tts.model_dir}")
    print(f"device: {args.device}   load: {time.perf_counter() - started:.2f}s")
    if english:
        # Azerbaijani normalisation and the stress layer do not apply.
        print(f"text  : voice=en  max-words={args.max_words or 'package'}  "
              f"censor={'off' if args.allow_profanity else 'on'}")
    else:
        print(f"text  : normalise={'off' if args.raw else 'on'}  "
              f"max-words={args.max_words or 'package'}  prosody={args.prosody}  "
              f"censor={'off' if args.allow_profanity else 'on'}")
    print()

    items = collect_items(args)
    if items is None:
        return 1
    if not items or not items[0][1].strip():
        print("error: the text is empty", file=sys.stderr)
        return 1

    options: dict[str, object] = dict(
        speed=args.speed,
        variation=args.variation,
        seed=args.seed,
        max_words=args.max_words,
        censor=not args.allow_profanity,
    )
    if not english:
        options.update(
            normalize=not args.raw,
            prosody=args.prosody,
            prosody_drop=args.prosody_drop,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    for name, source in items:
        chunks = (
            tts.prepare(
                source,
                max_words=args.max_words,
                censor=not args.allow_profanity,
            )
            if english
            else tts.prepare(
                source,
                normalize=not args.raw,
                max_words=args.max_words,
                censor=not args.allow_profanity,
            )
        )
        if not chunks:
            print(f"skipped (empty): {name}", file=sys.stderr)
            continue
        if args.show_text:
            for index, chunk in enumerate(chunks, start=1):
                print(f"    [{index}] {chunk}")
        destination = args.out / f"{name}.wav"
        began = time.perf_counter()
        waveform = tts.synthesize(source, **options)
        sf.write(destination, waveform, tts.sample_rate)
        elapsed = time.perf_counter() - began
        seconds = waveform.size / tts.sample_rate
        print(f"{destination}  {seconds:.2f}s  ({elapsed:.2f}s, {seconds / elapsed:.1f}x real time)")
        print(f"    {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
