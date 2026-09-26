"""Download the Azerbaijani parquet dataset and convert it into a manifest.

    python scripts/build_az_dataset.py --output data/az
    python scripts/build_az_dataset.py --output data/az --shards 1 --limit 50   # quick look

Produces data/az/audio/*.wav and data/az/metadata.jsonl, ready for
`inflect-preflight` and then `inflect-adapt prepare`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hf_dataset import (
    DEFAULT_REPO_ID,
    DEFAULT_SPEAKER,
    SPEAKER_LABEL_PATTERN,
    MAX_CLIPPED_FRACTION,
    MAX_SECONDS,
    MIN_SECONDS,
    convert,
    existing_speakers,
    shard_names,
)

TOTAL_SHARDS = 10


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "-o", type=Path, required=True, help="Destination directory.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--speaker",
        default=DEFAULT_SPEAKER,
        help=(
            "Voice label for every row. Ids follow alphabetical order, so keep "
            f"the zero-padded enrolment number: {SPEAKER_LABEL_PATTERN}"
        ),
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Enrol another voice into an existing output directory.",
    )
    parser.add_argument("--shards", type=int, default=TOTAL_SHARDS, help="How many shards to use.")
    parser.add_argument("--limit", type=int, default=None, help="Stop after N written clips.")
    parser.add_argument("--min-seconds", type=float, default=MIN_SECONDS)
    parser.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    parser.add_argument("--max-clipped-fraction", type=float, default=MAX_CLIPPED_FRACTION)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.shards <= TOTAL_SHARDS:
        print(f"error: --shards must be between 1 and {TOTAL_SHARDS}", file=sys.stderr)
        return 1

    from huggingface_hub import hf_hub_download

    names = shard_names(TOTAL_SHARDS)[: args.shards]
    print(f"downloading {len(names)} shard(s) from {args.repo_id} (cached after the first run)")
    paths = []
    for index, name in enumerate(names, start=1):
        print(f"  [{index}/{len(names)}] {name}")
        paths.append(Path(hf_hub_download(args.repo_id, name, repo_type="dataset")))

    print(f"\nconverting into {args.output.resolve()}")
    try:
        stats = convert(
            paths,
            args.output,
            speaker=args.speaker,
            min_seconds=args.min_seconds,
            max_seconds=args.max_seconds,
            max_clipped_fraction=args.max_clipped_fraction,
            limit=args.limit,
            append=args.append,
        )
    except (FileExistsError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print()
    print(stats.render())
    if stats.examples:
        print("\ntranscripts containing digits (normalize these by hand):")
        for example in stats.examples:
            print(f"  {example[:100]}")

    summary = args.output / "conversion-summary.json"
    summary.write_text(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "speaker": args.speaker,
                "speakers": existing_speakers(args.output),
                "shards": len(names),
                "written": stats.written,
                "hours": round(stats.hours, 4),
                "dropped": stats.dropped,
                "filters": {
                    "min_seconds": args.min_seconds,
                    "max_seconds": args.max_seconds,
                    "max_clipped_fraction": args.max_clipped_fraction,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nmanifest : {args.output / 'metadata.jsonl'}")
    print(f"summary  : {summary}")
    if not stats.written:
        print("error: no clips were written", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
