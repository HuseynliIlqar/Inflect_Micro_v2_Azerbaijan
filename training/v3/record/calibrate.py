"""Set the gates from v1's own distribution, measured exactly like the new takes.

    python calibrate.py --sample /workspace/rec/v1_sample --out /workspace/rec
    python calibrate.py --from-scores /workspace/rec/v1_scores.jsonl --out /workspace/rec   # no GPU

<sample>/manifest.jsonl lists {"file": "...wav", "text": "..."} for ~400 random v1 clips.
Writes <out>/v1_centroid.npy (the voice the new takes must match), <out>/v1_scores.jsonl
and <out>/gates.json. --from-scores rebuilds gates.json from an earlier v1_scores.jsonl.

The pitch-range gate sits at v1's 2nd percentile and the exclamation gate at its 25th: the
pilot's 10th percentile and median dropped many good takes, and the range measure did not
separate good from bad questions by ear (55% vs 65% good below / above v1's 10th percentile).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

LOG = logging.getLogger("calibrate")


def percentile(values: list[float], q: float) -> float:
    finite = [v for v in values if v == v]
    return round(float(np.percentile(finite, q)), 4)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sample", type=Path)
    source.add_argument("--from-scores", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if args.from_scores:
        scored = [json.loads(line) for line in args.from_scores.read_text(encoding="utf-8").splitlines() if line.strip()]
        write_gates(scored, None, args.out)
        return
    from score import GpuMeasures, measure

    manifest = [json.loads(line) for line in (args.sample / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    items = [(Path(m["file"]).stem, m["text"], str(args.sample / m["file"])) for m in manifest]
    scored = measure(items, GpuMeasures(), None, args.workers)
    vectors = np.array([row.pop("embedding") for row in scored])
    centroid = vectors.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    np.save(args.out / "v1_centroid.npy", centroid)
    for row, vector in zip(scored, vectors):
        row["similarity"] = round(float(vector @ centroid), 4)
    with (args.out / "v1_scores.jsonl").open("w", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_gates(scored, [m["text"] for m in manifest], args.out)


def write_gates(scored: list[dict], texts: list[str] | None, out: Path) -> None:
    column = lambda key: [row[key] for row in scored]  # noqa: E731
    gates = {
        "min_similarity": percentile(column("similarity"), 2),
        "min_buzz": percentile(column("buzz"), 10),
        "max_ring": percentile(column("ring"), 90),
        "min_range_st": percentile(column("range_st"), 2),
        "exclamation_range_st": percentile(column("range_st"), 25),
    }
    stats = {
        "clips": len(scored),
        "cer_p50": percentile(column("cer"), 50), "cer_p90": percentile(column("cer"), 90),
        "buzz_p50": percentile(column("buzz"), 50), "ring_p50": percentile(column("ring"), 50),
        "range_p50": percentile(column("range_st"), 50), "similarity_p50": percentile(column("similarity"), 50),
    }
    if texts is not None:
        questions = [row["final_st"] for row, text in zip(scored, texts) if text.rstrip().endswith("?")]
        stats.update(question_final_p50=percentile(questions, 50) if questions else None, questions=len(questions))
    elif (out / "gates.json").exists():  # v1_scores.jsonl has no texts: keep the earlier question figures
        earlier = json.loads((out / "gates.json").read_text())["v1"]
        stats.update({key: earlier.get(key) for key in ("question_final_p50", "questions")})
    (out / "gates.json").write_text(json.dumps({"gates": gates, "v1": stats}, indent=1))
    LOG.info("gates %s", gates)
    LOG.info("v1 %s", stats)


if __name__ == "__main__":
    main()
