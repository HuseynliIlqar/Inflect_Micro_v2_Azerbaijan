"""Choose the best passing take of every sentence and report what the gates did.

    python select_takes.py --scores /workspace/rec/scores --gates /workspace/rec/gates.json \
                           --takes A B C D E --question-takes B C D E                            --out /workspace/rec/selection [--ids ids.txt]

--question-takes limits yes/no questions (RISING) to the sampled takes: in the pilot's ear
test the deterministic take A sounded like a proper question in 8 of 24 cases, the sampled
takes in 81 of 120.

Writes <out>/selection.jsonl ({id, cat, take} or {id, cat, dropped: reason-per-take}),
<out>/summary.json (per-category pass rates and drop reasons, per-take pass rates, and the
bias guard: chosen takes against all takes) and <out>/listen.json (25 random and 25
borderline chosen clips per category, for the listening page).
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT))
from common import RISING, Gates, preference, rejection  # noqa: E402

GUARD_KEYS = ("seconds", "range_st", "buzz", "similarity", "cer")


def margin(cat: str, score: dict, gates: Gates) -> float:
    """How close a passing take came to its nearest gate, in rough relative units (smaller = closer)."""
    parts = [
        (score["similarity"] - gates.min_similarity) / 0.05,
        (score["buzz"] - gates.min_buzz) / 1.0,
        (gates.max_ring - score["ring"]) / 0.05,
        (score["range_st"] - gates.min_range_st) / 2.0,
    ]
    return min(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--takes", nargs="+", required=True)
    parser.add_argument("--question-takes", nargs="+", help="takes allowed for yes/no questions (default: all)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ids", type=Path)
    args = parser.parse_args()

    gates = Gates.from_dict(json.loads(args.gates.read_text())["gates"])
    rows = {r["id"]: r for r in map(json.loads, (KIT / "sentences_v2.jsonl").read_text(encoding="utf-8").splitlines())}
    if args.ids:
        wanted = set(args.ids.read_text(encoding="utf-8").split())
        rows = {k: v for k, v in rows.items() if k in wanted}
    scores = {take: {s["id"]: s for s in map(json.loads, (args.scores / f"{take}.jsonl").read_text(encoding="utf-8").splitlines())}
              for take in args.takes}

    selection, chosen_scores = [], []
    per_cat = defaultdict(Counter)
    take_pass = Counter()
    take_seen = Counter()
    reasons = defaultdict(Counter)
    for clip_id, row in rows.items():
        cat, text = row["cat"], row["text"]
        passing, why = [], {}
        for order, take in enumerate(args.takes):
            score = scores[take].get(clip_id)
            if score is None or (cat in RISING and args.question_takes and take not in args.question_takes):
                continue
            take_seen[take] += 1
            reason = rejection(cat, text, score, gates)
            if reason is None:
                take_pass[take] += 1
                passing.append((order, take, score))
            else:
                why[take] = reason
        per_cat[cat]["sentences"] += 1
        if passing:
            order, take, score = min(passing, key=lambda item: preference(cat, item[0], item[2]))
            selection.append({"id": clip_id, "cat": cat, "take": take, "margin": round(margin(cat, score, gates), 3),
                              "end_drop_st": score.get("end_drop_st"), "cer": score["cer"]})
            chosen_scores.append(score)
            per_cat[cat]["kept"] += 1
        else:
            selection.append({"id": clip_id, "cat": cat, "dropped": why})
            per_cat[cat]["dropped"] += 1
            for reason in set(why.values()):
                reasons[cat][reason] += 1

    all_scores = [s for take in args.takes for s in scores[take].values() if s["id"] in rows]
    median = lambda items, key: round(statistics.median(x[key] for x in items if x[key] == x[key]), 3)  # noqa: E731
    summary = {
        "takes": args.takes, "question_takes": args.question_takes or args.takes, "gates": gates.__dict__,
        "per_category": {cat: {**counts, "kept_share": round(counts["kept"] / counts["sentences"], 3),
                               "drop_reasons": dict(reasons[cat])} for cat, counts in sorted(per_cat.items())},
        "take_pass_rate": {take: round(take_pass[take] / max(take_seen[take], 1), 3) for take in args.takes},
        "chosen_take_share": dict(Counter(s.get("take") for s in selection if "take" in s)),
        "bias_guard": {key: {"chosen": median(chosen_scores, key), "all": median(all_scores, key)} for key in GUARD_KEYS},
        "question_end_drop_chosen_p10_p50": None,
    }
    drops = sorted(s.get("end_drop_st", float("nan")) for s, sel in zip(chosen_scores, [x for x in selection if "take" in x])
                   if sel["cat"] in RISING)
    drops = [d for d in drops if d == d]
    if drops:
        summary["question_end_drop_chosen_p10_p50"] = [round(drops[len(drops) // 10], 2), round(drops[len(drops) // 2], 2)]

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "selection.jsonl").open("w", encoding="utf-8") as handle:
        for item in selection:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))

    listen = {}
    rng = random.Random(0)
    by_cat = defaultdict(list)
    for item in selection:
        if "take" in item:
            by_cat[item["cat"]].append(item)
    for cat, items in sorted(by_cat.items()):
        borderline = sorted(items, key=lambda x: x["margin"])[:25]
        rest = [x for x in items if x not in borderline]
        listen[cat] = {"random": rng.sample(rest, min(25, len(rest))), "borderline": borderline}
    (args.out / "listen.json").write_text(json.dumps(listen, indent=1, ensure_ascii=False))
    print(json.dumps({k: summary[k] for k in ("per_category", "take_pass_rate", "chosen_take_share", "bias_guard")},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
