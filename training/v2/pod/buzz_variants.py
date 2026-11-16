import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/pod")
import buzz_measure as bm

OUT = Path("/workspace/asr_eval")
result = {}
for name in ("step-00030000", "exp-var-0.5", "exp-var-0.33", "exp-var-0.2"):
    rows = [bm.measure(str(OUT / name / f"{i:04d}.wav")) for i in range(0, 471, 6)]
    ok = lambda key: [r[key] for r in rows if r[key] == r[key]]  # noqa: E731
    result[name] = {"h2_4k": round(statistics.median(ok("h2-4k")), 2),
                    "ring_share": round(statistics.median(ok("ring_share")), 3), "clips": len(rows)}
(OUT / "buzz_variants.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
print(json.dumps(result, indent=1))
