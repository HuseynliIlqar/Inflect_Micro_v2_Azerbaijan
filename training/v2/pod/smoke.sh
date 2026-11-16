#!/usr/bin/env bash
# Short run with the real batch size through all three stages: checks the warm start
# from Avaz, the memory use and the speed before the long run is started.
#   bash smoke.sh
set -euo pipefail

KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK=/workspace
export PATH="$WORK/venv/bin:$PATH"
OUT="$WORK/runs/smoke"
STEPS=300

mkdir -p "$WORK/runs"
rm -rf "$OUT" "$OUT.speed"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
inflect-adapt train \
  --base "$WORK/base" \
  --dataset "$WORK/prepared" \
  --preset "$KIT/az-continue.json" \
  --output "$OUT" \
  --device cuda \
  --max-steps "$STEPS" \
  --posterior-warmup-steps 50 \
  --decoder-unfreeze-step 100 \
  --checkpoint-interval "$STEPS" \
  --validation-interval "$STEPS" \
  --log-interval 10 > "$OUT.log" 2>&1 &
TRAIN=$!
echo "$TRAIN" > "$OUT.pid"  # the monitor shows the run as alive while this process exists

# Seconds and step count every 15 s, for the speed of the decoder stage alone.
while kill -0 "$TRAIN" 2>/dev/null; do
  steps=$(wc -l < "$OUT/metrics.jsonl" 2>/dev/null || echo 0)
  echo "$(date +%s) $steps" >> "$OUT.speed"
  sleep 15
done
wait "$TRAIN" || { echo "=== smoke run FAILED"; tail -n 30 "$OUT.log"; exit 1; }

echo "=== warm start"
cat "$OUT/compatibility-report.json"
echo
echo "=== speed and memory"
python3 - "$OUT" <<'PY'
import json, sys
out = sys.argv[1]
rows = [json.loads(line) for line in open(f"{out}/metrics.jsonl")]
samples = [tuple(map(int, line.split())) for line in open(f"{out}.speed")]
decoder = [(t, s) for t, s in samples if 120 <= s < 300]
if len(decoder) >= 2:
    (t0, s0), (t1, s1) = decoder[0], decoder[-1]
    print(f"decoder stage: {(t1 - t0) / max(1, s1 - s0):.3f} s/step")
print("peak memory GB", max(r.get("peak_memory_gb", 0) for r in rows),
      "reserved GB", max(r.get("peak_reserved_gb", 0) for r in rows))
print("stages", sorted({r["stage"] for r in rows}))
last = rows[-1]
print("last", {k: last[k] for k in ("step", "loss_mel", "loss_kl", "loss_duration", "loss_g", "loss_d")})
PY
echo "=== listening samples (old Avaz and the step-$STEPS export)"
"$WORK/venv/bin/python" "$KIT/listen.py" --run "$OUT" --once
ls "$OUT/listen"
echo "=== smoke finished"
