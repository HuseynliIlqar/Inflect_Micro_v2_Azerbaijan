#!/usr/bin/env bash
# One line of progress for the main run: step, stage, losses, speed, time left.
#   bash status.sh
set -uo pipefail

OUT=/workspace/runs/main
METRICS="$OUT/metrics.jsonl"
[ -f "$METRICS" ] || { echo "no metrics yet"; tail -n 3 "$OUT.log" 2>/dev/null; exit 0; }

first=$(wc -l < "$METRICS")
sleep 30
second=$(wc -l < "$METRICS")

python3 - "$METRICS" "$first" "$second" <<'PY'
import json, sys
path, first, second = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
lines = open(path).read().splitlines()
last = json.loads(lines[-1])
recent = [json.loads(line) for line in lines[-200:]]
mean = lambda key: sum(r[key] for r in recent) / len(recent)
rate = (second - first) / 30
left = (100000 - last["step"]) / rate / 3600 if rate else float("nan")
print(f"step {last['step']} stage {last['stage']} mel {mean('loss_mel'):.3f} kl {mean('loss_kl'):.2f} "
      f"dur {mean('loss_duration'):.3f} | {1 / rate if rate else 0:.2f} s/step | ~{left:.1f} h left")
PY
if [ -f "$OUT.pid" ] && kill -0 "$(cat "$OUT.pid")" 2>/dev/null; then echo "running"; else echo "NOT RUNNING"; tail -n 5 "$OUT.log"; fi
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
df -h /workspace | tail -1
