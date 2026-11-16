#!/usr/bin/env bash
# Start (or resume) the main run detached, so a dropped SSH connection cannot stop it, plus the
# listening renderer that reads the fixed sentences with every new checkpoint.
#   bash train.sh
set -euo pipefail

KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK=/workspace
OUT="$WORK/runs/main"
LATEST="$OUT/checkpoints/latest.pth"
mkdir -p "$WORK/runs"

if [ -f "$OUT.pid" ] && kill -0 "$(cat "$OUT.pid")" 2>/dev/null; then
  echo "a run is already going (pid $(cat "$OUT.pid"))"
  exit 3
fi

ARGS=(train --base "$WORK/base" --dataset "$WORK/prepared" --preset "$KIT/az-continue.json"
      --output "$OUT" --device cuda)
if [ -f "$LATEST" ]; then
  ARGS+=(--resume "$LATEST")
  echo "resuming from $LATEST"
fi

# The Turkish run slowed from ~1.0 to ~2.4 s/step after a few thousand steps from allocator
# fragmentation; expandable segments avoid it.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
setsid nohup "$WORK/venv/bin/inflect-adapt" "${ARGS[@]}" >> "$OUT.log" 2>&1 < /dev/null &
echo $! > "$OUT.pid"
echo "training started (pid $(cat "$OUT.pid")), log $OUT.log"

if ! { [ -f "$OUT.listen.pid" ] && kill -0 "$(cat "$OUT.listen.pid")" 2>/dev/null; }; then
  setsid nohup "$WORK/venv/bin/python" "$KIT/listen.py" --run "$OUT" >> "$OUT.listen.log" 2>&1 < /dev/null &
  echo $! > "$OUT.listen.pid"
  echo "listening renderer started (pid $(cat "$OUT.listen.pid"))"
fi
