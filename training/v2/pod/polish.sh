#!/usr/bin/env bash
# Start (or resume) the decoder run with the new discriminators, detached, plus its listening renderer.
#   bash polish.sh [extra train_polish.py args]
# Stop the main run first: both cannot share the A40's memory at full batch.
set -euo pipefail

WORK=/workspace
OUT="$WORK/runs/polish"
mkdir -p "$WORK/runs"

if [ -f "$WORK/runs/main.pid" ] && kill -0 "$(cat "$WORK/runs/main.pid")" 2>/dev/null; then
  echo "the main run is still going; stop it first (kill \$(cat $WORK/runs/main.pid))" >&2
  exit 3
fi
if [ -f "$OUT.pid" ] && kill -0 "$(cat "$OUT.pid")" 2>/dev/null; then
  echo "the decoder run is already going (pid $(cat "$OUT.pid"))"
  exit 3
fi

cd "$WORK"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
setsid nohup "$WORK/venv/bin/python" "$WORK/polish/train_polish.py" \
  --init "$WORK/keep/adaptation-step-00020000.pth" --bigvgan "$WORK/bigvgan/disc_v2.pt" \
  --output "$OUT" "$@" >> "$OUT.log" 2>&1 < /dev/null &
echo $! > "$OUT.pid"
echo "decoder run started (pid $(cat "$OUT.pid")), log $OUT.log"

# Comparison clips for the validation section, then the listening renderer for this run.
mkdir -p "$OUT/validation"
"$WORK/venv/bin/python" "$WORK/pod/validation_before.py" --run "$OUT" > /dev/null 2>&1 || true
if ! { [ -f "$OUT.listen.pid" ] && kill -0 "$(cat "$OUT.listen.pid")" 2>/dev/null; }; then
  setsid nohup "$WORK/venv/bin/python" "$WORK/pod/listen.py" --run "$OUT" >> "$OUT.listen.log" 2>&1 < /dev/null &
  echo $! > "$OUT.listen.pid"
  echo "listening renderer started (pid $(cat "$OUT.listen.pid"))"
fi
