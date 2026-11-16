#!/usr/bin/env bash
# Unpack the upload, build the toolkit dataset with the stress frontend, and check it.
#   bash prepare.sh
set -euo pipefail

KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK=/workspace
PY="$WORK/venv/bin/python"
export PATH="$WORK/venv/bin:$PATH"

if [ -f "$WORK/data/.decoded" ]; then
  echo "=== FLAC already decoded and checked"
else
  echo "=== decoding FLAC to WAV and checking every sample"
  mkdir -p "$WORK/data"
  "$PY" "$KIT/decode_verify.py" "$WORK/upload" "$WORK/data"
  "$PY" "$KIT/dedup_manifest.py" "$WORK/data/metadata.jsonl"
  touch "$WORK/data/.decoded"
fi

echo "=== prepare (eSpeak az + stress rules)"
rm -rf "$WORK/prepared"
inflect-adapt prepare \
  --manifest "$WORK/data/metadata.jsonl" \
  --audio-root "$WORK/data" \
  --language az \
  --frontend custom \
  --frontend-hook "$KIT/az_stress_frontend.py:create_frontend" \
  --output "$WORK/prepared" \
  --min-duration-seconds 1.0 --max-duration-seconds 14.5 \
  --validation-fraction 0.02

echo "=== audit"
inflect-adapt audit --dataset "$WORK/prepared"

echo "=== phonemes match the Windows reference"
"$PY" "$KIT/check_phonemes.py" "$WORK/prepared" "$KIT/reference_phonemes.json" "$WORK/data/metadata.jsonl"

echo "=== symbols: base indices kept, nothing new"
"$PY" - "$WORK/prepared/symbols.json" "$WORK/base/symbols.json" <<'PY'
import json, sys
prepared = json.load(open(sys.argv[1], encoding="utf-8"))
prepared = prepared["symbols"] if isinstance(prepared, dict) else prepared
base = json.load(open(sys.argv[2], encoding="utf-8"))
base = base["symbols"] if isinstance(base, dict) else base
print("prepared", len(prepared), "base", len(base))
assert prepared[: len(base)] == base, "base symbol order changed"
assert len(prepared) == len(base), f"new symbols added: {prepared[len(base):]}"
PY
echo "=== prepare finished"
