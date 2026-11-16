#!/usr/bin/env bash
# Send the upload/ files the pod does not have yet (or has with a different size), in chunks, retrying
# after a dropped connection. The pod checks every sample afterwards (decode_verify.py), so a size match
# is only the test for "worth sending again".
#   bash upload_resume.sh
set -u

source /d/ktts/avaz-train/tools/podenv.sh
ROOT=/d/ktts/avaz-train
CHUNK=1500
WORK="$ROOT/upload_state"
mkdir -p "$WORK"

ssh_quiet() { timeout 120 ssh "${SSH_OPTS[@]}" -n "$POD_HOST" "$@"; }

for attempt in $(seq 1 30); do
  ssh_quiet 'cd /workspace && find upload -type f -printf "%p %s\n"' > "$WORK/remote.txt" || { echo "listing failed, retrying"; sleep 20; continue; }
  (cd "$ROOT" && find upload -type f -printf "%p %s\n") > "$WORK/local.txt"
  # Files whose "path size" line is missing on the pod.
  sort "$WORK/remote.txt" > "$WORK/remote.sorted"
  sort "$WORK/local.txt" > "$WORK/local.sorted"
  comm -23 "$WORK/local.sorted" "$WORK/remote.sorted" | cut -d' ' -f1 > "$WORK/missing.txt"
  left=$(wc -l < "$WORK/missing.txt")
  echo "$(date '+%T') attempt $attempt: $left files to send"
  [ "$left" -eq 0 ] && { echo "UPLOAD_COMPLETE"; exit 0; }
  split -l "$CHUNK" "$WORK/missing.txt" "$WORK/part."
  for part in "$WORK"/part.*; do
    if ! (cd "$ROOT" && tar --no-same-owner -cf - -T "$part") |
        timeout 1800 ssh "${SSH_OPTS[@]}" "$POD_HOST" 'tar --no-same-owner -C /workspace -xf -'; then
      echo "$(date '+%T') chunk failed, will list again"
      rm -f "$WORK"/part.*
      sleep 10
      continue 2
    fi
    rm -f "$part"
  done
done
echo "UPLOAD_GAVE_UP"
exit 1
