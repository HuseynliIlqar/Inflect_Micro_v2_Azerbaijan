#!/usr/bin/env bash
# Every 10 minutes: read the RunPod balance on this machine and write it to the pod for the monitor page.
# The RunPod key stays here; the pod only receives the number.
#   bash balance_watch.sh
set -u

source /d/ktts/avaz-train/tools/podenv.sh
R=/d/ktts/teacher/tools/runpodctl.exe
INTERVAL=600

while true; do
  balance=$("$R" user 2>/dev/null | grep clientBalance | tr -dc '0-9.')
  if [ -n "$balance" ]; then
    printf '{"balance": %s, "updated": %s}\n' "$balance" "$(date +%s)" |
      timeout 60 ssh "${SSH_OPTS[@]}" "$POD_HOST" 'mkdir -p /workspace/runs && cat > /workspace/runs/balance.json.tmp && mv /workspace/runs/balance.json.tmp /workspace/runs/balance.json' \
      && echo "$(date '+%F %T') balance $balance" || echo "$(date '+%F %T') could not reach the pod"
  fi
  sleep "$INTERVAL"
done
