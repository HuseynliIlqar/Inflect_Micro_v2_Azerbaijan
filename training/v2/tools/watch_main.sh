#!/usr/bin/env bash
# Watch the main run every 5 minutes and exit (one line saying why) when something needs attention:
#   MILESTONE  the run reached STEP and the listening samples for it exist
#   ENDED      the trainer is no longer running (finished or crashed)
#   LISTEN     the listening renderer died
#   TOPUP      the balance went up (the user added money)
#   LOWBAL     the balance fell under $1.50
#   bash watch_main.sh STEP
set -u

source /d/ktts/avaz-train/tools/podenv.sh
R=/d/ktts/teacher/tools/runpodctl.exe
MILESTONE="${1:?usage: watch_main.sh STEP}"
start_balance=$("$R" user 2>/dev/null | grep clientBalance | tr -dc '0-9.')
listen_dir=$(printf 'step-%08d' "$MILESTONE")

while true; do
  state=$(timeout 60 ssh "${SSH_OPTS[@]}" -n "$POD_HOST" "
    cd /workspace/runs
    steps=\$(wc -l < main/metrics.jsonl 2>/dev/null || echo 0)
    kill -0 \$(cat main.pid) 2>/dev/null && alive=1 || alive=0
    kill -0 \$(cat main.listen.pid) 2>/dev/null && listen=1 || listen=0
    [ -d main/listen/$listen_dir ] && heard=1 || heard=0
    echo \$steps \$alive \$listen \$heard" 2>/dev/null)
  balance=$("$R" user 2>/dev/null | grep clientBalance | tr -dc '0-9.')
  if [ -n "$state" ]; then
    read -r steps alive listen heard <<< "$state"
    [ "$alive" = 0 ] && { echo "ENDED at step $steps, balance $balance"; exit 0; }
    [ "$listen" = 0 ] && { echo "LISTEN renderer died at step $steps"; exit 0; }
    [ "$steps" -ge "$MILESTONE" ] && [ "$heard" = 1 ] && { echo "MILESTONE step $steps, balance $balance"; exit 0; }
  fi
  if [ -n "$balance" ] && [ -n "$start_balance" ]; then
    awk "BEGIN{exit !($balance > $start_balance + 3)}" && { echo "TOPUP balance $balance (was $start_balance), step ${steps:-?}"; exit 0; }
    awk "BEGIN{exit !($balance < 1.5)}" && { echo "LOWBAL balance $balance at step ${steps:-?}"; exit 0; }
  fi
  sleep 300
done
