#!/usr/bin/env bash
# Start the monitor page on port 8000 (RunPod serves it at https://<pod id>-8000.proxy.runpod.net), detached.
# Open it once with #token=<the token in /workspace/monitor.token>; the browser keeps the token afterwards.
#   bash monitor.sh
set -euo pipefail

WORK=/workspace
[ -s "$WORK/monitor.token" ] || "$WORK/venv/bin/python" -c "import secrets; print(secrets.token_urlsafe(24))" > "$WORK/monitor.token"
chmod 600 "$WORK/monitor.token"
mkdir -p "$WORK/runs"
pkill -f "avazmon.app:app_from_env" 2>/dev/null || true
cd "$WORK"
MONITOR_TOKEN="$(cat "$WORK/monitor.token")" MONITOR_RUNS="$WORK/runs" MONITOR_PRICE="${MONITOR_PRICE:-0.59}" \
  setsid nohup "$WORK/venv/bin/python" -m uvicorn avazmon.app:app_from_env --factory \
  --host 0.0.0.0 --port 8000 --proxy-headers > "$WORK/monitor.log" 2>&1 < /dev/null &
echo "monitor started (pid $!)"
