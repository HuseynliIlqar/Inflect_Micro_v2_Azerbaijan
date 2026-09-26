#!/usr/bin/env bash
# Training status: progress, speed, ETA, losses, GPU, disk.
#
#   bash status.sh              # one snapshot of the az-large run
#   bash status.sh live         # redraw every 10 s until Ctrl-C
#   bash status.sh live 30      # redraw every 30 s
#   bash status.sh az-quality   # another run under /workspace/runs
set -uo pipefail

# Live mode re-runs this same script, so the report itself stays single-purpose.
if [ "${1:-}" = "live" ] || [ "${1:-}" = "--watch" ]; then
  INTERVAL="${2:-10}"
  SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
  RUN_NAME="${3:-az-large}"
  echo "Live view (every ${INTERVAL}s). Stop with Ctrl-C"
  sleep 1
  while true; do
    clear
    bash "${SELF}" "${RUN_NAME}"
    echo
    echo "-- refreshes every ${INTERVAL}s, Ctrl-C to exit --"
    sleep "${INTERVAL}"
  done
fi

RUN="${1:-az-large}"
WORKDIR="${WORKDIR:-/workspace}"
OUTPUT="${OUTPUT:-${WORKDIR}/runs/${RUN}}"
PYTHON="${VENV:-/root/work/Inflect/.venv}/bin/python"
[ -x "$PYTHON" ] || PYTHON=python3

METRICS="${OUTPUT}/metrics.jsonl"
if [ ! -f "${METRICS}" ]; then
  echo "Run not found: ${OUTPUT}"
  [ -f "${OUTPUT%/}.log" ] && { echo "--- last log lines ---"; tail -5 "${OUTPUT%/}.log"; }
  exit 1
fi

RUNNING=$(pgrep -fc "inflect-adapt" 2>/dev/null || echo 0)
# utilization.gpu only reports whether a kernel was resident during the sample
# window, so it swings between 0 and 100 every step. Memory, power and clocks
# are the readings that actually say how loaded the card is.
GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.max.sm \
      --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || echo "n/a")

"$PYTHON" - "$OUTPUT" "$RUNNING" "$GPU" <<'PY'
import json, os, sys, time

output, running, gpu = sys.argv[1], int(sys.argv[2] or 0), sys.argv[3]
metrics = os.path.join(output, "metrics.jsonl")

rows = []
with open(metrics, encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a half-written final line while training is live
if not rows:
    print("metrics.jsonl is empty")
    raise SystemExit(1)

last = rows[-1]
step = last["step"]

options_path = os.path.join(output, "training-options.json")
total = 200000
if os.path.isfile(options_path):
    try:
        total = json.load(open(options_path, encoding="utf-8")).get("max_steps", total)
    except Exception:
        pass

# Wall clock: the run marker is written at startup, metrics on every step.
marker = os.path.join(output, "run-identity.json")
started = os.path.getmtime(marker) if os.path.isfile(marker) else os.path.getctime(metrics)
touched = os.path.getmtime(metrics)
elapsed = max(1.0, touched - started)
per_step = elapsed / step if step else 0.0
remaining = (total - step) * per_step
stale = time.time() - touched


def clock(seconds: float) -> str:
    seconds = int(max(0, seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes = seconds // 60
    return f"{days}d {hours:02d}h {minutes:02d}m" if days else f"{hours:02d}h {minutes:02d}m"


percent = 100.0 * step / total
filled = int(percent / 2.5)
bar = "#" * filled + "." * (40 - filled)

print(f"RUN        {os.path.basename(output)}")
print(f"state      {'RUNNING' if running else 'STOPPED'}   (last metric {int(stale)}s ago)")
print()
print(f"[{bar}] {percent:5.1f}%")
print(f"step       {step:,} / {total:,}")
print(f"epoch      {last.get('epoch', 0):,}")
print(f"stage      {last.get('stage', '?')}")
print()
print(f"speed      {per_step:.3f} s/step   ({1 / per_step:.2f} steps/s)" if per_step else "")
print(f"elapsed    {clock(elapsed)}")
print(f"remaining  {clock(remaining)}   (~${remaining / 3600 * 0.44:.0f})")
print()
print("loss       " + "  ".join(
    f"{name}={last[key]:.3f}"
    for name, key in (("g", "loss_g"), ("d", "loss_d"), ("mel", "loss_mel"),
                      ("kl", "loss_kl"), ("dur", "loss_duration"))
    if key in last
))
window = rows[-200:]
if len(window) > 20:
    half = len(window) // 2
    before = sum(r["loss_mel"] for r in window[:half]) / half
    after = sum(r["loss_mel"] for r in window[half:]) / (len(window) - half)
    arrow = "down" if after < before else "up"
    print(f"loss_mel   over the last {len(window)} steps {before:.4f} -> {after:.4f}  ({arrow})")

vram = last.get("peak_reserved_gb")
if vram:
    print(f"VRAM peak  {vram:.1f} GB")

fields = gpu.split(",")
if len(fields) == 8:
    used, total = float(fields[1]), float(fields[2])
    power, limit = float(fields[4]), float(fields[5])
    clock, clock_max = float(fields[6]), float(fields[7])
    print(f"GPU memory {used / 1024:.1f} / {total / 1024:.1f} GB  ({100 * used / total:.0f}%)")
    print(f"GPU power  {power:.0f} / {limit:.0f} W  ({100 * power / limit:.0f}%)   {fields[3]}C")
    # A sustained drop below the maximum clock is the real sign of throttling.
    throttle = "" if clock >= clock_max * 0.95 else "  <- THROTTLED"
    print(f"GPU clock  {clock:.0f} / {clock_max:.0f} MHz{throttle}")
    print(f"GPU util   {fields[0]}%  (instantaneous sample, swings 0-100 within a step)")
else:
    print(f"GPU        {gpu}")

checkpoints = os.path.join(output, "checkpoints")
exports = os.path.join(output, "exports")


def summarise(path: str, pattern: str) -> str:
    if not os.path.isdir(path):
        return "none"
    names = [n for n in os.listdir(path) if n.endswith(".pth") and pattern in n]
    size = sum(os.path.getsize(os.path.join(path, n)) for n in names) / 1e9
    return f"{len(names)} file(s), {size:.1f} GB"


print()
print(f"checkpoint {summarise(checkpoints, 'adaptation-step-')}   (last 5 kept)")
print(f"export     {summarise(exports, 'model-step-')}")

validation = os.path.join(output, "validation")
if os.path.isdir(validation):
    waves = sorted(n for n in os.listdir(validation) if n.endswith(".wav"))
    if waves:
        print(f"validation {len(waves)} clip(s), latest: {waves[-1]}")
PY

echo
# /workspace is a shared network mount, so its df totals describe the cluster,
# not this pod. The run's own footprint is the number that matters there.
printf "disk       run directory    %s\n" "$(du -sh "${OUTPUT}" 2>/dev/null | cut -f1)"
df -h / 2>/dev/null | awk 'NR==2 {printf "disk       container        %s / %s (%s)\n", $3, $2, $5}'
