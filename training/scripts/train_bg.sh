#!/usr/bin/env bash
# Start a training run detached, so it survives an SSH or JupyterLab disconnect.
#
#   bash train_bg.sh large      # start
#   bash train_bg.sh large tail # follow the log of a run already going
#
# A 200,000-step run lasts many hours. Started from a plain shell it receives
# SIGHUP the moment the terminal goes away, which on RunPod happens on every
# dropped connection or browser reload. tmux is used when present; otherwise the
# run is detached with setsid + nohup, which needs nothing preinstalled.
set -euo pipefail

PROFILE="${1:-}"
if [ -z "${PROFILE}" ]; then
  echo "usage: train_bg.sh {smoke|24gb|quality|large} [tail|extra args...]" >&2
  exit 2
fi
ACTION="${2:-start}"
# Anything after the profile (other than the literal "tail") is passed straight
# through to train_az.sh, e.g. --num-workers 5.
EXTRA=()
if [ "${ACTION}" != "tail" ]; then shift || true; EXTRA=("$@"); fi

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${WORKDIR:-/workspace}"
OUTPUT="${OUTPUT:-${WORKDIR}/runs/az-${PROFILE}}"
# The trainer refuses to start a new run in a nonempty output directory, so the
# launcher's own bookkeeping lives beside it rather than inside it.
LOG="${OUTPUT%/}.log"
PIDFILE="${OUTPUT%/}.pid"
SESSION="inflect-${PROFILE}"

follow() {
  echo "==> following ${LOG} (Ctrl-C stops watching, not training)"
  tail -f "${LOG}"
}

if [ "${ACTION}" = "tail" ]; then
  [ -f "${LOG}" ] || { echo "no log at ${LOG}" >&2; exit 1; }
  follow
fi

if [ -f "${PIDFILE}" ] && kill -0 "$(cat "${PIDFILE}")" 2>/dev/null; then
  echo "==> a run is already going (pid $(cat "${PIDFILE}"))." >&2
  echo "    follow it : bash train_bg.sh ${PROFILE} tail" >&2
  echo "    stop it   : kill $(cat "${PIDFILE}")" >&2
  exit 3
fi

mkdir -p "$(dirname "${OUTPUT}")"

if command -v tmux >/dev/null 2>&1; then
  # An already-running tmux server does not inherit this shell's environment,
  # so the split-storage variables are passed into the command explicitly.
  ENV_PREFIX="WORKDIR='${WORKDIR}' OUTPUT='${OUTPUT}'"
  [ -n "${BASE:-}" ] && ENV_PREFIX="${ENV_PREFIX} BASE='${BASE}'"
  [ -n "${DATASET:-}" ] && ENV_PREFIX="${ENV_PREFIX} DATASET='${DATASET}'"
  [ -n "${HF_HOME:-}" ] && ENV_PREFIX="${ENV_PREFIX} HF_HOME='${HF_HOME}'"
  [ -n "${VENV:-}" ] && ENV_PREFIX="${ENV_PREFIX} VENV='${VENV}'"
  tmux new-session -d -s "${SESSION}" \
    "${ENV_PREFIX} bash '${PROJECT}/scripts/train_az.sh' '${PROFILE}' ${EXTRA[*]} 2>&1 | tee -a '${LOG}'"
  echo "==> started in tmux session '${SESSION}'"
  echo "    attach  : tmux attach -t ${SESSION}"
  echo "    detach  : Ctrl-b then d"
else
  setsid nohup bash "${PROJECT}/scripts/train_az.sh" "${PROFILE}" "${EXTRA[@]}" \
    >>"${LOG}" 2>&1 < /dev/null &
  echo $! > "${PIDFILE}"
  echo "==> started detached (pid $(cat "${PIDFILE}")), tmux was not available"
  echo "    stop    : kill \$(cat ${PIDFILE})"
fi

echo "    log     : ${LOG}"
echo "    follow  : bash ${PROJECT}/scripts/train_bg.sh ${PROFILE} tail"
echo
echo "Safe to close the terminal now. After a reconnect, re-running"
echo "'bash train_bg.sh ${PROFILE}' resumes from the last checkpoint."
