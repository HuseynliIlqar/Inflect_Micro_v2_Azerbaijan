#!/usr/bin/env bash
# Launch an Azerbaijani adaptation run, resuming automatically if one exists.
#
#   bash train_az.sh smoke      # 400 steps, verifies the pipeline and measures speed
#   bash train_az.sh 24gb       # full run, batch 16 x 4   (needs >= 8 GB)
#   bash train_az.sh quality    # full run, batch 32 x 2   (needs >= 16 GB)
#   bash train_az.sh large      # full run, batch 64 x 1   (needs >= 32 GB, fastest)
#
# If the run directory already holds checkpoints/latest.pth the script continues
# from it, so re-running the same command after a pod restart picks up where the
# run stopped. Set FRESH=1 to refuse an existing run instead.
set -euo pipefail

PROFILE="${1:-}"
if [ -z "${PROFILE}" ]; then
  # Braces inside a ${var:?message} expansion terminate it early, so the
  # usage text is printed here instead.
  echo "usage: train_az.sh {smoke|24gb|quality|large} [extra args...]" >&2
  exit 2
fi
shift || true

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${WORKDIR:-/workspace}"
BASE="${BASE:-${WORKDIR}/base}"
DATASET="${DATASET:-${WORKDIR}/prepared/az}"

case "${PROFILE}" in
  smoke)   PRESET="az-smoke.json" ;;
  24gb)    PRESET="az-multispeaker-24gb.json" ;;
  quality) PRESET="az-multispeaker-quality.json" ;;
  large)   PRESET="az-large-gpu.json" ;;
  *) echo "unknown profile: ${PROFILE}" >&2; exit 2 ;;
esac

OUTPUT="${OUTPUT:-${WORKDIR}/runs/az-${PROFILE}}"
CONFIG="${PROJECT}/configs/${PRESET}"
LATEST="${OUTPUT}/checkpoints/latest.pth"

# A detached launcher does not inherit an activated venv, and `exec` would fail
# with a bare "not found" deep inside a log nobody is watching. Accept an
# explicit venv and check up front instead.
if [ -n "${VENV:-}" ] && [ -x "${VENV}/bin/inflect-adapt" ]; then
  PATH="${VENV}/bin:${PATH}"
  export PATH
fi
if ! command -v inflect-adapt >/dev/null 2>&1; then
  echo "error: inflect-adapt is not on PATH." >&2
  echo "       Activate the toolkit venv, or pass VENV=/path/to/.venv" >&2
  exit 4
fi

echo "==> preset  : ${CONFIG}"
echo "==> base    : ${BASE}"
echo "==> dataset : ${DATASET}"
echo "==> output  : ${OUTPUT}"

RESUME_ARGS=()
if [ -f "${LATEST}" ]; then
  if [ "${FRESH:-0}" = "1" ]; then
    echo "==> FRESH=1 but ${LATEST} exists; refusing to overwrite an existing run." >&2
    echo "    Move or delete ${OUTPUT} first, or unset FRESH to resume." >&2
    exit 3
  fi
  # Resume validates run identity itself and rejects a mismatched checkpoint,
  # so this can only continue the run that produced it.
  RESUME_ARGS=(--resume "${LATEST}")
  echo "==> resuming from ${LATEST}"
else
  echo "==> starting a new run"
fi

exec inflect-adapt train \
  --base "${BASE}" \
  --dataset "${DATASET}" \
  --preset "${CONFIG}" \
  --output "${OUTPUT}" \
  --device cuda \
  "${RESUME_ARGS[@]}" \
  "$@"
