#!/usr/bin/env bash
# Prepare a RunPod (or any Linux CUDA host) for Inflect-Micro-v2 adaptation.
#
#   bash runpod_setup.sh [WORKDIR]
#
# Expects this repository's configs/ and patches/ to be reachable; copy them to
# the pod first, or clone this project alongside.
set -euo pipefail

WORKDIR="${1:-/workspace}"
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="https://github.com/owenawsong/Inflect.git"
BASE_MODEL_REPO="owensong/Inflect-Micro-v2"

BASE_COMMIT="$(tr -d '[:space:]' < "${PROJECT}/patches/BASE_COMMIT.txt")"
PATCH="${PROJECT}/patches/multispeaker.patch"

echo "==> workdir       : ${WORKDIR}"
echo "==> project       : ${PROJECT}"
echo "==> upstream pin  : ${BASE_COMMIT}"

mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

# --- toolkit, pinned and patched -------------------------------------------
if [ ! -d Inflect ]; then
  git clone --filter=blob:none "${UPSTREAM}" Inflect
fi
cd Inflect
git fetch --depth 1 origin "${BASE_COMMIT}" || true
git checkout --quiet "${BASE_COMMIT}"
git checkout --quiet -- .
git apply --check "${PATCH}"
git apply "${PATCH}"
echo "==> multi-speaker patch applied"

# --- environment ------------------------------------------------------------
python -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel

# Install the CUDA build first so the toolkit does not pull a CPU-only torch.
# The pod template ships torch 2.4; the base model asks for >= 2.6, so it is
# replaced here. Override CUDA_TAG when the pod was created with CUDA 12.4.
CUDA_TAG="${CUDA_TAG:-cu128}"
echo "==> installing torch (${CUDA_TAG})"
python -m pip install torch --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
python -m pip install -e ./finetune
python -m pip install "huggingface_hub<1.0" num2words Unidecode

# --- base model -------------------------------------------------------------
cd "${WORKDIR}"
if [ ! -f base/model.pth ]; then
  hf download "${BASE_MODEL_REPO}" --local-dir base \
    --exclude "evaluation/*" "assets/*" "third_party/*"
fi

# --- verify -----------------------------------------------------------------
cd "${WORKDIR}/Inflect"
python -m pytest finetune/tests -q
python - <<'PY'
import torch
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"gpu   {props.name}  {props.total_memory / 1e9:.1f} GB")
PY

cat <<EOF

Setup complete.

  base model : ${WORKDIR}/base
  toolkit    : ${WORKDIR}/Inflect  (venv at Inflect/.venv)
  presets    : ${PROJECT}/configs

Next:
  source ${WORKDIR}/Inflect/.venv/bin/activate
  python -m pip install pyarrow -e ${PROJECT}          # converter + preflight

  # 1. build the manifest (filters here; prepare aborts on its first rejection)
  python ${PROJECT}/scripts/build_az_dataset.py --output ${WORKDIR}/data/az

  # 2. audit before spending GPU time
  inflect-preflight --manifest ${WORKDIR}/data/az/metadata.jsonl \
      --audio-root ${WORKDIR}/data/az --language az

  # 3. prepare (limits must match the converter)
  inflect-adapt prepare --manifest ${WORKDIR}/data/az/metadata.jsonl \
      --audio-root ${WORKDIR}/data/az \
      --language az --frontend espeak --output ${WORKDIR}/prepared/az \
      --min-duration-seconds 1.0 --max-duration-seconds 14.0 --validation-fraction 0.02
  inflect-adapt audit --dataset ${WORKDIR}/prepared/az

  # 4. measure throughput, then commit
  bash ${PROJECT}/scripts/train_az.sh smoke

  # 5. start the real run DETACHED so an SSH drop cannot kill it
  bash ${PROJECT}/scripts/train_bg.sh large
  bash ${PROJECT}/scripts/train_bg.sh large tail
EOF
