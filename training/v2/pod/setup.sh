#!/usr/bin/env bash
# Install the Inflect toolkit (pinned commit + Avaz's patch with the fast aligner) on the pod.
#   bash setup.sh            (run from /workspace/kit)
set -euo pipefail

KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK=/workspace
BASE_COMMIT="$(tr -d '[:space:]' < "$KIT/BASE_COMMIT.txt")"

echo "=== system packages"
apt-get update -qq && apt-get install -y -qq libsndfile1 git >/dev/null

echo "=== toolkit at ${BASE_COMMIT}"
cd "$WORK"
if [ ! -d Inflect ]; then
  git clone --quiet --filter=blob:none https://github.com/owenawsong/Inflect.git Inflect
fi
cd Inflect
git checkout --quiet "$BASE_COMMIT"
git checkout --quiet -- .
git apply --check "$KIT/multispeaker.patch"
git apply "$KIT/multispeaker.patch"
python3 "$KIT/add_walltime.py" finetune/inflect_finetune/training.py

echo "=== environment (reuses the image's CUDA torch)"
python3 -m venv --system-site-packages "$WORK/venv"
"$WORK/venv/bin/python" -m pip install -q --upgrade pip wheel
# Same eSpeak build and phonemizer as the stress rules were checked with on Windows.
"$WORK/venv/bin/python" -m pip install -q "espeakng-loader==0.2.4" "phonemizer==3.4.0" Unidecode num2words pytest fastapi uvicorn
"$WORK/venv/bin/python" -m pip install -q -e ./finetune

echo "=== toolkit tests"
"$WORK/venv/bin/python" -m pytest -q finetune/tests 2>&1 | tail -3

"$WORK/venv/bin/python" - <<'PY'
import torch, phonemizer, espeakng_loader
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("phonemizer", phonemizer.__version__)
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("gpu", p.name, round(p.total_memory / 1e9, 1), "GB")
PY
echo "=== setup finished"
