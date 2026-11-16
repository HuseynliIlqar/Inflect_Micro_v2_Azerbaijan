#!/usr/bin/env bash
# Second part of the battery: the hard-text renders (fixed import), after run_deep.sh has finished.
set -uo pipefail
cd /workspace/pod
L=/workspace/asr_eval
SITE=$(ls -d /workspace/venv_asr/lib/python3*/site-packages)
export HF_HOME=/workspace/hf
PY=/workspace/venv/bin/python
rm -f "$L/deep2.done"
while [ ! -f "$L/deep.done" ]; do sleep 10; done
echo "=== $(date -u +%T) hard text"
$PY hardtext.py render --package /workspace/final/avaz-v2 --name new
$PY hardtext.py render --package /workspace/final/avaz-v2 --name new-var0.33 --variation 0.333
$PY hardtext.py render --package /workspace/base --name base
LD_LIBRARY_PATH=$SITE/nvidia/cublas/lib:$SITE/nvidia/cudnn/lib /workspace/venv_asr/bin/python hardtext.py score > "$L/hardtext.log" 2>&1
echo "=== $(date -u +%T) finished"
touch "$L/deep2.done"
