#!/usr/bin/env bash
# Deep test battery, detached so an SSH drop does not stop it. Writes /workspace/asr_eval/deep.done at the end.
#   setsid nohup bash /workspace/pod/run_deep.sh > /workspace/asr_eval/deep.log 2>&1 < /dev/null &
set -uo pipefail
cd /workspace/pod
L=/workspace/asr_eval
SITE=$(ls -d /workspace/venv_asr/lib/python3*/site-packages)
export HF_HOME=/workspace/hf TORCH_HOME=/workspace/hf/torch
ASR="env LD_LIBRARY_PATH=$SITE/nvidia/cublas/lib:$SITE/nvidia/cudnn/lib /workspace/venv_asr/bin/python"
PY=/workspace/venv/bin/python
rm -f "$L/deep.done"

echo "=== $(date -u +%T) tools"; /workspace/venv/bin/pip install -q praat-parselmouth 2>&1 | tail -1
echo "=== $(date -u +%T) render"
$PY hardtext.py render --package /workspace/final/avaz-v2 --name new
$PY hardtext.py render --package /workspace/final/avaz-v2 --name new-var0.33 --variation 0.333
$PY hardtext.py render --package /workspace/base --name base
$PY seed_stability.py render
echo "=== $(date -u +%T) score"
$ASR hardtext.py score > "$L/hardtext.log" 2>&1 &
$ASR seed_stability.py score > "$L/seed_stability.log" 2>&1 &
$PY stress_check.py > "$L/stress_check.log" 2>&1 &
wait
echo "=== $(date -u +%T) speed"
$PY rtf.py > "$L/rtf.log" 2>&1
echo "=== $(date -u +%T) finished"
touch "$L/deep.done"
