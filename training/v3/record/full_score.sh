#!/usr/bin/env bash
# Scores each take as soon as full_generate.sh has finished it, then selects the takes.
# Gates come from v1's scores with the pilot's percentiles (calibrate.py --from-scores);
# the pilot's own scores move to scores_pilot, so every clip is scored by the same code.
# Run detached next to full_generate.sh:
#   setsid nohup bash full_score.sh > /workspace/rec/full_score.out 2>&1 < /dev/null &
set -uo pipefail
cd /workspace/record
export HF_HOME=/workspace/hf OMP_NUM_THREADS=1
SITE=$(ls -d /workspace/venv_eval/lib/python3*/site-packages)
EVAL="env LD_LIBRARY_PATH=$SITE/nvidia/cublas/lib:$SITE/nvidia/cudnn/lib /workspace/venv_eval/bin/python"
R=/workspace/rec
WORKERS=${WORKERS:-16}
stage() { echo "=== $(date -u +%T) $1"; echo "$1" > "$R/score_stage.txt"; }
fail() { echo "=== FAILED: $1"; echo "failed: $1" > "$R/score_stage.txt"; exit 1; }
rm -f "$R/full.done"

stage gates
[ -f "$R/gates_v0.json" ] || cp "$R/gates.json" "$R/gates_v0.json"
$EVAL calibrate.py --from-scores "$R/v1_scores.jsonl" --out "$R" || fail gates
[ -d "$R/scores_pilot" ] || mv "$R/scores" "$R/scores_pilot"

for take in A B C D E; do
  stage "wait_$take"
  until [ -f "$R/gen_$take.done" ]; do
    grep -q failed "$R/gen_stage.txt" 2>/dev/null && fail "generation failed"
    sleep 60
  done
  stage "score_$take"
  $EVAL score.py --take "$take" --takes-dir "$R/takes" --out "$R/scores" --centroid "$R/v1_centroid.npy" \
    --workers "$WORKERS" || fail "score_$take"
done
stage select
$EVAL select_takes.py --scores "$R/scores" --gates "$R/gates.json" --takes A B C D E --question-takes B C D E \
  --out "$R/selection" || fail select
stage done
touch "$R/full.done"
