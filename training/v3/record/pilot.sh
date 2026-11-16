#!/usr/bin/env bash
# The pilot: ~600 sentences (40 per category), take A with v1's exact settings, a
# determinism check, two takes with mild variety (B_CONFIG / C_CONFIG), then scoring,
# calibration against 400 v1 clips and selection. Run detached:
#   B_CONFIG='{...}' C_CONFIG='{...}' setsid nohup bash pilot.sh > /workspace/rec/pilot.out 2>&1 < /dev/null &
set -uo pipefail
cd /workspace/record
export HF_HOME=/workspace/hf OMP_NUM_THREADS=1
SITE=$(ls -d /workspace/venv_eval/lib/python3*/site-packages)
GEN=/workspace/venv_gen/bin/python
EVAL="env LD_LIBRARY_PATH=$SITE/nvidia/cublas/lib:$SITE/nvidia/cudnn/lib /workspace/venv_eval/bin/python"
R=/workspace/rec
mkdir -p "$R"
rm -f "$R/pilot.done"
stage() { echo "=== $(date -u +%T) $1"; echo "$1" > "$R/stage.txt"; }
fail() { echo "=== FAILED: $1"; echo "failed: $1" > "$R/stage.txt"; exit 1; }

stage calibrate
[ -f "$R/gates.json" ] || $EVAL calibrate.py --sample /workspace/v1_sample --out "$R" || fail calibrate

stage take_A
$GEN generate_takes.py --take A --ids pilot_ids.txt --out "$R/takes" || fail take_A
stage determinism
$GEN generate_takes.py --take A_seed1 --ids det_ids.txt --out "$R/takes" --seed 1 || fail take_A_seed1
$EVAL determinism.py "$R/takes/A" "$R/takes/A_seed1" || fail determinism
stage take_B
$GEN generate_takes.py --take B --ids pilot_ids.txt --out "$R/takes" --config "${B_CONFIG:?}" --seed 2 || fail take_B
stage take_C
$GEN generate_takes.py --take C --ids pilot_ids.txt --out "$R/takes" --config "${C_CONFIG:?}" --seed 3 || fail take_C

for take in A B C; do
  stage "score_$take"
  $EVAL score.py --take "$take" --takes-dir "$R/takes" --out "$R/scores" --centroid "$R/v1_centroid.npy" || fail "score_$take"
done
stage select
$EVAL select_takes.py --scores "$R/scores" --gates "$R/gates.json" --takes A B C --out "$R/pilot_selection" --ids pilot_ids.txt || fail select
stage done
touch "$R/pilot.done"
