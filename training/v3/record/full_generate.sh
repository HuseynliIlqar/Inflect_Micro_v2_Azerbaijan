#!/usr/bin/env bash
# The full recording (training/v3/PLAN.md 3.8): takes A, B, C over every sentence and D, E
# over the yes/no questions (A1-A3). Clips left by the pilot are reused, because the
# generator skips finished files. Writes $R/gen_<take>.done after each take, so
# full_score.sh can score it while the next take is generated. Run detached:
#   setsid nohup bash full_generate.sh > /workspace/rec/full_generate.out 2>&1 < /dev/null &
set -uo pipefail
cd /workspace/record
export HF_HOME=/workspace/hf OMP_NUM_THREADS=1
GEN=/workspace/venv_gen/bin/python
R=/workspace/rec
SAMPLED='{"position_temperature": 5.0}'
stage() { echo "=== $(date -u +%T) $1"; echo "$1" > "$R/gen_stage.txt"; }
fail() { echo "=== FAILED: $1"; echo "failed: $1" > "$R/gen_stage.txt"; exit 1; }
take() {  # take NAME [generate_takes.py options]
  local name=$1; shift
  [ -f "$R/gen_$name.done" ] && return 0
  stage "take_$name"
  $GEN generate_takes.py --take "$name" --out "$R/takes" "$@" || fail "take_$name"
  touch "$R/gen_$name.done"
}

$GEN -c "
import json
rows = [json.loads(line) for line in open('sentences_v2.jsonl', encoding='utf-8') if line.strip()]
open('question_ids.txt', 'w').write('\n'.join(r['id'] for r in rows if r['cat'] in ('A1', 'A2', 'A3')) + '\n')
" || fail question_ids

take A
take B --config "$SAMPLED" --seed 2
take C --config "$SAMPLED" --seed 3
take D --config "$SAMPLED" --seed 613 --ids question_ids.txt
take E --config "$SAMPLED" --seed 214 --ids question_ids.txt
stage done
