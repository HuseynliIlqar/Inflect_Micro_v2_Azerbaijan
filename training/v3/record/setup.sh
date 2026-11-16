#!/usr/bin/env bash
# Two isolated environments on the pod: one to generate (the teacher TTS), one to measure
# (Whisper, ECAPA, Praat). Same pins as the v1 corpus run, plus Praat and an older PyAV
# (faster-whisper 1.1.0 breaks on PyAV 14+).
# The teacher is named by exported variables: TEACHER_PIP (its pinned pip requirement) here,
# and the TEACHER_* variables listed in generate_takes.py.
set -euo pipefail

: "${TEACHER_PIP:?export TEACHER_PIP, the pinned pip requirement of the teacher package}"
: "${TEACHER_PACKAGE:?export TEACHER_PACKAGE (see generate_takes.py)}"
: "${TEACHER_CONFIG_CLASS:?export TEACHER_CONFIG_CLASS (see generate_takes.py)}"

export HF_HOME=/workspace/hf
mkdir -p /workspace/rec "$HF_HOME"
echo "=== system packages"
apt-get update -qq && apt-get install -y -qq ffmpeg libsndfile1 >/dev/null

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "=== generation environment"
uv venv -q --python 3.12 /workspace/venv_gen
uv pip install -q --python /workspace/venv_gen/bin/python "$TEACHER_PIP" soundfile numpy

echo "=== measurement environment"
uv venv -q --python 3.11 /workspace/venv_eval
uv pip install -q --python /workspace/venv_eval/bin/python \
  "torch==2.5.1" "torchaudio==2.5.1" "numpy<2" scipy soundfile librosa praat-parselmouth \
  "speechbrain==1.0.2" "huggingface_hub==0.25.2" \
  "faster-whisper==1.1.0" "av<14" requests nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"

echo "=== versions and teacher generation defaults"
/workspace/venv_gen/bin/python -c "
import importlib, os, torch
teacher = importlib.import_module(os.environ['TEACHER_PACKAGE'])
print('gen torch', torch.__version__, torch.cuda.is_available())
print('defaults', getattr(teacher, os.environ['TEACHER_CONFIG_CLASS'])())"
/workspace/venv_eval/bin/python -c "import torch, faster_whisper, parselmouth, speechbrain; print('eval torch', torch.__version__, torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== setup finished"
