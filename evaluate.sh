#!/bin/bash

# ============ Evaluate Piano Transcription System ============
export PYTHONPATH=$(pwd):$PYTHONPATH

# Workspace directory where intermediate results will be saved
WORKSPACE="./workspaces/piano_transcription_evaluate"

python3 pytorch/calculate_score_for_paper.py calculate_metrics \
  --workspace="$WORKSPACE" \
  --model_type='Note_pedal' \
  --augmentation='none' \
  --dataset='maestro' \
  --split='test'
