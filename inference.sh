#!/bin/bash

export PYTHONPATH=$(pwd):$PYTHONPATH

# Workspace directory where intermediate results will be saved
WORKSPACE="./workspaces/piano_transcription_finetune"

# Non-classical dataset directory (ensure this dataset is prepared beforehand)
DATASET_DIR="./datasets/data"

NOTE_CHECKPOINT_PATH="$WORKSPACE/final_Regress_onset_offset_frame_velocity_CRNN_100_iters.pth"
PEDAL_CHECKPOINT_PATH="$WORKSPACE/final_Regress_pedal_CRNN_100_iters.pth"
NOTE_PEDAL_CHECKPOINT_PATH="$WORKSPACE/final_combined_100_iters.pth"

python3 pytorch/combine_note_and_pedal_models.py \
  --note_checkpoint_path="$NOTE_CHECKPOINT_PATH" \
  --pedal_checkpoint_path="$PEDAL_CHECKPOINT_PATH" \
  --output_checkpoint_path="$NOTE_PEDAL_CHECKPOINT_PATH"

# ============ Inference using Fine-Tuned Model ============
# Run inference using the fine-tuned combined model
# python3 pytorch/inference.py \
#   --model_type='Note_pedal' \
#   --checkpoint_path="$NOTE_PEDAL_CHECKPOINT_PATH" \
#   --audio_path='path/to/your/audio/file.wav' \
#   --cuda

# ============ Evaluate Fine-Tuned Model (Optional) ============
# Inference probability for evaluation
python3 pytorch/calculate_score_for_paper.py infer_prob \
  --workspace="$WORKSPACE" \
  --model_type='Note_pedal' \
  --checkpoint_path="$NOTE_PEDAL_CHECKPOINT_PATH" \
  --augmentation='none' \
  --dataset="maestro" \
  --split='test' \
  --cuda

# Calculate metrics
python3 pytorch/calculate_score_for_paper.py calculate_metrics \
  --workspace="$WORKSPACE" \
  --model_type='Note_pedal' \
  --augmentation='none' \
  --dataset="maestro" \
  --split='test'
