#!/bin/bash

# ============ Evaluate Piano Transcription System ============
export PYTHONPATH=$(pwd):$PYTHONPATH

# Workspace directory where intermediate results will be saved
WORKSPACE="/workspace/test-finetune"

FT_NOTE_CHECKPOINT="/workspace/aug-finetune/checkpoints/main/Regress_onset_offset_frame_velocity_CRNN/augmentation=none/batch_size=6/5400-frame_ap-0.488.pth"
FT_PEDAL_CHECKPOINT="/workspace/aug-finetune/checkpoints/main/Regress_pedal_CRNN/augmentation=none/batch_size=8/3600-pedal_frame_mae-0.169.pth" 
FT_NOTE_PEDAL_CHECKPOINT="/workspace/aug-finetune/best/combined/best-note_pedal_bs6_mixed.pth"

OG_NOTE_PEDAL_CHECKPOINT="CRNN_note_F1=0.9677_pedal_F1=0.9186.pth"

#python3 pytorch/combine_note_and_pedal_models.py \
#  --note_checkpoint_path="$FT_NOTE_CHECKPOINT" \
#  --pedal_checkpoint_path="$FT_PEDAL_CHECKPOINT" \
#  --output_checkpoint_path="$FT_NOTE_PEDAL_CHECKPOINT"

#python3 pytorch/calculate_score_for_paper.py infer_prob \
#  --workspace="$WORKSPACE" \
#  --model_type='Note_pedal' \
#  --checkpoint_path="$FT_NOTE_PEDAL_CHECKPOINT" \
#  --augmentation='none' \
#  --dataset='maps' \
#  --split='test' \
#  --cuda

python3 pytorch/calculate_score_for_paper.py calculate_metrics \
  --workspace="$WORKSPACE" \
  --model_type='Note_pedal' \
  --augmentation='none' \
  --dataset='maps' \
  --split='test'
