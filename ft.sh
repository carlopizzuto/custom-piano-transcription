#!/bin/bash

# ============ Fine-Tune Piano Transcription System ============
export PYTHONPATH=$(pwd):$PYTHONPATH

# Download checkpoint and inference
CHECKPOINT_PATH="CRNN_note_F1=0.9677_pedal_F1=0.9186.pth"
wget -O "$CHECKPOINT_PATH" "https://zenodo.org/record/4034264/files/CRNN_note_F1=0.9677_pedal_F1=0.9186.pth?download=1"

# Note and pedal checkpoints
NOTE_CHECKPOINT="onset_offset_frame_velocity_pretrained.pth"
PEDAL_CHECKPOINT="pedal_pretrained.pth"

# Split combined checkpoint into note and pedal checkpoints
python3 pytorch/split_combined_checkpoint.py \
  --combined_checkpoint_path="$CHECKPOINT_PATH" \
  --note_checkpoint_path="$NOTE_CHECKPOINT" \
  --pedal_checkpoint_path="$PEDAL_CHECKPOINT"

# Workspace directory where intermediate results will be saved
WORKSPACE="./workspaces/piano_transcription_finetune"

# Non-classical dataset directory (ensure this dataset is prepared beforehand)
DATASET_DIR="./datasets/data"

# Pack audio files to HDF5 format for training 
python3 utils/features.py pack_maestro_dataset_to_hdf5 \
  --dataset_dir="$DATASET_DIR" \
  --workspace="$WORKSPACE"

# --- 1. Fine-Tune Note Transcription System ---
python3 pytorch/main.py train \
  --workspace="$WORKSPACE" \
  --model_type='Regress_onset_offset_frame_velocity_CRNN' \
  --loss_type='regress_onset_offset_frame_velocity_bce' \
  --augmentation='none' \
  --max_note_shift=0 \
  --batch_size=8 \
  --learning_rate=1e-5 \
  --reduce_iteration=10000 \
  --resume_iteration=0 \
  --early_stop=50000 \
  --cuda \
  --checkpoint_path="$NOTE_CHECKPOINT"

# --- 2. Fine-Tune Pedal Transcription System ---
python3 pytorch/main.py train \
  --workspace="$WORKSPACE" \
  --model_type='Regress_pedal_CRNN' \
  --loss_type='regress_pedal_bce' \
  --augmentation='none' \
  --max_note_shift=0 \
  --batch_size=8 \
  --learning_rate=1e-5 \
  --reduce_iteration=10000 \
  --resume_iteration=0 \
  --early_stop=50000 \
  --cuda \
  --checkpoint_path="$PEDAL_CHECKPOINT"

exit 0

# --- 3. Combine the Fine-Tuned Note and Pedal Models ---
# Update the paths to your fine-tuned model checkpoints
NOTE_CHECKPOINT_PATH="Regress_onset_offset_frame_velocity_CRNN_onset_F1=0.9677.pth"
PEDAL_CHECKPOINT_PATH="Regress_pedal_CRNN_onset_F1=0.9186.pth"
NOTE_PEDAL_CHECKPOINT_PATH="CRNN_note_F1=0.9677_pedal_F1=0.9186.pth"

python3 pytorch/combine_note_and_pedal_models.py \
  --note_checkpoint_path="$FINE_TUNED_NOTE_CHECKPOINT" \
  --pedal_checkpoint_path="$FINE_TUNED_PEDAL_CHECKPOINT" \
  --output_checkpoint_path="$FINE_TUNED_NOTE_PEDAL_CHECKPOINT"

# ============ Inference using Fine-Tuned Model ============
# Run inference using the fine-tuned combined model
python3 pytorch/inference.py \
  --model_type='Note_pedal' \
  --checkpoint_path="$FINE_TUNED_NOTE_PEDAL_CHECKPOINT" \
  --audio_path='path/to/your/audio/file.wav' \
  --cuda

# ============ Evaluate Fine-Tuned Model (Optional) ============
# Inference probability for evaluation
python3 pytorch/calculate_score_for_paper.py infer_prob \
  --workspace="$WORKSPACE" \
  --model_type='Note_pedal' \
  --checkpoint_path="$FINE_TUNED_NOTE_PEDAL_CHECKPOINT" \
  --augmentation='none' \
  --dataset='your_dataset' \
  --split='test' \
  --cuda

# Calculate metrics
python3 pytorch/calculate_score_for_paper.py calculate_metrics \
  --workspace="$WORKSPACE" \
  --model_type='Note_pedal' \
  --augmentation='none' \
  --dataset='your_dataset' \
  --split='test'