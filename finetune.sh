#!/bin/bash

# ============ Fine-tune pretrained model on non-classical piano music ============
# Pretrained model checkpoint path
CHECKPOINT_PATH="CRNN_note_F1=0.9677_pedal_F1=0.9186.pth"
wget -O $CHECKPOINT_PATH "https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
MODEL_TYPE="Note_pedal"

# Non-classical dataset directory (ensure this dataset is prepared beforehand)
DATASET_DIR="./datasets/non_classical_piano"

# Workspace directory where intermediate results will be saved
WORKSPACE="./workspaces/piano_transcription_finetune"

# Pack non-classical piano dataset to HDF5 format for training
python3 utils/features.py pack_maestro_dataset_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

# Split the combined checkpoint into separate note and pedal checkpoints
python3 pytorch/split_combined_checkpoint.py \
  --combined_checkpoint_path="CRNN_note_F1=0.9677_pedal_F1=0.9186.pth" \
  --note_checkpoint_path="Regress_onset_offset_frame_velocity_CRNN_pretrained.pth" \
  --pedal_checkpoint_path="Regress_pedal_CRNN_pretrained.pth"

# --- 1. Fine-tune the note transcription system ---
python3 pytorch/main.py train --workspace=$WORKSPACE \
    --model_type='Regress_onset_offset_frame_velocity_CRNN' \
    --loss_type='regress_onset_offset_frame_velocity_bce' \
    --augmentation='none' --max_note_shift=0 --batch_size=12 \
    --learning_rate=1e-4 --reduce_iteration=10000 --resume_iteration=0 \
    --early_stop=300000 --cuda --checkpoint_path='Regress_onset_offset_frame_velocity_CRNN_pretrained.pth'

# --- 2. Fine-tune the pedal transcription system ---
python3 pytorch/main.py train --workspace=$WORKSPACE \
    --model_type='Regress_pedal_CRNN' \
    --loss_type='regress_pedal_bce' \
    --augmentation='none' --max_note_shift=0 --batch_size=12 \
    --learning_rate=1e-4 --reduce_iteration=10000 --resume_iteration=0 \
    --early_stop=300000 --cuda --checkpoint_path='Regress_pedal_CRNN_pretrained.pth'

# --- 3. Combine the fine-tuned note and pedal models ---
# Paths to the fine-tuned note and pedal model checkpoints
FINE_TUNED_NOTE_CHECKPOINT_PATH="Regress_onset_offset_frame_velocity_CRNN_finetuned.pth"
FINE_TUNED_PEDAL_CHECKPOINT_PATH="Regress_pedal_CRNN_finetuned.pth"
FINE_TUNED_NOTE_PEDAL_CHECKPOINT_PATH="CRNN_note_pedal_finetuned.pth"

# Combine the fine-tuned note and pedal models into one checkpoint
python3 pytorch/combine_note_and_pedal_models.py \
    --note_checkpoint_path=$FINE_TUNED_NOTE_CHECKPOINT_PATH \
    --pedal_checkpoint_path=$FINE_TUNED_PEDAL_CHECKPOINT_PATH \
    --output_checkpoint_path=$FINE_TUNED_NOTE_PEDAL_CHECKPOINT_PATH

# ============ Evaluate the fine-tuned model (optional) ============
# Evaluate the fine-tuned model on non-classical test data
python3 pytorch/calculate_score_for_paper.py infer_prob --workspace=$WORKSPACE \
    --model_type='Note_pedal' --checkpoint_path=$FINE_TUNED_NOTE_PEDAL_CHECKPOINT_PATH \
    --augmentation='none' --dataset='non_classical' --split='test' --cuda

# Calculate metrics for the fine-tuned model
python3 pytorch/calculate_score_for_paper.py calculate_metrics --workspace=$WORKSPACE \
    --model_type='Note_pedal' --augmentation='aug' --dataset='non_classical' --split='test'