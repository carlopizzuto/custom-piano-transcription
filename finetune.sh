#!/bin/bash

# ============ Fine-tune pretrained model on non-classical piano music ============

PYTHON_PATH=$(which python)
echo "Using Python from: $PYTHON_PATH"


# Add Python path and training parameters
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export BATCH_SIZE=2
export MAX_EPOCH=2

export PYTHONWARNINGS="ignore::FutureWarning"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Set directories first
DATASET_DIR="./datasets/non_classical_piano"
WORKSPACE="./workspaces/piano_transcription_finetune"

# Create directories with relative paths
mkdir -p $WORKSPACE
mkdir -p "$WORKSPACE/checkpoints/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=none/max_note_shift=0/batch_size=4/"
mkdir -p "$WORKSPACE/checkpoints/Regress_pedal_CRNN/loss_type=regress_pedal_bce/augmentation=none/max_note_shift=0/batch_size=4/"

# Download pretrained model
CHECKPOINT_PATH="CRNN_note_F1=0.9677_pedal_F1=0.9186.pth"
# wget -O $CHECKPOINT_PATH "https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
MODEL_TYPE="Note_pedal"

# Pack dataset
$PYTHON_PATH utils/features.py pack_maestro_dataset_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

# Split checkpoint - specify full paths
NOTE_CHECKPOINT="$WORKSPACE/Regress_onset_offset_frame_velocity_CRNN_pretrained.pth"
PEDAL_CHECKPOINT="$WORKSPACE/Regress_pedal_CRNN_pretrained.pth"

$PYTHON_PATH pytorch/split_combined_checkpoint.py \
  --combined_checkpoint_path="$CHECKPOINT_PATH" \
  --note_checkpoint_path="$NOTE_CHECKPOINT" \
  --pedal_checkpoint_path="$PEDAL_CHECKPOINT"

# Fine-tune note model
$PYTHON_PATH pytorch/main.py train --workspace=$WORKSPACE \
    --model_type='Regress_onset_offset_frame_velocity_CRNN' \
    --loss_type='regress_onset_offset_frame_velocity_bce' \
    --augmentation='none' --max_note_shift=0 --batch_size=$BATCH_SIZE \
    --learning_rate=1e-4 --reduce_iteration=100 --resume_iteration=0 \
    --early_stop=1000 --checkpoint_path="$NOTE_CHECKPOINT" \
    --num_workers=0 

# Fine-tune pedal model
$PYTHON_PATH pytorch/main.py train --workspace=$WORKSPACE \
    --model_type='Regress_pedal_CRNN' \
    --loss_type='regress_pedal_bce' \
    --augmentation='none' --max_note_shift=0 --batch_size=$BATCH_SIZE \
    --learning_rate=1e-4 --reduce_iteration=100 --resume_iteration=0 \
    --early_stop=1000 --checkpoint_path="$PEDAL_CHECKPOINT" \
    --num_workers=0 

# Wait for training to complete and check status
sleep 5

# Check if training completed successfully
if [ $? -ne 0 ]; then
    echo "Training failed with exit code $?"
    exit 1
fi

# Additional check for model files
if [ ! -s "$FINE_TUNED_NOTE_CHECKPOINT" ] || [ ! -s "$FINE_TUNED_PEDAL_CHECKPOINT" ]; then
    echo "Error: Fine-tuned model files not found or empty. Training may have failed."
    echo "Note checkpoint: $FINE_TUNED_NOTE_CHECKPOINT"
    echo "Pedal checkpoint: $FINE_TUNED_PEDAL_CHECKPOINT"
    exit 1
fi

# Update paths to point to the trained models
FINE_TUNED_NOTE_CHECKPOINT="$WORKSPACE/checkpoints/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=none/max_note_shift=0/batch_size=4/1000_iterations.pth"
FINE_TUNED_PEDAL_CHECKPOINT="$WORKSPACE/checkpoints/Regress_pedal_CRNN/loss_type=regress_pedal_bce/augmentation=none/max_note_shift=0/batch_size=4/1000_iterations.pth"

# Combine fine-tuned models
FINE_TUNED_COMBINED_CHECKPOINT="$WORKSPACE/CRNN_note_pedal_finetuned.pth"

$PYTHON_PATH pytorch/combine_note_and_pedal_models.py \
    --note_checkpoint_path="$FINE_TUNED_NOTE_CHECKPOINT" \
    --pedal_checkpoint_path="$FINE_TUNED_PEDAL_CHECKPOINT" \
    --output_checkpoint_path="$FINE_TUNED_COMBINED_CHECKPOINT"

# Evaluate
$PYTHON_PATH pytorch/calculate_score_non_classical.py infer_prob --workspace=$WORKSPACE \
    --model_type='Note_pedal' --checkpoint_path="$FINE_TUNED_COMBINED_CHECKPOINT" \
    --augmentation='none' --dataset='non_classical' --split='test'

$PYTHON_PATH pytorch/calculate_score_non_classical.py calculate_metrics --workspace=$WORKSPACE \
    --model_type='Note_pedal' --augmentation='aug' --dataset='non_classical' --split='test'