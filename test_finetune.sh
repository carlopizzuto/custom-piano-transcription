#!/bin/bash

# ============ Test fine-tune with minimal dataset ============

PYTHON_PATH=$(which python)
echo "Using Python from: $PYTHON_PATH"

# Add to the beginning of your test_finetune.sh
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export TORCH_USE_MPS_ALLOCATOR=1
export PYTORCH_DEBUG=1

# Add Python path and training parameters
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export BATCH_SIZE=1
export MAX_EPOCH=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONWARNINGS="ignore::FutureWarning"

# Set directories
DATASET_DIR="./datasets/non_classical_piano"
WORKSPACE="./workspaces/piano_transcription_finetune_test"

# Create directories
mkdir -p $WORKSPACE
mkdir -p "$WORKSPACE/checkpoints/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=none/max_note_shift=0/batch_size=1/"
mkdir -p "$WORKSPACE/checkpoints/Regress_pedal_CRNN/loss_type=regress_pedal_bce/augmentation=none/max_note_shift=0/batch_size=1/"

# Download pretrained model
CHECKPOINT_PATH="CRNN_note_F1=0.9677_pedal_F1=0.9186.pth"
MODEL_TYPE="Note_pedal"

# Pack dataset
$PYTHON_PATH utils/features.py pack_maestro_dataset_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

# Add after the dataset packing step
$PYTHON_PATH utils/debug_sampler.py --hdf5s_dir="$WORKSPACE/hdf5s/maestro"

# Add before the training steps
echo "Checking HDF5 directory structure:"
ls -R "$WORKSPACE/hdf5s/maestro"

# Add after packing dataset
echo "Verifying HDF5 files:"
$PYTHON_PATH utils/verify_hdf5.py "$WORKSPACE/hdf5s/maestro"

# Split checkpoint
NOTE_CHECKPOINT="$WORKSPACE/Regress_onset_offset_frame_velocity_CRNN_pretrained.pth"
PEDAL_CHECKPOINT="$WORKSPACE/Regress_pedal_CRNN_pretrained.pth"

$PYTHON_PATH pytorch/split_combined_checkpoint.py \
  --combined_checkpoint_path="$CHECKPOINT_PATH" \
  --note_checkpoint_path="$NOTE_CHECKPOINT" \
  --pedal_checkpoint_path="$PEDAL_CHECKPOINT"

# Run training commands
$PYTHON_PATH pytorch/main.py train --workspace=$WORKSPACE \
    --model_type='Regress_onset_offset_frame_velocity_CRNN' \
    --loss_type='regress_onset_offset_frame_velocity_bce' \
    --augmentation='none' --max_note_shift=0 --batch_size=$BATCH_SIZE \
    --learning_rate=1e-4 --reduce_iteration=5 --resume_iteration=0 \
    --early_stop=10 --checkpoint_path="$NOTE_CHECKPOINT" \
    --num_workers=0 --mini_data --detect_anomaly

if [ $? -ne 0 ]; then
    echo "Note model training failed"
    exit 1
fi

$PYTHON_PATH pytorch/main.py train --workspace=$WORKSPACE \
    --model_type='Regress_pedal_CRNN' \
    --loss_type='regress_pedal_bce' \
    --augmentation='none' --max_note_shift=0 --batch_size=$BATCH_SIZE \
    --learning_rate=1e-4 --reduce_iteration=5 --resume_iteration=0 \
    --early_stop=10 --checkpoint_path="$PEDAL_CHECKPOINT" \
    --num_workers=0 --mini_data --detect_anomaly

if [ $? -ne 0 ]; then
    echo "Pedal model training failed"
    exit 1
fi