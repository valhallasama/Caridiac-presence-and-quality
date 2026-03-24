#!/bin/bash

# Resume training from Stage 2 using previous best checkpoint
# This loads the Stage 1 trained model and continues from epoch 50

CHECKPOINT="checkpoints/multitask_staged_20260324_141323/best_model.pth"
DATA_DIR="data/CAMUS_public/database_nifti"

echo "=========================================="
echo "Resume Training from Stage 2"
echo "=========================================="
echo ""
echo "Loading checkpoint: $CHECKPOINT"
echo "Starting from: Epoch 50 (Stage 2 beginning)"
echo ""
echo "Stage 2 (epochs 50-150): Multi-task learning"
echo "  - Segmentation (already trained)"
echo "  - Presence (geometry-based)"
echo "  - Quality (physics-based)"
echo ""
echo "Stage 3 (epochs 151-200): Full refinement"
echo "  - Add consistency constraints"
echo "  - Add anatomical constraints"
echo "=========================================="
echo ""

# Create checkpoint directory
mkdir -p checkpoints/multitask_stage2_resumed

# Resume training from Stage 2 (epoch 50)
python3 train_multitask.py \
    --data_dir "$DATA_DIR" \
    --checkpoint_dir "checkpoints/multitask_stage2_resumed" \
    --epochs 200 \
    --batch_size 16 \
    --lr 1e-4 \
    --synthetic_neg_prob 0.2 \
    --synthetic_partial_prob 0.3 \
    --resume "$CHECKPOINT" \
    --start_epoch 50 \
    2>&1 | tee checkpoints/multitask_stage2_resumed/training.log
