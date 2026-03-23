#!/bin/bash

# Multi-Task Training with Staged Strategy
# Mathematical redesign: Geometry-based presence + Physics-based quality

echo "=========================================="
echo "Multi-Task Staged Training"
echo "=========================================="
echo ""
echo "Architecture:"
echo "  Old: image → segmentation → presence → quality (fragile)"
echo "  New: image → (presence, quality, segmentation) (robust)"
echo ""
echo "Training Stages:"
echo "  Stage 1 (epochs 1-50):   Segmentation only"
echo "  Stage 2 (epochs 51-150): Multi-task (seg + presence + quality)"
echo "  Stage 3 (epochs 151-200): Full (+ consistency + anatomical)"
echo ""
echo "=========================================="
echo ""

# Configuration
DATA_DIR="data/CAMUS_public/database_nifti"
CHECKPOINT_DIR="checkpoints/multitask_staged_$(date +%Y%m%d_%H%M%S)"
EPOCHS=200
BATCH_SIZE=16
LR=0.0001
IMG_SIZE=256
SYNTHETIC_NEG_PROB=0.2
SYNTHETIC_PARTIAL_PROB=0.3
NUM_WORKERS=4

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    echo "Please ensure CAMUS dataset is available."
    exit 1
fi

# Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

# Save configuration
cat > "$CHECKPOINT_DIR/config.txt" << EOF
Multi-Task Staged Training Configuration
=========================================
Date: $(date)
Data Directory: $DATA_DIR
Checkpoint Directory: $CHECKPOINT_DIR
Epochs: $EPOCHS
Batch Size: $BATCH_SIZE
Learning Rate: $LR
Image Size: $IMG_SIZE
Synthetic Negative Probability: $SYNTHETIC_NEG_PROB
Synthetic Partial Probability: $SYNTHETIC_PARTIAL_PROB
Number of Workers: $NUM_WORKERS

Training Strategy:
- Stage 1 (epochs 1-50): Segmentation only (L_seg)
- Stage 2 (epochs 51-150): Multi-task (L_seg + L_pres + L_qual)
- Stage 3 (epochs 151-200): Full (+ L_cons + L_anat)

Loss Weights:
- Segmentation: 1.0
- Presence: 0.5
- Quality: 0.5
- Consistency: 0.2
- Anatomical: 0.1

Ground Truth:
- Presence: Geometry-based (area ratios)
- Quality: Physics-based (sharpness + contrast + structure)
EOF

echo "Configuration saved to: $CHECKPOINT_DIR/config.txt"
echo ""

# Start training
python3 train_multitask.py \
    --data_dir "$DATA_DIR" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --img_size $IMG_SIZE \
    --synthetic_neg_prob $SYNTHETIC_NEG_PROB \
    --synthetic_partial_prob $SYNTHETIC_PARTIAL_PROB \
    --num_workers $NUM_WORKERS \
    2>&1 | tee "$CHECKPOINT_DIR/training.log"

echo ""
echo "=========================================="
echo "Training complete!"
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo "Training log: $CHECKPOINT_DIR/training.log"
echo "=========================================="
