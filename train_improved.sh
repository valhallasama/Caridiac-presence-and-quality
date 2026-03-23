#!/bin/bash
# Training script with improved continuous quality calculation and dual quality heads
# Synthetic sample strategy: 10% negative, 20% partial, 70% original (less strict)

cd /home/tc115/Yue/Ultraprobe_guiding_system

python3 src/train.py \
  --data-dir data/CAMUS_public/database_nifti \
  --epochs 200 \
  --batch-size 8 \
  --lr 1e-4 \
  --num-workers 4 \
  --ckpt-tag continuous_dual_quality_v1 \
  --quality-source derived \
  --quality-loss smooth_l1 \
  --presence-loss-weight 0.3 \
  --view-loss-weight 0.25 \
  --synthetic-neg-prob 0.1 \
  --synthetic-partial-prob 0.2 \
  --synthetic-inpaint-radius 5 \
  --synthetic-neg-quality 0.0 \
  --synthetic-partial-quality 0.3

echo "Training completed!"
