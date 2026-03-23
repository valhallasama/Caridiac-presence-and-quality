#!/bin/bash
# Training script v3 with fan-area-normalized quality and adjusted synthetic ratios
# Key improvement: Quality normalized by fan area, not total frame area
# Synthetic sample strategy: 20% negative, 30% partial, 50% original

cd /home/tc115/Yue/Ultraprobe_guiding_system

python3 src/train.py \
  --data-dir data/CAMUS_public/database_nifti \
  --epochs 200 \
  --batch-size 8 \
  --lr 1e-4 \
  --num-workers 4 \
  --ckpt-tag fan_normalized_v3 \
  --quality-source derived \
  --quality-loss smooth_l1 \
  --presence-loss-weight 0.3 \
  --view-loss-weight 0.25 \
  --synthetic-neg-prob 0.2 \
  --synthetic-partial-prob 0.3 \
  --synthetic-inpaint-radius 5 \
  --synthetic-neg-quality 0.0 \
  --synthetic-partial-quality 0.3

echo "Training v3 completed!"
