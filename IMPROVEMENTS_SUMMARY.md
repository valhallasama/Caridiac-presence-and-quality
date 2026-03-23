# Model Improvements Summary

## Changes Implemented

### 1. Continuous Quality Calculation (dataset.py)

**Previous (Binary)**:
```python
structure_presence = (areas > 50).float()  # 0 or 1
quality = 0.45 * structure_presence[0] + 0.20 * structure_presence[1] + 0.25 * structure_presence[2] + 0.10 * lv_area_score
```

**New (Continuous)**:
```python
# Normalize each structure's area to 0-1 range based on expected coverage
lv_norm = min(float(areas[0]) / expected_lv_pixels, 1.0)      # 0-1 continuous
myo_norm = min(float(areas[1]) / expected_myo_pixels, 1.0)    # 0-1 continuous  
la_norm = min(float(areas[2]) / expected_la_pixels, 1.0)      # 0-1 continuous

quality = 0.45 * lv_norm + 0.20 * myo_norm + 0.25 * la_norm + 0.10 * lv_norm
```

**Benefits**:
- ✅ Gradual scaling: Small structures get proportionally small scores
- ✅ No cliff effects: Smooth transition from poor to good
- ✅ Better training signal: Network learns finer distinctions
- ✅ More realistic: Reflects actual image quality

### 2. Dual Quality Heads (model.py)

**Added two separate quality prediction heads**:

1. **quality_head**: Predicts derived quality (continuous from segmentation)
2. **camus_quality_head**: Predicts CAMUS expert quality labels (poor=0.3, medium=0.6, good=0.99)

**Architecture**:
```python
self.quality_head = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(576, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
)

self.camus_quality_head = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(576, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
)
```

**Benefits**:
- ✅ Learns both objective (segmentation-based) and subjective (expert) quality
- ✅ Can compare and ensemble both predictions
- ✅ More robust quality assessment

### 3. Adjusted Synthetic Sample Strategy

**Previous (inferred from checkpoint name)**:
- ~20% negative samples (fully removed hearts)
- ~20% partial samples (partially removed hearts)
- ~60% original CAMUS

**New (Less Strict)**:
- **10% negative samples** (--synthetic-neg-prob 0.1)
- **20% partial samples** (--synthetic-partial-prob 0.2)
- **70% original CAMUS**

**Rationale**:
- Model was too conservative/strict on real-world videos
- More original samples = better generalization
- Still includes challenging negatives for robustness

### 4. Training Configuration

**Loss weights**:
```python
loss = (
    2.0 * loss_seg                    # Segmentation (main task)
    + 0.5 * loss_aux                  # Auxiliary segmentation
    + 0.5 * loss_quality              # Derived quality (continuous)
    + 0.3 * loss_camus_quality        # CAMUS expert quality
    + 0.3 * loss_presence             # Presence detection
    + 0.25 * loss_view                # View classification
)
```

**Training parameters**:
- Epochs: 200
- Batch size: 8
- Learning rate: 1e-4
- Quality loss: smooth_l1 (for derived), mse (for CAMUS)
- Checkpoint tag: `continuous_dual_quality_v1`

## Expected Improvements

1. **Better quality predictions**: Continuous calculation provides more nuanced scores
2. **Less strict detection**: 70% original samples should improve real-world performance
3. **Dual quality assessment**: Can use both objective and expert-based quality
4. **Smoother learning**: Continuous labels provide better gradients

## Training Command

```bash
./train_improved.sh
```

Or manually:
```bash
python3 src/train.py \
  --data-dir data/CAMUS_public/database_nifti \
  --epochs 200 \
  --batch-size 8 \
  --lr 1e-4 \
  --ckpt-tag continuous_dual_quality_v1 \
  --quality-source derived \
  --synthetic-neg-prob 0.1 \
  --synthetic-partial-prob 0.2
```

## Checkpoints Location

- Best model: `checkpoints/best_model_continuous_dual_quality_v1.pth`
- Last model: `checkpoints/last_model_continuous_dual_quality_v1.pth`
- Periodic: `checkpoints/training_continuous_dual_quality_v1/checkpoint_epoch_*.pth`

## Comparison with Previous Model

| Aspect | Previous | New |
|--------|----------|-----|
| Quality calculation | Binary (0/1) | Continuous (0-1) |
| Quality heads | Single | Dual (derived + CAMUS) |
| Synthetic negatives | ~20% | 10% |
| Synthetic partials | ~20% | 20% |
| Original samples | ~60% | 70% |
| Expected strictness | High | Moderate |

## Testing After Training

Use the same test script with new checkpoint:
```bash
python3 test_video.py test/Bhanu_para_1_ultra.mp4 checkpoints/best_model_continuous_dual_quality_v1.pth
```
