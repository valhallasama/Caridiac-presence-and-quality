# Model Improvements V2 Summary

## Changes from V1 to V2

### Problem with V1
The continuous quality calculation made the model **too conservative**:
- Video 1: Detection dropped from 30.25s → 9.10s (-70%)
- Video 2: Detection dropped from 3.90s → 0.60s (-85%)
- Max presence scores dropped by 67-74%

**Root cause**: Continuous quality gave very low scores to small structures, making training labels too conservative.

### V2 Improvements

#### 1. **Rescaled Continuous Quality** ✅

**Previous (V1 - too conservative)**:
```python
lv_norm = min(float(areas[0]) / expected_lv_pixels, 1.0)
quality = 0.45 * lv_norm + 0.20 * myo_norm + 0.25 * la_norm + 0.10 * lv_norm
```

**New (V2 - rescaled)**:
```python
lv_norm = min(float(areas[0]) / expected_lv_pixels, 1.0)
raw_score = 0.45 * lv_norm + 0.20 * myo_norm + 0.25 * la_norm + 0.10 * lv_norm
quality = 0.3 + 0.7 * raw_score  # Baseline 0.3 + scaled contribution
```

**Quality Score Examples**:

| Structure Size | V1 (too low) | V2 (rescaled) | Binary (original) |
|----------------|--------------|---------------|-------------------|
| 10% of expected | 0.045 | **0.33** | 0.0 |
| 25% of expected | 0.11 | **0.38** | 0.0 |
| 50% of expected | 0.23 | **0.46** | 0.45 |
| 100% of expected | 0.45 | **0.62** | 0.45 |
| Full (all structures) | 1.0 | **1.0** | 1.0 |

**Benefits**:
- ✅ Still continuous (smooth gradients)
- ✅ Similar distribution to binary approach
- ✅ Small structures get reasonable credit (0.3-0.4 instead of 0.0-0.1)
- ✅ Prevents overly conservative predictions

#### 2. **Increased Synthetic Sample Ratios** ✅

**Previous (V1)**:
- 10% negative samples
- 20% partial samples
- 70% original CAMUS

**New (V2)**:
- **20% negative samples** (doubled)
- **30% partial samples** (increased)
- **50% original CAMUS** (reduced but still majority)

**Rationale**:
- More challenging negative/partial samples = more robust model
- Model learns to distinguish real cardiac views from artifacts
- Better generalization to real-world videos
- Balanced between robustness and accuracy

### Training Configuration V2

**Model architecture**:
- Dual quality heads (derived + CAMUS)
- MobileNetV3-Small backbone
- LiteUNetDecoder
- Multi-task learning (seg, presence, quality, view)

**Training parameters**:
```bash
--epochs 200
--batch-size 8
--lr 1e-4
--quality-source derived
--quality-loss smooth_l1
--synthetic-neg-prob 0.2
--synthetic-partial-prob 0.3
--synthetic-neg-quality 0.0
--synthetic-partial-quality 0.3
```

**Loss weights**:
```python
loss = (
    2.0 * loss_seg                    # Segmentation (main)
    + 0.5 * loss_aux                  # Auxiliary segmentation
    + 0.5 * loss_quality              # Derived quality (rescaled continuous)
    + 0.3 * loss_camus_quality        # CAMUS expert quality
    + 0.3 * loss_presence             # Presence detection
    + 0.25 * loss_view                # View classification
)
```

## Expected Improvements Over V1

1. **Better real-world detection**: Rescaled quality should restore detection sensitivity
2. **More robust**: Higher synthetic sample ratio improves generalization
3. **Continuous quality benefits**: Smooth gradients while maintaining good distribution
4. **Dual quality heads**: Can compare objective vs expert quality

## Comparison Table

| Aspect | V1 (failed) | V2 (improved) | Original |
|--------|-------------|---------------|----------|
| Quality calculation | Continuous (unscaled) | **Continuous (rescaled)** | Binary |
| Quality range | 0.0-1.0 | **0.3-1.0** | 0.0-1.0 |
| Synthetic negatives | 10% | **20%** | ~20% |
| Synthetic partials | 20% | **30%** | ~20% |
| Original samples | 70% | **50%** | ~60% |
| Expected sensitivity | Too low | **Balanced** | Good |

## Training Command

```bash
./train_improved_v2.sh
```

Or manually:
```bash
python3 src/train.py \
  --data-dir data/CAMUS_public/database_nifti \
  --epochs 200 \
  --batch-size 8 \
  --lr 1e-4 \
  --ckpt-tag rescaled_dual_quality_v2 \
  --quality-source derived \
  --synthetic-neg-prob 0.2 \
  --synthetic-partial-prob 0.3
```

## Checkpoints Location

- Best model: `checkpoints/best_model_rescaled_dual_quality_v2.pth`
- Last model: `checkpoints/last_model_rescaled_dual_quality_v2.pth`
- Periodic: `checkpoints/training_rescaled_dual_quality_v2/checkpoint_epoch_*.pth`

## Testing After Training

```bash
python3 test_video.py test/Bhanu_para_1_ultra.mp4 checkpoints/best_model_rescaled_dual_quality_v2.pth
python3 test_video.py test/Bhanu_para_2_ultra.mp4 checkpoints/best_model_rescaled_dual_quality_v2.pth
```

## Expected Results

With rescaled quality and increased synthetic samples, we expect:
- **Detection sensitivity**: Similar to original model (20-30s for Video 1)
- **Quality assessment**: More nuanced (continuous 0.3-1.0 instead of discrete)
- **Robustness**: Better handling of artifacts and partial views
- **CAMUS quality**: Additional expert-based quality metric

The model should combine the best of both approaches:
- ✅ Continuous quality (smooth, nuanced)
- ✅ Good detection sensitivity (rescaled baseline)
- ✅ Robust to artifacts (more synthetic samples)
- ✅ Dual quality metrics (objective + expert)
