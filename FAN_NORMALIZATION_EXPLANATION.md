# Fan Area Normalization - The Correct Solution

## The Problem (User's Insight)

The user correctly identified that quality scores were unfair because:

**CAMUS Dataset**:
- Fan region: ~80-90% of 256×256 frame
- LV in good image: ~10,000 pixels (15% of total frame)

**Test Video (App Recording)**:
- Fan region: ~40-50% of 256×256 frame (smaller due to app UI, borders)
- LV in good image: ~5,000 pixels (same 15% of FAN, but only 7.5% of total frame)

**Previous V2 Calculation**:
```python
expected_lv_pixels = 65536 * 0.15 = 9,830 pixels  # Based on TOTAL frame
actual_lv_pixels = 5,000 pixels
quality = 5000 / 9830 = 0.51  # Unfairly low!
```

The same quality cardiac view gets **half the score** just because the fan is smaller!

## The Solution: Fan-Area Normalization

### V3 Calculation (Correct)

```python
# Step 1: Compute actual fan area from segmentation mask
fan_area = (mask.sum(dim=0) > 0).sum()  # Pixels with any structure

# Step 2: Expected pixels as percentage of FAN AREA
expected_lv_pixels = fan_area * 0.20    # 20% of fan
expected_myo_pixels = fan_area * 0.30   # 30% of fan
expected_la_pixels = fan_area * 0.15    # 15% of fan

# Step 3: Normalize by fan area
lv_norm = actual_lv_pixels / expected_lv_pixels
```

### Example Comparison

**CAMUS Image** (large fan):
- Fan area: 60,000 pixels (90% of frame)
- LV detected: 12,000 pixels
- Expected LV: 60,000 × 0.20 = 12,000 pixels
- LV norm: 12,000 / 12,000 = **1.0** ✅

**Test Video** (small fan):
- Fan area: 30,000 pixels (45% of frame)
- LV detected: 6,000 pixels (same proportion!)
- Expected LV: 30,000 × 0.20 = 6,000 pixels
- LV norm: 6,000 / 6,000 = **1.0** ✅

**Same quality → Same score!**

## Why This Matters

### Before (V2 - Frame Normalization)

| Image Type | Fan Size | LV Pixels | Expected (15% of frame) | Quality |
|------------|----------|-----------|------------------------|---------|
| CAMUS | 90% | 10,000 | 9,830 | 1.0 ✅ |
| Test video | 45% | 5,000 | 9,830 | 0.51 ❌ |

### After (V3 - Fan Normalization)

| Image Type | Fan Size | LV Pixels | Expected (20% of fan) | Quality |
|------------|----------|-----------|----------------------|---------|
| CAMUS | 90% (60k) | 10,000 | 12,000 | 0.83 ✅ |
| Test video | 45% (30k) | 5,000 | 6,000 | 0.83 ✅ |

**Same cardiac view quality → Same score!**

## Technical Implementation

### Dataset.py Changes

```python
# Compute fan area (non-zero pixels in any structure)
fan_area = float((mask.sum(dim=0) > 0).sum())

# Expected pixel counts as percentage of FAN AREA (not total frame)
expected_lv_pixels = fan_area * 0.20    # 20% of fan area
expected_myo_pixels = fan_area * 0.30   # 30% of fan area
expected_la_pixels = fan_area * 0.15    # 15% of fan area

# Normalize to 0-1
lv_norm = min(float(areas[0]) / expected_lv_pixels, 1.0)
myo_norm = min(float(areas[1]) / expected_myo_pixels, 1.0)
la_norm = min(float(areas[2]) / expected_la_pixels, 1.0)

# Weighted combination with baseline
raw_score = 0.45 * lv_norm + 0.20 * myo_norm + 0.25 * la_norm + 0.10 * lv_norm
quality = 0.3 + 0.7 * raw_score
```

### Key Benefits

1. **Fair comparison**: Quality scores comparable across different fan sizes
2. **Robust to recording setup**: App recordings, DICOM exports, different devices
3. **Maintains continuous benefits**: Still smooth gradients for learning
4. **Baseline protection**: 0.3 baseline prevents near-zero scores

## Expected Results

With fan-area normalization, the model should:

1. **Learn consistent quality metrics**: Same cardiac view quality → same training label
2. **Generalize better**: Works on CAMUS, test videos, and real clinical data
3. **Detect properly**: Higher presence scores on test videos (no unfair penalty)
4. **Be robust**: Handles different fan sizes, recording setups, image qualities

## Training Configuration V3

```bash
--ckpt-tag fan_normalized_v3
--synthetic-neg-prob 0.2
--synthetic-partial-prob 0.3
--quality-source derived
```

This is the **correct** approach that addresses the root cause identified by the user.
