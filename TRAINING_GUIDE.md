# Ultraprobe Training Guide - Crash-Resistant Training

## ✅ Updated Training System

The training system now saves **full checkpoints** including:
- Model weights
- Optimizer state
- Epoch number
- Best validation Dice score
- Validation metrics

**Checkpoints saved:**
1. **Best model** - When validation Dice improves
2. **Last model** - After every epoch
3. **Periodic checkpoints** - Every 5 epochs

---

## 🚀 Start Fresh Training

```bash
cd /home/tc115/Yue/Ultraprobe_guiding_system

# Train from scratch with new checkpoint system
python3 src/train.py \
  --epochs 100 \
  --batch-size 8 \
  --lr 1e-4 \
  --ckpt-tag multi_v3
```

**What happens:**
- Saves checkpoints to `checkpoints/training_multi_v3/`
- Checkpoint every 5 epochs: `checkpoint_epoch_5.pth`, `checkpoint_epoch_10.pth`, etc.
- Best model: `checkpoints/best_model_multi_v3.pth`
- Last model: `checkpoints/last_model_multi_v3.pth`

---

## 🔄 Resume After Crash

If training crashes at epoch 47:

```bash
# Resume from last checkpoint
python3 resume_training.py \
  --checkpoint checkpoints/last_model_multi_v3.pth \
  --epochs 53 \
  --ckpt-tag multi_v3_resumed

# Or resume from specific epoch checkpoint
python3 resume_training.py \
  --checkpoint checkpoints/training_multi_v3/checkpoint_epoch_45.pth \
  --epochs 55 \
  --ckpt-tag multi_v3_resumed
```

**The resume script now:**
- ✅ Handles old checkpoint format (model weights only)
- ✅ Handles new checkpoint format (full state)
- ✅ Restores optimizer state
- ✅ Continues from exact epoch
- ✅ Uses `strict=False` for architecture changes

---

## 📁 Checkpoint Structure

```
checkpoints/
├── best_model_multi_v3.pth          # Best performing model
├── last_model_multi_v3.pth          # Most recent model
└── training_multi_v3/               # Periodic checkpoints
    ├── checkpoint_epoch_5.pth
    ├── checkpoint_epoch_10.pth
    ├── checkpoint_epoch_15.pth
    ├── checkpoint_epoch_20.pth
    └── ...
```

**Each checkpoint contains:**
```python
{
    'epoch': 45,
    'model_state_dict': {...},
    'optimizer_state_dict': {...},
    'best_val_dice': 0.8234,
    'val_metrics': {
        'loss': 0.1234,
        'dice': 0.8234,
        'centroid_error': 12.3,
        ...
    }
}
```

---

## 💡 Crash Recovery Examples

### **Scenario 1: Training crashes at epoch 67**

```bash
# Resume from last checkpoint (epoch 67)
python3 resume_training.py \
  --checkpoint checkpoints/last_model_multi_v3.pth \
  --epochs 33  # To complete 100 total epochs
```

### **Scenario 2: Want to go back to epoch 60**

```bash
# Resume from epoch 60 checkpoint
python3 resume_training.py \
  --checkpoint checkpoints/training_multi_v3/checkpoint_epoch_60.pth \
  --epochs 40  # To complete 100 total epochs
```

### **Scenario 3: Old checkpoint (no optimizer state)**

```bash
# Resume from old format checkpoint
python3 resume_training.py \
  --checkpoint checkpoints/best_model_multi.pth \
  --epochs 50
```

The script will:
- Detect old format
- Load model weights
- Use fresh optimizer
- Continue training

---

## 🎯 Training Commands

### **Fresh Training (Recommended)**

```bash
# Start new training with crash-resistant checkpoints
python3 src/train.py \
  --epochs 100 \
  --batch-size 8 \
  --lr 1e-4 \
  --ckpt-tag multi_v3
```

### **Resume from Crash**

```bash
# Resume from last checkpoint
python3 resume_training.py \
  --checkpoint checkpoints/last_model_multi_v3.pth \
  --epochs 50 \
  --ckpt-tag multi_v3_resumed
```

---

## 📊 Monitoring Training

```bash
# Watch training in real-time
tail -f nohup.out

# Or run in foreground
python3 src/train.py --epochs 100 --batch-size 8
```

---

## ✅ Summary

**Before (old system):**
- ❌ Only saved best and last model
- ❌ Lost optimizer state
- ❌ Couldn't resume from specific epoch
- ❌ Architecture changes broke loading

**After (new system):**
- ✅ Saves checkpoint every 5 epochs
- ✅ Saves full state (model + optimizer)
- ✅ Can resume from any epoch
- ✅ Handles architecture changes gracefully
- ✅ Never lose more than 5 epochs of work

**Start training now:**
```bash
cd /home/tc115/Yue/Ultraprobe_guiding_system
python3 src/train.py --epochs 100 --batch-size 8 --ckpt-tag multi_v3
```
