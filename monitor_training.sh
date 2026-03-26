#!/bin/bash

# Monitor training progress
# Shows last 30 lines of log and updates every 10 seconds

LOG_FILE="checkpoints/multitask_stage2_resumed/training.log"

echo "=========================================="
echo "Training Monitor"
echo "=========================================="
echo "Log file: $LOG_FILE"
echo "Press Ctrl+C to stop monitoring"
echo "=========================================="
echo ""

# Check if training is running
if ps aux | grep -E "python.*train_multitask" | grep -v grep > /dev/null; then
    echo "✓ Training process is running"
else
    echo "✗ Training process not found"
fi

echo ""
echo "Latest log output:"
echo "=========================================="

# Show last 30 lines and keep updating
tail -f -n 30 "$LOG_FILE"
