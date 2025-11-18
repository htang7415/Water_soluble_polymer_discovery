#!/bin/bash
# Stage 1: DFT χ Pretraining
# Train encoder + χ(T) head on large DFT dataset

set -e  # Exit on error

# Default config path
CONFIG_PATH="${1:-configs/config.yaml}"
DEVICE="${2:-cuda}"

echo "=================================================="
echo "Stage 1: DFT χ Pretraining"
echo "=================================================="
echo "Config: $CONFIG_PATH"
echo "Device: $DEVICE"
echo ""

# Run training
python -m src.training.train_dft \
    --config "$CONFIG_PATH" \
    --device "$DEVICE"

echo ""
echo "DFT pretraining completed!"
echo "Check results/ directory for outputs."
