#!/bin/bash
# Stage 2: Multi-Task Fine-Tuning
# Fine-tune encoder + χ head and train solubility head

set -e  # Exit on error

# Arguments
CONFIG_PATH="${1:-configs/config.yaml}"
PRETRAINED_PATH="${2}"
DEVICE="${3:-cuda}"

echo "=================================================="
echo "Stage 2: Multi-Task Fine-Tuning"
echo "=================================================="
echo "Config: $CONFIG_PATH"
echo "Pretrained model: $PRETRAINED_PATH"
echo "Device: $DEVICE"
echo ""

# Check if pretrained path provided
if [ -z "$PRETRAINED_PATH" ]; then
    echo "ERROR: Pretrained model path required!"
    echo "Usage: $0 <config_path> <pretrained_model_path> [device]"
    echo ""
    echo "Example:"
    echo "  $0 configs/config.yaml results/dft_pretrain_20250118_123456/checkpoints/best_model.pt cuda"
    exit 1
fi

# Check if pretrained model exists
if [ ! -f "$PRETRAINED_PATH" ]; then
    echo "ERROR: Pretrained model not found at: $PRETRAINED_PATH"
    exit 1
fi

# Run training
python -m src.training.train_multitask \
    --config "$CONFIG_PATH" \
    --pretrained "$PRETRAINED_PATH" \
    --device "$DEVICE"

echo ""
echo "Multi-task training completed!"
echo "Check results/ directory for outputs."
