#!/bin/bash
# Experimental χ K-Fold Cross-Validation
# Evaluate χ(T) head on small experimental dataset

set -e  # Exit on error

# Arguments
CONFIG_PATH="${1:-configs/config.yaml}"
DEVICE="${2:-cuda}"

echo "=================================================="
echo "Experimental χ K-Fold Cross-Validation"
echo "=================================================="
echo "Config: $CONFIG_PATH"
echo "Device: $DEVICE"
echo ""

# Run CV
python -m src.training.cv_exp_chi \
    --config "$CONFIG_PATH" \
    --device "$DEVICE"

echo ""
echo "Cross-validation completed!"
echo "Check results/ directory for aggregated metrics."
