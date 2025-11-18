#!/bin/bash
# Hyperparameter Optimization using Optuna
# Searches for best hyperparameters for multi-task model

set -e  # Exit on error

# Arguments
CONFIG_PATH="${1:-configs/config.yaml}"
N_TRIALS="${2:-100}"
DEVICE="${3:-cuda}"

echo "=================================================="
echo "Hyperparameter Optimization"
echo "=================================================="
echo "Config: $CONFIG_PATH"
echo "Number of trials: $N_TRIALS"
echo "Device: $DEVICE"
echo ""

# Run optimization
python scripts/hparam_opt.py \
    --config "$CONFIG_PATH" \
    --n-trials "$N_TRIALS" \
    --device "$DEVICE"

echo ""
echo "Hyperparameter search completed!"
echo "Best config saved to results/hparam_search/best_config.yaml"
