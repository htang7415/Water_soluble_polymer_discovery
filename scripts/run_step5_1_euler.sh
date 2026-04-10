#!/bin/bash
# Step 5_1 runner for Euler.

set -e
MODEL_SIZE=${1:-small}
POLYMER_FAMILY=${2:-}
RUNS=${3:-}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

STEP5_CONFIG=${STEP5_CONFIG:-configs/config5.yaml}
BASE_CONFIG=${BASE_CONFIG:-configs/config.yaml}
ALLOW_PARTIAL=${ALLOW_PARTIAL:-0}

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -x "/srv/home/htang228/anaconda3/bin/conda" ]; then
  eval "$(/srv/home/htang228/anaconda3/bin/conda shell.bash hook)"
else
  echo "Conda not found. Please load conda and rerun."
  exit 2
fi

conda activate euler_active_learning

mkdir -p "$PROJECT_ROOT/.cache" "$PROJECT_ROOT/.mplconfig"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-"$PROJECT_ROOT/.cache"}
export MPLCONFIGDIR=${MPLCONFIGDIR:-"$PROJECT_ROOT/.mplconfig"}

CMD=(
  python scripts/step5_1_compare_inverse_design.py
  --config "$STEP5_CONFIG"
  --base_config "$BASE_CONFIG"
  --model_size "$MODEL_SIZE"
)
if [ -n "$POLYMER_FAMILY" ]; then
  CMD+=(--c_target "$POLYMER_FAMILY")
fi
if [ -n "$RUNS" ]; then
  CMD+=(--runs "$RUNS")
fi
if [ "$ALLOW_PARTIAL" = "1" ]; then
  CMD+=(--allow_partial)
fi

echo "Step 5_1 (Euler): cross-run inverse design comparison"
echo "  Model size:     $MODEL_SIZE"
if [ -n "$POLYMER_FAMILY" ]; then
  echo "  Polymer family: $POLYMER_FAMILY"
else
  echo "  Polymer family: config5 default"
fi
echo "  Conda env:      euler_active_learning"
echo "  Config:         $STEP5_CONFIG"
echo "  XDG_CACHE_HOME: $XDG_CACHE_HOME"
echo "  MPLCONFIGDIR:   $MPLCONFIGDIR"
if [ -n "$RUNS" ]; then
  echo "  Runs:           $RUNS"
else
  echo "  Runs:           all enabled completed runs"
fi

"${CMD[@]}"

echo "Done!"
