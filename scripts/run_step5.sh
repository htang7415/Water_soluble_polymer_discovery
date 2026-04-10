#!/bin/bash
# Step 5 local runner.

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

mkdir -p "$PROJECT_ROOT/.cache" "$PROJECT_ROOT/.mplconfig"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-"$PROJECT_ROOT/.cache"}
export MPLCONFIGDIR=${MPLCONFIGDIR:-"$PROJECT_ROOT/.mplconfig"}

CMD=(
  python scripts/step5_inverse_design.py
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

echo "Step 5: inverse design benchmark"
echo "  Model size: $MODEL_SIZE"
if [ -n "$POLYMER_FAMILY" ]; then
  echo "  Polymer family: $POLYMER_FAMILY"
else
  echo "  Polymer family: config5 default"
fi
echo "  Config:     $STEP5_CONFIG"
echo "  XDG_CACHE_HOME: $XDG_CACHE_HOME"
echo "  MPLCONFIGDIR:   $MPLCONFIGDIR"
if [ -n "$RUNS" ]; then
  echo "  Runs:       $RUNS"
else
  echo "  Runs:       all enabled runs from config5"
fi

"${CMD[@]}"

echo "Done!"
