#!/bin/bash
# Step 6_3 local runner.

set -e
MODEL_SIZE=${1:-small}
RUNS=${2:-}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

STEP62_CONFIG=${STEP62_CONFIG:-configs/config6_2.yaml}
BASE_CONFIG=${BASE_CONFIG:-configs/config.yaml}
ALLOW_PARTIAL=${ALLOW_PARTIAL:-0}

mkdir -p "$PROJECT_ROOT/.cache" "$PROJECT_ROOT/.mplconfig"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-"$PROJECT_ROOT/.cache"}
export MPLCONFIGDIR=${MPLCONFIGDIR:-"$PROJECT_ROOT/.mplconfig"}

CMD=(
  python scripts/step6_3_compare_inverse_design.py
  --config "$STEP62_CONFIG"
  --base_config "$BASE_CONFIG"
  --model_size "$MODEL_SIZE"
)
if [ -n "$RUNS" ]; then
  CMD+=(--runs "$RUNS")
fi
if [ "$ALLOW_PARTIAL" = "1" ]; then
  CMD+=(--allow_partial)
fi

echo "Step 6_3: cross-run inverse design comparison"
echo "  Model size: $MODEL_SIZE"
echo "  Config:     $STEP62_CONFIG"
echo "  XDG_CACHE_HOME: $XDG_CACHE_HOME"
echo "  MPLCONFIGDIR:   $MPLCONFIGDIR"
if [ -n "$RUNS" ]; then
  echo "  Runs:       $RUNS"
else
  echo "  Runs:       all enabled completed runs"
fi

"${CMD[@]}"

echo "Done!"
