#!/bin/bash
# Step 5 Optuna HPO local runner.

set -e
MODEL_SIZE=${1:-small}
STUDY_FAMILIES=${2:-S1,S2,S3,S4_rl,S4_dpo}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

STEP5_CONFIG=${STEP5_CONFIG:-configs/config5.yaml}
BASE_CONFIG=${BASE_CONFIG:-configs/config.yaml}
SKIP_REFIT=${SKIP_REFIT:-0}
FRESH_STUDY=${FRESH_STUDY:-1}

mkdir -p "$PROJECT_ROOT/.cache" "$PROJECT_ROOT/.mplconfig"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-"$PROJECT_ROOT/.cache"}
export MPLCONFIGDIR=${MPLCONFIGDIR:-"$PROJECT_ROOT/.mplconfig"}

CMD=(
  python scripts/step5_optuna_hpo.py
  --config "$STEP5_CONFIG"
  --base_config "$BASE_CONFIG"
  --model_size "$MODEL_SIZE"
  --study_families "$STUDY_FAMILIES"
  --force_enable
)
if [ "$SKIP_REFIT" = "1" ]; then
  CMD+=(--skip_refit)
fi
if [ "$FRESH_STUDY" = "1" ]; then
  CMD+=(--fresh_study)
fi

echo "Step 5 HPO: Optuna study + best-trial refit"
echo "  Model size:     $MODEL_SIZE"
echo "  Study families: $STUDY_FAMILIES"
echo "  Config:         $STEP5_CONFIG"
echo "  Fresh study:    $FRESH_STUDY"
echo "  XDG_CACHE_HOME: $XDG_CACHE_HOME"
echo "  MPLCONFIGDIR:   $MPLCONFIGDIR"

"${CMD[@]}"

echo "Done!"
