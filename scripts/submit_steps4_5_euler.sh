#!/bin/bash
# Submit Step 4, then Step 5 and Step 5_1 on Euler.
# Assumes Step 0-3 outputs already exist.
# Usage:
#   bash scripts/submit_steps4_5_euler.sh <model_size> <polymer_family> [runs_csv]

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

mkdir -p logs

MODEL_SIZE=${1:-small}
POLYMER_FAMILY=${2:-}
RUNS=${3:-}

if [ -z "$POLYMER_FAMILY" ]; then
  echo "Usage: bash scripts/submit_steps4_5_euler.sh <model_size> <polymer_family> [runs_csv]"
  exit 2
fi

SIZE_TAG="m"
case "$MODEL_SIZE" in
  small|s) SIZE_TAG="s" ;;
  medium|m) SIZE_TAG="m" ;;
  large|l) SIZE_TAG="l" ;;
  xl|x) SIZE_TAG="xl" ;;
  *) SIZE_TAG="$MODEL_SIZE" ;;
esac

jid4=$(sbatch \
  --parsable \
  --job-name "smi_${SIZE_TAG}_4" \
  --output "logs/%x_%j.out" \
  --error "logs/%x_%j.err" \
  scripts/submit_step4_only_euler.sh \
  "$MODEL_SIZE")

STEP5_DEPENDENCY="afterok:${jid4}" \
  bash scripts/submit_step5_and_5_1_euler.sh "$MODEL_SIZE" "$POLYMER_FAMILY" "$RUNS"

echo "Submitted Step 4 -> Step 5 -> Step 5_1 on Euler:"
echo "  Step 4 job:      ${jid4}"
echo "  Model size:      ${MODEL_SIZE}"
echo "  Polymer family:  ${POLYMER_FAMILY}"
if [ -n "$RUNS" ]; then
  echo "  Step 5 runs:     ${RUNS}"
else
  echo "  Step 5 runs:     all enabled runs from config5"
fi
