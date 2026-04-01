#!/bin/bash
# Submit Step 6_2 runs plus dependent Step 6_3 comparison on NREL.

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

mkdir -p logs

MODEL_SIZE=${1:-small}
RUNS=${2:-}
STEP62_CONFIG=${STEP62_CONFIG:-configs/config6_2.yaml}
BASE_CONFIG=${BASE_CONFIG:-configs/config.yaml}

SIZE_TAG="m"
case "$MODEL_SIZE" in
  small|s) SIZE_TAG="s" ;;
  medium|m) SIZE_TAG="m" ;;
  large|l) SIZE_TAG="l" ;;
  xl|x) SIZE_TAG="xl" ;;
  *) SIZE_TAG="$MODEL_SIZE" ;;
esac

readarray -t META_LINES < <(MODEL_SIZE="$MODEL_SIZE" RUNS="$RUNS" STEP62_CONFIG="$STEP62_CONFIG" BASE_CONFIG="$BASE_CONFIG" python - <<'PY'
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(".").resolve()))
from src.step6_2.config import load_step6_2_config

resolved = load_step6_2_config(
    config_path=os.environ["STEP62_CONFIG"],
    base_config_path=os.environ["BASE_CONFIG"],
    model_size=os.environ.get("MODEL_SIZE") or None,
)
runs_csv = os.environ.get("RUNS", "").strip()
if runs_csv:
    requested = [item.strip() for item in runs_csv.split(",") if item.strip()]
    unknown = [run for run in requested if run not in resolved.enabled_runs]
    if unknown:
        raise ValueError(f"Requested runs are not enabled in config6_2: {unknown}")
    selected = requested
else:
    selected = list(resolved.enabled_runs)
print(f"BENCHMARK_ROOT={resolved.benchmark_root}")
print(f"COMPARE_ROOT={resolved.compare_root}")
for run_name in selected:
    print(f"RUN={run_name}")
PY
)

BENCHMARK_ROOT=""
COMPARE_ROOT=""
RUN_NAMES=()
for line in "${META_LINES[@]}"; do
  case "$line" in
    BENCHMARK_ROOT=*) BENCHMARK_ROOT=${line#BENCHMARK_ROOT=} ;;
    COMPARE_ROOT=*) COMPARE_ROOT=${line#COMPARE_ROOT=} ;;
    RUN=*) RUN_NAMES+=("${line#RUN=}") ;;
  esac
done

if [ ${#RUN_NAMES[@]} -eq 0 ]; then
  echo "No Step 6_2 runs selected."
  exit 2
fi

submitted_job_ids=()
for run_name in "${RUN_NAMES[@]}"; do
  run_dir="${BENCHMARK_ROOT}/${run_name}"
  is_complete=1
  for required_path in \
    "metrics/method_metrics.json" \
    "metrics/round_metrics.csv" \
    "metrics/target_row_summary.csv" \
    "metrics/evaluation_results.csv"; do
    if [ ! -f "${run_dir}/${required_path}" ]; then
      is_complete=0
      break
    fi
  done
  if [ "$is_complete" -eq 1 ]; then
    echo "Skipping completed Step 6_2 run: ${run_name}"
    continue
  fi
  safe_run_name=$(echo "$run_name" | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '_')
  jid=$(sbatch \
    --parsable \
    --account=nawimem \
    --job-name "smi_${SIZE_TAG}_6_2_${safe_run_name}" \
    --output "logs/%x_%j.out" \
    --error "logs/%x_%j.err" \
    --time=8-00:00:00 \
    --mem=256G \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --gres=gpu:1 \
    --chdir "$PROJECT_ROOT" \
    --wrap "bash scripts/run_step6_2_nrel.sh \"$MODEL_SIZE\" \"$run_name\"")
  submitted_job_ids+=("$jid")
  echo "Submitted Step 6_2 run ${run_name}: ${jid}"
done

STEP63_ARGS=(
  --parsable
  --account=nawimem
  --job-name "smi_${SIZE_TAG}_6_3"
  --output "logs/%x_%j.out"
  --error "logs/%x_%j.err"
  --time=1-00:00:00
  --mem=128G
  --nodes=1
  --ntasks=1
  --cpus-per-task=16
  --chdir "$PROJECT_ROOT"
)
if [ ${#submitted_job_ids[@]} -gt 0 ]; then
  dependency="afterok:$(IFS=:; echo "${submitted_job_ids[*]}")"
  STEP63_ARGS+=(--dependency "$dependency")
fi

jid63=$(sbatch \
  "${STEP63_ARGS[@]}" \
  --wrap "bash scripts/run_step6_3_nrel.sh \"$MODEL_SIZE\" \"$RUNS\"")

echo "Submitted Step 6_2 + Step 6_3 chain on NREL:"
if [ ${#submitted_job_ids[@]} -gt 0 ]; then
  echo "  Step 6_2 jobs: ${submitted_job_ids[*]}"
else
  echo "  Step 6_2 jobs: none submitted (all selected runs already complete)"
fi
echo "  Step 6_3 job:  ${jid63}"
echo "  Compare root:  ${COMPARE_ROOT}"
