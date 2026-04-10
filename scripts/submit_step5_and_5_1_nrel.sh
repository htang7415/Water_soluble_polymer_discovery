#!/bin/bash
# Submit Step 5 runs plus dependent Step 5_1 comparison on NREL.
# Usage:
#   bash scripts/submit_step5_and_5_1_nrel.sh <model_size> <polymer_family> [runs_csv]
# Optional env:
#   STEP5_DEPENDENCY=afterok:<jobid> to wait for upstream Step 4.

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

mkdir -p logs

MODEL_SIZE=${1:-small}
POLYMER_FAMILY=${2:-}
RUNS=${3:-}
STEP5_CONFIG=${STEP5_CONFIG:-configs/config5.yaml}
BASE_CONFIG=${BASE_CONFIG:-configs/config.yaml}
STEP5_DEPENDENCY=${STEP5_DEPENDENCY:-}

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -x "/home/htang/anaconda3/bin/conda" ]; then
  eval "$(/home/htang/anaconda3/bin/conda shell.bash hook)"
else
  echo "Conda not found on the login node. Please load conda and rerun."
  exit 2
fi

conda activate kl_active_learning

SIZE_TAG="m"
case "$MODEL_SIZE" in
  small|s) SIZE_TAG="s" ;;
  medium|m) SIZE_TAG="m" ;;
  large|l) SIZE_TAG="l" ;;
  xl|x) SIZE_TAG="xl" ;;
  *) SIZE_TAG="$MODEL_SIZE" ;;
esac

readarray -t META_LINES < <(MODEL_SIZE="$MODEL_SIZE" POLYMER_FAMILY="$POLYMER_FAMILY" RUNS="$RUNS" STEP5_CONFIG="$STEP5_CONFIG" BASE_CONFIG="$BASE_CONFIG" python - <<'PY'
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(".").resolve()))
from src.step5.config import load_step5_config

resolved = load_step5_config(
    config_path=os.environ["STEP5_CONFIG"],
    base_config_path=os.environ["BASE_CONFIG"],
    model_size=os.environ.get("MODEL_SIZE") or None,
    c_target_override=os.environ.get("POLYMER_FAMILY") or None,
)
runs_csv = os.environ.get("RUNS", "").strip()
if runs_csv:
    requested = [item.strip() for item in runs_csv.split(",") if item.strip()]
    unknown = [run for run in requested if run not in resolved.enabled_runs]
    if unknown:
        raise ValueError(f"Requested runs are not enabled in config5: {unknown}")
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
  echo "No Step 5 runs selected."
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
    echo "Skipping completed Step 5 run: ${run_name}"
    continue
  fi
  safe_run_name=$(echo "$run_name" | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '_')
  STEP5_JOB_ARGS=(
    --parsable
    --account=nawimem
    --job-name "smi_${SIZE_TAG}_5_${safe_run_name}"
    --output "logs/%x_%j.out"
    --error "logs/%x_%j.err"
    --time=8-00:00:00
    --mem=256G
    --nodes=1
    --ntasks=1
    --cpus-per-task=16
    --gres=gpu:1
    --chdir "$PROJECT_ROOT"
  )
  if [ -n "$STEP5_DEPENDENCY" ]; then
    STEP5_JOB_ARGS+=(--dependency "$STEP5_DEPENDENCY")
  fi
  jid=$(sbatch \
    "${STEP5_JOB_ARGS[@]}" \
    --wrap "bash scripts/run_step5_nrel.sh \"$MODEL_SIZE\" \"$POLYMER_FAMILY\" \"$run_name\"")
  submitted_job_ids+=("$jid")
  echo "Submitted Step 5 run ${run_name}: ${jid}"
done

STEP51_ARGS=(
  --parsable
  --account=nawimem
  --job-name "smi_${SIZE_TAG}_5_1"
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
  STEP51_ARGS+=(--dependency "$dependency")
elif [ -n "$STEP5_DEPENDENCY" ]; then
  STEP51_ARGS+=(--dependency "$STEP5_DEPENDENCY")
fi

jid51=$(sbatch \
  "${STEP51_ARGS[@]}" \
  --wrap "bash scripts/run_step5_1_nrel.sh \"$MODEL_SIZE\" \"$POLYMER_FAMILY\" \"$RUNS\"")

echo "Submitted Step 5 + Step 5_1 chain on NREL:"
if [ ${#submitted_job_ids[@]} -gt 0 ]; then
  echo "  Step 5 jobs: ${submitted_job_ids[*]}"
else
  echo "  Step 5 jobs: none submitted (all selected runs already complete)"
fi
echo "  Step 5_1 job:  ${jid51}"
echo "  Polymer family: ${POLYMER_FAMILY:-config5 default}"
echo "  Compare root:  ${COMPARE_ROOT}"
