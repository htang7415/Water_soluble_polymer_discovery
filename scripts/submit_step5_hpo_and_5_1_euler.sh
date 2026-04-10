#!/bin/bash
# Submit S0 baseline + per-family Step 5 HPO jobs + dependent Step 5_1 compare on Euler.

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

mkdir -p logs

MODEL_SIZE=${1:-small}
STUDY_FAMILIES=${2:-S1,S2,S3,S4_rl,S4_dpo}
INCLUDE_S0=${INCLUDE_S0:-1}
STEP5_CONFIG=${STEP5_CONFIG:-configs/config5.yaml}
BASE_CONFIG=${BASE_CONFIG:-configs/config.yaml}
FRESH_STUDY=${FRESH_STUDY:-1}
FORCE_RERUN_HPO=${FORCE_RERUN_HPO:-1}

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -x "/srv/home/htang228/anaconda3/bin/conda" ]; then
  eval "$(/srv/home/htang228/anaconda3/bin/conda shell.bash hook)"
else
  echo "Conda not found on the login node. Please load conda and rerun."
  exit 2
fi

conda activate euler_active_learning

SIZE_TAG="m"
case "$MODEL_SIZE" in
  small|s) SIZE_TAG="s" ;;
  medium|m) SIZE_TAG="m" ;;
  large|l) SIZE_TAG="l" ;;
  xl|x) SIZE_TAG="xl" ;;
  *) SIZE_TAG="$MODEL_SIZE" ;;
esac

readarray -t META_LINES < <(MODEL_SIZE="$MODEL_SIZE" STUDY_FAMILIES="$STUDY_FAMILIES" INCLUDE_S0="$INCLUDE_S0" STEP5_CONFIG="$STEP5_CONFIG" BASE_CONFIG="$BASE_CONFIG" python - <<'PY'
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(".").resolve()))
from src.step5.config import load_step5_config
from src.step5.study_families import STUDY_BASE_RUNS

resolved = load_step5_config(
    config_path=os.environ["STEP5_CONFIG"],
    base_config_path=os.environ["BASE_CONFIG"],
    model_size=os.environ.get("MODEL_SIZE") or None,
)
families = [item.strip() for item in os.environ.get("STUDY_FAMILIES", "").split(",") if item.strip()]
unknown = [family for family in families if family not in STUDY_BASE_RUNS]
if unknown:
    raise ValueError(f"Unknown Step 5 HPO study families: {unknown}")
compare_runs = []
if os.environ.get("INCLUDE_S0", "1") == "1":
    compare_runs.append("S0_raw_unconditional")
for family in families:
    compare_runs.append(f"{STUDY_BASE_RUNS[family]}_optuna")
print(f"BENCHMARK_ROOT={resolved.benchmark_root}")
print(f"COMPARE_ROOT={resolved.compare_root}")
print(f"COMPARE_RUNS={','.join(compare_runs)}")
for family in families:
    print(f"FAMILY={family}")
PY
)

BENCHMARK_ROOT=""
COMPARE_ROOT=""
COMPARE_RUNS=""
FAMILIES=()
for line in "${META_LINES[@]}"; do
  case "$line" in
    BENCHMARK_ROOT=*) BENCHMARK_ROOT=${line#BENCHMARK_ROOT=} ;;
    COMPARE_ROOT=*) COMPARE_ROOT=${line#COMPARE_ROOT=} ;;
    COMPARE_RUNS=*) COMPARE_RUNS=${line#COMPARE_RUNS=} ;;
    FAMILY=*) FAMILIES+=("${line#FAMILY=}") ;;
  esac
done

if [ -z "$BENCHMARK_ROOT" ] || [ -z "$COMPARE_ROOT" ] || [ -z "$COMPARE_RUNS" ]; then
  echo "Failed to resolve Step 5/6_3 metadata on the login node."
  exit 2
fi
if [ -n "$STUDY_FAMILIES" ] && [ ${#FAMILIES[@]} -eq 0 ]; then
  echo "Failed to resolve Step 5 HPO study families on the login node."
  exit 2
fi

submitted_job_ids=()

if [ "$INCLUDE_S0" = "1" ]; then
  s0_dir="${BENCHMARK_ROOT}/S0_raw_unconditional"
  s0_complete=1
  for required_path in \
    "metrics/method_metrics.json" \
    "metrics/round_metrics.csv" \
    "metrics/target_row_summary.csv" \
    "metrics/evaluation_results.csv"; do
    if [ ! -f "${s0_dir}/${required_path}" ]; then
      s0_complete=0
      break
    fi
  done
  if [ "$s0_complete" -eq 0 ]; then
    jid_s0=$(sbatch \
      --parsable \
      --job-name "smi_${SIZE_TAG}_5_s0" \
      --output "logs/%x_%j.out" \
      --error "logs/%x_%j.err" \
      --time=8-00:00:00 \
      --mem=256G \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task=16 \
      --partition=pdelab \
      --gres=gpu:1 \
      --chdir "$PROJECT_ROOT" \
      --wrap "bash scripts/run_step5_euler.sh \"$MODEL_SIZE\" \"S0_raw_unconditional\"")
    submitted_job_ids+=("$jid_s0")
    echo "Submitted Step 5 S0 baseline: ${jid_s0}"
  else
    echo "Skipping completed Step 5 S0 baseline"
  fi
fi

for family in "${FAMILIES[@]}"; do
  base_run=$(STUDY_FAMILY="$family" python - <<'PY'
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(".").resolve()))
from src.step5.study_families import STUDY_BASE_RUNS
print(STUDY_BASE_RUNS[os.environ["STUDY_FAMILY"]])
PY
)
  tuned_run="${base_run}_optuna"
  tuned_dir="${BENCHMARK_ROOT}/${tuned_run}"
  tuned_complete=1
  for required_path in \
    "metrics/method_metrics.json" \
    "metrics/round_metrics.csv" \
    "metrics/target_row_summary.csv" \
    "metrics/evaluation_results.csv"; do
    if [ ! -f "${tuned_dir}/${required_path}" ]; then
      tuned_complete=0
      break
    fi
  done
  if [ "$tuned_complete" -eq 1 ] && [ "$FORCE_RERUN_HPO" != "1" ]; then
    echo "Skipping completed HPO refit run: ${tuned_run}"
    continue
  fi
  if [ "$tuned_complete" -eq 1 ] && [ "$FORCE_RERUN_HPO" = "1" ]; then
    echo "Re-running completed HPO refit run: ${tuned_run}"
  fi
  family_tag=$(echo "$family" | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '_')
  jid=$(sbatch \
    --parsable \
    --job-name "smi_${SIZE_TAG}_5_hpo_${family_tag}" \
    --output "logs/%x_%j.out" \
    --error "logs/%x_%j.err" \
    --time=8-00:00:00 \
    --mem=256G \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --partition=pdelab \
    --gres=gpu:1 \
    --chdir "$PROJECT_ROOT" \
    --wrap "FRESH_STUDY=\"$FRESH_STUDY\" bash scripts/run_step5_hpo_euler.sh \"$MODEL_SIZE\" \"$family\"")
  submitted_job_ids+=("$jid")
  echo "Submitted Step 5 HPO family ${family}: ${jid}"
done

STEP51_ARGS=(
  --parsable
  --job-name "smi_${SIZE_TAG}_5_1_optuna"
  --output "logs/%x_%j.out"
  --error "logs/%x_%j.err"
  --time=1-00:00:00
  --mem=128G
  --nodes=1
  --ntasks=1
  --cpus-per-task=16
  --partition=pdelab
  --chdir "$PROJECT_ROOT"
)
if [ ${#submitted_job_ids[@]} -gt 0 ]; then
  dependency="afterany:$(IFS=:; echo "${submitted_job_ids[*]}")"
  STEP51_ARGS+=(--dependency "$dependency")
fi

jid51=$(sbatch \
  "${STEP51_ARGS[@]}" \
  --wrap "ALLOW_PARTIAL=\"1\" bash scripts/run_step5_1_euler.sh \"$MODEL_SIZE\" \"$COMPARE_RUNS\"")

echo "Submitted Step 5 HPO + Step 5_1 chain on Euler:"
if [ ${#submitted_job_ids[@]} -gt 0 ]; then
  echo "  Producer jobs: ${submitted_job_ids[*]}"
else
  echo "  Producer jobs: none submitted (all selected outputs already complete)"
fi
echo "  Step 5_1 job: ${jid51}"
echo "  Compare runs: ${COMPARE_RUNS}"
echo "  Compare root: ${COMPARE_ROOT}"
