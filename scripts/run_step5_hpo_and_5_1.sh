#!/bin/bash
# One-command local workflow: S0 baseline + Step 5 HPO/refit + Step 5_1 compare.

set -e
MODEL_SIZE=${1:-small}
POLYMER_FAMILY=${2:-}
STUDY_FAMILIES=${3:-S1,S2,S3,S4_rl,S4_ppo,S4_grpo,S4_dpo}
INCLUDE_S0=${INCLUDE_S0:-1}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

if [ "$INCLUDE_S0" = "1" ]; then
  bash scripts/run_step5.sh "$MODEL_SIZE" "$POLYMER_FAMILY" "S0_raw_unconditional"
fi

bash scripts/run_step5_hpo.sh "$MODEL_SIZE" "$POLYMER_FAMILY" "$STUDY_FAMILIES"

COMPARE_RUNS=$(INCLUDE_S0="$INCLUDE_S0" STUDY_FAMILIES="$STUDY_FAMILIES" python - <<'PY'
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(".").resolve()))
from src.step5.study_families import STUDY_BASE_RUNS

runs = []
if os.environ.get("INCLUDE_S0", "1") == "1":
    runs.append("S0_raw_unconditional")
for family in [item.strip() for item in os.environ.get("STUDY_FAMILIES", "").split(",") if item.strip()]:
    base = STUDY_BASE_RUNS[family]
    runs.append(f"{base}_optuna")
print(",".join(runs))
PY
)

echo "Step 5_1 compare runs: ${COMPARE_RUNS}"
bash scripts/run_step5_1.sh "$MODEL_SIZE" "$POLYMER_FAMILY" "$COMPARE_RUNS"

echo "Done!"
