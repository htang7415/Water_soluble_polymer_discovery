# Commands (Polymer Workflow)

All commands below assume `split_mode=polymer`.

## Defaults
```bash
MODEL_SIZE=small
```

## One-time setup
```bash
pip install -e .
bash scripts/run_step0.sh
```

## Precompute Step 3 targets for cluster workflows
```bash
# required before submit_all_* because the cluster full-chain scripts do not run Step 3
bash scripts/run_step3.sh
```

## Run Step 5 and Step 5_1 in terminal
```bash
# development / non-HPO path
# Step 5 benchmark for the target family configured in configs/config5.yaml
bash scripts/run_step5.sh "$MODEL_SIZE"

# optional: restrict Step 5 to a subset of enabled runs
bash scripts/run_step5.sh "$MODEL_SIZE" "S0_raw_unconditional,S1_guided_frozen,S2_conditional"

# Step 5_1 comparison across the completed Step 5 runs
bash scripts/run_step5_1.sh "$MODEL_SIZE"

# optional: compare only a subset of runs
bash scripts/run_step5_1.sh "$MODEL_SIZE" "S2_conditional,S3_conditional_guided,S4_rl_finetuned"
```

## Run Step 5 smoke tests
```bash
# isolated Step 5 smoke run using configs/config5_smoke.yaml
# defaults: 1 target row, 1 sample, 1 round, short S2/RL/DPO schedules
bash scripts/run_step5_smoke.sh "$MODEL_SIZE"

# optional: choose a different run subset and output suffix
bash scripts/run_step5_smoke.sh "$MODEL_SIZE" "S4_dpo" "__smoke_dpo"

# optional: override smoke settings via env vars
STEP5_SMOKE_S2_MAX_STEPS=8 STEP5_SMOKE_MAX_TARGET_ROWS=2 \
  bash scripts/run_step5_smoke.sh "$MODEL_SIZE" "S2_conditional,S4_rl_finetuned" "__smoke_custom"
```

## Run Step 5 smoke suite
```bash
# runs S2_conditional, S4_dpo, and S4_rl_finetuned back-to-back
# then checks their key artifacts and schedule overrides
bash scripts/test_step5_smoke.sh "$MODEL_SIZE"

# optional: use a different isolated suffix root
bash scripts/test_step5_smoke.sh "$MODEL_SIZE" "__smoke_suite_alt"

# optional: run and validate only one case from the suite
bash scripts/test_step5_smoke.sh "$MODEL_SIZE" "__smoke_suite_rl" "S4_rl_finetuned"
```

## Default: one command for Step 5 HPO + best refits + Step 5_1
```bash
# default local workflow:
# 1. run fixed S0 baseline
# 2. run Optuna HPO for S1/S2/S3/S4_rl/S4_dpo
# 3. refit each family with its best hyperparameters
# 4. compare S0 + tuned _optuna runs in Step 5_1
bash scripts/run_step5_hpo_and_5_1.sh "$MODEL_SIZE"

# optional: restrict HPO to a subset of study families
bash scripts/run_step5_hpo_and_5_1.sh "$MODEL_SIZE" "S2,S3,S4_rl"
```

## Run Step 5 separately in terminal
```bash
# reuses existing Step 1-4 outputs for the same MODEL_SIZE
bash scripts/run_step5.sh "$MODEL_SIZE"
```

## Run Step 4_4 (polymer)
```bash
# default: compare DiT Step 4 vs traditional Step 4_3 for polymer / small
python traditional_step4/scripts/step4_4_compare_dit_vs_traditional.py \
  --config traditional_step4/configs/config_traditional.yaml \
  --split_mode polymer \
  --model_sizes small

# optional: restrict comparison to specific model sizes
python traditional_step4/scripts/step4_4_compare_dit_vs_traditional.py \
  --config traditional_step4/configs/config_traditional.yaml \
  --split_mode polymer \
  --model_sizes small medium large xl

# optional: override the DiT config path used to locate DiT results
python traditional_step4/scripts/step4_4_compare_dit_vs_traditional.py \
  --config traditional_step4/configs/config_traditional.yaml \
  --split_mode polymer \
  --model_sizes small \
  --dit_config configs/config.yaml
```

## Fast local end-to-end
```bash
# runs Steps 1-4 locally; Step 3 is included in this wrapper
bash scripts/run_steps1_4.sh "$MODEL_SIZE"
python traditional_step4/scripts/step4_4_compare_dit_vs_traditional.py \
  --config traditional_step4/configs/config_traditional.yaml \
  --split_mode polymer \
  --model_sizes "$MODEL_SIZE"
```

## Cluster submit (Euler, full chain)
```bash
# submits Steps 1,2,4,5 and then Step 5_1 comparison
# prerequisites: run Step 0 once and precompute Step 3 targets locally
bash scripts/submit_all_euler.sh "$MODEL_SIZE"

# traditional baselines
sbatch scripts/submit_step4_3_euler.sh polymer
sbatch scripts/submit_step4_4_euler.sh polymer
```

## Cluster submit (NREL, full chain)
```bash
# submits Steps 1,2,4,5 and then Step 5_1 comparison
# prerequisites: run Step 0 once and precompute Step 3 targets locally
bash scripts/submit_all_nrel.sh "$MODEL_SIZE"

# traditional baselines
sbatch scripts/submit_step4_3_nrel.sh polymer
sbatch scripts/submit_step4_4_nrel.sh polymer
```

## Cluster submit (Step 5 + Step 5_1)
```bash
# Step 5 / 5_1 uses configs/config5.yaml for c_target and enabled runs

# Euler: submit one GPU Step 5 job per selected run, then a dependent Step 5_1 compare job
bash scripts/submit_step5_and_5_1_euler.sh "$MODEL_SIZE"

# optional: restrict to a subset of enabled runs
bash scripts/submit_step5_and_5_1_euler.sh "$MODEL_SIZE" "S2_conditional,S3_conditional_guided,S4_rl_finetuned"

# NREL
bash scripts/submit_step5_and_5_1_nrel.sh "$MODEL_SIZE"

# optional: restrict to a subset of enabled runs
bash scripts/submit_step5_and_5_1_nrel.sh "$MODEL_SIZE" "S2_conditional,S3_conditional_guided,S4_rl_finetuned"
```

## Cluster submit (one command: Step 5 HPO + best refits + Step 5_1)
```bash
# Euler: fixed S0 baseline + one GPU HPO job per study family + dependent Step 5_1 compare
bash scripts/submit_step5_hpo_and_5_1_euler.sh "$MODEL_SIZE"

# optional: restrict HPO to a subset of study families
bash scripts/submit_step5_hpo_and_5_1_euler.sh "$MODEL_SIZE" "S2,S3,S4_rl"

# NREL
bash scripts/submit_step5_hpo_and_5_1_nrel.sh "$MODEL_SIZE"

# optional: restrict HPO to a subset of study families
bash scripts/submit_step5_hpo_and_5_1_nrel.sh "$MODEL_SIZE" "S2,S3,S4_rl"
```

## Cluster run wrappers (Step 5 / Step 5_1 only)
```bash
# Euler
bash scripts/run_step5_euler.sh "$MODEL_SIZE"
bash scripts/run_step5_1_euler.sh "$MODEL_SIZE"
bash scripts/run_step5_hpo_euler.sh "$MODEL_SIZE"

# NREL
bash scripts/run_step5_nrel.sh "$MODEL_SIZE"
bash scripts/run_step5_1_nrel.sh "$MODEL_SIZE"
bash scripts/run_step5_hpo_nrel.sh "$MODEL_SIZE"
```

## Monitor
```bash
squeue -u "$USER"
```
