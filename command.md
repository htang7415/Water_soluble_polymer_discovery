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

## Run Step 5-8 in terminal
```bash
# reuses existing Step 1-4 outputs for the same MODEL_SIZE
bash scripts/run_steps5_8.sh "$MODEL_SIZE"
```

## Run Step 6_2 and Step 6_3 in terminal
```bash
# development / non-HPO path
# Step 6_2 benchmark for the target family configured in configs/config6_2.yaml
bash scripts/run_step6_2.sh "$MODEL_SIZE"

# optional: restrict Step 6_2 to a subset of enabled runs
bash scripts/run_step6_2.sh "$MODEL_SIZE" "S0_raw_unconditional,S1_guided_frozen,S2_conditional"

# Step 6_3 comparison across the completed Step 6_2 runs
bash scripts/run_step6_3.sh "$MODEL_SIZE"

# optional: compare only a subset of runs
bash scripts/run_step6_3.sh "$MODEL_SIZE" "S2_conditional,S3_conditional_guided,S4_rl_finetuned"
```

## Default: one command for Step 6_2 HPO + best refits + Step 6_3
```bash
# default local workflow:
# 1. run fixed S0 baseline
# 2. run Optuna HPO for S1/S2/S3/S4_rl/S4_dpo
# 3. refit each family with its best hyperparameters
# 4. compare S0 + tuned _optuna runs in Step 6_3
bash scripts/run_step6_2_hpo_and_6_3.sh "$MODEL_SIZE"

# optional: restrict HPO to a subset of study families
bash scripts/run_step6_2_hpo_and_6_3.sh "$MODEL_SIZE" "S2,S3,S4_rl"
```

## Run Steps 5-8 separately in terminal
```bash
# reuses existing Step 1-4 outputs for the same MODEL_SIZE
bash scripts/run_step5.sh "$MODEL_SIZE"
bash scripts/run_step6.sh "$MODEL_SIZE"
bash scripts/run_step7.sh "$MODEL_SIZE" polymer
bash scripts/run_step8.sh "$MODEL_SIZE"
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
# runs Steps 1-8 locally; Step 3 is included in this wrapper
bash scripts/run_steps1_8.sh "$MODEL_SIZE"
python traditional_step4/scripts/step4_4_compare_dit_vs_traditional.py \
  --config traditional_step4/configs/config_traditional.yaml \
  --split_mode polymer \
  --model_sizes "$MODEL_SIZE"
```

## Cluster submit (Euler, full chain)
```bash
# submits Steps 1,2,4,5,6,7,8
# prerequisites: run Step 0 once and precompute Step 3 targets locally
bash scripts/submit_all_euler.sh "$MODEL_SIZE"

# traditional baselines
sbatch scripts/submit_step4_3_euler.sh polymer
sbatch scripts/submit_step4_4_euler.sh polymer
```

## Cluster submit (NREL, full chain)
```bash
# submits Steps 1,2,4,5,6,7,8
# prerequisites: run Step 0 once and precompute Step 3 targets locally
bash scripts/submit_all_nrel.sh "$MODEL_SIZE"

# traditional baselines
sbatch scripts/submit_step4_3_nrel.sh polymer
sbatch scripts/submit_step4_4_nrel.sh polymer
```

## Cluster submit (Steps 5-8 only)
```bash
# Euler
bash scripts/submit_steps5_8_euler.sh "$MODEL_SIZE"

# NREL
bash scripts/submit_steps5_8_nrel.sh "$MODEL_SIZE"
```

## Cluster submit (Step 6_2 + Step 6_3)
```bash
# Step 6_2 / 6_3 uses configs/config6_2.yaml for c_target and enabled runs

# Euler: submit one GPU Step 6_2 job per selected run, then a dependent Step 6_3 compare job
bash scripts/submit_step6_2_and_6_3_euler.sh "$MODEL_SIZE"

# optional: restrict to a subset of enabled runs
bash scripts/submit_step6_2_and_6_3_euler.sh "$MODEL_SIZE" "S2_conditional,S3_conditional_guided,S4_rl_finetuned"

# NREL
bash scripts/submit_step6_2_and_6_3_nrel.sh "$MODEL_SIZE"

# optional: restrict to a subset of enabled runs
bash scripts/submit_step6_2_and_6_3_nrel.sh "$MODEL_SIZE" "S2_conditional,S3_conditional_guided,S4_rl_finetuned"
```

## Cluster submit (one command: Step 6_2 HPO + best refits + Step 6_3)
```bash
# Euler: fixed S0 baseline + one GPU HPO job per study family + dependent Step 6_3 compare
bash scripts/submit_step6_2_hpo_and_6_3_euler.sh "$MODEL_SIZE"

# optional: restrict HPO to a subset of study families
bash scripts/submit_step6_2_hpo_and_6_3_euler.sh "$MODEL_SIZE" "S2,S3,S4_rl"

# NREL
bash scripts/submit_step6_2_hpo_and_6_3_nrel.sh "$MODEL_SIZE"

# optional: restrict HPO to a subset of study families
bash scripts/submit_step6_2_hpo_and_6_3_nrel.sh "$MODEL_SIZE" "S2,S3,S4_rl"
```

## Cluster run wrappers (Step 6_2 / Step 6_3 only)
```bash
# Euler
bash scripts/run_step6_2_euler.sh "$MODEL_SIZE"
bash scripts/run_step6_3_euler.sh "$MODEL_SIZE"
bash scripts/run_step6_2_hpo_euler.sh "$MODEL_SIZE"

# NREL
bash scripts/run_step6_2_nrel.sh "$MODEL_SIZE"
bash scripts/run_step6_3_nrel.sh "$MODEL_SIZE"
bash scripts/run_step6_2_hpo_nrel.sh "$MODEL_SIZE"
```

## Run Step 8 only
```bash
# local direct terminal command
python scripts/step8_build_paper_package.py --config configs/config.yaml --model_size "$MODEL_SIZE" --split_mode polymer

# NREL (activates kl_active_learning + safe cache dirs)
bash scripts/run_step8_nrel.sh "$MODEL_SIZE"

# Euler (activates euler_active_learning + safe cache dirs)
bash scripts/run_step8_euler.sh "$MODEL_SIZE"
```

## Monitor
```bash
squeue -u "$USER"
```
