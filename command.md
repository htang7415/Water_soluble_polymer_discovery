# Commands

Assumes `split_mode=polymer`.

```bash
MODEL_SIZE=small
POLYMER_FAMILY=polyimide
RUNS="S0_raw_unconditional,S2_conditional,S4_dpo"  # optional
```

Valid `POLYMER_FAMILY` values: `polyimide`, `polyamide`, `polyester`, `polyether`, `polyacrylate`, `polystyrene`, `polysulfone`.

## Setup

```bash
pip install -e .
bash scripts/run_step0.sh
```

## Slurm Step 1+2

Submit Step 1 backbone training and Step 2 sampling in the same Slurm job. These commands only need `MODEL_SIZE`.

```bash
# Euler: 1 GPU
sbatch scripts/submit_steps1_2_euler.sh "$MODEL_SIZE"

# NREL: 4 GPUs for Step 1 DDP, then 1 GPU for Step 2 sampling
sbatch scripts/submit_steps1_2_nrel.sh "$MODEL_SIZE"

# NREL optional: pass partition/qos at submit time if needed
sbatch --partition=<gpu_partition> --qos=<qos> scripts/submit_steps1_2_nrel.sh "$MODEL_SIZE"
```

After Step 1+2 completes, run Step 3 once if it is not already done:

```bash
bash scripts/run_step3.sh
```

## Slurm Step 4

Euler submits Step4_1, Step4_2, and Step4_3 together, then Step4_4 after all three succeed.

```bash
# Euler
bash scripts/submit_step4_euler.sh "$MODEL_SIZE"

# NREL
bash scripts/submit_step4_only_nrel.sh "$MODEL_SIZE"
```

## Slurm Step 5: No HPO

Run only after Step 4 succeeds.

```bash
# Euler
bash scripts/submit_step5_and_5_1_euler.sh "$MODEL_SIZE" "$POLYMER_FAMILY"
bash scripts/submit_step5_and_5_1_euler.sh "$MODEL_SIZE" "$POLYMER_FAMILY" "$RUNS"

# NREL
bash scripts/submit_step5_and_5_1_nrel.sh "$MODEL_SIZE" "$POLYMER_FAMILY"
bash scripts/submit_step5_and_5_1_nrel.sh "$MODEL_SIZE" "$POLYMER_FAMILY" "$RUNS"
```

## Local

```bash
bash scripts/run_step4.sh "$MODEL_SIZE"
bash scripts/run_step5.sh "$MODEL_SIZE" "$POLYMER_FAMILY"
bash scripts/run_step5.sh "$MODEL_SIZE" "$POLYMER_FAMILY" "$RUNS"
bash scripts/run_step5_1.sh "$MODEL_SIZE" "$POLYMER_FAMILY"
bash scripts/run_step5_1.sh "$MODEL_SIZE" "$POLYMER_FAMILY" "$RUNS"
```

## Step 5 HPO

Defaults to the fixed `S0` baseline plus HPO study families: `S1`, `S2`, `S3`, `S4_rl`, `S4_ppo`, `S4_grpo`, `S4_dpo`.

```bash
bash scripts/run_step5_hpo_and_5_1.sh "$MODEL_SIZE" "$POLYMER_FAMILY"
bash scripts/submit_step5_hpo_and_5_1_euler.sh "$MODEL_SIZE" "$POLYMER_FAMILY"
bash scripts/submit_step5_hpo_and_5_1_nrel.sh "$MODEL_SIZE" "$POLYMER_FAMILY"
```

## Smoke

```bash
bash scripts/run_step5_smoke.sh "$MODEL_SIZE"
bash scripts/run_step5_smoke.sh "$MODEL_SIZE" "S4_dpo" "__smoke_dpo"
bash scripts/test_step5_smoke.sh "$MODEL_SIZE"
```

## Traditional Baselines

```bash
python traditional_step4/scripts/step4_4_compare_dit_vs_traditional.py \
  --config traditional_step4/configs/config_traditional.yaml \
  --split_mode polymer \
  --model_sizes "$MODEL_SIZE" \
  --dit_config configs/config.yaml

sbatch scripts/submit_step4_3_euler.sh polymer
sbatch scripts/submit_step4_4_euler.sh polymer traditional_step4/configs/config_traditional.yaml "" "$MODEL_SIZE"
sbatch scripts/submit_step4_3_nrel.sh polymer
sbatch scripts/submit_step4_4_nrel.sh polymer
```

## Monitor

```bash
squeue -u "$USER"
```
