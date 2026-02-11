# Water_soluble_polymer_discovery

NREL production submit chain:
- `Step1+2` in one job (`scripts/submit_steps1_2_nrel.sh`)
- `Step4_1` and `Step4_2` as two parallel jobs (`scripts/submit_step4_only_nrel.sh ... step4_1/step4_2`)
- `Step5+6` in one job (`scripts/submit_steps5_6_nrel.sh`)
- Wrapper: `bash scripts/submit_all_nrel.sh <model_size> [partition] [qos]`

Note: this chain expects Step 3 outputs to already exist and does not submit Step 0/Step 3.
