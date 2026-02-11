# Water_soluble_polymer_discovery

Euler/NREL production submit chain:
- `Step1+2` in one job (`scripts/submit_steps1_2_euler.sh` or `scripts/submit_steps1_2_nrel.sh`)
- `Step4_1` and `Step4_2` as two parallel jobs (`scripts/submit_step4_only_<cluster>.sh ... step4_1/step4_2`)
- `Step5+6` in one job (`scripts/submit_steps5_6_euler.sh` or `scripts/submit_steps5_6_nrel.sh`)
- Euler wrapper: `bash scripts/submit_all_euler.sh <model_sizes> <split_mode>`
- NREL wrapper: `bash scripts/submit_all_nrel.sh <model_sizes> <split_mode>`

Note: this chain expects Step 3 outputs to already exist and does not submit Step 0/Step 3.
Results namespace: runs are separated by both model size and split mode, e.g. `results_small_polymer` and `results_small_random`.
