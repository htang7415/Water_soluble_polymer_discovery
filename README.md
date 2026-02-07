# Water_soluble_polymer_discovery

Physics-guided `χ(T,ϕ)` modeling and inverse design for water-soluble polymer discovery.

## Workflow
- Step 0/1/2 unchanged (data prep, backbone training, sampling).
- Step 3: learn `χ_target` from labeled dataset (`water_soluble`, `χ`).
- Step 4: train physics-guided `χ(T,ϕ)` model with Optuna, then retrain final model with selected best hyperparameters.
- Step 5: inverse design for water-soluble polymers using Step 3 learned `χ_target`.
- Step 6: polymer-family class + water-soluble inverse design using Step 3 learned `χ_target`.

Dataset: `Data/chi/_50_polymers_T_phi.csv`
- 50 polymers (25 water-soluble, 25 water-insoluble)
- 5 temperatures x 5 `ϕ` values per polymer
- 1250 rows total
- Default model size is `small`.

## Core model
\[
\chi(T,\phi;p)=\left(a_0+\frac{a_1}{T}+a_2\ln T+a_3T\right)\left(1+b_1(1-\phi)+b_2(1-\phi)^2\right)
\]

## Quick run
```bash
./scripts/run_steps1_6.sh small
```

## Step-by-step
```bash
./scripts/run_step1.sh small
./scripts/run_step2.sh small
./scripts/run_step3.sh
./scripts/run_step4.sh small
./scripts/run_step5.sh small
./scripts/run_step6.sh small
```

## Output recheck
```bash
python scripts/recheck_pipeline_outputs.py --config configs/config.yaml --model_size small --split_mode polymer
```
This generates step-by-step audit CSVs and figures under `results*/pipeline_recheck/<split_mode>/`.

Run settings note:
- For Step 3/4/5/6, `split_mode` is read from `configs/config.yaml` (`chi_training.split_mode`).
- Step 3 does not use `model_size`.
- Steps 4/5/6 use `model_size` for namespaced outputs/checkpoints.

## Sampling quality controls (Step 2)
To improve `Frac star=2` and validity, Step 2 now uses:
- exact star-budget constraints during reverse diffusion (`target_stars`, default `2`);
- top-k/top-p token filtering (`top_k`, `top_p`);
- post-fix order: paren/ring/bond repair first, then final star-count correction.

Main config keys in `configs/config.yaml`:
- `sampling.temperature`
- `sampling.top_k`
- `sampling.top_p`
- `sampling.target_stars`

## Step 4 Optuna outputs
- Search space is controlled by `chi_training.optuna_search_space` in `configs/config.yaml`.
- Trial records:
  - `step4_chi_training/<split_mode>/tuning/optuna_trials.csv`
  - `step4_chi_training/<split_mode>/tuning/optuna_optimization_chi_r2.csv`
  - `step4_chi_training/<split_mode>/tuning/optuna_optimization_chi_r2.png`
- Final selected training hyperparameters:
  - `step4_chi_training/<split_mode>/metrics/chosen_hyperparameters.json`
  - `step4_chi_training/<split_mode>/metrics/hyperparameter_selection_summary.json`

## HPC submit wrappers
- Euler: `scripts/submit_all_euler.sh`
- NREL: `scripts/submit_all_nrel.sh`

See `Pipeline.md` and `results.md` for details.
