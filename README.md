# Water_soluble_polymer_discovery

Physics-guided `χ(T,ϕ)` modeling and inverse design for water-soluble polymer discovery.

## Workflow
- Step 0/1/2 unchanged (data prep, backbone training, sampling).
- Step 3: learn `χ_target` from labeled dataset (`water_soluble`, `χ`).
- Step 4: train physics-guided `χ(T,ϕ)` model with Optuna.
- Step 5: inverse design for water-soluble polymers using Step 3 learned `χ_target`.
- Step 6: polymer-family class + water-soluble inverse design using Step 3 learned `χ_target`.

Dataset: `Data/chi/_50_polymers_T_phi.csv`
- 50 polymers (25 water-soluble, 25 water-insoluble)
- 5 temperatures x 5 `ϕ` values per polymer
- 1250 rows total

## Core model
\[
\chi(T,\phi;p)=\left(a_0+\frac{a_1}{T}+a_2\ln T+a_3T\right)\left(1+b_1(1-\phi)+b_2(1-\phi)^2\right)
\]

## Quick run
```bash
./scripts/run_steps1_6.sh medium polymer 10000 balanced_accuracy 0 1 0.05 0.25 all 0.5 novel
```

## Step-by-step
```bash
./scripts/run_step1.sh medium
./scripts/run_step2.sh 10000 medium
./scripts/run_step3.sh polymer medium balanced_accuracy 0
./scripts/run_step4.sh polymer medium 1
./scripts/run_step5.sh polymer medium 0.05 0.25 novel
./scripts/run_step6.sh polymer medium all 0.05 0.25 0.5 novel
```

## HPC submit wrappers
- Euler: `scripts/submit_all_euler.sh`
- NREL: `scripts/submit_all_nrel.sh`

See `Pipeline.md` and `results.md` for details.
