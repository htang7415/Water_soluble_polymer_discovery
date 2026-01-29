from typing import Dict

import optuna


def create_study(name: str, direction: str = "minimize"):
    pruner = optuna.pruners.MedianPruner()
    normalized = direction
    if direction in {"min", "max"}:
        normalized = "minimize" if direction == "min" else "maximize"
    return optuna.create_study(study_name=name, direction=normalized, pruner=pruner)


def save_study(study: optuna.Study, path: str) -> None:
    df = study.trials_dataframe()
    df.to_csv(path, index=False)


def trial_params_to_json(trial: optuna.Trial) -> Dict:
    return trial.params
