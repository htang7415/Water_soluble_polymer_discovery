from typing import Dict

import optuna


def create_study(name: str, direction: str = "min"):
    pruner = optuna.pruners.MedianPruner()
    return optuna.create_study(study_name=name, direction=direction, pruner=pruner)


def save_study(study: optuna.Study, path: str) -> None:
    df = study.trials_dataframe()
    df.to_csv(path, index=False)


def trial_params_to_json(trial: optuna.Trial) -> Dict:
    return trial.params
