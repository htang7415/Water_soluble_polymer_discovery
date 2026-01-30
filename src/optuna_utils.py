from typing import Dict, Callable

import json
from pathlib import Path

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


def trial_logger(path: str) -> Callable[[optuna.Study, optuna.trial.FrozenTrial], None]:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        payload = {
            "number": trial.number,
            "state": str(trial.state),
            "value": trial.value,
            "params": trial.params,
            "user_attrs": trial.user_attrs,
        }
        with log_path.open("a") as f:
            f.write(json.dumps(payload) + "\n")

    return _callback
