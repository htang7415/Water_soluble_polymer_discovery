#!/usr/bin/env python
"""Run Step 6_2 Optuna HPO studies."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.step6_2.config import load_step6_2_config
from src.step6_2.hpo import STUDY_BASE_RUNS, run_optuna_study


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step 6_2 Optuna HPO.")
    parser.add_argument("--config", default="configs/config6_2.yaml")
    parser.add_argument("--base_config", default="configs/config.yaml")
    parser.add_argument("--model_size", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--study_families",
        default="S1,S2,S3,S4_rl,S4_dpo",
        help="Comma-separated study families to run.",
    )
    parser.add_argument("--skip_refit", action="store_true", help="Skip the best-trial full-budget refit.")
    parser.add_argument(
        "--force_enable",
        action="store_true",
        help="Force-enable step6_2_hpo for this invocation without editing configs/config6_2.yaml.",
    )
    parser.add_argument(
        "--fresh_study",
        action="store_true",
        help="Delete existing per-family Step 6_2 HPO study artifacts before rerunning.",
    )
    args = parser.parse_args()

    resolved = load_step6_2_config(
        config_path=args.config,
        base_config_path=args.base_config,
        model_size=args.model_size,
        force_hpo_enabled=bool(args.force_enable),
    )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    requested = [item.strip() for item in str(args.study_families).split(",") if item.strip()]
    unknown = [item for item in requested if item not in STUDY_BASE_RUNS]
    if unknown:
        raise ValueError(f"Unknown Step 6_2 study families: {unknown}")

    for study_family in requested:
        result = run_optuna_study(
            resolved=resolved,
            study_family=study_family,
            config_path=args.config,
            base_config_path=args.base_config,
            model_size=args.model_size,
            device=device,
            refit_best=not args.skip_refit,
            fresh_study=bool(args.fresh_study),
        )
        print(
            f"[step6_2_hpo] {study_family} best_trial={int(result['best_trial'].number)} "
            f"best_value={float(result['best_trial'].value):.6f}"
        )


if __name__ == "__main__":
    main()
