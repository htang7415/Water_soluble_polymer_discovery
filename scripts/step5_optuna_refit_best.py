#!/usr/bin/env python
"""Refit Step 5 runs from previously saved Optuna best_params.yaml artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.step5.config import load_step5_config
from src.step5.hpo import STUDY_BASE_RUNS, refit_best_trial


def main() -> None:
    parser = argparse.ArgumentParser(description="Refit Step 5 best Optuna trials.")
    parser.add_argument("--config", default="configs/config5.yaml")
    parser.add_argument("--base_config", default="configs/config.yaml")
    parser.add_argument("--model_size", default=None)
    parser.add_argument(
        "--c_target",
        "--polymer_family",
        dest="c_target",
        default=None,
        help="Override step5.c_target polymer-family target.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--study_families",
        default="S1,S2,S3,S4_rl,S4_ppo,S4_grpo,S4_dpo",
        help="Comma-separated study families to refit.",
    )
    parser.add_argument(
        "--force_enable",
        action="store_true",
        help="Force-enable step5_hpo for this invocation without editing configs/config5.yaml.",
    )
    parser.add_argument(
        "--fresh_refit",
        action="store_true",
        help="Delete existing tuned benchmark outputs before rerunning the refit.",
    )
    args = parser.parse_args()

    resolved = load_step5_config(
        config_path=args.config,
        base_config_path=args.base_config,
        model_size=args.model_size,
        c_target_override=args.c_target,
        force_hpo_enabled=bool(args.force_enable),
    )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    requested = [item.strip() for item in str(args.study_families).split(",") if item.strip()]
    unknown = [item for item in requested if item not in STUDY_BASE_RUNS]
    if unknown:
        raise ValueError(f"Unknown Step 5 study families: {unknown}")

    for study_family in requested:
        result = refit_best_trial(
            resolved=resolved,
            study_family=study_family,
            config_path=args.config,
            base_config_path=args.base_config,
            model_size=args.model_size,
            device=device,
            fresh_refit=bool(args.fresh_refit),
        )
        if result is None:
            print(f"[step5_hpo_refit] {study_family} skipped: no best_params.yaml with refittable params")
            continue
        print(
            f"[step5_hpo_refit] {study_family} refit_complete "
            f"best_params={result['best_params_path']}"
        )


if __name__ == "__main__":
    main()
