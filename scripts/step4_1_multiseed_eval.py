#!/usr/bin/env python
"""Run Step4_1 across multiple seeds and aggregate miscible-focused diagnostics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, save_config
from src.utils.model_scales import get_results_dir


def _parse_seeds(seed_text: str) -> List[int]:
    seeds: List[int] = []
    for part in str(seed_text).split(","):
        part = part.strip()
        if not part:
            continue
        seeds.append(int(part))
    if not seeds:
        raise ValueError("Expected at least one seed")
    return seeds


def _extract_miscible_metrics(metrics_dir: Path) -> dict:
    class_metrics = pd.read_csv(metrics_dir / "chi_metrics_by_class.csv")
    poly_r2_metrics = pd.read_csv(metrics_dir / "chi_metrics_polymer_r2_distribution.csv")

    class_col = "group" if "group" in class_metrics.columns else "water_miscible"
    miscible_class = class_metrics[
        (class_metrics["split"] == "test")
        & (pd.to_numeric(class_metrics[class_col], errors="coerce") == 1)
    ]
    if miscible_class.empty:
        raise ValueError(f"No miscible test row found in {metrics_dir / 'chi_metrics_by_class.csv'}")

    miscible_poly = poly_r2_metrics[
        (poly_r2_metrics["split"] == "test")
        & (pd.to_numeric(poly_r2_metrics["water_miscible"], errors="coerce") == 1)
    ]
    if miscible_poly.empty:
        raise ValueError(f"No miscible test row found in {metrics_dir / 'chi_metrics_polymer_r2_distribution.csv'}")

    class_row = miscible_class.iloc[0]
    poly_row = miscible_poly.iloc[0]
    return {
        "miscible_test_calib_slope": float(class_row.get("calib_slope", np.nan)),
        "miscible_test_nrmse": float(class_row.get("nrmse", np.nan)),
        "miscible_test_pct_poly_r2_gt_0": float(poly_row.get("pct_poly_r2_gt_0", np.nan)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step4_1 multi-seed evaluation")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Base config path")
    parser.add_argument("--model_size", type=str, default="small", help="Step1 model size")
    parser.add_argument("--split_mode", type=str, default="polymer", choices=["polymer", "random"])
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46", help="Comma-separated random seeds")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results_multiseed/step4_1_multiseed_eval",
        help="Directory for aggregate outputs and temp configs",
    )
    args, passthrough = parser.parse_known_args()

    seeds = _parse_seeds(args.seeds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_cfg_dir = output_dir / "temp_configs"
    temp_cfg_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_config(args.config)
    run_rows = []

    for seed in seeds:
        seed_config = deepcopy(base_config)
        seed_config.setdefault("data", {})
        seed_config["data"]["random_seed"] = int(seed)
        seed_config.setdefault("paths", {})
        seed_results_root = output_dir / f"seed_{seed}"
        seed_config["paths"]["results_dir"] = str(seed_results_root)

        cfg_path = temp_cfg_dir / f"config_seed_{seed}.yaml"
        save_config(seed_config, str(cfg_path))

        cmd = [
            sys.executable,
            "scripts/step4_train_chi_model.py",
            "--config",
            str(cfg_path),
            "--stage",
            "step4_1",
            "--model_size",
            str(args.model_size),
            "--split_mode",
            str(args.split_mode),
        ] + passthrough
        subprocess.run(cmd, check=True)

        results_dir = Path(get_results_dir(args.model_size, seed_config["paths"]["results_dir"], split_mode=None))
        metrics_dir = results_dir / "step4_1_regression" / args.split_mode / "metrics"
        metrics = _extract_miscible_metrics(metrics_dir)
        metrics["seed"] = int(seed)
        metrics["metrics_dir"] = str(metrics_dir)
        run_rows.append(metrics)

    run_df = pd.DataFrame(run_rows).sort_values("seed").reset_index(drop=True)
    run_df.to_csv(output_dir / "step4_1_multiseed_runs.csv", index=False)

    summary_rows = []
    for metric in [
        "miscible_test_calib_slope",
        "miscible_test_nrmse",
        "miscible_test_pct_poly_r2_gt_0",
    ]:
        values = pd.to_numeric(run_df[metric], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        summary_rows.append(
            {
                "metric": metric,
                "mean": float(np.mean(finite)) if finite.size > 0 else np.nan,
                "std": float(np.std(finite)) if finite.size > 0 else np.nan,
                "n_runs": int(finite.size),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "step4_1_multiseed_summary.csv", index=False)
    with open(output_dir / "step4_1_multiseed_summary.json", "w") as f:
        json.dump(
            {
                "config_path": str(Path(args.config).resolve()),
                "model_size": str(args.model_size),
                "split_mode": str(args.split_mode),
                "seeds": [int(seed) for seed in seeds],
                "passthrough_args": passthrough,
                "summary": summary_rows,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
