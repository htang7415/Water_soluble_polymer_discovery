#!/usr/bin/env python
"""Run Step4_3_1 across multiple seeds and aggregate miscible-focused diagnostics."""

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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.config import save_config  # noqa: E402
from traditional_step4.src.common import get_traditional_results_dir, load_traditional_config  # noqa: E402


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


def _resolve_recommended_model_metrics_dir(stage_metrics_dir: Path) -> tuple[str, Path]:
    leaderboard_path = stage_metrics_dir / "model_leaderboard.csv"
    if not leaderboard_path.exists():
        raise FileNotFoundError(f"Missing leaderboard file: {leaderboard_path}")

    leaderboard_df = pd.read_csv(leaderboard_path)
    if leaderboard_df.empty:
        raise ValueError(f"Leaderboard is empty: {leaderboard_path}")

    if "rank" in leaderboard_df.columns:
        ranked = leaderboard_df.sort_values("rank", ascending=True).reset_index(drop=True)
        top_row = ranked.iloc[0]
    else:
        top_row = leaderboard_df.iloc[0]

    model_name = str(top_row.get("model_name", "")).strip()
    metrics_dir_raw = str(top_row.get("metrics_dir", "")).strip()
    if not model_name:
        raise ValueError(f"Leaderboard row missing model_name: {leaderboard_path}")
    if not metrics_dir_raw:
        raise ValueError(f"Leaderboard row missing metrics_dir: {leaderboard_path}")
    return model_name, Path(metrics_dir_raw)


def _extract_miscible_metrics(metrics_dir: Path) -> dict:
    class_metrics_path = metrics_dir / "chi_metrics_by_class.csv"
    poly_r2_metrics_path = metrics_dir / "chi_metrics_polymer_r2_distribution.csv"
    if not class_metrics_path.exists():
        raise FileNotFoundError(f"Missing required metrics file: {class_metrics_path}")
    if not poly_r2_metrics_path.exists():
        raise FileNotFoundError(
            "Missing required metrics file: "
            f"{poly_r2_metrics_path}. Re-run Step4_3_1 with the updated script so the new comparison metrics are generated."
        )

    class_metrics = pd.read_csv(class_metrics_path)
    poly_r2_metrics = pd.read_csv(poly_r2_metrics_path)

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
    parser = argparse.ArgumentParser(description="Run Step4_3_1 multi-seed evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="traditional_step4/configs/config_traditional.yaml",
        help="Base traditional config path",
    )
    parser.add_argument("--split_mode", type=str, default="polymer", choices=["polymer", "random"])
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46", help="Comma-separated random seeds")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="traditional_step4/results_multiseed/step4_3_1_multiseed_eval",
        help="Directory for aggregate outputs and temp configs",
    )
    args, passthrough = parser.parse_known_args()

    seeds = _parse_seeds(args.seeds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_cfg_dir = output_dir / "temp_configs"
    temp_cfg_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_traditional_config(args.config)
    run_rows = []

    for seed in seeds:
        seed_config = deepcopy(base_config)
        seed_config.setdefault("data", {})
        seed_config["data"]["random_seed"] = int(seed)
        seed_config.setdefault("paths", {})
        seed_results_root = output_dir / f"seed_{seed}"
        seed_config["paths"]["results_root"] = str(seed_results_root)
        seed_config.setdefault("traditional_step4", {})
        trad_cfg = seed_config["traditional_step4"]
        if not isinstance(trad_cfg, dict):
            raise ValueError("traditional_step4 config section must be a mapping")
        trad_cfg.setdefault("shared", {})
        if not isinstance(trad_cfg["shared"], dict):
            raise ValueError("traditional_step4.shared must be a mapping")
        trad_cfg["shared"]["split_mode"] = str(args.split_mode)

        cfg_path = temp_cfg_dir / f"config_seed_{seed}.yaml"
        save_config(seed_config, str(cfg_path))

        cmd = [
            sys.executable,
            "traditional_step4/scripts/step4_3_train_traditional.py",
            "--config",
            str(cfg_path),
            "--stage",
            "step4_3_1",
            "--split_mode",
            str(args.split_mode),
        ] + passthrough
        subprocess.run(cmd, check=True)

        results_dir = get_traditional_results_dir(results_root=seed_config["paths"]["results_root"], split_mode=None)
        stage_dir = results_dir / "step4_3_traditional" / "step4_3_1_regression" / args.split_mode
        stage_metrics_dir = stage_dir / "metrics"
        best_model_name, model_metrics_dir = _resolve_recommended_model_metrics_dir(stage_metrics_dir)
        metrics = _extract_miscible_metrics(model_metrics_dir)
        metrics["seed"] = int(seed)
        metrics["selected_model_name"] = best_model_name
        metrics["stage_metrics_dir"] = str(stage_metrics_dir)
        metrics["model_metrics_dir"] = str(model_metrics_dir)
        run_rows.append(metrics)

    run_df = pd.DataFrame(run_rows).sort_values("seed").reset_index(drop=True)
    run_df.to_csv(output_dir / "step4_3_1_multiseed_runs.csv", index=False)

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
    summary_df.to_csv(output_dir / "step4_3_1_multiseed_summary.csv", index=False)
    with open(output_dir / "step4_3_1_multiseed_summary.json", "w") as f:
        json.dump(
            {
                "config_path": str(Path(args.config).resolve()),
                "split_mode": str(args.split_mode),
                "seeds": [int(seed) for seed in seeds],
                "passthrough_args": passthrough,
                "selection_rule": "use step4_3_1 model_leaderboard rank 1 per seed",
                "summary": summary_rows,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
