#!/usr/bin/env python3
"""Compare Step 5 HPO running-best metrics across methods."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.step5.config import load_step5_config
from src.step5.plotting import apply_step5_plot_style

METHOD_LABELS = {
    "S1": "S1: conditional sampling",
    "S2": "S2: conditional diffusion",
    "S3": "S3: conditional guided",
    "S4_dpo": "S4: rl-DPO",
    "S4_grpo": "S4: rl-GRPO",
    "S4_ppo": "S4: rl-PPO",
    "S4_rl": "S4: rl-finetuned",
}


def _load_trials_df(study_root: Path) -> pd.DataFrame:
    db_path = study_root / "optuna.db"
    if not db_path.exists():
        raise FileNotFoundError(f"Missing Optuna database: {db_path}")
    query = """
    WITH attr AS (
        SELECT
            trial_id,
            MAX(CASE WHEN "key" = 'mean_property_success_hit_rate_reporting' THEN value_json END)
                AS mean_property_success_hit_rate,
            MAX(CASE WHEN "key" = 'mean_success_hit_rate_reporting' THEN value_json END)
                AS mean_success_hit_rate,
            MAX(CASE WHEN "key" = 'mean_success_hit_rate_discovery' THEN value_json END)
                AS mean_success_hit_rate_discovery
        FROM trial_user_attributes
        GROUP BY trial_id
    )
    SELECT
        t.number AS trial_number,
        t.state AS state,
        attr.mean_property_success_hit_rate,
        attr.mean_success_hit_rate,
        attr.mean_success_hit_rate_discovery
    FROM trials t
    LEFT JOIN attr ON attr.trial_id = t.trial_id
    ORDER BY t.number
    """
    with sqlite3.connect(db_path) as conn:
        trials_df = pd.read_sql_query(query, conn)
    return trials_df


def _running_best(values: Iterable[float]) -> list[float]:
    best_so_far = float("nan")
    running_best: list[float] = []
    for value in values:
        if np.isfinite(value):
            if not np.isfinite(best_so_far):
                best_so_far = float(value)
            else:
                best_so_far = max(float(best_so_far), float(value))
        running_best.append(float(best_so_far) if np.isfinite(best_so_far) else float("nan"))
    return running_best


def _build_plot_frame(study_root: Path, metric_col: str) -> pd.DataFrame:
    trials_df = _load_trials_df(study_root)
    frame = trials_df.loc[:, ["trial_number", metric_col]].copy()
    frame["trial_number"] = pd.to_numeric(frame["trial_number"], errors="coerce")
    frame[metric_col] = pd.to_numeric(frame[metric_col], errors="coerce")
    frame = frame.loc[frame["trial_number"].notna()].sort_values("trial_number", kind="mergesort").reset_index(drop=True)
    frame["running_best"] = _running_best(frame[metric_col].to_numpy(dtype=float))
    return frame


def _plot_metric(study_root: Path, metric_col: str, ylabel: str, output_name: str, *, font_size: int, dpi: int) -> None:
    apply_step5_plot_style(font_size=font_size, dpi=dpi)
    fig, ax = plt.subplots(figsize=(8, 6))

    plotted_any = False
    ymax = 0.0
    for method_dir in sorted(path for path in study_root.iterdir() if path.is_dir()):
        try:
            frame = _build_plot_frame(method_dir, metric_col=metric_col)
        except Exception:
            continue
        if frame.empty or not np.isfinite(frame["running_best"].to_numpy(dtype=float)).any():
            continue
        plotted_any = True
        ymax = max(ymax, float(np.nanmax(frame["running_best"].to_numpy(dtype=float))))
        ax.plot(
            frame["trial_number"].astype(int),
            frame["running_best"].astype(float),
            linewidth=2.0,
            marker="o",
            markersize=4.0,
            label=METHOD_LABELS.get(method_dir.name, method_dir.name),
        )

    if plotted_any:
        ax.set_ylim(0.0, max(1.0, min(1.02, ymax * 1.05 if ymax > 0.0 else 1.0)))
        ax.legend(frameon=False, loc="lower right")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    else:
        ax.text(0.5, 0.5, "No completed trials", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylim(0.0, 1.0)

    ax.set_xlabel("Trial")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(study_root / output_name, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Step 5 HPO running-best curves across methods.")
    parser.add_argument(
        "--study-root",
        type=Path,
        default=None,
        help="Directory containing per-method Step 5 HPO studies. If omitted, resolve from the Step 5 config.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/config5.yaml"))
    parser.add_argument("--base_config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--model_size", default="small")
    parser.add_argument("--c_target", default="")
    parser.add_argument("--font-size", type=int, default=16)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    if args.study_root is not None:
        study_root = args.study_root.resolve()
    else:
        resolved = load_step5_config(
            config_path=args.config,
            base_config_path=args.base_config,
            model_size=args.model_size,
            c_target_override=(args.c_target or None),
        )
        study_root = (resolved.results_dir / "step5_hpo" / resolved.split_mode / resolved.c_target).resolve()
    study_root.mkdir(parents=True, exist_ok=True)

    _plot_metric(
        study_root,
        metric_col="mean_property_success_hit_rate",
        ylabel="Property success hit rate",
        output_name="hpo_compare_best_mean_property_success_hit_rate.png",
        font_size=int(args.font_size),
        dpi=int(args.dpi),
    )
    _plot_metric(
        study_root,
        metric_col="mean_success_hit_rate",
        ylabel="Success hit rate",
        output_name="hpo_compare_best_mean_success_hit_rate.png",
        font_size=int(args.font_size),
        dpi=int(args.dpi),
    )
    stale_output = study_root / "hpo_compare_best_mean_success_hit_rate_discovery.png"
    if stale_output.exists():
        stale_output.unlink()


if __name__ == "__main__":
    main()
