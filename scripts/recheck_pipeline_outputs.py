#!/usr/bin/env python
"""Recheck pipeline outputs step-by-step and save clear audit CSVs/figures."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.utils.model_scales import get_results_dir


def _expected_files(split_mode: str) -> Dict[str, List[str]]:
    return {
        "step0_data_prep": [
            "metrics/step_summary.csv",
            "metrics/artifact_manifest.csv",
            "metrics/tokenizer_roundtrip.csv",
            "metrics/unlabeled_data_stats.csv",
            "figures/length_hist_train_val.png",
        ],
        "step1_backbone": [
            "metrics/step_summary.csv",
            "metrics/artifact_manifest.csv",
            "metrics/backbone_loss_curve.csv",
            "figures/backbone_loss_curve.png",
        ],
        "step2_sampling": [
            "metrics/step_summary.csv",
            "metrics/artifact_manifest.csv",
            "metrics/generated_samples.csv",
            "metrics/sampling_generative_metrics.csv",
            "figures/sa_hist_train_vs_uncond.png",
        ],
        f"step3_chi_target_learning/{split_mode}": [
            "metrics/step_summary.csv",
            "metrics/artifact_manifest.csv",
            "metrics/chi_target_for_inverse_design.csv",
            "metrics/chi_target_summary.json",
            "figures/chi_target_heatmap.png",
        ],
        "step4_chi_training": [
            f"pipeline_metrics/both_{split_mode}/step_summary.csv",
            f"step4_1_regression/{split_mode}/metrics/chi_metrics_overall.csv",
            f"step4_1_regression/{split_mode}/metrics/polymer_coefficients_regression_only.csv",
            "step4_2_classification/metrics/class_metrics_overall.csv",
            f"step4_1_regression/{split_mode}/figures/chi_loss_curve.png",
        ],
        f"step5_water_soluble_inverse_design/{split_mode}": [
            "metrics/step_summary.csv",
            "metrics/artifact_manifest.csv",
            "metrics/inverse_target_metrics.csv",
            "metrics/step5_summary.json",
            "figures/topk_target_success_curve.png",
        ],
        f"step6_polymer_class_water_soluble_inverse_design/{split_mode}": [
            "metrics/step_summary.csv",
            "metrics/artifact_manifest.csv",
            "metrics/inverse_target_metrics.csv",
            "metrics/step6_summary.json",
            "figures/topk_target_success_curve.png",
        ],
    }


def _count_artifacts(step_path: Path) -> Tuple[int, int, int]:
    if not step_path.exists():
        return 0, 0, 0
    csv_count = len(list(step_path.rglob("*.csv")))
    fig_count = len(list(step_path.rglob("*.png"))) + len(list(step_path.rglob("*.pdf")))
    total = len([p for p in step_path.rglob("*") if p.is_file()])
    return csv_count, fig_count, total


def _make_figures(check_df: pd.DataFrame, fig_dir: Path, dpi: int = 600) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Completion by step
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        tmp = check_df.copy()
        tmp["ok"] = tmp["missing_required_count"].eq(0).astype(int)
        ax.bar(tmp["step_name"], tmp["ok"], color=["#54a24b" if v == 1 else "#e45756" for v in tmp["ok"]])
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Complete (1=yes)")
        ax.set_xlabel("Step")
        ax.set_title("Pipeline Step Completion")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(fig_dir / "step_completion.png", dpi=dpi)
        plt.close(fig)

        # Artifact count overview
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = range(len(check_df))
        ax.bar([i - 0.15 for i in x], check_df["csv_count"], width=0.3, label="CSV", color="#4c78a8")
        ax.bar([i + 0.15 for i in x], check_df["figure_count"], width=0.3, label="Figure", color="#f58518")
        ax.set_xticks(list(x))
        ax.set_xticklabels(check_df["step_name"], rotation=30)
        ax.set_xlabel("Step")
        ax.set_ylabel("File count")
        ax.set_title("CSV and Figure Counts by Step")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "step_artifact_counts.png", dpi=dpi)
        plt.close(fig)
    except Exception as exc:
        (fig_dir / "figure_generation_error.txt").write_text(str(exc))


def main(args):
    config = load_config(args.config)
    base_results_dir = Path(config["paths"]["results_dir"])
    split_mode = args.split_mode
    results_dir = Path(get_results_dir(args.model_size, config["paths"]["results_dir"], split_mode))
    results_dir_nosplit = Path(get_results_dir(args.model_size, config["paths"]["results_dir"], None))

    out_dir = results_dir / "pipeline_recheck" / split_mode
    metrics_dir = out_dir / "metrics"
    figures_dir = out_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    expected = _expected_files(split_mode)

    step_rows = []
    missing_rows = []
    manifest_rows = []

    for step_rel, reqs in expected.items():
        step_name = step_rel.split("/")[0]
        if step_name in {"step0_data_prep", "step3_chi_target_learning"}:
            step_root = base_results_dir
        elif step_name == "step4_chi_training":
            step_root = results_dir_nosplit
        else:
            step_root = results_dir
        step_path = step_root / step_rel
        exists = step_path.exists()

        csv_count, fig_count, total_count = _count_artifacts(step_path)
        missing_count = 0

        for req in reqs:
            target = step_path / req
            ok = target.exists()
            if not ok:
                missing_count += 1
                missing_rows.append(
                    {
                        "step_name": step_name,
                        "step_dir": str(step_path),
                        "step_root": str(step_root),
                        "missing_relative_path": req,
                    }
                )

        if exists:
            for p in sorted(step_path.rglob("*")):
                if p.is_file():
                    manifest_rows.append(
                        {
                            "step_name": step_name,
                            "step_dir": str(step_path),
                            "step_root": str(step_root),
                            "relative_path": str(p.relative_to(step_path)),
                            "suffix": p.suffix.lower(),
                            "size_bytes": int(p.stat().st_size),
                        }
                    )

        step_rows.append(
            {
                "step_name": step_name,
                "step_dir": str(step_path),
                "step_root": str(step_root),
                "exists": int(exists),
                "required_file_count": int(len(reqs)),
                "missing_required_count": int(missing_count),
                "csv_count": int(csv_count),
                "figure_count": int(fig_count),
                "total_file_count": int(total_count),
            }
        )

    check_df = pd.DataFrame(step_rows)
    missing_df = pd.DataFrame(missing_rows)
    manifest_df = pd.DataFrame(manifest_rows)

    check_df.to_csv(metrics_dir / "step_output_checklist.csv", index=False)
    if not missing_df.empty:
        missing_df.to_csv(metrics_dir / "missing_required_files.csv", index=False)
    else:
        pd.DataFrame(columns=["step_name", "step_dir", "step_root", "missing_relative_path"]).to_csv(
            metrics_dir / "missing_required_files.csv", index=False
        )

    if not manifest_df.empty:
        manifest_df.to_csv(metrics_dir / "all_artifacts_manifest.csv", index=False)
    else:
        pd.DataFrame(columns=["step_name", "step_dir", "step_root", "relative_path", "suffix", "size_bytes"]).to_csv(
            metrics_dir / "all_artifacts_manifest.csv", index=False
        )

    if not args.skip_figures:
        _make_figures(check_df, figures_dir, dpi=int(config.get("plotting", {}).get("dpi", 600)))

    print("=" * 70)
    print("Pipeline recheck complete")
    print(f"Results dir: {results_dir}")
    print(f"Checklist: {metrics_dir / 'step_output_checklist.csv'}")
    print(f"Missing:   {metrics_dir / 'missing_required_files.csv'}")
    print(f"Manifest:  {metrics_dir / 'all_artifacts_manifest.csv'}")
    print(f"Figures:   {figures_dir}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recheck pipeline outputs step-by-step")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config path")
    parser.add_argument("--model_size", type=str, default="small", choices=["small", "medium", "large", "xl"], help="Model size namespace")
    parser.add_argument("--split_mode", type=str, default="polymer", choices=["polymer", "random"], help="Split mode for chi steps")
    parser.add_argument("--skip_figures", action="store_true", help="Skip figure generation and only write CSV reports")
    args = parser.parse_args()
    main(args)
