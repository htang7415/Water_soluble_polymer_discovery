#!/usr/bin/env python
"""Recheck pipeline outputs step-by-step and save clear audit CSVs/figures."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.utils.figure_style import apply_publication_figure_style
from src.utils.model_scales import get_results_dir


PRIORITY_METRICS: Dict[str, List[str]] = {
    "step0_data_prep": [
        "train_samples",
        "val_samples",
        "train_roundtrip_pct",
        "val_roundtrip_pct",
        "train_mean_length",
        "val_mean_length",
        "train_mean_sa",
        "val_mean_sa",
    ],
    "step1_backbone": [
        "best_val_loss",
        "n_train_loss_points",
        "n_val_loss_points",
        "train_samples",
        "val_samples",
        "num_params",
        "num_trainable_params",
        "max_steps",
    ],
    "step2_sampling": [
        "generated_count",
        "accepted_count_for_evaluation",
        "samples_per_sec",
        "validity",
        "uniqueness",
        "novelty",
        "avg_diversity",
        "mean_sa",
        "target_polymer_count_selected",
        "target_polymer_selection_success_rate",
    ],
    "step3_chi_target_learning": [
        "global_chi_target",
        "global_balanced_accuracy",
        "global_test_balanced_accuracy",
        "condition_test_balanced_accuracy",
        "condition_test_coverage",
        "n_conditions",
        "mean_condition_balanced_accuracy",
    ],
    "step4_1_regression": [
        "step4_1_test_r2",
        "step4_1_test_rmse",
        "step4_1_test_mae",
        "step4_1_post_optuna_cv_test_r2_mean",
        "step4_1_post_optuna_cv_train_r2_mean",
        "step4_1_n_epochs",
        "n_data_rows_step4_1",
        "n_polymers_step4_1",
    ],
    "step4_2_classification": [
        "step4_2_test_balanced_accuracy",
        "step4_2_test_auroc",
        "step4_2_post_optuna_cv_test_balanced_accuracy_mean",
        "step4_2_post_optuna_cv_train_balanced_accuracy_mean",
        "step4_2_n_epochs",
        "n_data_rows_step4_2",
        "n_polymers_step4_2",
    ],
    "step5_water_soluble_inverse_design": [
        "target_success_rate",
        "mean_top1_abs_error",
        "mean_mrr_joint",
        "mean_joint_hit_rate",
        "target_polymer_count_selected",
        "target_polymer_selection_success_rate",
        "target_polymer_diversity",
        "target_polymer_mean_sa",
    ],
    "step6_polymer_class_water_soluble_inverse_design": [
        "target_success_rate",
        "mean_top1_abs_error",
        "mean_mrr_joint",
        "mean_joint_hit_rate",
        "target_polymer_count_selected",
        "target_polymer_selection_success_rate",
        "target_polymer_diversity",
        "target_polymer_mean_sa",
    ],
}


PRIMARY_METRIC_CANDIDATES: Dict[str, List[str]] = {
    "step0_data_prep": ["train_roundtrip_pct", "val_roundtrip_pct"],
    "step1_backbone": ["best_val_loss", "aux_min_val_loss"],
    "step2_sampling": ["novelty", "validity", "samples_per_sec"],
    "step3_chi_target_learning": ["global_test_balanced_accuracy", "condition_test_balanced_accuracy"],
    "step4_1_regression": ["step4_1_test_r2", "aux_test_r2"],
    "step4_2_classification": ["step4_2_test_balanced_accuracy", "aux_test_balanced_accuracy"],
    "step5_water_soluble_inverse_design": ["target_success_rate", "aux_target_success_rate"],
    "step6_polymer_class_water_soluble_inverse_design": ["target_success_rate", "aux_target_success_rate"],
}


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
        f"step4_1_regression/{split_mode}": [
            "pipeline_metrics/step_summary.csv",
            "metrics/chi_metrics_overall.csv",
            "metrics/polymer_coefficients_regression_only.csv",
            "figures/chi_loss_curve.png",
        ],
        "step4_2_classification": [
            "pipeline_metrics/step_summary.csv",
            "metrics/class_metrics_overall.csv",
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


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _as_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            out = float(int(value))
        else:
            out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _read_step_summary(step_name: str, step_path: Path) -> Dict[str, object]:
    candidates: List[Path] = []
    if step_name in {"step4_1_regression", "step4_2_classification"}:
        candidates.append(step_path / "pipeline_metrics" / "step_summary.csv")
    candidates.append(step_path / "metrics" / "step_summary.csv")

    for csv_path in candidates:
        df = _safe_read_csv(csv_path)
        if not df.empty:
            return df.iloc[0].to_dict()
    return {}


def _extract_aux_metrics(step_name: str, step_path: Path) -> List[Tuple[str, float, str]]:
    rows: List[Tuple[str, float, str]] = []

    def add_metric(name: str, value, source: str = "aux") -> None:
        f = _as_float(value)
        if f is not None:
            rows.append((name, f, source))

    if step_name == "step1_backbone":
        val_df = _safe_read_csv(step_path / "metrics" / "backbone_val_loss.csv")
        if not val_df.empty and "val_loss" in val_df.columns:
            vals = pd.to_numeric(val_df["val_loss"], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                add_metric("aux_min_val_loss", float(np.min(vals)))
                add_metric("aux_last_val_loss", float(vals[-1]))

    if step_name == "step2_sampling":
        gen_df = _safe_read_csv(step_path / "metrics" / "sampling_generative_metrics.csv")
        if not gen_df.empty:
            row = gen_df.iloc[0]
            for key in ["validity", "uniqueness", "novelty", "avg_diversity", "mean_sa", "samples_per_sec"]:
                if key in gen_df.columns:
                    add_metric(f"aux_{key}", row[key])

    if step_name == "step3_chi_target_learning":
        holdout_df = _safe_read_csv(step_path / "metrics" / "chi_target_holdout_metrics.csv")
        if not holdout_df.empty and "scope" in holdout_df.columns:
            for scope in ["global", "conditional"]:
                sub = holdout_df[holdout_df["scope"] == scope]
                if sub.empty:
                    continue
                row = sub.iloc[0]
                for key in ["balanced_accuracy", "accuracy", "f1", "youden_j"]:
                    if key in holdout_df.columns:
                        add_metric(f"aux_{scope}_{key}", row[key])

    if step_name == "step4_1_regression":
        reg_df = _safe_read_csv(step_path / "metrics" / "chi_metrics_overall.csv")
        if not reg_df.empty:
            if "split" in reg_df.columns:
                sub = reg_df[reg_df["split"] == "test"]
                row = sub.iloc[0] if not sub.empty else reg_df.iloc[-1]
            else:
                row = reg_df.iloc[-1]
            for key in ["mae", "rmse", "r2"]:
                if key in reg_df.columns:
                    add_metric(f"aux_test_{key}", row[key])

        cv_df = _safe_read_csv(step_path / "metrics" / "cv_best_params" / "cv_metrics_summary.csv")
        if not cv_df.empty and "cv_split" in cv_df.columns:
            sub = cv_df[cv_df["cv_split"] == "test"]
            if not sub.empty:
                row = sub.iloc[0]
                for key in ["r2_mean", "rmse_mean", "mae_mean"]:
                    if key in cv_df.columns:
                        add_metric(f"aux_post_optuna_cv_test_{key}", row[key])

    if step_name == "step4_2_classification":
        cls_df = _safe_read_csv(step_path / "metrics" / "class_metrics_overall.csv")
        if not cls_df.empty:
            if "split" in cls_df.columns:
                sub = cls_df[cls_df["split"] == "test"]
                row = sub.iloc[0] if not sub.empty else cls_df.iloc[-1]
            else:
                row = cls_df.iloc[-1]
            for key in ["balanced_accuracy", "auroc", "auprc", "brier"]:
                if key in cls_df.columns:
                    add_metric(f"aux_test_{key}", row[key])

        cv_df = _safe_read_csv(step_path / "metrics" / "cv_best_params" / "cv_metrics_summary.csv")
        if not cv_df.empty and "cv_split" in cv_df.columns:
            sub = cv_df[cv_df["cv_split"] == "test"]
            if not sub.empty:
                row = sub.iloc[0]
                for key in ["balanced_accuracy_mean", "auroc_mean", "auprc_mean", "brier_mean"]:
                    if key in cv_df.columns:
                        add_metric(f"aux_post_optuna_cv_test_{key}", row[key])

    if step_name in {"step5_water_soluble_inverse_design", "step6_polymer_class_water_soluble_inverse_design"}:
        agg_df = _safe_read_csv(step_path / "metrics" / "inverse_aggregate_metrics.csv")
        if not agg_df.empty:
            if "scope" in agg_df.columns:
                sub = agg_df[agg_df["scope"] == "overall"]
                row = sub.iloc[0] if not sub.empty else agg_df.iloc[-1]
            else:
                row = agg_df.iloc[-1]
            for key in [
                "target_success_rate",
                "mean_top1_abs_error",
                "mean_mrr_joint",
                "mean_joint_hit_rate",
                "mean_soluble_hit_rate",
                "mean_property_hit_rate",
            ]:
                if key in agg_df.columns:
                    add_metric(f"aux_{key}", row[key])

    return rows


def _collect_step_metric_rows(step_name: str, step_path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    metric_order = 0

    summary = _read_step_summary(step_name=step_name, step_path=step_path)
    numeric_summary: Dict[str, float] = {}
    for key, value in summary.items():
        f = _as_float(value)
        if f is not None:
            numeric_summary[str(key)] = f

    preferred = PRIORITY_METRICS.get(step_name, [])
    chosen: List[str] = []
    for key in preferred:
        if key in numeric_summary and key not in chosen:
            chosen.append(key)

    for key in sorted(numeric_summary.keys()):
        if key not in chosen:
            chosen.append(key)

    chosen = chosen[:14]
    chosen_set = set(chosen)

    for key in chosen:
        rows.append(
            {
                "step_name": step_name,
                "metric_name": key,
                "metric_value": float(numeric_summary[key]),
                "source": "step_summary",
                "metric_order": metric_order,
            }
        )
        metric_order += 1

    for metric_name, metric_value, source in _extract_aux_metrics(step_name=step_name, step_path=step_path):
        if metric_name in chosen_set:
            continue
        rows.append(
            {
                "step_name": step_name,
                "metric_name": metric_name,
                "metric_value": float(metric_value),
                "source": source,
                "metric_order": metric_order,
            }
        )
        metric_order += 1

    return rows


def _collect_step_figure_paths(step_path: Path) -> List[Path]:
    fig_root = step_path / "figures"
    if not fig_root.exists():
        return []
    return sorted([p for p in fig_root.rglob("*.png") if p.is_file()])


def _pick_representative_figure(fig_paths: List[Path]) -> Optional[Path]:
    if not fig_paths:
        return None
    preferred_fragments = [
        "topk_target_success_curve",
        "chi_parity_test",
        "class_parity_test",
        "chi_target_heatmap",
        "chi_loss_curve",
        "class_loss_curve",
        "backbone_loss_curve",
        "sa_hist_train_vs_uncond",
        "length_hist_train_val",
    ]
    by_name = sorted(fig_paths, key=lambda p: p.name)
    for frag in preferred_fragments:
        for p in by_name:
            if frag in p.name:
                return p
    return by_name[0]


def _make_step_gallery(step_name: str, step_path: Path, out_png: Path, max_panels: int, dpi: int) -> None:
    fig_paths = _collect_step_figure_paths(step_path)

    if len(fig_paths) == 0:
        fig, ax = plt.subplots(figsize=(7.0, 3.2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No figure files found", ha="center", va="center", fontsize=15)
        ax.set_title(f"{step_name} figure gallery")
        fig.tight_layout()
        fig.savefig(out_png, dpi=dpi)
        plt.close(fig)
        return

    show_paths = fig_paths[: max(1, int(max_panels))]
    n = len(show_paths)
    cols = min(3, max(1, int(math.ceil(math.sqrt(n)))))
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows))
    axes_arr = np.atleast_1d(axes).reshape(-1)

    for i, ax in enumerate(axes_arr):
        if i >= n:
            ax.axis("off")
            continue
        p = show_paths[i]
        rel = p.relative_to(step_path / "figures")
        try:
            img = plt.imread(p)
            ax.imshow(img)
            ax.axis("off")
        except Exception:
            ax.axis("off")
            ax.text(0.5, 0.5, "Could not load image", ha="center", va="center", fontsize=15)
        ax.set_title(str(rel), fontsize=15)

    fig.suptitle(f"{step_name} figure gallery ({n}/{len(fig_paths)})")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def _make_pipeline_storyboard(step_items: List[Tuple[str, Path]], fig_dir: Path, dpi: int) -> None:
    out_png = fig_dir / "pipeline_storyboard.png"
    n = len(step_items)
    cols = min(3, max(1, n))
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.4 * rows))
    axes_arr = np.atleast_1d(axes).reshape(-1)

    for i, ax in enumerate(axes_arr):
        if i >= n:
            ax.axis("off")
            continue

        step_name, step_path = step_items[i]
        rep = _pick_representative_figure(_collect_step_figure_paths(step_path))
        if rep is None:
            ax.axis("off")
            ax.text(0.5, 0.5, "No figure", ha="center", va="center", fontsize=15)
            ax.set_title(step_name, fontsize=15)
            continue

        try:
            img = plt.imread(rep)
            ax.imshow(img)
            ax.axis("off")
        except Exception:
            ax.axis("off")
            ax.text(0.5, 0.5, "Could not load image", ha="center", va="center", fontsize=15)

        rel = rep.relative_to(step_path / "figures") if (step_path / "figures") in rep.parents else rep.name
        ax.set_title(f"{step_name}\n{rel}", fontsize=15)

    fig.suptitle("Pipeline storyboard (representative figure per step)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def _make_step_metric_cards(metric_df: pd.DataFrame, step_order: List[str], fig_dir: Path, dpi: int) -> None:
    out_png = fig_dir / "step_metric_cards.png"
    n = max(1, len(step_order))
    fig, axes = plt.subplots(n, 1, figsize=(10.5, max(3.2, 2.2 * n)))
    axes_arr = np.atleast_1d(axes).reshape(-1)

    for i, step_name in enumerate(step_order):
        ax = axes_arr[i]
        ax.axis("off")
        sub = metric_df[metric_df["step_name"] == step_name].sort_values(["metric_order", "metric_name"]) if not metric_df.empty else pd.DataFrame()
        sub = sub.head(12)
        lines: List[str] = []
        for _, row in sub.iterrows():
            metric_name = str(row["metric_name"])
            metric_value = float(row["metric_value"])
            lines.append(f"{metric_name:<45} = {metric_value:>12.5g}")

        if len(lines) == 0:
            lines = ["No numeric metrics found"]

        ax.text(
            0.01,
            0.97,
            "\n".join(lines),
            va="top",
            ha="left",
            fontsize=15,
            family="monospace",
            transform=ax.transAxes,
        )
        ax.set_title(step_name, loc="left", fontsize=15)

    fig.suptitle("Step metric snapshot (from step summaries + auxiliary metric files)")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def _make_primary_metric_figure(metric_df: pd.DataFrame, step_order: List[str], fig_dir: Path, dpi: int) -> None:
    out_png = fig_dir / "step_primary_metrics.png"

    labels: List[str] = []
    values: List[float] = []
    names: List[str] = []

    for step_name in step_order:
        sub = metric_df[metric_df["step_name"] == step_name] if not metric_df.empty else pd.DataFrame()
        if sub.empty:
            continue

        metric_row = None
        for cand in PRIMARY_METRIC_CANDIDATES.get(step_name, []):
            hit = sub[sub["metric_name"] == cand]
            if not hit.empty:
                metric_row = hit.iloc[0]
                break

        if metric_row is None:
            metric_row = sub.sort_values(["metric_order", "metric_name"]).iloc[0]

        labels.append(step_name)
        values.append(float(metric_row["metric_value"]))
        names.append(str(metric_row["metric_name"]))

    if len(labels) == 0:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.axis("off")
        ax.text(0.5, 0.5, "No primary metrics available", ha="center", va="center", fontsize=15)
        fig.tight_layout()
        fig.savefig(out_png, dpi=dpi)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color="#4c78a8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title("Primary metric by step")
    for i, b in enumerate(bars):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{names[i]}\n{values[i]:.4g}",
            ha="center",
            va="bottom",
            fontsize=15,
        )
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def _make_figures(
    check_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    step_items: List[Tuple[str, Path]],
    fig_dir: Path,
    dpi: int = 600,
    max_gallery_panels: int = 9,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    apply_publication_figure_style(font_size=15, dpi=int(dpi), remove_titles=True)
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

        step_order = [name for name, _ in step_items]
        _make_primary_metric_figure(metric_df=metric_df, step_order=step_order, fig_dir=fig_dir, dpi=dpi)
        _make_step_metric_cards(metric_df=metric_df, step_order=step_order, fig_dir=fig_dir, dpi=dpi)
        _make_pipeline_storyboard(step_items=step_items, fig_dir=fig_dir, dpi=dpi)

        for step_name, step_path in step_items:
            safe_name = step_name.replace("/", "_")
            _make_step_gallery(
                step_name=step_name,
                step_path=step_path,
                out_png=fig_dir / f"gallery_{safe_name}.png",
                max_panels=max_gallery_panels,
                dpi=dpi,
            )
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
    step_items: List[Tuple[str, Path]] = []

    for step_rel, reqs in expected.items():
        step_name = step_rel.split("/")[0]
        if step_name in {"step0_data_prep", "step3_chi_target_learning"}:
            step_root = base_results_dir
        elif step_name in {"step4_1_regression", "step4_2_classification"}:
            step_root = results_dir_nosplit
        else:
            step_root = results_dir
        step_path = step_root / step_rel
        step_items.append((step_name, step_path))
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

    metric_rows: List[Dict[str, object]] = []
    figure_rows: List[Dict[str, object]] = []
    for step_name, step_path in step_items:
        metric_rows.extend(_collect_step_metric_rows(step_name=step_name, step_path=step_path))

        for p in _collect_step_figure_paths(step_path):
            rel = p.relative_to(step_path)
            figure_rows.append(
                {
                    "step_name": step_name,
                    "step_dir": str(step_path),
                    "figure_relative_path": str(rel),
                    "figure_name": p.name,
                    "size_bytes": int(p.stat().st_size),
                }
            )

    metric_df = pd.DataFrame(metric_rows)
    if metric_df.empty:
        metric_df = pd.DataFrame(columns=["step_name", "metric_name", "metric_value", "source", "metric_order"])
    metric_df.to_csv(metrics_dir / "step_metric_snapshot_long.csv", index=False)

    if not metric_df.empty:
        wide_df = (
            metric_df.sort_values(["step_name", "metric_order", "metric_name"])
            .pivot_table(index="step_name", columns="metric_name", values="metric_value", aggfunc="first")
            .reset_index()
        )
        wide_df.to_csv(metrics_dir / "step_metric_snapshot_wide.csv", index=False)
    else:
        pd.DataFrame(columns=["step_name"]).to_csv(metrics_dir / "step_metric_snapshot_wide.csv", index=False)

    figure_df = pd.DataFrame(figure_rows)
    if figure_df.empty:
        figure_df = pd.DataFrame(columns=["step_name", "step_dir", "figure_relative_path", "figure_name", "size_bytes"])
    figure_df.to_csv(metrics_dir / "step_figure_inventory.csv", index=False)

    if not args.skip_figures:
        _make_figures(
            check_df=check_df,
            metric_df=metric_df,
            step_items=step_items,
            fig_dir=figures_dir,
            dpi=int(config.get("plotting", {}).get("dpi", 600)),
            max_gallery_panels=int(args.max_gallery_panels),
        )

    print("=" * 70)
    print("Pipeline recheck complete")
    print(f"Results dir: {results_dir}")
    print(f"Checklist:       {metrics_dir / 'step_output_checklist.csv'}")
    print(f"Missing:         {metrics_dir / 'missing_required_files.csv'}")
    print(f"Manifest:        {metrics_dir / 'all_artifacts_manifest.csv'}")
    print(f"Metric snapshot: {metrics_dir / 'step_metric_snapshot_long.csv'}")
    print(f"Figure index:    {metrics_dir / 'step_figure_inventory.csv'}")
    print(f"Figures:         {figures_dir}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recheck pipeline outputs step-by-step")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config path")
    parser.add_argument("--model_size", type=str, default="small", choices=["small", "medium", "large", "xl"], help="Model size namespace")
    parser.add_argument("--split_mode", type=str, default="polymer", choices=["polymer", "random"], help="Split mode for chi steps")
    parser.add_argument("--skip_figures", action="store_true", help="Skip figure generation and only write CSV reports")
    parser.add_argument("--max_gallery_panels", type=int, default=9, help="Max images shown per-step gallery figure")
    args = parser.parse_args()
    main(args)
