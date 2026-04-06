"""Shared plotting helpers for Step 6_2 and Step 6_3."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.figure_style import apply_publication_figure_style


def apply_step62_plot_style(*, font_size: int = 16, dpi: int = 600) -> None:
    """Apply the shared Step 6_2 plotting style."""

    apply_publication_figure_style(font_size=font_size, dpi=dpi, remove_titles=True)


def _comparison_success_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    mean_col = "comparison_mean_success_hit_rate" if "comparison_mean_success_hit_rate" in df.columns else "mean_success_hit_rate"
    std_col = "comparison_std_success_hit_rate" if "comparison_std_success_hit_rate" in df.columns else "std_success_hit_rate"
    label = (
        str(df["comparison_metric_label"].iloc[0])
        if "comparison_metric_label" in df.columns and not df.empty
        else "Success hit rate"
    )
    return mean_col, std_col, label


def _family_success_columns(df: pd.DataFrame) -> tuple[str, str]:
    mean_col = "best_comparison_success_hit_rate" if "best_comparison_success_hit_rate" in df.columns else "best_mean_success_hit_rate"
    label = (
        str(df["comparison_metric_label"].iloc[0])
        if "comparison_metric_label" in df.columns and not df.empty
        else "Best mean success hit rate"
    )
    return mean_col, label


def plot_per_target_success(
    target_row_summary_df: pd.DataFrame,
    output_path: Path,
    *,
    font_size: int = 16,
    dpi: int = 600,
) -> None:
    """Write a simple per-target success-rate plot."""

    apply_step62_plot_style(font_size=font_size, dpi=dpi)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = target_row_summary_df.sort_values(["temperature", "phi", "target_row_id"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        range(len(ordered)),
        ordered["mean_success_hit_rate"].astype(float),
        yerr=ordered["std_success_hit_rate"].astype(float),
        color="#4c78a8",
        alpha=0.85,
    )
    ax.set_xlabel("Target row")
    ax.set_ylabel("Success hit rate")
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(ordered["target_row_key"].tolist(), rotation=90)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_success_gate_funnel(
    evaluation_df: pd.DataFrame,
    output_path: Path,
    *,
    font_size: int = 16,
    dpi: int = 600,
) -> None:
    """Write a simple gate-funnel plot for one run."""

    apply_step62_plot_style(font_size=font_size, dpi=dpi)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gates = ["valid_ok", "novel_ok", "star_ok", "sa_ok", "soluble_ok", "class_ok", "chi_ok", "success_hit"]
    rates = [float(evaluation_df[gate].astype(float).mean()) for gate in gates]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(gates, rates, marker="o", color="#f58518")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Gate")
    ax.set_ylabel("Pass rate")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_generated_chi_vs_target(
    evaluation_df: pd.DataFrame,
    output_path: Path,
    *,
    font_size: int = 16,
    dpi: int = 600,
) -> None:
    """Write generated chi versus target-threshold scatter plot."""

    apply_step62_plot_style(font_size=font_size, dpi=dpi)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    valid = evaluation_df.loc[evaluation_df["chi_pred_target"].notna()].copy()

    fig, ax = plt.subplots(figsize=(6, 6))
    if not valid.empty:
        ax.scatter(
            valid["chi_target"].astype(float),
            valid["chi_pred_target"].astype(float),
            s=14,
            alpha=0.6,
            color="#0C5DA5",
        )
        lower = min(valid["chi_target"].min(), valid["chi_pred_target"].min())
        upper = max(valid["chi_target"].max(), valid["chi_pred_target"].max())
        ax.plot([lower, upper], [lower, upper], linestyle="--", color="#222222", linewidth=1.2)
    ax.set_xlabel("Target chi")
    ax.set_ylabel("Predicted chi")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_overall_success_all_runs(
    run_comparison_df: pd.DataFrame,
    output_path: Path,
    *,
    font_size: int = 16,
    dpi: int = 600,
) -> None:
    """Bar chart of mean success hit rate for all compared runs."""

    apply_step62_plot_style(font_size=font_size, dpi=dpi)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mean_col, std_col, label = _comparison_success_columns(run_comparison_df)
    ordered = run_comparison_df.sort_values([mean_col, "run_name"], ascending=[False, True]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        range(len(ordered)),
        ordered[mean_col].astype(float),
        yerr=ordered[std_col].astype(float),
        color="#4c78a8",
        alpha=0.85,
    )
    ax.set_xlabel("Run")
    ax.set_ylabel(f"Mean {label.lower()}")
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(ordered["run_name"].tolist(), rotation=45, ha="right")
    ax.set_ylim(0.0, max(1.0, float(np.nanmax(ordered[mean_col].to_numpy(dtype=float))) * 1.1))
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_overall_success_by_family(
    canonical_family_df: pd.DataFrame,
    output_path: Path,
    *,
    font_size: int = 16,
    dpi: int = 600,
) -> None:
    """Bar chart of best run per canonical family."""

    apply_step62_plot_style(font_size=font_size, dpi=dpi)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mean_col, label = _family_success_columns(canonical_family_df)
    ordered = canonical_family_df.sort_values([mean_col, "canonical_family"], ascending=[False, True]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        range(len(ordered)),
        ordered[mean_col].astype(float),
        color="#54a24b",
        alpha=0.85,
    )
    ax.set_xlabel("Canonical family")
    ax.set_ylabel(label)
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(ordered["canonical_family"].tolist())
    for idx, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            str(ordered.iloc[idx]["best_run_name"]),
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=max(8, font_size - 4),
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_per_target_success_compare(
    per_target_run_df: pd.DataFrame,
    output_path: Path,
    *,
    font_size: int = 16,
    dpi: int = 600,
) -> None:
    """Line plot of per-target success hit rate across compared runs."""

    apply_step62_plot_style(font_size=font_size, dpi=dpi)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mean_col, _std_col, label = _comparison_success_columns(per_target_run_df)
    ordered = per_target_run_df.sort_values(["temperature", "phi", "target_row_id", "run_name"]).reset_index(drop=True)
    target_order = (
        ordered[["target_row_id", "target_row_key", "temperature", "phi"]]
        .drop_duplicates()
        .sort_values(["temperature", "phi", "target_row_id"])
        .reset_index(drop=True)
    )
    target_positions = {int(row["target_row_id"]): idx for idx, row in target_order.iterrows()}

    fig, ax = plt.subplots(figsize=(11, 5))
    for run_name, sub in ordered.groupby("run_name", sort=True):
        sub = sub.sort_values(["temperature", "phi", "target_row_id"])
        xs = [target_positions[int(target_row_id)] for target_row_id in sub["target_row_id"]]
        ys = sub[mean_col].astype(float).tolist()
        ax.plot(xs, ys, marker="o", linewidth=1.5, alpha=0.9, label=str(run_name))
    ax.set_xlabel("Target row")
    ax.set_ylabel(f"Mean {label.lower()}")
    ax.set_xticks(range(len(target_order)))
    ax.set_xticklabels(target_order["target_row_key"].tolist(), rotation=90)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="best", fontsize=max(8, font_size - 4), ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_per_target_difficulty_ranked(
    difficulty_df: pd.DataFrame,
    output_path: Path,
    *,
    font_size: int = 16,
    dpi: int = 600,
) -> None:
    """Bar chart of target-row difficulty ranked by mean success across runs."""

    apply_step62_plot_style(font_size=font_size, dpi=dpi)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = difficulty_df.sort_values(["difficulty_rank", "target_row_key"]).reset_index(drop=True)
    label = (
        str(ordered["comparison_metric_label"].iloc[0])
        if "comparison_metric_label" in ordered.columns and not ordered.empty
        else "Success hit rate"
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        range(len(ordered)),
        ordered["mean_success_hit_rate_across_runs"].astype(float),
        yerr=ordered["std_success_hit_rate_across_runs"].astype(float),
        color="#e45756",
        alpha=0.85,
    )
    ax.set_xlabel("Target row")
    ax.set_ylabel(f"Mean {label.lower()} across runs")
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(ordered["target_row_key"].tolist(), rotation=90)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_success_gate_funnel_compare(
    run_comparison_df: pd.DataFrame,
    output_path: Path,
    *,
    font_size: int = 16,
    dpi: int = 600,
) -> None:
    """Compare gate pass rates across runs."""

    apply_step62_plot_style(font_size=font_size, dpi=dpi)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mean_col, _std_col, _label = _comparison_success_columns(run_comparison_df)
    compare_metric_name = (
        str(run_comparison_df["comparison_metric_name"].iloc[0])
        if "comparison_metric_name" in run_comparison_df.columns and not run_comparison_df.empty
        else "reporting_success_hit_rate"
    )
    sa_gate = "sa_ok_discovery" if compare_metric_name.startswith("discovery_") else "sa_ok"
    gates = ["valid_ok", "novel_ok", "star_ok", sa_gate, "soluble_ok", "class_ok", "chi_ok", "success_hit"]

    fig, ax = plt.subplots(figsize=(11, 5))
    ordered = run_comparison_df.sort_values([mean_col, "run_name"], ascending=[False, True]).reset_index(drop=True)
    for _, row in ordered.iterrows():
        rates = []
        for gate in gates[:-1]:
            rates.append(float(row[f"mean_{gate}_rate"]))
        rates.append(float(row[mean_col]))
        ax.plot(gates, rates, marker="o", linewidth=1.5, alpha=0.9, label=str(row["run_name"]))
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Gate")
    ax.set_ylabel("Mean pass rate")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="best", fontsize=max(8, font_size - 4), ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_success_vs_oracle_budget(
    run_comparison_df: pd.DataFrame,
    output_path: Path,
    *,
    font_size: int = 16,
    dpi: int = 600,
) -> None:
    """Scatter plot of success hit rate against average oracle-call budget."""

    apply_step62_plot_style(font_size=font_size, dpi=dpi)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mean_col, _std_col, label = _comparison_success_columns(run_comparison_df)
    ordered = run_comparison_df.sort_values([mean_col, "run_name"], ascending=[False, True]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    x = ordered["mean_total_oracle_calls"].astype(float)
    y = ordered[mean_col].astype(float)
    ax.scatter(x, y, s=60, alpha=0.85, color="#b279a2")
    for _, row in ordered.iterrows():
        ax.annotate(str(row["run_name"]), (float(row["mean_total_oracle_calls"]), float(row[mean_col])))
    ax.set_xlabel("Mean total oracle calls")
    ax.set_ylabel(f"Mean {label.lower()}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
