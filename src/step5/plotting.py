"""Shared plotting helpers for Step 5 and Step 5_1."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.figure_style import apply_publication_figure_style


def apply_step5_plot_style(*, font_size: int = 16, dpi: int = 600) -> None:
    """Apply the shared Step 5 plotting style."""

    apply_publication_figure_style(font_size=font_size, dpi=dpi, remove_titles=True)


def _add_compact_target_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "target_label" in out.columns:
        return out
    if "target_row_id" not in out.columns:
        return out
    sort_cols = [col for col in ["temperature", "phi", "target_row_id"] if col in out.columns]
    target_cols = list(dict.fromkeys(["target_row_id", *sort_cols]))
    target_order = out[target_cols].drop_duplicates("target_row_id")
    if sort_cols:
        target_order = target_order.sort_values(sort_cols, kind="mergesort")
    target_order = target_order.reset_index(drop=True)
    width = max(2, len(str(max(1, len(target_order)))))
    label_map = {
        int(row["target_row_id"]): f"T{idx + 1:0{width}d}"
        for idx, row in target_order.iterrows()
    }
    out["target_label"] = out["target_row_id"].map(lambda value: label_map.get(int(value), str(value)))
    return out


def _run_label(row: pd.Series) -> str:
    return str(row.get("run_label", row.get("run_name", "")))


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

    apply_step5_plot_style(font_size=font_size, dpi=dpi)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = _add_compact_target_labels(
        target_row_summary_df.sort_values(["temperature", "phi", "target_row_id"]).reset_index(drop=True)
    )
    mean_col = "mean_property_success_hit_rate"
    std_col = "std_property_success_hit_rate"
    ylabel = "Property success hit rate"
    if mean_col not in ordered.columns:
        mean_col = "mean_success_hit_rate"
        std_col = "std_success_hit_rate"
        ylabel = "Success hit rate"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        range(len(ordered)),
        ordered[mean_col].astype(float),
        yerr=ordered[std_col].astype(float) if std_col in ordered.columns else None,
        color="#4c78a8",
        alpha=0.85,
    )
    ax.set_xlabel("Target row")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(ordered["target_label"].tolist(), rotation=90)
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

    apply_step5_plot_style(font_size=font_size, dpi=dpi)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if "property_success_hit_discovery" in evaluation_df.columns:
        gates = [
            "valid_ok",
            "novel_ok",
            "star_ok",
            "sa_ok_discovery",
            "soluble_ok",
            "chi_ok",
            "property_success_hit_discovery",
        ]
    elif "property_success_hit" in evaluation_df.columns:
        gates = [
            "valid_ok",
            "novel_ok",
            "star_ok",
            "sa_ok",
            "soluble_ok",
            "chi_ok",
            "property_success_hit",
        ]
    else:
        gates = ["valid_ok", "novel_ok", "star_ok", "sa_ok", "soluble_ok", "chi_ok", "success_hit"]
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

    apply_step5_plot_style(font_size=font_size, dpi=dpi)
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

    apply_step5_plot_style(font_size=font_size, dpi=dpi)
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
    ax.set_xticklabels([_run_label(row) for _, row in ordered.iterrows()], rotation=0)
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

    apply_step5_plot_style(font_size=font_size, dpi=dpi)
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
            str(ordered.iloc[idx].get("best_run_label", ordered.iloc[idx]["best_run_name"])),
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=max(8, font_size - 4),
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_hpo_best_success_curve(
    trials_df: pd.DataFrame,
    output_path: Path,
    *,
    font_size: int = 16,
    dpi: int = 600,
) -> None:
    """Plot running-best success hit rate across HPO trials."""

    apply_step5_plot_style(font_size=font_size, dpi=dpi)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = trials_df.copy()
    if "trial_number" not in ordered.columns:
        ordered["trial_number"] = np.arange(len(ordered), dtype=int)
    ordered["trial_number"] = pd.to_numeric(ordered["trial_number"], errors="coerce")
    ordered = ordered.loc[ordered["trial_number"].notna()].sort_values("trial_number", kind="mergesort").reset_index(drop=True)

    metric_col = "mean_success_hit_rate_discovery"
    ylabel = "Best success hit rate so far (discovery)"
    if metric_col not in ordered.columns:
        metric_col = "mean_success_hit_rate"
        ylabel = "Best success hit rate so far"
    metric_values = (
        pd.to_numeric(ordered[metric_col], errors="coerce")
        if metric_col in ordered.columns
        else pd.Series(np.nan, index=ordered.index, dtype=float)
    )

    running_best: list[float] = []
    best_so_far = float("nan")
    for value in metric_values.tolist():
        if np.isfinite(value):
            if not np.isfinite(best_so_far):
                best_so_far = float(value)
            else:
                best_so_far = max(float(best_so_far), float(value))
        running_best.append(float(best_so_far) if np.isfinite(best_so_far) else float("nan"))
    ordered["best_success_hit_rate_so_far"] = running_best

    fig, ax = plt.subplots(figsize=(7, 5))
    if not ordered.empty and np.isfinite(np.asarray(running_best, dtype=float)).any():
        ax.plot(
            ordered["trial_number"].astype(int),
            ordered["best_success_hit_rate_so_far"].astype(float),
            color="#d62728",
            linewidth=2.0,
            marker="o",
            markersize=4.5,
        )
        ymax = float(np.nanmax(ordered["best_success_hit_rate_so_far"].to_numpy(dtype=float)))
        ax.set_ylim(0.0, max(1.0, min(1.02, ymax * 1.05 if ymax > 0.0 else 1.0)))
    else:
        ax.text(0.5, 0.5, "No completed trials", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Trial")
    ax.set_ylabel(ylabel)
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

    apply_step5_plot_style(font_size=font_size, dpi=dpi)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mean_col, _std_col, label = _comparison_success_columns(per_target_run_df)
    ordered = _add_compact_target_labels(
        per_target_run_df.sort_values(["temperature", "phi", "target_row_id", "run_name"]).reset_index(drop=True)
    )
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
        run_display = str(sub["run_label"].iloc[0]) if "run_label" in sub.columns else str(run_name)
        ax.plot(xs, ys, marker="o", linewidth=1.5, alpha=0.9, label=run_display)
    ax.set_xlabel("Target row")
    ax.set_ylabel(f"Mean {label.lower()}")
    ax.set_xticks(range(len(target_order)))
    if "target_label" not in target_order.columns:
        target_order = _add_compact_target_labels(target_order)
    ax.set_xticklabels(target_order["target_label"].tolist(), rotation=90)
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

    apply_step5_plot_style(font_size=font_size, dpi=dpi)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = _add_compact_target_labels(
        difficulty_df.sort_values(["difficulty_rank", "target_row_key"]).reset_index(drop=True)
    )
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
    ax.set_xticklabels(ordered["target_label"].tolist(), rotation=90)
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

    apply_step5_plot_style(font_size=font_size, dpi=dpi)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mean_col, _std_col, _label = _comparison_success_columns(run_comparison_df)
    compare_metric_name = (
        str(run_comparison_df["comparison_metric_name"].iloc[0])
        if "comparison_metric_name" in run_comparison_df.columns and not run_comparison_df.empty
        else "reporting_success_hit_rate"
    )
    is_discovery_metric = "discovery" in compare_metric_name
    is_property_metric = compare_metric_name.startswith("property_")
    sa_gate = "sa_ok_discovery" if is_discovery_metric else "sa_ok"
    if is_property_metric:
        final_gate = "property_success_hit_discovery" if is_discovery_metric else "property_success_hit"
    else:
        final_gate = "success_hit_discovery" if is_discovery_metric else "success_hit"
    gates = ["valid_ok", "novel_ok", "star_ok", sa_gate, "soluble_ok", "chi_ok", final_gate]
    if not is_property_metric:
        gates.insert(-2, "class_ok")

    fig, ax = plt.subplots(figsize=(11, 5))
    ordered = run_comparison_df.sort_values([mean_col, "run_name"], ascending=[False, True]).reset_index(drop=True)
    for _, row in ordered.iterrows():
        rates = []
        for gate in gates[:-1]:
            rates.append(float(row[f"mean_{gate}_rate"]))
        rates.append(float(row[mean_col]))
        ax.plot(gates, rates, marker="o", linewidth=1.5, alpha=0.9, label=_run_label(row))
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

    apply_step5_plot_style(font_size=font_size, dpi=dpi)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mean_col, _std_col, label = _comparison_success_columns(run_comparison_df)
    ordered = run_comparison_df.sort_values([mean_col, "run_name"], ascending=[False, True]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    x = ordered["mean_total_oracle_calls"].astype(float)
    y = ordered[mean_col].astype(float)
    ax.scatter(x, y, s=60, alpha=0.85, color="#b279a2")
    for _, row in ordered.iterrows():
        ax.annotate(
            _run_label(row),
            (float(row["mean_total_oracle_calls"]), float(row[mean_col])),
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("Mean total oracle calls")
    ax.set_ylabel(f"Mean {label.lower()}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_supervised_training_curves(
    history_df: pd.DataFrame,
    output_path: Path,
    *,
    font_size: int = 16,
    dpi: int = 600,
) -> None:
    """Plot S2-family supervised training curves."""

    apply_step5_plot_style(font_size=font_size, dpi=dpi)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=False)
    ordered = history_df.sort_values(["run_label", "global_step"], kind="mergesort")
    for _run_name, sub in ordered.groupby("run_name", sort=True):
        label = str(sub["run_label"].iloc[0]) if "run_label" in sub.columns else str(_run_name)
        x = sub["global_step"].astype(float)
        if "train_diffusion_loss_window" in sub.columns:
            axes[0].plot(x, sub["train_diffusion_loss_window"].astype(float), linewidth=1.5, label=label)
        if "val_diffusion_loss" in sub.columns:
            axes[1].plot(x, sub["val_diffusion_loss"].astype(float), linewidth=1.5, label=label)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Train diffusion loss")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Val diffusion loss")
    axes[1].legend(loc="best", fontsize=max(8, font_size - 4), ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_alignment_training_curves(
    history_df: pd.DataFrame,
    output_path: Path,
    *,
    font_size: int = 16,
    dpi: int = 600,
) -> None:
    """Plot S4 RL/PPO/GRPO training diagnostics."""

    apply_step5_plot_style(font_size=font_size, dpi=dpi)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=False)
    axes_flat = axes.reshape(-1)
    plot_specs = [
        ("loss", "Loss"),
        ("baseline_reward", "Reward"),
        ("trajectory_kl_mean", "KL"),
        ("proxy_property_success_hit_rate_discovery", "Proxy success"),
    ]
    ordered = history_df.sort_values(["run_label", "step_idx"], kind="mergesort")
    for _run_name, sub in ordered.groupby("run_name", sort=True):
        label = str(sub["run_label"].iloc[0]) if "run_label" in sub.columns else str(_run_name)
        x = sub["step_idx"].astype(float)
        for ax, (column, ylabel) in zip(axes_flat, plot_specs):
            if column in sub.columns and sub[column].notna().any():
                ax.plot(x, sub[column].astype(float), linewidth=1.4, label=label)
            ax.set_xlabel("Step")
            ax.set_ylabel(ylabel)
    axes_flat[-1].legend(loc="best", fontsize=max(8, font_size - 4), ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_dpo_training_curves(
    history_df: pd.DataFrame,
    output_path: Path,
    *,
    font_size: int = 16,
    dpi: int = 600,
) -> None:
    """Plot S4 DPO training diagnostics."""

    apply_step5_plot_style(font_size=font_size, dpi=dpi)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=False)
    plot_specs = [
        ("train_dpo_loss", "Train DPO loss"),
        ("val_dpo_loss", "Val DPO loss"),
        ("val_preference_accuracy", "Val pref accuracy"),
    ]
    ordered = history_df.sort_values(["run_label", "epoch_idx"], kind="mergesort")
    for _run_name, sub in ordered.groupby("run_name", sort=True):
        label = str(sub["run_label"].iloc[0]) if "run_label" in sub.columns else str(_run_name)
        x = sub["epoch_idx"].astype(float)
        for ax, (column, ylabel) in zip(axes, plot_specs):
            if column in sub.columns and sub[column].notna().any():
                ax.plot(x, sub[column].astype(float), linewidth=1.5, label=label)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
    axes[-1].legend(loc="best", fontsize=max(8, font_size - 4))
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_chi_vs_target_compare(
    evaluation_df: pd.DataFrame,
    output_path: Path,
    *,
    font_size: int = 16,
    dpi: int = 600,
) -> None:
    """Plot generated chi predictions against target chi across compared runs."""

    apply_step5_plot_style(font_size=font_size, dpi=dpi)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    valid = evaluation_df.loc[
        evaluation_df["chi_pred_target"].notna() & evaluation_df["chi_target"].notna()
    ].copy()
    fig, ax = plt.subplots(figsize=(7, 6))
    if not valid.empty:
        for _run_name, sub in valid.groupby("run_name", sort=True):
            label = str(sub["run_label"].iloc[0]) if "run_label" in sub.columns else str(_run_name)
            ax.scatter(
                sub["chi_target"].astype(float),
                sub["chi_pred_target"].astype(float),
                s=12,
                alpha=0.45,
                label=label,
            )
        lower = min(valid["chi_target"].min(), valid["chi_pred_target"].min())
        upper = max(valid["chi_target"].max(), valid["chi_pred_target"].max())
        ax.plot([lower, upper], [lower, upper], linestyle="--", color="#222222", linewidth=1.2)
        ax.legend(loc="best", fontsize=max(8, font_size - 5), ncol=2)
    ax.set_xlabel("Target chi")
    ax.set_ylabel("Predicted chi")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
