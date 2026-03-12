#!/usr/bin/env python
"""Step 4_4: compare DiT Step4 vs traditional Step4_3 metrics across model sizes."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import get_traditional_results_dir, load_traditional_config, normalize_split_mode  # noqa: E402
from src.utils.config import load_config, save_config  # noqa: E402
from src.utils.figure_style import apply_publication_figure_style  # noqa: E402
from src.utils.model_scales import get_results_dir  # noqa: E402
from src.utils.reporting import save_artifact_manifest, save_step_summary, write_initial_log  # noqa: E402


VALID_MODEL_SIZES = {"small", "medium", "large", "xl"}
PIPELINE_COLORS = {"DiT": "#E64B35", "Traditional": "#1A80BB", "Tie": "#7A7A7A"}
BAR_COLORS = {
    "DiT_train": "#4DBBD5",
    "DiT_test": "#E64B35",
    "Traditional_train": "#8CC5E3",
    "Traditional_test": "#1A80BB",
}
BAR_EDGE_COLORS = {
    "DiT_train": "#2E8193",
    "DiT_test": "#A93828",
    "Traditional_train": "#5F9DBC",
    "Traditional_test": "#125B84",
}
ROW_BAND_COLORS = {"DiT": "#FDF0EC", "Traditional": "#EEF7FC", "Tie": "#F2F2F2"}
MISSING_INPUT_COLUMNS = ["model_size", "stage", "path", "error"]
REGRESSION_SUMMARY_COLUMNS = [
    "split_mode",
    "n_model_sizes",
    "n_comparison_rows",
    "n_traditional_models",
    "mean_delta_r2_traditional_minus_dit",
    "mean_delta_rmse_traditional_minus_dit",
    "mean_delta_mae_traditional_minus_dit",
    "traditional_wins_r2_count",
    "traditional_wins_rmse_count",
    "traditional_wins_mae_count",
]
CLASSIFICATION_SUMMARY_COLUMNS = [
    "split_mode",
    "n_model_sizes",
    "n_comparison_rows",
    "n_traditional_models",
    "mean_delta_balanced_accuracy_traditional_minus_dit",
    "mean_delta_auroc_traditional_minus_dit",
    "mean_delta_f1_traditional_minus_dit",
    "traditional_wins_balanced_accuracy_count",
    "traditional_wins_auroc_count",
    "traditional_wins_f1_count",
]


def _seed_everything_simple(seed: int) -> Dict[str, object]:
    random.seed(int(seed))
    np.random.seed(int(seed))
    return {"seed": int(seed), "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def _save_run_metadata_simple(output_dir: Path, config_path: str, seed_info: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config_path": str(config_path),
        "seed": int(seed_info.get("seed", 0)),
        "timestamp_utc": str(seed_info.get("timestamp_utc", "")),
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(payload, f, indent=2)


def _load_split_row(csv_path: Path, split_name: str) -> pd.Series:
    if not csv_path.exists():
        raise FileNotFoundError(f"metrics file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "split" not in df.columns:
        raise ValueError(f"Expected 'split' column in {csv_path}")
    split_df = df[df["split"].astype(str).str.lower() == str(split_name).strip().lower()]
    if split_df.empty:
        raise ValueError(f"No {split_name} row in {csv_path}")
    return split_df.iloc[0]


def _load_model_rows_from_summary(summary_csv: Path, *, required_metric_columns: List[str]) -> pd.DataFrame:
    if not summary_csv.exists():
        raise FileNotFoundError(f"model summary file not found: {summary_csv}")
    df = pd.read_csv(summary_csv)
    if df.empty:
        raise ValueError(f"model summary is empty: {summary_csv}")
    if "model_name" not in df.columns:
        raise ValueError(f"Expected 'model_name' column in {summary_csv}")
    missing_metrics = [col for col in required_metric_columns if col not in df.columns]
    if missing_metrics:
        raise ValueError(f"Missing required metric columns in {summary_csv}: {missing_metrics}")

    valid = df.copy()
    for col in required_metric_columns:
        valid[col] = pd.to_numeric(valid[col], errors="coerce")
    valid = valid[valid[required_metric_columns].notna().any(axis=1)].copy()
    if valid.empty:
        raise ValueError(f"No valid metric rows in {summary_csv}")

    if "rank" in valid.columns:
        valid["__rank"] = pd.to_numeric(valid["rank"], errors="coerce")
    else:
        valid["__rank"] = np.nan
    valid["__rank"] = valid["__rank"].fillna(np.inf)
    valid["__model_name_sort"] = valid["model_name"].astype(str).str.lower()
    valid = valid.sort_values(["__rank", "__model_name_sort"]).drop(columns=["__rank", "__model_name_sort"])
    return valid.reset_index(drop=True)


def _safe_get(row: pd.Series, key: str) -> float:
    if key not in row.index:
        return np.nan
    try:
        return float(row[key])
    except Exception:
        return np.nan


def _first_existing_path(candidates: List[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _resolve_model_sizes(args_sizes: List[str], cfg_sizes: List[str]) -> List[str]:
    if args_sizes:
        sizes = [str(s).strip().lower() for s in args_sizes]
    elif cfg_sizes:
        sizes = [str(s).strip().lower() for s in cfg_sizes]
    else:
        sizes = ["small", "medium", "large", "xl"]
    invalid = [s for s in sizes if s not in VALID_MODEL_SIZES]
    if invalid:
        raise ValueError(f"Invalid model size(s): {invalid}. Valid choices: {sorted(VALID_MODEL_SIZES)}")
    return sizes


def _metric_higher_is_better(metric: str) -> bool:
    return str(metric).strip().lower() not in {"rmse", "mae"}


def _metric_direction_note(metric: str) -> str:
    return "Higher is better" if _metric_higher_is_better(metric) else "Lower is better"


def _compute_dit_advantage(dit_vals: np.ndarray, trad_vals: np.ndarray, *, higher_is_better: bool) -> np.ndarray:
    if higher_is_better:
        return dit_vals - trad_vals
    return trad_vals - dit_vals


def _winner_labels_from_advantage(advantage: np.ndarray, tol: float = 1.0e-12) -> np.ndarray:
    winners = np.full(len(advantage), "Tie", dtype=object)
    winners[advantage > tol] = "DiT"
    winners[advantage < -tol] = "Traditional"
    return winners


def _winner_note_text(winner: str, advantage: float) -> str:
    if winner == "Tie" or (not np.isfinite(advantage)):
        return "Tie"
    return f"{winner} +{abs(float(advantage)):.3f}"


def _draw_test_winner_bands(ax: plt.Axes, x_values: np.ndarray, winners: np.ndarray) -> None:
    for xi, winner in zip(x_values, winners):
        ax.axvspan(
            xi - 0.46,
            xi + 0.46,
            color=ROW_BAND_COLORS.get(str(winner), ROW_BAND_COLORS["Tie"]),
            alpha=0.65,
            zorder=0,
        )


def _annotate_bar_values(ax: plt.Axes, bars, *, color: str, font_size: int, pad_fraction: float = 0.02) -> None:
    ymin, ymax = ax.get_ylim()
    y_range = max(float(ymax - ymin), 1.0e-9)
    pad = pad_fraction * y_range
    for bar in bars:
        height = float(bar.get_height())
        if not np.isfinite(height):
            continue
        y = height + pad if height >= 0 else height - pad
        ax.text(
            float(bar.get_x()) + float(bar.get_width()) / 2.0,
            y,
            f"{height:.3f}",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=font_size,
            color=color,
            rotation=90,
        )


def _is_constant_reference(values: np.ndarray, tol: float = 1.0e-12) -> tuple[bool, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False, float("nan")
    ref = float(np.median(finite))
    return bool(np.max(np.abs(finite - ref)) <= tol), ref


def _format_method_display_name(name: str) -> str:
    text = str(name).strip()
    if text.lower() == "dit":
        return "DiT"
    return text.replace("_", " ")


def _apply_axis_text_sizes(ax: plt.Axes, font_size: int, legend=None) -> None:
    ax.xaxis.label.set_size(font_size)
    ax.yaxis.label.set_size(font_size)
    ax.title.set_size(font_size)
    ax.tick_params(axis="both", labelsize=font_size)
    current_legend = legend if legend is not None else ax.get_legend()
    if current_legend is not None:
        title = current_legend.get_title()
        if title is not None:
            title.set_fontsize(font_size)
        for text in current_legend.get_texts():
            text.set_fontsize(font_size)


def _build_method_metric_frame(
    df: pd.DataFrame,
    metric: str,
    *,
    model_size_order: List[str] | None = None,
) -> pd.DataFrame:
    required_cols = [
        "traditional_model_name",
        f"dit_train_{metric}",
        f"dit_{metric}",
        f"traditional_train_{metric}",
        f"traditional_{metric}",
    ]
    if any(col not in df.columns for col in required_cols):
        return pd.DataFrame(columns=["method_label", "method_type", "train_value", "test_value"])

    cols = [c for c in ["model_size", "traditional_rank", "traditional_model_name"] if c in df.columns] + required_cols[1:]
    plot_df = _sort_by_model_size(df[cols], model_size_order=model_size_order)
    include_model_size = "model_size" in plot_df.columns and plot_df["model_size"].astype(str).nunique() > 1
    rows: List[Dict[str, object]] = []

    if "model_size" in plot_df.columns:
        groups = list(plot_df.groupby("model_size", sort=False))
    else:
        groups = [("", plot_df)]

    for model_size, group in groups:
        group = _sort_by_model_size(group, model_size_order=model_size_order)
        if group.empty:
            continue
        first = group.iloc[0]
        dit_label = _format_method_display_name("DiT")
        if include_model_size:
            dit_label = f"{model_size} | {dit_label}"
        rows.append(
            {
                "method_label": dit_label,
                "method_type": "DiT",
                "train_value": _safe_get(first, f"dit_train_{metric}"),
                "test_value": _safe_get(first, f"dit_{metric}"),
            }
        )
        for _, row in group.iterrows():
            method_label = _format_method_display_name(str(row.get("traditional_model_name", "")))
            if include_model_size:
                method_label = f"{model_size} | {method_label}"
            rows.append(
                {
                    "method_label": method_label,
                    "method_type": "Traditional",
                    "train_value": _safe_get(row, f"traditional_train_{metric}"),
                    "test_value": _safe_get(row, f"traditional_{metric}"),
                }
            )
    method_df = pd.DataFrame(rows)
    if method_df.empty:
        return pd.DataFrame(columns=["method_label", "method_type", "train_value", "test_value"])
    method_df["train_value"] = pd.to_numeric(method_df["train_value"], errors="coerce")
    method_df["test_value"] = pd.to_numeric(method_df["test_value"], errors="coerce")
    method_df = method_df.dropna(subset=["train_value", "test_value"], how="all").reset_index(drop=True)
    return method_df


def _grouped_method_bar_axis(
    ax: plt.Axes,
    method_df: pd.DataFrame,
    *,
    ylabel: str,
    font_size: int,
    show_xticklabels: bool,
    annotate_values: bool,
    legend_labels: bool,
) -> None:
    if method_df.empty:
        return
    labels = method_df["method_label"].astype(str).tolist()
    method_types = method_df["method_type"].astype(str).tolist()
    train_vals = pd.to_numeric(method_df["train_value"], errors="coerce").to_numpy(dtype=float)
    test_vals = pd.to_numeric(method_df["test_value"], errors="coerce").to_numpy(dtype=float)
    x = np.arange(len(labels), dtype=float)
    width = 0.34

    for xi, method_type in zip(x, method_types):
        if str(method_type).lower() == "dit":
            ax.axvspan(xi - 0.52, xi + 0.52, color=ROW_BAND_COLORS["DiT"], alpha=0.85, zorder=0)

    train_bars = ax.bar(
        x - width / 2.0,
        train_vals,
        width=width,
        color=BAR_COLORS["Traditional_train"],
        edgecolor=BAR_EDGE_COLORS["Traditional_train"],
        linewidth=0.8,
        label="Train" if legend_labels else None,
        zorder=3,
    )
    test_bars = ax.bar(
        x + width / 2.0,
        test_vals,
        width=width,
        color=BAR_COLORS["Traditional_test"],
        edgecolor=BAR_EDGE_COLORS["Traditional_test"],
        linewidth=0.8,
        label="Test" if legend_labels else None,
        zorder=3,
    )

    for idx, method_type in enumerate(method_types):
        if str(method_type).lower() != "dit":
            continue
        train_bars[idx].set_facecolor(BAR_COLORS["DiT_train"])
        train_bars[idx].set_edgecolor(BAR_EDGE_COLORS["DiT_train"])
        train_bars[idx].set_linewidth(1.4)
        test_bars[idx].set_facecolor(BAR_COLORS["DiT_test"])
        test_bars[idx].set_edgecolor(BAR_EDGE_COLORS["DiT_test"])
        test_bars[idx].set_linewidth(1.4)

    all_vals = np.concatenate([train_vals[np.isfinite(train_vals)], test_vals[np.isfinite(test_vals)]])
    if all_vals.size == 0:
        return
    ymin = min(0.0, float(np.min(all_vals)))
    ymax = float(np.max(all_vals))
    pad = 0.20 * (ymax - ymin) if abs(ymax - ymin) > 1.0e-12 else max(0.12 * max(abs(ymax), 1.0), 0.1)
    ax.set_ylim(ymin - 0.04 * pad, ymax + 1.75 * pad)

    if annotate_values:
        _annotate_bar_values(ax, train_bars, color="#5E6374", font_size=font_size, pad_fraction=0.025)
        _annotate_bar_values(ax, test_bars, color="#7A372A", font_size=font_size, pad_fraction=0.025)

    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Methods")
    ax.set_xticks(x)
    if show_xticklabels:
        ax.set_xticklabels(labels, rotation=24, ha="right")
    else:
        ax.set_xticklabels([])
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45, zorder=0)
    _apply_axis_text_sizes(ax, font_size)


def _sort_by_model_size(df: pd.DataFrame, model_size_order: List[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    sort_cols: List[str] = []
    ascending: List[bool] = []
    temp_cols: List[str] = []

    if model_size_order and "model_size" in out.columns:
        rank = {str(s): i for i, s in enumerate(model_size_order)}
        out["__size_rank"] = out["model_size"].astype(str).map(lambda s: rank.get(str(s), len(rank)))
        sort_cols.append("__size_rank")
        ascending.append(True)
        temp_cols.append("__size_rank")

    if "traditional_rank" in out.columns:
        out["__traditional_rank"] = pd.to_numeric(out["traditional_rank"], errors="coerce").fillna(np.inf)
        sort_cols.append("__traditional_rank")
        ascending.append(True)
        temp_cols.append("__traditional_rank")

    if "traditional_model_name" in out.columns:
        out["__traditional_model_name"] = out["traditional_model_name"].astype(str).str.lower()
        sort_cols.append("__traditional_model_name")
        ascending.append(True)
        temp_cols.append("__traditional_model_name")

    if sort_cols:
        out = out.sort_values(sort_cols, ascending=ascending)
    if temp_cols:
        out = out.drop(columns=temp_cols)
    return out.reset_index(drop=True)


def _add_comparison_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        out["comparison_label"] = pd.Series(dtype=object)
        return out
    if "traditional_model_name" not in out.columns:
        out["comparison_label"] = out["model_size"].astype(str) if "model_size" in out.columns else out.index.astype(str)
        return out
    model_label = out["traditional_model_name"].astype(str).map(_format_method_display_name)
    include_model_size = "model_size" in out.columns and out["model_size"].astype(str).nunique() > 1
    if include_model_size:
        out["comparison_label"] = out["model_size"].astype(str) + " | " + model_label
    else:
        out["comparison_label"] = model_label
    return out


def _barplot_two_models(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    out_png: Path,
    dpi: int,
    font_size: int,
    model_size_order: List[str] | None = None,
) -> None:
    if df.empty:
        return
    method_df = _build_method_metric_frame(df, metric, model_size_order=model_size_order)
    if method_df.empty:
        return

    apply_publication_figure_style(font_size=font_size, dpi=dpi, remove_titles=True)
    fig_width = max(10.5, 1.28 * len(method_df) + 3.8)
    fig, ax = plt.subplots(figsize=(fig_width, 6.8))
    _grouped_method_bar_axis(
        ax,
        method_df,
        ylabel=ylabel,
        font_size=font_size,
        show_xticklabels=True,
        annotate_values=True,
        legend_labels=True,
    )
    ax.set_title(title, loc="left", fontsize=font_size)
    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2, frameon=False, fontsize=font_size)
    _apply_axis_text_sizes(ax, font_size, legend=legend)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.26, top=0.84)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def _overview_panel_two_models(
    df: pd.DataFrame,
    metric_specs: List[tuple[str, str]],
    title: str,
    out_png: Path,
    dpi: int,
    font_size: int,
    model_size_order: List[str] | None = None,
) -> None:
    if df.empty or len(metric_specs) == 0:
        return
    apply_publication_figure_style(font_size=font_size, dpi=dpi, remove_titles=True)
    max_methods = 0
    method_frames: List[pd.DataFrame] = []
    for metric, _ in metric_specs:
        method_df = _build_method_metric_frame(df, metric, model_size_order=model_size_order)
        method_frames.append(method_df)
        max_methods = max(max_methods, len(method_df))
    if max_methods == 0:
        return

    fig_width = max(16.0, 4.9 * len(metric_specs) + 1.15 * max_methods)
    fig, axes = plt.subplots(1, len(metric_specs), figsize=(fig_width, 7.4), sharey=False)
    if len(metric_specs) == 1:
        axes = [axes]
    legend_handles = None
    legend_labels = None
    for idx, (ax, (metric, ylabel), method_df) in enumerate(zip(axes, metric_specs, method_frames)):
        if method_df.empty:
            ax.set_axis_off()
            continue
        _grouped_method_bar_axis(
            ax,
            method_df,
            ylabel=ylabel,
            font_size=font_size,
            show_xticklabels=True,
            annotate_values=True,
            legend_labels=(idx == 0),
        )
        ax.set_title(ylabel, fontsize=font_size, pad=8)
        if idx == 0:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        _apply_axis_text_sizes(ax, font_size)
    if legend_handles and legend_labels:
        legend = fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=2,
            frameon=False,
            fontsize=font_size,
        )
        legend.get_title().set_fontsize(font_size)
    fig.suptitle(title, y=0.985, fontsize=font_size)
    fig.subplots_adjust(left=0.05, right=0.995, bottom=0.28, top=0.80, wspace=0.22)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def _delta_heatmap(
    df: pd.DataFrame,
    delta_specs: List[tuple[str, str, str, str, str, bool]],
    title: str,
    out_png: Path,
    dpi: int,
    font_size: int,
    model_size_order: List[str] | None = None,
) -> None:
    if df.empty or len(delta_specs) == 0:
        return
    cols = [c for c in ["model_size", "traditional_rank", "traditional_model_name", "comparison_label"] if c in df.columns]
    for metric_label, traditional_train_col, traditional_test_col, dit_train_col, dit_test_col, _ in delta_specs:
        cols.extend([traditional_train_col, traditional_test_col, dit_train_col, dit_test_col])
    for c in cols:
        if c not in df.columns:
            return
    plot_df = _sort_by_model_size(df[cols], model_size_order=model_size_order)
    plot_df = _add_comparison_labels(plot_df)
    method_labels = plot_df["comparison_label"].astype(str).tolist()
    row_labels: List[str] = []
    row_values: List[np.ndarray] = []
    for metric_label, traditional_train_col, traditional_test_col, dit_train_col, dit_test_col, higher_is_better in delta_specs:
        traditional_train = pd.to_numeric(plot_df[traditional_train_col], errors="coerce").to_numpy(dtype=float)
        traditional_test = pd.to_numeric(plot_df[traditional_test_col], errors="coerce").to_numpy(dtype=float)
        dit_train = pd.to_numeric(plot_df[dit_train_col], errors="coerce").to_numpy(dtype=float)
        dit_test = pd.to_numeric(plot_df[dit_test_col], errors="coerce").to_numpy(dtype=float)
        row_labels.append(f"Train {metric_label}")
        row_values.append(_compute_dit_advantage(dit_train, traditional_train, higher_is_better=higher_is_better))
        row_labels.append(f"Test {metric_label}")
        row_values.append(_compute_dit_advantage(dit_test, traditional_test, higher_is_better=higher_is_better))
    heat = pd.DataFrame(row_values, index=row_labels, columns=method_labels)
    if heat.empty:
        return
    arr = heat.to_numpy(dtype=float)
    finite_vals = arr[np.isfinite(arr)]
    vmax = float(np.max(np.abs(finite_vals))) if finite_vals.size > 0 else 1.0
    if (not np.isfinite(vmax)) or vmax < 1.0e-8:
        vmax = 1.0
    annot = heat.apply(lambda col: col.map(lambda v: "" if pd.isna(v) else f"{float(v):+0.3f}"))

    apply_publication_figure_style(font_size=font_size, dpi=dpi, remove_titles=True)
    fig_width = max(11.5, 1.85 * heat.shape[1] + 3.4)
    fig_height = max(6.2, 1.10 * heat.shape[0] + 2.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        heat,
        cmap=sns.blend_palette(
            [PIPELINE_COLORS["Traditional"], "#f7f7f7", PIPELINE_COLORS["DiT"]],
            as_cmap=True,
        ),
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        annot=annot,
        fmt="",
        annot_kws={"fontsize": font_size, "color": "#111111", "fontweight": "semibold"},
        linewidths=1.1,
        linecolor="#d9dee7",
        mask=heat.isna(),
        cbar_kws={"label": "DiT advantage (+ favors DiT)"},
        ax=ax,
    )
    ax.set_xlabel("Methods", fontsize=font_size)
    ax.set_ylabel("Metrics", fontsize=font_size)
    ax.set_title(title, loc="left", fontsize=font_size)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=24, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    _apply_axis_text_sizes(ax, font_size)
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.set_label("DiT advantage (+ favors DiT)", size=font_size)
        cbar.ax.tick_params(labelsize=font_size)
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.23, top=0.90)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def _winner_count_figure(reg_df: pd.DataFrame, cls_df: pd.DataFrame, out_png: Path, dpi: int, font_size: int) -> None:
    rows: List[Dict[str, object]] = []
    specs = [
        (reg_df, "Regression", "winner_r2", "R2"),
        (reg_df, "Regression", "winner_rmse", "RMSE"),
        (reg_df, "Regression", "winner_mae", "MAE"),
        (cls_df, "Classification", "winner_balanced_accuracy", "Balanced accuracy"),
        (cls_df, "Classification", "winner_auroc", "AUROC"),
        (cls_df, "Classification", "winner_f1", "F1"),
    ]
    for df, task, winner_col, metric_name in specs:
        if df.empty or winner_col not in df.columns:
            continue
        vc = df[winner_col].astype(str).str.strip().str.lower().value_counts()
        rows.append({"metric": f"{task}: {metric_name}", "winner": "DiT", "count": int(vc.get("dit", 0))})
        rows.append({"metric": f"{task}: {metric_name}", "winner": "Traditional", "count": int(vc.get("traditional", 0))})
    if len(rows) == 0:
        return
    plot_df = pd.DataFrame(rows)
    metric_order = [
        "Regression: R2",
        "Regression: RMSE",
        "Regression: MAE",
        "Classification: Balanced accuracy",
        "Classification: AUROC",
        "Classification: F1",
    ]
    plot_df["metric"] = pd.Categorical(plot_df["metric"], categories=metric_order, ordered=True)
    plot_df = plot_df.sort_values(["metric", "winner"]).reset_index(drop=True)

    apply_publication_figure_style(font_size=font_size, dpi=dpi, remove_titles=True)
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    sns.barplot(
        data=plot_df,
        y="metric",
        x="count",
        hue="winner",
        palette={"DiT": PIPELINE_COLORS["DiT"], "Traditional": PIPELINE_COLORS["Traditional"]},
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3, fontsize=font_size)
    ax.set_xlabel("Winning model pairs", fontsize=font_size)
    ax.set_ylabel("", fontsize=font_size)
    ax.set_title("Winner counts by metric", loc="left", fontsize=font_size)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.45)
    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=font_size)
    _apply_axis_text_sizes(ax, font_size, legend=legend)
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def _missing_inputs_figure(missing_df: pd.DataFrame, out_png: Path, dpi: int, font_size: int) -> None:
    if missing_df.empty:
        return
    plot_df = (
        missing_df.groupby(["stage", "model_size"], as_index=False)
        .size()
        .rename(columns={"size": "n_missing"})
        .sort_values(["stage", "model_size"])
        .reset_index(drop=True)
    )
    apply_publication_figure_style(font_size=font_size, dpi=dpi, remove_titles=True)
    fig, ax = plt.subplots(figsize=(11, 5.4))
    sns.barplot(data=plot_df, x="stage", y="n_missing", hue="model_size", ax=ax)
    ax.set_xlabel("Missing/invalid input stage", fontsize=font_size)
    ax.set_ylabel("Count", fontsize=font_size)
    ax.set_title("Step4_4 missing/invalid inputs by stage", fontsize=font_size)
    ax.tick_params(axis="x", rotation=20, labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)
    legend = ax.legend(title="Model size", loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=font_size)
    _apply_axis_text_sizes(ax, font_size, legend=legend)
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def _no_data_notice_figure(
    split_mode: str,
    model_sizes: List[str],
    missing_df: pd.DataFrame,
    out_png: Path,
    dpi: int,
    font_size: int,
) -> None:
    apply_publication_figure_style(font_size=font_size, dpi=dpi, remove_titles=True)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axis("off")
    lines = [
        "Step4_4 comparison: no valid paired metrics available to plot.",
        f"Split mode: {split_mode}",
        f"Requested model sizes: {', '.join(model_sizes)}",
        f"Missing/invalid input rows: {int(len(missing_df))}",
        "See metrics/missing_or_invalid_inputs.csv for exact missing paths and errors.",
    ]
    ax.text(0.02, 0.95, "\n".join(lines), va="top", ha="left", fontsize=font_size)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 4_4: compare DiT Step4 vs traditional Step4_3")
    parser.add_argument("--config", type=str, default="traditional_step4/configs/config_traditional.yaml")
    parser.add_argument(
        "--split_mode",
        type=str,
        required=True,
        choices=["polymer", "random"],
        help="Comparison split mode namespace.",
    )
    parser.add_argument(
        "--model_sizes",
        type=str,
        nargs="*",
        default=None,
        help="Optional model-size overrides. Defaults to config step4_4_comparation.model_sizes.",
    )
    parser.add_argument(
        "--dit_config",
        type=str,
        default=None,
        help="Optional path to DiT config (fallback: config_traditional.paths.dit_config_path or configs/config.yaml).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override (default: config data.random_seed or 42).")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    split_mode = normalize_split_mode(args.split_mode)

    config = load_traditional_config(args.config)
    trad_cfg = config.get("traditional_step4", {})
    if not isinstance(trad_cfg, dict):
        raise ValueError("Missing 'traditional_step4' section in config_traditional.yaml")
    compare_cfg = trad_cfg.get("step4_4_comparation", {})
    if not isinstance(compare_cfg, dict):
        compare_cfg = {}

    paths_cfg = config.get("paths", {})
    results_root = str(paths_cfg.get("results_root", "traditional_step4"))
    default_dit_config = str(paths_cfg.get("dit_config_path", "configs/config.yaml"))
    dit_config_path = args.dit_config or default_dit_config
    dit_config = load_config(dit_config_path)
    dit_results_base = str(dit_config.get("paths", {}).get("results_dir", "results"))
    data_cfg = config.get("data", {}) if isinstance(config.get("data", {}), dict) else {}
    seed = int(args.seed if args.seed is not None else data_cfg.get("random_seed", 42))

    model_sizes = _resolve_model_sizes(args.model_sizes or [], compare_cfg.get("model_sizes", []))

    output_dir = Path(results_root) / f"results_4_4_comparation_{split_mode}"
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    regression_figures_dir = figures_dir / "regression"
    classification_figures_dir = figures_dir / "classification"
    shared_figures_dir = figures_dir / "shared"
    for d in [output_dir, metrics_dir, figures_dir, regression_figures_dir, classification_figures_dir, shared_figures_dir]:
        d.mkdir(parents=True, exist_ok=True)
    for png_path in figures_dir.rglob("*.png"):
        if png_path.is_file():
            png_path.unlink()

    seed_info = _seed_everything_simple(seed)
    effective_config = copy.deepcopy(config)
    effective_data_cfg = effective_config.setdefault("data", {})
    if isinstance(effective_data_cfg, dict):
        effective_data_cfg["random_seed"] = int(seed)
    effective_paths_cfg = effective_config.setdefault("paths", {})
    if isinstance(effective_paths_cfg, dict):
        effective_paths_cfg["dit_config_path"] = str(dit_config_path)
    effective_trad_cfg = effective_config.setdefault("traditional_step4", {})
    if not isinstance(effective_trad_cfg, dict):
        effective_trad_cfg = {}
        effective_config["traditional_step4"] = effective_trad_cfg
    effective_shared_cfg = effective_trad_cfg.setdefault("shared", {})
    if isinstance(effective_shared_cfg, dict):
        effective_shared_cfg["split_mode"] = str(split_mode)
    effective_compare_cfg = effective_trad_cfg.setdefault("step4_4_comparation", {})
    if isinstance(effective_compare_cfg, dict):
        effective_compare_cfg["model_sizes"] = list(model_sizes)

    save_config(effective_config, output_dir / "config_used.yaml")
    _save_run_metadata_simple(output_dir, args.config, seed_info)
    write_initial_log(
        step_dir=output_dir,
        step_name="step4_4_comparation",
        context={
            "split_mode": split_mode,
            "model_sizes": ",".join(model_sizes),
            "dit_config_path": str(dit_config_path),
            "dit_results_base_dir": str(dit_results_base),
            "results_root": str(results_root),
        },
    )

    missing_rows: List[Dict[str, str]] = []
    reg_rows: List[Dict[str, object]] = []
    cls_rows: List[Dict[str, object]] = []

    for model_size in model_sizes:
        dit_results_dir_split = Path(get_results_dir(model_size=model_size, base_dir=dit_results_base, split_mode=split_mode))
        dit_results_dir_root = Path(get_results_dir(model_size=model_size, base_dir=dit_results_base, split_mode=None))
        trad_results_dir = get_traditional_results_dir(results_root=results_root)
        trad_results_dir_legacy = get_traditional_results_dir(results_root=results_root, split_mode=split_mode)

        dit_reg_csv = _first_existing_path(
            [
                dit_results_dir_split / "step4_1_regression" / split_mode / "metrics" / "chi_metrics_overall.csv",
                dit_results_dir_root / "step4_1_regression" / split_mode / "metrics" / "chi_metrics_overall.csv",
                dit_results_dir_root / "step4_chi_training" / "step4_1_regression" / split_mode / "metrics" / "chi_metrics_overall.csv",
                dit_results_dir_split / "step4_chi_training" / split_mode / "step4_1_regression" / "metrics" / "chi_metrics_overall.csv",
            ]
        )
        dit_cls_csv = _first_existing_path(
            [
                dit_results_dir_split / "step4_2_classification" / "metrics" / "class_metrics_overall.csv",
                dit_results_dir_root / "step4_2_classification" / "metrics" / "class_metrics_overall.csv",
                dit_results_dir_root / "step4_chi_training" / "step4_2_classification" / "metrics" / "class_metrics_overall.csv",
                dit_results_dir_split / "step4_chi_training" / split_mode / "step4_2_classification" / "metrics" / "class_metrics_overall.csv",
            ]
        )
        trad_reg_summary_csv = _first_existing_path(
            [
                (
                    trad_results_dir
                    / "step4_3_traditional"
                    / "step4_3_1_regression"
                    / split_mode
                    / "metrics"
                    / "model_metrics_summary.csv"
                ),
                (
                    trad_results_dir_legacy
                    / "step4_3_traditional"
                    / split_mode
                    / "step4_3_1_regression"
                    / "metrics"
                    / "model_metrics_summary.csv"
                ),
            ]
        )
        trad_cls_summary_csv = _first_existing_path(
            [
                (
                    trad_results_dir
                    / "step4_3_traditional"
                    / "step4_3_2_classification"
                    / "metrics"
                    / "model_metrics_summary.csv"
                ),
                (
                    trad_results_dir_legacy
                    / "step4_3_traditional"
                    / split_mode
                    / "step4_3_2_classification"
                    / "metrics"
                    / "model_metrics_summary.csv"
                ),
            ]
        )

        reg_ready = True
        try:
            dit_reg_train = _load_split_row(dit_reg_csv, "train")
            dit_reg_test = _load_split_row(dit_reg_csv, "test")
        except Exception as exc:
            reg_ready = False
            missing_rows.append({"model_size": model_size, "stage": "step4_1_regression", "path": str(dit_reg_csv), "error": str(exc)})
        try:
            trad_reg_df = _load_model_rows_from_summary(
                trad_reg_summary_csv,
                required_metric_columns=["test_r2", "test_rmse", "test_mae"],
            )
        except Exception as exc:
            reg_ready = False
            missing_rows.append(
                {
                    "model_size": model_size,
                    "stage": "step4_3_1_regression_best_model_selection",
                    "path": str(trad_reg_summary_csv),
                    "error": str(exc),
                }
            )

        if reg_ready:
            for _, trad_reg in trad_reg_df.iterrows():
                row = {
                    "model_size": model_size,
                    "traditional_model_name": str(trad_reg.get("model_name", "")),
                    "traditional_rank": _safe_get(trad_reg, "rank"),
                    "traditional_param_source": str(trad_reg.get("param_source", "")),
                    "traditional_quality_status": str(trad_reg.get("quality_status", "")),
                    "traditional_metrics_dir": str(trad_reg.get("metrics_dir", "")),
                    "traditional_figures_dir": str(trad_reg.get("figures_dir", "")),
                    "traditional_checkpoint": str(trad_reg.get("checkpoint", "")),
                    "traditional_tuning_dir": str(trad_reg.get("tuning_dir", "")),
                    "dit_train_r2": _safe_get(dit_reg_train, "r2"),
                    "dit_train_rmse": _safe_get(dit_reg_train, "rmse"),
                    "dit_train_mae": _safe_get(dit_reg_train, "mae"),
                    "dit_r2": _safe_get(dit_reg_test, "r2"),
                    "dit_rmse": _safe_get(dit_reg_test, "rmse"),
                    "dit_mae": _safe_get(dit_reg_test, "mae"),
                    "traditional_train_r2": _safe_get(trad_reg, "train_r2"),
                    "traditional_train_rmse": _safe_get(trad_reg, "train_rmse"),
                    "traditional_train_mae": _safe_get(trad_reg, "train_mae"),
                    "traditional_r2": _safe_get(trad_reg, "test_r2"),
                    "traditional_rmse": _safe_get(trad_reg, "test_rmse"),
                    "traditional_mae": _safe_get(trad_reg, "test_mae"),
                }
                row["delta_r2_traditional_minus_dit"] = row["traditional_r2"] - row["dit_r2"]
                row["delta_rmse_traditional_minus_dit"] = row["traditional_rmse"] - row["dit_rmse"]
                row["delta_mae_traditional_minus_dit"] = row["traditional_mae"] - row["dit_mae"]
                row["winner_r2"] = "traditional" if row["delta_r2_traditional_minus_dit"] > 0 else "dit"
                row["winner_rmse"] = "traditional" if row["delta_rmse_traditional_minus_dit"] < 0 else "dit"
                row["winner_mae"] = "traditional" if row["delta_mae_traditional_minus_dit"] < 0 else "dit"
                reg_rows.append(row)

        cls_ready = True
        try:
            dit_cls_train = _load_split_row(dit_cls_csv, "train")
            dit_cls_test = _load_split_row(dit_cls_csv, "test")
        except Exception as exc:
            cls_ready = False
            missing_rows.append({"model_size": model_size, "stage": "step4_2_classification", "path": str(dit_cls_csv), "error": str(exc)})
        try:
            trad_cls_df = _load_model_rows_from_summary(
                trad_cls_summary_csv,
                required_metric_columns=["test_balanced_accuracy", "test_auroc", "test_f1"],
            )
        except Exception as exc:
            cls_ready = False
            missing_rows.append(
                {
                    "model_size": model_size,
                    "stage": "step4_3_2_classification_best_model_selection",
                    "path": str(trad_cls_summary_csv),
                    "error": str(exc),
                }
            )

        if cls_ready:
            for _, trad_cls in trad_cls_df.iterrows():
                row = {
                    "model_size": model_size,
                    "traditional_model_name": str(trad_cls.get("model_name", "")),
                    "traditional_rank": _safe_get(trad_cls, "rank"),
                    "traditional_param_source": str(trad_cls.get("param_source", "")),
                    "traditional_quality_status": str(trad_cls.get("quality_status", "")),
                    "traditional_metrics_dir": str(trad_cls.get("metrics_dir", "")),
                    "traditional_figures_dir": str(trad_cls.get("figures_dir", "")),
                    "traditional_checkpoint": str(trad_cls.get("checkpoint", "")),
                    "traditional_tuning_dir": str(trad_cls.get("tuning_dir", "")),
                    "dit_train_balanced_accuracy": _safe_get(dit_cls_train, "balanced_accuracy"),
                    "dit_train_auroc": _safe_get(dit_cls_train, "auroc"),
                    "dit_train_f1": _safe_get(dit_cls_train, "f1"),
                    "dit_balanced_accuracy": _safe_get(dit_cls_test, "balanced_accuracy"),
                    "dit_auroc": _safe_get(dit_cls_test, "auroc"),
                    "dit_f1": _safe_get(dit_cls_test, "f1"),
                    "traditional_train_balanced_accuracy": _safe_get(trad_cls, "train_balanced_accuracy"),
                    "traditional_train_auroc": _safe_get(trad_cls, "train_auroc"),
                    "traditional_train_f1": _safe_get(trad_cls, "train_f1"),
                    "traditional_balanced_accuracy": _safe_get(trad_cls, "test_balanced_accuracy"),
                    "traditional_auroc": _safe_get(trad_cls, "test_auroc"),
                    "traditional_f1": _safe_get(trad_cls, "test_f1"),
                }
                row["delta_balanced_accuracy_traditional_minus_dit"] = row["traditional_balanced_accuracy"] - row["dit_balanced_accuracy"]
                row["delta_auroc_traditional_minus_dit"] = row["traditional_auroc"] - row["dit_auroc"]
                row["delta_f1_traditional_minus_dit"] = row["traditional_f1"] - row["dit_f1"]
                row["winner_balanced_accuracy"] = "traditional" if row["delta_balanced_accuracy_traditional_minus_dit"] > 0 else "dit"
                row["winner_auroc"] = "traditional" if row["delta_auroc_traditional_minus_dit"] > 0 else "dit"
                row["winner_f1"] = "traditional" if row["delta_f1_traditional_minus_dit"] > 0 else "dit"
                cls_rows.append(row)

    missing_df = pd.DataFrame(missing_rows, columns=MISSING_INPUT_COLUMNS)
    missing_df.to_csv(metrics_dir / "missing_or_invalid_inputs.csv", index=False)

    reg_columns = [
        "model_size",
        "traditional_model_name",
        "traditional_rank",
        "traditional_param_source",
        "traditional_quality_status",
        "traditional_metrics_dir",
        "traditional_figures_dir",
        "traditional_checkpoint",
        "traditional_tuning_dir",
        "dit_train_r2",
        "dit_train_rmse",
        "dit_train_mae",
        "dit_r2",
        "dit_rmse",
        "dit_mae",
        "traditional_train_r2",
        "traditional_train_rmse",
        "traditional_train_mae",
        "traditional_r2",
        "traditional_rmse",
        "traditional_mae",
        "delta_r2_traditional_minus_dit",
        "delta_rmse_traditional_minus_dit",
        "delta_mae_traditional_minus_dit",
        "winner_r2",
        "winner_rmse",
        "winner_mae",
    ]
    cls_columns = [
        "model_size",
        "traditional_model_name",
        "traditional_rank",
        "traditional_param_source",
        "traditional_quality_status",
        "traditional_metrics_dir",
        "traditional_figures_dir",
        "traditional_checkpoint",
        "traditional_tuning_dir",
        "dit_train_balanced_accuracy",
        "dit_train_auroc",
        "dit_train_f1",
        "dit_balanced_accuracy",
        "dit_auroc",
        "dit_f1",
        "traditional_train_balanced_accuracy",
        "traditional_train_auroc",
        "traditional_train_f1",
        "traditional_balanced_accuracy",
        "traditional_auroc",
        "traditional_f1",
        "delta_balanced_accuracy_traditional_minus_dit",
        "delta_auroc_traditional_minus_dit",
        "delta_f1_traditional_minus_dit",
        "winner_balanced_accuracy",
        "winner_auroc",
        "winner_f1",
    ]
    if reg_rows:
        reg_df = _sort_by_model_size(pd.DataFrame(reg_rows), model_size_order=model_sizes)
        reg_df = _add_comparison_labels(reg_df)
    else:
        reg_df = pd.DataFrame(columns=reg_columns + ["comparison_label"])
    if cls_rows:
        cls_df = _sort_by_model_size(pd.DataFrame(cls_rows), model_size_order=model_sizes)
        cls_df = _add_comparison_labels(cls_df)
    else:
        cls_df = pd.DataFrame(columns=cls_columns + ["comparison_label"])
    reg_df.to_csv(metrics_dir / "regression_model_size_comparison.csv", index=False)
    cls_df.to_csv(metrics_dir / "classification_model_size_comparison.csv", index=False)

    if not reg_df.empty:
        reg_summary = pd.DataFrame(
            [
                {
                    "split_mode": split_mode,
                    "n_model_sizes": int(reg_df["model_size"].astype(str).nunique()),
                    "n_comparison_rows": int(len(reg_df)),
                    "n_traditional_models": int(reg_df["traditional_model_name"].astype(str).nunique()),
                    "mean_delta_r2_traditional_minus_dit": float(reg_df["delta_r2_traditional_minus_dit"].mean()),
                    "mean_delta_rmse_traditional_minus_dit": float(reg_df["delta_rmse_traditional_minus_dit"].mean()),
                    "mean_delta_mae_traditional_minus_dit": float(reg_df["delta_mae_traditional_minus_dit"].mean()),
                    "traditional_wins_r2_count": int((reg_df["winner_r2"] == "traditional").sum()),
                    "traditional_wins_rmse_count": int((reg_df["winner_rmse"] == "traditional").sum()),
                    "traditional_wins_mae_count": int((reg_df["winner_mae"] == "traditional").sum()),
                }
            ],
            columns=REGRESSION_SUMMARY_COLUMNS,
        )
        reg_summary.to_csv(metrics_dir / "regression_comparation_summary.csv", index=False)
    else:
        pd.DataFrame(columns=REGRESSION_SUMMARY_COLUMNS).to_csv(metrics_dir / "regression_comparation_summary.csv", index=False)

    if not cls_df.empty:
        cls_summary = pd.DataFrame(
            [
                {
                    "split_mode": split_mode,
                    "n_model_sizes": int(cls_df["model_size"].astype(str).nunique()),
                    "n_comparison_rows": int(len(cls_df)),
                    "n_traditional_models": int(cls_df["traditional_model_name"].astype(str).nunique()),
                    "mean_delta_balanced_accuracy_traditional_minus_dit": float(
                        cls_df["delta_balanced_accuracy_traditional_minus_dit"].mean()
                    ),
                    "mean_delta_auroc_traditional_minus_dit": float(cls_df["delta_auroc_traditional_minus_dit"].mean()),
                    "mean_delta_f1_traditional_minus_dit": float(cls_df["delta_f1_traditional_minus_dit"].mean()),
                    "traditional_wins_balanced_accuracy_count": int((cls_df["winner_balanced_accuracy"] == "traditional").sum()),
                    "traditional_wins_auroc_count": int((cls_df["winner_auroc"] == "traditional").sum()),
                    "traditional_wins_f1_count": int((cls_df["winner_f1"] == "traditional").sum()),
                }
            ],
            columns=CLASSIFICATION_SUMMARY_COLUMNS,
        )
        cls_summary.to_csv(metrics_dir / "classification_comparation_summary.csv", index=False)
    else:
        pd.DataFrame(columns=CLASSIFICATION_SUMMARY_COLUMNS).to_csv(
            metrics_dir / "classification_comparation_summary.csv", index=False
        )

    dpi = int(config.get("plotting", {}).get("dpi", 600))
    font_size = int(config.get("plotting", {}).get("font_size", 16))
    if not reg_df.empty:
        _barplot_two_models(
            df=reg_df,
            metric="r2",
            ylabel="R2",
            title=f"Regression R2 ({split_mode})",
            out_png=regression_figures_dir / "regression_r2_by_method.png",
            dpi=dpi,
            font_size=font_size,
            model_size_order=model_sizes,
        )
        _barplot_two_models(
            df=reg_df,
            metric="rmse",
            ylabel="RMSE",
            title=f"Regression RMSE ({split_mode})",
            out_png=regression_figures_dir / "regression_rmse_by_method.png",
            dpi=dpi,
            font_size=font_size,
            model_size_order=model_sizes,
        )
        _barplot_two_models(
            df=reg_df,
            metric="mae",
            ylabel="MAE",
            title=f"Regression MAE ({split_mode})",
            out_png=regression_figures_dir / "regression_mae_by_method.png",
            dpi=dpi,
            font_size=font_size,
            model_size_order=model_sizes,
        )
        _overview_panel_two_models(
            df=reg_df,
            metric_specs=[("r2", "R2"), ("rmse", "RMSE"), ("mae", "MAE")],
            title=f"Regression Overview ({split_mode})",
            out_png=regression_figures_dir / "regression_overview_panel.png",
            dpi=dpi,
            font_size=font_size,
            model_size_order=model_sizes,
        )
        _delta_heatmap(
            df=reg_df,
            delta_specs=[
                ("R2", "traditional_train_r2", "traditional_r2", "dit_train_r2", "dit_r2", True),
                ("RMSE", "traditional_train_rmse", "traditional_rmse", "dit_train_rmse", "dit_rmse", False),
                ("MAE", "traditional_train_mae", "traditional_mae", "dit_train_mae", "dit_mae", False),
            ],
            title=f"Regression Delta to DiT ({split_mode})",
            out_png=regression_figures_dir / "regression_delta_heatmap.png",
            dpi=dpi,
            font_size=font_size,
            model_size_order=model_sizes,
        )

    if not cls_df.empty:
        _barplot_two_models(
            df=cls_df,
            metric="balanced_accuracy",
            ylabel="Balanced accuracy",
            title=f"Classification Balanced Accuracy ({split_mode})",
            out_png=classification_figures_dir / "classification_balanced_accuracy_by_method.png",
            dpi=dpi,
            font_size=font_size,
            model_size_order=model_sizes,
        )
        _barplot_two_models(
            df=cls_df,
            metric="auroc",
            ylabel="AUROC",
            title=f"Classification AUROC ({split_mode})",
            out_png=classification_figures_dir / "classification_auroc_by_method.png",
            dpi=dpi,
            font_size=font_size,
            model_size_order=model_sizes,
        )
        _barplot_two_models(
            df=cls_df,
            metric="f1",
            ylabel="F1",
            title=f"Classification F1 ({split_mode})",
            out_png=classification_figures_dir / "classification_f1_by_method.png",
            dpi=dpi,
            font_size=font_size,
            model_size_order=model_sizes,
        )
        _overview_panel_two_models(
            df=cls_df,
            metric_specs=[
                ("balanced_accuracy", "Balanced accuracy"),
                ("auroc", "AUROC"),
                ("f1", "F1"),
            ],
            title=f"Classification Overview ({split_mode})",
            out_png=classification_figures_dir / "classification_overview_panel.png",
            dpi=dpi,
            font_size=font_size,
            model_size_order=model_sizes,
        )
        _delta_heatmap(
            df=cls_df,
            delta_specs=[
                (
                    "Balanced acc.",
                    "traditional_train_balanced_accuracy",
                    "traditional_balanced_accuracy",
                    "dit_train_balanced_accuracy",
                    "dit_balanced_accuracy",
                    True,
                ),
                ("AUROC", "traditional_train_auroc", "traditional_auroc", "dit_train_auroc", "dit_auroc", True),
                ("F1", "traditional_train_f1", "traditional_f1", "dit_train_f1", "dit_f1", True),
            ],
            title=f"Classification Delta to DiT ({split_mode})",
            out_png=classification_figures_dir / "classification_delta_heatmap.png",
            dpi=dpi,
            font_size=font_size,
            model_size_order=model_sizes,
        )

    _winner_count_figure(
        reg_df=reg_df,
        cls_df=cls_df,
        out_png=shared_figures_dir / "winner_counts_by_metric.png",
        dpi=dpi,
        font_size=font_size,
    )
    _missing_inputs_figure(
        missing_df=missing_df,
        out_png=shared_figures_dir / "missing_inputs_by_stage.png",
        dpi=dpi,
        font_size=font_size,
    )
    if reg_df.empty and cls_df.empty:
        _no_data_notice_figure(
            split_mode=split_mode,
            model_sizes=model_sizes,
            missing_df=missing_df,
            out_png=shared_figures_dir / "comparison_no_valid_data_notice.png",
            dpi=dpi,
            font_size=font_size,
        )

    save_artifact_manifest(step_dir=output_dir, metrics_dir=metrics_dir, figures_dir=figures_dir, dpi=dpi)
    generated_figures = sorted(str(p.relative_to(figures_dir)) for p in figures_dir.rglob("*.png") if p.is_file())
    payload = {
        "split_mode": split_mode,
        "model_sizes": model_sizes,
        "n_regression_rows": int(len(reg_df)),
        "n_regression_models": int(reg_df["traditional_model_name"].astype(str).nunique()) if "traditional_model_name" in reg_df.columns else 0,
        "n_classification_rows": int(len(cls_df)),
        "n_classification_models": int(cls_df["traditional_model_name"].astype(str).nunique()) if "traditional_model_name" in cls_df.columns else 0,
        "n_missing_or_invalid": int(len(missing_df)),
        "n_figures_generated": int(len(generated_figures)),
        "figure_files": generated_figures,
    }
    with open(metrics_dir / "comparation_run_summary.json", "w") as f:
        json.dump(payload, f, indent=2)
    save_step_summary(
        {
            "step": "step4_4_comparation",
            "split_mode": split_mode,
            "model_sizes": ",".join(model_sizes),
            "output_dir": str(output_dir),
            "n_regression_rows": int(len(reg_df)),
            "n_regression_models": int(reg_df["traditional_model_name"].astype(str).nunique()) if "traditional_model_name" in reg_df.columns else 0,
            "n_classification_rows": int(len(cls_df)),
            "n_classification_models": int(cls_df["traditional_model_name"].astype(str).nunique()) if "traditional_model_name" in cls_df.columns else 0,
            "n_missing_or_invalid": int(len(missing_df)),
            "n_figures_generated": int(len(generated_figures)),
        },
        metrics_dir=metrics_dir,
    )
    print(f"Step4_4 comparation outputs: {output_dir}")


if __name__ == "__main__":
    main()
