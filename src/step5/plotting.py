"""Shared plotting helpers for Step 5 and Step 5_1."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd

from src.utils.figure_style import apply_publication_figure_style, save_publication_figure


_GRID_COLOR = "#D8DEE4"
_TEXT_COLOR = "#24292F"
_EDGE_COLOR = "#2A2A2A"
_STEP5_COLORS = [
    "#0072B2",
    "#009E73",
    "#E69F00",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#882255",
    "#44AA99",
    "#999933",
    "#332288",
]
_RUN_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "h", "<", ">"]
_RUN_COLOR_RULES = [
    (("s4_dpo", "s4d"), "#CC79A7"),
    (("s4_rl", "s4r"), "#56B4E9"),
    (("s4_ppo", "s4p"), "#882255"),
    (("s4_grpo", "s4g"), "#44AA99"),
    (("s0",), "#0072B2"),
    (("s1",), "#009E73"),
    (("s2",), "#E69F00"),
    (("s3",), "#D55E00"),
    (("s4",), "#666666"),
]
_GATE_LABELS = {
    "valid_ok": "Valid",
    "novel_ok": "Novel",
    "star_ok": "Two stars",
    "sa_ok": "SA",
    "sa_ok_reporting": "SA\n(reporting)",
    "sa_ok_discovery": "SA\n(discovery)",
    "soluble_ok": "Soluble",
    "class_ok": "Target\nclass",
    "class_ok_loose": "Class\n(loose)",
    "class_ok_strict": "Class\n(strict)",
    "chi_ok": "Chi",
    "chi_band_ok": "Chi band",
    "success_hit": "Full\nsuccess",
    "success_hit_discovery": "Discovery\nsuccess",
    "property_success_hit": "Property\nsuccess",
    "property_success_hit_discovery": "Property\ndiscovery",
}


def apply_step5_plot_style(*, font_size: int = 16, dpi: int = 600) -> None:
    """Apply the shared Step 5 plotting style."""

    apply_publication_figure_style(font_size=font_size, dpi=dpi, remove_titles=True)


def _save_step5_figure(
    fig,
    output_path: Path,
    *,
    dpi: int,
    rect: tuple[float, float, float, float] | None = None,
) -> None:
    if rect is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=rect)
    save_publication_figure(fig, output_path, dpi=dpi, write_pdf=False)


def _style_axis(ax, *, grid_axis: str | None = "y") -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(_EDGE_COLOR)
        spine.set_linewidth(0.9)
    ax.tick_params(axis="both", colors=_TEXT_COLOR, width=0.9, length=4)
    if grid_axis:
        ax.grid(
            True,
            axis=grid_axis,
            linestyle=(0, (3, 3)),
            linewidth=0.65,
            color=_GRID_COLOR,
            alpha=0.9,
        )
    else:
        ax.grid(False)


def _format_rate_axis(ax, *, axis: str = "y", upper: float = 1.02) -> None:
    if axis == "x":
        ax.set_xlim(0.0, max(1.0, float(upper)))
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    else:
        ax.set_ylim(0.0, max(1.0, float(upper)))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def _metric_axis_label(label: str, *, mean: bool = True, across_runs: bool = False) -> str:
    text = str(label).strip().lower()
    if "property" in text and "discovery" in text:
        base = "property discovery success rate"
    elif "property" in text:
        base = "property success rate"
    elif "discovery" in text:
        base = "discovery success rate"
    else:
        base = "success rate"
    if across_runs:
        return f"Mean {base} across runs"
    return f"Mean {base}" if mean else base.capitalize()


def _gate_label(gate: str) -> str:
    if gate in _GATE_LABELS:
        return _GATE_LABELS[gate]
    label = str(gate).replace("_hit_rate", "").replace("_rate", "").replace("_ok", "")
    chunks = [chunk for chunk in label.split("_") if chunk]
    if not chunks:
        return str(gate)
    if len(chunks) <= 2:
        return " ".join(chunks).capitalize()
    return "\n".join([" ".join(chunks[:-1]).capitalize(), chunks[-1]])


def _run_color(value: object, index: int = 0) -> str:
    key = str(value).strip().lower()
    for needles, color in _RUN_COLOR_RULES:
        if any(key.startswith(needle) or needle in key for needle in needles):
            return color
    return _STEP5_COLORS[int(index) % len(_STEP5_COLORS)]


def _run_marker(index: int) -> str:
    return _RUN_MARKERS[int(index) % len(_RUN_MARKERS)]


def _legend_font_size(font_size: int) -> int:
    return int(font_size)


def _add_shared_legend(
    legend_ax,
    axes,
    *,
    font_size: int,
    ncol: int = 1,
    bbox_to_anchor: tuple[float, float] = (1.01, 1.0),
) -> None:
    handles = []
    labels = []
    seen = set()
    for ax in np.ravel(axes):
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if not label or str(label).startswith("_") or label in seen:
                continue
            handles.append(handle)
            labels.append(label)
            seen.add(label)
    if not handles:
        return
    legend_ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=bbox_to_anchor,
        borderaxespad=0.0,
        fontsize=_legend_font_size(font_size),
        ncol=ncol,
        handlelength=1.7,
        columnspacing=0.9,
    )


def _figure_width(count: int, *, base: float = 5.6, per_item: float = 0.35, max_width: float = 12.0) -> float:
    return min(max_width, max(base, 2.6 + per_item * max(1, int(count))))


def _target_rotation(count: int) -> int:
    return 90 if int(count) > 12 else 0


def _set_integer_x_axis(ax, values, *, minimum_span: int = 1) -> None:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if numeric.size == 0:
        ax.set_xlim(0.0, float(max(1, minimum_span)))
        ax.set_xticks([0, max(1, minimum_span)])
        return
    lo_data = float(np.nanmin(numeric))
    hi_data = float(np.nanmax(numeric))
    lo = min(0, int(np.floor(lo_data)))
    hi = max(int(np.ceil(hi_data)), lo + int(max(1, minimum_span)))
    ax.set_xlim(float(lo), float(hi))
    ax.set_xticks(np.arange(lo, hi + 1, dtype=int))


def _add_bar_value_labels(ax, bars, values, *, font_size: int, rate: bool = True) -> None:
    for bar, value in zip(bars, values):
        if not np.isfinite(float(value)):
            continue
        label = f"{float(value):.0%}" if rate else f"{float(value):.2g}"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(float(value) + 0.025, 1.055) if rate else float(value),
            label,
            ha="center",
            va="bottom",
            fontsize=int(font_size),
            color=_TEXT_COLOR,
        )


def _add_horizontal_value_labels(ax, values, *, font_size: int) -> None:
    for idx, value in enumerate(values):
        if not np.isfinite(float(value)):
            continue
        if float(value) >= 0.96:
            x = max(0.02, float(value) - 0.025)
            ha = "right"
            color = "white"
        else:
            x = min(1.045, max(0.02, float(value) + 0.018))
            ha = "left"
            color = _TEXT_COLOR
        ax.text(
            x,
            idx,
            f"{float(value):.0%}",
            va="center",
            ha=ha,
            fontsize=int(font_size),
            color=color,
        )


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


def _add_summary_box(ax, text: str, *, loc: tuple[float, float] = (0.03, 0.97)) -> None:
    if not str(text).strip():
        return
    ax.text(
        float(loc[0]),
        float(loc[1]),
        str(text),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=int(plt.rcParams.get("font.size", 16)),
        color=_TEXT_COLOR,
        linespacing=1.15,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "#C7CDD1",
            "linewidth": 0.6,
            "alpha": 0.9,
        },
    )


def _apply_parity_axis(ax, x: pd.Series, y: pd.Series) -> None:
    x_vals = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_vals = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    finite_x = x_vals[np.isfinite(x_vals)]
    finite_y = y_vals[np.isfinite(y_vals)]
    if finite_x.size == 0 or finite_y.size == 0:
        return
    lo = float(min(finite_x.min(), finite_y.min()))
    hi = float(max(finite_x.max(), finite_y.max()))
    span = max(hi - lo, 1.0e-8)
    pad = 0.04 * span
    lo_plot = lo - pad
    hi_plot = hi + pad
    ax.plot(
        [lo_plot, hi_plot],
        [lo_plot, hi_plot],
        linestyle=(0, (4, 3)),
        color=_EDGE_COLOR,
        linewidth=1.1,
        zorder=1,
    )
    ax.set_xlim(lo_plot, hi_plot)
    ax.set_ylim(lo_plot, hi_plot)
    ax.set_aspect("equal", adjustable="box")
    _style_axis(ax, grid_axis="both")


def _safe_series_correlation(x: pd.Series, y: pd.Series) -> float:
    x_num = pd.to_numeric(x, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")
    valid = x_num.notna() & y_num.notna()
    if int(valid.sum()) < 2:
        return float("nan")
    return float(x_num.loc[valid].corr(y_num.loc[valid]))


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

    values = ordered[mean_col].astype(float).to_numpy(dtype=float)
    errors = ordered[std_col].astype(float).to_numpy(dtype=float) if std_col in ordered.columns else None
    fig, ax = plt.subplots(figsize=(_figure_width(len(ordered), base=6.0, per_item=0.38), 4.6))
    bars = ax.bar(
        range(len(ordered)),
        values,
        yerr=errors,
        color="#4E79A7",
        edgecolor=_EDGE_COLOR,
        linewidth=0.7,
        alpha=0.92,
        error_kw={"elinewidth": 0.9, "capthick": 0.9, "capsize": 3, "ecolor": _EDGE_COLOR},
    )
    ax.set_xlabel("Target row")
    ax.set_ylabel(_metric_axis_label(ylabel, mean=False))
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(ordered["target_label"].tolist(), rotation=_target_rotation(len(ordered)))
    _format_rate_axis(ax, upper=1.08)
    _style_axis(ax, grid_axis="y")
    if len(ordered) <= 12:
        _add_bar_value_labels(ax, bars, values, font_size=font_size)
    _save_step5_figure(fig, output_path, dpi=dpi)
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
    rates = [float(pd.to_numeric(evaluation_df[gate], errors="coerce").fillna(0.0).mean()) for gate in gates]

    fig, ax = plt.subplots(figsize=(6.8, max(3.8, 0.45 * len(gates) + 1.3)))
    y_pos = np.arange(len(gates), dtype=int)
    colors = [_STEP5_COLORS[idx % len(_STEP5_COLORS)] for idx in range(len(gates))]
    ax.barh(
        y_pos,
        rates,
        color=colors,
        edgecolor=_EDGE_COLOR,
        linewidth=0.7,
        alpha=0.92,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([_gate_label(gate) for gate in gates])
    ax.invert_yaxis()
    ax.set_xlabel("Pass rate")
    ax.set_ylabel("Screening gate")
    _format_rate_axis(ax, axis="x", upper=1.08)
    _style_axis(ax, grid_axis="x")
    _add_horizontal_value_labels(ax, rates, font_size=font_size)
    _save_step5_figure(fig, output_path, dpi=dpi)
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
    valid = evaluation_df.loc[
        evaluation_df["chi_pred_target"].notna() & evaluation_df["chi_target"].notna()
    ].copy()

    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    if not valid.empty:
        ax.scatter(
            valid["chi_target"].astype(float),
            valid["chi_pred_target"].astype(float),
            s=20,
            alpha=0.62,
            color="#0072B2",
            edgecolors="white",
            linewidths=0.25,
            zorder=2,
        )
        _apply_parity_axis(ax, valid["chi_target"], valid["chi_pred_target"])
        corr = _safe_series_correlation(valid["chi_target"], valid["chi_pred_target"])
        _add_summary_box(
            ax,
            f"n={len(valid)}\nr={corr:.3f}" if np.isfinite(corr) else f"n={len(valid)}",
        )
    else:
        _style_axis(ax, grid_axis="both")
    ax.set_xlabel("Target chi")
    ax.set_ylabel("Predicted chi")
    _save_step5_figure(fig, output_path, dpi=dpi)
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

    labels = [_run_label(row) for _, row in ordered.iterrows()]
    values = ordered[mean_col].astype(float).to_numpy(dtype=float)
    errors = ordered[std_col].astype(float).to_numpy(dtype=float) if std_col in ordered.columns else None
    colors = [_run_color(label, idx) for idx, label in enumerate(labels)]

    fig, ax = plt.subplots(figsize=(_figure_width(len(ordered), base=6.2, per_item=0.55), 4.8))
    bars = ax.bar(
        range(len(ordered)),
        values,
        yerr=errors,
        color=colors,
        edgecolor=_EDGE_COLOR,
        linewidth=0.7,
        alpha=0.92,
        error_kw={"elinewidth": 0.9, "capthick": 0.9, "capsize": 3, "ecolor": _EDGE_COLOR},
    )
    ax.set_xlabel("Run")
    ax.set_ylabel(_metric_axis_label(label))
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(labels, rotation=0)
    _format_rate_axis(ax, upper=1.08)
    _style_axis(ax, grid_axis="y")
    _add_bar_value_labels(ax, bars, values, font_size=font_size)
    _save_step5_figure(fig, output_path, dpi=dpi)
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

    values = ordered[mean_col].astype(float).to_numpy(dtype=float)
    labels = ordered["canonical_family"].astype(str).tolist()
    fig, ax = plt.subplots(figsize=(_figure_width(len(ordered), base=5.8, per_item=0.55), 4.7))
    bars = ax.bar(
        range(len(ordered)),
        values,
        color=[_run_color(label, idx) for idx, label in enumerate(labels)],
        edgecolor=_EDGE_COLOR,
        linewidth=0.7,
        alpha=0.92,
    )
    ax.set_xlabel("Canonical family")
    ax.set_ylabel(_metric_axis_label(label, mean=False))
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(labels)
    _format_rate_axis(ax, upper=1.08)
    _style_axis(ax, grid_axis="y")
    for idx, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(float(values[idx]) + 0.025, 1.055),
            str(ordered.iloc[idx].get("best_run_label", ordered.iloc[idx]["best_run_name"])),
            rotation=0,
            ha="center",
            va="bottom",
            fontsize=int(font_size),
            color=_TEXT_COLOR,
        )
    _save_step5_figure(fig, output_path, dpi=dpi)
    plt.close(fig)


def plot_hpo_best_metric_curve(
    trials_df: pd.DataFrame,
    output_path: Path,
    *,
    metric_candidates: list[str],
    output_column: str,
    ylabel: str,
    font_size: int = 16,
    dpi: int = 600,
) -> None:
    """Plot a running-best HPO metric across trials."""

    apply_step5_plot_style(font_size=font_size, dpi=dpi)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = trials_df.copy()
    if "trial_number" not in ordered.columns:
        ordered["trial_number"] = np.arange(len(ordered), dtype=int)
    ordered["trial_number"] = pd.to_numeric(ordered["trial_number"], errors="coerce")
    ordered = ordered.loc[ordered["trial_number"].notna()].sort_values("trial_number", kind="mergesort").reset_index(drop=True)

    metric_col = next((column for column in metric_candidates if column in ordered.columns), None)
    metric_values = (
        pd.to_numeric(ordered[metric_col], errors="coerce")
        if metric_col is not None
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
    ordered[output_column] = running_best

    is_rate_metric = "rate" in str(ylabel).lower() or "success" in str(ylabel).lower()
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    if not ordered.empty and np.isfinite(np.asarray(running_best, dtype=float)).any():
        x_values = ordered["trial_number"].to_numpy(dtype=int)
        y_values = ordered[output_column].to_numpy(dtype=float)
        ax.plot(
            x_values,
            y_values,
            color="#D55E00",
            linewidth=1.8,
            marker="o",
            markersize=4.8,
            markeredgecolor="white",
            markeredgewidth=0.4,
        )
        ymax = float(np.nanmax(ordered[output_column].to_numpy(dtype=float)))
        if is_rate_metric:
            _format_rate_axis(ax, upper=max(1.02, min(1.08, ymax * 1.12 if ymax > 0.0 else 1.02)))
        else:
            ax.set_ylim(0.0, ymax * 1.12 if ymax > 0.0 else 1.0)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    else:
        ax.text(
            0.5,
            0.5,
            "No completed trials",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=_TEXT_COLOR,
        )
        if is_rate_metric:
            _format_rate_axis(ax, upper=1.02)
        else:
            ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Trial")
    ax.set_ylabel(_metric_axis_label(ylabel, mean=False) if is_rate_metric else ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _style_axis(ax, grid_axis="y")
    _save_step5_figure(fig, output_path, dpi=dpi)
    plt.close(fig)


def plot_hpo_best_success_curve(
    trials_df: pd.DataFrame,
    output_path: Path,
    *,
    font_size: int = 16,
    dpi: int = 600,
) -> None:
    """Plot running-best success hit rate across HPO trials."""

    plot_hpo_best_metric_curve(
        trials_df,
        output_path,
        metric_candidates=["mean_success_hit_rate_discovery", "mean_success_hit_rate"],
        output_column="best_success_hit_rate_so_far",
        ylabel="Best success hit rate so far",
        font_size=font_size,
        dpi=dpi,
    )


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

    run_groups = list(ordered.groupby("run_name", sort=True))
    fig, ax = plt.subplots(figsize=(_figure_width(len(target_order), base=7.0, per_item=0.5, max_width=11.5), 4.8))
    for idx, (run_name, sub) in enumerate(run_groups):
        sub = sub.sort_values(["temperature", "phi", "target_row_id"])
        xs = [target_positions[int(target_row_id)] for target_row_id in sub["target_row_id"]]
        ys = sub[mean_col].astype(float).tolist()
        run_display = str(sub["run_label"].iloc[0]) if "run_label" in sub.columns else str(run_name)
        ax.plot(
            xs,
            ys,
            marker=_run_marker(idx),
            linewidth=1.7,
            markersize=4.8,
            alpha=0.92,
            label=run_display,
            color=_run_color(run_display, idx),
            markeredgecolor="white",
            markeredgewidth=0.35,
        )
    ax.set_xlabel("Target row")
    ax.set_ylabel(_metric_axis_label(label))
    ax.set_xticks(range(len(target_order)))
    if "target_label" not in target_order.columns:
        target_order = _add_compact_target_labels(target_order)
    ax.set_xticklabels(target_order["target_label"].tolist(), rotation=_target_rotation(len(target_order)))
    _format_rate_axis(ax, upper=1.04)
    _style_axis(ax, grid_axis="y")
    legend_cols = 1 if len(run_groups) <= 8 else 2
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        fontsize=_legend_font_size(font_size),
        ncol=legend_cols,
        handlelength=1.6,
        columnspacing=0.9,
    )
    _save_step5_figure(fig, output_path, dpi=dpi, rect=(0, 0, 0.82, 1))
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

    values = ordered["mean_success_hit_rate_across_runs"].astype(float).to_numpy(dtype=float)
    errors = ordered["std_success_hit_rate_across_runs"].astype(float).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(_figure_width(len(ordered), base=6.2, per_item=0.38), 4.6))
    bars = ax.bar(
        range(len(ordered)),
        values,
        yerr=errors,
        color="#D55E00",
        edgecolor=_EDGE_COLOR,
        linewidth=0.7,
        alpha=0.92,
        error_kw={"elinewidth": 0.9, "capthick": 0.9, "capsize": 3, "ecolor": _EDGE_COLOR},
    )
    ax.set_xlabel("Target row")
    ax.set_ylabel(_metric_axis_label(label, across_runs=True))
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(ordered["target_label"].tolist(), rotation=_target_rotation(len(ordered)))
    _format_rate_axis(ax, upper=1.08)
    _style_axis(ax, grid_axis="y")
    if len(ordered) <= 12:
        _add_bar_value_labels(ax, bars, values, font_size=font_size)
    _save_step5_figure(fig, output_path, dpi=dpi)
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

    fig, ax = plt.subplots(figsize=(7.6, max(4.5, 0.46 * len(gates) + 1.5)))
    ordered = run_comparison_df.sort_values([mean_col, "run_name"], ascending=[False, True]).reset_index(drop=True)
    y_pos = np.arange(len(gates), dtype=int)
    for idx, row in ordered.iterrows():
        rates = []
        for gate in gates[:-1]:
            rates.append(float(row.get(f"mean_{gate}_rate", float("nan"))))
        rates.append(float(row[mean_col]))
        run_display = _run_label(row)
        ax.plot(
            rates,
            y_pos,
            marker=_run_marker(idx),
            linewidth=1.7,
            markersize=4.8,
            alpha=0.92,
            label=run_display,
            color=_run_color(run_display, idx),
            markeredgecolor="white",
            markeredgewidth=0.35,
        )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([_gate_label(gate) for gate in gates])
    ax.invert_yaxis()
    ax.set_xlabel("Mean pass rate")
    ax.set_ylabel("Screening gate")
    _format_rate_axis(ax, axis="x", upper=1.08)
    _style_axis(ax, grid_axis="x")
    legend_cols = 1 if len(ordered) <= 8 else 2
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        fontsize=_legend_font_size(font_size),
        ncol=legend_cols,
        handlelength=1.6,
        columnspacing=0.9,
    )
    _save_step5_figure(fig, output_path, dpi=dpi, rect=(0, 0, 0.8, 1))
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

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    x = ordered["mean_total_oracle_calls"].astype(float)
    y = ordered[mean_col].astype(float)
    labels = [_run_label(row) for _, row in ordered.iterrows()]
    colors = [_run_color(label, idx) for idx, label in enumerate(labels)]
    ax.scatter(
        x,
        y,
        s=54,
        alpha=0.9,
        color=colors,
        edgecolors="white",
        linewidths=0.45,
        zorder=3,
    )
    for idx, row in ordered.iterrows():
        offset = (5, 5 if idx % 2 == 0 else -10)
        ax.annotate(
            _run_label(row),
            (float(row["mean_total_oracle_calls"]), float(row[mean_col])),
            xytext=offset,
            textcoords="offset points",
            fontsize=int(font_size),
            color=_TEXT_COLOR,
        )
    ax.set_xlabel("Mean oracle calls")
    ax.set_ylabel(_metric_axis_label(label))
    _format_rate_axis(ax, upper=1.04)
    _style_axis(ax, grid_axis="both")
    _save_step5_figure(fig, output_path, dpi=dpi)
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
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharex=False)
    ordered = history_df.sort_values(["run_label", "global_step"], kind="mergesort")
    run_groups = list(ordered.groupby("run_name", sort=True))
    for idx, (_run_name, sub) in enumerate(run_groups):
        label = str(sub["run_label"].iloc[0]) if "run_label" in sub.columns else str(_run_name)
        x = pd.to_numeric(sub["global_step"], errors="coerce").to_numpy(dtype=float)
        if "train_diffusion_loss_window" in sub.columns:
            y_train = pd.to_numeric(sub["train_diffusion_loss_window"], errors="coerce").to_numpy(dtype=float)
            axes[0].plot(x, y_train, linewidth=1.6, label=label, color=_run_color(label, idx))
        if "val_diffusion_loss" in sub.columns:
            y_val = pd.to_numeric(sub["val_diffusion_loss"], errors="coerce").to_numpy(dtype=float)
            axes[1].plot(x, y_val, linewidth=1.6, label=label, color=_run_color(label, idx))
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Train diffusion loss")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Val diffusion loss")
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        _style_axis(ax, grid_axis="both")
    _add_shared_legend(axes[1], axes, font_size=font_size)
    _save_step5_figure(fig, output_path, dpi=dpi, rect=(0, 0, 0.84, 1))
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
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 7.0), sharex=False)
    axes_flat = axes.reshape(-1)
    plot_specs = [
        ("loss", "Loss"),
        ("baseline_reward", "Reward"),
        ("trajectory_kl_mean", "KL"),
        ("proxy_property_success_hit_rate_discovery", "Proxy success"),
    ]
    ordered = history_df.sort_values(["run_label", "step_idx"], kind="mergesort")
    run_groups = list(ordered.groupby("run_name", sort=True))
    for idx, (_run_name, sub) in enumerate(run_groups):
        label = str(sub["run_label"].iloc[0]) if "run_label" in sub.columns else str(_run_name)
        x = pd.to_numeric(sub["step_idx"], errors="coerce").to_numpy(dtype=float)
        for ax, (column, ylabel) in zip(axes_flat, plot_specs):
            if column in sub.columns and sub[column].notna().any():
                y = pd.to_numeric(sub[column], errors="coerce").to_numpy(dtype=float)
                ax.plot(x, y, linewidth=1.45, label=label, color=_run_color(label, idx))
            ax.set_xlabel("Step")
            ax.set_ylabel(ylabel)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            _style_axis(ax, grid_axis="both")
    _add_shared_legend(axes_flat[-1], axes_flat, font_size=font_size)
    _save_step5_figure(fig, output_path, dpi=dpi, rect=(0, 0, 0.84, 1))
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
    metric_candidates = [
        [("train_dpo_loss", "DPO loss", False)],
        [
            ("val_preference_accuracy", "Preference accuracy", True),
            ("train_preference_accuracy", "Preference accuracy", True),
            ("val_dpo_loss", "DPO loss", False),
        ],
        [
            ("proxy_property_success_hit_rate_discovery", "Proxy success", True),
            ("proxy_property_success_hit_rate_reporting", "Proxy success", True),
            ("train_margin_mean", "DPO margin", False),
        ],
    ]
    plot_specs: list[tuple[str, str, bool]] = []
    for candidates in metric_candidates:
        for column, ylabel, is_rate in candidates:
            if column in history_df.columns and pd.to_numeric(history_df[column], errors="coerce").notna().any():
                plot_specs.append((column, ylabel, is_rate))
                break
    if not plot_specs:
        return

    fig, axes = plt.subplots(1, len(plot_specs), figsize=(5.6 * len(plot_specs), 4.8), sharex=False)
    axes = np.atleast_1d(axes)
    ordered = history_df.sort_values(["run_label", "epoch_idx"], kind="mergesort")
    run_groups = list(ordered.groupby("run_name", sort=True))
    all_epoch_values: list[float] = []
    for idx, (_run_name, sub) in enumerate(run_groups):
        label = str(sub["run_label"].iloc[0]) if "run_label" in sub.columns else str(_run_name)
        x = pd.to_numeric(sub["epoch_idx"], errors="coerce").to_numpy(dtype=float)
        all_epoch_values.extend([float(value) for value in x if np.isfinite(value)])
        for ax, (column, ylabel, is_rate) in zip(axes, plot_specs):
            if column in sub.columns and sub[column].notna().any():
                y = pd.to_numeric(sub[column], errors="coerce").to_numpy(dtype=float)
                ax.plot(
                    x,
                    y,
                    linewidth=1.55,
                    marker=_run_marker(idx),
                    markersize=4.5,
                    label=label,
                    color=_run_color(label, idx),
                )
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            if is_rate:
                _format_rate_axis(ax, upper=1.02)
            else:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            _style_axis(ax, grid_axis="both")
    for ax in axes:
        _set_integer_x_axis(ax, all_epoch_values, minimum_span=2)
    if len(run_groups) > 1:
        _add_shared_legend(axes[-1], axes, font_size=font_size)
    _save_step5_figure(fig, output_path, dpi=dpi, rect=(0.02, 0.02, 0.9, 0.98))
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
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    if not valid.empty:
        run_groups = list(valid.groupby("run_name", sort=True))
        for idx, (_run_name, sub) in enumerate(run_groups):
            label = str(sub["run_label"].iloc[0]) if "run_label" in sub.columns else str(_run_name)
            ax.scatter(
                sub["chi_target"].astype(float),
                sub["chi_pred_target"].astype(float),
                s=15,
                alpha=0.52,
                label=label,
                color=_run_color(label, idx),
                edgecolors="white",
                linewidths=0.2,
                zorder=2,
            )
        _apply_parity_axis(ax, valid["chi_target"], valid["chi_pred_target"])
        corr = _safe_series_correlation(valid["chi_target"], valid["chi_pred_target"])
        _add_summary_box(
            ax,
            f"n={len(valid)}\nr={corr:.3f}" if np.isfinite(corr) else f"n={len(valid)}",
        )
        legend_cols = 1 if len(run_groups) <= 8 else 2
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            fontsize=_legend_font_size(font_size),
            ncol=legend_cols,
            handlelength=1.2,
            columnspacing=0.8,
        )
    else:
        _style_axis(ax, grid_axis="both")
    ax.set_xlabel("Target chi")
    ax.set_ylabel("Predicted chi")
    _save_step5_figure(fig, output_path, dpi=dpi, rect=(0, 0, 0.82, 1))
    plt.close(fig)
