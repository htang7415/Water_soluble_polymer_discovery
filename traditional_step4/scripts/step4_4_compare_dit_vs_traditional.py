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


def _load_test_row(csv_path: Path) -> pd.Series:
    if not csv_path.exists():
        raise FileNotFoundError(f"metrics file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "split" not in df.columns:
        raise ValueError(f"Expected 'split' column in {csv_path}")
    test_df = df[df["split"] == "test"]
    if test_df.empty:
        raise ValueError(f"No test row in {csv_path}")
    return test_df.iloc[0]


def _load_best_model_row_from_summary(
    summary_csv: Path,
    *,
    primary_metric: str,
    primary_higher_is_better: bool,
    tiebreak_metrics: List[tuple[str, bool]] | None = None,
) -> pd.Series:
    if not summary_csv.exists():
        raise FileNotFoundError(f"model summary file not found: {summary_csv}")
    df = pd.read_csv(summary_csv)
    if df.empty:
        raise ValueError(f"model summary is empty: {summary_csv}")
    if "model_name" not in df.columns:
        raise ValueError(f"Expected 'model_name' column in {summary_csv}")
    if primary_metric not in df.columns:
        raise ValueError(f"Expected '{primary_metric}' column in {summary_csv}")

    score = pd.to_numeric(df[primary_metric], errors="coerce")
    valid = df[score.notna()].copy()
    if valid.empty:
        raise ValueError(f"No valid '{primary_metric}' values in {summary_csv}")

    sort_cols: List[str] = []
    ascending: List[bool] = []

    valid["__primary"] = pd.to_numeric(valid[primary_metric], errors="coerce")
    sort_cols.append("__primary")
    ascending.append(not bool(primary_higher_is_better))

    for metric_name, metric_higher_is_better in (tiebreak_metrics or []):
        if metric_name not in valid.columns:
            continue
        numeric_col = f"__tb_{metric_name}"
        valid[numeric_col] = pd.to_numeric(valid[metric_name], errors="coerce")
        fill_value = -np.inf if metric_higher_is_better else np.inf
        valid[numeric_col] = valid[numeric_col].fillna(fill_value)
        sort_cols.append(numeric_col)
        ascending.append(not bool(metric_higher_is_better))

    valid = valid.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    best = valid.iloc[0].copy()
    return best


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
    plot_df = df[["model_size", f"dit_{metric}", f"traditional_{metric}"]].copy()
    if model_size_order:
        present = set(plot_df["model_size"].astype(str).tolist())
        ordered = [str(s) for s in model_size_order if str(s) in present]
        if ordered:
            plot_df["model_size"] = pd.Categorical(plot_df["model_size"], categories=ordered, ordered=True)
            plot_df = plot_df.sort_values("model_size").reset_index(drop=True)
    plot_df = plot_df.rename(
        columns={
            f"dit_{metric}": "DiT",
            f"traditional_{metric}": "Traditional",
        }
    )
    melted = plot_df.melt(id_vars="model_size", var_name="pipeline", value_name="value")
    apply_publication_figure_style(font_size=font_size, dpi=dpi, remove_titles=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(data=melted, x="model_size", y="value", hue="pipeline", ax=ax)
    ax.set_xlabel("Model size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def _sort_by_model_size(df: pd.DataFrame, model_size_order: List[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    if model_size_order:
        rank = {str(s): i for i, s in enumerate(model_size_order)}
        out["__size_rank"] = out["model_size"].astype(str).map(lambda s: rank.get(str(s), len(rank)))
        out = out.sort_values("__size_rank").drop(columns=["__size_rank"])
    return out.reset_index(drop=True)


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
    cols = ["model_size"] + [f"dit_{m}" for m, _ in metric_specs] + [f"traditional_{m}" for m, _ in metric_specs]
    for c in cols:
        if c not in df.columns:
            return
    plot_df = _sort_by_model_size(df[cols], model_size_order=model_size_order)
    x = np.arange(len(plot_df), dtype=float)
    xlabels = plot_df["model_size"].astype(str).tolist()

    apply_publication_figure_style(font_size=font_size, dpi=dpi, remove_titles=True)
    fig, axes = plt.subplots(1, len(metric_specs), figsize=(5.0 * len(metric_specs), 4.2))
    if len(metric_specs) == 1:
        axes = [axes]
    for ax, (metric, ylabel) in zip(axes, metric_specs):
        dit_vals = pd.to_numeric(plot_df[f"dit_{metric}"], errors="coerce").to_numpy(dtype=float)
        trad_vals = pd.to_numeric(plot_df[f"traditional_{metric}"], errors="coerce").to_numpy(dtype=float)
        ax.plot(x, dit_vals, marker="o", linewidth=2.0, color="#1f77b4", label="DiT")
        ax.plot(x, trad_vals, marker="s", linewidth=2.0, color="#d62728", label="Traditional")
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        ax.set_xlabel("Model size")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.45)
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.suptitle(title, y=1.04)
    fig.tight_layout(rect=(0, 0, 0.94, 1))
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def _delta_heatmap(
    df: pd.DataFrame,
    delta_specs: List[tuple[str, str]],
    title: str,
    out_png: Path,
    dpi: int,
    font_size: int,
    model_size_order: List[str] | None = None,
) -> None:
    if df.empty or len(delta_specs) == 0:
        return
    cols = ["model_size"] + [c for c, _ in delta_specs]
    for c in cols:
        if c not in df.columns:
            return
    plot_df = _sort_by_model_size(df[cols], model_size_order=model_size_order).set_index("model_size")
    heat = plot_df.rename(columns={c: label for c, label in delta_specs})
    heat = heat.apply(pd.to_numeric, errors="coerce")
    if heat.empty:
        return
    arr = heat.to_numpy(dtype=float)
    finite_vals = arr[np.isfinite(arr)]
    vmax = float(np.max(np.abs(finite_vals))) if finite_vals.size > 0 else 1.0
    if (not np.isfinite(vmax)) or vmax < 1.0e-8:
        vmax = 1.0

    apply_publication_figure_style(font_size=font_size, dpi=dpi, remove_titles=True)
    fig, ax = plt.subplots(figsize=(1.8 * heat.shape[1] + 2.2, 0.9 * heat.shape[0] + 2.2))
    sns.heatmap(
        heat,
        cmap="RdBu_r",
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Traditional - DiT"},
        ax=ax,
    )
    ax.set_xlabel("Metric")
    ax.set_ylabel("Model size")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def _winner_count_figure(reg_df: pd.DataFrame, cls_df: pd.DataFrame, out_png: Path, dpi: int, font_size: int) -> None:
    rows: List[Dict[str, object]] = []
    specs = [
        (reg_df, "Regression", "winner_r2", "R2"),
        (reg_df, "Regression", "winner_rmse", "RMSE"),
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

    apply_publication_figure_style(font_size=font_size, dpi=dpi, remove_titles=True)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    sns.barplot(
        data=plot_df,
        x="metric",
        y="count",
        hue="winner",
        palette={"DiT": "#1f77b4", "Traditional": "#d62728"},
        ax=ax,
    )
    ax.set_xlabel("Metric")
    ax.set_ylabel("Model-size wins")
    ax.set_title("Winner counts by metric")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
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
    ax.set_xlabel("Missing/invalid input stage")
    ax.set_ylabel("Count")
    ax.set_title("Step4_4 missing/invalid inputs by stage")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Model size", loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
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
    for d in [output_dir, metrics_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

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
                dit_results_dir_root / "step4_chi_training" / "step4_1_regression" / split_mode / "metrics" / "chi_metrics_overall.csv",
                dit_results_dir_split / "step4_chi_training" / split_mode / "step4_1_regression" / "metrics" / "chi_metrics_overall.csv",
            ]
        )
        dit_cls_csv = _first_existing_path(
            [
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
            dit_reg = _load_test_row(dit_reg_csv)
        except Exception as exc:
            reg_ready = False
            missing_rows.append({"model_size": model_size, "stage": "step4_1_regression", "path": str(dit_reg_csv), "error": str(exc)})
        try:
            trad_reg = _load_best_model_row_from_summary(
                trad_reg_summary_csv,
                primary_metric="test_r2",
                primary_higher_is_better=True,
                tiebreak_metrics=[("test_rmse", False), ("test_mae", False)],
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
            row = {
                "model_size": model_size,
                "dit_r2": _safe_get(dit_reg, "r2"),
                "dit_rmse": _safe_get(dit_reg, "rmse"),
                "dit_mae": _safe_get(dit_reg, "mae"),
                "traditional_best_model_name": str(trad_reg.get("model_name", "")),
                "traditional_r2": _safe_get(trad_reg, "test_r2"),
                "traditional_rmse": _safe_get(trad_reg, "test_rmse"),
                "traditional_mae": _safe_get(trad_reg, "test_mae"),
            }
            row["delta_r2_traditional_minus_dit"] = _safe_get(pd.Series(row), "traditional_r2") - _safe_get(pd.Series(row), "dit_r2")
            row["delta_rmse_traditional_minus_dit"] = _safe_get(pd.Series(row), "traditional_rmse") - _safe_get(pd.Series(row), "dit_rmse")
            row["delta_mae_traditional_minus_dit"] = _safe_get(pd.Series(row), "traditional_mae") - _safe_get(pd.Series(row), "dit_mae")
            row["winner_r2"] = "traditional" if row["delta_r2_traditional_minus_dit"] > 0 else "dit"
            row["winner_rmse"] = "traditional" if row["delta_rmse_traditional_minus_dit"] < 0 else "dit"
            reg_rows.append(row)

        cls_ready = True
        try:
            dit_cls = _load_test_row(dit_cls_csv)
        except Exception as exc:
            cls_ready = False
            missing_rows.append({"model_size": model_size, "stage": "step4_2_classification", "path": str(dit_cls_csv), "error": str(exc)})
        try:
            trad_cls = _load_best_model_row_from_summary(
                trad_cls_summary_csv,
                primary_metric="test_balanced_accuracy",
                primary_higher_is_better=True,
                tiebreak_metrics=[("test_auroc", True), ("test_f1", True)],
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
            row = {
                "model_size": model_size,
                "dit_balanced_accuracy": _safe_get(dit_cls, "balanced_accuracy"),
                "dit_auroc": _safe_get(dit_cls, "auroc"),
                "dit_f1": _safe_get(dit_cls, "f1"),
                "traditional_best_model_name": str(trad_cls.get("model_name", "")),
                "traditional_balanced_accuracy": _safe_get(trad_cls, "test_balanced_accuracy"),
                "traditional_auroc": _safe_get(trad_cls, "test_auroc"),
                "traditional_f1": _safe_get(trad_cls, "test_f1"),
            }
            row["delta_balanced_accuracy_traditional_minus_dit"] = _safe_get(pd.Series(row), "traditional_balanced_accuracy") - _safe_get(
                pd.Series(row), "dit_balanced_accuracy"
            )
            row["delta_auroc_traditional_minus_dit"] = _safe_get(pd.Series(row), "traditional_auroc") - _safe_get(pd.Series(row), "dit_auroc")
            row["delta_f1_traditional_minus_dit"] = _safe_get(pd.Series(row), "traditional_f1") - _safe_get(pd.Series(row), "dit_f1")
            row["winner_balanced_accuracy"] = "traditional" if row["delta_balanced_accuracy_traditional_minus_dit"] > 0 else "dit"
            row["winner_auroc"] = "traditional" if row["delta_auroc_traditional_minus_dit"] > 0 else "dit"
            row["winner_f1"] = "traditional" if row["delta_f1_traditional_minus_dit"] > 0 else "dit"
            cls_rows.append(row)

    missing_df = pd.DataFrame(missing_rows)
    missing_df.to_csv(metrics_dir / "missing_or_invalid_inputs.csv", index=False)

    reg_columns = [
        "model_size",
        "dit_r2",
        "dit_rmse",
        "dit_mae",
        "traditional_r2",
        "traditional_rmse",
        "traditional_mae",
        "delta_r2_traditional_minus_dit",
        "delta_rmse_traditional_minus_dit",
        "delta_mae_traditional_minus_dit",
        "winner_r2",
        "winner_rmse",
    ]
    cls_columns = [
        "model_size",
        "dit_balanced_accuracy",
        "dit_auroc",
        "dit_f1",
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
    else:
        reg_df = pd.DataFrame(columns=reg_columns)
    if cls_rows:
        cls_df = _sort_by_model_size(pd.DataFrame(cls_rows), model_size_order=model_sizes)
    else:
        cls_df = pd.DataFrame(columns=cls_columns)
    reg_df.to_csv(metrics_dir / "regression_model_size_comparison.csv", index=False)
    cls_df.to_csv(metrics_dir / "classification_model_size_comparison.csv", index=False)

    if not reg_df.empty:
        reg_summary = pd.DataFrame(
            [
                {
                    "split_mode": split_mode,
                    "n_model_sizes": int(len(reg_df)),
                    "mean_delta_r2_traditional_minus_dit": float(reg_df["delta_r2_traditional_minus_dit"].mean()),
                    "mean_delta_rmse_traditional_minus_dit": float(reg_df["delta_rmse_traditional_minus_dit"].mean()),
                    "traditional_wins_r2_count": int((reg_df["winner_r2"] == "traditional").sum()),
                    "traditional_wins_rmse_count": int((reg_df["winner_rmse"] == "traditional").sum()),
                }
            ]
        )
        reg_summary.to_csv(metrics_dir / "regression_comparation_summary.csv", index=False)
    else:
        pd.DataFrame().to_csv(metrics_dir / "regression_comparation_summary.csv", index=False)

    if not cls_df.empty:
        cls_summary = pd.DataFrame(
            [
                {
                    "split_mode": split_mode,
                    "n_model_sizes": int(len(cls_df)),
                    "mean_delta_balanced_accuracy_traditional_minus_dit": float(
                        cls_df["delta_balanced_accuracy_traditional_minus_dit"].mean()
                    ),
                    "mean_delta_auroc_traditional_minus_dit": float(cls_df["delta_auroc_traditional_minus_dit"].mean()),
                    "mean_delta_f1_traditional_minus_dit": float(cls_df["delta_f1_traditional_minus_dit"].mean()),
                    "traditional_wins_balanced_accuracy_count": int((cls_df["winner_balanced_accuracy"] == "traditional").sum()),
                    "traditional_wins_auroc_count": int((cls_df["winner_auroc"] == "traditional").sum()),
                    "traditional_wins_f1_count": int((cls_df["winner_f1"] == "traditional").sum()),
                }
            ]
        )
        cls_summary.to_csv(metrics_dir / "classification_comparation_summary.csv", index=False)
    else:
        pd.DataFrame().to_csv(metrics_dir / "classification_comparation_summary.csv", index=False)

    dpi = int(config.get("plotting", {}).get("dpi", 600))
    font_size = int(config.get("plotting", {}).get("font_size", 16))
    if not reg_df.empty:
        _barplot_two_models(
            df=reg_df,
            metric="r2",
            ylabel="Test R2",
            title=f"Step4_1 vs Step4_3_1 (split={split_mode})",
            out_png=figures_dir / "regression_test_r2_by_model_size.png",
            dpi=dpi,
            font_size=font_size,
            model_size_order=model_sizes,
        )
        _barplot_two_models(
            df=reg_df,
            metric="rmse",
            ylabel="Test RMSE",
            title=f"Step4_1 vs Step4_3_1 (split={split_mode})",
            out_png=figures_dir / "regression_test_rmse_by_model_size.png",
            dpi=dpi,
            font_size=font_size,
            model_size_order=model_sizes,
        )
        _barplot_two_models(
            df=reg_df,
            metric="mae",
            ylabel="Test MAE",
            title=f"Step4_1 vs Step4_3_1 (split={split_mode})",
            out_png=figures_dir / "regression_test_mae_by_model_size.png",
            dpi=dpi,
            font_size=font_size,
            model_size_order=model_sizes,
        )
        _overview_panel_two_models(
            df=reg_df,
            metric_specs=[("r2", "Test R2"), ("rmse", "Test RMSE"), ("mae", "Test MAE")],
            title=f"Regression comparison overview (split={split_mode})",
            out_png=figures_dir / "regression_overview_panel.png",
            dpi=dpi,
            font_size=font_size,
            model_size_order=model_sizes,
        )
        _delta_heatmap(
            df=reg_df,
            delta_specs=[
                ("delta_r2_traditional_minus_dit", "Δ R2"),
                ("delta_rmse_traditional_minus_dit", "Δ RMSE"),
                ("delta_mae_traditional_minus_dit", "Δ MAE"),
            ],
            title="Regression deltas by model size",
            out_png=figures_dir / "regression_delta_heatmap.png",
            dpi=dpi,
            font_size=font_size,
            model_size_order=model_sizes,
        )

    if not cls_df.empty:
        _barplot_two_models(
            df=cls_df,
            metric="balanced_accuracy",
            ylabel="Test balanced accuracy",
            title=f"Step4_2 vs Step4_3_2 (split={split_mode})",
            out_png=figures_dir / "classification_test_balanced_accuracy_by_model_size.png",
            dpi=dpi,
            font_size=font_size,
            model_size_order=model_sizes,
        )
        _barplot_two_models(
            df=cls_df,
            metric="auroc",
            ylabel="Test AUROC",
            title=f"Step4_2 vs Step4_3_2 (split={split_mode})",
            out_png=figures_dir / "classification_test_auroc_by_model_size.png",
            dpi=dpi,
            font_size=font_size,
            model_size_order=model_sizes,
        )
        _barplot_two_models(
            df=cls_df,
            metric="f1",
            ylabel="Test F1",
            title=f"Step4_2 vs Step4_3_2 (split={split_mode})",
            out_png=figures_dir / "classification_test_f1_by_model_size.png",
            dpi=dpi,
            font_size=font_size,
            model_size_order=model_sizes,
        )
        _overview_panel_two_models(
            df=cls_df,
            metric_specs=[
                ("balanced_accuracy", "Test balanced accuracy"),
                ("auroc", "Test AUROC"),
                ("f1", "Test F1"),
            ],
            title=f"Classification comparison overview (split={split_mode})",
            out_png=figures_dir / "classification_overview_panel.png",
            dpi=dpi,
            font_size=font_size,
            model_size_order=model_sizes,
        )
        _delta_heatmap(
            df=cls_df,
            delta_specs=[
                ("delta_balanced_accuracy_traditional_minus_dit", "Δ balanced accuracy"),
                ("delta_auroc_traditional_minus_dit", "Δ AUROC"),
                ("delta_f1_traditional_minus_dit", "Δ F1"),
            ],
            title="Classification deltas by model size",
            out_png=figures_dir / "classification_delta_heatmap.png",
            dpi=dpi,
            font_size=font_size,
            model_size_order=model_sizes,
        )

    _winner_count_figure(
        reg_df=reg_df,
        cls_df=cls_df,
        out_png=figures_dir / "winner_counts_by_metric.png",
        dpi=dpi,
        font_size=font_size,
    )
    _missing_inputs_figure(
        missing_df=missing_df,
        out_png=figures_dir / "missing_inputs_by_stage.png",
        dpi=dpi,
        font_size=font_size,
    )
    if reg_df.empty and cls_df.empty:
        _no_data_notice_figure(
            split_mode=split_mode,
            model_sizes=model_sizes,
            missing_df=missing_df,
            out_png=figures_dir / "comparison_no_valid_data_notice.png",
            dpi=dpi,
            font_size=font_size,
        )

    save_artifact_manifest(step_dir=output_dir, metrics_dir=metrics_dir, figures_dir=figures_dir, dpi=dpi)
    generated_figures = sorted(p.name for p in figures_dir.glob("*.png") if p.is_file())
    payload = {
        "split_mode": split_mode,
        "model_sizes": model_sizes,
        "n_regression_rows": int(len(reg_df)),
        "n_classification_rows": int(len(cls_df)),
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
            "n_classification_rows": int(len(cls_df)),
            "n_missing_or_invalid": int(len(missing_df)),
            "n_figures_generated": int(len(generated_figures)),
        },
        metrics_dir=metrics_dir,
    )
    print(f"Step4_4 comparation outputs: {output_dir}")


if __name__ == "__main__":
    main()
