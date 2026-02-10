#!/usr/bin/env python
"""Step 3: Learn chi_target thresholds from labeled chi data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chi.data import load_chi_dataset
from src.utils.config import load_config, save_config
from src.utils.reproducibility import save_run_metadata, seed_everything
from src.utils.reporting import save_step_summary, save_artifact_manifest, write_initial_log


CLASS_NAME_MAP = {1: "Water-soluble", 0: "Water-insoluble"}


def _default_chi_config(config: Dict) -> Dict:
    chi_cfg = config.get("chi_training", {})
    shared = chi_cfg.get("shared", {}) if isinstance(chi_cfg.get("shared", {}), dict) else {}
    step3_cfg = (
        chi_cfg.get("step3_target_learning", {})
        if isinstance(chi_cfg.get("step3_target_learning", {}), dict)
        else {}
    )

    defaults = {
        "dataset_path": "Data/chi/_50_polymers_T_phi.csv",
        "split_mode": "polymer",
        "target_objective": "balanced_accuracy",
        "target_bootstrap_repeats": 800,
    }

    out = defaults.copy()
    out["dataset_path"] = str(shared.get("dataset_path", chi_cfg.get("dataset_path", defaults["dataset_path"])))
    out["split_mode"] = str(shared.get("split_mode", chi_cfg.get("split_mode", defaults["split_mode"])))
    out["target_objective"] = str(
        step3_cfg.get("target_objective", chi_cfg.get("target_objective", defaults["target_objective"]))
    )
    out["target_bootstrap_repeats"] = int(
        step3_cfg.get("target_bootstrap_repeats", chi_cfg.get("target_bootstrap_repeats", defaults["target_bootstrap_repeats"]))
    )
    return out


def _set_plot_style(font_size: int) -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "legend.fontsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
        }
    )


def _confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y = y_true.astype(int)
    p = y_pred.astype(int)

    tp = int(np.sum((y == 1) & (p == 1)))
    tn = int(np.sum((y == 0) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))

    n = max(len(y), 1)
    acc = (tp + tn) / n
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    balanced_acc = 0.5 * (recall + specificity)
    youden_j = recall + specificity - 1.0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float(acc),
        "balanced_accuracy": float(balanced_acc),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "youden_j": float(youden_j),
    }


def _threshold_candidates(values: np.ndarray) -> np.ndarray:
    u = np.sort(np.unique(values.astype(float)))
    if len(u) == 0:
        return np.array([])
    if len(u) == 1:
        return np.array([u[0]])
    mids = 0.5 * (u[:-1] + u[1:])
    eps = max(1e-8, (u[-1] - u[0]) * 1e-6)
    candidates = np.concatenate([[u[0] - eps], mids, [u[-1] + eps]])
    return candidates


def _scan_thresholds(x_chi: np.ndarray, y_label: np.ndarray, positive_when_low: bool) -> pd.DataFrame:
    rows = []
    for thr in _threshold_candidates(x_chi):
        pred = (x_chi <= thr).astype(int) if positive_when_low else (x_chi >= thr).astype(int)
        metric = _confusion_metrics(y_label, pred)
        metric["threshold"] = float(thr)
        rows.append(metric)
    return pd.DataFrame(rows)


def _pick_best_threshold(scan_df: pd.DataFrame, objective: str) -> pd.Series:
    if scan_df.empty:
        return pd.Series(dtype=float)
    if objective not in scan_df.columns:
        raise ValueError(f"Unknown objective: {objective}")

    best = scan_df.sort_values(
        by=[objective, "youden_j", "f1", "threshold"],
        ascending=[False, False, False, True],
    ).iloc[0]
    return best


def _bootstrap_threshold_ci(
    x_chi: np.ndarray,
    y_label: np.ndarray,
    objective: str,
    positive_when_low: bool,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    thresholds = []

    n = len(x_chi)
    if n == 0:
        return {
            "bootstrap_n": 0,
            "chi_target_boot_mean": np.nan,
            "chi_target_boot_std": np.nan,
            "chi_target_boot_q025": np.nan,
            "chi_target_boot_q500": np.nan,
            "chi_target_boot_q975": np.nan,
        }

    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        xb = x_chi[idx]
        yb = y_label[idx]
        if len(np.unique(yb)) < 2:
            continue
        scan = _scan_thresholds(xb, yb, positive_when_low=positive_when_low)
        if scan.empty:
            continue
        best = _pick_best_threshold(scan, objective)
        thresholds.append(float(best["threshold"]))

    if not thresholds:
        return {
            "bootstrap_n": 0,
            "chi_target_boot_mean": np.nan,
            "chi_target_boot_std": np.nan,
            "chi_target_boot_q025": np.nan,
            "chi_target_boot_q500": np.nan,
            "chi_target_boot_q975": np.nan,
        }

    thr = np.asarray(thresholds, dtype=float)
    return {
        "bootstrap_n": int(len(thr)),
        "chi_target_boot_mean": float(np.mean(thr)),
        "chi_target_boot_std": float(np.std(thr, ddof=0)),
        "chi_target_boot_q025": float(np.quantile(thr, 0.025)),
        "chi_target_boot_q500": float(np.quantile(thr, 0.500)),
        "chi_target_boot_q975": float(np.quantile(thr, 0.975)),
    }


def _make_figures(
    cond_best: pd.DataFrame,
    global_best: pd.DataFrame,
    df: pd.DataFrame,
    objective: str,
    fig_dir: Path,
    dpi: int,
    font_size: int,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    _set_plot_style(font_size)

    # chi_target heatmap
    pivot_thr = cond_best.pivot(index="temperature", columns="phi", values="chi_target")
    fig, ax = plt.subplots(figsize=(5.8, 5))
    sns.heatmap(
        pivot_thr,
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "χ_target"},
        ax=ax,
    )
    ax.set_xlabel("ϕ")
    ax.set_ylabel("Temperature (K)")
    ax.set_title("Condition-wise χ_target")
    fig.tight_layout()
    fig.savefig(fig_dir / "chi_target_heatmap.png", dpi=dpi)
    plt.close(fig)

    # quality heatmap
    pivot_obj = cond_best.pivot(index="temperature", columns="phi", values=objective)
    obj_vals = cond_best[objective].to_numpy(dtype=float)
    if np.isfinite(obj_vals).any():
        obj_min = float(np.nanmin(obj_vals))
        obj_max = float(np.nanmax(obj_vals))
        if np.isclose(obj_min, obj_max):
            pad = 0.02
        else:
            pad = max(0.01, 0.10 * (obj_max - obj_min))
        obj_vmin = max(0.0, obj_min - pad)
        obj_vmax = min(1.0, obj_max + pad)
        if np.isclose(obj_vmin, obj_vmax):
            obj_vmin = max(0.0, obj_vmin - 0.05)
            obj_vmax = min(1.0, obj_vmax + 0.05)
    else:
        obj_vmin, obj_vmax = 0.0, 1.0
    fig, ax = plt.subplots(figsize=(5.8, 5))
    sns.heatmap(
        pivot_obj,
        cmap="YlGnBu",
        vmin=obj_vmin,
        vmax=obj_vmax,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": objective.replace("_", " ").title()},
        ax=ax,
    )
    ax.set_xlabel("ϕ")
    ax.set_ylabel("Temperature (K)")
    ax.set_title(f"Threshold quality ({objective})")
    fig.tight_layout()
    fig.savefig(fig_dir / f"chi_target_{objective}_heatmap.png", dpi=dpi)
    plt.close(fig)

    # χ distribution with global threshold
    gthr = float(global_best["chi_target"].iloc[0])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.kdeplot(
        data=df[df["water_soluble"] == 1],
        x="chi",
        fill=True,
        alpha=0.4,
        label="Water-soluble",
        color="#1f77b4",
        ax=ax,
    )
    sns.kdeplot(
        data=df[df["water_soluble"] == 0],
        x="chi",
        fill=True,
        alpha=0.4,
        label="Water-insoluble",
        color="#d62728",
        ax=ax,
    )
    ax.axvline(gthr, color="black", linestyle="--", linewidth=2, label=f"Global χ_target={gthr:.3f}")
    ax.set_xlabel("χ")
    ax.set_ylabel("Density")
    ax.set_title("Global χ threshold from labels")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "chi_distribution_global_threshold.png", dpi=dpi)
    plt.close(fig)

    # threshold vs temperature with bootstrap CI
    fig, ax = plt.subplots(figsize=(6, 5))
    for phi, sub in cond_best.groupby("phi"):
        sub = sub.sort_values("temperature")
        ax.plot(sub["temperature"], sub["chi_target"], marker="o", linewidth=2, label=f"ϕ={phi:.1f}")
        if {"chi_target_boot_q025", "chi_target_boot_q975"}.issubset(sub.columns):
            lo = sub["chi_target_boot_q025"].to_numpy(dtype=float)
            hi = sub["chi_target_boot_q975"].to_numpy(dtype=float)
            if np.isfinite(lo).any() and np.isfinite(hi).any():
                ax.fill_between(sub["temperature"], lo, hi, alpha=0.15)

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("χ_target")
    ax.set_title("χ_target trend vs temperature")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "chi_target_vs_temperature_with_ci.png", dpi=dpi)
    plt.close(fig)


def _condition_stability_report(
    raw_df: pd.DataFrame,
    scan_df: pd.DataFrame,
    cond_best: pd.DataFrame,
    objective: str,
) -> pd.DataFrame:
    rows = []
    for (t, phi), sub in raw_df.groupby(["temperature", "phi"]):
        t = float(t)
        phi = float(phi)
        scan_sub = scan_df[(scan_df["temperature"] == t) & (scan_df["phi"] == phi)].copy()
        best_sub = cond_best[(cond_best["temperature"] == t) & (cond_best["phi"] == phi)].copy()
        if scan_sub.empty or best_sub.empty:
            continue

        max_obj = float(scan_sub[objective].max())
        tie_mask = np.isclose(scan_sub[objective].to_numpy(dtype=float), max_obj, atol=1e-12, rtol=1e-12)
        tie_df = scan_sub.loc[tie_mask].sort_values("threshold")

        chosen = float(best_sub["chi_target"].iloc[0])
        tie_thr_min = float(tie_df["threshold"].min()) if not tie_df.empty else np.nan
        tie_thr_max = float(tie_df["threshold"].max()) if not tie_df.empty else np.nan
        chosen_on_tie_edge = (
            bool(np.isclose(chosen, tie_thr_min, atol=1e-12, rtol=1e-12))
            or bool(np.isclose(chosen, tie_thr_max, atol=1e-12, rtol=1e-12))
        )

        ci_lo = float(best_sub.get("chi_target_boot_q025", pd.Series([np.nan])).iloc[0])
        ci_hi = float(best_sub.get("chi_target_boot_q975", pd.Series([np.nan])).iloc[0])
        ci_width = ci_hi - ci_lo if np.isfinite(ci_lo) and np.isfinite(ci_hi) else np.nan

        rows.append(
            {
                "temperature": t,
                "phi": phi,
                "n_samples": int(len(sub)),
                "n_soluble": int((sub["water_soluble"] == 1).sum()),
                "n_insoluble": int((sub["water_soluble"] == 0).sum()),
                "chosen_chi_target": chosen,
                "chosen_balanced_accuracy": float(best_sub["balanced_accuracy"].iloc[0]),
                "objective_max": max_obj,
                "n_tied_thresholds": int(len(tie_df)),
                "tie_threshold_min": tie_thr_min,
                "tie_threshold_max": tie_thr_max,
                "tie_threshold_span": (
                    float(tie_thr_max - tie_thr_min)
                    if np.isfinite(tie_thr_min) and np.isfinite(tie_thr_max)
                    else np.nan
                ),
                "chosen_on_tie_edge": bool(chosen_on_tie_edge),
                "chi_target_boot_q025": ci_lo,
                "chi_target_boot_q975": ci_hi,
                "chi_target_boot_ci_width": ci_width,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["temperature", "phi"]).reset_index(drop=True)


def _round_float_columns(df: pd.DataFrame, ndigits: int = 4) -> pd.DataFrame:
    out = df.copy()
    float_cols = out.select_dtypes(include=["float16", "float32", "float64"]).columns
    if len(float_cols) > 0:
        out.loc[:, float_cols] = out[float_cols].round(ndigits)
    return out


def main(args):
    config = load_config(args.config)
    chi_cfg = _default_chi_config(config)

    split_mode = str(chi_cfg["split_mode"]).strip().lower()
    if split_mode not in {"polymer", "random"}:
        raise ValueError("split_mode must be one of {'polymer','random'}")

    dataset_path = args.dataset_path or chi_cfg["dataset_path"]
    objective = args.objective or str(chi_cfg.get("target_objective", "balanced_accuracy"))
    if objective not in {"balanced_accuracy", "youden_j", "f1", "accuracy"}:
        raise ValueError(f"Unsupported objective: {objective}")

    n_bootstrap = int(args.bootstrap_repeats or chi_cfg.get("target_bootstrap_repeats", 800))
    if n_bootstrap < 0:
        raise ValueError("bootstrap_repeats must be >= 0")

    positive_when_low = True  # physical rule requested by user: soluble if chi <= chi_target

    results_dir = Path(config["paths"]["results_dir"])
    step_dir = results_dir / "step3_chi_target_learning" / split_mode
    metrics_dir = step_dir / "metrics"
    figures_dir = step_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["data"]["random_seed"])
    seed_info = seed_everything(seed)
    save_config(config, step_dir / "config_used.yaml")
    save_run_metadata(step_dir, args.config, seed_info)
    write_initial_log(
        step_dir=step_dir,
        step_name="step3_chi_target_learning",
        context={
            "config_path": args.config,
            "results_dir": str(results_dir),
            "split_mode": split_mode,
            "dataset_path": dataset_path,
            "objective": objective,
            "bootstrap_repeats": n_bootstrap,
            "include_insoluble_targets": args.include_insoluble_targets,
            "random_seed": seed,
        },
    )

    print("=" * 70)
    print("Step 3: χ_target learning from data + water-soluble labels")
    print(f"dataset={dataset_path}")
    print(f"objective={objective}")
    print(f"bootstrap_repeats={n_bootstrap}")
    print("rule: predict soluble when χ <= χ_target")
    print("=" * 70)

    df = load_chi_dataset(dataset_path)
    x_all = df["chi"].to_numpy(dtype=float)
    y_all = df["water_soluble"].to_numpy(dtype=int)

    # global scan
    global_scan = _scan_thresholds(x_all, y_all, positive_when_low=positive_when_low)
    global_best_row = _pick_best_threshold(global_scan, objective)
    global_boot = _bootstrap_threshold_ci(
        x_chi=x_all,
        y_label=y_all,
        objective=objective,
        positive_when_low=positive_when_low,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    global_best = pd.DataFrame([global_best_row.to_dict()])
    global_best.insert(0, "scope", "global")
    global_best = global_best.rename(columns={"threshold": "chi_target"})
    for k, v in global_boot.items():
        global_best[k] = v

    # condition-wise scan
    scan_rows = []
    best_rows = []
    for idx, ((t, phi), sub) in enumerate(df.groupby(["temperature", "phi"])):
        x = sub["chi"].to_numpy(dtype=float)
        y = sub["water_soluble"].to_numpy(dtype=int)
        scan = _scan_thresholds(x, y, positive_when_low=positive_when_low)
        if scan.empty:
            continue
        scan.insert(0, "temperature", float(t))
        scan.insert(1, "phi", float(phi))
        scan_rows.append(scan)

        best = _pick_best_threshold(scan.drop(columns=["temperature", "phi"]), objective)
        boot = _bootstrap_threshold_ci(
            x_chi=x,
            y_label=y,
            objective=objective,
            positive_when_low=positive_when_low,
            n_bootstrap=n_bootstrap,
            seed=seed + idx + 1,
        )
        row = {"temperature": float(t), "phi": float(phi), **best.to_dict(), **boot}
        best_rows.append(row)

    scan_df = pd.concat(scan_rows, ignore_index=True) if scan_rows else pd.DataFrame()
    cond_best = pd.DataFrame(best_rows)
    if cond_best.empty:
        raise RuntimeError("Failed to estimate condition-wise chi_target.")

    cond_best = cond_best.rename(columns={"threshold": "chi_target"})
    cond_best = cond_best.sort_values(["temperature", "phi"]).reset_index(drop=True)
    stability_df = _condition_stability_report(df, scan_df, cond_best, objective)

    # recommended targets for inverse design
    target_rows = []
    for _, row in cond_best.iterrows():
        target_rows.append(
            {
                "target_class": 1,
                "target_class_name": CLASS_NAME_MAP[1],
                "temperature": float(row["temperature"]),
                "phi": float(row["phi"]),
                "target_chi": float(row["chi_target"]),
                "property_rule": "upper_bound",
                "target_source": f"condition_{objective}",
                "quality_balanced_accuracy": float(row["balanced_accuracy"]),
                "quality_youden_j": float(row["youden_j"]),
                "chi_target_boot_q025": float(row.get("chi_target_boot_q025", np.nan)),
                "chi_target_boot_q975": float(row.get("chi_target_boot_q975", np.nan)),
            }
        )
        if args.include_insoluble_targets:
            target_rows.append(
                {
                    "target_class": 0,
                    "target_class_name": CLASS_NAME_MAP[0],
                    "temperature": float(row["temperature"]),
                    "phi": float(row["phi"]),
                    "target_chi": float(row["chi_target"]),
                    "property_rule": "lower_bound",
                    "target_source": f"condition_{objective}",
                    "quality_balanced_accuracy": float(row["balanced_accuracy"]),
                    "quality_youden_j": float(row["youden_j"]),
                    "chi_target_boot_q025": float(row.get("chi_target_boot_q025", np.nan)),
                    "chi_target_boot_q975": float(row.get("chi_target_boot_q975", np.nan)),
                }
            )

    targets_df = pd.DataFrame(target_rows)
    targets_df.insert(0, "target_id", np.arange(1, len(targets_df) + 1))

    # Save CSV metrics rounded to 4 decimals for readability.
    _round_float_columns(global_scan, ndigits=4).to_csv(metrics_dir / "chi_target_global_scan.csv", index=False)
    _round_float_columns(global_best, ndigits=4).to_csv(metrics_dir / "chi_target_global_best.csv", index=False)
    _round_float_columns(scan_df, ndigits=4).to_csv(metrics_dir / "chi_target_scan_by_condition.csv", index=False)
    _round_float_columns(cond_best, ndigits=4).to_csv(metrics_dir / "chi_target_best_by_condition.csv", index=False)
    _round_float_columns(stability_df, ndigits=4).to_csv(metrics_dir / "chi_target_condition_stability.csv", index=False)
    _round_float_columns(targets_df, ndigits=4).to_csv(metrics_dir / "chi_target_for_inverse_design.csv", index=False)

    summary = {
        "dataset_path": str(dataset_path),
        "objective": objective,
        "rule": "soluble_if_chi_leq_target",
        "bootstrap_repeats": int(n_bootstrap),
        "n_rows": int(len(df)),
        "n_polymers": int(df["Polymer"].nunique()),
        "n_soluble": int(df[df["water_soluble"] == 1]["Polymer"].nunique()),
        "n_insoluble": int(df[df["water_soluble"] == 0]["Polymer"].nunique()),
        "global_chi_target": float(global_best["chi_target"].iloc[0]),
        "global_balanced_accuracy": float(global_best["balanced_accuracy"].iloc[0]),
        "global_youden_j": float(global_best["youden_j"].iloc[0]),
        "global_chi_target_boot_q025": float(global_best["chi_target_boot_q025"].iloc[0]),
        "global_chi_target_boot_q975": float(global_best["chi_target_boot_q975"].iloc[0]),
        "n_conditions": int(cond_best.shape[0]),
        "mean_condition_balanced_accuracy": float(cond_best["balanced_accuracy"].mean()),
        "std_condition_balanced_accuracy": float(cond_best["balanced_accuracy"].std()),
        "n_conditions_with_tied_optimal_thresholds": int((stability_df["n_tied_thresholds"] > 1).sum()) if not stability_df.empty else 0,
        "mean_condition_bootstrap_ci_width": (
            float(stability_df["chi_target_boot_ci_width"].mean())
            if not stability_df.empty
            else np.nan
        ),
    }
    with open(metrics_dir / "chi_target_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    summary_csv = {
        k: (round(v, 4) if isinstance(v, float) and np.isfinite(v) else v)
        for k, v in summary.items()
    }
    save_step_summary(summary_csv, metrics_dir)

    # figures
    dpi = int(config.get("plotting", {}).get("dpi", 600))
    font_size = int(config.get("plotting", {}).get("font_size", 12))
    _make_figures(cond_best, global_best, df, objective, figures_dir, dpi=dpi, font_size=font_size)
    save_artifact_manifest(step_dir=step_dir, metrics_dir=metrics_dir, figures_dir=figures_dir)

    print("Step 3 complete.")
    print(f"Recommended targets: {metrics_dir / 'chi_target_for_inverse_design.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3: learn chi_target from labeled dataset")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config path")
    parser.add_argument("--dataset_path", type=str, default=None, help="Override path to chi dataset")
    parser.add_argument("--objective", type=str, default=None, choices=["balanced_accuracy", "youden_j", "f1", "accuracy"], help="Threshold selection objective")
    parser.add_argument("--bootstrap_repeats", type=int, default=None, help="Bootstrap repeats for threshold CI")
    parser.add_argument("--include_insoluble_targets", action="store_true", help="Also emit class=0 targets using lower_bound rule")
    args = parser.parse_args()
    main(args)
