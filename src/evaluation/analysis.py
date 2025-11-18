"""
Analysis utilities for polymer-water interaction predictions.

Provides functions for analyzing relationships between chi, solubility,
A-parameter distributions, and uncertainty calibration.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import compute_classification_metrics, compute_regression_metrics
from .uncertainty import (
    calibration_bins_analysis,
    compute_uncertainty_metrics,
)

logger = logging.getLogger("polymer_chi_ml.analysis")


def analyze_chi_solubility_relationship(
    chi_rt: np.ndarray,
    solubility_labels: np.ndarray,
    chi_rt_pred: Optional[np.ndarray] = None,
) -> Dict[str, Union[float, int, Dict]]:
    """
    Analyze the relationship between chi_RT and solubility.

    Computes statistics and statistical tests to understand how chi at
    reference temperature correlates with binary solubility.

    Args:
        chi_rt: Chi at reference temperature (true values), shape (n_samples,)
        solubility_labels: Binary solubility labels (0=insoluble, 1=soluble)
        chi_rt_pred: Optional predicted chi_RT for additional analysis

    Returns:
        Dictionary containing:
            - n_samples: Total number of samples
            - n_soluble: Number of soluble samples
            - n_insoluble: Number of insoluble samples
            - chi_soluble_mean: Mean chi_RT for soluble samples
            - chi_soluble_std: Std chi_RT for soluble samples
            - chi_insoluble_mean: Mean chi_RT for insoluble samples
            - chi_insoluble_std: Std chi_RT for insoluble samples
            - mann_whitney_u: Mann-Whitney U test statistic
            - mann_whitney_p: p-value for Mann-Whitney U test
            - effect_size: Cohen's d effect size
            - point_biserial_r: Point-biserial correlation
            - point_biserial_p: p-value for point-biserial correlation
            - predicted_analysis: (if chi_rt_pred provided) similar stats for predictions

    Example:
        >>> analysis = analyze_chi_solubility_relationship(
        ...     chi_rt_true, solubility_labels, chi_rt_pred
        ... )
        >>> print(f"Chi_RT difference p-value: {analysis['mann_whitney_p']:.4e}")
    """
    # Filter NaN values
    valid_mask = ~(np.isnan(chi_rt) | np.isnan(solubility_labels))
    chi_rt_valid = chi_rt[valid_mask]
    sol_valid = solubility_labels[valid_mask].astype(int)

    n_samples = len(chi_rt_valid)
    n_soluble = np.sum(sol_valid == 1)
    n_insoluble = np.sum(sol_valid == 0)

    results = {
        "n_samples": int(n_samples),
        "n_soluble": int(n_soluble),
        "n_insoluble": int(n_insoluble),
    }

    if n_samples == 0:
        logger.warning("No valid samples for chi-solubility analysis")
        return results

    # Split by solubility class
    chi_soluble = chi_rt_valid[sol_valid == 1]
    chi_insoluble = chi_rt_valid[sol_valid == 0]

    # Descriptive statistics
    if len(chi_soluble) > 0:
        results["chi_soluble_mean"] = float(np.mean(chi_soluble))
        results["chi_soluble_std"] = float(np.std(chi_soluble))
        results["chi_soluble_median"] = float(np.median(chi_soluble))
        results["chi_soluble_min"] = float(np.min(chi_soluble))
        results["chi_soluble_max"] = float(np.max(chi_soluble))
    else:
        results["chi_soluble_mean"] = np.nan
        results["chi_soluble_std"] = np.nan
        results["chi_soluble_median"] = np.nan
        results["chi_soluble_min"] = np.nan
        results["chi_soluble_max"] = np.nan

    if len(chi_insoluble) > 0:
        results["chi_insoluble_mean"] = float(np.mean(chi_insoluble))
        results["chi_insoluble_std"] = float(np.std(chi_insoluble))
        results["chi_insoluble_median"] = float(np.median(chi_insoluble))
        results["chi_insoluble_min"] = float(np.min(chi_insoluble))
        results["chi_insoluble_max"] = float(np.max(chi_insoluble))
    else:
        results["chi_insoluble_mean"] = np.nan
        results["chi_insoluble_std"] = np.nan
        results["chi_insoluble_median"] = np.nan
        results["chi_insoluble_min"] = np.nan
        results["chi_insoluble_max"] = np.nan

    # Mann-Whitney U test (non-parametric test for difference in distributions)
    if len(chi_soluble) > 0 and len(chi_insoluble) > 0:
        try:
            u_stat, p_value = stats.mannwhitneyu(
                chi_soluble, chi_insoluble, alternative="two-sided"
            )
            results["mann_whitney_u"] = float(u_stat)
            results["mann_whitney_p"] = float(p_value)
        except Exception as e:
            logger.warning(f"Failed to compute Mann-Whitney U test: {e}")
            results["mann_whitney_u"] = np.nan
            results["mann_whitney_p"] = np.nan

        # Effect size (Cohen's d)
        try:
            pooled_std = np.sqrt(
                (
                    (len(chi_soluble) - 1) * np.var(chi_soluble, ddof=1)
                    + (len(chi_insoluble) - 1) * np.var(chi_insoluble, ddof=1)
                )
                / (len(chi_soluble) + len(chi_insoluble) - 2)
            )
            if pooled_std > 0:
                cohens_d = (np.mean(chi_soluble) - np.mean(chi_insoluble)) / pooled_std
                results["effect_size_cohens_d"] = float(cohens_d)
            else:
                results["effect_size_cohens_d"] = np.nan
        except Exception as e:
            logger.warning(f"Failed to compute Cohen's d: {e}")
            results["effect_size_cohens_d"] = np.nan
    else:
        results["mann_whitney_u"] = np.nan
        results["mann_whitney_p"] = np.nan
        results["effect_size_cohens_d"] = np.nan

    # Point-biserial correlation (correlation between continuous and binary variable)
    if len(chi_rt_valid) > 1:
        try:
            r_pb, p_value = stats.pointbiserialr(sol_valid, chi_rt_valid)
            results["point_biserial_r"] = float(r_pb)
            results["point_biserial_p"] = float(p_value)
        except Exception as e:
            logger.warning(f"Failed to compute point-biserial correlation: {e}")
            results["point_biserial_r"] = np.nan
            results["point_biserial_p"] = np.nan
    else:
        results["point_biserial_r"] = np.nan
        results["point_biserial_p"] = np.nan

    # If predicted chi_RT provided, compute similar analysis
    if chi_rt_pred is not None:
        pred_mask = valid_mask & ~np.isnan(chi_rt_pred)
        if np.sum(pred_mask) > 0:
            chi_rt_pred_valid = chi_rt_pred[pred_mask]
            sol_pred_valid = solubility_labels[pred_mask].astype(int)

            chi_pred_soluble = chi_rt_pred_valid[sol_pred_valid == 1]
            chi_pred_insoluble = chi_rt_pred_valid[sol_pred_valid == 0]

            pred_analysis = {
                "n_samples": int(len(chi_rt_pred_valid)),
                "chi_soluble_mean": (
                    float(np.mean(chi_pred_soluble))
                    if len(chi_pred_soluble) > 0
                    else np.nan
                ),
                "chi_insoluble_mean": (
                    float(np.mean(chi_pred_insoluble))
                    if len(chi_pred_insoluble) > 0
                    else np.nan
                ),
            }

            # Statistical test on predictions
            if len(chi_pred_soluble) > 0 and len(chi_pred_insoluble) > 0:
                try:
                    u_stat, p_value = stats.mannwhitneyu(
                        chi_pred_soluble, chi_pred_insoluble, alternative="two-sided"
                    )
                    pred_analysis["mann_whitney_p"] = float(p_value)
                except Exception:
                    pred_analysis["mann_whitney_p"] = np.nan
            else:
                pred_analysis["mann_whitney_p"] = np.nan

            results["predicted_analysis"] = pred_analysis

    return results


def analyze_A_sign_distribution(
    A_pred: np.ndarray,
    A_true: Optional[np.ndarray] = None,
) -> Dict[str, Union[int, float, Dict]]:
    """
    Analyze the distribution of A parameter signs.

    The sign of A determines phase behavior:
    - A > 0: UCST (Upper Critical Solution Temperature) behavior
    - A < 0: LCST (Lower Critical Solution Temperature) behavior

    Args:
        A_pred: Predicted A parameters, shape (n_samples,)
        A_true: Optional true A parameters for comparison

    Returns:
        Dictionary containing:
            - n_samples: Total number of samples
            - n_positive: Number of positive A (UCST)
            - n_negative: Number of negative A (LCST)
            - n_near_zero: Number near zero (|A| < 1)
            - mean_A: Mean A value
            - std_A: Std of A values
            - median_A: Median A value
            - positive_fraction: Fraction with A > 0
            - true_comparison: (if A_true provided) agreement analysis

    Example:
        >>> analysis = analyze_A_sign_distribution(A_pred, A_true)
        >>> print(f"UCST fraction: {analysis['positive_fraction']:.2%}")
    """
    # Filter NaN values
    valid_mask = ~np.isnan(A_pred)
    A_pred_valid = A_pred[valid_mask]

    n_samples = len(A_pred_valid)

    results = {
        "n_samples": int(n_samples),
    }

    if n_samples == 0:
        logger.warning("No valid samples for A sign distribution analysis")
        return results

    # Count signs
    n_positive = np.sum(A_pred_valid > 0)
    n_negative = np.sum(A_pred_valid < 0)
    n_near_zero = np.sum(np.abs(A_pred_valid) < 1)

    results.update(
        {
            "n_positive_A": int(n_positive),
            "n_negative_A": int(n_negative),
            "n_near_zero_A": int(n_near_zero),
            "positive_fraction": float(n_positive / n_samples),
            "negative_fraction": float(n_negative / n_samples),
            "near_zero_fraction": float(n_near_zero / n_samples),
        }
    )

    # Descriptive statistics
    results.update(
        {
            "mean_A": float(np.mean(A_pred_valid)),
            "std_A": float(np.std(A_pred_valid)),
            "median_A": float(np.median(A_pred_valid)),
            "min_A": float(np.min(A_pred_valid)),
            "max_A": float(np.max(A_pred_valid)),
            "q25_A": float(np.percentile(A_pred_valid, 25)),
            "q75_A": float(np.percentile(A_pred_valid, 75)),
        }
    )

    # If true A provided, analyze agreement
    if A_true is not None:
        both_valid_mask = valid_mask & ~np.isnan(A_true)
        if np.sum(both_valid_mask) > 0:
            A_true_valid = A_true[both_valid_mask]
            A_pred_comp = A_pred[both_valid_mask]

            # Sign agreement
            sign_true = np.sign(A_true_valid)
            sign_pred = np.sign(A_pred_comp)
            sign_agreement = np.mean(sign_true == sign_pred)

            # Correlation
            try:
                pearson_r, pearson_p = stats.pearsonr(A_true_valid, A_pred_comp)
                spearman_r, spearman_p = stats.spearmanr(A_true_valid, A_pred_comp)
            except Exception as e:
                logger.warning(f"Failed to compute A correlations: {e}")
                pearson_r = pearson_p = np.nan
                spearman_r = spearman_p = np.nan

            true_comparison = {
                "n_samples": int(np.sum(both_valid_mask)),
                "sign_agreement": float(sign_agreement),
                "pearson_r": float(pearson_r),
                "pearson_p": float(pearson_p),
                "spearman_r": float(spearman_r),
                "spearman_p": float(spearman_p),
                "mae": float(np.mean(np.abs(A_pred_comp - A_true_valid))),
            }

            results["true_comparison"] = true_comparison

    return results


def analyze_uncertainty_calibration(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    n_bins: int = 5,
) -> Dict[str, Union[float, Dict]]:
    """
    Analyze uncertainty calibration.

    Checks if predicted uncertainties are well-calibrated by testing whether
    high uncertainty predictions have higher errors.

    Args:
        y_true: True values, shape (n_samples,)
        y_pred_mean: Mean predictions, shape (n_samples,)
        y_pred_std: Prediction standard deviations, shape (n_samples,)
        n_bins: Number of bins for calibration analysis

    Returns:
        Dictionary containing:
            - correlation: Spearman correlation between |error| and uncertainty
            - p_value: p-value for correlation
            - mean_uncertainty: Mean predicted uncertainty
            - std_uncertainty: Std of predicted uncertainties
            - bins_analysis: Binned calibration analysis

    Example:
        >>> analysis = analyze_uncertainty_calibration(
        ...     chi_true, chi_mean, chi_std, n_bins=5
        ... )
        >>> print(f"Calibration correlation: {analysis['correlation']:.4f}")
    """
    # Compute overall uncertainty metrics
    uncertainty_metrics = compute_uncertainty_metrics(
        y_true, y_pred_mean, y_pred_std
    )

    # Perform binned analysis
    bins_analysis = calibration_bins_analysis(
        y_true, y_pred_mean, y_pred_std, n_bins=n_bins
    )

    # Combine results
    results = {
        "correlation": uncertainty_metrics["correlation"],
        "p_value": uncertainty_metrics["p_value"],
        "mean_uncertainty": uncertainty_metrics["mean_uncertainty"],
        "std_uncertainty": uncertainty_metrics["std_uncertainty"],
        "bins_analysis": {
            "bin_edges": bins_analysis["bin_edges"].tolist(),
            "bin_means": bins_analysis["bin_means"].tolist(),
            "bin_errors": bins_analysis["bin_errors"].tolist(),
            "bin_counts": bins_analysis["bin_counts"].tolist(),
        },
    }

    return results


def create_results_summary(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Optional[Union[str, Path]] = None,
    pretty_print: bool = True,
) -> str:
    """
    Create comprehensive results summary from metrics dictionary.

    Args:
        metrics_dict: Nested dictionary of metrics
            e.g., {"dft": {...}, "exp": {...}, "solubility": {...}}
        save_path: Optional path to save summary as JSON
        pretty_print: If True, format JSON with indentation

    Returns:
        JSON string of formatted results

    Example:
        >>> metrics = {
        ...     "dft_test": compute_regression_metrics(y_true, y_pred),
        ...     "solubility_test": compute_classification_metrics(y_true, y_prob),
        ... }
        >>> summary = create_results_summary(metrics, "results/summary.json")
    """
    # Create formatted summary
    summary = {
        "metrics": metrics_dict,
        "summary_statistics": {},
    }

    # Extract key metrics for quick reference
    for dataset_name, metrics in metrics_dict.items():
        if "mae" in metrics and "r2" in metrics:
            # Regression metrics
            summary["summary_statistics"][dataset_name] = {
                "type": "regression",
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "n_samples": metrics.get("n_samples", 0),
            }
        elif "roc_auc" in metrics:
            # Classification metrics
            summary["summary_statistics"][dataset_name] = {
                "type": "classification",
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "n_samples": metrics.get("n_samples", 0),
            }

    # Convert to JSON string
    if pretty_print:
        json_str = json.dumps(summary, indent=2, default=_json_serializer)
    else:
        json_str = json.dumps(summary, default=_json_serializer)

    # Save if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            f.write(json_str)

        logger.info(f"Results summary saved to {save_path}")

    return json_str


def _json_serializer(obj):
    """Custom JSON serializer for numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return str(obj)


def create_latex_table(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Optional[Union[str, Path]] = None,
    caption: str = "Model Performance Metrics",
    label: str = "tab:metrics",
) -> str:
    """
    Create LaTeX table from metrics dictionary for publication.

    Args:
        metrics_dict: Nested dictionary of metrics
        save_path: Optional path to save LaTeX table
        caption: Table caption
        label: Table label for referencing

    Returns:
        LaTeX table string

    Example:
        >>> metrics = {
        ...     "DFT Test": {"mae": 0.05, "rmse": 0.08, "r2": 0.92},
        ...     "Exp CV": {"mae": 0.12, "rmse": 0.18, "r2": 0.78},
        ... }
        >>> latex = create_latex_table(metrics, "results/table.tex")
    """
    # Start LaTeX table
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
    ]

    # Determine columns based on metrics present
    sample_metrics = next(iter(metrics_dict.values()))
    is_regression = "mae" in sample_metrics
    is_classification = "roc_auc" in sample_metrics

    if is_regression:
        lines.extend(
            [
                "\\begin{tabular}{lcccc}",
                "\\toprule",
                "Dataset & MAE & RMSE & $R^2$ & N \\\\",
                "\\midrule",
            ]
        )

        for dataset_name, metrics in metrics_dict.items():
            mae = metrics.get("mae", np.nan)
            rmse = metrics.get("rmse", np.nan)
            r2 = metrics.get("r2", np.nan)
            n = metrics.get("n_samples", 0)

            lines.append(
                f"{dataset_name} & {mae:.4f} & {rmse:.4f} & {r2:.4f} & {n} \\\\"
            )

    elif is_classification:
        lines.extend(
            [
                "\\begin{tabular}{lccccc}",
                "\\toprule",
                "Dataset & ROC-AUC & PR-AUC & Accuracy & F1 & N \\\\",
                "\\midrule",
            ]
        )

        for dataset_name, metrics in metrics_dict.items():
            roc_auc = metrics.get("roc_auc", np.nan)
            pr_auc = metrics.get("pr_auc", np.nan)
            accuracy = metrics.get("accuracy", np.nan)
            f1 = metrics.get("f1", np.nan)
            n = metrics.get("n_samples", 0)

            lines.append(
                f"{dataset_name} & {roc_auc:.4f} & {pr_auc:.4f} & "
                f"{accuracy:.4f} & {f1:.4f} & {n} \\\\"
            )

    # Close table
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )

    latex_str = "\n".join(lines)

    # Save if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            f.write(latex_str)

        logger.info(f"LaTeX table saved to {save_path}")

    return latex_str
