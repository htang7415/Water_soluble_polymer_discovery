"""
Publication-quality plotting utilities for model evaluation.

Provides visualization functions for regression and classification tasks,
following configuration settings for consistent styling.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from ..utils.config import Config
from .metrics import compute_regression_metrics

logger = logging.getLogger("polymer_chi_ml.plots")


def _setup_plot_style(config: Config) -> None:
    """
    Set up matplotlib style from config.

    Args:
        config: Configuration object with plotting settings
    """
    try:
        plt.style.use(config.plotting.style)
    except Exception as e:
        logger.warning(f"Failed to set plot style '{config.plotting.style}': {e}")
        # Fallback to default
        plt.style.use("default")

    # Set font size
    plt.rcParams.update({"font.size": config.plotting.font_size})


def _save_figure(
    fig: plt.Figure,
    save_path: Union[str, Path],
    config: Config,
) -> None:
    """
    Save figure in configured formats (PNG and/or PDF).

    Args:
        fig: Matplotlib figure
        save_path: Base path for saving (without extension)
        config: Configuration object
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    dpi = config.plotting.dpi

    if config.plotting.save_png:
        png_path = save_path.with_suffix(".png")
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved plot to {png_path}")

    if config.plotting.save_pdf:
        pdf_path = save_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        logger.info(f"Saved plot to {pdf_path}")


def plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Union[str, Path],
    config: Config,
    title: str = "Parity Plot",
    xlabel: str = r"$\chi$ (True)",
    ylabel: str = r"$\chi$ (Predicted)",
    show_metrics: bool = True,
) -> plt.Figure:
    """
    Create parity plot: predicted vs true values with y=x reference line.

    Args:
        y_true: True values, shape (n_samples,)
        y_pred: Predicted values, shape (n_samples,)
        save_path: Path to save figure (without extension)
        config: Configuration object
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        show_metrics: If True, annotate plot with metrics

    Returns:
        Matplotlib figure

    Example:
        >>> fig = plot_parity(chi_true, chi_pred, "results/parity.png", config)
    """
    _setup_plot_style(config)

    # Create figure
    fig, ax = plt.subplots(
        figsize=tuple(config.plotting.figure_size)
    )

    # Filter NaN values
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    if len(y_true_valid) == 0:
        logger.warning("No valid data to plot in parity plot")
        plt.close(fig)
        return fig

    # Scatter plot
    ax.scatter(
        y_true_valid,
        y_pred_valid,
        alpha=0.5,
        s=20,
        edgecolors="none",
    )

    # y=x reference line
    min_val = min(y_true_valid.min(), y_pred_valid.min())
    max_val = max(y_true_valid.max(), y_pred_valid.max())
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "k--",
        linewidth=1.5,
        label="y=x",
        zorder=10,
    )

    # Add metrics annotation
    if show_metrics:
        metrics = compute_regression_metrics(y_true_valid, y_pred_valid)
        metrics_text = (
            f"MAE = {metrics['mae']:.4f}\n"
            f"RMSE = {metrics['rmse']:.4f}\n"
            f"R² = {metrics['r2']:.4f}\n"
            f"Spearman r = {metrics['spearman_r']:.4f}\n"
            f"n = {metrics['n_samples']}"
        )
        ax.text(
            0.05,
            0.95,
            metrics_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=config.plotting.font_size - 1,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()

    # Save figure
    _save_figure(fig, save_path, config)

    return fig


def plot_parity_with_temperature(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    temperatures: np.ndarray,
    save_path: Union[str, Path],
    config: Config,
    title: str = r"Parity Plot Colored by Temperature",
    xlabel: str = r"$\chi$ (True)",
    ylabel: str = r"$\chi$ (Predicted)",
) -> plt.Figure:
    """
    Create parity plot with points colored by temperature.

    Args:
        y_true: True chi values, shape (n_samples,)
        y_pred: Predicted chi values, shape (n_samples,)
        temperatures: Temperature values in Kelvin, shape (n_samples,)
        save_path: Path to save figure (without extension)
        config: Configuration object
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label

    Returns:
        Matplotlib figure

    Example:
        >>> fig = plot_parity_with_temperature(
        ...     chi_true, chi_pred, T, "results/parity_T.png", config
        ... )
    """
    _setup_plot_style(config)

    # Create figure
    fig, ax = plt.subplots(
        figsize=tuple(config.plotting.figure_size)
    )

    # Filter NaN values
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(temperatures))
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    T_valid = temperatures[valid_mask]

    if len(y_true_valid) == 0:
        logger.warning("No valid data to plot in temperature parity plot")
        plt.close(fig)
        return fig

    # Scatter plot with temperature coloring
    scatter = ax.scatter(
        y_true_valid,
        y_pred_valid,
        c=T_valid,
        cmap=config.plotting.colormap,
        alpha=0.6,
        s=20,
        edgecolors="none",
    )

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Temperature (K)", rotation=270, labelpad=20)

    # y=x reference line
    min_val = min(y_true_valid.min(), y_pred_valid.min())
    max_val = max(y_true_valid.max(), y_pred_valid.max())
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "k--",
        linewidth=1.5,
        label="y=x",
        zorder=10,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()

    # Save figure
    _save_figure(fig, save_path, config)

    return fig


def plot_residual_vs_temperature(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    temperatures: np.ndarray,
    save_path: Union[str, Path],
    config: Config,
    title: str = "Residual vs Temperature",
    xlabel: str = "Temperature (K)",
    ylabel: str = r"Residual ($\chi_{pred} - \chi_{true}$)",
) -> plt.Figure:
    """
    Plot residuals (y_pred - y_true) vs temperature.

    Useful for identifying temperature-dependent biases in predictions.

    Args:
        y_true: True chi values, shape (n_samples,)
        y_pred: Predicted chi values, shape (n_samples,)
        temperatures: Temperature values in Kelvin, shape (n_samples,)
        save_path: Path to save figure (without extension)
        config: Configuration object
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label

    Returns:
        Matplotlib figure

    Example:
        >>> fig = plot_residual_vs_temperature(
        ...     chi_true, chi_pred, T, "results/residual_T.png", config
        ... )
    """
    _setup_plot_style(config)

    # Create figure
    fig, ax = plt.subplots(
        figsize=tuple(config.plotting.figure_size)
    )

    # Filter NaN values
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(temperatures))
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    T_valid = temperatures[valid_mask]

    if len(y_true_valid) == 0:
        logger.warning("No valid data to plot in residual plot")
        plt.close(fig)
        return fig

    # Compute residuals
    residuals = y_pred_valid - y_true_valid

    # Scatter plot
    ax.scatter(
        T_valid,
        residuals,
        alpha=0.5,
        s=20,
        edgecolors="none",
    )

    # Zero reference line
    ax.axhline(y=0, color="k", linestyle="--", linewidth=1.5, label="Zero error")

    # Add mean residual line
    mean_residual = np.mean(residuals)
    ax.axhline(
        y=mean_residual,
        color="r",
        linestyle="-",
        linewidth=1.5,
        alpha=0.7,
        label=f"Mean residual = {mean_residual:.4f}",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    _save_figure(fig, save_path, config)

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Union[str, Path],
    config: Config,
    title: str = "ROC Curve",
) -> plt.Figure:
    """
    Plot ROC (Receiver Operating Characteristic) curve.

    Args:
        y_true: True binary labels (0 or 1), shape (n_samples,)
        y_prob: Predicted probabilities for positive class, shape (n_samples,)
        save_path: Path to save figure (without extension)
        config: Configuration object
        title: Plot title

    Returns:
        Matplotlib figure

    Example:
        >>> fig = plot_roc_curve(y_true, y_prob, "results/roc.png", config)
    """
    _setup_plot_style(config)

    # Create figure
    fig, ax = plt.subplots(
        figsize=tuple(config.plotting.figure_size)
    )

    # Filter NaN values
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_prob))
    y_true_valid = y_true[valid_mask].astype(int)
    y_prob_valid = y_prob[valid_mask]

    if len(y_true_valid) == 0:
        logger.warning("No valid data to plot ROC curve")
        plt.close(fig)
        return fig

    # Check if both classes are present
    if len(np.unique(y_true_valid)) < 2:
        logger.warning("ROC curve requires both classes; cannot plot")
        plt.close(fig)
        return fig

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true_valid, y_prob_valid)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    ax.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"ROC curve (AUC = {roc_auc:.4f})",
    )

    # Diagonal reference (random classifier)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random classifier")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()

    # Save figure
    _save_figure(fig, save_path, config)

    return fig


def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Union[str, Path],
    config: Config,
    title: str = "Precision-Recall Curve",
) -> plt.Figure:
    """
    Plot Precision-Recall curve.

    Args:
        y_true: True binary labels (0 or 1), shape (n_samples,)
        y_prob: Predicted probabilities for positive class, shape (n_samples,)
        save_path: Path to save figure (without extension)
        config: Configuration object
        title: Plot title

    Returns:
        Matplotlib figure

    Example:
        >>> fig = plot_pr_curve(y_true, y_prob, "results/pr_curve.png", config)
    """
    _setup_plot_style(config)

    # Create figure
    fig, ax = plt.subplots(
        figsize=tuple(config.plotting.figure_size)
    )

    # Filter NaN values
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_prob))
    y_true_valid = y_true[valid_mask].astype(int)
    y_prob_valid = y_prob[valid_mask]

    if len(y_true_valid) == 0:
        logger.warning("No valid data to plot PR curve")
        plt.close(fig)
        return fig

    # Check if both classes are present
    if len(np.unique(y_true_valid)) < 2:
        logger.warning("PR curve requires both classes; cannot plot")
        plt.close(fig)
        return fig

    # Compute PR curve
    precision, recall, thresholds = precision_recall_curve(
        y_true_valid, y_prob_valid
    )
    pr_auc = auc(recall, precision)

    # Plot PR curve
    ax.plot(
        recall,
        precision,
        linewidth=2,
        label=f"PR curve (AUC = {pr_auc:.4f})",
    )

    # Baseline (random classifier) - proportion of positive class
    baseline = np.sum(y_true_valid) / len(y_true_valid)
    ax.axhline(
        y=baseline,
        color="k",
        linestyle="--",
        linewidth=1.5,
        label=f"Baseline (P={baseline:.3f})",
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()

    # Save figure
    _save_figure(fig, save_path, config)

    return fig


def plot_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Union[str, Path],
    config: Config,
    n_bins: int = 10,
    title: str = "Calibration Plot",
) -> plt.Figure:
    """
    Plot calibration (reliability) diagram.

    Shows whether predicted probabilities match actual frequencies.

    Args:
        y_true: True binary labels (0 or 1), shape (n_samples,)
        y_prob: Predicted probabilities, shape (n_samples,)
        save_path: Path to save figure (without extension)
        config: Configuration object
        n_bins: Number of bins for calibration curve
        title: Plot title

    Returns:
        Matplotlib figure

    Example:
        >>> fig = plot_calibration(y_true, y_prob, "results/calibration.png", config)
    """
    _setup_plot_style(config)

    # Create figure
    fig, ax = plt.subplots(
        figsize=tuple(config.plotting.figure_size)
    )

    # Filter NaN values
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_prob))
    y_true_valid = y_true[valid_mask].astype(int)
    y_prob_valid = y_prob[valid_mask]

    if len(y_true_valid) == 0:
        logger.warning("No valid data to plot calibration curve")
        plt.close(fig)
        return fig

    # Check if both classes are present
    if len(np.unique(y_true_valid)) < 2:
        logger.warning("Calibration curve requires both classes; cannot plot")
        plt.close(fig)
        return fig

    # Compute calibration curve
    try:
        prob_true, prob_pred = calibration_curve(
            y_true_valid,
            y_prob_valid,
            n_bins=n_bins,
            strategy="uniform",
        )

        # Plot calibration curve
        ax.plot(
            prob_pred,
            prob_true,
            marker="o",
            linewidth=2,
            label="Model",
        )

        # Perfect calibration reference
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")

        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])

    except Exception as e:
        logger.error(f"Failed to compute calibration curve: {e}")
        ax.text(
            0.5,
            0.5,
            "Failed to compute calibration",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    plt.tight_layout()

    # Save figure
    _save_figure(fig, save_path, config)

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: Union[str, Path],
    config: Config,
    labels: Optional[list] = None,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
) -> plt.Figure:
    """
    Plot confusion matrix as heatmap.

    Args:
        cm: Confusion matrix, shape (2, 2) for binary classification
        save_path: Path to save figure (without extension)
        config: Configuration object
        labels: Class labels (default: ['Insoluble', 'Soluble'])
        title: Plot title
        cmap: Colormap for heatmap

    Returns:
        Matplotlib figure

    Example:
        >>> from .metrics import compute_confusion_matrix
        >>> cm, counts = compute_confusion_matrix(y_true, y_prob)
        >>> fig = plot_confusion_matrix(cm, "results/confusion.png", config)
    """
    _setup_plot_style(config)

    if labels is None:
        labels = ["Insoluble", "Soluble"]

    # Create figure
    fig, ax = plt.subplots(
        figsize=tuple(config.plotting.figure_size)
    )

    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt=".0f" if cm.dtype == int else ".2f",
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={"label": "Count"},
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    plt.tight_layout()

    # Save figure
    _save_figure(fig, save_path, config)

    return fig


def plot_chi_rt_vs_solubility(
    chi_rt: np.ndarray,
    solubility_labels: np.ndarray,
    save_path: Union[str, Path],
    config: Config,
    plot_type: str = "box",
    title: str = r"$\chi_{RT}$ Distribution by Solubility Class",
) -> plt.Figure:
    """
    Plot chi_RT distribution separated by solubility class.

    Args:
        chi_rt: Chi at reference temperature, shape (n_samples,)
        solubility_labels: Binary solubility labels (0=insoluble, 1=soluble)
        save_path: Path to save figure (without extension)
        config: Configuration object
        plot_type: Type of plot - "box", "violin", or "strip"
        title: Plot title

    Returns:
        Matplotlib figure

    Example:
        >>> fig = plot_chi_rt_vs_solubility(
        ...     chi_rt, solubility, "results/chi_rt_distribution.png", config
        ... )
    """
    _setup_plot_style(config)

    # Create figure
    fig, ax = plt.subplots(
        figsize=tuple(config.plotting.figure_size)
    )

    # Filter NaN values
    valid_mask = ~(np.isnan(chi_rt) | np.isnan(solubility_labels))
    chi_rt_valid = chi_rt[valid_mask]
    sol_valid = solubility_labels[valid_mask].astype(int)

    if len(chi_rt_valid) == 0:
        logger.warning("No valid data to plot chi_RT distribution")
        plt.close(fig)
        return fig

    # Create labels for plot
    class_names = ["Insoluble", "Soluble"]
    plot_data = []
    plot_labels = []

    for class_val in [0, 1]:
        mask = sol_valid == class_val
        if np.sum(mask) > 0:
            plot_data.append(chi_rt_valid[mask])
            plot_labels.append(class_names[class_val])

    # Create plot based on type
    if plot_type == "box":
        ax.boxplot(
            plot_data,
            labels=plot_labels,
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor="lightblue", alpha=0.7),
        )
    elif plot_type == "violin":
        parts = ax.violinplot(
            plot_data,
            positions=range(len(plot_data)),
            widths=0.7,
            showmeans=True,
            showmedians=True,
        )
        ax.set_xticks(range(len(plot_labels)))
        ax.set_xticklabels(plot_labels)
    elif plot_type == "strip":
        # Strip plot using seaborn
        import pandas as pd

        df = pd.DataFrame(
            {
                "chi_RT": chi_rt_valid,
                "Solubility": [class_names[v] for v in sol_valid],
            }
        )
        sns.stripplot(
            data=df,
            x="Solubility",
            y="chi_RT",
            alpha=0.5,
            ax=ax,
        )
    else:
        logger.warning(f"Unknown plot_type '{plot_type}', using box plot")
        ax.boxplot(plot_data, labels=plot_labels, widths=0.6)

    ax.set_xlabel("Solubility Class")
    ax.set_ylabel(r"$\chi_{RT}$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    # Add statistical test (Mann-Whitney U)
    if len(plot_data) == 2 and len(plot_data[0]) > 0 and len(plot_data[1]) > 0:
        try:
            from scipy.stats import mannwhitneyu

            stat, p_value = mannwhitneyu(plot_data[0], plot_data[1])
            ax.text(
                0.95,
                0.95,
                f"Mann-Whitney U\np = {p_value:.4f}",
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                fontsize=config.plotting.font_size - 1,
            )
        except Exception as e:
            logger.warning(f"Failed to compute Mann-Whitney U test: {e}")

    plt.tight_layout()

    # Save figure
    _save_figure(fig, save_path, config)

    return fig
