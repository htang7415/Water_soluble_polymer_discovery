"""
Evaluation module for polymer-water interaction ML models.

Provides comprehensive metrics, plotting utilities, uncertainty quantification,
and analysis tools for model evaluation.
"""

from .metrics import (
    compute_regression_metrics,
    compute_classification_metrics,
    compute_confusion_matrix,
)
from .plots import (
    plot_parity,
    plot_parity_with_temperature,
    plot_residual_vs_temperature,
    plot_roc_curve,
    plot_pr_curve,
    plot_calibration,
    plot_confusion_matrix,
    plot_chi_rt_vs_solubility,
)
from .uncertainty import (
    enable_mc_dropout,
    mc_predict,
    mc_predict_batch,
)
from .analysis import (
    analyze_chi_solubility_relationship,
    analyze_A_sign_distribution,
    analyze_uncertainty_calibration,
    create_results_summary,
)

__all__ = [
    # Metrics
    "compute_regression_metrics",
    "compute_classification_metrics",
    "compute_confusion_matrix",
    # Plots
    "plot_parity",
    "plot_parity_with_temperature",
    "plot_residual_vs_temperature",
    "plot_roc_curve",
    "plot_pr_curve",
    "plot_calibration",
    "plot_confusion_matrix",
    "plot_chi_rt_vs_solubility",
    # Uncertainty
    "enable_mc_dropout",
    "mc_predict",
    "mc_predict_batch",
    # Analysis
    "analyze_chi_solubility_relationship",
    "analyze_A_sign_distribution",
    "analyze_uncertainty_calibration",
    "create_results_summary",
]
