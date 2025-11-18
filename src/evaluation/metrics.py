"""
Metrics for evaluating regression and classification performance.

Provides comprehensive evaluation metrics with robust handling of edge cases,
NaN values, and empty data.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

logger = logging.getLogger("polymer_chi_ml.metrics")


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Compute comprehensive regression metrics.

    Handles edge cases gracefully:
    - Empty arrays return NaN for all metrics
    - Arrays with NaN values are filtered before computation
    - Constant predictions (zero variance) return NaN for R²

    Args:
        y_true: True values, shape (n_samples,)
        y_pred: Predicted values, shape (n_samples,)
        prefix: Optional prefix for metric names (e.g., "dft_" -> "dft_mae")

    Returns:
        Dictionary containing:
            - mae: Mean Absolute Error
            - rmse: Root Mean Squared Error
            - r2: R² (coefficient of determination)
            - spearman_r: Spearman rank correlation coefficient
            - spearman_p: p-value for Spearman correlation
            - n_samples: Number of valid samples used

    Example:
        >>> metrics = compute_regression_metrics(y_true, y_pred, prefix="test_")
        >>> print(f"Test MAE: {metrics['test_mae']:.4f}")
    """
    # Validate inputs
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Shape mismatch: y_true has {len(y_true)} samples, "
            f"y_pred has {len(y_pred)} samples"
        )

    # Handle empty arrays
    if len(y_true) == 0:
        logger.warning("Empty arrays provided to compute_regression_metrics")
        return {
            f"{prefix}mae": np.nan,
            f"{prefix}rmse": np.nan,
            f"{prefix}r2": np.nan,
            f"{prefix}spearman_r": np.nan,
            f"{prefix}spearman_p": np.nan,
            f"{prefix}n_samples": 0,
        }

    # Filter out NaN values
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    n_valid = np.sum(valid_mask)

    if n_valid == 0:
        logger.warning("All values are NaN in compute_regression_metrics")
        return {
            f"{prefix}mae": np.nan,
            f"{prefix}rmse": np.nan,
            f"{prefix}r2": np.nan,
            f"{prefix}spearman_r": np.nan,
            f"{prefix}spearman_p": np.nan,
            f"{prefix}n_samples": 0,
        }

    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    # Compute metrics
    mae = mean_absolute_error(y_true_valid, y_pred_valid)
    mse = mean_squared_error(y_true_valid, y_pred_valid)
    rmse = np.sqrt(mse)

    # R² - handle constant predictions
    try:
        r2 = r2_score(y_true_valid, y_pred_valid)
        # R² can be negative for very poor fits; clip to reasonable range for display
        # but keep actual value for analysis
    except Exception as e:
        logger.warning(f"Failed to compute R²: {e}")
        r2 = np.nan

    # Spearman correlation - requires at least 2 samples
    if n_valid >= 2:
        try:
            spearman_r, spearman_p = stats.spearmanr(y_true_valid, y_pred_valid)
        except Exception as e:
            logger.warning(f"Failed to compute Spearman correlation: {e}")
            spearman_r = np.nan
            spearman_p = np.nan
    else:
        spearman_r = np.nan
        spearman_p = np.nan

    return {
        f"{prefix}mae": float(mae),
        f"{prefix}rmse": float(rmse),
        f"{prefix}r2": float(r2),
        f"{prefix}spearman_r": float(spearman_r),
        f"{prefix}spearman_p": float(spearman_p),
        f"{prefix}n_samples": int(n_valid),
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Compute comprehensive binary classification metrics.

    Handles edge cases gracefully:
    - Empty arrays return NaN for all metrics
    - Arrays with NaN values are filtered before computation
    - Single-class data returns NaN for metrics requiring both classes

    Args:
        y_true: True binary labels (0 or 1), shape (n_samples,)
        y_prob: Predicted probabilities for positive class, shape (n_samples,)
        threshold: Decision threshold for converting probabilities to labels
        prefix: Optional prefix for metric names

    Returns:
        Dictionary containing:
            - roc_auc: Area Under ROC Curve
            - pr_auc: Area Under Precision-Recall Curve (average precision)
            - accuracy: Classification accuracy
            - balanced_accuracy: Balanced accuracy (average of recall per class)
            - precision: Precision for positive class
            - recall: Recall for positive class (sensitivity, TPR)
            - f1: F1 score for positive class
            - mcc: Matthews Correlation Coefficient
            - brier: Brier score (mean squared error of probability estimates)
            - n_samples: Number of valid samples
            - n_positive: Number of positive samples
            - n_negative: Number of negative samples

    Example:
        >>> metrics = compute_classification_metrics(y_true, y_prob, prefix="val_")
        >>> print(f"Validation ROC-AUC: {metrics['val_roc_auc']:.4f}")
    """
    # Validate inputs
    y_true = np.asarray(y_true).flatten()
    y_prob = np.asarray(y_prob).flatten()

    if len(y_true) != len(y_prob):
        raise ValueError(
            f"Shape mismatch: y_true has {len(y_true)} samples, "
            f"y_prob has {len(y_prob)} samples"
        )

    # Handle empty arrays
    if len(y_true) == 0:
        logger.warning("Empty arrays provided to compute_classification_metrics")
        return _empty_classification_metrics(prefix)

    # Filter out NaN values
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_prob))
    n_valid = np.sum(valid_mask)

    if n_valid == 0:
        logger.warning("All values are NaN in compute_classification_metrics")
        return _empty_classification_metrics(prefix)

    y_true_valid = y_true[valid_mask].astype(int)
    y_prob_valid = y_prob[valid_mask]

    # Convert probabilities to binary predictions
    y_pred_valid = (y_prob_valid >= threshold).astype(int)

    # Count classes
    n_positive = np.sum(y_true_valid == 1)
    n_negative = np.sum(y_true_valid == 0)

    # Initialize metrics dictionary
    metrics = {
        f"{prefix}n_samples": int(n_valid),
        f"{prefix}n_positive": int(n_positive),
        f"{prefix}n_negative": int(n_negative),
    }

    # Check if we have both classes (required for some metrics)
    has_both_classes = (n_positive > 0) and (n_negative > 0)

    # ROC-AUC requires both classes
    if has_both_classes:
        try:
            roc_auc = roc_auc_score(y_true_valid, y_prob_valid)
        except Exception as e:
            logger.warning(f"Failed to compute ROC-AUC: {e}")
            roc_auc = np.nan
    else:
        logger.warning("ROC-AUC requires both classes; setting to NaN")
        roc_auc = np.nan

    metrics[f"{prefix}roc_auc"] = float(roc_auc)

    # PR-AUC (average precision) requires both classes
    if has_both_classes:
        try:
            pr_auc = average_precision_score(y_true_valid, y_prob_valid)
        except Exception as e:
            logger.warning(f"Failed to compute PR-AUC: {e}")
            pr_auc = np.nan
    else:
        logger.warning("PR-AUC requires both classes; setting to NaN")
        pr_auc = np.nan

    metrics[f"{prefix}pr_auc"] = float(pr_auc)

    # Brier score (always computable)
    try:
        brier = brier_score_loss(y_true_valid, y_prob_valid)
    except Exception as e:
        logger.warning(f"Failed to compute Brier score: {e}")
        brier = np.nan

    metrics[f"{prefix}brier"] = float(brier)

    # Accuracy (always computable)
    try:
        accuracy = accuracy_score(y_true_valid, y_pred_valid)
    except Exception as e:
        logger.warning(f"Failed to compute accuracy: {e}")
        accuracy = np.nan

    metrics[f"{prefix}accuracy"] = float(accuracy)

    # Balanced accuracy requires both classes
    if has_both_classes:
        try:
            balanced_acc = balanced_accuracy_score(y_true_valid, y_pred_valid)
        except Exception as e:
            logger.warning(f"Failed to compute balanced accuracy: {e}")
            balanced_acc = np.nan
    else:
        balanced_acc = np.nan

    metrics[f"{prefix}balanced_accuracy"] = float(balanced_acc)

    # Precision, recall, F1 - use zero_division parameter for edge cases
    try:
        precision = precision_score(
            y_true_valid, y_pred_valid, zero_division=0
        )
    except Exception as e:
        logger.warning(f"Failed to compute precision: {e}")
        precision = np.nan

    metrics[f"{prefix}precision"] = float(precision)

    try:
        recall = recall_score(
            y_true_valid, y_pred_valid, zero_division=0
        )
    except Exception as e:
        logger.warning(f"Failed to compute recall: {e}")
        recall = np.nan

    metrics[f"{prefix}recall"] = float(recall)

    try:
        f1 = f1_score(
            y_true_valid, y_pred_valid, zero_division=0
        )
    except Exception as e:
        logger.warning(f"Failed to compute F1: {e}")
        f1 = np.nan

    metrics[f"{prefix}f1"] = float(f1)

    # Matthews Correlation Coefficient - requires both classes
    if has_both_classes:
        try:
            mcc = matthews_corrcoef(y_true_valid, y_pred_valid)
        except Exception as e:
            logger.warning(f"Failed to compute MCC: {e}")
            mcc = np.nan
    else:
        mcc = np.nan

    metrics[f"{prefix}mcc"] = float(mcc)

    return metrics


def _empty_classification_metrics(prefix: str) -> Dict[str, float]:
    """Return dictionary with NaN values for all classification metrics."""
    return {
        f"{prefix}roc_auc": np.nan,
        f"{prefix}pr_auc": np.nan,
        f"{prefix}accuracy": np.nan,
        f"{prefix}balanced_accuracy": np.nan,
        f"{prefix}precision": np.nan,
        f"{prefix}recall": np.nan,
        f"{prefix}f1": np.nan,
        f"{prefix}mcc": np.nan,
        f"{prefix}brier": np.nan,
        f"{prefix}n_samples": 0,
        f"{prefix}n_positive": 0,
        f"{prefix}n_negative": 0,
    }


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    normalize: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Compute confusion matrix with optional normalization.

    Args:
        y_true: True binary labels (0 or 1), shape (n_samples,)
        y_prob: Predicted probabilities, shape (n_samples,)
        threshold: Decision threshold
        normalize: Normalization mode - 'true', 'pred', 'all', or None

    Returns:
        Tuple of:
            - cm: Confusion matrix, shape (2, 2)
                  [[TN, FP],
                   [FN, TP]]
            - counts: Dictionary with 'TP', 'TN', 'FP', 'FN' counts

    Example:
        >>> cm, counts = compute_confusion_matrix(y_true, y_prob)
        >>> print(f"True Positives: {counts['TP']}")
    """
    # Validate inputs
    y_true = np.asarray(y_true).flatten()
    y_prob = np.asarray(y_prob).flatten()

    # Filter out NaN values
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_prob))
    y_true_valid = y_true[valid_mask].astype(int)
    y_prob_valid = y_prob[valid_mask]

    # Convert probabilities to predictions
    y_pred_valid = (y_prob_valid >= threshold).astype(int)

    # Compute confusion matrix
    cm = confusion_matrix(
        y_true_valid,
        y_pred_valid,
        labels=[0, 1],
        normalize=normalize,
    )

    # Extract counts (from unnormalized matrix)
    if normalize is not None:
        cm_counts = confusion_matrix(y_true_valid, y_pred_valid, labels=[0, 1])
    else:
        cm_counts = cm

    tn, fp, fn, tp = cm_counts.ravel()

    counts = {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    return cm, counts
