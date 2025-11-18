"""
Uncertainty quantification using Monte Carlo Dropout.

Provides utilities for enabling dropout during inference and computing
prediction uncertainties through multiple forward passes.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger("polymer_chi_ml.uncertainty")


def enable_mc_dropout(model: nn.Module) -> None:
    """
    Enable Monte Carlo Dropout by keeping dropout active during inference.

    Sets all Dropout layers in the model to training mode while keeping
    other layers (BatchNorm, etc.) in evaluation mode.

    Args:
        model: PyTorch model

    Example:
        >>> model.eval()  # Set model to eval mode
        >>> enable_mc_dropout(model)  # Re-enable dropout layers
        >>> # Now model will use dropout during inference
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
    logger.debug("MC Dropout enabled: Dropout layers set to training mode")


def mc_predict(
    model: nn.Module,
    x: torch.Tensor,
    T_ref: float,
    n_samples: int = 50,
    device: str = "cuda",
    predict_solubility: bool = False,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Perform MC Dropout prediction with uncertainty estimation.

    Runs multiple forward passes with dropout enabled and computes mean
    and standard deviation of predictions.

    Args:
        model: PyTorch model (should be in eval mode)
        x: Input features, shape (batch_size, feature_dim)
        T_ref: Reference temperature for chi_RT prediction
        n_samples: Number of forward passes for MC sampling
        device: Device to run inference on
        predict_solubility: If True, also predict solubility

    Returns:
        Dictionary containing for each output:
            - (mean, std) tuples as numpy arrays
        Keys: 'chi_RT', 'A', 'B', and optionally 'p_soluble'

    Example:
        >>> model.eval()
        >>> predictions = mc_predict(
        ...     model, x_test, T_ref=298.0, n_samples=50, device="cuda"
        ... )
        >>> chi_mean, chi_std = predictions['chi_RT']
        >>> print(f"Chi prediction: {chi_mean[0]:.4f} ± {chi_std[0]:.4f}")
    """
    # Ensure model is in eval mode first
    model.eval()

    # Enable MC Dropout
    enable_mc_dropout(model)

    # Move tensors to device
    x = x.to(device)
    T_ref_tensor = torch.tensor(T_ref, device=device, dtype=x.dtype)

    # Storage for predictions
    batch_size = x.shape[0]
    chi_RT_samples = np.zeros((n_samples, batch_size))
    A_samples = np.zeros((n_samples, batch_size))
    B_samples = np.zeros((n_samples, batch_size))

    if predict_solubility:
        p_soluble_samples = np.zeros((n_samples, batch_size))

    # Run multiple forward passes
    with torch.no_grad():
        for i in range(n_samples):
            outputs = model(
                x,
                temperature=T_ref_tensor,
                predict_solubility=predict_solubility,
            )

            # Store predictions
            chi_RT_samples[i] = outputs["chi_RT"].cpu().numpy()
            A_samples[i] = outputs["A"].cpu().numpy()
            B_samples[i] = outputs["B"].cpu().numpy()

            if predict_solubility:
                p_soluble_samples[i] = outputs["p_soluble"].cpu().numpy()

    # Compute mean and std
    results = {
        "chi_RT": (
            np.mean(chi_RT_samples, axis=0),
            np.std(chi_RT_samples, axis=0),
        ),
        "A": (
            np.mean(A_samples, axis=0),
            np.std(A_samples, axis=0),
        ),
        "B": (
            np.mean(B_samples, axis=0),
            np.std(B_samples, axis=0),
        ),
    }

    if predict_solubility:
        results["p_soluble"] = (
            np.mean(p_soluble_samples, axis=0),
            np.std(p_soluble_samples, axis=0),
        )

    logger.debug(
        f"MC Dropout prediction completed with {n_samples} samples "
        f"for batch of size {batch_size}"
    )

    return results


def mc_predict_batch(
    model: nn.Module,
    dataloader: DataLoader,
    T_ref: float,
    n_samples: int = 50,
    device: str = "cuda",
    predict_solubility: bool = False,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Perform MC Dropout prediction for entire dataset using DataLoader.

    Args:
        model: PyTorch model (should be in eval mode)
        dataloader: DataLoader providing batches of input features
        T_ref: Reference temperature for chi_RT prediction
        n_samples: Number of forward passes for MC sampling
        device: Device to run inference on
        predict_solubility: If True, also predict solubility

    Returns:
        Dictionary containing for each output:
            - (mean, std) tuples as numpy arrays with shape (n_total_samples,)
        Keys: 'chi_RT', 'A', 'B', and optionally 'p_soluble'

    Example:
        >>> from torch.utils.data import DataLoader, TensorDataset
        >>> dataset = TensorDataset(x_test)
        >>> loader = DataLoader(dataset, batch_size=256)
        >>> predictions = mc_predict_batch(
        ...     model, loader, T_ref=298.0, n_samples=50
        ... )
        >>> chi_mean, chi_std = predictions['chi_RT']
    """
    # Ensure model is in eval mode first
    model.eval()

    # Enable MC Dropout
    enable_mc_dropout(model)

    # Storage for all predictions
    all_chi_RT_mean = []
    all_chi_RT_std = []
    all_A_mean = []
    all_A_std = []
    all_B_mean = []
    all_B_std = []

    if predict_solubility:
        all_p_soluble_mean = []
        all_p_soluble_std = []

    # Process batches
    for batch in dataloader:
        # Handle different dataloader formats
        if isinstance(batch, (list, tuple)):
            x_batch = batch[0]
        else:
            x_batch = batch

        # Get predictions for this batch
        batch_predictions = mc_predict(
            model,
            x_batch,
            T_ref=T_ref,
            n_samples=n_samples,
            device=device,
            predict_solubility=predict_solubility,
        )

        # Accumulate results
        chi_mean, chi_std = batch_predictions["chi_RT"]
        all_chi_RT_mean.append(chi_mean)
        all_chi_RT_std.append(chi_std)

        A_mean, A_std = batch_predictions["A"]
        all_A_mean.append(A_mean)
        all_A_std.append(A_std)

        B_mean, B_std = batch_predictions["B"]
        all_B_mean.append(B_mean)
        all_B_std.append(B_std)

        if predict_solubility:
            p_mean, p_std = batch_predictions["p_soluble"]
            all_p_soluble_mean.append(p_mean)
            all_p_soluble_std.append(p_std)

    # Concatenate all batches
    results = {
        "chi_RT": (
            np.concatenate(all_chi_RT_mean),
            np.concatenate(all_chi_RT_std),
        ),
        "A": (
            np.concatenate(all_A_mean),
            np.concatenate(all_A_std),
        ),
        "B": (
            np.concatenate(all_B_mean),
            np.concatenate(all_B_std),
        ),
    }

    if predict_solubility:
        results["p_soluble"] = (
            np.concatenate(all_p_soluble_mean),
            np.concatenate(all_p_soluble_std),
        )

    n_total = len(results["chi_RT"][0])
    logger.info(
        f"MC Dropout batch prediction completed for {n_total} samples "
        f"with {n_samples} forward passes each"
    )

    return results


def compute_uncertainty_metrics(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
) -> Dict[str, float]:
    """
    Compute metrics for uncertainty calibration analysis.

    Checks if predicted uncertainties correlate with prediction errors.

    Args:
        y_true: True values, shape (n_samples,)
        y_pred_mean: Mean predictions, shape (n_samples,)
        y_pred_std: Prediction standard deviations, shape (n_samples,)

    Returns:
        Dictionary containing:
            - correlation: Spearman correlation between |error| and uncertainty
            - p_value: p-value for correlation test
            - mean_uncertainty: Mean of predicted uncertainties
            - std_uncertainty: Std of predicted uncertainties

    Example:
        >>> metrics = compute_uncertainty_metrics(y_true, chi_mean, chi_std)
        >>> print(f"Error-uncertainty correlation: {metrics['correlation']:.4f}")
    """
    # Filter NaN values
    valid_mask = ~(
        np.isnan(y_true) | np.isnan(y_pred_mean) | np.isnan(y_pred_std)
    )
    y_true_valid = y_true[valid_mask]
    y_mean_valid = y_pred_mean[valid_mask]
    y_std_valid = y_pred_std[valid_mask]

    if len(y_true_valid) < 2:
        return {
            "correlation": np.nan,
            "p_value": np.nan,
            "mean_uncertainty": np.nan,
            "std_uncertainty": np.nan,
        }

    # Compute absolute errors
    abs_errors = np.abs(y_mean_valid - y_true_valid)

    # Compute Spearman correlation between errors and uncertainties
    from scipy.stats import spearmanr

    try:
        correlation, p_value = spearmanr(abs_errors, y_std_valid)
    except Exception as e:
        logger.warning(f"Failed to compute uncertainty correlation: {e}")
        correlation = np.nan
        p_value = np.nan

    return {
        "correlation": float(correlation),
        "p_value": float(p_value),
        "mean_uncertainty": float(np.mean(y_std_valid)),
        "std_uncertainty": float(np.std(y_std_valid)),
    }


def calibration_bins_analysis(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    n_bins: int = 5,
) -> Dict[str, np.ndarray]:
    """
    Analyze uncertainty calibration by binning samples by predicted uncertainty.

    For well-calibrated uncertainties, high-uncertainty predictions should have
    higher errors.

    Args:
        y_true: True values, shape (n_samples,)
        y_pred_mean: Mean predictions, shape (n_samples,)
        y_pred_std: Prediction standard deviations, shape (n_samples,)
        n_bins: Number of uncertainty bins

    Returns:
        Dictionary containing:
            - bin_edges: Uncertainty bin edges
            - bin_means: Mean uncertainty in each bin
            - bin_errors: Mean absolute error in each bin
            - bin_counts: Number of samples in each bin

    Example:
        >>> analysis = calibration_bins_analysis(y_true, chi_mean, chi_std)
        >>> print("Uncertainty bins:", analysis['bin_means'])
        >>> print("Corresponding MAE:", analysis['bin_errors'])
    """
    # Filter NaN values
    valid_mask = ~(
        np.isnan(y_true) | np.isnan(y_pred_mean) | np.isnan(y_pred_std)
    )
    y_true_valid = y_true[valid_mask]
    y_mean_valid = y_pred_mean[valid_mask]
    y_std_valid = y_pred_std[valid_mask]

    # Compute absolute errors
    abs_errors = np.abs(y_mean_valid - y_true_valid)

    # Create bins based on uncertainty quantiles
    bin_edges = np.percentile(
        y_std_valid, np.linspace(0, 100, n_bins + 1)
    )

    # Analyze each bin
    bin_means = []
    bin_errors = []
    bin_counts = []

    for i in range(n_bins):
        if i == n_bins - 1:
            # Last bin includes right edge
            mask = (y_std_valid >= bin_edges[i]) & (y_std_valid <= bin_edges[i + 1])
        else:
            mask = (y_std_valid >= bin_edges[i]) & (y_std_valid < bin_edges[i + 1])

        if np.sum(mask) > 0:
            bin_means.append(np.mean(y_std_valid[mask]))
            bin_errors.append(np.mean(abs_errors[mask]))
            bin_counts.append(np.sum(mask))
        else:
            bin_means.append(np.nan)
            bin_errors.append(np.nan)
            bin_counts.append(0)

    return {
        "bin_edges": bin_edges,
        "bin_means": np.array(bin_means),
        "bin_errors": np.array(bin_errors),
        "bin_counts": np.array(bin_counts),
    }
