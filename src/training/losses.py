"""
Multi-task loss functions for chi(T) prediction and solubility classification.

Implements:
- MSE loss for DFT chi (using chi_pred = A/T_ref + B)
- MSE loss for experimental chi (using chi_pred = A/T + B)
- Weighted BCE loss for solubility classification
- Combined multi-task loss with lambda weights
- Masking logic for missing labels
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.config import Config

logger = logging.getLogger("polymer_chi_ml.losses")


def chi_dft_loss(
    A: torch.Tensor,
    B: torch.Tensor,
    chi_true: torch.Tensor,
    temperature: Optional[torch.Tensor] = None,
    T_ref: Optional[float] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    MSE loss for DFT chi prediction at given temperature(s).

    Computes chi_pred = A/T + B and compares with chi_true.
    Can use either a tensor of temperatures (one per sample) or a single T_ref float.

    Args:
        A: Predicted temperature coefficient, shape (batch_size,)
        B: Predicted constant term, shape (batch_size,)
        chi_true: True DFT chi values, shape (batch_size,)
        temperature: Temperature tensor for each sample, shape (batch_size,). Takes precedence if provided.
        T_ref: Single reference temperature (float). Used if temperature is None. Defaults to 298.0.
        mask: Optional boolean mask for valid samples, shape (batch_size,)

    Returns:
        loss: Mean squared error loss (scalar)

    Example:
        >>> # Using per-sample temperatures
        >>> loss = chi_dft_loss(A, B, chi_true, temperature=temp_tensor)
        >>> # Using single reference temperature
        >>> loss = chi_dft_loss(A, B, chi_true, T_ref=298.0)
    """
    # Determine which temperature to use
    if temperature is not None:
        # Use per-sample temperatures
        T = temperature
    elif T_ref is not None:
        # Use single reference temperature
        T = T_ref
    else:
        # Default to 298 K
        T = 298.0

    # Compute predicted chi at temperature(s)
    chi_pred = A / T + B

    # Apply mask if provided
    if mask is not None:
        chi_pred = chi_pred[mask]
        chi_true = chi_true[mask]

    # MSE loss
    loss = F.mse_loss(chi_pred, chi_true)

    return loss


def chi_exp_loss(
    A: torch.Tensor,
    B: torch.Tensor,
    chi_true: torch.Tensor,
    temperature: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    MSE loss for experimental chi prediction at given temperatures.

    Computes chi_pred = A/T + B for each sample's temperature and compares with chi_true.

    Args:
        A: Predicted temperature coefficient, shape (batch_size,)
        B: Predicted constant term, shape (batch_size,)
        chi_true: True experimental chi values, shape (batch_size,)
        temperature: Temperature in Kelvin for each sample, shape (batch_size,)
        mask: Optional boolean mask for valid samples, shape (batch_size,)

    Returns:
        loss: Mean squared error loss (scalar)

    Example:
        >>> loss = chi_exp_loss(A, B, chi_true, temperature)
    """
    # Compute predicted chi at given temperatures
    chi_pred = A / temperature + B

    # Apply mask if provided
    if mask is not None:
        chi_pred = chi_pred[mask]
        chi_true = chi_true[mask]

    # MSE loss
    loss = F.mse_loss(chi_pred, chi_true)

    return loss


def solubility_loss(
    p_soluble: torch.Tensor,
    soluble_true: torch.Tensor,
    class_weight_pos: float = 1.0,
    class_weight_neg: float = 1.0,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Weighted binary cross-entropy loss for solubility classification.

    Handles class imbalance using per-class weights.

    Args:
        p_soluble: Predicted probability of being soluble, shape (batch_size,)
        soluble_true: True binary labels (1=soluble, 0=insoluble), shape (batch_size,)
        class_weight_pos: Weight for positive class (soluble)
        class_weight_neg: Weight for negative class (insoluble)
        mask: Optional boolean mask for valid samples, shape (batch_size,)

    Returns:
        loss: Weighted binary cross-entropy loss (scalar)

    Example:
        >>> loss = solubility_loss(p_soluble, soluble_true,
        ...                        class_weight_pos=2.0, class_weight_neg=1.0)
    """
    # Apply mask if provided
    if mask is not None:
        p_soluble = p_soluble[mask]
        soluble_true = soluble_true[mask]

    # Compute per-sample weights based on true labels
    # weights[i] = class_weight_pos if soluble_true[i] == 1 else class_weight_neg
    weights = torch.where(
        soluble_true == 1,
        torch.tensor(class_weight_pos, device=soluble_true.device, dtype=torch.float32),
        torch.tensor(class_weight_neg, device=soluble_true.device, dtype=torch.float32),
    )

    # Binary cross-entropy loss with weights
    loss = F.binary_cross_entropy(p_soluble, soluble_true, weight=weights)

    return loss


def multitask_loss(
    outputs: Dict[str, torch.Tensor],
    chi_dft_true: Optional[torch.Tensor] = None,
    chi_exp_true: Optional[torch.Tensor] = None,
    temperature_exp: Optional[torch.Tensor] = None,
    soluble_true: Optional[torch.Tensor] = None,
    config: Optional[Config] = None,
    lambda_dft: float = 0.5,
    lambda_exp: float = 2.0,
    lambda_sol: float = 1.0,
    class_weight_pos: float = 2.0,
    class_weight_neg: float = 1.0,
    T_ref: float = 298.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combined multi-task loss for DFT chi + experimental chi + solubility.

    Each task is optional (controlled by providing corresponding labels).
    Uses masking to handle missing labels within a batch.

    Args:
        outputs: Model outputs dictionary containing 'A', 'B', and optionally 'p_soluble'
        chi_dft_true: True DFT chi values, shape (batch_size,) or None
        chi_exp_true: True experimental chi values, shape (batch_size,) or None
        temperature_exp: Temperature for exp chi, shape (batch_size,) or None
        soluble_true: True solubility labels, shape (batch_size,) or None
        config: Config object (if provided, overrides lambda and weight parameters)
        lambda_dft: Weight for DFT chi loss
        lambda_exp: Weight for experimental chi loss
        lambda_sol: Weight for solubility loss
        class_weight_pos: Weight for positive class in solubility
        class_weight_neg: Weight for negative class in solubility
        T_ref: Reference temperature for DFT chi

    Returns:
        total_loss: Weighted sum of task losses (scalar)
        loss_dict: Dictionary of individual task losses for logging

    Example:
        >>> # Multi-task training with all tasks
        >>> total_loss, losses = multitask_loss(
        ...     outputs, chi_dft_true, chi_exp_true, temperature_exp,
        ...     soluble_true, config=config
        ... )
        >>>
        >>> # Training with only DFT chi (Stage 1)
        >>> total_loss, losses = multitask_loss(
        ...     outputs, chi_dft_true=chi_dft_true, config=config
        ... )
    """
    # Extract model outputs
    A = outputs["A"]
    B = outputs["B"]

    # Use config parameters if provided
    if config is not None:
        lambda_dft = config.loss_weights.lambda_dft
        lambda_exp = config.loss_weights.lambda_exp
        lambda_sol = config.loss_weights.lambda_sol
        class_weight_pos = config.solubility.class_weight_pos
        class_weight_neg = config.solubility.class_weight_neg
        T_ref = config.model.T_ref_K

    loss_dict = {}
    total_loss = torch.tensor(0.0, device=A.device, dtype=A.dtype)

    # DFT chi loss
    if chi_dft_true is not None:
        # Create mask for non-NaN values
        mask_dft = ~torch.isnan(chi_dft_true)

        if mask_dft.any():
            loss_dft = chi_dft_loss(A, B, chi_dft_true, T_ref=T_ref, mask=mask_dft)
            total_loss = total_loss + lambda_dft * loss_dft
            loss_dict["loss_dft"] = loss_dft.item()
        else:
            loss_dict["loss_dft"] = 0.0

    # Experimental chi loss
    if chi_exp_true is not None and temperature_exp is not None:
        # Create mask for non-NaN values
        mask_exp = ~torch.isnan(chi_exp_true)

        if mask_exp.any():
            loss_exp = chi_exp_loss(A, B, chi_exp_true, temperature_exp, mask=mask_exp)
            total_loss = total_loss + lambda_exp * loss_exp
            loss_dict["loss_exp"] = loss_exp.item()
        else:
            loss_dict["loss_exp"] = 0.0

    # Solubility loss
    if soluble_true is not None:
        # Ensure p_soluble is in outputs
        if "p_soluble" not in outputs:
            raise ValueError(
                "Model outputs missing 'p_soluble'. "
                "Set predict_solubility=True in model forward pass."
            )

        p_soluble = outputs["p_soluble"]

        # Create mask for non-NaN values
        mask_sol = ~torch.isnan(soluble_true)

        if mask_sol.any():
            loss_sol = solubility_loss(
                p_soluble, soluble_true,
                class_weight_pos=class_weight_pos,
                class_weight_neg=class_weight_neg,
                mask=mask_sol
            )
            total_loss = total_loss + lambda_sol * loss_sol
            loss_dict["loss_sol"] = loss_sol.item()
        else:
            loss_dict["loss_sol"] = 0.0

    # Add total loss to dict
    loss_dict["loss_total"] = total_loss.item()

    return total_loss, loss_dict


def compute_metrics_regression(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute regression metrics (MAE, RMSE, R²).

    Args:
        y_pred: Predicted values, shape (n,)
        y_true: True values, shape (n,)
        mask: Optional boolean mask for valid samples

    Returns:
        Dictionary of metrics: mae, rmse, r2
    """
    # Apply mask
    if mask is not None:
        y_pred = y_pred[mask]
        y_true = y_true[mask]

    # Move to CPU for numpy operations
    y_pred = y_pred.detach().cpu()
    y_true = y_true.detach().cpu()

    # Compute metrics
    mae = torch.mean(torch.abs(y_pred - y_true)).item()
    rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()

    # R² score
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def compute_metrics_classification(
    p_pred: torch.Tensor,
    y_true: torch.Tensor,
    threshold: float = 0.5,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute classification metrics (accuracy, precision, recall, F1, ROC-AUC).

    Args:
        p_pred: Predicted probabilities, shape (n,)
        y_true: True binary labels, shape (n,)
        threshold: Decision threshold for converting probabilities to labels
        mask: Optional boolean mask for valid samples

    Returns:
        Dictionary of metrics: accuracy, precision, recall, f1, roc_auc
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    # Apply mask
    if mask is not None:
        p_pred = p_pred[mask]
        y_true = y_true[mask]

    # Move to CPU and convert to numpy
    p_pred_np = p_pred.detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy()

    # Convert probabilities to binary predictions
    y_pred_np = (p_pred_np >= threshold).astype(int)

    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(y_true_np, y_pred_np),
        "precision": precision_score(y_true_np, y_pred_np, zero_division=0),
        "recall": recall_score(y_true_np, y_pred_np, zero_division=0),
        "f1": f1_score(y_true_np, y_pred_np, zero_division=0),
    }

    # ROC-AUC (only if both classes present)
    if len(set(y_true_np)) > 1:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true_np, p_pred_np)
        except ValueError:
            metrics["roc_auc"] = 0.0
    else:
        metrics["roc_auc"] = 0.0

    return metrics
