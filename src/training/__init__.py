"""
Training module for polymer-water interaction ML models.

Provides loss functions and training scripts for:
- Stage 1: DFT chi pretraining
- Stage 2: Multi-task fine-tuning (DFT chi + exp chi + solubility)
- Cross-validation for experimental chi
"""

from .losses import (
    chi_dft_loss,
    chi_exp_loss,
    solubility_loss,
    multitask_loss,
)

__all__ = [
    "chi_dft_loss",
    "chi_exp_loss",
    "solubility_loss",
    "multitask_loss",
]
