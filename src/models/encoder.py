"""
Shared encoder network for polymer feature embedding.

Transforms input features (Morgan FP + descriptors) into a latent representation.
"""

import logging
from typing import List, Optional

import torch
import torch.nn as nn

from ..utils.config import Config

logger = logging.getLogger("polymer_chi_ml.encoder")


class Encoder(nn.Module):
    """
    Shared encoder MLP that maps polymer features to latent representation.

    Architecture (configurable):
        Input (input_dim) -> Hidden layers -> Latent (latent_dim)

    Each hidden layer includes:
        - Linear transformation
        - Optional BatchNorm
        - Activation (ReLU, ELU, or LeakyReLU)
        - Dropout

    Args:
        input_dim: Input feature dimension
        config: Configuration object containing encoder hyperparameters

    Example:
        >>> config = load_config("configs/config.yaml")
        >>> encoder = Encoder(input_dim=2061, config=config)
        >>> z = encoder(x)  # x: (batch, 2061) -> z: (batch, 128)
    """

    def __init__(self, input_dim: int, config: Config):
        """Initialize encoder with configuration."""
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = config.model.encoder_hidden_dims
        self.latent_dim = config.model.encoder_latent_dim
        self.dropout_p = config.model.encoder_dropout
        self.activation_name = config.model.encoder_activation
        self.use_batchnorm = config.model.encoder_use_batchnorm

        # Build activation function
        self.activation = self._build_activation(self.activation_name)

        # Build network layers
        layers = []
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if self.use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(self.activation)
            layers.append(nn.Dropout(p=self.dropout_p))

            prev_dim = hidden_dim

        # Final layer to latent dimension (no dropout after this)
        layers.append(nn.Linear(prev_dim, self.latent_dim))
        layers.append(self.activation)

        self.network = nn.Sequential(*layers)

        # Log architecture
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"Encoder initialized: {input_dim} -> {self.hidden_dims} -> {self.latent_dim}, "
            f"{n_params:,} parameters"
        )

    def _build_activation(self, name: str) -> nn.Module:
        """
        Build activation function from name.

        Args:
            name: Activation name ('relu', 'elu', 'leaky_relu')

        Returns:
            Activation module
        """
        name = name.lower()

        if name == "relu":
            return nn.ReLU()
        elif name == "elu":
            return nn.ELU()
        elif name == "leaky_relu":
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            logger.warning(f"Unknown activation '{name}', defaulting to ReLU")
            return nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.

        Args:
            x: Input features, shape (batch_size, input_dim)

        Returns:
            z: Latent representation, shape (batch_size, latent_dim)
        """
        return self.network(x)

    def get_output_dim(self) -> int:
        """Get latent dimension (output size)."""
        return self.latent_dim
