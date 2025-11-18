"""
Multi-task model for polymer-water interaction prediction.

Components:
- Shared encoder: polymer features -> latent representation z
- Chi head: z -> (A, B) for chi(T) = A/T + B
- Solubility head: [z, chi_RT] -> P(soluble)
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..utils.config import Config
from .encoder import Encoder

logger = logging.getLogger("polymer_chi_ml.multitask")


class ChiHead(nn.Module):
    """
    Chi(T) prediction head: outputs A and B such that chi(T) = A/T + B.

    Args:
        input_dim: Input dimension (latent_dim from encoder)
        config: Configuration object

    Example:
        >>> head = ChiHead(input_dim=128, config=config)
        >>> A, B = head(z)  # z: (batch, 128) -> A, B: (batch,)
    """

    def __init__(self, input_dim: int, config: Config):
        """Initialize chi head."""
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = config.model.chi_head_hidden_dim
        self.dropout_p = config.model.chi_head_dropout
        self.activation_name = config.model.chi_head_activation

        # Build activation
        self.activation = self._build_activation(self.activation_name)

        # Network: latent -> hidden -> 2 outputs (A, B)
        self.network = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            self.activation,
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.hidden_dim, 2),  # Output: [A, B]
        )

        logger.debug(f"ChiHead initialized: {input_dim} -> {self.hidden_dim} -> 2")

    def _build_activation(self, name: str) -> nn.Module:
        """Build activation function from name."""
        name = name.lower()
        if name == "relu":
            return nn.ReLU()
        elif name == "elu":
            return nn.ELU()
        elif name == "leaky_relu":
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            return nn.ReLU()

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through chi head.

        Args:
            z: Latent representation, shape (batch_size, input_dim)

        Returns:
            A: Temperature coefficient, shape (batch_size,)
            B: Constant term, shape (batch_size,)
        """
        output = self.network(z)  # (batch, 2)
        A = output[:, 0]  # (batch,)
        B = output[:, 1]  # (batch,)
        return A, B


class SolubilityHead(nn.Module):
    """
    Solubility prediction head: takes [z, chi_RT] and outputs P(soluble).

    Args:
        latent_dim: Latent dimension from encoder
        config: Configuration object

    Example:
        >>> head = SolubilityHead(latent_dim=128, config=config)
        >>> p_soluble = head(z, chi_RT)  # (batch, 128), (batch,) -> (batch,)
    """

    def __init__(self, latent_dim: int, config: Config):
        """Initialize solubility head."""
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = config.model.sol_head_hidden_dim
        self.dropout_p = config.model.sol_head_dropout
        self.activation_name = config.model.sol_head_activation

        # Input: [z, chi_RT] -> dimension is latent_dim + 1
        self.input_dim = latent_dim + 1

        # Build activation
        self.activation = self._build_activation(self.activation_name)

        # Network: [z, chi_RT] -> hidden -> 1 (logit)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            self.activation,
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.hidden_dim, 1),  # Output: logit
            nn.Sigmoid(),  # Convert to probability
        )

        logger.debug(
            f"SolubilityHead initialized: {self.input_dim} -> {self.hidden_dim} -> 1"
        )

    def _build_activation(self, name: str) -> nn.Module:
        """Build activation function from name."""
        name = name.lower()
        if name == "relu":
            return nn.ReLU()
        elif name == "elu":
            return nn.ELU()
        elif name == "leaky_relu":
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            return nn.ReLU()

    def forward(self, z: torch.Tensor, chi_RT: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through solubility head.

        Args:
            z: Latent representation, shape (batch_size, latent_dim)
            chi_RT: Chi at reference temperature, shape (batch_size,)

        Returns:
            p_soluble: Probability of being soluble, shape (batch_size,)
        """
        # Concatenate z and chi_RT
        chi_RT = chi_RT.unsqueeze(1)  # (batch, 1)
        x_concat = torch.cat([z, chi_RT], dim=1)  # (batch, latent_dim + 1)

        # Forward pass
        p_soluble = self.network(x_concat).squeeze(1)  # (batch,)

        return p_soluble


class MultiTaskChiSolubilityModel(nn.Module):
    """
    Multi-task model for chi(T) and solubility prediction.

    Components:
    - Encoder: x -> z
    - Chi head: z -> (A, B), where chi(T) = A/T + B
    - Solubility head: [z, chi_RT] -> P(soluble)

    Args:
        input_dim: Feature dimension
        config: Configuration object

    Example:
        >>> model = MultiTaskChiSolubilityModel(input_dim=2061, config=config)
        >>> outputs = model(x, temperature=298.0, predict_solubility=True)
        >>> chi_pred = outputs['chi']
        >>> p_soluble = outputs['p_soluble']
    """

    def __init__(self, input_dim: int, config: Config):
        """Initialize multi-task model."""
        super().__init__()

        self.input_dim = input_dim
        self.config = config
        self.T_ref = config.model.T_ref_K

        # Build components
        self.encoder = Encoder(input_dim, config)
        self.chi_head = ChiHead(self.encoder.get_output_dim(), config)
        self.solubility_head = SolubilityHead(self.encoder.get_output_dim(), config)

        # Log model size
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"MultiTaskChiSolubilityModel initialized with {n_params:,} parameters"
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode polymer features to latent representation.

        Args:
            x: Feature tensor, shape (batch_size, input_dim)

        Returns:
            z: Latent representation, shape (batch_size, latent_dim)
        """
        return self.encoder(x)

    def predict_AB(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict A and B parameters for chi(T) = A/T + B.

        Args:
            x: Feature tensor, shape (batch_size, input_dim)

        Returns:
            A: Temperature coefficient, shape (batch_size,)
            B: Constant term, shape (batch_size,)
        """
        z = self.encode(x)
        A, B = self.chi_head(z)
        return A, B

    def predict_chi(self, x: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Predict chi at given temperature(s).

        Args:
            x: Feature tensor, shape (batch_size, input_dim)
            T: Temperature in Kelvin, shape (batch_size,) or scalar

        Returns:
            chi: Predicted chi values, shape (batch_size,)
        """
        A, B = self.predict_AB(x)

        # Ensure T has the right shape
        if T.dim() == 0:  # Scalar
            T = T.expand_as(A)

        chi = A / T + B
        return chi

    def predict_chi_RT(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict chi at reference temperature T_ref.

        Args:
            x: Feature tensor, shape (batch_size, input_dim)

        Returns:
            chi_RT: Predicted chi at T_ref, shape (batch_size,)
        """
        T_ref_tensor = torch.tensor(self.T_ref, device=x.device, dtype=x.dtype)
        return self.predict_chi(x, T_ref_tensor)

    def predict_solubility(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict solubility probability.

        Internally computes chi_RT and passes [z, chi_RT] to solubility head.

        Args:
            x: Feature tensor, shape (batch_size, input_dim)

        Returns:
            p_soluble: Probability of being soluble, shape (batch_size,)
        """
        z = self.encode(x)
        chi_RT = self.predict_chi_RT(x)
        p_soluble = self.solubility_head(z, chi_RT)
        return p_soluble

    def forward(
        self,
        x: torch.Tensor,
        temperature: Optional[torch.Tensor] = None,
        predict_solubility: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Unified forward pass for multi-task prediction.

        Args:
            x: Feature tensor, shape (batch_size, input_dim)
            temperature: Temperature for chi prediction, shape (batch_size,) or scalar.
                        If None, uses T_ref.
            predict_solubility: If True, also predict solubility

        Returns:
            Dictionary containing:
                - 'z': Latent representation (batch_size, latent_dim)
                - 'A': Temperature coefficient (batch_size,)
                - 'B': Constant term (batch_size,)
                - 'chi': Predicted chi at given temperature (batch_size,)
                - 'chi_RT': Predicted chi at T_ref (batch_size,)
                - 'p_soluble': (optional) Solubility probability (batch_size,)

        Example:
            >>> # For DFT chi prediction
            >>> outputs = model(x, temperature=T_dft)
            >>> chi_pred = outputs['chi']
            >>>
            >>> # For multi-task prediction
            >>> outputs = model(x, temperature=T_exp, predict_solubility=True)
            >>> chi_pred = outputs['chi']
            >>> p_soluble = outputs['p_soluble']
        """
        # Encode
        z = self.encode(x)

        # Predict A, B
        A, B = self.chi_head(z)

        # Determine temperature for chi prediction
        if temperature is None:
            temperature = torch.tensor(self.T_ref, device=x.device, dtype=x.dtype)

        # Ensure temperature has the right shape
        if temperature.dim() == 0:  # Scalar
            temperature = temperature.expand(x.size(0))

        # Predict chi at given temperature
        chi = A / temperature + B

        # Predict chi at T_ref (always compute for potential solubility prediction)
        T_ref_tensor = torch.tensor(self.T_ref, device=x.device, dtype=x.dtype)
        chi_RT = A / T_ref_tensor + B

        # Build output dictionary
        outputs = {
            "z": z,
            "A": A,
            "B": B,
            "chi": chi,
            "chi_RT": chi_RT,
        }

        # Optionally predict solubility
        if predict_solubility:
            p_soluble = self.solubility_head(z, chi_RT)
            outputs["p_soluble"] = p_soluble

        return outputs

    def freeze_encoder(self) -> None:
        """Freeze encoder parameters (for fine-tuning)."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder parameters frozen")

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        logger.info("Encoder parameters unfrozen")

    def freeze_chi_head(self) -> None:
        """Freeze chi head parameters."""
        for param in self.chi_head.parameters():
            param.requires_grad = False
        logger.info("Chi head parameters frozen")

    def unfreeze_chi_head(self) -> None:
        """Unfreeze chi head parameters."""
        for param in self.chi_head.parameters():
            param.requires_grad = True
        logger.info("Chi head parameters unfrozen")

    def load_encoder_and_chi_head(self, checkpoint_path: str) -> None:
        """
        Load pretrained encoder and chi head weights from Stage 1.

        Args:
            checkpoint_path: Path to Stage 1 checkpoint

        Example:
            >>> model.load_encoder_and_chi_head("results/dft_pretrain/best_model.pt")
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Load encoder and chi_head (solubility_head will be randomly initialized)
        # Filter out solubility_head parameters
        encoder_chi_state = {
            k: v for k, v in state_dict.items()
            if k.startswith("encoder.") or k.startswith("chi_head.")
        }

        # Load with strict=False to allow missing solubility_head
        missing_keys, unexpected_keys = self.load_state_dict(
            encoder_chi_state, strict=False
        )

        logger.info(
            f"Loaded encoder and chi_head from {checkpoint_path}. "
            f"Missing keys: {len(missing_keys)} (expected: solubility_head), "
            f"Unexpected keys: {len(unexpected_keys)}"
        )
