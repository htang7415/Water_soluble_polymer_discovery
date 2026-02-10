"""Physics-guided model for polymer-water chi(T, phi) prediction."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


COEFF_NAMES = ["a0", "a1", "a2", "a3", "b1", "b2"]


def _validate_formula_inputs_torch(temperature: torch.Tensor) -> None:
    if torch.any(~torch.isfinite(temperature)):
        raise ValueError("temperature contains non-finite values")
    if torch.any(temperature <= 0):
        raise ValueError("temperature must be > 0 for chi(T, phi) formula")


def _validate_formula_inputs_numpy(temperature) -> None:
    import numpy as np

    arr = np.asarray(temperature, dtype=float)
    if np.any(~np.isfinite(arr)):
        raise ValueError("temperature contains non-finite values")
    if np.any(arr <= 0):
        raise ValueError("temperature must be > 0 for chi(T, phi) formula")



def _build_mlp(input_dim: int, hidden_sizes: Iterable[int], dropout: float) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = input_dim
    for h in hidden_sizes:
        layers.extend([
            nn.Linear(prev, h),
            nn.ReLU(),
            nn.Dropout(dropout),
        ])
        prev = h
    return nn.Sequential(*layers) if layers else nn.Sequential(nn.Identity())



def chi_formula_torch(coefficients: torch.Tensor, temperature: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Compute chi(T, phi) from coefficients.

    coefficients: [..., 6] order = a0,a1,a2,a3,b1,b2
    temperature: [...]
    phi: [...]
    """
    _validate_formula_inputs_torch(temperature)
    a0, a1, a2, a3, b1, b2 = [coefficients[..., i] for i in range(6)]
    base = a0 + a1 / temperature + a2 * torch.log(temperature) + a3 * temperature
    one_minus_phi = 1.0 - phi
    modifier = 1.0 + b1 * one_minus_phi + b2 * (one_minus_phi ** 2)
    return base * modifier



def predict_chi_from_coefficients(coefficients, temperature, phi):
    """Numpy-friendly chi formula wrapper used in analysis scripts."""
    import numpy as np

    _validate_formula_inputs_numpy(temperature)
    coeffs = np.asarray(coefficients)
    temperature = np.asarray(temperature)
    phi = np.asarray(phi)

    a0 = coeffs[..., 0]
    a1 = coeffs[..., 1]
    a2 = coeffs[..., 2]
    a3 = coeffs[..., 3]
    b1 = coeffs[..., 4]
    b2 = coeffs[..., 5]

    base = a0 + a1 / temperature + a2 * np.log(temperature) + a3 * temperature
    one_minus_phi = 1.0 - phi
    modifier = 1.0 + b1 * one_minus_phi + b2 * (one_minus_phi ** 2)
    return base * modifier


class PhysicsGuidedChiModel(nn.Module):
    """Map polymer embedding to chi coefficients + class logit."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_sizes: List[int] | Tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.hidden_sizes = list(hidden_sizes)
        self.dropout = float(dropout)

        self.encoder = _build_mlp(self.embedding_dim, self.hidden_sizes, self.dropout)
        head_dim = self.embedding_dim if not self.hidden_sizes else self.hidden_sizes[-1]
        self.coeff_head = nn.Linear(head_dim, 6)
        self.class_head = nn.Linear(head_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, embedding: torch.Tensor, temperature: torch.Tensor, phi: torch.Tensor):
        features = self.encoder(embedding)
        coeffs = self.coeff_head(features)
        class_logit = self.class_head(features).squeeze(-1)
        chi_pred = chi_formula_torch(coeffs, temperature, phi)
        return {
            "coefficients": coeffs,
            "class_logit": class_logit,
            "chi_pred": chi_pred,
        }

    def compute_loss(
        self,
        embedding: torch.Tensor,
        temperature: torch.Tensor,
        phi: torch.Tensor,
        chi_true: torch.Tensor,
        class_label: torch.Tensor,
        lambda_bce: float = 0.1,
    ):
        out = self.forward(embedding, temperature, phi)
        mse = F.mse_loss(out["chi_pred"], chi_true)
        bce = F.binary_cross_entropy_with_logits(out["class_logit"], class_label)
        total = mse + float(lambda_bce) * bce
        out.update({"loss": total, "loss_mse": mse, "loss_bce": bce})
        return out


class SolubilityClassifier(nn.Module):
    """Map polymer embedding to soluble/insoluble class logit."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_sizes: List[int] | Tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.hidden_sizes = list(hidden_sizes)
        self.dropout = float(dropout)

        self.encoder = _build_mlp(self.embedding_dim, self.hidden_sizes, self.dropout)
        head_dim = self.embedding_dim if not self.hidden_sizes else self.hidden_sizes[-1]
        self.class_head = nn.Linear(head_dim, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.encoder(embedding)
        class_logit = self.class_head(features).squeeze(-1)
        return {
            "class_logit": class_logit,
        }

    def compute_loss(self, embedding: torch.Tensor, class_label: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.forward(embedding=embedding)
        bce = F.binary_cross_entropy_with_logits(out["class_logit"], class_label)
        out.update({"loss": bce})
        return out
