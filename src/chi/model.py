"""Physics-guided model for polymer-water chi(T, phi) prediction."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.chi.constants import COEFF_NAMES

def _validate_formula_inputs_torch(temperature: torch.Tensor, phi: torch.Tensor | None = None) -> None:
    if torch.any(~torch.isfinite(temperature)):
        raise ValueError("temperature contains non-finite values")
    if torch.any(temperature <= 0):
        raise ValueError("temperature must be > 0 for chi(T, phi) formula")
    if phi is not None:
        if torch.any(~torch.isfinite(phi)):
            raise ValueError("phi contains non-finite values")
        tol = 1.0e-8
        if torch.any((phi < -tol) | (phi > 1.0 + tol)):
            raise ValueError("phi must be within [0, 1] for chi(T, phi) formula")


def _validate_formula_inputs_numpy(temperature, phi=None) -> None:
    import numpy as np

    temp_arr = np.asarray(temperature, dtype=float)
    if np.any(~np.isfinite(temp_arr)):
        raise ValueError("temperature contains non-finite values")
    if np.any(temp_arr <= 0):
        raise ValueError("temperature must be > 0 for chi(T, phi) formula")
    if phi is not None:
        phi_arr = np.asarray(phi, dtype=float)
        if np.any(~np.isfinite(phi_arr)):
            raise ValueError("phi contains non-finite values")
        tol = 1.0e-8
        if np.any((phi_arr < -tol) | (phi_arr > 1.0 + tol)):
            raise ValueError("phi must be within [0, 1] for chi(T, phi) formula")



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
    _validate_formula_inputs_torch(temperature, phi=phi)
    a0, a1, a2, a3, b1, b2 = [coefficients[..., i] for i in range(6)]
    base = a0 + a1 / temperature + a2 * torch.log(temperature) + a3 * temperature
    one_minus_phi = 1.0 - phi
    modifier = 1.0 + b1 * one_minus_phi + b2 * (one_minus_phi ** 2)
    return base * modifier



def predict_chi_from_coefficients(coefficients, temperature, phi):
    """Numpy-friendly chi formula wrapper used in analysis scripts."""
    import numpy as np

    _validate_formula_inputs_numpy(temperature, phi=phi)
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


def predict_chi_mean_std_from_coefficients(coeff_mean, coeff_std, temperature, phi):
    """Approximate chi mean/std from coefficient mean/std using variance propagation.

    This approximation ignores covariance terms between coefficients.
    """
    import numpy as np

    _validate_formula_inputs_numpy(temperature, phi=phi)
    mean = np.asarray(coeff_mean, dtype=float)
    std = np.asarray(coeff_std, dtype=float)
    if mean.shape != std.shape:
        raise ValueError(
            f"coeff_mean and coeff_std must have identical shape, got {mean.shape} vs {std.shape}"
        )
    if mean.shape[-1] != 6:
        raise ValueError(f"Expected coefficient last dimension=6, got {mean.shape[-1]}")
    if np.any(std < 0):
        raise ValueError("coeff_std must be non-negative")

    temperature = np.asarray(temperature, dtype=float)
    phi = np.asarray(phi, dtype=float)

    a0, a1, a2, a3, b1, b2 = [mean[..., i] for i in range(6)]
    s0, s1, s2, s3, s4, s5 = [std[..., i] for i in range(6)]

    log_t = np.log(temperature)
    one_minus_phi = 1.0 - phi

    base_mean = a0 + a1 / temperature + a2 * log_t + a3 * temperature
    mod_mean = 1.0 + b1 * one_minus_phi + b2 * (one_minus_phi ** 2)
    chi_mean = base_mean * mod_mean

    var_base = (
        (s0 ** 2)
        + ((s1 / temperature) ** 2)
        + ((s2 * log_t) ** 2)
        + ((s3 * temperature) ** 2)
    )
    var_mod = (
        ((s4 * one_minus_phi) ** 2)
        + ((s5 * (one_minus_phi ** 2)) ** 2)
    )
    # Product variance approximation for independent factors:
    # Var(XY) = E[X]^2 Var(Y) + E[Y]^2 Var(X) + Var(X)Var(Y)
    var_chi = (mod_mean ** 2) * var_base + (base_mean ** 2) * var_mod + var_base * var_mod
    chi_std = np.sqrt(np.clip(var_chi, a_min=0.0, a_max=None))
    return chi_mean, chi_std


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


class BackbonePhysicsGuidedChiModel(nn.Module):
    """Backbone encoder + physics-guided chi head used by Step 4 finetuning."""

    def __init__(
        self,
        backbone: nn.Module,
        chi_head: PhysicsGuidedChiModel,
        timestep: int,
        pooling: str = "mean",
    ):
        super().__init__()
        self.backbone = backbone
        self.chi_head = chi_head
        self.timestep = int(timestep)
        self.pooling = pooling

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size = int(input_ids.shape[0])
        timesteps = torch.full((batch_size,), self.timestep, device=input_ids.device, dtype=torch.long)
        return self.backbone.get_pooled_output(
            input_ids=input_ids,
            timesteps=timesteps,
            attention_mask=attention_mask,
            pooling=self.pooling,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: torch.Tensor,
        phi: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        embedding = self._encode(input_ids=input_ids, attention_mask=attention_mask)
        return self.chi_head(embedding=embedding, temperature=temperature, phi=phi)


class BackboneSolubilityClassifierModel(nn.Module):
    """Backbone encoder + solubility classifier head used by Step 4 finetuning."""

    def __init__(
        self,
        backbone: nn.Module,
        classifier_head: SolubilityClassifier,
        timestep: int,
        pooling: str = "mean",
    ):
        super().__init__()
        self.backbone = backbone
        self.classifier_head = classifier_head
        self.timestep = int(timestep)
        self.pooling = pooling

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size = int(input_ids.shape[0])
        timesteps = torch.full((batch_size,), self.timestep, device=input_ids.device, dtype=torch.long)
        return self.backbone.get_pooled_output(
            input_ids=input_ids,
            timesteps=timesteps,
            attention_mask=attention_mask,
            pooling=self.pooling,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        embedding = self._encode(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier_head(embedding=embedding)

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        class_label: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        embedding = self._encode(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier_head.compute_loss(embedding=embedding, class_label=class_label)
