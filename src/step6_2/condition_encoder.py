"""Condition encoder for Step 6_2 conditional diffusion models."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConditionEncoder(nn.Module):
    """Encode the 7-scalar condition bundle or a CFG-dropped branch."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        self.dropped_condition_embedding = nn.Parameter(torch.zeros(self.output_dim))
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.dropped_condition_embedding)

    def forward(
        self,
        condition_bundle: torch.Tensor,
        condition_drop_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if condition_bundle.dim() != 2 or condition_bundle.shape[-1] != self.input_dim:
            raise ValueError(
                f"condition_bundle must have shape [batch, {self.input_dim}], got {tuple(condition_bundle.shape)}"
            )
        cond = self.mlp(condition_bundle)
        if condition_drop_mask is None:
            return cond
        dropped = self.dropped_condition_embedding.unsqueeze(0).expand(condition_bundle.shape[0], -1)
        return torch.where(condition_drop_mask.unsqueeze(-1), dropped, cond)
