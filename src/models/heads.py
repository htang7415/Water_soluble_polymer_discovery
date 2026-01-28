from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], out_dim: int, dropout: float):
        super().__init__()
        dims = [in_dim] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RegressionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        self.mlp = MLP(in_dim, hidden_dims, 1, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)


class ChiHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], dropout: float, mode: str = "M1"):
        super().__init__()
        out_dim = 2 if mode == "M1" else 3
        self.mode = mode
        self.mlp = MLP(in_dim, hidden_dims, out_dim, dropout)

    def forward(self, x: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        coeffs = self.mlp(x)
        if self.mode == "M1":
            a = coeffs[:, 0]
            b = coeffs[:, 1]
            return a + b / temperature
        a = coeffs[:, 0]
        b = coeffs[:, 1]
        c = coeffs[:, 2]
        return a + b / temperature + c * temperature


class SolubilityHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        self.mlp = MLP(in_dim, hidden_dims, 1, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)


class HansenHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        self.mlp = MLP(in_dim, hidden_dims, 3, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
