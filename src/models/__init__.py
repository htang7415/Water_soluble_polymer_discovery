"""
Model components for polymer-water interaction prediction.

Exports:
- Encoder: Shared feature encoder
- ChiHead: Chi(T) = A/T + B prediction head
- SolubilityHead: Solubility classification head
- MultiTaskChiSolubilityModel: Complete multi-task model
"""

from .encoder import Encoder
from .multitask_model import ChiHead, MultiTaskChiSolubilityModel, SolubilityHead

__all__ = [
    "Encoder",
    "ChiHead",
    "SolubilityHead",
    "MultiTaskChiSolubilityModel",
]
