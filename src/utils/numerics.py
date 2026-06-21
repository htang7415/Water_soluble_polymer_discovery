"""Numerical helper utilities."""

from __future__ import annotations

import numpy as np


def stable_sigmoid(x):
    """Numerically stable sigmoid for numpy arrays/scalars."""
    arr = np.asarray(x, dtype=np.float64)
    out = np.empty_like(arr, dtype=np.float64)
    pos = arr >= 0
    neg = ~pos

    out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
    exp_x = np.exp(arr[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out

