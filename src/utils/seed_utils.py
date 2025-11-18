"""
Reproducibility utilities for seeding random number generators.
"""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Random seed value
        deterministic: If True, sets PyTorch to deterministic mode (slower but fully reproducible)

    Note:
        Deterministic mode may impact performance but ensures complete reproducibility.
        For CUDA operations, deterministic=True sets:
        - torch.backends.cudnn.deterministic = True
        - torch.backends.cudnn.benchmark = False
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Set deterministic behavior for CUDA
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # For better performance with variable input sizes
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int, base_seed: Optional[int] = None) -> None:
    """
    Initialize worker seed for DataLoader workers.

    Args:
        worker_id: Worker ID (automatically passed by DataLoader)
        base_seed: Base seed to use. If None, uses default seed.

    Usage:
        DataLoader(..., worker_init_fn=lambda wid: worker_init_fn(wid, seed))
    """
    if base_seed is None:
        base_seed = torch.initial_seed() % 2**32

    seed = base_seed + worker_id
    np.random.seed(seed)
    random.seed(seed)
