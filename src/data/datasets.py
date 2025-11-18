"""
PyTorch Dataset classes for DFT chi, experimental chi, and solubility data.

All datasets operate on pre-featurized data and provide consistent interfaces
for the training pipeline.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("polymer_chi_ml.datasets")


class DFTChiDataset(Dataset):
    """
    Dataset for DFT-computed chi (polymer-water interaction parameter).

    Each sample contains:
    - x: Feature vector for polymer
    - chi_dft: DFT-computed chi value
    - temperature: Temperature in Kelvin
    - smiles: Original SMILES string

    Args:
        df: DataFrame with columns ['SMILES', 'chi', 'temp']
        features: 2D array of features (n_polymers, feature_dim)
        smiles_to_idx: Mapping from SMILES to feature array index

    Example:
        >>> df = pd.read_csv("data/raw/dft_chi.csv")
        >>> dataset = DFTChiDataset(df, features, smiles_to_idx)
        >>> x, chi, temp, smiles = dataset[0]
    """

    def __init__(
        self,
        df: pd.DataFrame,
        features: np.ndarray,
        smiles_to_idx: Dict[str, int],
    ):
        """Initialize dataset with data and features."""
        self.df = df.reset_index(drop=True)
        self.features = features
        self.smiles_to_idx = smiles_to_idx

        # Validate required columns
        required_cols = ["SMILES", "chi", "temp"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"DataFrame missing required column: {col}")

        # Filter to only SMILES with valid features
        valid_mask = df["SMILES"].isin(smiles_to_idx.keys())
        if not valid_mask.all():
            n_invalid = (~valid_mask).sum()
            logger.warning(
                f"Filtering {n_invalid} samples with missing features"
            )
            self.df = self.df[valid_mask].reset_index(drop=True)

        logger.info(f"DFTChiDataset initialized with {len(self.df)} samples")

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            x: Feature tensor (feature_dim,)
            chi_dft: DFT chi value (scalar)
            temperature: Temperature in K (scalar)
            smiles: SMILES string
        """
        row = self.df.iloc[idx]
        smiles = row["SMILES"]
        chi_dft = float(row["chi"])
        temp = float(row["temp"])

        # Get features
        feat_idx = self.smiles_to_idx[smiles]
        x = torch.from_numpy(self.features[feat_idx]).float()

        return x, torch.tensor(chi_dft, dtype=torch.float32), \
               torch.tensor(temp, dtype=torch.float32), smiles


class ExpChiDataset(Dataset):
    """
    Dataset for experimental chi measurements at various temperatures.

    Each sample contains:
    - x: Feature vector for polymer
    - chi_exp: Experimental chi value
    - temperature: Measurement temperature in Kelvin
    - smiles: Original SMILES string

    Note: A single polymer (SMILES) can appear multiple times at different temperatures.

    Args:
        df: DataFrame with columns ['SMILES', 'chi', 'temp']
        features: 2D array of features (n_polymers, feature_dim)
        smiles_to_idx: Mapping from SMILES to feature array index

    Example:
        >>> df = pd.read_csv("data/raw/exp_chi.csv")
        >>> dataset = ExpChiDataset(df, features, smiles_to_idx)
        >>> x, chi, temp, smiles = dataset[0]
    """

    def __init__(
        self,
        df: pd.DataFrame,
        features: np.ndarray,
        smiles_to_idx: Dict[str, int],
    ):
        """Initialize dataset with data and features."""
        self.df = df.reset_index(drop=True)
        self.features = features
        self.smiles_to_idx = smiles_to_idx

        # Validate required columns
        required_cols = ["SMILES", "chi", "temp"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"DataFrame missing required column: {col}")

        # Filter to only SMILES with valid features
        valid_mask = df["SMILES"].isin(smiles_to_idx.keys())
        if not valid_mask.all():
            n_invalid = (~valid_mask).sum()
            logger.warning(
                f"Filtering {n_invalid} samples with missing features"
            )
            self.df = self.df[valid_mask].reset_index(drop=True)

        # Log statistics
        n_unique_smiles = self.df["SMILES"].nunique()
        logger.info(
            f"ExpChiDataset initialized with {len(self.df)} measurements "
            f"from {n_unique_smiles} unique polymers"
        )

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            x: Feature tensor (feature_dim,)
            chi_exp: Experimental chi value (scalar)
            temperature: Temperature in K (scalar)
            smiles: SMILES string
        """
        row = self.df.iloc[idx]
        smiles = row["SMILES"]
        chi_exp = float(row["chi"])
        temp = float(row["temp"])

        # Get features
        feat_idx = self.smiles_to_idx[smiles]
        x = torch.from_numpy(self.features[feat_idx]).float()

        return x, torch.tensor(chi_exp, dtype=torch.float32), \
               torch.tensor(temp, dtype=torch.float32), smiles


class SolubilityDataset(Dataset):
    """
    Dataset for binary water solubility classification.

    Each sample contains:
    - x: Feature vector for polymer
    - soluble: Binary label (1 = soluble, 0 = insoluble)
    - smiles: Original SMILES string

    Args:
        df: DataFrame with columns ['SMILES', 'soluble']
        features: 2D array of features (n_polymers, feature_dim)
        smiles_to_idx: Mapping from SMILES to feature array index

    Example:
        >>> df = pd.read_csv("data/raw/solubility.csv")
        >>> dataset = SolubilityDataset(df, features, smiles_to_idx)
        >>> x, label, smiles = dataset[0]
    """

    def __init__(
        self,
        df: pd.DataFrame,
        features: np.ndarray,
        smiles_to_idx: Dict[str, int],
    ):
        """Initialize dataset with data and features."""
        self.df = df.reset_index(drop=True)
        self.features = features
        self.smiles_to_idx = smiles_to_idx

        # Validate required columns
        required_cols = ["SMILES", "soluble"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"DataFrame missing required column: {col}")

        # Filter to only SMILES with valid features
        valid_mask = df["SMILES"].isin(smiles_to_idx.keys())
        if not valid_mask.all():
            n_invalid = (~valid_mask).sum()
            logger.warning(
                f"Filtering {n_invalid} samples with missing features"
            )
            self.df = self.df[valid_mask].reset_index(drop=True)

        # Ensure labels are binary
        unique_labels = self.df["soluble"].unique()
        if not set(unique_labels).issubset({0, 1}):
            logger.warning(
                f"Non-binary labels found: {unique_labels}. "
                f"Ensure 'soluble' column contains only 0 or 1."
            )

        # Log class distribution
        n_soluble = (self.df["soluble"] == 1).sum()
        n_insoluble = (self.df["soluble"] == 0).sum()
        logger.info(
            f"SolubilityDataset initialized with {len(self.df)} samples: "
            f"{n_soluble} soluble, {n_insoluble} insoluble "
            f"(balance: {n_soluble / len(self.df):.2%})"
        )

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            x: Feature tensor (feature_dim,)
            soluble: Binary label (0 or 1)
            smiles: SMILES string
        """
        row = self.df.iloc[idx]
        smiles = row["SMILES"]
        soluble = int(row["soluble"])

        # Get features
        feat_idx = self.smiles_to_idx[smiles]
        x = torch.from_numpy(self.features[feat_idx]).float()

        return x, torch.tensor(soluble, dtype=torch.float32), smiles


def collate_dft_chi(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]
) -> Dict[str, torch.Tensor]:
    """
    Collate function for DFTChiDataset.

    Args:
        batch: List of (x, chi_dft, temperature, smiles) tuples

    Returns:
        Dictionary with batched tensors:
            - x: (batch_size, feature_dim)
            - chi_dft: (batch_size,)
            - temperature: (batch_size,)
            - smiles: List of SMILES strings
    """
    x_list, chi_list, temp_list, smiles_list = zip(*batch)

    return {
        "x": torch.stack(x_list),
        "chi_dft": torch.stack(chi_list),
        "temperature": torch.stack(temp_list),
        "smiles": smiles_list,
    }


def collate_exp_chi(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]
) -> Dict[str, torch.Tensor]:
    """
    Collate function for ExpChiDataset.

    Args:
        batch: List of (x, chi_exp, temperature, smiles) tuples

    Returns:
        Dictionary with batched tensors:
            - x: (batch_size, feature_dim)
            - chi_exp: (batch_size,)
            - temperature: (batch_size,)
            - smiles: List of SMILES strings
    """
    x_list, chi_list, temp_list, smiles_list = zip(*batch)

    return {
        "x": torch.stack(x_list),
        "chi_exp": torch.stack(chi_list),
        "temperature": torch.stack(temp_list),
        "smiles": smiles_list,
    }


def collate_solubility(
    batch: List[Tuple[torch.Tensor, torch.Tensor, str]]
) -> Dict[str, torch.Tensor]:
    """
    Collate function for SolubilityDataset.

    Args:
        batch: List of (x, soluble, smiles) tuples

    Returns:
        Dictionary with batched tensors:
            - x: (batch_size, feature_dim)
            - soluble: (batch_size,)
            - smiles: List of SMILES strings
    """
    x_list, soluble_list, smiles_list = zip(*batch)

    return {
        "x": torch.stack(x_list),
        "soluble": torch.stack(soluble_list),
        "smiles": smiles_list,
    }
