"""
Data splitting utilities for DFT chi, experimental chi, and solubility datasets.

Implements:
- 80/10/10 splits for DFT chi and solubility (stratified by SMILES)
- k-fold cross-validation for experimental chi (at SMILES level)
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from ..utils.config import Config

logger = logging.getLogger("polymer_chi_ml.splits")


def create_dft_splits(
    df: pd.DataFrame,
    config: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create 80/10/10 train/val/test splits for DFT chi data.

    Args:
        df: DataFrame with DFT chi data (must have 'SMILES' column)
        config: Configuration object

    Returns:
        train_df: Training set (80%)
        val_df: Validation set (10%)
        test_df: Test set (10%)

    Example:
        >>> df = pd.read_csv("data/raw/dft_chi.csv")
        >>> train, val, test = create_dft_splits(df, config)
    """
    train_frac = config.splits.train_frac
    val_frac = config.splits.val_frac
    test_frac = config.splits.test_frac
    seed = config.splits.split_seed

    # Validate fractions sum to 1
    total_frac = train_frac + val_frac + test_frac
    if not np.isclose(total_frac, 1.0):
        raise ValueError(
            f"Split fractions must sum to 1.0, got {total_frac}"
        )

    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_frac,
        random_state=seed,
        shuffle=True,
    )

    # Second split: separate train and val from remaining data
    # Adjust val fraction relative to train+val size
    val_frac_adjusted = val_frac / (train_frac + val_frac)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_frac_adjusted,
        random_state=seed,
        shuffle=True,
    )

    logger.info(
        f"DFT splits created: train={len(train_df)}, "
        f"val={len(val_df)}, test={len(test_df)}"
    )

    return train_df, val_df, test_df


def create_solubility_splits(
    df: pd.DataFrame,
    config: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create 80/10/10 train/val/test splits for solubility data, stratified by SMILES.

    All rows with the same SMILES go into the same split.
    Stratification is done by the 'soluble' label to preserve class balance.

    Args:
        df: DataFrame with solubility data (must have 'SMILES' and 'soluble' columns)
        config: Configuration object

    Returns:
        train_df: Training set (80%)
        val_df: Validation set (10%)
        test_df: Test set (10%)

    Example:
        >>> df = pd.read_csv("data/raw/solubility.csv")
        >>> train, val, test = create_solubility_splits(df, config)
    """
    train_frac = config.splits.train_frac
    val_frac = config.splits.val_frac
    test_frac = config.splits.test_frac
    seed = config.splits.split_seed
    stratify = config.solubility.stratify

    # Validate fractions
    total_frac = train_frac + val_frac + test_frac
    if not np.isclose(total_frac, 1.0):
        raise ValueError(
            f"Split fractions must sum to 1.0, got {total_frac}"
        )

    # Get unique SMILES with their labels
    # If a SMILES appears multiple times (shouldn't happen for solubility),
    # use the most common label
    smiles_labels = df.groupby("SMILES")["soluble"].agg(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
    ).reset_index()

    unique_smiles = smiles_labels["SMILES"].values
    labels = smiles_labels["soluble"].values

    # First split: separate test SMILES
    if stratify:
        try:
            train_val_smiles, test_smiles = train_test_split(
                unique_smiles,
                test_size=test_frac,
                random_state=seed,
                stratify=labels,
                shuffle=True,
            )
            # Get labels for train+val for second split
            train_val_labels = smiles_labels[
                smiles_labels["SMILES"].isin(train_val_smiles)
            ]["soluble"].values
        except ValueError as e:
            logger.warning(
                f"Stratified split failed ({e}), using non-stratified split"
            )
            stratify = False

    if not stratify:
        train_val_smiles, test_smiles = train_test_split(
            unique_smiles,
            test_size=test_frac,
            random_state=seed,
            shuffle=True,
        )
        train_val_labels = None

    # Second split: separate train and val SMILES
    val_frac_adjusted = val_frac / (train_frac + val_frac)

    if stratify and train_val_labels is not None:
        try:
            train_smiles, val_smiles = train_test_split(
                train_val_smiles,
                test_size=val_frac_adjusted,
                random_state=seed,
                stratify=train_val_labels,
                shuffle=True,
            )
        except ValueError as e:
            logger.warning(
                f"Stratified split failed for val ({e}), using non-stratified"
            )
            train_smiles, val_smiles = train_test_split(
                train_val_smiles,
                test_size=val_frac_adjusted,
                random_state=seed,
                shuffle=True,
            )
    else:
        train_smiles, val_smiles = train_test_split(
            train_val_smiles,
            test_size=val_frac_adjusted,
            random_state=seed,
            shuffle=True,
        )

    # Map back to full DataFrame
    train_df = df[df["SMILES"].isin(train_smiles)].reset_index(drop=True)
    val_df = df[df["SMILES"].isin(val_smiles)].reset_index(drop=True)
    test_df = df[df["SMILES"].isin(test_smiles)].reset_index(drop=True)

    # Log statistics
    logger.info(
        f"Solubility splits created (SMILES-level stratified): "
        f"train={len(train_df)} ({len(train_smiles)} SMILES), "
        f"val={len(val_df)} ({len(val_smiles)} SMILES), "
        f"test={len(test_df)} ({len(test_smiles)} SMILES)"
    )

    # Log class balance
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        n_soluble = (split_df["soluble"] == 1).sum()
        balance = n_soluble / len(split_df) if len(split_df) > 0 else 0
        logger.info(
            f"  {split_name}: {n_soluble}/{len(split_df)} soluble ({balance:.2%})"
        )

    return train_df, val_df, test_df


def create_exp_chi_splits(
    df: pd.DataFrame,
    config: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create 80/10/10 train/val/test splits for experimental chi data, grouped by SMILES.

    All measurements for a given polymer (same SMILES) go into the same split.
    This prevents data leakage where the same polymer appears in both train and test.

    Args:
        df: DataFrame with experimental chi data (must have 'SMILES' column)
        config: Configuration object

    Returns:
        train_df: Training set (80%)
        val_df: Validation set (10%)
        test_df: Test set (10%)

    Example:
        >>> df = pd.read_csv("data/raw/exp_chi.csv")
        >>> train, val, test = create_exp_chi_splits(df, config)
    """
    train_frac = config.splits.train_frac
    val_frac = config.splits.val_frac
    test_frac = config.splits.test_frac
    seed = config.splits.split_seed

    # Validate fractions
    total_frac = train_frac + val_frac + test_frac
    if not np.isclose(total_frac, 1.0):
        raise ValueError(
            f"Split fractions must sum to 1.0, got {total_frac}"
        )

    # Get unique SMILES
    unique_smiles = df["SMILES"].unique()
    n_unique = len(unique_smiles)

    # First split: separate test SMILES
    train_val_smiles, test_smiles = train_test_split(
        unique_smiles,
        test_size=test_frac,
        random_state=seed,
        shuffle=True,
    )

    # Second split: separate train and val SMILES
    val_frac_adjusted = val_frac / (train_frac + val_frac)
    train_smiles, val_smiles = train_test_split(
        train_val_smiles,
        test_size=val_frac_adjusted,
        random_state=seed,
        shuffle=True,
    )

    # Map SMILES back to DataFrame rows
    train_df = df[df["SMILES"].isin(train_smiles)].copy()
    val_df = df[df["SMILES"].isin(val_smiles)].copy()
    test_df = df[df["SMILES"].isin(test_smiles)].copy()

    logger.info(
        f"Exp chi splits created (SMILES-grouped): "
        f"train={len(train_df)} measurements ({len(train_smiles)} SMILES), "
        f"val={len(val_df)} measurements ({len(val_smiles)} SMILES), "
        f"test={len(test_df)} measurements ({len(test_smiles)} SMILES)"
    )

    return train_df, val_df, test_df


def create_exp_chi_cv_splits(
    df: pd.DataFrame,
    config: Config,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create k-fold cross-validation splits for experimental chi data at SMILES level.

    All measurements for a given SMILES go into the same fold.

    Args:
        df: DataFrame with experimental chi data (must have 'SMILES' column)
        config: Configuration object

    Returns:
        List of (train_indices, val_indices) tuples for each fold

    Example:
        >>> df = pd.read_csv("data/raw/exp_chi.csv")
        >>> folds = create_exp_chi_cv_splits(df, config)
        >>> for fold_idx, (train_idx, val_idx) in enumerate(folds):
        ...     train_df = df.iloc[train_idx]
        ...     val_df = df.iloc[val_idx]
    """
    k_folds = config.cv.exp_chi_k_folds
    shuffle = config.cv.exp_chi_shuffle
    seed = config.cv.exp_chi_shuffle_seed

    # Get unique SMILES
    unique_smiles = df["SMILES"].unique()
    n_unique = len(unique_smiles)

    if k_folds > n_unique:
        logger.warning(
            f"k_folds ({k_folds}) > n_unique_smiles ({n_unique}), "
            f"reducing k to {n_unique}"
        )
        k_folds = n_unique

    # Create KFold splitter
    kfold = KFold(n_splits=k_folds, shuffle=shuffle, random_state=seed if shuffle else None)

    # Generate splits at SMILES level
    folds = []
    for fold_idx, (train_smiles_idx, val_smiles_idx) in enumerate(kfold.split(unique_smiles)):
        # Get SMILES for this fold
        train_smiles = unique_smiles[train_smiles_idx]
        val_smiles = unique_smiles[val_smiles_idx]

        # Map to DataFrame indices
        train_mask = df["SMILES"].isin(train_smiles)
        val_mask = df["SMILES"].isin(val_smiles)

        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]

        folds.append((train_indices, val_indices))

        logger.info(
            f"Fold {fold_idx + 1}/{k_folds}: "
            f"train={len(train_indices)} samples ({len(train_smiles)} SMILES), "
            f"val={len(val_indices)} samples ({len(val_smiles)} SMILES)"
        )

    logger.info(
        f"Created {k_folds}-fold CV splits for experimental chi "
        f"({n_unique} unique SMILES, {len(df)} total measurements)"
    )

    return folds


def get_smiles_sets(
    df: pd.DataFrame,
    config: Config,
) -> Tuple[set, set, set]:
    """
    Get train/val/test SMILES sets for stratified splitting.

    Utility function for multi-task training where you need to know
    which SMILES belong to which split.

    Args:
        df: DataFrame with 'SMILES' and optionally 'soluble' columns
        config: Configuration object

    Returns:
        train_smiles: Set of training SMILES
        val_smiles: Set of validation SMILES
        test_smiles: Set of test SMILES
    """
    if "soluble" in df.columns:
        # Use solubility splits (stratified)
        train_df, val_df, test_df = create_solubility_splits(df, config)
    else:
        # Use DFT-style splits (non-stratified)
        train_df, val_df, test_df = create_dft_splits(df, config)

    train_smiles = set(train_df["SMILES"].unique())
    val_smiles = set(val_df["SMILES"].unique())
    test_smiles = set(test_df["SMILES"].unique())

    return train_smiles, val_smiles, test_smiles
