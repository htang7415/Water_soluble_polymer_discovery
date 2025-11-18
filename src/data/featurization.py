"""
Polymer SMILES featurization using Morgan fingerprints and RDKit descriptors.

Handles polymer repeat-unit SMILES containing two '*' connection points.
Provides caching mechanism for computed features.
"""

import hashlib
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from ..utils.config import Config

logger = logging.getLogger("polymer_chi_ml.featurization")


class PolymerFeaturizer:
    """
    Convert polymer repeat-unit SMILES to fixed-length feature vectors.

    Features include:
    - Morgan (circular) fingerprints
    - RDKit molecular descriptors

    Handles SMILES with '*' connection points by replacing with dummy atoms.
    Implements caching to avoid recomputation.

    Args:
        config: Configuration object containing featurization parameters
        cache_dir: Directory for caching computed features

    Example:
        >>> config = load_config("configs/config.yaml")
        >>> featurizer = PolymerFeaturizer(config)
        >>> features = featurizer.featurize(["*CC*", "*C(C)C*"])
        >>> print(features.shape)
        (2, 2061)  # 2048 Morgan bits + 13 descriptors
    """

    def __init__(self, config: Config, cache_dir: Optional[Path] = None):
        """Initialize featurizer with configuration."""
        self.config = config
        self.morgan_radius = config.chem.morgan_radius
        self.morgan_n_bits = config.chem.morgan_n_bits
        self.descriptor_list = config.chem.descriptor_list
        self.dummy_atom = config.chem.smiles_dummy_replacement
        self.force_recompute = config.chem.force_recompute_features

        # Set cache directory
        if cache_dir is None:
            cache_dir = Path(config.paths.processed_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Feature dimension (computed after first featurization)
        self.feature_dim: Optional[int] = None

        # Descriptor functions mapping
        self._descriptor_functions = self._build_descriptor_map()

        logger.info(
            f"PolymerFeaturizer initialized: "
            f"Morgan(radius={self.morgan_radius}, nBits={self.morgan_n_bits}), "
            f"{len(self.descriptor_list)} descriptors"
        )

    def _build_descriptor_map(self) -> Dict[str, callable]:
        """Build mapping from descriptor names to RDKit functions."""
        descriptor_map = {
            "MolWt": Descriptors.MolWt,
            "MolLogP": Descriptors.MolLogP,
            "LogP": Descriptors.MolLogP,  # Alias
            "TPSA": Descriptors.TPSA,
            "NumHDonors": Descriptors.NumHDonors,
            "NumHAcceptors": Descriptors.NumHAcceptors,
            "FractionCSP3": Descriptors.FractionCSP3,
            "NumAromaticRings": Descriptors.NumAromaticRings,
            "NumAliphaticRings": Descriptors.NumAliphaticRings,
            "NumRotatableBonds": Descriptors.NumRotatableBonds,
            "NumHeteroatoms": Descriptors.NumHeteroatoms,
            "FormalCharge": Chem.rdmolops.GetFormalCharge,
            "HeavyAtomCount": Descriptors.HeavyAtomCount,
            "RingCount": Descriptors.RingCount,
            "MolMR": Descriptors.MolMR,
        }
        return descriptor_map

    def _get_cache_key(self) -> str:
        """
        Generate unique cache key based on featurization parameters.

        Returns:
            MD5 hash of featurization parameters
        """
        params = {
            "morgan_radius": self.morgan_radius,
            "morgan_n_bits": self.morgan_n_bits,
            "descriptors": sorted(self.descriptor_list),
            "dummy_atom": self.dummy_atom,
        }
        param_str = str(params)
        return hashlib.md5(param_str.encode()).hexdigest()[:16]

    def _get_cache_path(self, data_hash: str) -> Path:
        """
        Get cache file path for a specific dataset.

        Args:
            data_hash: Hash of input data (e.g., from SMILES list)

        Returns:
            Path to cache file
        """
        cache_key = self._get_cache_key()
        cache_filename = f"features_{cache_key}_{data_hash}.pkl"
        return self.cache_dir / cache_filename

    def _compute_data_hash(self, smiles_list: List[str]) -> str:
        """
        Compute hash of input SMILES list for cache validation.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            MD5 hash of concatenated SMILES
        """
        combined = "".join(sorted(smiles_list))
        return hashlib.md5(combined.encode()).hexdigest()[:16]

    def _preprocess_smiles(self, smiles: str) -> str:
        """
        Preprocess SMILES by replacing '*' connection points with dummy atom.

        Args:
            smiles: Polymer repeat-unit SMILES with '*' connection points

        Returns:
            Modified SMILES with dummy atoms

        Example:
            >>> self._preprocess_smiles("*CC(C)C*")
            "CC(C)CC"  # If dummy_atom = "C"
        """
        return smiles.replace("*", self.dummy_atom)

    def _smiles_to_mol(self, smiles: str) -> Optional[Chem.Mol]:
        """
        Convert SMILES to RDKit Mol object.

        Args:
            smiles: SMILES string (already preprocessed)

        Returns:
            RDKit Mol object or None if parsing failed
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Failed to parse SMILES: {smiles}")
                return None
            return mol
        except Exception as e:
            logger.warning(f"Error parsing SMILES '{smiles}': {e}")
            return None

    def _compute_morgan_fp(self, mol: Chem.Mol) -> np.ndarray:
        """
        Compute Morgan fingerprint as binary vector.

        Args:
            mol: RDKit Mol object

        Returns:
            Binary fingerprint array (float32)
        """
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=self.morgan_radius,
            nBits=self.morgan_n_bits
        )
        # Convert to numpy array
        arr = np.zeros((self.morgan_n_bits,), dtype=np.float32)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def _compute_descriptors(self, mol: Chem.Mol) -> np.ndarray:
        """
        Compute RDKit descriptors.

        Args:
            mol: RDKit Mol object

        Returns:
            Descriptor vector (float32)
        """
        desc_values = []
        for desc_name in self.descriptor_list:
            if desc_name not in self._descriptor_functions:
                logger.warning(f"Unknown descriptor: {desc_name}, skipping")
                desc_values.append(0.0)
                continue

            try:
                desc_func = self._descriptor_functions[desc_name]
                value = desc_func(mol)

                # Handle NaN or inf
                if not np.isfinite(value):
                    logger.warning(
                        f"Non-finite descriptor {desc_name} for mol, using 0.0"
                    )
                    value = 0.0

                desc_values.append(float(value))

            except Exception as e:
                logger.warning(f"Error computing {desc_name}: {e}, using 0.0")
                desc_values.append(0.0)

        return np.array(desc_values, dtype=np.float32)

    def featurize_single(self, smiles: str) -> Optional[np.ndarray]:
        """
        Featurize a single SMILES string.

        Args:
            smiles: Polymer repeat-unit SMILES (with '*')

        Returns:
            Feature vector (1D array) or None if featurization failed
        """
        # Preprocess SMILES
        preprocessed = self._preprocess_smiles(smiles)

        # Convert to mol
        mol = self._smiles_to_mol(preprocessed)
        if mol is None:
            return None

        # Compute Morgan FP
        morgan_fp = self._compute_morgan_fp(mol)

        # Compute descriptors
        descriptors = self._compute_descriptors(mol)

        # Concatenate
        features = np.concatenate([morgan_fp, descriptors])

        return features

    def featurize(
        self,
        smiles_list: List[str],
        use_cache: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Featurize a list of SMILES strings.

        Args:
            smiles_list: List of polymer SMILES
            use_cache: If True, use cached features if available

        Returns:
            features: 2D array of shape (n_valid, feature_dim)
            smiles_to_idx: Mapping from SMILES to row index in features array

        Raises:
            ValueError: If no valid SMILES could be featurized

        Example:
            >>> features, mapping = featurizer.featurize(["*CC*", "*C(C)C*"])
            >>> print(features.shape)
            (2, 2061)
            >>> print(mapping["*CC*"])
            0
        """
        # Check cache
        data_hash = self._compute_data_hash(smiles_list)
        cache_path = self._get_cache_path(data_hash)

        if use_cache and not self.force_recompute and cache_path.exists():
            logger.info(f"Loading features from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)

            features = cache_data["features"]
            smiles_to_idx = cache_data["smiles_to_idx"]
            self.feature_dim = features.shape[1]

            logger.info(
                f"Loaded {len(features)} cached features, dim={self.feature_dim}"
            )
            return features, smiles_to_idx

        # Featurize each SMILES
        logger.info(f"Featurizing {len(smiles_list)} SMILES strings...")

        feature_list = []
        smiles_to_idx = {}
        failed_smiles = []

        for smiles in smiles_list:
            # Skip duplicates
            if smiles in smiles_to_idx:
                continue

            features = self.featurize_single(smiles)

            if features is None:
                failed_smiles.append(smiles)
                continue

            # Add to list
            smiles_to_idx[smiles] = len(feature_list)
            feature_list.append(features)

        # Check if any valid features
        if len(feature_list) == 0:
            raise ValueError(
                f"No valid SMILES could be featurized. "
                f"Failed SMILES: {failed_smiles[:10]}"
            )

        # Convert to array
        features_array = np.array(feature_list, dtype=np.float32)
        self.feature_dim = features_array.shape[1]

        # Log statistics
        logger.info(
            f"Featurization complete: {len(features_array)} valid, "
            f"{len(failed_smiles)} failed, feature_dim={self.feature_dim}"
        )

        if failed_smiles:
            logger.warning(
                f"Failed to featurize {len(failed_smiles)} SMILES. "
                f"Examples: {failed_smiles[:5]}"
            )

        # Save to cache
        if use_cache:
            cache_data = {
                "features": features_array,
                "smiles_to_idx": smiles_to_idx,
                "config": {
                    "morgan_radius": self.morgan_radius,
                    "morgan_n_bits": self.morgan_n_bits,
                    "descriptors": self.descriptor_list,
                    "dummy_atom": self.dummy_atom,
                },
            }

            with open(cache_path, "wb") as f:
                pickle.dump(cache_data, f)

            logger.info(f"Saved features to cache: {cache_path}")

        return features_array, smiles_to_idx

    def get_feature_dim(self) -> int:
        """
        Get feature dimension (must be called after featurization).

        Returns:
            Feature vector dimension

        Raises:
            RuntimeError: If called before featurization
        """
        if self.feature_dim is None:
            raise RuntimeError(
                "Feature dimension unknown. Call featurize() first."
            )
        return self.feature_dim


# Convenience functions for backward compatibility and ease of use


def compute_features(
    smiles_list: List[str],
    config: Config,
    cache_dir: Optional[Path] = None,
    use_cache: bool = True,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Compute features for a list of SMILES strings.

    Thin wrapper around PolymerFeaturizer for convenience.

    Args:
        smiles_list: List of polymer SMILES strings
        config: Configuration object
        cache_dir: Optional cache directory (defaults to config.paths.processed_dir)
        use_cache: Whether to use cached features

    Returns:
        Tuple of (feature_array, smiles_to_index_dict)

    Example:
        >>> features, smiles_map = compute_features(
        ...     ["*CC*", "*C(C)C*"],
        ...     config,
        ...     use_cache=True
        ... )
    """
    if cache_dir is None:
        cache_dir = Path(config.paths.processed_dir)

    featurizer = PolymerFeaturizer(config, cache_dir=cache_dir)
    features, smiles_to_idx = featurizer.featurize(smiles_list, use_cache=use_cache)
    return features, smiles_to_idx


def load_or_compute_features(
    dataframe,
    config: Config,
    cache_prefix: str = "",
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Load features from cache or compute them from a DataFrame.

    Convenience function for loading/computing features from a pandas DataFrame
    containing a 'SMILES' column. Automatically handles caching with a prefix.

    Args:
        dataframe: pandas DataFrame with 'SMILES' column
        config: Configuration object
        cache_prefix: Prefix for cache files (e.g., 'dft', 'exp', 'sol')
        logger: Optional logger for progress messages

    Returns:
        Tuple of (feature_array, smiles_to_index_dict)

    Raises:
        ValueError: If DataFrame doesn't have 'SMILES' column

    Example:
        >>> import pandas as pd
        >>> df = pd.read_csv("data.csv")
        >>> features, smiles_map = load_or_compute_features(
        ...     df, config, cache_prefix="dft"
        ... )
    """
    if "SMILES" not in dataframe.columns:
        raise ValueError("DataFrame must have 'SMILES' column")

    if logger is None:
        logger = logging.getLogger("polymer_chi_ml.featurization")

    # Get unique SMILES from dataframe
    smiles_list = dataframe["SMILES"].unique().tolist()

    logger.info(f"Featurizing {len(smiles_list)} unique SMILES (prefix: {cache_prefix})")

    # Create cache directory
    cache_dir = Path(config.paths.processed_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Initialize featurizer
    featurizer = PolymerFeaturizer(config, cache_dir=cache_dir)

    # Build cache key with prefix
    if cache_prefix:
        # Temporarily modify cache dir to include prefix
        original_cache_dir = featurizer.cache_dir
        featurizer.cache_dir = cache_dir / cache_prefix
        featurizer.cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Check if we should force recompute
        use_cache = not config.chem.get("force_recompute_features", False)

        # Compute features
        features, smiles_to_idx = featurizer.featurize(smiles_list, use_cache=use_cache)

        logger.info(
            f"Featurization complete: {features.shape[0]} polymers, "
            f"{features.shape[1]} features per polymer"
        )

        return features, smiles_to_idx

    finally:
        # Restore original cache dir if modified
        if cache_prefix:
            featurizer.cache_dir = original_cache_dir
