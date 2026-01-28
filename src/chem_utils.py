"""Chemistry helpers (RDKit) for sampling evaluation."""

from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, Descriptors
from rdkit import DataStructs
from rdkit.Chem import RDConfig
import os
import sys

rdBase.DisableLog("rdApp.*")

# SA score (if available)
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
try:
    import sascorer
except Exception:
    sascorer = None


def canonicalize_smiles(smiles: str) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def check_validity(smiles: str) -> bool:
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


def count_stars(smiles: str) -> int:
    return smiles.count("*")


def compute_sa_score(smiles: str) -> Optional[float]:
    if sascorer is None:
        return compute_sa_score_fallback(smiles)
    try:
        smiles_clean = smiles.replace("*", "[H]")
        mol = Chem.MolFromSmiles(smiles_clean)
        if mol is None:
            mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return float(sascorer.calculateScore(mol))
    except Exception:
        return None


def compute_sa_score_fallback(smiles: str) -> Optional[float]:
    try:
        smiles_clean = smiles.replace("*", "[H]")
        mol = Chem.MolFromSmiles(smiles_clean)
        if mol is None:
            mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mw = Descriptors.MolWt(mol)
        num_rings = Descriptors.RingCount(mol)
        num_rot = Descriptors.NumRotatableBonds(mol)
        score = 1.0 + (mw / 500.0) + (num_rings * 0.5) + (num_rot * 0.1)
        score = min(max(score, 1.0), 10.0)
        return float(score)
    except Exception:
        return None


def compute_fingerprint(smiles: str, fp_type: str = "morgan", radius: int = 2, n_bits: int = 2048) -> Optional[np.ndarray]:
    try:
        smiles_clean = smiles.replace("*", "[H]")
        mol = Chem.MolFromSmiles(smiles_clean)
        if mol is None:
            mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        if fp_type == "morgan":
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        elif fp_type == "maccs":
            fp = AllChem.GetMACCSKeysFingerprint(mol)
        else:
            raise ValueError(f"Unknown fp_type: {fp_type}")
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return None


def compute_tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    intersection = np.sum(fp1 & fp2)
    union = np.sum(fp1 | fp2)
    if union == 0:
        return 0.0
    return float(intersection / union)


def compute_pairwise_diversity(fingerprints: List[np.ndarray]) -> float:
    n = len(fingerprints)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            sim = compute_tanimoto_similarity(fingerprints[i], fingerprints[j])
            total += (1.0 - sim)
            count += 1
    return total / count if count > 0 else 0.0


def batch_compute_fingerprints(
    smiles_list: List[str],
    fp_type: str = "morgan",
    radius: int = 2,
    n_bits: int = 2048,
) -> Tuple[List[np.ndarray], List[int]]:
    fps = []
    idxs = []
    for i, s in enumerate(smiles_list):
        fp = compute_fingerprint(s, fp_type, radius, n_bits)
        if fp is not None:
            fps.append(fp)
            idxs.append(i)
    return fps, idxs
