"""Polymer family classifier using SMARTS patterns."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdmolops
from tqdm import tqdm


# Polymer classes whose defining functional group is expected to be on the
# repeat-unit backbone (shortest path between the two ``*`` connection points).
# Classes like polyacrylate (pendant ester) and polystyrene (pendant phenyl)
# are intentionally excluded because their functional groups are side-chain.
BACKBONE_CLASS_MATCH_CLASSES: Set[str] = {
    "polyimide",
    "polyester",
    "polyamide",
    "polyurethane",
    "polyether",
    "polysiloxane",
    "polycarbonate",
    "polysulfone",
}


class PolymerClassifier:
    """Classify polymers into families using SMARTS patterns."""

    DEFAULT_PATTERNS = {
        "polyimide": "[#6][CX3](=[OX1])[NX3][CX3](=[OX1])[#6]",
        "polyester": "[#6][CX3](=[OX1])[OX2][#6]",
        "polyamide": "[#6][CX3](=[OX1])[NX3;!$([N]([C](=O))[C](=O))][#6;!$([CX3](=[OX1]))]",
        "polyurethane": "[#6][OX2][CX3](=[OX1])[NX3][#6]",
        "polyether": "[#6;!$([CX3](=[OX1]))][OX2][#6;!$([CX3](=[OX1]))]",
        "polysiloxane": "[Si][OX2][Si]",
        "polycarbonate": "[#6][OX2][CX3](=[OX1])[OX2][#6]",
        "polysulfone": "[#6][SX4](=[OX1])(=[OX1])[#6]",
        "polyacrylate": "[#6]-[#6](=O)-[#8]",
        "polystyrene": "[#6]-[#6](c1ccccc1)-[#6]",
    }

    def __init__(self, patterns: Optional[Dict[str, str]] = None):
        self.patterns = patterns if patterns else self.DEFAULT_PATTERNS
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        self.compiled_patterns: Dict[str, Chem.Mol] = {}
        for name, smarts in self.patterns.items():
            try:
                mol = Chem.MolFromSmarts(smarts)
                if mol is not None:
                    self.compiled_patterns[name] = mol
            except Exception:
                print(f"Warning: Could not compile SMARTS pattern for {name}: {smarts}")

    def classify(self, smiles: str) -> Dict[str, bool]:
        smiles_clean = smiles.replace("*", "[*]")

        try:
            mol = Chem.MolFromSmiles(smiles_clean)
            if mol is None:
                mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {name: False for name in self.patterns}
        except Exception:
            return {name: False for name in self.patterns}

        results: Dict[str, bool] = {}
        for name, pattern in self.compiled_patterns.items():
            try:
                results[name] = bool(mol.HasSubstructMatch(pattern))
            except Exception:
                results[name] = False

        # Ensure every configured class exists in output.
        for name in self.patterns:
            results.setdefault(name, False)
        return results

    def get_class_label(self, smiles: str) -> Optional[str]:
        classification = self.classify(smiles)
        for name, matched in classification.items():
            if matched:
                return name
        return None

    def batch_classify(self, smiles_list: List[str], show_progress: bool = True) -> pd.DataFrame:
        rows = []
        iterator = tqdm(smiles_list, desc="Classifying") if show_progress else smiles_list

        for smiles in iterator:
            row = {"smiles": smiles}
            row.update(self.classify(smiles))
            rows.append(row)

        return pd.DataFrame(rows)

    def classify_backbone(self, smiles: str) -> Dict[str, bool]:
        """Classify polymer checking that the functional group is on the backbone.

        For classes in ``BACKBONE_CLASS_MATCH_CLASSES``, the SMARTS match must
        lie on the shortest path between the two ``*`` connection points.
        Other classes fall back to the standard ``classify`` logic.
        """
        smiles_clean = smiles.replace("*", "[*]")
        try:
            mol = Chem.MolFromSmiles(smiles_clean)
            if mol is None:
                mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {name: False for name in self.patterns}
        except Exception:
            return {name: False for name in self.patterns}

        backbone_set = _get_backbone_atom_set(mol)

        results: Dict[str, bool] = {}
        for name, pattern in self.compiled_patterns.items():
            try:
                if backbone_set is not None and name in BACKBONE_CLASS_MATCH_CLASSES:
                    results[name] = _has_backbone_substructure_match(
                        mol, pattern, backbone_set
                    )
                else:
                    results[name] = bool(mol.HasSubstructMatch(pattern))
            except Exception:
                results[name] = False

        for name in self.patterns:
            results.setdefault(name, False)
        return results

    def filter_by_class(self, smiles_list: List[str], target_class: str) -> List[str]:
        if target_class not in self.patterns:
            raise ValueError(f"Unknown class: {target_class}")

        matched = []
        for smiles in smiles_list:
            classification = self.classify(smiles)
            if classification.get(target_class, False):
                matched.append(smiles)
        return matched


def _get_backbone_atom_set(mol: Chem.Mol) -> Optional[Set[int]]:
    """Return the set of atom indices on the backbone (shortest path between ``*`` atoms).

    Returns ``None`` if the molecule does not have exactly two ``*`` atoms
    or no path exists between them.
    """
    star_indices = [
        atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0
    ]
    if len(star_indices) != 2:
        return None
    try:
        path = rdmolops.GetShortestPath(mol, star_indices[0], star_indices[1])
    except Exception:
        return None
    if not path:
        return None
    return set(path)


def _has_backbone_substructure_match(
    mol: Chem.Mol,
    pattern: Chem.Mol,
    backbone_set: Set[int],
) -> bool:
    """Check if *pattern* matches atoms predominantly on the backbone.

    For each substructure match, counts how many matched atoms are on the
    backbone path.  A match is accepted if at least ``ceil(n_atoms / 2)``
    matched atoms are on the backbone—this tolerates pendant ``=O`` oxygens
    while requiring the core chain atoms to be backbone-resident.
    """
    matches = mol.GetSubstructMatches(pattern)
    if not matches:
        return False
    min_on_backbone = max(2, (pattern.GetNumAtoms() + 1) // 2)
    for match in matches:
        on_backbone = sum(1 for idx in match if idx in backbone_set)
        if on_backbone >= min_on_backbone:
            return True
    return False


def is_class_motif_on_backbone(smiles: str, smarts: str) -> bool:
    """Convenience function: does *smarts* match backbone atoms in *smiles*?

    Returns ``False`` if the molecule is invalid, has != 2 ``*`` atoms, or
    no SMARTS match sits on the backbone.
    """
    smiles_clean = smiles.replace("*", "[*]")
    try:
        mol = Chem.MolFromSmiles(smiles_clean)
        if mol is None:
            mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
    except Exception:
        return False

    backbone_set = _get_backbone_atom_set(mol)
    if backbone_set is None:
        return False

    try:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            return False
    except Exception:
        return False

    return _has_backbone_substructure_match(mol, pattern, backbone_set)
