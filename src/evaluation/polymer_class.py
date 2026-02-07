"""Polymer family classifier using SMARTS patterns."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from rdkit import Chem
from tqdm import tqdm


class PolymerClassifier:
    """Classify polymers into families using SMARTS patterns."""

    DEFAULT_PATTERNS = {
        "polyimide": "[#6](=O)-[#7]-[#6](=O)",
        "polyester": "[#6](=O)-[#8]-[#6]",
        "polyamide": "[#6](=O)-[#7]-[#6]",
        "polyurethane": "[#8]-[#6](=O)-[#7]",
        "polyether": "[#6]-[#8]-[#6]",
        "polysiloxane": "[Si]-[#8]-[Si]",
        "polycarbonate": "[#8]-[#6](=O)-[#8]",
        "polysulfone": "[#6]-[S](=O)(=O)-[#6]",
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

    def filter_by_class(self, smiles_list: List[str], target_class: str) -> List[str]:
        if target_class not in self.patterns:
            raise ValueError(f"Unknown class: {target_class}")

        matched = []
        for smiles in smiles_list:
            classification = self.classify(smiles)
            if classification.get(target_class, False):
                matched.append(smiles)
        return matched
