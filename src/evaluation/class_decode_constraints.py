"""Utilities for Step 6 decode-time polymer-class constraints."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from src.data.tokenizer import PSmilesTokenizer
from src.evaluation.polymer_class import PolymerClassifier


# Short, token-safe motif fragments used when automatic mining is unavailable.
DEFAULT_CLASS_MOTIFS: Dict[str, List[str]] = {
    "polyimide": ["CC(=O)NC(=O)C", "C(=O)NC(=O)", "NC(=O)C(=O)"],
    "polyester": ["CC(=O)OC", "C(=O)OC", "OC(=O)C"],
    "polyamide": ["CC(=O)NC", "C(=O)NC", "NC(=O)C"],
    "polyurethane": ["OC(=O)NC", "NC(=O)OC", "C(=O)NCO"],
    "polyether": ["COCCOC", "COC", "CCOCC"],
    "polysiloxane": ["SiOSi", "OSiO", "SiOCCOSi"],
    "polycarbonate": ["OC(=O)OC", "COC(=O)OC", "OC(=O)O"],
    "polysulfone": ["CS(=O)(=O)C", "S(=O)(=O)", "CS(=O)(=O)CC"],
    "polyacrylate": ["CC(=O)OC", "CC(C)C(=O)O", "C(C)C(=O)O"],
    "polystyrene": ["Cc1ccccc1", "c1ccccc1C", "CC(c1ccccc1)"],
}

_BOND_TOKENS = {"-", "=", "#", "/", "\\"}
_FORBIDDEN_EDGE_TOKENS = _BOND_TOKENS | {"(", ")", "."}


def load_decode_constraint_source_smiles(data_dir: Path) -> List[str]:
    """Load a small local polymer corpus for motif mining."""
    candidates = [
        data_dir / "Polymer" / "PolyInfo_Homopolymer.csv",
        data_dir / "water_solvent" / "water_miscible_polymer.csv",
        data_dir / "water_solvent" / "water_immiscible_polymer.csv",
    ]

    smiles_values: List[str] = []
    seen: set[str] = set()
    for csv_path in candidates:
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        smiles_col = next(
            (col for col in ["SMILES", "smiles", "p_smiles"] if col in df.columns),
            None,
        )
        if smiles_col is None:
            continue

        for smi in df[smiles_col].dropna().astype(str).tolist():
            value = smi.strip()
            if not value or value in seen:
                continue
            seen.add(value)
            smiles_values.append(value)
    return smiles_values


def load_class_motifs_from_json(path: Path, target_class: str) -> List[str]:
    """Load class motifs from a JSON file.

    Accepted file shapes:
    - {"polyimide": ["...", "..."]}
    - {"target_class": "polyimide", "motifs": ["...", "..."]}
    - ["...", "..."]  (applies only when used for a single class)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [str(x).strip() for x in data if str(x).strip()]

    if isinstance(data, dict):
        if isinstance(data.get("motifs"), list):
            if "target_class" in data and str(data["target_class"]).strip().lower() not in {
                "",
                target_class,
            }:
                raise ValueError(
                    f"decode constraint JSON target_class={data['target_class']} does not match requested class={target_class}"
                )
            return [str(x).strip() for x in data["motifs"] if str(x).strip()]

        if target_class in data and isinstance(data[target_class], list):
            return [str(x).strip() for x in data[target_class] if str(x).strip()]

    raise ValueError(f"Could not find motifs for class '{target_class}' in {path}")


def _is_valid_fragment_tokens(tokens: Sequence[str]) -> bool:
    if not tokens:
        return False
    if any(token == "*" for token in tokens):
        return False
    if tokens[0] in _FORBIDDEN_EDGE_TOKENS or tokens[-1] in _FORBIDDEN_EDGE_TOKENS:
        return False
    if any(token.isdigit() or token.startswith("%") for token in tokens):
        return False

    paren_depth = 0
    atom_like_count = 0
    for token in tokens:
        if token == "(":
            paren_depth += 1
        elif token == ")":
            paren_depth -= 1
            if paren_depth < 0:
                return False
        if token.startswith("[") or token[:1].isalpha():
            atom_like_count += 1
    if paren_depth != 0:
        return False
    if atom_like_count < 3:
        return False
    if len("".join(tokens)) < 6:
        return False
    return True


def _count_fragment_support(
    smiles_list: Iterable[str],
    tokenizer: PSmilesTokenizer,
    *,
    min_token_len: int,
    max_token_len: int,
) -> Counter:
    counts: Counter = Counter()
    for smiles in smiles_list:
        tokens = tokenizer.tokenize(str(smiles))
        seen_in_sample: set[str] = set()
        n_tokens = len(tokens)
        for start in range(n_tokens):
            upper = min(max_token_len, n_tokens - start)
            for span_len in range(min_token_len, upper + 1):
                frag_tokens = tokens[start : start + span_len]
                if not _is_valid_fragment_tokens(frag_tokens):
                    continue
                seen_in_sample.add("".join(frag_tokens))
        counts.update(seen_in_sample)
    return counts


def _normalize_fragments(
    fragments: Sequence[str],
    tokenizer: PSmilesTokenizer,
    *,
    max_motifs: int,
) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for raw in fragments:
        frag = str(raw).strip()
        if not frag or frag in seen:
            continue
        tokens = tokenizer.tokenize(frag)
        if "".join(tokens) != frag:
            continue
        token_ids = [tokenizer.vocab.get(token, tokenizer.unk_token_id) for token in tokens]
        if any(token_id == tokenizer.unk_token_id for token_id in token_ids):
            continue
        if not _is_valid_fragment_tokens(tokens):
            continue
        seen.add(frag)
        out.append(frag)
        if len(out) >= int(max_motifs):
            break
    return out


def mine_class_decode_motifs(
    *,
    target_class: str,
    tokenizer: PSmilesTokenizer,
    source_smiles: Sequence[str],
    patterns: Dict[str, str],
    max_motifs: int = 6,
    min_token_len: int = 4,
    max_token_len: int = 10,
    min_support: int = 3,
    min_precision: float = 0.75,
) -> List[str]:
    """Mine short class-enriched p-SMILES fragments."""
    if not source_smiles:
        return []

    classifier = PolymerClassifier(patterns=patterns)
    positive: List[str] = []
    negative: List[str] = []
    target_key = str(target_class).strip().lower()
    for smiles in source_smiles:
        try:
            is_positive = bool(classifier.classify(str(smiles)).get(target_key, False))
        except Exception:
            is_positive = False
        if is_positive:
            positive.append(str(smiles))
        else:
            negative.append(str(smiles))

    if not positive:
        return []

    pos_counts = _count_fragment_support(
        positive,
        tokenizer,
        min_token_len=min_token_len,
        max_token_len=max_token_len,
    )
    neg_counts = _count_fragment_support(
        negative,
        tokenizer,
        min_token_len=min_token_len,
        max_token_len=max_token_len,
    )

    ranked: List[Tuple[float, int, int, str]] = []
    for fragment, pos_support in pos_counts.items():
        neg_support = int(neg_counts.get(fragment, 0))
        total_support = pos_support + neg_support
        if pos_support < int(min_support) or total_support <= 0:
            continue
        precision = float(pos_support) / float(total_support)
        if precision < float(min_precision):
            continue
        ranked.append((precision, pos_support, len(fragment), fragment))

    ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return _normalize_fragments([fragment for _, _, _, fragment in ranked], tokenizer, max_motifs=max_motifs)


def resolve_class_decode_motifs(
    *,
    target_class: str,
    tokenizer: PSmilesTokenizer,
    source_smiles: Sequence[str],
    patterns: Dict[str, str],
    configured_bank_path: Path | None = None,
    max_motifs: int = 6,
    resolution_strategy: str = "configured_or_local_mined_or_defaults",
) -> Tuple[List[str], str]:
    """Resolve decode motifs from user config, mined corpus fragments, or defaults.

    resolution_strategy controls fallback order:
    - configured_or_local_mined_or_defaults: current legacy behavior
    - configured_or_defaults: prefer explicit or curated defaults; skip auto-mined local motifs
    - configured_only: require an explicit configured motif bank
    - defaults_only: require built-in defaults
    """
    target_key = str(target_class).strip().lower()
    strategy = str(resolution_strategy).strip().lower()
    valid_strategies = {
        "configured_or_local_mined_or_defaults",
        "configured_or_defaults",
        "configured_only",
        "defaults_only",
    }
    if strategy not in valid_strategies:
        raise ValueError(
            f"Unknown decode-constraint resolution_strategy='{resolution_strategy}'. "
            f"Expected one of {sorted(valid_strategies)}"
        )

    if configured_bank_path is not None:
        configured = load_class_motifs_from_json(configured_bank_path, target_key)
        normalized = _normalize_fragments(configured, tokenizer, max_motifs=max_motifs)
        if not normalized:
            raise ValueError(
                f"Configured decode-constraint motif bank {configured_bank_path} contains no valid motifs for class '{target_key}'"
            )
        return normalized, "configured_json"

    if strategy == "configured_only":
        raise ValueError(
            f"Configured decode-constraint motif bank required for class '{target_key}', but none was provided."
        )

    if strategy == "configured_or_local_mined_or_defaults":
        mined = mine_class_decode_motifs(
            target_class=target_key,
            tokenizer=tokenizer,
            source_smiles=source_smiles,
            patterns=patterns,
            max_motifs=max_motifs,
        )
        if mined:
            return mined, "mined_local_corpus"

    fallback = _normalize_fragments(DEFAULT_CLASS_MOTIFS.get(target_key, []), tokenizer, max_motifs=max_motifs)
    if fallback:
        return fallback, "fallback_defaults"

    raise ValueError(
        f"Could not resolve any decode-time motifs for class '{target_key}'. "
        "Provide decode_constraint_motif_bank_json or add defaults for this class."
    )
