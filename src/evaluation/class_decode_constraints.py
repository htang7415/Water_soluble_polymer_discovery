"""Utilities for Step 6 decode-time polymer-class constraints."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from src.data.tokenizer import PSmilesTokenizer
from src.evaluation.polymer_class import BACKBONE_CLASS_MATCH_CLASSES, PolymerClassifier


# Short, token-safe motif fragments used when automatic mining is unavailable.
# Longer motifs with aromatic context are listed first to better survive
# diffusion infilling while preserving SMARTS-matchable imide/ester/amide bonds.
DEFAULT_CLASS_MOTIFS: Dict[str, List[str]] = {
    "polyimide": [
        # SMARTS: [#6][CX3](=[OX1])[NX3][CX3](=[OX1])[#6]
        "c(C(=O))NC(=O)c",     # aromatic-context imide linkage
        "c(C(=O)NC(=O))c",     # aromatic-sandwich imide
        "CC(=O)NC(=O)CC",       # longer aliphatic imide
        "CC(=O)NC(=O)C",        # classic imide motif
        "NC(=O)C(=O)C",         # partial imide with trailing context
        "cC(=O)NC(=O)c",       # aromatic-edge imide core
    ],
    "polyester": [
        # SMARTS: [#6][CX3](=[OX1])[OX2][#6]
        "c(C(=O)O)c",          # aromatic-context ester
        "c(OC(=O))c",          # reversed aromatic ester
        "CC(=O)OCC",            # longer aliphatic ester
        "CC(=O)OC",             # classic ester motif
        "C(=O)OC",              # minimal ester core
        "OC(=O)C",              # reversed ester
    ],
    "polyamide": [
        # SMARTS: [#6][CX3](=[OX1])[NX3;!$([N]([C](=O))[C](=O))][#6;!$([CX3](=[OX1]))]
        "c(C(=O)NC)c",         # aromatic-context amide with continuation
        "c(C(=O)N)c",          # aromatic-context amide
        "CC(=O)NCC",            # longer aliphatic amide
        "CC(=O)NC",             # classic amide motif
        "C(=O)NC",              # minimal amide core
        "NC(=O)C",              # reversed amide
    ],
    "polyurethane": [
        # SMARTS: [#6][OX2][CX3](=[OX1])[NX3][#6]
        "c(OC(=O)N)c",         # aromatic-context urethane
        "COC(=O)NCC",           # longer urethane linkage
        "CCOC(=O)NC",           # aliphatic urethane with extra context
        "OC(=O)NC",             # classic urethane
        "NC(=O)OC",             # reversed urethane
        "C(=O)NCO",             # partial urethane
    ],
    "polyether": [
        # SMARTS: [#6;!$([CX3](=[OX1]))][OX2][#6;!$([CX3](=[OX1]))]
        "CCOCCOC",              # long ether chain
        "cOCCOc",               # aromatic-context ether
        "CCOCCOCC",             # extended ether
        "COCCOC",               # classic ether chain
        "CCCOCC",               # short ether with context
    ],
    "polysiloxane": [
        # SMARTS: [Si][OX2][Si]
        "SiOSiOSi",            # extended siloxane chain
        "CSiOSiC",              # Si with carbon context
        "CSiOSiOC",             # longer siloxane with C context
        "SiOCCOSi",             # siloxane with carbon spacer
    ],
    "polycarbonate": [
        # SMARTS: [#6][OX2][CX3](=[OX1])[OX2][#6]
        "c(OC(=O)O)c",         # aromatic-context carbonate
        "COC(=O)OCC",           # longer carbonate
        "CCOC(=O)OC",           # aliphatic carbonate with extra context
        "COC(=O)OC",            # extended carbonate
        "OC(=O)OC",             # classic carbonate
    ],
    "polysulfone": [
        # SMARTS: [#6][SX4](=[OX1])(=[OX1])[#6]
        "c(S(=O)(=O))c",       # aromatic-context sulfone
        "CCS(=O)(=O)CC",       # longer sulfone
        "CS(=O)(=O)CC",         # extended sulfone
        "CS(=O)(=O)C",          # classic sulfone motif
    ],
    "polyacrylate": [
        # SMARTS: [#6]-[#6](=O)-[#8]
        "c(C(=O)OC)c",         # aromatic-context acrylate
        "CC(C)C(=O)OC",        # methacrylate-like with ester
        "CC(=O)OCC",            # longer acrylate
        "CC(=O)OC",             # classic acrylate
        "CC(C)C(=O)O",          # partial methacrylate
        "C(C)C(=O)O",           # minimal acrylate
    ],
    "polystyrene": [
        # SMARTS: [#6]-[#6](c1ccccc1)-[#6]
        # Ring digits are forbidden in motifs, so we provide aromatic-C chains
        # that the diffusion model can complete into phenyl rings.
        "CC(cccccc)C",          # backbone with 6 aromatic C (full ring hint)
        "CC(ccccc)C",           # backbone with 5 aromatic C
        "CC(cccc)C",            # backbone with 4 aromatic C
        "CC(ccc)C",             # backbone with 3 aromatic C
    ],
}

# Backbone-template core fragments used by Step 6_2 constrained sampling.
# These are inserted into a fixed ``*...core...*`` scaffold so the defining
# backbone linkage is already present before diffusion infills the remaining
# context. Side-chain classes intentionally do not use this path.
#
# Keep these fragments aliphatic-only. Aromatic lowercase ``c`` fragments are
# fragile here because the sampler cannot guarantee the ring closures needed to
# make a fixed aromatic subgraph valid.
DEFAULT_BACKBONE_TEMPLATE_CORES: Dict[str, List[str]] = {
    "polyimide": [
        "CCC(=O)NC(=O)CCC",
        "CCOC(=O)NC(=O)OCC",
        "CC(=O)NC(=O)CC",
        "CC(=O)NC(=O)C",
    ],
    "polyester": [
        "CCOC(=O)OCCC",
        "CC(=O)OCC",
        "CC(=O)OC",
    ],
    "polyamide": [
        "CCC(=O)NCCC",
        "CC(=O)NCC",
        "CC(=O)NC",
    ],
    "polyurethane": [
        "CCOC(=O)NC",
        "COC(=O)NCC",
    ],
    "polyether": [
        "CCOCCOCCOCC",
        "CCOCCOCCOC",
        "CCOCCOCC",
        "COCCOC",
    ],
    "polysiloxane": [
        "CSiOSiC",
        "SiOSiOSi",
        "CSiOSiOC",
    ],
    "polycarbonate": [
        "CCOC(=O)OCCO",
        "CCOC(=O)OC",
        "COC(=O)OC",
    ],
    "polysulfone": [
        "CCCS(=O)(=O)CCC",
        "CCS(=O)(=O)CCOCC",
        "CCS(=O)(=O)CC",
        "CS(=O)(=O)CC",
    ],
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


def resolve_class_backbone_template_cores(
    *,
    target_class: str,
    tokenizer: PSmilesTokenizer,
    max_templates: int = 3,
) -> Tuple[List[str], str]:
    """Resolve default backbone-template cores for backbone-defined classes."""
    target_key = str(target_class).strip().lower()
    if target_key not in BACKBONE_CLASS_MATCH_CLASSES:
        return [], "not_backbone_class"

    fragments = _normalize_fragments(
        DEFAULT_BACKBONE_TEMPLATE_CORES.get(target_key, []),
        tokenizer,
        max_motifs=max_templates,
    )
    if fragments:
        return fragments, "default_backbone_templates"
    return [], "unavailable"


def resolve_class_decode_length_prior(
    *,
    target_class: str,
    tokenizer: PSmilesTokenizer,
    source_smiles: Sequence[str],
    patterns: Dict[str, str],
    max_length: int,
) -> Tuple[List[int], str]:
    """Resolve a lightweight class-specific token-length prior from local source polymers."""
    if not source_smiles:
        return [], "unavailable"

    classifier = PolymerClassifier(patterns=patterns)
    target_key = str(target_class).strip().lower()
    lengths: List[int] = []
    for smiles in source_smiles:
        try:
            is_positive = bool(classifier.classify(str(smiles)).get(target_key, False))
        except Exception:
            is_positive = False
        if not is_positive:
            continue
        token_len = len(tokenizer.tokenize(str(smiles))) + 2
        token_len = max(2, min(int(max_length), int(token_len)))
        lengths.append(token_len)

    if not lengths:
        return [], "unavailable"
    return lengths, "local_class_corpus"


def compute_class_token_logit_bias(
    *,
    target_class: str,
    tokenizer: PSmilesTokenizer,
    source_smiles: Sequence[str],
    patterns: Dict[str, str],
    bias_strength: float = 1.5,
    min_class_count: int = 10,
) -> Optional[List[float]]:
    """Compute per-token logit bias to steer diffusion sampling toward target class.

    Computes log frequency ratio of tokens in the target class vs the rest of the
    corpus, then scales by ``bias_strength``.  Tokens enriched in the target class
    receive positive bias; depleted tokens receive negative bias.

    Returns:
        List of length ``tokenizer.vocab_size`` with additive logit biases,
        or ``None`` if the class corpus is too small.
    """
    if not source_smiles:
        return None

    classifier = PolymerClassifier(patterns=patterns)
    target_key = str(target_class).strip().lower()
    positive_smiles: List[str] = []
    negative_smiles: List[str] = []
    for smi in source_smiles:
        try:
            is_pos = bool(classifier.classify(str(smi)).get(target_key, False))
        except Exception:
            is_pos = False
        if is_pos:
            positive_smiles.append(str(smi))
        else:
            negative_smiles.append(str(smi))

    if len(positive_smiles) < int(min_class_count):
        return None

    vocab_size = tokenizer.vocab_size
    pos_counts = [0] * vocab_size
    neg_counts = [0] * vocab_size
    pos_total = 0
    neg_total = 0

    for smi in positive_smiles:
        tokens = tokenizer.tokenize(smi)
        for tok in tokens:
            tid = tokenizer.vocab.get(tok, tokenizer.unk_token_id)
            pos_counts[tid] += 1
            pos_total += 1
    for smi in negative_smiles:
        tokens = tokenizer.tokenize(smi)
        for tok in tokens:
            tid = tokenizer.vocab.get(tok, tokenizer.unk_token_id)
            neg_counts[tid] += 1
            neg_total += 1

    if pos_total == 0 or neg_total == 0:
        return None

    smoothing = 1.0
    bias = [0.0] * vocab_size
    for tid in range(vocab_size):
        pos_freq = (pos_counts[tid] + smoothing) / (pos_total + smoothing * vocab_size)
        neg_freq = (neg_counts[tid] + smoothing) / (neg_total + smoothing * vocab_size)
        bias[tid] = float(bias_strength) * math.log(pos_freq / neg_freq)

    # Zero out special tokens
    for special in tokenizer.SPECIAL_TOKENS:
        if special in tokenizer.vocab:
            bias[tokenizer.vocab[special]] = 0.0
    # Zero out star token (star placement is handled by constraints, not bias)
    star_id = tokenizer.get_star_token_id()
    if 0 <= star_id < vocab_size:
        bias[star_id] = 0.0

    return bias
