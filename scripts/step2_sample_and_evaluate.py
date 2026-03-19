#!/usr/bin/env python
"""Step 2: Sample from backbone and evaluate generative metrics."""

from __future__ import annotations

import os
import sys
import argparse
import re
import time
import json
from pathlib import Path
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

import torch
import pandas as pd
import numpy as np
from collections import Counter

from src.utils.config import load_config, save_config
from src.utils.plotting import PlotUtils
from src.utils.chemistry import (
    compute_sa_score,
    count_stars,
    check_validity,
    canonicalize_smiles,
    has_terminal_connection_stars,
    batch_compute_fingerprints,
    compute_pairwise_diversity,
)
from src.utils.model_scales import get_model_config, get_results_dir
from src.data.tokenizer import PSmilesTokenizer
from src.model.backbone import DiffusionBackbone
from src.model.diffusion import DiscreteMaskingDiffusion
from src.sampling.sampler import ConstrainedSampler
from src.evaluation.generative_metrics import GenerativeEvaluator
from src.evaluation.polymer_class import PolymerClassifier
from src.utils.reproducibility import seed_everything, save_run_metadata
from src.utils.reporting import save_step_summary, save_artifact_manifest, write_initial_log



# Constraint logging helpers
BOND_CHARS = set(['-', '=', '#', '/', '\\'])


def _passes_target_star_rule(smiles: str, target_stars: int) -> bool:
    """Require the requested star count, plus terminal '*' atoms for 2-star p-SMILES."""
    if count_stars(smiles) != int(target_stars):
        return False
    if int(target_stars) == 2:
        return has_terminal_connection_stars(smiles, expected_stars=2)
    return True


def _sa_score_cached(smiles: str, cache: dict[str, float | None]) -> float | None:
    key = str(smiles)
    if key not in cache:
        cache[key] = compute_sa_score(key)
    return cache[key]


def _smiles_constraint_violations(smiles: str) -> dict:
    if not smiles:
        return {
            "star_count": True,
            "terminal_star_atoms": True,
            "bond_placement": True,
            "paren_balance": True,
            "empty_parens": True,
            "ring_closure": True,
        }

    star_violation = count_stars(smiles) != 2
    terminal_star_violation = (not star_violation) and (not has_terminal_connection_stars(smiles, expected_stars=2))
    empty_parens = "()" in smiles

    # Parenthesis balance
    depth = 0
    paren_violation = False
    for ch in smiles:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth < 0:
                paren_violation = True
                break
    if depth != 0:
        paren_violation = True

    # Bond placement (heuristic)
    bond_violation = False
    prev = None
    for ch in smiles:
        if ch in BOND_CHARS:
            if prev is None or prev in BOND_CHARS or prev in '()':
                bond_violation = True
                break
        if ch.strip() == "":
            continue
        prev = ch

    # Ring closure (digits and %nn tokens must appear exactly twice)
    ring_tokens = re.findall(r'%\d{2}', smiles)
    no_percent = re.sub(r'%\d{2}', '', smiles)
    ring_tokens += re.findall(r'\d', no_percent)
    ring_violation = False
    if ring_tokens:
        counts = Counter(ring_tokens)
        ring_violation = any(c != 2 for c in counts.values())

    return {
        "star_count": star_violation,
        "terminal_star_atoms": terminal_star_violation,
        "bond_placement": bond_violation,
        "paren_balance": paren_violation,
        "empty_parens": empty_parens,
        "ring_closure": ring_violation,
    }


def compute_smiles_constraint_metrics(smiles_list, method, representation, model_size):
    total = len(smiles_list)
    violations = {
        "star_count": 0,
        "terminal_star_atoms": 0,
        "bond_placement": 0,
        "paren_balance": 0,
        "empty_parens": 0,
        "ring_closure": 0,
    }

    for smiles in smiles_list:
        flags = _smiles_constraint_violations(smiles)
        for key, violated in flags.items():
            if violated:
                violations[key] += 1

    rows = []
    for constraint, count in violations.items():
        rate = count / total if total > 0 else 0.0
        rows.append({
            "method": method,
            "representation": representation,
            "model_size": model_size,
            "constraint": constraint,
            "total": total,
            "violations": count,
            "violation_rate": round(rate, 4),
        })
    return rows


def _filter_valid_samples(
    smiles_list,
    require_target_stars: bool,
    target_stars: int,
    required_polymer_class: str | None = None,
    polymer_classifier: PolymerClassifier | None = None,
    enforce_backbone_class_match: bool = False,
):
    valid = []
    for smiles in smiles_list:
        if not check_validity(smiles):
            continue
        if require_target_stars and not _passes_target_star_rule(smiles, target_stars):
            continue
        if required_polymer_class is not None and polymer_classifier is not None:
            if enforce_backbone_class_match:
                if not bool(polymer_classifier.classify_backbone(smiles).get(required_polymer_class, False)):
                    continue
            elif not bool(polymer_classifier.classify(smiles).get(required_polymer_class, False)):
                continue
        valid.append(smiles)
    return valid


def _filter_step2_target_candidates(
    smiles_list,
    training_canonical: set[str],
    seen_canonical: set[str],
    require_target_stars: bool,
    target_stars: int,
    sa_max: float,
    sa_cache: dict[str, float | None] | None = None,
    required_polymer_class: str | None = None,
    polymer_classifier: PolymerClassifier | None = None,
    enforce_novelty: bool = True,
    enforce_unique: bool = True,
    enforce_sa: bool = True,
    enforce_backbone_class_match: bool = False,
):
    sa_cache = sa_cache if sa_cache is not None else {}
    accepted = []
    for smiles in smiles_list:
        if not check_validity(smiles):
            continue
        if require_target_stars and not _passes_target_star_rule(smiles, target_stars):
            continue
        canonical = canonicalize_smiles(smiles)
        if canonical is None:
            continue
        if required_polymer_class is not None and polymer_classifier is not None:
            if enforce_backbone_class_match:
                if not bool(polymer_classifier.classify_backbone(smiles).get(required_polymer_class, False)):
                    continue
            elif not bool(polymer_classifier.classify(smiles).get(required_polymer_class, False)):
                continue
        if enforce_novelty and canonical in training_canonical:
            continue
        if enforce_unique and canonical in seen_canonical:
            continue
        sa = None
        if enforce_sa:
            sa = _sa_score_cached(smiles, sa_cache)
            if sa is None or float(sa) >= float(sa_max):
                continue
        if enforce_unique:
            seen_canonical.add(canonical)
        accepted.append(
            {
                "smiles": smiles,
                "canonical_smiles": canonical,
                "sa_score": float(sa) if sa is not None else np.nan,
            }
        )
    return accepted


def _load_decode_constraint_fragments(path: Path, target_class: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [str(x).strip() for x in data if str(x).strip()]

    if isinstance(data, dict):
        if isinstance(data.get("motifs"), list):
            target_in_file = str(data.get("target_class", target_class)).strip().lower()
            if target_in_file not in {"", str(target_class).strip().lower()}:
                raise ValueError(
                    f"decode constraint target_class mismatch: requested={target_class}, file={target_in_file}"
                )
            return [str(x).strip() for x in data["motifs"] if str(x).strip()]
        target_key = str(target_class).strip().lower()
        if target_key in data and isinstance(data[target_key], list):
            return [str(x).strip() for x in data[target_key] if str(x).strip()]

    raise ValueError(f"Could not load motifs for class '{target_class}' from {path}")


def _load_decode_constraint_length_prior(path: Path, target_class: str) -> List[int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_lengths: List[int] = []
    if isinstance(data, list):
        raw_lengths = [int(x) for x in data]
    elif isinstance(data, dict):
        target_key = str(target_class).strip().lower()
        if isinstance(data.get("lengths"), list):
            target_in_file = str(data.get("target_class", target_key)).strip().lower()
            if target_in_file not in {"", target_key}:
                raise ValueError(
                    f"decode constraint length prior target_class mismatch: requested={target_key}, file={target_in_file}"
                )
            raw_lengths = [int(x) for x in data["lengths"]]
        elif target_key in data and isinstance(data[target_key], list):
            raw_lengths = [int(x) for x in data[target_key]]

    return [int(x) for x in raw_lengths if int(x) >= 2]


def _prepare_decode_constraint_token_ids(
    tokenizer: PSmilesTokenizer,
    fragments: List[str],
) -> List[List[int]]:
    token_ids: List[List[int]] = []
    for fragment in fragments:
        tokens = tokenizer.tokenize(str(fragment))
        if not tokens or "".join(tokens) != str(fragment):
            continue
        ids = [tokenizer.vocab.get(token, tokenizer.unk_token_id) for token in tokens]
        if any(token_id == tokenizer.unk_token_id for token_id in ids):
            continue
        if tokenizer.get_star_token_id() in ids:
            continue
        token_ids.append(ids)
    return token_ids


def _build_decode_constraint_spans(
    *,
    motif_token_ids: List[List[int]],
    lengths: List[int],
    center_min_frac: float,
    center_max_frac: float,
    seq_length: int,
) -> tuple[List[List[int]], List[int], List[int]]:
    if not motif_token_ids:
        raise ValueError("motif_token_ids is empty")
    if not lengths:
        return [], [], []
    if not (0.0 <= center_min_frac <= center_max_frac <= 1.0):
        raise ValueError(
            "decode constraint center fractions must satisfy 0 <= min <= max <= 1"
        )

    max_motif_len = max(len(ids) for ids in motif_token_ids)
    adjusted_lengths = [
        min(
            seq_length,
            max(int(raw_length), max_motif_len + 4),
        )
        for raw_length in lengths
    ]

    chosen_spans: List[List[int]] = []
    start_positions: List[int] = []
    for effective_length in adjusted_lengths:
        fitting = [ids for ids in motif_token_ids if len(ids) <= max(1, effective_length - 2)]
        if not fitting:
            raise ValueError(
                f"No decode-time motif fits effective sequence length {effective_length}. "
                "Increase allowed lengths or shorten motifs."
            )
        motif_ids = fitting[np.random.randint(0, len(fitting))]
        max_start = int(effective_length) - 1 - len(motif_ids)
        if max_start < 1:
            raise ValueError(
                f"Motif length {len(motif_ids)} does not fit sequence length {effective_length}"
            )

        if center_min_frac == center_max_frac:
            center_frac = center_min_frac
        else:
            center_frac = float(np.random.uniform(center_min_frac, center_max_frac))
        center_target = center_frac * float(max(1, effective_length - 1))
        candidate_start = int(round(center_target - (0.5 * len(motif_ids))))
        start = max(1, min(max_start, candidate_start))
        chosen_spans.append(motif_ids)
        start_positions.append(start)

    return chosen_spans, start_positions, adjusted_lengths


def _place_multi_spans_for_length(
    *,
    motifs: List[List[int]],
    effective_length: int,
    center_min_frac: float,
    center_max_frac: float,
    min_gap_tokens: int = 2,
) -> List[int]:
    if not motifs:
        return []

    usable_token_count = max(0, int(effective_length) - 2)
    required_token_count = sum(len(motif_ids) for motif_ids in motifs) + (
        max(0, len(motifs) - 1) * int(min_gap_tokens)
    )
    if required_token_count > usable_token_count:
        raise ValueError(
            f"Could not place {len(motifs)} decode-time motifs within effective length {effective_length}"
        )

    if len(motifs) == 1:
        motif_ids = motifs[0]
        max_start = int(effective_length) - 1 - len(motif_ids)
        center_frac = (
            center_min_frac
            if center_min_frac == center_max_frac
            else float(np.random.uniform(center_min_frac, center_max_frac))
        )
        center_target = center_frac * float(max(1, effective_length - 1))
        candidate_start = int(round(center_target - (0.5 * len(motif_ids))))
        return [max(1, min(max_start, candidate_start))]

    anchors = np.linspace(center_min_frac, center_max_frac, len(motifs) + 2)[1:-1]
    future_requirements: List[int] = [0] * len(motifs)
    running_requirement = 0
    for idx in range(len(motifs) - 1, -1, -1):
        future_requirements[idx] = running_requirement
        running_requirement += len(motifs[idx]) + int(min_gap_tokens)

    starts: List[int] = []
    prev_end = 1
    for idx, (anchor_frac, motif_ids) in enumerate(zip(anchors, motifs)):
        max_start = (
            int(effective_length) - 1 - len(motif_ids) - future_requirements[idx]
        )
        candidate_start = int(round(anchor_frac * float(max(1, effective_length - 1)) - (0.5 * len(motif_ids))))
        min_start = 1 if idx == 0 else prev_end + int(min_gap_tokens)
        if max_start < min_start:
            raise ValueError(
                f"Could not place {len(motifs)} decode-time motifs within effective length {effective_length}"
            )
        start = max(min_start, min(max_start, candidate_start))
        starts.append(start)
        prev_end = start + len(motif_ids)

    return starts


def _build_decode_constraint_multi_spans(
    *,
    motif_token_ids: List[List[int]],
    lengths: List[int],
    center_min_frac: float,
    center_max_frac: float,
    seq_length: int,
    spans_per_sample: int,
    min_gap_tokens: int = 2,
) -> tuple[List[List[List[int]]], List[List[int]], List[int]]:
    if spans_per_sample < 1:
        raise ValueError(f"spans_per_sample must be >= 1, got {spans_per_sample}")
    if spans_per_sample == 1:
        spans, starts, adjusted = _build_decode_constraint_spans(
            motif_token_ids=motif_token_ids,
            lengths=lengths,
            center_min_frac=center_min_frac,
            center_max_frac=center_max_frac,
            seq_length=seq_length,
        )
        return [[span] for span in spans], [[start] for start in starts], adjusted

    max_motif_len = max(len(ids) for ids in motif_token_ids)
    adjusted_lengths = [
        min(
            seq_length,
            max(int(raw_length), max_motif_len + 4),
        )
        for raw_length in lengths
    ]

    chosen_spans: List[List[List[int]]] = []
    start_positions: List[List[int]] = []
    for effective_length in adjusted_lengths:
        fitting = [ids for ids in motif_token_ids if len(ids) <= max(1, effective_length - 2)]
        if not fitting:
            raise ValueError(
                f"No decode-time motif fits effective sequence length {effective_length}. "
                "Increase allowed lengths or shorten motifs."
            )
        min_len = min(len(ids) for ids in fitting)
        shortest_pool = [ids for ids in fitting if len(ids) <= (min_len + 1)]
        usable_pool = shortest_pool if shortest_pool else fitting
        max_spans_fit = max(
            1,
            int((max(1, effective_length - 2) + int(min_gap_tokens)) // (min_len + int(min_gap_tokens))),
        )
        sample_span_count = max(1, min(int(spans_per_sample), int(max_spans_fit)))

        sample_spans: List[List[int]] | None = None
        sample_starts: List[int] | None = None
        available_token_count = max(0, int(effective_length) - 2)
        for current_span_count in range(sample_span_count, 0, -1):
            pool = usable_pool if current_span_count > 1 else fitting
            for _ in range(32):
                candidate_spans = [
                    list(pool[np.random.randint(0, len(pool))])
                    for _ in range(current_span_count)
                ]
                required_token_count = sum(len(ids) for ids in candidate_spans) + (
                    max(0, current_span_count - 1) * int(min_gap_tokens)
                )
                if required_token_count > available_token_count:
                    continue
                try:
                    candidate_starts = _place_multi_spans_for_length(
                        motifs=candidate_spans,
                        effective_length=int(effective_length),
                        center_min_frac=float(center_min_frac),
                        center_max_frac=float(center_max_frac),
                        min_gap_tokens=int(min_gap_tokens),
                    )
                except ValueError:
                    continue
                sample_spans = candidate_spans
                sample_starts = candidate_starts
                break
            if sample_spans is not None and sample_starts is not None:
                break
        if sample_spans is None or sample_starts is None:
            raise ValueError(
                f"Could not construct {sample_span_count} decode-time motifs within effective length {effective_length}"
            )
        chosen_spans.append(sample_spans)
        start_positions.append(sample_starts)

    return chosen_spans, start_positions, adjusted_lengths


def _select_target_polymers(
    generated_smiles,
    training_smiles,
    target_count: int,
    target_stars: int,
    sa_max: float,
    total_sampling_points: int | None = None,
    sa_cache: dict[str, float | None] | None = None,
):
    sa_cache = sa_cache if sa_cache is not None else {}
    training_canonical = {canonicalize_smiles(s) or s for s in training_smiles}
    rows = []
    for idx, smiles in enumerate(generated_smiles, start=1):
        is_valid = check_validity(smiles)
        star_count = count_stars(smiles)
        terminal_star_ok = bool(is_valid) and (
            has_terminal_connection_stars(smiles, expected_stars=2) if int(target_stars) == 2 else True
        )
        canonical = canonicalize_smiles(smiles) if is_valid else None
        is_novel = bool(canonical) and canonical not in training_canonical
        sa = _sa_score_cached(smiles, sa_cache) if is_valid else None
        sa_ok = sa is not None and float(sa) < float(sa_max)
        rows.append(
            {
                "sample_index": idx,
                "smiles": smiles,
                "canonical_smiles": canonical,
                "is_valid": int(is_valid),
                "star_count": int(star_count),
                "terminal_star_ok": int(terminal_star_ok),
                "is_novel": int(is_novel),
                "sa_score": float(sa) if sa is not None else np.nan,
                "sa_ok": int(sa_ok),
            }
        )

    all_df = pd.DataFrame(rows)
    real_total_generated = int(total_sampling_points) if total_sampling_points is not None else int(len(all_df))
    if all_df.empty:
        summary = {
            "target_count_requested": int(target_count),
            "total_generated": int(real_total_generated),
            "total_evaluated_for_filters": 0,
            "filter_pass_count": 0,
            "filter_pass_unique": 0,
            "target_count_selected": 0,
            "selection_success_rate": 0.0,
            "final_diversity": 0.0,
            "final_mean_sa": np.nan,
            "final_std_sa": np.nan,
            "final_frac_star_eq_target": 0.0,
            "final_novelty": 0.0,
            "final_uniqueness": 0.0,
        }
        return pd.DataFrame(), summary

    filter_mask = (
        (all_df["is_valid"] == 1)
        & (all_df["star_count"] == int(target_stars))
        & ((all_df["terminal_star_ok"] == 1) if int(target_stars) == 2 else True)
        & (all_df["is_novel"] == 1)
        & (all_df["sa_ok"] == 1)
    )
    filtered = all_df.loc[filter_mask].copy()
    filtered = filtered.drop_duplicates(subset=["canonical_smiles"], keep="first").reset_index(drop=True)

    selected = filtered.head(int(target_count)).copy()
    selected.insert(0, "target_rank", np.arange(1, len(selected) + 1))
    selected["is_unique"] = 1
    selected["passes_all_filters"] = 1

    diversity = 0.0
    if len(selected) >= 2:
        fps, _ = batch_compute_fingerprints(selected["smiles"].astype(str).tolist())
        if len(fps) >= 2:
            diversity = float(compute_pairwise_diversity(fps))

    sa_vals = selected["sa_score"].to_numpy(dtype=float) if not selected.empty else np.array([])
    summary = {
        "target_count_requested": int(target_count),
        "total_generated": int(real_total_generated),
        "total_evaluated_for_filters": int(len(all_df)),
        "filter_pass_count": int(filter_mask.sum()),
        "filter_pass_unique": int(len(filtered)),
        "target_count_selected": int(len(selected)),
        "selection_success_rate": float(len(selected) / real_total_generated) if real_total_generated > 0 else 0.0,
        "final_diversity": float(diversity),
        "final_mean_sa": float(np.nanmean(sa_vals)) if sa_vals.size else np.nan,
        "final_std_sa": float(np.nanstd(sa_vals)) if sa_vals.size else np.nan,
        "final_frac_star_eq_target": float(np.mean(selected["star_count"] == int(target_stars))) if len(selected) else 0.0,
        "final_novelty": float(np.mean(selected["is_novel"])) if len(selected) else 0.0,
        "final_uniqueness": 1.0 if len(selected) > 0 else 0.0,
    }
    return selected, summary


def _append_step_log(step_dir: Path, lines) -> None:
    log_path = Path(step_dir) / "log.txt"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n")
        for line in lines:
            f.write(f"{line}\n")


def main(args):
    """Main function."""
    # Load config
    config = load_config(args.config)
    shared_cfg = config.get("chi_training", {}).get("shared", {})
    default_split_mode = str(shared_cfg.get("split_mode", "polymer")).strip().lower()
    if default_split_mode not in {"polymer", "random"}:
        default_split_mode = "polymer"
    split_mode = str(args.split_mode).strip().lower() if args.split_mode is not None else default_split_mode
    if split_mode not in {"polymer", "random"}:
        raise ValueError(f"split_mode must be one of ['polymer', 'random'], got: {split_mode}")
    sampling_cfg = config.get('sampling', {})
    temperature = float(args.temperature if args.temperature is not None else sampling_cfg.get('temperature', 1.0))
    top_k = args.top_k if args.top_k is not None else sampling_cfg.get('top_k', None)
    top_k = int(top_k) if top_k is not None else None
    if top_k is not None and top_k <= 0:
        top_k = None
    top_p = args.top_p if args.top_p is not None else sampling_cfg.get('top_p', None)
    top_p = float(top_p) if top_p is not None else None
    if top_p is not None and not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    target_stars = int(args.target_stars if args.target_stars is not None else sampling_cfg.get('target_stars', 2))
    if target_stars < 0:
        raise ValueError(f"target_stars must be >= 0, got {target_stars}")
    variable_length = bool(sampling_cfg.get('variable_length', False))
    if args.variable_length:
        variable_length = True
    variable_length_min_tokens = int(
        args.min_length
        if args.min_length is not None
        else sampling_cfg.get('variable_length_min_tokens', 20)
    )
    variable_length_max_tokens = int(
        args.max_length
        if args.max_length is not None
        else sampling_cfg.get('variable_length_max_tokens', 100)
    )
    variable_length_samples_per_length = int(
        args.samples_per_length
        if args.samples_per_length is not None
        else sampling_cfg.get('variable_length_samples_per_length', 16)
    )
    if variable_length_min_tokens < 1:
        raise ValueError(f"variable_length_min_tokens must be >= 1, got {variable_length_min_tokens}")
    if variable_length_max_tokens < 1:
        raise ValueError(f"variable_length_max_tokens must be >= 1, got {variable_length_max_tokens}")
    if variable_length_max_tokens < variable_length_min_tokens:
        raise ValueError(
            "variable_length_max_tokens must be >= variable_length_min_tokens, "
            f"got {variable_length_max_tokens} < {variable_length_min_tokens}"
        )
    if variable_length_samples_per_length < 1:
        raise ValueError(
            "variable_length_samples_per_length must be >= 1, "
            f"got {variable_length_samples_per_length}"
        )
    if args.valid_only and args.no_valid_only:
        raise ValueError("Use only one of --valid_only or --no_valid_only")
    if args.valid_only_require_target_stars and args.valid_only_allow_non_target_stars:
        raise ValueError("Use only one of --valid_only_require_target_stars or --valid_only_allow_non_target_stars")
    if args.valid_only_fail_on_shortfall and args.valid_only_continue_on_shortfall:
        raise ValueError("Use only one of --valid_only_fail_on_shortfall or --valid_only_continue_on_shortfall")
    valid_only = bool(sampling_cfg.get('valid_only', False))
    if args.valid_only:
        valid_only = True
    elif args.no_valid_only:
        valid_only = False

    valid_only_require_target_stars = bool(sampling_cfg.get('valid_only_require_target_stars', True))
    if args.valid_only_require_target_stars:
        valid_only_require_target_stars = True
    elif args.valid_only_allow_non_target_stars:
        valid_only_require_target_stars = False

    valid_only_max_rounds = int(
        args.valid_only_max_rounds
        if args.valid_only_max_rounds is not None
        else sampling_cfg.get('valid_only_max_rounds', 20)
    )
    if valid_only_max_rounds < 1:
        raise ValueError(f"valid_only_max_rounds must be >= 1, got {valid_only_max_rounds}")

    valid_only_min_samples_per_round = int(
        args.valid_only_min_samples_per_round
        if args.valid_only_min_samples_per_round is not None
        else sampling_cfg.get('valid_only_min_samples_per_round', 256)
    )
    if valid_only_min_samples_per_round < 1:
        raise ValueError(
            f"valid_only_min_samples_per_round must be >= 1, got {valid_only_min_samples_per_round}"
        )
    valid_only_fail_on_shortfall = bool(sampling_cfg.get('valid_only_fail_on_shortfall', False))
    if args.valid_only_fail_on_shortfall:
        valid_only_fail_on_shortfall = True
    elif args.valid_only_continue_on_shortfall:
        valid_only_fail_on_shortfall = False
    valid_only_skip_novelty_filter = bool(getattr(args, "valid_only_skip_novelty_filter", False))
    valid_only_skip_sa_filter = bool(getattr(args, "valid_only_skip_sa_filter", False))
    target_polymer_count = int(
        args.target_polymer_count
        if args.target_polymer_count is not None
        else sampling_cfg.get("target_polymer_count", 100)
    )
    target_sa_max = float(sampling_cfg.get("target_sa_max", 4.0))
    if target_polymer_count < 1:
        raise ValueError(f"sampling.target_polymer_count must be >=1, got {target_polymer_count}")
    generation_goal = int(target_polymer_count)
    decode_constraint_class = (
        str(args.decode_constraint_class).strip().lower()
        if getattr(args, "decode_constraint_class", None) is not None
        else None
    )
    if decode_constraint_class == "":
        decode_constraint_class = None
    decode_constraint_motif_bank_json = (
        Path(args.decode_constraint_motif_bank_json)
        if getattr(args, "decode_constraint_motif_bank_json", None) is not None
        else None
    )
    decode_constraint_length_prior_json = (
        Path(args.decode_constraint_length_prior_json)
        if getattr(args, "decode_constraint_length_prior_json", None) is not None
        else None
    )
    decode_constraint_enabled = bool(decode_constraint_class is not None and decode_constraint_motif_bank_json is not None)
    if (decode_constraint_class is None) != (decode_constraint_motif_bank_json is None):
        raise ValueError(
            "decode constraint requires both --decode_constraint_class and --decode_constraint_motif_bank_json"
        )
    decode_constraint_spans_per_sample = int(
        args.decode_constraint_spans_per_sample
        if getattr(args, "decode_constraint_spans_per_sample", None) is not None
        else 1
    )
    decode_constraint_center_min_frac = float(
        args.decode_constraint_center_min_frac
        if getattr(args, "decode_constraint_center_min_frac", None) is not None
        else 0.25
    )
    decode_constraint_center_max_frac = float(
        args.decode_constraint_center_max_frac
        if getattr(args, "decode_constraint_center_max_frac", None) is not None
        else 0.75
    )
    decode_constraint_enforce_class_match = bool(
        decode_constraint_enabled and getattr(args, "decode_constraint_enforce_class_match", False)
    )
    decode_constraint_enforce_backbone_class_match = bool(
        decode_constraint_enforce_class_match
        and getattr(args, "decode_constraint_enforce_backbone_class_match", False)
    )
    if decode_constraint_enabled and not decode_constraint_motif_bank_json.exists():
        raise FileNotFoundError(
            f"decode constraint motif bank not found: {decode_constraint_motif_bank_json}"
        )
    if decode_constraint_length_prior_json is not None and not decode_constraint_length_prior_json.exists():
        raise FileNotFoundError(
            f"decode constraint length prior not found: {decode_constraint_length_prior_json}"
        )
    if decode_constraint_enabled and not (0.0 <= decode_constraint_center_min_frac <= decode_constraint_center_max_frac <= 1.0):
        raise ValueError(
            "decode constraint center fractions must satisfy 0 <= min <= max <= 1"
        )
    if decode_constraint_spans_per_sample < 1:
        raise ValueError(f"decode_constraint_spans_per_sample must be >= 1, got {decode_constraint_spans_per_sample}")

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Override results_dir if model_size specified
    base_results_dir = config['paths']['results_dir']
    results_dir = Path(get_results_dir(args.model_size, base_results_dir, split_mode))

    # Create output directories. Inverse-design steps can override this so each
    # run keeps its own fresh resampling artifacts instead of reusing Step 2 outputs.
    step_dir = Path(args.output_step_dir) if args.output_step_dir else (results_dir / 'step2_sampling')
    metrics_dir = step_dir / 'metrics'
    figures_dir = step_dir / 'figures'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    run_random_seed = int(args.random_seed) if args.random_seed is not None else int(config['data']['random_seed'])
    seed_info = seed_everything(run_random_seed)
    save_config(config, step_dir / 'config_used.yaml')
    save_run_metadata(step_dir, args.config, seed_info)
    write_initial_log(
        step_dir=step_dir,
        step_name="step2_sampling",
        context={
            "config_path": args.config,
            "model_size": args.model_size,
            "split_mode": split_mode,
            "results_dir": str(results_dir),
            "output_step_dir": str(step_dir),
            "target_polymer_goal": generation_goal,
            "target_polymer_count_config": target_polymer_count,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "target_stars": target_stars,
            "target_sa_max": target_sa_max,
            "variable_length": variable_length,
            "variable_length_min_tokens": variable_length_min_tokens,
            "variable_length_max_tokens": variable_length_max_tokens,
            "variable_length_samples_per_length": variable_length_samples_per_length,
            "valid_only": valid_only,
            "valid_only_require_target_stars": valid_only_require_target_stars,
            "valid_only_max_rounds": valid_only_max_rounds,
            "valid_only_min_samples_per_round": valid_only_min_samples_per_round,
            "valid_only_fail_on_shortfall": valid_only_fail_on_shortfall,
            "valid_only_skip_novelty_filter": bool(valid_only_skip_novelty_filter),
            "valid_only_skip_sa_filter": bool(valid_only_skip_sa_filter),
            "decode_constraint_enabled": bool(decode_constraint_enabled),
            "decode_constraint_class": decode_constraint_class,
            "decode_constraint_motif_bank_json": None if decode_constraint_motif_bank_json is None else str(decode_constraint_motif_bank_json),
            "decode_constraint_length_prior_json": None if decode_constraint_length_prior_json is None else str(decode_constraint_length_prior_json),
            "decode_constraint_spans_per_sample": int(decode_constraint_spans_per_sample),
            "decode_constraint_center_min_frac": float(decode_constraint_center_min_frac),
            "decode_constraint_center_max_frac": float(decode_constraint_center_max_frac),
            "decode_constraint_enforce_class_match": bool(decode_constraint_enforce_class_match),
            "random_seed": run_random_seed,
        },
    )

    print("=" * 50)
    print("Step 2: Sampling and Generative Evaluation")
    if args.model_size:
        print(f"Model Size: {args.model_size}")
    print("=" * 50)

    # Load tokenizer (from base results dir which has the tokenizer)
    print("\n1. Loading tokenizer...")
    tokenizer_path = results_dir / 'tokenizer.json'
    if not tokenizer_path.exists():
        tokenizer_path = Path(base_results_dir) / 'tokenizer.json'
    tokenizer = PSmilesTokenizer.load(tokenizer_path)
    if variable_length and variable_length_max_tokens > int(tokenizer.max_length):
        print(
            "   variable_length_max_tokens exceeds tokenizer.max_length; "
            f"clipping {variable_length_max_tokens} -> {tokenizer.max_length}"
        )
        variable_length_max_tokens = int(tokenizer.max_length)
    if variable_length and variable_length_min_tokens > int(tokenizer.max_length):
        raise ValueError(
            "variable_length_min_tokens exceeds tokenizer.max_length: "
            f"{variable_length_min_tokens} > {tokenizer.max_length}"
        )
    decode_constraint_fragments: List[str] = []
    decode_constraint_token_ids: List[List[int]] = []
    decode_constraint_length_prior_lengths: List[int] = []
    decode_constraint_classifier: PolymerClassifier | None = None
    if decode_constraint_enabled:
        decode_constraint_fragments = _load_decode_constraint_fragments(
            decode_constraint_motif_bank_json,
            decode_constraint_class,
        )
        decode_constraint_token_ids = _prepare_decode_constraint_token_ids(
            tokenizer=tokenizer,
            fragments=decode_constraint_fragments,
        )
        if not decode_constraint_token_ids:
            raise ValueError(
                f"decode constraint motif bank {decode_constraint_motif_bank_json} has no valid motifs for class '{decode_constraint_class}'"
            )
        if max(len(ids) for ids in decode_constraint_token_ids) + 4 > int(tokenizer.max_length):
            raise ValueError(
                "decode constraint motifs are too long for tokenizer.max_length="
                f"{tokenizer.max_length}"
            )
        if decode_constraint_length_prior_json is not None:
            decode_constraint_length_prior_lengths = _load_decode_constraint_length_prior(
                decode_constraint_length_prior_json,
                decode_constraint_class,
            )
            decode_constraint_length_prior_lengths = [
                max(max(len(ids) for ids in decode_constraint_token_ids) + 4, min(int(tokenizer.max_length), int(x)))
                for x in decode_constraint_length_prior_lengths
            ]
        if decode_constraint_enforce_class_match:
            patterns = {str(k).strip().lower(): v for k, v in config.get("polymer_classes", {}).items()}
            if decode_constraint_class not in patterns:
                raise ValueError(
                    f"decode constraint class '{decode_constraint_class}' not found in config polymer_classes"
                )
            decode_constraint_classifier = PolymerClassifier(patterns=patterns)
        print(
            "   decode constraint active: "
            f"class={decode_constraint_class}, motifs={len(decode_constraint_token_ids)}, "
            f"spans_per_sample={decode_constraint_spans_per_sample}, "
            f"center={decode_constraint_center_min_frac:.2f}-{decode_constraint_center_max_frac:.2f}, "
            f"enforce_class_match={decode_constraint_enforce_class_match}, "
            f"enforce_backbone={decode_constraint_enforce_backbone_class_match}"
        )
        if decode_constraint_length_prior_lengths:
            print(
                "   decode class length prior: "
                f"count={len(decode_constraint_length_prior_lengths)}, "
                f"min={min(decode_constraint_length_prior_lengths)}, "
                f"max={max(decode_constraint_length_prior_lengths)}"
            )
        _append_step_log(
            step_dir=step_dir,
            lines=[
                "decode_constraint:",
                f"enabled: {bool(decode_constraint_enabled)}",
                f"class: {decode_constraint_class}",
                f"motif_bank_json: {decode_constraint_motif_bank_json}",
                f"motif_count: {len(decode_constraint_token_ids)}",
                f"spans_per_sample: {int(decode_constraint_spans_per_sample)}",
                f"length_prior_json: {decode_constraint_length_prior_json}",
                f"length_prior_count: {len(decode_constraint_length_prior_lengths)}",
                f"center_min_frac: {decode_constraint_center_min_frac:.4f}",
                f"center_max_frac: {decode_constraint_center_max_frac:.4f}",
                f"enforce_class_match: {bool(decode_constraint_enforce_class_match)}",
            ],
        )

    # Load training data for novelty computation (from base results dir)
    print("\n2. Loading training data...")
    train_path = results_dir / 'train_unlabeled.csv'
    if not train_path.exists():
        train_path = Path(base_results_dir) / 'train_unlabeled.csv'
    train_df = pd.read_csv(train_path)
    training_smiles = set(train_df['p_smiles'].tolist())
    print(f"Training set size: {len(training_smiles)}")

    # Load model
    print("\n3. Loading model...")
    checkpoint_path = args.checkpoint or (results_dir / 'step1_backbone' / 'checkpoints' / 'backbone_best.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Get backbone config based on model_size
    backbone_config = get_model_config(args.model_size, config, model_type='sequence')
    backbone = DiffusionBackbone(
        vocab_size=tokenizer.vocab_size,
        hidden_size=backbone_config['hidden_size'],
        num_layers=backbone_config['num_layers'],
        num_heads=backbone_config['num_heads'],
        ffn_hidden_size=backbone_config['ffn_hidden_size'],
        max_position_embeddings=backbone_config['max_position_embeddings'],
        num_diffusion_steps=config['diffusion']['num_steps'],
        dropout=backbone_config['dropout'],
        pad_token_id=tokenizer.pad_token_id
    )

    model = DiscreteMaskingDiffusion(
        backbone=backbone,
        num_steps=config['diffusion']['num_steps'],
        beta_min=config['diffusion']['beta_min'],
        beta_max=config['diffusion']['beta_max'],
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # Handle torch.compile() state dict (keys have _orig_mod. prefix)
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Create sampler
    print("\n4. Creating sampler...")
    print(f"   temperature={temperature}, top_k={top_k}, top_p={top_p}, target_stars={target_stars}")
    sampler = ConstrainedSampler(
        diffusion_model=model,
        tokenizer=tokenizer,
        num_steps=config['diffusion']['num_steps'],
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        target_stars=target_stars,
        use_constraints=config['sampling'].get('use_constraints', True),
        device=device
    )

    # Apply class-enriched token logit bias if provided (Step 6 decode constraints)
    if getattr(args, "decode_constraint_class_token_bias_json", None):
        _bias_json_path = Path(args.decode_constraint_class_token_bias_json)
        if _bias_json_path.exists():
            with open(_bias_json_path, "r", encoding="utf-8") as f:
                _bias_data = json.load(f)
            _bias_values = _bias_data.get("bias") if isinstance(_bias_data, dict) else _bias_data
            if isinstance(_bias_values, list) and len(_bias_values) == tokenizer.vocab_size:
                sampler.set_class_token_logit_bias(_bias_values)
                print(f"   class token logit bias loaded: {_bias_json_path.name}")

    # Sample
    sampling_start = time.time()
    batch_size = args.batch_size or config['sampling']['batch_size']
    print(f"\n5. Sampling target polymers (goal={generation_goal}, batch_size={batch_size})...")

    def sample_candidates(n_requested: int, show_progress: bool = True, spans_per_sample_override: int | None = None):
        effective_spans_per_sample = (
            int(spans_per_sample_override)
            if spans_per_sample_override is not None
            else int(decode_constraint_spans_per_sample)
        )
        length_buffer = max((len(ids) for ids in decode_constraint_token_ids), default=0) + 4
        def _sample_decode_constraint_lengths(count: int) -> List[int]:
            if decode_constraint_length_prior_lengths:
                sampled_lengths = np.random.choice(
                    np.asarray(decode_constraint_length_prior_lengths, dtype=np.int32),
                    size=int(count),
                    replace=True,
                )
                return [int(x) for x in sampled_lengths.tolist()]
            return []

        if variable_length:
            min_len = variable_length_min_tokens
            max_len = variable_length_max_tokens
            if decode_constraint_enabled:
                min_len = max(min_len, length_buffer)
            if max_len < min_len:
                raise ValueError(
                    "decode constraint requires longer sequences than allowed by variable_length_max_tokens: "
                    f"min_required={min_len}, max_allowed={max_len}"
                )
            if show_progress:
                print(
                    "   Using variable length sampling "
                    f"(range: {min_len}-{max_len}, "
                    f"samples_per_length={variable_length_samples_per_length})"
                )
            if not decode_constraint_enabled:
                _, sampled_smiles = sampler.sample_variable_length(
                    n_requested,
                    length_range=(min_len, max_len),
                    batch_size=batch_size,
                    samples_per_length=variable_length_samples_per_length,
                    show_progress=show_progress
                )
                return sampled_smiles

            lengths = _sample_decode_constraint_lengths(int(n_requested))
            if not lengths:
                lengths = [
                    int(np.random.randint(min_len, max_len + 1))
                    for _ in range(int(n_requested))
                ]
            multi_spans, multi_span_starts, lengths = _build_decode_constraint_multi_spans(
                motif_token_ids=decode_constraint_token_ids,
                lengths=lengths,
                center_min_frac=decode_constraint_center_min_frac,
                center_max_frac=decode_constraint_center_max_frac,
                seq_length=int(tokenizer.max_length),
                spans_per_sample=effective_spans_per_sample,
            )
            if effective_spans_per_sample == 1:
                _, sampled_smiles = sampler.sample_batch_with_fixed_spans(
                    n_requested,
                    seq_length=int(tokenizer.max_length),
                    span_token_ids=[sample_spans[0] for sample_spans in multi_spans],
                    span_start_positions=[sample_starts[0] for sample_starts in multi_span_starts],
                    batch_size=batch_size,
                    show_progress=show_progress,
                    lengths=lengths,
                )
            else:
                _, sampled_smiles = sampler.sample_batch_with_multiple_fixed_spans(
                    n_requested,
                    seq_length=int(tokenizer.max_length),
                    span_token_ids=multi_spans,
                    span_start_positions=multi_span_starts,
                    batch_size=batch_size,
                    show_progress=show_progress,
                    lengths=lengths,
                )
            return sampled_smiles

        replace = n_requested > len(train_df)
        if decode_constraint_enabled and decode_constraint_length_prior_lengths:
            lengths = _sample_decode_constraint_lengths(int(n_requested))
        else:
            sampled = train_df['p_smiles'].sample(
                n=n_requested,
                replace=replace,
                random_state=np.random.randint(0, 2**31 - 1)
            )
            lengths = [
                min(len(tokenizer.tokenize(s)) + 2, tokenizer.max_length)
                for s in sampled.tolist()
            ]
        if decode_constraint_enabled:
            lengths = [max(int(length), length_buffer) for length in lengths]
        if show_progress:
            if decode_constraint_enabled and decode_constraint_length_prior_lengths:
                print(
                    "   Using class length prior "
                    f"(min={min(lengths)}, max={max(lengths)}, n_prior={len(decode_constraint_length_prior_lengths)})"
                )
            else:
                print(f"   Using training length distribution (min={min(lengths)}, max={max(lengths)})")
        if not decode_constraint_enabled:
            _, sampled_smiles = sampler.sample_batch(
                n_requested,
                seq_length=tokenizer.max_length,
                batch_size=batch_size,
                show_progress=show_progress,
                lengths=lengths
            )
            return sampled_smiles

        multi_spans, multi_span_starts, lengths = _build_decode_constraint_multi_spans(
            motif_token_ids=decode_constraint_token_ids,
            lengths=lengths,
            center_min_frac=decode_constraint_center_min_frac,
            center_max_frac=decode_constraint_center_max_frac,
            seq_length=int(tokenizer.max_length),
            spans_per_sample=effective_spans_per_sample,
        )
        if effective_spans_per_sample == 1:
            _, sampled_smiles = sampler.sample_batch_with_fixed_spans(
                n_requested,
                seq_length=tokenizer.max_length,
                span_token_ids=[sample_spans[0] for sample_spans in multi_spans],
                span_start_positions=[sample_starts[0] for sample_starts in multi_span_starts],
                batch_size=batch_size,
                show_progress=show_progress,
                lengths=lengths
            )
        else:
            _, sampled_smiles = sampler.sample_batch_with_multiple_fixed_spans(
                n_requested,
                seq_length=tokenizer.max_length,
                span_token_ids=multi_spans,
                span_start_positions=multi_span_starts,
                batch_size=batch_size,
                show_progress=show_progress,
                lengths=lengths
            )
        return sampled_smiles

    valid_only_stats = {
        "valid_only": bool(valid_only),
        "valid_only_rounds": 0,
        "valid_only_raw_generated": 0,
        "valid_only_acceptance_rate": None,
        "valid_only_rejected_count": 0,
        "valid_only_target_met": False,
        "valid_only_shortfall_count": int(generation_goal),
    }
    sa_cache: dict[str, float | None] = {}
    training_canonical = {canonicalize_smiles(s) or s for s in training_smiles}
    if valid_only:
        print(
            f"   Valid-only mode ON (require_target_stars={valid_only_require_target_stars}, "
            f"max_rounds={valid_only_max_rounds}, "
            f"skip_novelty={valid_only_skip_novelty_filter}, "
            f"skip_sa={valid_only_skip_sa_filter})"
        )
        generated_smiles = []
        seen_canonical = set()
        total_raw_generated = 0
        round_rows = []

        # Adaptive spans escalation for rare polymer classes:
        # After consecutive zero-acceptance rounds, increase spans_per_sample.
        adaptive_consecutive_zero = 0
        adaptive_escalation_after = 5  # escalate after this many consecutive 0% rounds
        adaptive_current_spans = int(decode_constraint_spans_per_sample)
        adaptive_spans_max = max(int(decode_constraint_spans_per_sample) + 3, 6)

        for round_idx in range(1, valid_only_max_rounds + 1):
            remaining = generation_goal - len(generated_smiles)
            if remaining <= 0:
                break

            n_requested = max(
                remaining,
                valid_only_min_samples_per_round,
            )
            print(
                f"   Valid-only round {round_idx}/{valid_only_max_rounds}: "
                f"request={n_requested}, remaining_target={remaining}"
            )
            spans_override = adaptive_current_spans if decode_constraint_enabled else None
            batch_smiles = sample_candidates(
                n_requested=n_requested,
                show_progress=True,
                spans_per_sample_override=spans_override,
            )
            total_raw_generated += len(batch_smiles)

            batch_accepted = _filter_step2_target_candidates(
                smiles_list=batch_smiles,
                training_canonical=training_canonical,
                seen_canonical=seen_canonical,
                require_target_stars=valid_only_require_target_stars,
                target_stars=target_stars,
                sa_max=target_sa_max,
                sa_cache=sa_cache,
                required_polymer_class=decode_constraint_class if decode_constraint_enforce_class_match else None,
                polymer_classifier=decode_constraint_classifier,
                enforce_novelty=not valid_only_skip_novelty_filter,
                enforce_unique=True,
                enforce_sa=not valid_only_skip_sa_filter,
                enforce_backbone_class_match=decode_constraint_enforce_backbone_class_match,
            )
            accepted = min(remaining, len(batch_accepted))
            if accepted > 0:
                generated_smiles.extend([row["smiles"] for row in batch_accepted[:accepted]])

            round_acceptance = (len(batch_accepted) / len(batch_smiles)) if len(batch_smiles) > 0 else 0.0
            round_rows.append(
                {
                    "round": round_idx,
                    "remaining_target_before_round": remaining,
                    "requested_samples": int(len(batch_smiles)),
                    "strict_candidates_found": int(len(batch_accepted)),
                    "accepted_this_round": int(accepted),
                    "accepted_cumulative": int(len(generated_smiles)),
                    "round_acceptance_rate": float(round_acceptance),
                    "adaptive_spans_per_sample": adaptive_current_spans if decode_constraint_enabled else None,
                }
            )
            print(
                f"     strict={len(batch_accepted)}/{len(batch_smiles)} "
                f"({100.0 * round_acceptance:.2f}%), accepted={accepted}, "
                f"cumulative={len(generated_smiles)}/{generation_goal}"
            )

            # Adaptive spans escalation: increase motif density on persistent failure
            if decode_constraint_enabled and round_acceptance == 0.0:
                adaptive_consecutive_zero += 1
                if (
                    adaptive_consecutive_zero >= adaptive_escalation_after
                    and adaptive_current_spans < adaptive_spans_max
                ):
                    adaptive_current_spans += 1
                    adaptive_consecutive_zero = 0
                    print(
                        f"   Adaptive escalation: spans_per_sample increased to "
                        f"{adaptive_current_spans} (max={adaptive_spans_max})"
                    )
            else:
                adaptive_consecutive_zero = 0

        shortfall_count = int(max(generation_goal - len(generated_smiles), 0))
        if shortfall_count > 0:
            shortfall_msg = (
                "Valid-only sampling did not reach requested samples. "
                f"accepted={len(generated_smiles)}, requested={generation_goal}, "
                f"shortfall={shortfall_count}, max_rounds={valid_only_max_rounds}."
            )
            if valid_only_fail_on_shortfall:
                raise RuntimeError(f"{shortfall_msg} Increase valid_only_max_rounds or loosen filters.")
            print(f"WARNING: {shortfall_msg} Continuing with accepted subset.")

        generated_smiles = generated_smiles[:generation_goal]
        rounds_df = pd.DataFrame(round_rows)
        rounds_df.to_csv(metrics_dir / 'valid_only_sampling_rounds.csv', index=False)
        print(f"Saved valid-only round log: {metrics_dir / 'valid_only_sampling_rounds.csv'}")

        acceptance_rate = (len(generated_smiles) / total_raw_generated) if total_raw_generated > 0 else 0.0
        valid_only_stats.update(
            {
                "valid_only_rounds": int(len(round_rows)),
                "valid_only_raw_generated": int(total_raw_generated),
                "valid_only_acceptance_rate": float(acceptance_rate),
                "valid_only_rejected_count": int(max(total_raw_generated - len(generated_smiles), 0)),
                "valid_only_target_met": bool(len(generated_smiles) >= generation_goal),
                "valid_only_shortfall_count": int(max(generation_goal - len(generated_smiles), 0)),
            }
        )
        total_sampling_points = int(total_raw_generated)
    else:
        raw_smiles = sample_candidates(n_requested=generation_goal, show_progress=True)
        filtered = _filter_valid_samples(
            smiles_list=raw_smiles,
            require_target_stars=valid_only_require_target_stars,
            target_stars=target_stars,
            required_polymer_class=decode_constraint_class if decode_constraint_enforce_class_match else None,
            polymer_classifier=decode_constraint_classifier,
            enforce_backbone_class_match=decode_constraint_enforce_backbone_class_match,
        )
        generated_smiles = filtered[:generation_goal]
        valid_only_stats.update(
            {
                "valid_only_rounds": 1,
                "valid_only_raw_generated": int(len(raw_smiles)),
                "valid_only_acceptance_rate": float(len(generated_smiles) / len(raw_smiles)) if raw_smiles else 0.0,
                "valid_only_rejected_count": int(max(len(raw_smiles) - len(generated_smiles), 0)),
                "valid_only_target_met": bool(len(generated_smiles) >= generation_goal),
                "valid_only_shortfall_count": int(max(generation_goal - len(generated_smiles), 0)),
            }
        )
        total_sampling_points = int(len(raw_smiles))

    sampling_time_sec = time.time() - sampling_start

    # Save generated samples
    samples_df = pd.DataFrame({'smiles': generated_smiles})
    samples_df.to_csv(metrics_dir / 'generated_samples.csv', index=False)
    print(f"Saved {len(generated_smiles)} generated samples")
    if valid_only_stats["valid_only"]:
        print(
            "Valid-only acceptance: "
            f"{valid_only_stats['valid_only_raw_generated']} -> {len(generated_smiles)} "
            f"(rate={100.0 * float(valid_only_stats['valid_only_acceptance_rate']):.2f}%)"
        )

    # Evaluate
    print("\n6. Evaluating generative metrics...")
    method_name = "Bi_Diffusion"
    representation_name = "SMILES"
    model_size_label = args.model_size or "small"
    evaluator = GenerativeEvaluator(training_smiles)
    metrics = evaluator.evaluate(
        generated_smiles,
        sample_id=f'uncond_{generation_goal}_best_checkpoint',
        show_progress=True,
        sampling_time_sec=sampling_time_sec,
        method=method_name,
        representation=representation_name,
        model_size=model_size_label
    )

    # Save metrics
    metrics_csv = evaluator.format_metrics_csv(metrics)
    metrics_csv.to_csv(metrics_dir / 'sampling_generative_metrics.csv', index=False)

    constraint_rows = compute_smiles_constraint_metrics(generated_smiles, method_name, representation_name, model_size_label)
    pd.DataFrame(constraint_rows).to_csv(metrics_dir / 'constraint_metrics.csv', index=False)

    if args.evaluate_ood:
        foundation_dir = Path(args.foundation_results_dir)
        d1_path = foundation_dir / "embeddings_d1.npy"
        d2_path = foundation_dir / "embeddings_d2.npy"
        gen_path = Path(args.generated_embeddings_path) if args.generated_embeddings_path else None
        if d1_path.exists() and d2_path.exists():
            try:
                from shared.ood_metrics import compute_ood_metrics_from_files
                ood_metrics = compute_ood_metrics_from_files(d1_path, d2_path, gen_path, k=args.ood_k)
                ood_row = {
                    "method": method_name,
                    "representation": representation_name,
                    "model_size": model_size_label,
                    **ood_metrics
                }
                pd.DataFrame([ood_row]).to_csv(metrics_dir / "metrics_ood.csv", index=False)
            except Exception as exc:
                print(f"OOD evaluation failed: {exc}")
        else:
            print("OOD embeddings not found; skipping OOD evaluation.")

    # Print metrics
    print("\nGenerative Metrics:")
    print(f"  Validity: {metrics['validity']:.4f}")
    print(f"  Validity (star=2): {metrics['validity_two_stars']:.4f}")
    print(f"  Uniqueness: {metrics['uniqueness']:.4f}")
    print(f"  Novelty: {metrics['novelty']:.4f}")
    print(f"  Diversity: {metrics['avg_diversity']:.4f}")
    print(f"  Frac star=2: {metrics['frac_star_eq_2']:.4f}")
    print(f"  Mean SA: {metrics['mean_sa']:.4f}")
    print(f"  Std SA: {metrics['std_sa']:.4f}")

    # Final target polymer selection for downstream inverse design
    target_df, target_summary = _select_target_polymers(
        generated_smiles=generated_smiles,
        training_smiles=training_smiles,
        target_count=generation_goal,
        target_stars=target_stars,
        sa_max=target_sa_max,
        total_sampling_points=total_sampling_points,
        sa_cache=sa_cache,
    )
    target_df.to_csv(metrics_dir / "target_polymers.csv", index=False)
    pd.DataFrame([target_summary]).to_csv(metrics_dir / "target_polymer_selection_summary.csv", index=False)

    if target_summary["target_count_selected"] < target_summary["target_count_requested"]:
        print(
            "Warning: selected target polymers are fewer than requested. "
            f"selected={target_summary['target_count_selected']}, "
            f"requested={target_summary['target_count_requested']}"
        )

    print("\nTarget polymer selection (valid + star=2 + novel + SA<4 + unique):")
    print(f"  Requested: {target_summary['target_count_requested']}")
    print(f"  Selected: {target_summary['target_count_selected']}")
    print(f"  Success rate: {target_summary['selection_success_rate']:.4f}")
    print(f"  Diversity: {target_summary['final_diversity']:.4f}")
    print(f"  Mean SA: {target_summary['final_mean_sa']:.4f}")

    # Create plots
    print("\n7. Creating plots...")
    plotter = PlotUtils(
        figure_size=tuple(config['plotting']['figure_size']),
        font_size=config['plotting']['font_size'],
        dpi=config['plotting']['dpi']
    )

    # Get valid samples
    valid_smiles = evaluator.get_valid_samples(generated_smiles, require_two_stars=True)

    # SA histogram: train vs generated
    train_sa = [_sa_score_cached(s, sa_cache) for s in list(training_smiles)[:5000]]
    train_sa = [s for s in train_sa if s is not None]
    gen_sa = [_sa_score_cached(s, sa_cache) for s in valid_smiles[:5000]]
    gen_sa = [s for s in gen_sa if s is not None]

    plotter.histogram(
        data=[train_sa, gen_sa],
        labels=['Train', 'Generated'],
        xlabel='SA Score',
        ylabel='Count',
        title='SA Score: Train vs Generated',
        save_path=figures_dir / 'sa_hist_train_vs_uncond.png',
        bins=50,
        style='step'
    )

    # Length histogram: train vs generated
    train_lengths = [len(s) for s in list(training_smiles)[:5000]]
    gen_lengths = [len(s) for s in valid_smiles[:5000]]

    plotter.histogram(
        data=[train_lengths, gen_lengths],
        labels=['Train', 'Generated'],
        xlabel='SMILES Length',
        ylabel='Count',
        title='Length: Train vs Generated',
        save_path=figures_dir / 'length_hist_train_vs_uncond.png',
        bins=50,
        style='step'
    )

    # Star count histogram
    star_counts = [count_stars(s) for s in valid_smiles]

    plotter.star_count_bar(
        star_counts=star_counts,
        expected_count=2,
        xlabel='Star Count',
        ylabel='Count',
        title='Star Count Distribution',
        save_path=figures_dir / 'star_count_hist_uncond.png'
    )

    # Save standardized summary and artifact manifest.
    summary = {
        "step": "step2_sampling",
        "model_size": model_size_label,
        "generation_goal": int(generation_goal),
        "generated_count": int(total_sampling_points),
        "accepted_count_for_evaluation": int(len(generated_smiles)),
        "variable_length": bool(variable_length),
        "variable_length_min_tokens": int(variable_length_min_tokens),
        "variable_length_max_tokens": int(variable_length_max_tokens),
        "variable_length_samples_per_length": int(variable_length_samples_per_length),
        "temperature": float(temperature),
        "top_k": int(top_k) if top_k is not None else None,
        "top_p": float(top_p) if top_p is not None else None,
        "target_stars": int(target_stars),
        "valid_only": bool(valid_only_stats["valid_only"]),
        "valid_only_rounds": int(valid_only_stats["valid_only_rounds"]),
        "valid_only_raw_generated": int(valid_only_stats["valid_only_raw_generated"]),
        "valid_only_acceptance_rate": float(valid_only_stats["valid_only_acceptance_rate"]),
        "valid_only_rejected_count": int(valid_only_stats["valid_only_rejected_count"]),
        "valid_only_target_met": bool(valid_only_stats["valid_only_target_met"]),
        "valid_only_shortfall_count": int(valid_only_stats["valid_only_shortfall_count"]),
        "valid_only_skip_novelty_filter": bool(valid_only_skip_novelty_filter),
        "valid_only_skip_sa_filter": bool(valid_only_skip_sa_filter),
        "decode_constraint_enabled": bool(decode_constraint_enabled),
        "decode_constraint_class": decode_constraint_class,
        "decode_constraint_motif_count": int(len(decode_constraint_token_ids)),
        "decode_constraint_spans_per_sample": int(decode_constraint_spans_per_sample),
        "decode_constraint_length_prior_count": int(len(decode_constraint_length_prior_lengths)),
        "decode_constraint_center_min_frac": float(decode_constraint_center_min_frac),
        "decode_constraint_center_max_frac": float(decode_constraint_center_max_frac),
        "decode_constraint_enforce_class_match": bool(decode_constraint_enforce_class_match),
        "sampling_time_sec": float(sampling_time_sec),
        "samples_per_sec": float(metrics.get("samples_per_sec", 0.0)),
        "validity": float(metrics.get("validity", 0.0)),
        "validity_two_stars": float(metrics.get("validity_two_stars", 0.0)),
        "uniqueness": float(metrics.get("uniqueness", 0.0)),
        "novelty": float(metrics.get("novelty", 0.0)),
        "avg_diversity": float(metrics.get("avg_diversity", 0.0)),
        "frac_star_eq_2": float(metrics.get("frac_star_eq_2", 0.0)),
        "mean_sa": float(metrics.get("mean_sa", 0.0)),
        "std_sa": float(metrics.get("std_sa", 0.0)),
        "target_polymer_count_requested": int(target_summary["target_count_requested"]),
        "target_polymer_count_selected": int(target_summary["target_count_selected"]),
        "target_polymer_selection_success_rate": float(target_summary["selection_success_rate"]),
        "target_polymer_diversity": float(target_summary["final_diversity"]),
        "target_polymer_mean_sa": float(target_summary["final_mean_sa"]),
        "target_polymer_std_sa": float(target_summary["final_std_sa"]),
        "target_polymer_novelty": float(target_summary["final_novelty"]),
        "target_polymer_uniqueness": float(target_summary["final_uniqueness"]),
        "target_polymer_frac_star_eq_target": float(target_summary["final_frac_star_eq_target"]),
    }
    save_step_summary(summary, metrics_dir)
    save_artifact_manifest(step_dir=step_dir, metrics_dir=metrics_dir, figures_dir=figures_dir)
    _append_step_log(
        step_dir=step_dir,
        lines=[
            "final_target_polymer_selection:",
            f"total_sampling_examples: {target_summary['total_generated']}",
            f"target_count_requested: {target_summary['target_count_requested']}",
            f"target_count_selected: {target_summary['target_count_selected']}",
            f"success_rate: {target_summary['selection_success_rate']:.6f}",
            f"diversity: {target_summary['final_diversity']:.6f}",
            f"mean_sa: {target_summary['final_mean_sa']:.6f}",
            f"std_sa: {target_summary['final_std_sa']:.6f}",
            f"novelty: {target_summary['final_novelty']:.6f}",
            f"uniqueness: {target_summary['final_uniqueness']:.6f}",
            f"frac_star_eq_{target_stars}: {target_summary['final_frac_star_eq_target']:.6f}",
            "target_csv: metrics/target_polymers.csv",
        ],
    )

    print("\n" + "=" * 50)
    print("Sampling and evaluation complete!")
    print(f"Results saved to: {metrics_dir}")
    print(f"Figures saved to: {figures_dir}")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample and evaluate generative model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--model_size', type=str, default='small',
                        choices=['small', 'medium', 'large', 'xl'],
                        help='Model size preset (small: ~12M, medium: ~50M, large: ~150M, xl: ~400M)')
    parser.add_argument('--split_mode', type=str, default=None, choices=['polymer', 'random'],
                        help='Optional split-mode namespace for results dir (default: config chi_training.shared.split_mode)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--output_step_dir', type=str, default=None,
                        help='Optional output directory override for this sampling run')
    parser.add_argument('--random_seed', type=int, default=None,
                        help='Optional random seed override for this sampling run')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for sampling (default: from config)')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Sampling temperature (default: sampling.temperature in config)')
    parser.add_argument('--top_k', type=int, default=None,
                        help='Top-k token filter (default: sampling.top_k in config)')
    parser.add_argument('--top_p', type=float, default=None,
                        help='Top-p nucleus filter (default: sampling.top_p in config)')
    parser.add_argument('--target_stars', type=int, default=None,
                        help='Target number of "*" tokens (default: sampling.target_stars in config)')
    parser.add_argument('--target_polymer_count', type=int, default=None,
                        help='Target number of accepted polymers to generate for this sampling run (default: sampling.target_polymer_count in config)')
    parser.add_argument('--valid_only', action='store_true',
                        help='Enable valid-only resampling mode')
    parser.add_argument('--no_valid_only', action='store_true',
                        help='Disable valid-only resampling mode')
    parser.add_argument('--valid_only_require_target_stars', action='store_true',
                        help='In valid-only mode, require star count == target_stars')
    parser.add_argument('--valid_only_allow_non_target_stars', action='store_true',
                        help='In valid-only mode, do not enforce target star count')
    parser.add_argument('--valid_only_max_rounds', type=int, default=None,
                        help='Maximum rounds in valid-only mode (default: sampling.valid_only_max_rounds)')
    parser.add_argument('--valid_only_min_samples_per_round', type=int, default=None,
                        help='Minimum requested samples per valid-only round (default: sampling.valid_only_min_samples_per_round)')
    parser.add_argument('--valid_only_fail_on_shortfall', action='store_true',
                        help='Raise an error if valid-only rounds do not reach target count')
    parser.add_argument('--valid_only_continue_on_shortfall', action='store_true',
                        help='Continue with accepted subset if valid-only rounds do not reach target count')
    parser.add_argument('--valid_only_skip_novelty_filter', action='store_true',
                        help='In valid-only mode, do not require novelty during acceptance; defer novelty filtering downstream')
    parser.add_argument('--valid_only_skip_sa_filter', action='store_true',
                        help='In valid-only mode, do not require SA < target_sa_max during acceptance; defer SA filtering downstream')
    parser.add_argument('--variable_length', action='store_true',
                        help='Enable variable length sampling (or set sampling.variable_length=true in config)')
    parser.add_argument('--min_length', type=int, default=None,
                        help='Minimum sequence length for variable length sampling (default: sampling.variable_length_min_tokens)')
    parser.add_argument('--max_length', type=int, default=None,
                        help='Maximum sequence length for variable length sampling (default: sampling.variable_length_max_tokens)')
    parser.add_argument('--samples_per_length', type=int, default=None,
                        help='Samples per length in variable length mode (default: sampling.variable_length_samples_per_length)')
    parser.add_argument('--decode_constraint_class', type=str, default=None,
                        help='Optional decode-time polymer class constraint (used by Step 6 only)')
    parser.add_argument('--decode_constraint_motif_bank_json', type=str, default=None,
                        help='JSON file containing decode-time motif fragments for the target class')
    parser.add_argument('--decode_constraint_length_prior_json', type=str, default=None,
                        help='Optional JSON file containing class-specific token lengths for decode-time sampling')
    parser.add_argument('--decode_constraint_spans_per_sample', type=int, default=None,
                        help='Number of fixed decode-time motifs to inject per sampled polymer')
    parser.add_argument('--decode_constraint_center_min_frac', type=float, default=None,
                        help='Lower bound for motif center placement as a fraction of sequence length')
    parser.add_argument('--decode_constraint_center_max_frac', type=float, default=None,
                        help='Upper bound for motif center placement as a fraction of sequence length')
    parser.add_argument('--decode_constraint_enforce_class_match', action='store_true',
                        help='Filter accepted samples to exact target polymer-class SMARTS matches')
    parser.add_argument('--decode_constraint_enforce_backbone_class_match', action='store_true',
                        help='Require class-defining functional group to be on the backbone path between * atoms')
    parser.add_argument('--decode_constraint_class_token_bias_json', type=str, default=None,
                        help='Path to JSON with per-token logit bias for class-steered sampling')
    parser.add_argument("--evaluate_ood", action="store_true",
                        help="Compute OOD metrics if embeddings are available")
    parser.add_argument("--foundation_results_dir", type=str,
                        default="../Multi_View_Foundation/results",
                        help="Path to Multi_View_Foundation results directory")
    parser.add_argument("--generated_embeddings_path", type=str, default=None,
                        help="Optional path to generated embeddings (.npy)")
    parser.add_argument("--ood_k", type=int, default=1,
                        help="k for nearest-neighbor distance in OOD metrics")

    args = parser.parse_args()
    main(args)
