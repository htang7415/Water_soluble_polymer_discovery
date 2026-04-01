"""Frozen-model sampling helpers for Step 6_2 S0/S1."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.chi.embeddings import load_backbone_from_step1
from src.evaluation.class_decode_constraints import (
    compute_class_token_logit_bias,
    load_decode_constraint_source_smiles,
    resolve_class_decode_length_prior,
    resolve_class_decode_motifs,
)
from src.model.diffusion import DiscreteMaskingDiffusion
from src.sampling.sampler import ConstrainedSampler
from src.step6_2.config import ResolvedStep62Config
from src.data.tokenizer import PSmilesTokenizer


@dataclass
class ResolvedClassSamplingPrior:
    """Resolved class-aware sampling priors for frozen-model runs."""

    target_class: str
    source_smiles: List[str]
    motifs: List[str]
    motif_source: str
    motif_token_ids: List[List[int]]
    spans_per_sample: int
    center_min_frac: float
    center_max_frac: float
    length_prior_lengths: List[int]
    length_prior_source: Optional[str]
    fallback_source_lengths: List[int]
    class_token_logit_bias: Optional[List[float]]
    class_token_bias_strength: float


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
        raise ValueError("decode constraint center fractions must satisfy 0 <= min <= max <= 1")

    max_motif_len = max(len(ids) for ids in motif_token_ids)
    adjusted_lengths = [
        min(seq_length, max(int(raw_length), max_motif_len + 4))
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
        center_frac = (
            center_min_frac
            if center_min_frac == center_max_frac
            else float(np.random.uniform(center_min_frac, center_max_frac))
        )
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
        max_start = int(effective_length) - 1 - len(motif_ids) - future_requirements[idx]
        candidate_start = int(
            round(anchor_frac * float(max(1, effective_length - 1)) - (0.5 * len(motif_ids)))
        )
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
        min(seq_length, max(int(raw_length), max_motif_len + 4))
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


def load_step1_diffusion(
    resolved: ResolvedStep62Config,
    *,
    device: str,
) -> tuple[PSmilesTokenizer, DiscreteMaskingDiffusion, Path]:
    """Load tokenizer + Step 1 diffusion model for S0/S1."""

    try:
        tokenizer, backbone, checkpoint_path = load_backbone_from_step1(
            config=resolved.base_config,
            model_size=resolved.model_size,
            split_mode=resolved.split_mode,
            checkpoint_path=None,
            device=device,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Step 6_2 frozen sampling requires an existing Step 1 backbone checkpoint and tokenizer. "
            f"model_size={resolved.model_size!r}, split_mode={resolved.split_mode!r}. "
            f"Original error: {exc}"
        ) from exc
    diffusion = DiscreteMaskingDiffusion(
        backbone=backbone,
        num_steps=resolved.base_config["diffusion"]["num_steps"],
        beta_min=resolved.base_config["diffusion"]["beta_min"],
        beta_max=resolved.base_config["diffusion"]["beta_max"],
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    ).to(device)
    diffusion.eval()
    return tokenizer, diffusion, checkpoint_path


def _token_lengths_from_smiles(
    tokenizer: PSmilesTokenizer,
    smiles_list: Iterable[str],
) -> List[int]:
    lengths: List[int] = []
    for smiles in smiles_list:
        token_len = len(tokenizer.tokenize(str(smiles))) + 2
        token_len = max(2, min(int(tokenizer.max_length), int(token_len)))
        lengths.append(token_len)
    return lengths


def resolve_class_sampling_prior(
    resolved: ResolvedStep62Config,
    run_cfg: Dict[str, object],
    tokenizer: PSmilesTokenizer,
    *,
    metrics_dir: Optional[Path] = None,
) -> ResolvedClassSamplingPrior:
    """Resolve Step 6-style class-aware sampling priors for S0/S1."""

    step6_cfg = (
        resolved.base_config.get("chi_training", {}).get("step6_class_inverse_design", {})
        if isinstance(resolved.base_config.get("chi_training", {}).get("step6_class_inverse_design", {}), dict)
        else {}
    )
    target_class = resolved.c_target
    source_smiles = load_decode_constraint_source_smiles(Path(resolved.base_config["paths"]["data_dir"]))
    resolution_strategy = str(
        step6_cfg.get("decode_constraint_resolution_strategy", "configured_or_defaults")
    ).strip().lower()
    configured_bank_path_raw = step6_cfg.get("decode_constraint_motif_bank_json")
    configured_bank_path = (
        Path(configured_bank_path_raw).resolve()
        if configured_bank_path_raw not in {None, "", "null"}
        else None
    )
    resolve_source_smiles = (
        source_smiles if resolution_strategy == "configured_or_local_mined_or_defaults" else []
    )
    motifs, motif_source = resolve_class_decode_motifs(
        target_class=target_class,
        tokenizer=tokenizer,
        source_smiles=resolve_source_smiles,
        patterns=resolved.polymer_patterns,
        configured_bank_path=configured_bank_path,
        max_motifs=int(step6_cfg.get("decode_constraint_max_motifs", 6)),
        resolution_strategy=resolution_strategy,
    )
    motif_token_ids = _prepare_decode_constraint_token_ids(tokenizer, motifs)
    use_class_length_prior = bool(step6_cfg.get("decode_constraint_use_class_length_prior", True))
    length_prior_lengths: List[int] = []
    length_prior_source: Optional[str] = None
    if use_class_length_prior:
        length_prior_lengths, length_prior_source = resolve_class_decode_length_prior(
            target_class=target_class,
            tokenizer=tokenizer,
            source_smiles=source_smiles,
            patterns=resolved.polymer_patterns,
            max_length=int(tokenizer.max_length),
        )
    fallback_source_lengths = _token_lengths_from_smiles(tokenizer, source_smiles)
    class_token_bias_strength = float(run_cfg.get("class_token_bias_strength", 1.5))
    class_token_logit_bias = compute_class_token_logit_bias(
        target_class=target_class,
        tokenizer=tokenizer,
        source_smiles=source_smiles,
        patterns=resolved.polymer_patterns,
        bias_strength=class_token_bias_strength,
    )

    prior = ResolvedClassSamplingPrior(
        target_class=target_class,
        source_smiles=source_smiles,
        motifs=motifs,
        motif_source=motif_source,
        motif_token_ids=motif_token_ids,
        spans_per_sample=int(step6_cfg.get("decode_constraint_spans_per_sample", 2)),
        center_min_frac=float(step6_cfg.get("decode_constraint_center_min_frac", 0.25)),
        center_max_frac=float(step6_cfg.get("decode_constraint_center_max_frac", 0.75)),
        length_prior_lengths=length_prior_lengths,
        length_prior_source=length_prior_source,
        fallback_source_lengths=fallback_source_lengths,
        class_token_logit_bias=class_token_logit_bias,
        class_token_bias_strength=class_token_bias_strength,
    )

    if metrics_dir is not None:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        with open(metrics_dir / "decode_constraint_motif_bank_resolved.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "target_class": target_class,
                    "source": motif_source,
                    "motifs": motifs,
                },
                handle,
                indent=2,
            )
        if length_prior_lengths:
            with open(metrics_dir / "decode_constraint_length_prior_resolved.json", "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "target_class": target_class,
                        "source": length_prior_source,
                        "lengths": length_prior_lengths,
                    },
                    handle,
                    indent=2,
                )
        if class_token_logit_bias is not None:
            with open(metrics_dir / "decode_constraint_class_token_bias.json", "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "target_class": target_class,
                        "bias_strength": class_token_bias_strength,
                        "bias": class_token_logit_bias,
                    },
                    handle,
                )

    return prior


def _sample_lengths(
    *,
    prior: ResolvedClassSamplingPrior,
    tokenizer: PSmilesTokenizer,
    num_samples: int,
    sampling_cfg: Dict[str, object],
) -> List[int]:
    if prior.length_prior_lengths:
        sampled = np.random.choice(
            np.asarray(prior.length_prior_lengths, dtype=np.int32),
            size=int(num_samples),
            replace=True,
        )
        lengths = [int(x) for x in sampled.tolist()]
    elif bool(sampling_cfg.get("variable_length", False)):
        min_len = int(sampling_cfg.get("variable_length_min_tokens", 12))
        max_len = int(min(int(tokenizer.max_length), sampling_cfg.get("variable_length_max_tokens", 100)))
        lengths = [int(np.random.randint(min_len, max_len + 1)) for _ in range(int(num_samples))]
    elif prior.fallback_source_lengths:
        sampled = np.random.choice(
            np.asarray(prior.fallback_source_lengths, dtype=np.int32),
            size=int(num_samples),
            replace=True,
        )
        lengths = [int(x) for x in sampled.tolist()]
    else:
        lengths = [int(tokenizer.max_length)] * int(num_samples)

    motif_buffer = max((len(ids) for ids in prior.motif_token_ids), default=0) + 4
    if motif_buffer > 0:
        lengths = [max(int(length), int(motif_buffer)) for length in lengths]
    return [min(int(tokenizer.max_length), int(length)) for length in lengths]


def create_constrained_sampler(
    *,
    diffusion_model: DiscreteMaskingDiffusion,
    tokenizer: PSmilesTokenizer,
    resolved: ResolvedStep62Config,
    prior: ResolvedClassSamplingPrior,
    device: str,
) -> ConstrainedSampler:
    """Create a Step 6_2 sampler with shared class-prior settings."""

    sampling_cfg = resolved.base_config.get("sampling", {})
    sampler = ConstrainedSampler(
        diffusion_model=diffusion_model,
        tokenizer=tokenizer,
        num_steps=resolved.base_config["diffusion"]["num_steps"],
        temperature=float(resolved.step6_2["sampling_temperature"]),
        top_k=sampling_cfg.get("top_k"),
        top_p=sampling_cfg.get("top_p"),
        target_stars=int(sampling_cfg.get("target_stars", 2)),
        use_constraints=bool(sampling_cfg.get("use_constraints", True)),
        device=device,
    )
    sampler.set_class_token_bias_start_frac(float(resolved.step6_2.get("class_token_bias_start_frac", 0.0)))
    if prior.class_token_logit_bias is not None:
        sampler.set_class_token_logit_bias(prior.class_token_logit_bias)
    return sampler


def sample_with_class_prior(
    *,
    sampler: ConstrainedSampler,
    tokenizer: PSmilesTokenizer,
    prior: ResolvedClassSamplingPrior,
    resolved: ResolvedStep62Config,
    num_samples: int,
    show_progress: bool = True,
) -> Tuple[List[str], Dict[str, object]]:
    """Sample polymers with Step 6 class priors using a sampler-compatible backend."""

    sampling_cfg = resolved.base_config.get("sampling", {})
    lengths = _sample_lengths(
        prior=prior,
        tokenizer=tokenizer,
        num_samples=num_samples,
        sampling_cfg=sampling_cfg,
    )

    if prior.motif_token_ids:
        multi_spans, multi_span_starts, lengths = _build_decode_constraint_multi_spans(
            motif_token_ids=prior.motif_token_ids,
            lengths=lengths,
            center_min_frac=prior.center_min_frac,
            center_max_frac=prior.center_max_frac,
            seq_length=int(tokenizer.max_length),
            spans_per_sample=prior.spans_per_sample,
        )
        if prior.spans_per_sample == 1:
            _, smiles = sampler.sample_batch_with_fixed_spans(
                num_samples=num_samples,
                seq_length=int(tokenizer.max_length),
                span_token_ids=[sample_spans[0] for sample_spans in multi_spans],
                span_start_positions=[sample_starts[0] for sample_starts in multi_span_starts],
                batch_size=int(sampling_cfg.get("batch_size", 128)),
                show_progress=show_progress,
                lengths=lengths,
            )
        else:
            _, smiles = sampler.sample_batch_with_multiple_fixed_spans(
                num_samples=num_samples,
                seq_length=int(tokenizer.max_length),
                span_token_ids=multi_spans,
                span_start_positions=multi_span_starts,
                batch_size=int(sampling_cfg.get("batch_size", 128)),
                show_progress=show_progress,
                lengths=lengths,
            )
    else:
        _, smiles = sampler.sample_batch(
            num_samples=num_samples,
            seq_length=int(tokenizer.max_length),
            batch_size=int(sampling_cfg.get("batch_size", 128)),
            show_progress=show_progress,
            lengths=lengths,
        )

    metadata = {
        "num_samples": int(num_samples),
        "spans_per_sample": int(prior.spans_per_sample),
        "motif_count": int(len(prior.motifs)),
        "motif_source": prior.motif_source,
        "length_prior_count": int(len(prior.length_prior_lengths)),
        "length_prior_source": prior.length_prior_source,
        "class_token_bias_enabled": bool(prior.class_token_logit_bias is not None),
        "class_token_bias_strength": float(prior.class_token_bias_strength),
    }
    return smiles, metadata


def sample_unconditional_with_class_prior(
    *,
    sampler: ConstrainedSampler,
    tokenizer: PSmilesTokenizer,
    prior: ResolvedClassSamplingPrior,
    resolved: ResolvedStep62Config,
    num_samples: int,
    show_progress: bool = True,
) -> Tuple[List[str], Dict[str, object]]:
    """Backward-compatible alias for S0/S1 class-aware sampling."""

    return sample_with_class_prior(
        sampler=sampler,
        tokenizer=tokenizer,
        prior=prior,
        resolved=resolved,
        num_samples=num_samples,
        show_progress=show_progress,
    )
