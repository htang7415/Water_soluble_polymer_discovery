#!/usr/bin/env python
"""Step 2: Sample from backbone and evaluate generative metrics."""

import os
import sys
import argparse
import re
import time
import math
from pathlib import Path

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
    batch_compute_fingerprints,
    compute_pairwise_diversity,
)
from src.utils.model_scales import get_model_config, get_results_dir
from src.data.tokenizer import PSmilesTokenizer
from src.model.backbone import DiffusionBackbone
from src.model.diffusion import DiscreteMaskingDiffusion
from src.sampling.sampler import ConstrainedSampler
from src.evaluation.generative_metrics import GenerativeEvaluator
from src.utils.reproducibility import seed_everything, save_run_metadata
from src.utils.reporting import save_step_summary, save_artifact_manifest, write_initial_log



# Constraint logging helpers
BOND_CHARS = set(['-', '=', '#', '/', '\\'])


def _smiles_constraint_violations(smiles: str) -> dict:
    if not smiles:
        return {
            "star_count": True,
            "bond_placement": True,
            "paren_balance": True,
            "empty_parens": True,
            "ring_closure": True,
        }

    star_violation = count_stars(smiles) != 2
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
        "bond_placement": bond_violation,
        "paren_balance": paren_violation,
        "empty_parens": empty_parens,
        "ring_closure": ring_violation,
    }


def compute_smiles_constraint_metrics(smiles_list, method, representation, model_size):
    total = len(smiles_list)
    violations = {
        "star_count": 0,
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
):
    valid = []
    for smiles in smiles_list:
        if not check_validity(smiles):
            continue
        if require_target_stars and count_stars(smiles) != target_stars:
            continue
        valid.append(smiles)
    return valid


def _select_target_polymers(
    generated_smiles,
    training_smiles,
    target_count: int,
    target_stars: int,
    sa_max: float,
):
    training_canonical = {canonicalize_smiles(s) or s for s in training_smiles}
    rows = []
    for idx, smiles in enumerate(generated_smiles, start=1):
        is_valid = check_validity(smiles)
        star_count = count_stars(smiles)
        canonical = canonicalize_smiles(smiles) if is_valid else None
        is_novel = bool(canonical) and canonical not in training_canonical
        sa = compute_sa_score(smiles) if is_valid else None
        sa_ok = sa is not None and float(sa) < float(sa_max)
        rows.append(
            {
                "sample_index": idx,
                "smiles": smiles,
                "canonical_smiles": canonical,
                "is_valid": int(is_valid),
                "star_count": int(star_count),
                "is_novel": int(is_novel),
                "sa_score": float(sa) if sa is not None else np.nan,
                "sa_ok": int(sa_ok),
            }
        )

    all_df = pd.DataFrame(rows)
    if all_df.empty:
        summary = {
            "target_count_requested": int(target_count),
            "total_generated": 0,
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
        "total_generated": int(len(all_df)),
        "filter_pass_count": int(filter_mask.sum()),
        "filter_pass_unique": int(len(filtered)),
        "target_count_selected": int(len(selected)),
        "selection_success_rate": float(len(selected) / len(all_df)) if len(all_df) > 0 else 0.0,
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
    sampling_cfg = config.get('sampling', {})
    num_samples = int(args.num_samples if args.num_samples is not None else config.get('sampling', {}).get('num_samples', 10000))
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
    if args.valid_only and args.no_valid_only:
        raise ValueError("Use only one of --valid_only or --no_valid_only")
    if args.valid_only_require_target_stars and args.valid_only_allow_non_target_stars:
        raise ValueError("Use only one of --valid_only_require_target_stars or --valid_only_allow_non_target_stars")
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

    valid_only_oversample_factor = float(
        args.valid_only_oversample_factor
        if args.valid_only_oversample_factor is not None
        else sampling_cfg.get('valid_only_oversample_factor', 1.3)
    )
    if valid_only_oversample_factor < 1.0:
        raise ValueError(f"valid_only_oversample_factor must be >= 1.0, got {valid_only_oversample_factor}")

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
    target_polymer_count = int(sampling_cfg.get("target_polymer_count", 100))
    target_sa_max = float(sampling_cfg.get("target_sa_max", 4.0))
    if target_polymer_count < 1:
        raise ValueError(f"sampling.target_polymer_count must be >=1, got {target_polymer_count}")

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Override results_dir if model_size specified
    base_results_dir = config['paths']['results_dir']
    results_dir = Path(get_results_dir(args.model_size, base_results_dir))

    # Create output directories
    step_dir = results_dir / 'step2_sampling'
    metrics_dir = step_dir / 'metrics'
    figures_dir = step_dir / 'figures'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    seed_info = seed_everything(config['data']['random_seed'])
    save_config(config, step_dir / 'config_used.yaml')
    save_run_metadata(step_dir, args.config, seed_info)
    write_initial_log(
        step_dir=step_dir,
        step_name="step2_sampling",
        context={
            "config_path": args.config,
            "model_size": args.model_size,
            "results_dir": str(results_dir),
            "num_samples": num_samples,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "target_stars": target_stars,
            "valid_only": valid_only,
            "valid_only_require_target_stars": valid_only_require_target_stars,
            "valid_only_oversample_factor": valid_only_oversample_factor,
            "valid_only_max_rounds": valid_only_max_rounds,
            "valid_only_min_samples_per_round": valid_only_min_samples_per_round,
            "random_seed": config['data']['random_seed'],
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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

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

    # Sample
    sampling_start = time.time()
    batch_size = args.batch_size or config['sampling']['batch_size']
    print(f"\n5. Sampling {num_samples} polymers (batch_size={batch_size})...")

    def sample_candidates(n_requested: int, show_progress: bool = True):
        if args.variable_length:
            if show_progress:
                print(f"   Using variable length sampling (range: {args.min_length}-{args.max_length})")
            _, sampled_smiles = sampler.sample_variable_length(
                num_samples=n_requested,
                length_range=(args.min_length, args.max_length),
                batch_size=batch_size,
                samples_per_length=args.samples_per_length,
                show_progress=show_progress
            )
            return sampled_smiles

        replace = n_requested > len(train_df)
        sampled = train_df['p_smiles'].sample(
            n=n_requested,
            replace=replace,
            random_state=np.random.randint(0, 2**31 - 1)
        )
        lengths = [
            min(len(tokenizer.tokenize(s)) + 2, tokenizer.max_length)
            for s in sampled.tolist()
        ]
        if show_progress:
            print(f"   Using training length distribution (min={min(lengths)}, max={max(lengths)})")
        _, sampled_smiles = sampler.sample_batch(
            num_samples=n_requested,
            seq_length=tokenizer.max_length,
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
    }
    if valid_only:
        print(
            f"   Valid-only mode ON (require_target_stars={valid_only_require_target_stars}, "
            f"oversample_factor={valid_only_oversample_factor}, max_rounds={valid_only_max_rounds})"
        )
        generated_smiles = []
        total_raw_generated = 0
        round_rows = []

        for round_idx in range(1, valid_only_max_rounds + 1):
            remaining = num_samples - len(generated_smiles)
            if remaining <= 0:
                break

            n_requested = max(
                remaining,
                int(math.ceil(remaining * valid_only_oversample_factor)),
                valid_only_min_samples_per_round,
            )
            print(
                f"   Valid-only round {round_idx}/{valid_only_max_rounds}: "
                f"request={n_requested}, remaining_target={remaining}"
            )
            batch_smiles = sample_candidates(n_requested=n_requested, show_progress=True)
            total_raw_generated += len(batch_smiles)

            batch_valid = _filter_valid_samples(
                smiles_list=batch_smiles,
                require_target_stars=valid_only_require_target_stars,
                target_stars=target_stars,
            )
            accepted = min(remaining, len(batch_valid))
            if accepted > 0:
                generated_smiles.extend(batch_valid[:accepted])

            round_acceptance = (len(batch_valid) / len(batch_smiles)) if len(batch_smiles) > 0 else 0.0
            round_rows.append(
                {
                    "round": round_idx,
                    "remaining_target_before_round": remaining,
                    "requested_samples": int(len(batch_smiles)),
                    "valid_candidates_found": int(len(batch_valid)),
                    "accepted_this_round": int(accepted),
                    "accepted_cumulative": int(len(generated_smiles)),
                    "round_acceptance_rate": float(round_acceptance),
                }
            )
            print(
                f"     valid={len(batch_valid)}/{len(batch_smiles)} "
                f"({100.0 * round_acceptance:.2f}%), accepted={accepted}, "
                f"cumulative={len(generated_smiles)}/{num_samples}"
            )

        if len(generated_smiles) < num_samples:
            raise RuntimeError(
                "Valid-only sampling did not reach requested samples. "
                f"accepted={len(generated_smiles)}, requested={num_samples}. "
                "Increase valid_only_max_rounds or valid_only_oversample_factor."
            )

        generated_smiles = generated_smiles[:num_samples]
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
            }
        )
    else:
        generated_smiles = sample_candidates(n_requested=num_samples, show_progress=True)
        valid_only_stats.update(
            {
                "valid_only_rounds": 1,
                "valid_only_raw_generated": int(len(generated_smiles)),
                "valid_only_acceptance_rate": 1.0,
                "valid_only_rejected_count": 0,
            }
        )

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
        sample_id=f'uncond_{num_samples}_best_checkpoint',
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
        target_count=target_polymer_count,
        target_stars=target_stars,
        sa_max=target_sa_max,
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
    train_sa = [compute_sa_score(s) for s in list(training_smiles)[:5000]]
    train_sa = [s for s in train_sa if s is not None]
    gen_sa = [compute_sa_score(s) for s in valid_smiles[:5000]]
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
        "num_samples_requested": int(num_samples),
        "num_samples_generated": int(len(generated_smiles)),
        "temperature": float(temperature),
        "top_k": int(top_k) if top_k is not None else None,
        "top_p": float(top_p) if top_p is not None else None,
        "target_stars": int(target_stars),
        "valid_only": bool(valid_only_stats["valid_only"]),
        "valid_only_rounds": int(valid_only_stats["valid_only_rounds"]),
        "valid_only_raw_generated": int(valid_only_stats["valid_only_raw_generated"]),
        "valid_only_acceptance_rate": float(valid_only_stats["valid_only_acceptance_rate"]),
        "valid_only_rejected_count": int(valid_only_stats["valid_only_rejected_count"]),
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
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to generate (default: sampling.num_samples in config)')
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
    parser.add_argument('--valid_only', action='store_true',
                        help='Enable valid-only resampling mode')
    parser.add_argument('--no_valid_only', action='store_true',
                        help='Disable valid-only resampling mode')
    parser.add_argument('--valid_only_require_target_stars', action='store_true',
                        help='In valid-only mode, require star count == target_stars')
    parser.add_argument('--valid_only_allow_non_target_stars', action='store_true',
                        help='In valid-only mode, do not enforce target star count')
    parser.add_argument('--valid_only_oversample_factor', type=float, default=None,
                        help='Oversampling factor per valid-only round (default: sampling.valid_only_oversample_factor)')
    parser.add_argument('--valid_only_max_rounds', type=int, default=None,
                        help='Maximum rounds in valid-only mode (default: sampling.valid_only_max_rounds)')
    parser.add_argument('--valid_only_min_samples_per_round', type=int, default=None,
                        help='Minimum requested samples per valid-only round (default: sampling.valid_only_min_samples_per_round)')
    parser.add_argument('--variable_length', action='store_true',
                        help='Enable variable length sampling')
    parser.add_argument('--min_length', type=int, default=20,
                        help='Minimum sequence length for variable length sampling')
    parser.add_argument('--max_length', type=int, default=100,
                        help='Maximum sequence length for variable length sampling')
    parser.add_argument('--samples_per_length', type=int, default=16,
                        help='Samples per length in variable length mode (controls diversity)')
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
