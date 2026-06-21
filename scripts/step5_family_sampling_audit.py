#!/usr/bin/env python
"""Quick Step 5 family sampling audit for iterative robustness checks."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.polymer_class import BACKBONE_CLASS_MATCH_CLASSES, PolymerClassifier
from src.step5.conditional_sampling import create_conditional_sampler, sample_conditional_with_class_prior
from src.step5.config import build_run_config, load_step5_config
from src.step5.dataset import build_inference_condition_bundle
from src.step5.evaluation import (
    aggregate_round_metrics,
    aggregate_target_row_metrics,
    build_generated_samples_frame,
    build_method_metrics,
    evaluate_generated_samples,
    load_step5_evaluator,
    summarize_target_rows,
)
from src.step5.frozen_sampling import (
    ClassConstrainedSamplingQuotaError,
    create_constrained_sampler,
    load_step1_diffusion,
    resolve_class_sampling_prior,
    sample_unconditional_with_class_prior,
)
from src.step5.run_core import (
    _load_s2_training_artifacts_from_existing_checkpoint,
    run_single_target_sampling,
)
from src.utils.chemistry import canonicalize_smiles, check_validity, count_stars, has_terminal_connection_stars
from src.utils.reproducibility import seed_everything


def _parse_csv_arg(raw: str) -> List[str]:
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def _parse_int_csv_arg(raw: str) -> List[int]:
    return [int(token) for token in _parse_csv_arg(raw)]


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build_temp_step5_config(
    *,
    config_path: str,
    c_target: str,
    enabled_runs: Iterable[str],
) -> str:
    payload = deepcopy(_load_yaml(config_path))
    step5_cfg = payload.setdefault("step5", {})
    step5_cfg["c_target"] = str(c_target)
    step5_cfg["enabled_runs"] = [str(name) for name in enabled_runs]
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
        return handle.name


def _resolve_available_families(config_path: str, families_arg: Optional[str]) -> List[str]:
    if families_arg:
        return [token.lower() for token in _parse_csv_arg(families_arg)]
    payload = _load_yaml(config_path)
    step5_cfg = payload.get("step5", {})
    return [str(name).strip().lower() for name in step5_cfg.get("available_target_classes", [])]


def _family_suffix_token(families: List[str], families_arg: Optional[str]) -> str:
    if not families:
        return "f0"
    if not families_arg:
        return "fall"
    return "f" + "-".join(str(family).strip().lower() for family in families)


def _find_latest_conditional_checkpoint(results_dir: Path, family: str) -> Optional[Path]:
    family_root = results_dir / "step5_inverse_design" / "polymer" / str(family).strip().lower()
    if not family_root.exists():
        return None
    candidates = [
        path
        for path in family_root.rglob("conditional_diffusion_best.pt")
        if "_warm_start" not in path.parts
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda path: (path.stat().st_mtime, -len(path.parts), str(path)), reverse=True)
    return candidates[0]


def _unexpected_atoms(smiles: str) -> List[str]:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles.replace("*", "[*]"))
    if mol is None:
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    seen: set[str] = set()
    ordered: List[str] = []
    for atom in mol.GetAtoms():
        symbol = str(atom.GetSymbol())
        if symbol in seen:
            continue
        seen.add(symbol)
        ordered.append(symbol)
    return ordered


def _class_ok_for_family(classifier: PolymerClassifier, family: str, smiles: str) -> Tuple[bool, bool]:
    family = str(family).strip().lower()
    loose = bool(classifier.classify(smiles).get(family, False))
    if family in BACKBONE_CLASS_MATCH_CLASSES:
        strict = bool(classifier.classify_backbone(smiles).get(family, False))
    else:
        strict = loose
    return loose, strict


def _call_summary_row(
    *,
    family: str,
    mode: str,
    seed: int,
    target_row: pd.Series,
    meta: Dict[str, Any],
    quota_ok: bool,
    returned_count: int,
    shared_seen_scope: str,
    checkpoint_path: Optional[Path],
) -> Dict[str, Any]:
    filter_counts = dict(meta.get("filter_rejection_counts", {})) if isinstance(meta, dict) else {}
    return {
        "family": str(family),
        "mode": str(mode),
        "seed": int(seed),
        "target_row_id": int(target_row["target_row_id"]),
        "target_row_key": str(target_row["target_row_key"]),
        "temperature": float(target_row["temperature"]),
        "phi": float(target_row["phi"]),
        "chi_target": float(target_row["chi_target"]),
        "quota_ok": int(bool(quota_ok)),
        "returned_count": int(returned_count),
        "class_match_acceptance_rate": float(meta.get("class_match_acceptance_rate", 0.0)),
        "class_match_sampling_attempts": int(meta.get("class_match_sampling_attempts", 0)),
        "total_raw_samples_drawn": int(meta.get("total_raw_samples_drawn", 0)),
        "target_class_candidate_count": int(filter_counts.get("target_class_candidate_count", 0)),
        "star_filter_rejected_count": int(filter_counts.get("star_filter_rejected_count", 0)),
        "sidechain_backbone_hybrid_rejected_count": int(
            filter_counts.get("sidechain_backbone_hybrid_rejected_count", 0)
        ),
        "unexpected_atoms_rejected_count": int(filter_counts.get("unexpected_atoms_rejected_count", 0)),
        "duplicate_canonical_rejected_count": int(filter_counts.get("duplicate_canonical_rejected_count", 0)),
        "shared_seen_scope": str(shared_seen_scope),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else "",
    }


def _accepted_sample_rows(
    *,
    family: str,
    mode: str,
    seed: int,
    target_row: pd.Series,
    smiles_list: List[str],
    shared_seen_scope: str,
    classifier: PolymerClassifier,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sample_index, smiles in enumerate(smiles_list, start=1):
        valid_ok = bool(check_validity(smiles))
        canonical = canonicalize_smiles(smiles) if valid_ok else None
        class_loose_ok, class_strict_ok = _class_ok_for_family(classifier, family, smiles) if valid_ok else (False, False)
        rows.append(
            {
                "family": str(family),
                "mode": str(mode),
                "seed": int(seed),
                "target_row_id": int(target_row["target_row_id"]),
                "target_row_key": str(target_row["target_row_key"]),
                "sample_index": int(sample_index),
                "smiles": str(smiles),
                "canonical_smiles": canonical,
                "valid_ok": int(valid_ok),
                "class_ok": int(class_loose_ok),
                "backbone_class_ok": int(class_strict_ok),
                "star_ok": int(valid_ok and count_stars(smiles) == 2 and has_terminal_connection_stars(smiles, expected_stars=2)),
                "atom_symbols": ",".join(_unexpected_atoms(smiles)) if valid_ok else "",
                "shared_seen_scope": str(shared_seen_scope),
            }
        )
    return rows


def _run_s0_call(
    *,
    resolved,
    run_cfg: Dict[str, Any],
    target_row: pd.Series,
    device: str,
    generation_budget: int,
    num_steps: Optional[int],
    seen_canonical_smiles: Optional[set[str]],
    shared_artifacts: Tuple[Any, Any, Path],
) -> Tuple[List[str], Dict[str, Any], Optional[Path]]:
    tokenizer, diffusion_model, checkpoint_path = shared_artifacts
    prior = resolve_class_sampling_prior(resolved, run_cfg, tokenizer)
    sampler = create_constrained_sampler(
        diffusion_model=diffusion_model,
        tokenizer=tokenizer,
        resolved=resolved,
        prior=prior,
        device=device,
    )
    if num_steps is not None:
        sampler.num_steps = int(num_steps)
    smiles, meta = sample_unconditional_with_class_prior(
        sampler=sampler,
        tokenizer=tokenizer,
        prior=prior,
        resolved=resolved,
        num_samples=int(generation_budget),
        show_progress=False,
        seen_canonical_smiles=seen_canonical_smiles,
    )
    return smiles, meta, checkpoint_path


def _run_s2_call(
    *,
    resolved,
    run_cfg: Dict[str, Any],
    target_row: pd.Series,
    device: str,
    generation_budget: int,
    num_steps: Optional[int],
    seen_canonical_smiles: Optional[set[str]],
    shared_artifacts,
) -> Tuple[List[str], Dict[str, Any], Optional[Path]]:
    artifacts = shared_artifacts
    prior = resolve_class_sampling_prior(resolved, run_cfg, artifacts.tokenizer)
    chi_goal = float(target_row["chi_target"])
    condition_bundle = torch.tensor(
        build_inference_condition_bundle(
            temperature=float(target_row["temperature"]),
            phi=float(target_row["phi"]),
            chi_goal=chi_goal,
            scaler=artifacts.scaler,
            soluble=1,
        ),
        dtype=torch.float32,
        device=device,
    )
    sampler = create_conditional_sampler(
        diffusion_model=artifacts.diffusion_model,
        tokenizer=artifacts.tokenizer,
        resolved=resolved,
        prior=prior,
        condition_bundle=condition_bundle,
        cfg_scale=float(run_cfg["s2"]["cfg_scale"]),
        device=device,
        num_steps=num_steps,
    )
    smiles, meta = sample_conditional_with_class_prior(
        sampler=sampler,
        tokenizer=artifacts.tokenizer,
        prior=prior,
        resolved=resolved,
        num_samples=int(generation_budget),
        show_progress=False,
        seen_canonical_smiles=seen_canonical_smiles,
    )
    return smiles, meta, artifacts.checkpoint_path


def _run_s3_call(
    *,
    resolved,
    run_cfg: Dict[str, Any],
    target_row: pd.Series,
    device: str,
    generation_budget: int,
    seen_canonical_smiles: Optional[set[str]],
    shared_artifacts,
    evaluator,
) -> Tuple[List[str], Dict[str, Any], Optional[Path]]:
    artifacts = shared_artifacts
    prior = resolve_class_sampling_prior(resolved, run_cfg, artifacts.tokenizer)
    smiles, _guidance_stats, meta = run_single_target_sampling(
        run_cfg=run_cfg,
        resolved=resolved,
        target_row=target_row,
        tokenizer=artifacts.tokenizer,
        diffusion_model=artifacts.diffusion_model,
        prior=prior,
        evaluator=evaluator,
        s2_scaler=artifacts.scaler,
        device=device,
        generation_budget=int(generation_budget),
        seen_canonical_smiles=seen_canonical_smiles,
    )
    return smiles, meta, artifacts.checkpoint_path


def _run_s1_call(
    *,
    resolved,
    run_cfg: Dict[str, Any],
    target_row: pd.Series,
    device: str,
    generation_budget: int,
    seen_canonical_smiles: Optional[set[str]],
    shared_artifacts: Tuple[Any, Any, Path],
    evaluator,
) -> Tuple[List[str], Dict[str, Any], Optional[Path]]:
    tokenizer, diffusion_model, checkpoint_path = shared_artifacts
    prior = resolve_class_sampling_prior(resolved, run_cfg, tokenizer)
    smiles, _guidance_stats, meta = run_single_target_sampling(
        run_cfg=run_cfg,
        resolved=resolved,
        target_row=target_row,
        tokenizer=tokenizer,
        diffusion_model=diffusion_model,
        prior=prior,
        evaluator=evaluator,
        device=device,
        generation_budget=int(generation_budget),
        seen_canonical_smiles=seen_canonical_smiles,
    )
    return smiles, meta, checkpoint_path


def _summarize_family(call_df: pd.DataFrame, sample_df: pd.DataFrame) -> Dict[str, Any]:
    valid_samples = sample_df[sample_df["valid_ok"] == 1].copy()
    valid_samples = valid_samples[valid_samples["canonical_smiles"].notna()]
    n_valid = int(len(valid_samples))
    n_unique = int(valid_samples["canonical_smiles"].nunique()) if n_valid > 0 else 0
    duplicate_rate = (1.0 - (float(n_unique) / float(n_valid))) if n_valid > 0 else 0.0
    return {
        "family": str(call_df["family"].iloc[0]) if not call_df.empty else "",
        "mode": str(call_df["mode"].iloc[0]) if not call_df.empty else "",
        "shared_seen_scope": str(call_df["shared_seen_scope"].iloc[0]) if not call_df.empty else "",
        "n_calls": int(len(call_df)),
        "quota_ok_rate": float(call_df["quota_ok"].mean()) if not call_df.empty else 0.0,
        "returned_sample_rate": float(call_df["returned_count"].mean()) if not call_df.empty else 0.0,
        "mean_acceptance_rate": float(call_df["class_match_acceptance_rate"].mean()) if not call_df.empty else 0.0,
        "mean_attempts": float(call_df["class_match_sampling_attempts"].mean()) if not call_df.empty else 0.0,
        "mean_raw_draws": float(call_df["total_raw_samples_drawn"].mean()) if not call_df.empty else 0.0,
        "mean_duplicate_rejections": float(call_df["duplicate_canonical_rejected_count"].mean()) if not call_df.empty else 0.0,
        "mean_hybrid_rejections": float(call_df["sidechain_backbone_hybrid_rejected_count"].mean()) if not call_df.empty else 0.0,
        "valid_ok_rate": float(sample_df["valid_ok"].mean()) if not sample_df.empty else 0.0,
        "class_ok_rate": float(sample_df["class_ok"].mean()) if not sample_df.empty else 0.0,
        "backbone_class_ok_rate": float(sample_df["backbone_class_ok"].mean()) if not sample_df.empty else 0.0,
        "star_ok_rate": float(sample_df["star_ok"].mean()) if not sample_df.empty else 0.0,
        "canonical_uniqueness_rate": (float(n_unique) / float(n_valid)) if n_valid > 0 else 0.0,
        "canonical_duplicate_rate": float(duplicate_rate),
        "num_valid_samples": int(n_valid),
        "num_unique_valid_canonicals": int(n_unique),
    }


def _augment_summary_with_evaluation(
    summary_row: Dict[str, Any],
    *,
    evaluation_df: pd.DataFrame,
    method_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    if evaluation_df.empty:
        summary_row.update(
            {
                "novel_ok_rate": 0.0,
                "soluble_ok_rate": 0.0,
                "chi_ok_rate": 0.0,
                "chi_band_ok_rate": 0.0,
                "property_frontier_hit_rate": 0.0,
                "property_frontier_band_hit_rate": 0.0,
                "sa_blocked_property_frontier_rate": 0.0,
                "sa_ok_discovery_rate": 0.0,
                "property_success_hit_rate": 0.0,
                "property_success_hit_rate_discovery": 0.0,
                "success_hit_rate": 0.0,
                "success_hit_rate_discovery": 0.0,
                "macro_target_row_property_success_hit_rate": 0.0,
                "macro_target_row_property_success_hit_rate_discovery": 0.0,
                "macro_target_row_success_hit_rate": 0.0,
                "macro_target_row_success_hit_rate_discovery": 0.0,
            }
        )
        return summary_row

    frontier_hit_mask = evaluation_df["soluble_ok"].astype(int) & evaluation_df["chi_ok"].astype(int)
    frontier_band_hit_mask = evaluation_df["soluble_ok"].astype(int) & evaluation_df["chi_band_ok"].astype(int)
    sa_blocked_frontier_mask = frontier_hit_mask & (1 - evaluation_df["sa_ok_discovery"].astype(int))
    summary_row.update(
        {
            "novel_ok_rate": float(evaluation_df["novel_ok"].mean()),
            "soluble_ok_rate": float(evaluation_df["soluble_ok"].mean()),
            "chi_ok_rate": float(evaluation_df["chi_ok"].mean()),
            "chi_band_ok_rate": float(evaluation_df["chi_band_ok"].mean()),
            "property_frontier_hit_rate": float(frontier_hit_mask.mean()),
            "property_frontier_band_hit_rate": float(frontier_band_hit_mask.mean()),
            "sa_blocked_property_frontier_rate": float(sa_blocked_frontier_mask.mean()),
            "sa_ok_discovery_rate": float(evaluation_df["sa_ok_discovery"].mean()),
            "property_success_hit_rate": float(evaluation_df["property_success_hit"].mean()),
            "property_success_hit_rate_discovery": float(evaluation_df["property_success_hit_discovery"].mean()),
            "success_hit_rate": float(evaluation_df["success_hit"].mean()),
            "success_hit_rate_discovery": float(evaluation_df["success_hit_discovery"].mean()),
            "macro_target_row_property_success_hit_rate": float(
                method_metrics.get("macro_average_row_mean_property_success_hit_rate", 0.0) or 0.0
            ),
            "macro_target_row_property_success_hit_rate_discovery": float(
                method_metrics.get("macro_average_row_mean_property_success_hit_rate_discovery", 0.0) or 0.0
            ),
            "macro_target_row_success_hit_rate": float(
                method_metrics.get("macro_average_row_mean_success_hit_rate", 0.0) or 0.0
            ),
            "macro_target_row_success_hit_rate_discovery": float(
                method_metrics.get("macro_average_row_mean_success_hit_rate_discovery", 0.0) or 0.0
            ),
        }
    )
    return summary_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick Step 5 family sampling audit.")
    parser.add_argument("--config", default="configs/config5.yaml")
    parser.add_argument("--base_config", default="configs/config.yaml")
    parser.add_argument("--model_size", default="small")
    parser.add_argument("--device", default=None)
    parser.add_argument("--mode", choices=["S0", "S1", "S2", "S3"], default="S0")
    parser.add_argument("--families", default=None)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--seeds", default="42,43")
    parser.add_argument("--generation_budget", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--shared_seen_scope", choices=["none", "per_seed"], default="none")
    parser.add_argument("--evaluate_step4", action="store_true")
    parser.add_argument("--skip_novelty_reference", action="store_true")
    parser.add_argument("--output_dir", default="results_small/step5_inverse_design/polymer/_family_quality_audit")
    args = parser.parse_args()

    device = str(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    families = _resolve_available_families(args.config, args.families)
    seeds = _parse_int_csv_arg(args.seeds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classifier = PolymerClassifier()
    run_name_map = {
        "S0": "S0_raw_unconditional",
        "S1": "S1_guided_frozen",
        "S2": "S2_conditional",
        "S3": "S3_conditional_guided",
    }
    run_name = run_name_map[str(args.mode)]
    call_rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    missing_families: List[Dict[str, Any]] = []
    evaluation_frames: List[pd.DataFrame] = []
    target_row_metric_frames: List[pd.DataFrame] = []
    target_row_summary_frames: List[pd.DataFrame] = []
    round_metric_frames: List[pd.DataFrame] = []
    method_metric_rows: List[Dict[str, Any]] = []

    shared_step1_artifacts: Optional[Tuple[Any, Any, Path]] = None

    for family in families:
        print(f"audit_start family={family} mode={args.mode} rows={int(args.rows)} seeds={seeds}")
        temp_config_path = _build_temp_step5_config(
            config_path=args.config,
            c_target=family,
            enabled_runs=[run_name],
        )
        resolved = load_step5_config(
            config_path=temp_config_path,
            base_config_path=args.base_config,
            model_size=args.model_size,
        )
        run_cfg = build_run_config(resolved, run_name)
        target_rows_df = resolved.target_family_df.head(int(args.rows)).copy()
        if target_rows_df.empty:
            missing_families.append({"family": str(family), "reason": "no_target_rows"})
            print(f"audit_skip family={family} reason=no_target_rows")
            continue

        family_evaluator = None
        if args.mode in {"S1", "S3"} or args.evaluate_step4:
            family_evaluator = load_step5_evaluator(
                resolved,
                device=device,
                skip_novelty_reference=bool(args.skip_novelty_reference),
            )

        checkpoint_path: Optional[Path] = None
        if args.mode in {"S0", "S1"}:
            if shared_step1_artifacts is None:
                shared_step1_artifacts = load_step1_diffusion(resolved, device=device)
            mode_context: Dict[str, Any] = {"shared_artifacts": shared_step1_artifacts}
        else:
            checkpoint_path = _find_latest_conditional_checkpoint(resolved.results_dir, family)
            if checkpoint_path is None:
                missing_families.append({"family": str(family), "reason": "missing_conditional_checkpoint"})
                print(f"audit_skip family={family} reason=missing_conditional_checkpoint")
                continue
            metrics_dir = checkpoint_path.parent.parent / "metrics"
            mode_context = {
                "checkpoint_path": checkpoint_path,
                "shared_artifacts": _load_s2_training_artifacts_from_existing_checkpoint(
                    resolved=resolved,
                    run_cfg=run_cfg,
                    checkpoint_path=checkpoint_path,
                    metrics_dir=metrics_dir,
                    device=device,
                ),
            }

        family_call_rows: List[Dict[str, Any]] = []
        family_sample_rows: List[Dict[str, Any]] = []
        family_generated_frames: List[pd.DataFrame] = []
        sample_id_start = 1

        for seed in seeds:
            seen_cache: Optional[set[str]] = set() if args.shared_seen_scope == "per_seed" else None
            seed_everything(int(seed), deterministic=True)
            for _, target_row in target_rows_df.iterrows():
                try:
                    if args.mode == "S0":
                        accepted_smiles, meta, used_checkpoint = _run_s0_call(
                            resolved=resolved,
                            run_cfg=run_cfg,
                            target_row=target_row,
                            device=device,
                            generation_budget=int(args.generation_budget),
                            num_steps=args.num_steps,
                            seen_canonical_smiles=seen_cache,
                            shared_artifacts=mode_context["shared_artifacts"],
                        )
                    elif args.mode == "S1":
                        accepted_smiles, meta, used_checkpoint = _run_s1_call(
                            resolved=resolved,
                            run_cfg=run_cfg,
                            target_row=target_row,
                            device=device,
                            generation_budget=int(args.generation_budget),
                            seen_canonical_smiles=seen_cache,
                            shared_artifacts=mode_context["shared_artifacts"],
                            evaluator=family_evaluator,
                        )
                    elif args.mode == "S2":
                        accepted_smiles, meta, used_checkpoint = _run_s2_call(
                            resolved=resolved,
                            run_cfg=run_cfg,
                            target_row=target_row,
                            device=device,
                            generation_budget=int(args.generation_budget),
                            num_steps=args.num_steps,
                            seen_canonical_smiles=seen_cache,
                            shared_artifacts=mode_context["shared_artifacts"],
                        )
                    else:
                        accepted_smiles, meta, used_checkpoint = _run_s3_call(
                            resolved=resolved,
                            run_cfg=run_cfg,
                            target_row=target_row,
                            device=device,
                            generation_budget=int(args.generation_budget),
                            seen_canonical_smiles=seen_cache,
                            shared_artifacts=mode_context["shared_artifacts"],
                            evaluator=family_evaluator,
                        )
                    quota_ok = True
                except ClassConstrainedSamplingQuotaError as exc:
                    accepted_smiles = list(exc.accepted_smiles)
                    meta = dict(exc.metadata)
                    used_checkpoint = checkpoint_path
                    quota_ok = False

                family_call_rows.append(
                    _call_summary_row(
                        family=family,
                        mode=args.mode,
                        seed=int(seed),
                        target_row=target_row,
                        meta=meta,
                        quota_ok=quota_ok,
                        returned_count=len(accepted_smiles),
                        shared_seen_scope=args.shared_seen_scope,
                        checkpoint_path=used_checkpoint,
                    )
                )
                family_sample_rows.extend(
                    _accepted_sample_rows(
                        family=family,
                        mode=args.mode,
                        seed=int(seed),
                        target_row=target_row,
                        smiles_list=accepted_smiles,
                        shared_seen_scope=args.shared_seen_scope,
                        classifier=classifier,
                    )
                )
                generated_df = build_generated_samples_frame(
                    accepted_smiles,
                    target_row=target_row,
                    round_id=1,
                    sampling_seed=int(seed),
                    run_name=f"audit_{args.mode.lower()}_{family}",
                    canonical_family=str(family),
                    sample_id_start=sample_id_start,
                )
                sample_id_start += int(len(generated_df))
                if not generated_df.empty:
                    family_generated_frames.append(generated_df)

        family_call_df = pd.DataFrame(family_call_rows)
        family_sample_df = pd.DataFrame(family_sample_rows)
        if not family_sample_df.empty:
            duplicate_mask = family_sample_df["canonical_smiles"].duplicated(keep=False)
            family_sample_df["duplicate_within_audit"] = duplicate_mask.fillna(False).astype(int)
        family_summary_row = _summarize_family(family_call_df, family_sample_df)
        if args.evaluate_step4 and family_generated_frames:
            family_generated_df = pd.concat(family_generated_frames, ignore_index=True)
            family_evaluation_df = evaluate_generated_samples(family_generated_df, family_evaluator)
            family_target_row_metrics_df = aggregate_target_row_metrics(family_evaluation_df)
            family_target_row_summary_df = summarize_target_rows(family_target_row_metrics_df)
            family_round_metrics_df = aggregate_round_metrics(family_evaluation_df, family_target_row_metrics_df)
            family_method_metrics = build_method_metrics(family_round_metrics_df, family_target_row_summary_df)
            family_method_metrics.update(
                {
                    "family": str(family),
                    "mode": str(args.mode),
                    "shared_seen_scope": str(args.shared_seen_scope),
                    "mean_class_match_acceptance_rate": float(family_call_df["class_match_acceptance_rate"].mean())
                    if "class_match_acceptance_rate" in family_call_df.columns and not family_call_df.empty
                    else 0.0,
                    "mean_total_raw_samples_drawn": float(family_call_df["total_raw_samples_drawn"].mean())
                    if "total_raw_samples_drawn" in family_call_df.columns and not family_call_df.empty
                    else 0.0,
                }
            )
            family_summary_row = _augment_summary_with_evaluation(
                family_summary_row,
                evaluation_df=family_evaluation_df,
                method_metrics=family_method_metrics,
            )
            evaluation_frames.append(family_evaluation_df)
            target_row_metric_frames.append(family_target_row_metrics_df)
            target_row_summary_frames.append(family_target_row_summary_df)
            round_metric_frames.append(family_round_metrics_df)
            method_metric_rows.append(family_method_metrics)
        summary_rows.append(family_summary_row)
        call_rows.extend(family_call_rows)
        sample_rows.extend(family_sample_rows)
        print(f"audit_done family={family} calls={len(family_call_rows)} samples={len(family_sample_rows)}")

    summary_df = pd.DataFrame(summary_rows)
    call_df = pd.DataFrame(call_rows)
    sample_df = pd.DataFrame(sample_rows)
    missing_df = pd.DataFrame(missing_families)
    evaluation_df = pd.concat(evaluation_frames, ignore_index=True) if evaluation_frames else pd.DataFrame()
    target_row_metrics_df = (
        pd.concat(target_row_metric_frames, ignore_index=True) if target_row_metric_frames else pd.DataFrame()
    )
    target_row_summary_df = (
        pd.concat(target_row_summary_frames, ignore_index=True) if target_row_summary_frames else pd.DataFrame()
    )
    round_metrics_df = pd.concat(round_metric_frames, ignore_index=True) if round_metric_frames else pd.DataFrame()
    method_metrics_df = pd.DataFrame(method_metric_rows)

    suffix = (
        f"{args.mode.lower()}_rows{int(args.rows)}_seeds{'-'.join(str(seed) for seed in seeds)}"
        f"_g{int(args.generation_budget)}_{args.shared_seen_scope}_{_family_suffix_token(families, args.families)}"
    )
    if args.num_steps is not None:
        suffix += f"_steps{int(args.num_steps)}"

    summary_path = output_dir / f"family_sampling_audit_{suffix}_summary.csv"
    call_path = output_dir / f"family_sampling_audit_{suffix}_calls.csv"
    sample_path = output_dir / f"family_sampling_audit_{suffix}_samples.csv"
    missing_path = output_dir / f"family_sampling_audit_{suffix}_missing.csv"
    manifest_path = output_dir / f"family_sampling_audit_{suffix}.json"
    evaluation_path = output_dir / f"family_sampling_audit_{suffix}_evaluation.csv"
    target_row_metrics_path = output_dir / f"family_sampling_audit_{suffix}_target_row_metrics.csv"
    target_row_summary_path = output_dir / f"family_sampling_audit_{suffix}_target_row_summary.csv"
    round_metrics_path = output_dir / f"family_sampling_audit_{suffix}_round_metrics.csv"
    method_metrics_path = output_dir / f"family_sampling_audit_{suffix}_method_metrics.csv"

    summary_df.to_csv(summary_path, index=False)
    call_df.to_csv(call_path, index=False)
    sample_df.to_csv(sample_path, index=False)
    missing_df.to_csv(missing_path, index=False)
    if args.evaluate_step4:
        evaluation_df.to_csv(evaluation_path, index=False)
        target_row_metrics_df.to_csv(target_row_metrics_path, index=False)
        target_row_summary_df.to_csv(target_row_summary_path, index=False)
        round_metrics_df.to_csv(round_metrics_path, index=False)
        method_metrics_df.to_csv(method_metrics_path, index=False)

    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "mode": args.mode,
                "device": device,
                "families": families,
                "rows": int(args.rows),
                "seeds": seeds,
                "generation_budget": int(args.generation_budget),
                "num_steps": args.num_steps,
                "shared_seen_scope": args.shared_seen_scope,
                "summary_path": str(summary_path),
                "calls_path": str(call_path),
                "samples_path": str(sample_path),
                "missing_path": str(missing_path),
                "evaluate_step4": bool(args.evaluate_step4),
                "skip_novelty_reference": bool(args.skip_novelty_reference),
                "evaluation_path": str(evaluation_path) if args.evaluate_step4 else "",
                "target_row_metrics_path": str(target_row_metrics_path) if args.evaluate_step4 else "",
                "target_row_summary_path": str(target_row_summary_path) if args.evaluate_step4 else "",
                "round_metrics_path": str(round_metrics_path) if args.evaluate_step4 else "",
                "method_metrics_path": str(method_metrics_path) if args.evaluate_step4 else "",
            },
            handle,
            indent=2,
        )

    print(f"summary={summary_path}")
    print(f"calls={call_path}")
    print(f"samples={sample_path}")
    print(f"missing={missing_path}")
    if args.evaluate_step4:
        print(f"evaluation={evaluation_path}")
        print(f"target_row_metrics={target_row_metrics_path}")
        print(f"target_row_summary={target_row_summary_path}")
        print(f"round_metrics={round_metrics_path}")
        print(f"method_metrics={method_metrics_path}")


if __name__ == "__main__":
    main()
