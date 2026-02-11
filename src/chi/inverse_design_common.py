"""Common utilities for chi inverse-design steps."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.chi.embeddings import load_backbone_from_step1
from src.chi.model import (
    BackbonePhysicsGuidedChiModel,
    BackboneSolubilityClassifierModel,
    PhysicsGuidedChiModel,
    SolubilityClassifier,
)
from src.chi.constants import COEFF_NAMES
from src.utils.chemistry import canonicalize_smiles, check_validity, count_stars
from src.utils.numerics import stable_sigmoid


CLASS_NAME_MAP = {1: "Water-soluble", 0: "Water-insoluble"}


def default_chi_config(config: Dict, step: str | None = None) -> Dict:
    chi_cfg = config.get("chi_training", {})
    shared = chi_cfg.get("shared", {}) if isinstance(chi_cfg.get("shared", {}), dict) else {}
    shared_embedding = shared.get("embedding", {}) if isinstance(shared.get("embedding", {}), dict) else {}
    step5_cfg = (
        chi_cfg.get("step5_inverse_design", {})
        if isinstance(chi_cfg.get("step5_inverse_design", {}), dict)
        else {}
    )
    step6_cfg = (
        chi_cfg.get("step6_class_inverse_design", {})
        if isinstance(chi_cfg.get("step6_class_inverse_design", {}), dict)
        else {}
    )

    defaults = {
        "split_mode": "polymer",
        "epsilon": 0.05,
        "class_weight": None,  # deprecated (kept for backward-compatible parsing)
        "polymer_class_weight": None,  # deprecated (kept for backward-compatible parsing)
        "candidate_source": "novel",
        "property_rule": "upper_bound",
        "coverage_topk": 5,
        "target_temperature": 293.15,
        "target_phi": 0.2,
        "target_polymer_class": "all",
        "target_polymer_count": 100,
        "target_sa_max": 4.0,
        "embedding_batch_size": 128,
        "embedding_timestep": int(config.get("training_property", {}).get("default_timestep", 1)),
        "uncertainty_enabled": False,
        "uncertainty_mc_samples": 20,
        "uncertainty_class_z": 1.0,
        "uncertainty_property_z": 1.0,
        "uncertainty_score_weight": 0.0,
        "uncertainty_seed": int(config.get("data", {}).get("random_seed", 42)),
    }

    out = defaults.copy()
    out["split_mode"] = str(shared.get("split_mode", chi_cfg.get("split_mode", defaults["split_mode"])))
    out["embedding_batch_size"] = int(
        shared_embedding.get("batch_size", chi_cfg.get("embedding_batch_size", defaults["embedding_batch_size"]))
    )
    out["embedding_timestep"] = int(
        shared_embedding.get("timestep", chi_cfg.get("embedding_timestep", defaults["embedding_timestep"]))
    )

    if step == "step5":
        out.update(step5_cfg)
    elif step == "step6":
        out.update(step6_cfg)
    else:
        # Backward-compatible union if caller does not specify a step.
        out.update(step5_cfg)
        out.update(step6_cfg)

    # Legacy flat keys still override defaults when present.
    for key in [
        "epsilon",
        "class_weight",
        "polymer_class_weight",
        "candidate_source",
        "property_rule",
        "coverage_topk",
        "target_temperature",
        "target_phi",
        "target_polymer_class",
        "target_polymer_count",
        "target_sa_max",
        "uncertainty_enabled",
        "uncertainty_mc_samples",
        "uncertainty_class_z",
        "uncertainty_property_z",
        "uncertainty_score_weight",
        "uncertainty_seed",
    ]:
        if key in chi_cfg:
            out[key] = chi_cfg[key]

    return out


def set_plot_style(font_size: int) -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "legend.fontsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
        }
    )


def parse_candidate_source(value: str) -> str:
    v = value.strip().lower()
    if v in {"novel", "generated", "step2"}:
        return "novel"
    if v in {"known", "step4", "training"}:
        return "known"
    if v in {"hybrid", "both"}:
        return "hybrid"
    raise ValueError(
        "candidate_source must be one of {'novel','known','hybrid'} "
        "(aliases: generated/step2 -> novel, step4/training -> known, both -> hybrid)"
    )


def _normalize_checkpoint_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(str(k).startswith("_orig_mod.") for k in state_dict.keys()):
        return {str(k).replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _load_known_candidates_from_step4_metrics(
    step4_reg_metrics_dir: Path,
    step4_cls_metrics_dir: Path,
    training_canonical: set[str],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    reg_candidates = [
        step4_reg_metrics_dir / "polymer_coefficients_regression_only.csv",
        step4_reg_metrics_dir / "polymer_coefficients.csv",
    ]
    reg_path = next((p for p in reg_candidates if p.exists()), reg_candidates[0])
    if not reg_path.exists():
        raise FileNotFoundError(
            "Known candidate source requires Step4 regression polymer coefficients. "
            f"Expected one of: {', '.join(str(p) for p in reg_candidates)}"
        )
    reg_df = pd.read_csv(reg_path)

    required_reg = {"polymer_id", "Polymer", "SMILES", *COEFF_NAMES}
    missing_reg = sorted(required_reg - set(reg_df.columns))
    if missing_reg:
        raise ValueError(f"Regression known-candidate file missing columns: {missing_reg}")

    cls_all = step4_cls_metrics_dir / "class_predictions_all.csv"
    if cls_all.exists():
        cls_df = pd.read_csv(cls_all)
        cls_source = cls_all
    else:
        split_paths = [
            step4_cls_metrics_dir / "class_predictions_train.csv",
            step4_cls_metrics_dir / "class_predictions_val.csv",
            step4_cls_metrics_dir / "class_predictions_test.csv",
        ]
        available = [p for p in split_paths if p.exists()]
        if not available:
            raise FileNotFoundError(
                "Known candidate source requires Step4 classification predictions. "
                f"Expected: {cls_all} or split files under {step4_cls_metrics_dir}"
            )
        cls_df = pd.concat([pd.read_csv(p) for p in available], ignore_index=True)
        cls_source = available[0]

    required_cls = {"polymer_id", "class_prob"}
    missing_cls = sorted(required_cls - set(cls_df.columns))
    if missing_cls:
        raise ValueError(f"Classification known-candidate file missing columns: {missing_cls}")

    by_poly = cls_df.groupby("polymer_id", as_index=False)
    cls_poly = by_poly["class_prob"].mean().rename(columns={"class_prob": "class_prob"})
    cls_poly["class_prob_std"] = (
        by_poly["class_prob"].std(ddof=0)["class_prob"].fillna(0.0).astype(float)
    )
    if "class_logit" in cls_df.columns:
        cls_poly["class_logit"] = by_poly["class_logit"].mean()["class_logit"]
        cls_poly["class_logit_std"] = (
            by_poly["class_logit"].std(ddof=0)["class_logit"].fillna(0.0).astype(float)
        )
    if "water_soluble" in cls_df.columns:
        cls_poly["water_soluble"] = by_poly["water_soluble"].max()["water_soluble"]

    out = reg_df.merge(cls_poly, on="polymer_id", how="left")
    if out["class_prob"].isna().any():
        missing = int(out["class_prob"].isna().sum())
        raise ValueError(
            "Failed to attach class probabilities for known candidates: "
            f"{missing} polymers missing class_prob after merge."
        )

    out = out.copy()
    out["canonical_smiles"] = out["SMILES"].astype(str).map(canonicalize_smiles)
    out["canonical_smiles"] = out["canonical_smiles"].where(out["canonical_smiles"].notna(), out["SMILES"].astype(str))
    out["candidate_source"] = "known_step4"
    out["is_novel_vs_train"] = (~out["canonical_smiles"].isin(training_canonical)).astype(int)
    if "water_soluble" not in out.columns:
        out["water_soluble"] = -1
    if "class_logit" not in out.columns:
        out["class_logit"] = np.nan
    if "class_prob_std" not in out.columns:
        out["class_prob_std"] = 0.0
    if "class_logit_std" not in out.columns:
        out["class_logit_std"] = 0.0
    for name in COEFF_NAMES:
        std_col = f"{name}_std"
        if std_col not in out.columns:
            out[std_col] = 0.0
    out["class_prob_std"] = out["class_prob_std"].fillna(0.0).astype(float)
    out["class_logit_std"] = out["class_logit_std"].fillna(0.0).astype(float)
    for name in COEFF_NAMES:
        out[f"{name}_std"] = out[f"{name}_std"].fillna(0.0).astype(float)

    summary = {
        "known_regression_coefficients_csv": str(reg_path),
        "known_class_predictions_csv": str(cls_source),
        "known_candidate_count": int(len(out)),
    }
    return out, summary


def resolve_training_smiles(results_dir: Path, base_results_dir: Path) -> set[str]:
    train_path = results_dir / "train_unlabeled.csv"
    if not train_path.exists():
        train_path = base_results_dir / "train_unlabeled.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Training set not found for novelty reference: {train_path}")

    train_df = pd.read_csv(train_path)
    if "p_smiles" in train_df.columns:
        smiles_col = "p_smiles"
    elif "smiles" in train_df.columns:
        smiles_col = "smiles"
    else:
        raise ValueError(f"Training CSV missing smiles column: {train_path}")

    canonical = train_df[smiles_col].astype(str).map(canonicalize_smiles)
    canonical = canonical[canonical.notna()]
    return set(canonical.tolist())


def prepare_novel_candidates(
    generated_csv: Path,
    smiles_column: str,
    training_canonical: set[str],
    require_two_stars: bool = True,
    max_novel_candidates: int | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if not generated_csv.exists():
        raise FileNotFoundError(f"Generated samples CSV not found: {generated_csv}")

    raw_df = pd.read_csv(generated_csv)
    if smiles_column not in raw_df.columns:
        fallback_cols = ["smiles", "p_smiles", "SMILES"]
        found = next((c for c in fallback_cols if c in raw_df.columns), None)
        if found is None:
            raise ValueError(f"Generated CSV missing smiles column '{smiles_column}'")
        smiles_column = found

    input_smiles = raw_df[smiles_column].astype(str).tolist()

    rows = []
    n_valid = 0
    n_valid_two_star = 0
    for smi in input_smiles:
        if not check_validity(smi):
            continue
        n_valid += 1
        if require_two_stars and count_stars(smi) != 2:
            continue
        n_valid_two_star += 1
        canon = canonicalize_smiles(smi)
        if canon is None:
            continue
        rows.append({"SMILES": smi, "canonical_smiles": canon})

    cand_df = pd.DataFrame(rows)
    n_valid_unique = int(cand_df["canonical_smiles"].nunique()) if not cand_df.empty else 0

    if cand_df.empty:
        summary = {
            "n_generated_input": int(len(input_smiles)),
            "n_valid": int(n_valid),
            "n_valid_two_stars": int(n_valid_two_star),
            "n_valid_unique": int(n_valid_unique),
            "n_novel_unique": 0,
            "novel_fraction_among_valid_unique": 0.0,
        }
        return cand_df, summary

    cand_df = cand_df.drop_duplicates(subset=["canonical_smiles"]).reset_index(drop=True)
    novel_mask = ~cand_df["canonical_smiles"].isin(training_canonical)
    cand_df = cand_df[novel_mask].reset_index(drop=True)

    if max_novel_candidates is not None and max_novel_candidates > 0:
        cand_df = cand_df.head(int(max_novel_candidates)).copy()

    cand_df = cand_df.reset_index(drop=True)
    cand_df["polymer_id"] = np.arange(len(cand_df), dtype=int)
    cand_df["Polymer"] = [f"Novel_{i + 1:06d}" for i in range(len(cand_df))]

    n_novel_unique = int(len(cand_df))
    summary = {
        "n_generated_input": int(len(input_smiles)),
        "n_valid": int(n_valid),
        "n_valid_two_stars": int(n_valid_two_star),
        "n_valid_unique": int(n_valid_unique),
        "n_novel_unique": int(n_novel_unique),
        "novel_fraction_among_valid_unique": float(n_novel_unique / n_valid_unique) if n_valid_unique > 0 else 0.0,
    }
    return cand_df, summary


@torch.no_grad()
def infer_coefficients_for_novel_candidates(
    novel_df: pd.DataFrame,
    config: Dict,
    model_size: str | None,
    chi_checkpoint_path: Path,
    class_checkpoint_path: Path | None,
    backbone_checkpoint_path: str | None,
    device: str,
    timestep: int,
    pooling: str,
    batch_size: int,
    uncertainty_enabled: bool = False,
    uncertainty_mc_samples: int = 20,
    uncertainty_seed: int | None = None,
) -> pd.DataFrame:
    mc_enabled = bool(uncertainty_enabled and int(uncertainty_mc_samples) >= 2)
    if novel_df.empty:
        return pd.DataFrame(
            columns=[
                "polymer_id",
                "Polymer",
                "SMILES",
                "canonical_smiles",
                "water_soluble",
                "class_logit",
                "class_logit_std",
                "class_prob",
                "class_prob_std",
                *COEFF_NAMES,
                *[f"{name}_std" for name in COEFF_NAMES],
                "candidate_source",
                "is_novel_vs_train",
            ]
        )

    if uncertainty_seed is not None:
        torch.manual_seed(int(uncertainty_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(uncertainty_seed))

    tokenizer, step1_backbone, _ = load_backbone_from_step1(
        config=config,
        model_size=model_size,
        checkpoint_path=backbone_checkpoint_path,
        device=device,
    )

    reg_ckpt = torch.load(chi_checkpoint_path, map_location=device, weights_only=True)
    reg_state = _normalize_checkpoint_state_dict(reg_ckpt["model_state_dict"])
    reg_finetune_last_layers = int(reg_ckpt.get("finetune_last_layers", 0) or 0)
    reg_timestep = int(reg_ckpt.get("timestep_for_embedding", timestep))

    reg_model = None
    if reg_finetune_last_layers > 0:
        _, reg_backbone, _ = load_backbone_from_step1(
            config=config,
            model_size=model_size,
            checkpoint_path=backbone_checkpoint_path,
            device=device,
        )
        reg_head = PhysicsGuidedChiModel(
            embedding_dim=int(reg_ckpt["embedding_dim"]),
            hidden_sizes=list(reg_ckpt["hidden_sizes"]),
            dropout=float(reg_ckpt["dropout"]),
        )
        reg_model = BackbonePhysicsGuidedChiModel(
            backbone=reg_backbone,
            chi_head=reg_head,
            timestep=reg_timestep,
            pooling=pooling,
        ).to(device)
        reg_model.load_state_dict(reg_state, strict=True)
    else:
        reg_model = PhysicsGuidedChiModel(
            embedding_dim=int(reg_ckpt["embedding_dim"]),
            hidden_sizes=list(reg_ckpt["hidden_sizes"]),
            dropout=float(reg_ckpt["dropout"]),
        ).to(device)
        if isinstance(reg_state, dict) and any(str(k).startswith("chi_head.") for k in reg_state.keys()):
            reg_state = {
                str(k)[len("chi_head."):]: v
                for k, v in reg_state.items()
                if str(k).startswith("chi_head.")
            }
        reg_model.load_state_dict(reg_state, strict=True)
    if mc_enabled:
        reg_model.train()
    else:
        reg_model.eval()

    cls_model = None
    cls_finetune_last_layers = 0
    cls_timestep = int(timestep)
    if class_checkpoint_path is not None and class_checkpoint_path.exists():
        cls_ckpt = torch.load(class_checkpoint_path, map_location=device, weights_only=True)
        cls_state = _normalize_checkpoint_state_dict(cls_ckpt["model_state_dict"])
        cls_finetune_last_layers = int(cls_ckpt.get("finetune_last_layers", 0) or 0)
        cls_timestep = int(cls_ckpt.get("timestep_for_embedding", timestep))
        if cls_finetune_last_layers > 0:
            _, cls_backbone, _ = load_backbone_from_step1(
                config=config,
                model_size=model_size,
                checkpoint_path=backbone_checkpoint_path,
                device=device,
            )
            cls_head = SolubilityClassifier(
                embedding_dim=int(cls_ckpt["embedding_dim"]),
                hidden_sizes=list(cls_ckpt["hidden_sizes"]),
                dropout=float(cls_ckpt["dropout"]),
            )
            cls_model = BackboneSolubilityClassifierModel(
                backbone=cls_backbone,
                classifier_head=cls_head,
                timestep=cls_timestep,
                pooling=pooling,
            ).to(device)
            cls_model.load_state_dict(cls_state, strict=True)
        else:
            cls_model = SolubilityClassifier(
                embedding_dim=int(cls_ckpt["embedding_dim"]),
                hidden_sizes=list(cls_ckpt["hidden_sizes"]),
                dropout=float(cls_ckpt["dropout"]),
            ).to(device)
            if isinstance(cls_state, dict) and any(str(k).startswith("classifier_head.") for k in cls_state.keys()):
                cls_state = {
                    str(k)[len("classifier_head."):]: v
                    for k, v in cls_state.items()
                    if str(k).startswith("classifier_head.")
                }
            cls_model.load_state_dict(cls_state, strict=True)
        if mc_enabled:
            cls_model.train()
        else:
            cls_model.eval()

    if reg_finetune_last_layers > 0:
        warnings.warn(
            (
                f"Loaded Step4 regression checkpoint with finetuned backbone "
                f"(finetune_last_layers={reg_finetune_last_layers}) for novel-candidate inference."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
    if cls_model is not None and cls_finetune_last_layers > 0:
        warnings.warn(
            (
                f"Loaded Step4 classification checkpoint with finetuned backbone "
                f"(finetune_last_layers={cls_finetune_last_layers}) for novel-candidate inference."
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    reg_needs_step1_embeddings = reg_finetune_last_layers == 0
    cls_needs_step1_embeddings = cls_model is not None and cls_finetune_last_layers == 0
    if not (reg_needs_step1_embeddings or cls_needs_step1_embeddings):
        # Avoid keeping an unused backbone resident when both heads are end-to-end finetuned.
        step1_backbone = None

    coeff_mean_list: List[np.ndarray] = []
    coeff_std_list: List[np.ndarray] = []
    logit_mean_list: List[np.ndarray] = []
    logit_std_list: List[np.ndarray] = []
    prob_mean_list: List[np.ndarray] = []
    prob_std_list: List[np.ndarray] = []
    smiles_list = novel_df["SMILES"].astype(str).tolist()
    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i : i + batch_size]
        encoded = tokenizer.batch_encode(batch_smiles)
        input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long, device=device)
        attention_mask = torch.tensor(encoded["attention_mask"], dtype=torch.long, device=device)

        emb_reg = None
        emb_cls = None
        if reg_needs_step1_embeddings:
            t_reg = torch.full((input_ids.shape[0],), reg_timestep, device=device, dtype=torch.long)
            emb_reg = step1_backbone.get_pooled_output(
                input_ids=input_ids,
                timesteps=t_reg,
                attention_mask=attention_mask,
                pooling=pooling,
            )
        if cls_needs_step1_embeddings:
            if reg_needs_step1_embeddings and cls_timestep == reg_timestep:
                emb_cls = emb_reg
            else:
                t_cls = torch.full((input_ids.shape[0],), cls_timestep, device=device, dtype=torch.long)
                emb_cls = step1_backbone.get_pooled_output(
                    input_ids=input_ids,
                    timesteps=t_cls,
                    attention_mask=attention_mask,
                    pooling=pooling,
                )

        t_dummy = torch.full((input_ids.shape[0],), 300.0, dtype=torch.float32, device=device)
        phi_dummy = torch.full((input_ids.shape[0],), 0.5, dtype=torch.float32, device=device)

        def _forward_once() -> Tuple[np.ndarray, np.ndarray]:
            if reg_finetune_last_layers > 0:
                reg_out_local = reg_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    temperature=t_dummy,
                    phi=phi_dummy,
                )
            else:
                reg_out_local = reg_model(
                    embedding=emb_reg,
                    temperature=t_dummy,
                    phi=phi_dummy,
                )
            coeff_local = reg_out_local["coefficients"].detach().cpu().numpy()
            if cls_model is not None:
                if cls_finetune_last_layers > 0:
                    cls_out_local = cls_model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    cls_out_local = cls_model(embedding=emb_cls)
                logit_local = cls_out_local["class_logit"].detach().cpu().numpy()
            else:
                logit_local = reg_out_local["class_logit"].detach().cpu().numpy()
            return coeff_local, logit_local

        if mc_enabled:
            coeff_samples: List[np.ndarray] = []
            logit_samples: List[np.ndarray] = []
            prob_samples: List[np.ndarray] = []
            for _ in range(int(uncertainty_mc_samples)):
                coeff_s, logit_s = _forward_once()
                coeff_samples.append(coeff_s)
                logit_samples.append(logit_s)
                prob_samples.append(stable_sigmoid(logit_s))
            coeff_stack = np.stack(coeff_samples, axis=0)
            logit_stack = np.stack(logit_samples, axis=0)
            prob_stack = np.stack(prob_samples, axis=0)
            coeff_mean_list.append(np.mean(coeff_stack, axis=0))
            coeff_std_list.append(np.std(coeff_stack, axis=0, ddof=0))
            logit_mean_list.append(np.mean(logit_stack, axis=0))
            logit_std_list.append(np.std(logit_stack, axis=0, ddof=0))
            prob_mean_list.append(np.mean(prob_stack, axis=0))
            prob_std_list.append(np.std(prob_stack, axis=0, ddof=0))
        else:
            coeff_det, logit_det = _forward_once()
            prob_det = stable_sigmoid(logit_det)
            coeff_mean_list.append(coeff_det)
            coeff_std_list.append(np.zeros_like(coeff_det))
            logit_mean_list.append(logit_det)
            logit_std_list.append(np.zeros_like(logit_det))
            prob_mean_list.append(prob_det)
            prob_std_list.append(np.zeros_like(prob_det))

    coeff_mean = np.concatenate(coeff_mean_list, axis=0) if coeff_mean_list else np.zeros((0, 6), dtype=float)
    coeff_std = np.concatenate(coeff_std_list, axis=0) if coeff_std_list else np.zeros((0, 6), dtype=float)
    logit_mean = np.concatenate(logit_mean_list, axis=0) if logit_mean_list else np.zeros((0,), dtype=float)
    logit_std = np.concatenate(logit_std_list, axis=0) if logit_std_list else np.zeros((0,), dtype=float)
    prob_mean = np.concatenate(prob_mean_list, axis=0) if prob_mean_list else np.zeros((0,), dtype=float)
    prob_std = np.concatenate(prob_std_list, axis=0) if prob_std_list else np.zeros((0,), dtype=float)

    result = novel_df[["polymer_id", "Polymer", "SMILES", "canonical_smiles"]].copy()
    result["water_soluble"] = -1
    result["class_logit"] = logit_mean
    result["class_logit_std"] = logit_std
    result["class_prob"] = prob_mean
    result["class_prob_std"] = prob_std
    for idx, name in enumerate(COEFF_NAMES):
        result[name] = coeff_mean[:, idx]
        result[f"{name}_std"] = coeff_std[:, idx]
    result["candidate_source"] = "novel_generated"
    result["is_novel_vs_train"] = 1
    return result


def load_soluble_targets(
    targets_csv: str | None,
    results_dir: Path,
    base_results_dir: Path | None,
    split_mode: str,
    target_temperature: float | None = None,
    target_phi: float | None = None,
) -> Tuple[pd.DataFrame, str | None]:
    if targets_csv:
        target_path = Path(targets_csv)
    else:
        candidate_paths = [
            results_dir / "step3_chi_target_learning" / split_mode / "metrics" / "chi_target_for_inverse_design.csv"
        ]
        if base_results_dir is not None:
            base_path = Path(base_results_dir) / "step3_chi_target_learning" / split_mode / "metrics" / "chi_target_for_inverse_design.csv"
            if str(base_path) not in {str(p) for p in candidate_paths}:
                candidate_paths.append(base_path)
        target_path = next((p for p in candidate_paths if p.exists()), candidate_paths[0])
    if not target_path.exists():
        raise FileNotFoundError(
            "Learned χ_target file not found. "
            "Run Step 3 first or provide --targets_csv. "
            f"Expected: {target_path}"
        )
    target_df = pd.read_csv(target_path)

    if "target_class" in target_df.columns:
        target_df = target_df[target_df["target_class"].astype(int) == 1].copy()
    else:
        target_df["target_class"] = 1

    required_cols = {"temperature", "phi", "target_chi"}
    missing = required_cols - set(target_df.columns)
    if missing:
        raise ValueError(f"targets CSV missing required columns: {sorted(missing)}")

    if "target_class_name" not in target_df.columns:
        target_df["target_class_name"] = CLASS_NAME_MAP[1]
    if "property_rule" not in target_df.columns:
        target_df["property_rule"] = "upper_bound"
    if "target_id" not in target_df.columns:
        target_df.insert(0, "target_id", np.arange(1, len(target_df) + 1))

    target_df = target_df.copy()
    target_df["temperature"] = target_df["temperature"].astype(float)
    target_df["phi"] = target_df["phi"].astype(float)

    if target_temperature is not None:
        t_value = float(target_temperature)
        t_mask = np.isclose(target_df["temperature"].to_numpy(dtype=float), t_value, atol=1.0e-8)
        if not np.any(t_mask):
            available_t = sorted(target_df["temperature"].astype(float).unique().tolist())
            raise ValueError(
                f"No targets found for target_temperature={t_value}. Available temperatures: {available_t}"
            )
        target_df = target_df.loc[t_mask].copy()

    if target_phi is not None:
        phi_value = float(target_phi)
        phi_mask = np.isclose(target_df["phi"].to_numpy(dtype=float), phi_value, atol=1.0e-8)
        if not np.any(phi_mask):
            available_phi = sorted(target_df["phi"].astype(float).unique().tolist())
            raise ValueError(
                f"No targets found for target_phi={phi_value}. Available phi values: {available_phi}"
            )
        target_df = target_df.loc[phi_mask].copy()

    target_df = target_df.sort_values(["temperature", "phi"]).reset_index(drop=True)
    if target_df.empty:
        raise ValueError("No water-soluble targets found in χ_target file (class=1).")
    target_df["target_id"] = np.arange(1, len(target_df) + 1)
    return target_df, str(target_path)


def build_candidate_pool(
    args,
    config: Dict,
    chi_cfg: Dict,
    results_dir: Path,
    base_results_dir: Path,
    step4_reg_metrics_dir: Path,
    step4_cls_metrics_dir: Path,
    device: str,
) -> Tuple[pd.DataFrame, Dict[str, object], set[str]]:
    source = parse_candidate_source(args.candidate_source)
    training_canonical = resolve_training_smiles(results_dir, base_results_dir)
    uncertainty_enabled = bool(getattr(args, "uncertainty_enabled", chi_cfg.get("uncertainty_enabled", False)))
    uncertainty_mc_samples = int(
        getattr(args, "uncertainty_mc_samples", chi_cfg.get("uncertainty_mc_samples", 20))
        or chi_cfg.get("uncertainty_mc_samples", 20)
    )
    uncertainty_seed = getattr(args, "uncertainty_seed", chi_cfg.get("uncertainty_seed", None))

    summary: Dict[str, object] = {
        "candidate_source": source,
        "step4_regression_metrics_dir": str(step4_reg_metrics_dir),
        "step4_classification_metrics_dir": str(step4_cls_metrics_dir),
        "uncertainty_enabled": bool(uncertainty_enabled),
        "uncertainty_mc_samples": int(uncertainty_mc_samples),
        "uncertainty_seed": None if uncertainty_seed is None else int(uncertainty_seed),
    }

    pool_frames: List[pd.DataFrame] = []

    if source in {"known", "hybrid"}:
        known_df, known_summary = _load_known_candidates_from_step4_metrics(
            step4_reg_metrics_dir=step4_reg_metrics_dir,
            step4_cls_metrics_dir=step4_cls_metrics_dir,
            training_canonical=training_canonical,
        )
        summary.update(known_summary)
        pool_frames.append(known_df)

    if source in {"novel", "hybrid"}:
        if args.generated_csv:
            generated_csv = Path(args.generated_csv)
        else:
            default_candidates = [
                results_dir / "step2_sampling" / "metrics" / "target_polymers.csv",
                results_dir / "step2_sampling" / "metrics" / "generated_samples.csv",
            ]
            generated_csv = next((p for p in default_candidates if p.exists()), default_candidates[0])

        novel_df, novel_summary = prepare_novel_candidates(
            generated_csv=generated_csv,
            smiles_column=args.generated_smiles_column,
            training_canonical=training_canonical,
            require_two_stars=(not args.allow_non_two_stars),
            max_novel_candidates=args.max_novel_candidates,
        )
        summary.update(novel_summary)
        summary["generated_csv"] = str(generated_csv)

        default_reg_candidates = [
            step4_reg_metrics_dir.parent / "checkpoints" / "chi_regression_best.pt",
            step4_reg_metrics_dir.parent / "checkpoints" / "chi_physics_best.pt",
            results_dir / "checkpoints" / "chi_regression_best.pt",
            results_dir / "checkpoints" / "chi_physics_best.pt",
        ]
        if getattr(args, "step4_checkpoint", None):
            chi_checkpoint = Path(args.step4_checkpoint)
        else:
            chi_checkpoint = next((p for p in default_reg_candidates if p.exists()), default_reg_candidates[0])
        if not chi_checkpoint.exists():
            raise FileNotFoundError(f"Step4 chi checkpoint not found: {chi_checkpoint}")

        if getattr(args, "step4_class_checkpoint", None):
            class_checkpoint = Path(args.step4_class_checkpoint)
        else:
            default_cls_candidates = [
                step4_cls_metrics_dir.parent / "checkpoints" / "chi_classifier_best.pt",
                results_dir / "checkpoints" / "chi_classifier_best.pt",
            ]
            class_checkpoint = next((p for p in default_cls_candidates if p.exists()), default_cls_candidates[0])
        if not class_checkpoint.exists():
            raise FileNotFoundError(
                "Step4 classification checkpoint not found. "
                f"Expected: {class_checkpoint}. "
                "Run Step 4 to produce Step4_2 checkpoint or pass --step4_class_checkpoint."
            )

        novel_coeff_df = infer_coefficients_for_novel_candidates(
            novel_df=novel_df,
            config=config,
            model_size=args.model_size,
            chi_checkpoint_path=chi_checkpoint,
            class_checkpoint_path=class_checkpoint,
            backbone_checkpoint_path=args.backbone_checkpoint,
            device=device,
            timestep=int(chi_cfg.get("embedding_timestep", 1)),
            pooling=args.embedding_pooling,
            batch_size=int(args.embedding_batch_size or chi_cfg.get("embedding_batch_size", 128)),
            uncertainty_enabled=uncertainty_enabled,
            uncertainty_mc_samples=uncertainty_mc_samples,
            uncertainty_seed=uncertainty_seed,
        )
        summary["novel_candidate_count"] = int(len(novel_coeff_df))
        pool_frames.append(novel_coeff_df)

    if not pool_frames:
        empty = pd.DataFrame(
            columns=[
                "polymer_id",
                "Polymer",
                "SMILES",
                "canonical_smiles",
                "water_soluble",
                "class_logit",
                "class_logit_std",
                "class_prob",
                "class_prob_std",
                *COEFF_NAMES,
                *[f"{name}_std" for name in COEFF_NAMES],
                "candidate_source",
                "is_novel_vs_train",
            ]
        )
        summary["candidate_count_total"] = 0
        summary["candidate_count_after_dedup"] = 0
        return empty, summary, training_canonical

    coeff_df = pd.concat(pool_frames, ignore_index=True)
    if "canonical_smiles" not in coeff_df.columns:
        coeff_df["canonical_smiles"] = coeff_df["SMILES"].astype(str).map(canonicalize_smiles)
    coeff_df["canonical_smiles"] = coeff_df["canonical_smiles"].where(
        coeff_df["canonical_smiles"].notna(),
        coeff_df["SMILES"].astype(str),
    )
    if "candidate_source" not in coeff_df.columns:
        coeff_df["candidate_source"] = "unknown"
    if "is_novel_vs_train" not in coeff_df.columns:
        coeff_df["is_novel_vs_train"] = (~coeff_df["canonical_smiles"].isin(training_canonical)).astype(int)
    if "class_prob_std" not in coeff_df.columns:
        coeff_df["class_prob_std"] = 0.0
    if "class_logit_std" not in coeff_df.columns:
        coeff_df["class_logit_std"] = 0.0
    for name in COEFF_NAMES:
        std_col = f"{name}_std"
        if std_col not in coeff_df.columns:
            coeff_df[std_col] = 0.0
    coeff_df["class_prob_std"] = coeff_df["class_prob_std"].fillna(0.0).astype(float)
    coeff_df["class_logit_std"] = coeff_df["class_logit_std"].fillna(0.0).astype(float)
    for name in COEFF_NAMES:
        coeff_df[f"{name}_std"] = coeff_df[f"{name}_std"].fillna(0.0).astype(float)

    summary["candidate_count_total"] = int(len(coeff_df))
    coeff_df = coeff_df.drop_duplicates(subset=["canonical_smiles"], keep="first").reset_index(drop=True)
    coeff_df["source_polymer_id"] = coeff_df.get("polymer_id", pd.Series([-1] * len(coeff_df))).astype(int)
    coeff_df["polymer_id"] = np.arange(len(coeff_df), dtype=int)
    summary["candidate_count_after_dedup"] = int(len(coeff_df))
    return coeff_df, summary, training_canonical
