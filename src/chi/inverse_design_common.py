"""Common utilities for chi inverse-design steps."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch

from src.chi.embeddings import load_backbone_from_step1
from src.chi.model import PhysicsGuidedChiModel, SolubilityClassifier
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
        "class_weight": 0.25,
        "polymer_class_weight": 0.50,
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
    raise ValueError("candidate_source supports only 'novel' (aliases: generated, step2)")


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
) -> pd.DataFrame:
    if novel_df.empty:
        return pd.DataFrame(
            columns=[
                "polymer_id",
                "Polymer",
                "SMILES",
                "canonical_smiles",
                "water_soluble",
                "class_logit",
                "class_prob",
                "a0",
                "a1",
                "a2",
                "a3",
                "b1",
                "b2",
                "candidate_source",
                "is_novel_vs_train",
            ]
        )

    tokenizer, backbone, _ = load_backbone_from_step1(
        config=config,
        model_size=model_size,
        checkpoint_path=backbone_checkpoint_path,
        device=device,
    )

    ckpt = torch.load(chi_checkpoint_path, map_location=device, weights_only=True)
    finetune_last_layers = int(ckpt.get("finetune_last_layers", 0) or 0)
    if finetune_last_layers > 0:
        warnings.warn(
            (
                f"Step4 checkpoint {chi_checkpoint_path} was trained with "
                f"finetune_last_layers={finetune_last_layers}. "
                "Novel-candidate inference uses the Step1 backbone encoder by design, "
                "so Step4 finetuned backbone weights are not applied here."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
    reg_model = PhysicsGuidedChiModel(
        embedding_dim=int(ckpt["embedding_dim"]),
        hidden_sizes=list(ckpt["hidden_sizes"]),
        dropout=float(ckpt["dropout"]),
    ).to(device)
    reg_state = ckpt["model_state_dict"]
    if isinstance(reg_state, dict) and any(str(k).startswith("chi_head.") for k in reg_state.keys()):
        reg_state = {
            str(k)[len("chi_head."):]: v
            for k, v in reg_state.items()
            if str(k).startswith("chi_head.")
        }
    reg_model.load_state_dict(reg_state, strict=True)
    reg_model.eval()

    cls_model = None
    if class_checkpoint_path is not None and class_checkpoint_path.exists():
        cls_ckpt = torch.load(class_checkpoint_path, map_location=device, weights_only=True)
        cls_finetune_last_layers = int(cls_ckpt.get("finetune_last_layers", 0) or 0)
        if cls_finetune_last_layers > 0:
            warnings.warn(
                (
                    f"Step4 classifier checkpoint {class_checkpoint_path} was trained with "
                    f"finetune_last_layers={cls_finetune_last_layers}. "
                    "Novel-candidate inference uses Step1 backbone embeddings by design, "
                    "so Step4 finetuned classifier-backbone weights are not applied here."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
        cls_model = SolubilityClassifier(
            embedding_dim=int(cls_ckpt["embedding_dim"]),
            hidden_sizes=list(cls_ckpt["hidden_sizes"]),
            dropout=float(cls_ckpt["dropout"]),
        ).to(device)
        cls_state = cls_ckpt["model_state_dict"]
        if isinstance(cls_state, dict) and any(str(k).startswith("classifier_head.") for k in cls_state.keys()):
            cls_state = {
                str(k)[len("classifier_head."):]: v
                for k, v in cls_state.items()
                if str(k).startswith("classifier_head.")
            }
        cls_model.load_state_dict(cls_state, strict=True)
        cls_model.eval()

    embeddings = []
    smiles_list = novel_df["SMILES"].astype(str).tolist()
    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i : i + batch_size]
        encoded = tokenizer.batch_encode(batch_smiles)
        input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long, device=device)
        attention_mask = torch.tensor(encoded["attention_mask"], dtype=torch.long, device=device)
        t = torch.full((input_ids.shape[0],), int(timestep), device=device, dtype=torch.long)

        pooled = backbone.get_pooled_output(
            input_ids=input_ids,
            timesteps=t,
            attention_mask=attention_mask,
            pooling=pooling,
        )
        embeddings.append(pooled.detach().cpu())

    emb = torch.cat(embeddings, dim=0).to(device)
    t_dummy = torch.full((emb.shape[0],), 300.0, dtype=torch.float32, device=device)
    phi_dummy = torch.full((emb.shape[0],), 0.5, dtype=torch.float32, device=device)
    out = reg_model(embedding=emb, temperature=t_dummy, phi=phi_dummy)

    coeff = out["coefficients"].detach().cpu().numpy()
    if cls_model is not None:
        cls_out = cls_model(embedding=emb)
        logit = cls_out["class_logit"].detach().cpu().numpy()
        prob = stable_sigmoid(logit)
    else:
        logit = out["class_logit"].detach().cpu().numpy()
        prob = stable_sigmoid(logit)

    result = novel_df[["polymer_id", "Polymer", "SMILES", "canonical_smiles"]].copy()
    result["water_soluble"] = -1
    result["class_logit"] = logit
    result["class_prob"] = prob
    for idx, name in enumerate(["a0", "a1", "a2", "a3", "b1", "b2"]):
        result[name] = coeff[:, idx]
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

    summary: Dict[str, object] = {
        "candidate_source": source,
        "step4_regression_metrics_dir": str(step4_reg_metrics_dir),
        "step4_classification_metrics_dir": str(step4_cls_metrics_dir),
    }

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
    )
    summary["novel_candidate_count"] = int(len(novel_coeff_df))
    return novel_coeff_df, summary, training_canonical
