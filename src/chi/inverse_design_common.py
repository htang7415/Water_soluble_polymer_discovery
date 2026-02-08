"""Common utilities for chi inverse-design steps."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch

from src.chi.embeddings import load_backbone_from_step1
from src.chi.model import PhysicsGuidedChiModel
from src.utils.chemistry import canonicalize_smiles, check_validity, count_stars


CLASS_NAME_MAP = {1: "Water-soluble", 0: "Water-insoluble"}


def default_chi_config(config: Dict) -> Dict:
    chi_cfg = config.get("chi_training", {})
    defaults = {
        "split_mode": "polymer",
        "epsilon": 0.05,
        "class_weight": 0.25,
        "polymer_class_weight": 0.50,
        "candidate_source": "known",
        "property_rule": "upper_bound",
        "coverage_topk": 5,
        "target_polymer_class": "all",
        "embedding_batch_size": 128,
        "embedding_timestep": int(config.get("training_property", {}).get("default_timestep", 1)),
    }
    out = defaults.copy()
    out.update(chi_cfg)
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
    if v in {"known", "step4", "in_dataset"}:
        return "known"
    if v in {"novel", "generated", "step2"}:
        return "novel"
    if v in {"hybrid", "both", "all"}:
        return "hybrid"
    raise ValueError("candidate_source must be one of: known, novel, hybrid")


def check_required_step4_files(step4_metrics_dir: Path) -> None:
    needed = [
        step4_metrics_dir / "chi_predictions_all.csv",
        step4_metrics_dir / "polymer_coefficients.csv",
    ]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Inverse-design step requires Step 4 outputs. Missing files:\n"
            + "\n".join(missing)
        )


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

    canon = []
    for s in train_df[smiles_col].astype(str).tolist():
        c = canonicalize_smiles(s)
        if c is not None:
            canon.append(c)
    return set(canon)


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

    ckpt = torch.load(chi_checkpoint_path, map_location=device, weights_only=False)
    model = PhysicsGuidedChiModel(
        embedding_dim=int(ckpt["embedding_dim"]),
        hidden_sizes=list(ckpt["hidden_sizes"]),
        dropout=float(ckpt["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

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
    out = model(embedding=emb, temperature=t_dummy, phi=phi_dummy)

    coeff = out["coefficients"].detach().cpu().numpy()
    logit = out["class_logit"].detach().cpu().numpy()
    prob = 1.0 / (1.0 + np.exp(-logit))

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
    step4_metrics_dir: Path,
    device: str,
) -> Tuple[pd.DataFrame, Dict[str, object], set[str]]:
    source = parse_candidate_source(args.candidate_source)
    known_df = pd.read_csv(step4_metrics_dir / "polymer_coefficients.csv")
    known_df["candidate_source"] = "known_dataset"
    known_df["canonical_smiles"] = known_df["SMILES"].astype(str).apply(lambda s: canonicalize_smiles(s) or s)

    training_canonical = resolve_training_smiles(results_dir, base_results_dir)
    known_df["is_novel_vs_train"] = (~known_df["canonical_smiles"].isin(training_canonical)).astype(int)

    summary: Dict[str, object] = {
        "candidate_source": source,
        "known_candidate_count": int(len(known_df)),
        "known_novel_count": int(known_df["is_novel_vs_train"].sum()),
    }

    if source == "known":
        return known_df, summary, training_canonical

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

    chi_checkpoint = (
        Path(args.step4_checkpoint)
        if getattr(args, "step4_checkpoint", None)
        else (results_dir / "checkpoints" / "chi_physics_best.pt")
    )
    if not chi_checkpoint.exists():
        raise FileNotFoundError(f"Step4 chi checkpoint not found: {chi_checkpoint}")

    novel_coeff_df = infer_coefficients_for_novel_candidates(
        novel_df=novel_df,
        config=config,
        model_size=args.model_size,
        chi_checkpoint_path=chi_checkpoint,
        backbone_checkpoint_path=args.backbone_checkpoint,
        device=device,
        timestep=int(chi_cfg.get("embedding_timestep", 1)),
        pooling=args.embedding_pooling,
        batch_size=int(args.embedding_batch_size or chi_cfg.get("embedding_batch_size", 128)),
    )
    summary["novel_candidate_count"] = int(len(novel_coeff_df))

    if source == "novel":
        return novel_coeff_df, summary, training_canonical

    if not novel_coeff_df.empty:
        novel_coeff_df = novel_coeff_df.copy()
        offset = int(known_df["polymer_id"].max()) + 1
        novel_coeff_df["polymer_id"] = novel_coeff_df["polymer_id"].astype(int) + offset
    merged = pd.concat([known_df, novel_coeff_df], ignore_index=True)
    summary["hybrid_candidate_count"] = int(len(merged))
    return merged, summary, training_canonical
