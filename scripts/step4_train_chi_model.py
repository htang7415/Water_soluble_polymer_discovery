#!/usr/bin/env python
"""Step 4: split training with Step4_1 regression and Step4_2 classification."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chi.data import (
    COEFF_NAMES,
    SplitConfig,
    add_split_column,
    fill_missing_polymer_names_from_smiles,
    load_chi_dataset,
    make_split_assignments,
)
from src.chi.embeddings import (
    build_or_load_embedding_cache,
    embedding_table_from_cache,
    load_backbone_from_step1,
)
from src.chi.metrics import (
    classification_metrics,
    hit_metrics,
    metrics_by_group,
    polymer_balanced_nrmse,
    polymer_r2_distribution,
    regression_metrics,
)
from src.chi.model import DirectChiRegressor, SolubilityClassifier
from src.utils.config import load_step4_config, save_config
from src.utils.figure_style import apply_publication_figure_style
from src.utils.model_scales import get_model_config, get_results_dir
from src.utils.numerics import stable_sigmoid
from src.utils.reproducibility import save_run_metadata, seed_everything
from src.utils.reporting import save_step_summary, save_artifact_manifest, write_initial_log


WATER_SOLUBLE_PALETTE = {0: "#d62728", 1: "#1f77b4"}
CLASS_LABEL_INTERNAL = "water_miscible"
CLASS_LABEL_PUBLIC = "water_miscible"
CLASS_DISPLAY_LABELS = {0: "water-immiscible", 1: "water-miscible"}
CLASS_DISPLAY_TICKLABELS = ["water-\nimmiscible", "water-\nmiscible"]
CLASS_DISPLAY_LEGEND_TITLE = "Class"


def _class_display_label(value: object) -> str:
    try:
        class_value = int(float(value))
    except Exception:
        return str(value)
    return CLASS_DISPLAY_LABELS.get(class_value, str(value))


def _class_display_labels(values: Iterable[object]) -> List[str]:
    return [_class_display_label(value) for value in values]


@dataclass
class TrainConfig:
    split_mode: str
    holdout_test_ratio: float
    seed: int
    batch_size: int
    num_epochs: int
    patience: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    use_scheduler: bool
    scheduler_min_lr: float
    hidden_sizes: List[int]
    dropout: float
    tune: bool
    n_trials: int
    tuning_epochs: int
    tuning_patience: int
    tuning_objective: str
    tuning_cv_folds: int
    budget_search_epochs: int
    budget_search_patience: int
    epoch_selection_metric: str
    loss_weighting: str
    loss_weight_clip_ratio: float
    scheduler_t_max: Optional[int]
    nrmse_std_floor: float
    nrmse_clip: float
    timestep_for_embedding: int
    finetune_last_layers: int
    optuna_search_space: Dict[str, object]


class ChiDataset(Dataset):
    """Row-level chi dataset with cached polymer embeddings."""

    def __init__(self, df: pd.DataFrame, embedding_table: np.ndarray, weights: Optional[np.ndarray] = None):
        self.df = df.reset_index(drop=True).copy()
        self.embedding_table = embedding_table.astype(np.float32)

        polymer_ids = self.df["polymer_id"].to_numpy(dtype=np.int64)
        self.embedding = torch.tensor(self.embedding_table[polymer_ids], dtype=torch.float32)
        self.temperature = torch.tensor(self.df["temperature"].to_numpy(dtype=np.float32), dtype=torch.float32)
        self.phi = torch.tensor(self.df["phi"].to_numpy(dtype=np.float32), dtype=torch.float32)
        self.chi = torch.tensor(self.df["chi"].to_numpy(dtype=np.float32), dtype=torch.float32)
        self.label = torch.tensor(self.df["water_miscible"].to_numpy(dtype=np.float32), dtype=torch.float32)
        if weights is None:
            weight_array = np.ones(len(self.df), dtype=np.float32)
        else:
            weight_array = np.asarray(weights, dtype=np.float32).reshape(-1)
            if len(weight_array) != len(self.df):
                raise ValueError(f"ChiDataset weights length mismatch: expected {len(self.df)}, got {len(weight_array)}")
        self.weight = torch.tensor(weight_array, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "embedding": self.embedding[idx],
            "temperature": self.temperature[idx],
            "phi": self.phi[idx],
            "chi": self.chi[idx],
            "label": self.label[idx],
            "weight": self.weight[idx],
        }


class ChiTokenDataset(Dataset):
    """Row-level chi dataset with tokenized polymer SMILES for backbone finetuning."""

    def __init__(self, df: pd.DataFrame, tokenizer, weights: Optional[np.ndarray] = None):
        self.df = df.reset_index(drop=True).copy()
        encoded = tokenizer.batch_encode(self.df["SMILES"].astype(str).tolist())
        self.input_ids = torch.tensor(np.asarray(encoded["input_ids"], dtype=np.int64), dtype=torch.long)
        self.attention_mask = torch.tensor(np.asarray(encoded["attention_mask"], dtype=np.int64), dtype=torch.long)
        self.temperature = torch.tensor(self.df["temperature"].to_numpy(dtype=np.float32), dtype=torch.float32)
        self.phi = torch.tensor(self.df["phi"].to_numpy(dtype=np.float32), dtype=torch.float32)
        self.chi = torch.tensor(self.df["chi"].to_numpy(dtype=np.float32), dtype=torch.float32)
        self.label = torch.tensor(self.df["water_miscible"].to_numpy(dtype=np.float32), dtype=torch.float32)
        if weights is None:
            weight_array = np.ones(len(self.df), dtype=np.float32)
        else:
            weight_array = np.asarray(weights, dtype=np.float32).reshape(-1)
            if len(weight_array) != len(self.df):
                raise ValueError(
                    f"ChiTokenDataset weights length mismatch: expected {len(self.df)}, got {len(weight_array)}"
                )
        self.weight = torch.tensor(weight_array, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "temperature": self.temperature[idx],
            "phi": self.phi[idx],
            "chi": self.chi[idx],
            "label": self.label[idx],
            "weight": self.weight[idx],
        }


class BackboneDirectChiModel(nn.Module):
    """End-to-end Step 4 model: backbone encoder + direct chi regression head."""

    def __init__(
        self,
        backbone: nn.Module,
        chi_head: DirectChiRegressor,
        timestep: int,
        pooling: str = "mean",
    ):
        super().__init__()
        self.backbone = backbone
        self.chi_head = chi_head
        self.timestep = int(timestep)
        self.pooling = pooling

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size = int(input_ids.shape[0])
        timesteps = torch.full((batch_size,), self.timestep, device=input_ids.device, dtype=torch.long)
        return self.backbone.get_pooled_output(
            input_ids=input_ids,
            timesteps=timesteps,
            attention_mask=attention_mask,
            pooling=self.pooling,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: torch.Tensor,
        phi: torch.Tensor,
    ):
        embedding = self._encode(input_ids=input_ids, attention_mask=attention_mask)
        return self.chi_head(embedding=embedding, temperature=temperature, phi=phi)

class BackboneSolubilityClassifierModel(nn.Module):
    """End-to-end Step 4_2 model: backbone encoder + solubility classifier head."""

    def __init__(
        self,
        backbone: nn.Module,
        classifier_head: SolubilityClassifier,
        timestep: int,
        pooling: str = "mean",
    ):
        super().__init__()
        self.backbone = backbone
        self.classifier_head = classifier_head
        self.timestep = int(timestep)
        self.pooling = pooling

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size = int(input_ids.shape[0])
        timesteps = torch.full((batch_size,), self.timestep, device=input_ids.device, dtype=torch.long)
        return self.backbone.get_pooled_output(
            input_ids=input_ids,
            timesteps=timesteps,
            attention_mask=attention_mask,
            pooling=self.pooling,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        embedding = self._encode(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier_head(embedding=embedding)

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        class_label: torch.Tensor,
    ):
        embedding = self._encode(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier_head.compute_loss(embedding=embedding, class_label=class_label)


def _set_finetune_last_layers(backbone: nn.Module, finetune_last_layers: int) -> int:
    n_layers = int(len(backbone.layers))
    if finetune_last_layers < 0 or finetune_last_layers > n_layers:
        raise ValueError(
            f"finetune_last_layers must be in [0, {n_layers}] for this backbone, got {finetune_last_layers}"
        )
    for p in backbone.parameters():
        p.requires_grad = False
    if finetune_last_layers > 0:
        for layer in backbone.layers[-finetune_last_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in backbone.final_norm.parameters():
            p.requires_grad = True
    return n_layers


def _resolve_optuna_search_space(config: Dict, chi_cfg: Dict) -> Dict[str, object]:
    """Resolve Step 4 Optuna search space with legacy fallback support."""
    defaults: Dict[str, object] = {
        "num_layers": [1, 2, 3],
        "hidden_units": [64, 128, 256, 512],
        "dropout": [0.0, 0.1, 0.2, 0.3],
        "learning_rate": [1e-4, 5e-3],   # continuous range when len==2
        "learning_rate_log": True,
        "weight_decay": [1e-7, 1e-3],    # continuous range when len==2
        "weight_decay_log": True,
        "batch_size": [16, 32, 64, 128, 256],
    }

    # Preferred: chi_training.step4_1_regression.optuna_search_space
    step41_cfg = chi_cfg.get("step4_1_regression", {}) if isinstance(chi_cfg.get("step4_1_regression", {}), dict) else {}
    user_space = step41_cfg.get("optuna_search_space", {})
    if not isinstance(user_space, dict) or len(user_space) == 0:
        user_space = chi_cfg.get("optuna_search_space", {})
    if not isinstance(user_space, dict):
        user_space = {}

    # Backward compatibility: hyperparameter_tuning.search_space
    if not user_space:
        legacy = config.get("hyperparameter_tuning", {}).get("search_space", {})
        if isinstance(legacy, dict) and legacy:
            mapped = {}
            if "num_layers" in legacy:
                mapped["num_layers"] = legacy["num_layers"]
            if "neurons" in legacy:
                mapped["hidden_units"] = legacy["neurons"]
            if "dropout" in legacy:
                mapped["dropout"] = legacy["dropout"]
            if "learning_rate" in legacy:
                mapped["learning_rate"] = legacy["learning_rate"]
            if "batch_size" in legacy:
                mapped["batch_size"] = legacy["batch_size"]
            user_space = mapped

    out = defaults.copy()
    for key, val in user_space.items():
        out[key] = val
    return out


def _normalize_tuning_objective(value: str) -> str:
    v = str(value).strip().lower()
    if v in {"val_r2", "r2", "maximize_val_r2"}:
        return "val_r2"
    if v in {"val_poly_nrmse", "poly_nrmse", "minimize_val_poly_nrmse"}:
        return "val_poly_nrmse"
    raise ValueError("chi_training.step4_1_regression.tuning_objective must be one of {'val_r2','val_poly_nrmse'}")


def _normalize_epoch_selection_metric(value: str) -> str:
    v = str(value).strip().lower()
    if v in {"val_rmse", "rmse"}:
        return "val_rmse"
    if v in {"val_poly_nrmse", "poly_nrmse"}:
        return "val_poly_nrmse"
    raise ValueError(
        "chi_training.step4_1_regression.epoch_selection_metric must be one of {'val_rmse','val_poly_nrmse'}"
    )


def _tuning_objective_direction(name: str) -> str:
    normalized = _normalize_tuning_objective(name)
    if normalized == "val_r2":
        return "maximize"
    if normalized == "val_poly_nrmse":
        return "minimize"
    raise ValueError(f"Unsupported tuning objective: {name}")


def _metric_is_lower_better(name: str) -> bool:
    normalized = str(name).strip().lower()
    if normalized in {"val_rmse", "rmse", "val_poly_nrmse", "poly_nrmse"}:
        return True
    if normalized in {"val_r2", "r2"}:
        return False
    raise ValueError(f"Unsupported metric comparator for Step4_1: {name}")


def _describe_tuning_objective(train_cfg: TrainConfig) -> str:
    if train_cfg.tuning_objective == "val_r2":
        return "maximize_val_r2"
    if train_cfg.tuning_objective == "val_poly_nrmse":
        return "minimize_val_poly_nrmse"
    return str(train_cfg.tuning_objective)


def _normalize_water_soluble_column(df: pd.DataFrame) -> pd.DataFrame:
    label_aliases = {
        "water_soluble",
        "water_solubel",
        "water_solubility",
        "water_miscible",
        "water miscible",
        "watermiscible",
        "water_missible",
    }
    out = df.copy()

    matched = []
    for col in out.columns:
        key = str(col).strip().lower()
        if key in label_aliases:
            matched.append(col)

    if not matched:
        return out

    if CLASS_LABEL_INTERNAL not in out.columns:
        primary = matched[0]
        if primary != CLASS_LABEL_INTERNAL:
            out = out.rename(columns={primary: CLASS_LABEL_INTERNAL})
        matched = [CLASS_LABEL_INTERNAL] + [c for c in matched if c != primary]

    for col in matched:
        if col == CLASS_LABEL_INTERNAL or col not in out.columns:
            continue
        out[CLASS_LABEL_INTERNAL] = out[CLASS_LABEL_INTERNAL].where(
            out[CLASS_LABEL_INTERNAL].notna(),
            out[col],
        )

    # Public-facing alias for updated label naming in downstream tables/plots.
    out[CLASS_LABEL_PUBLIC] = out[CLASS_LABEL_INTERNAL]
    return out


def _resolve_classification_dataset_paths(csv_path: str | Path | List[str] | Tuple[str, ...]) -> List[Path]:
    specs: List[str] = []
    if isinstance(csv_path, (list, tuple)):
        specs = [str(x).strip() for x in csv_path if str(x).strip()]
    else:
        raw = str(csv_path).strip()
        if len(raw) == 0:
            raise ValueError("Step4_2 classification dataset path is empty.")
        if "," in raw:
            specs = [x.strip() for x in raw.split(",") if x.strip()]
        else:
            specs = [raw]

    paths: List[Path] = []
    for spec in specs:
        p = Path(spec)
        if p.is_dir():
            csvs = sorted(q for q in p.glob("*.csv") if q.is_file())
            if len(csvs) == 0:
                raise FileNotFoundError(f"No CSV files found under Step4_2 dataset directory: {p}")
            paths.extend(csvs)
        else:
            if not p.exists():
                raise FileNotFoundError(f"Step4_2 classification dataset not found: {p}")
            paths.append(p)

    if len(paths) == 0:
        raise ValueError("No classification dataset CSV paths resolved for Step4_2.")
    return paths


def _serialize_path_spec(value: object) -> object:
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return str(value)


def _load_step42_classification_dataset(
    csv_path: str | Path | List[str] | Tuple[str, ...],
    default_temperature: float = 293.15,
    default_phi: float = 0.2,
    default_chi: float = 0.0,
) -> pd.DataFrame:
    """Load Step4_2 classification dataset and normalize required columns.

    Step4_2 now supports one or more classification-only CSV files that may only include
    Polymer/SMILES/water_miscible. Missing physics columns are filled with
    deterministic defaults so existing dataloader interfaces remain unchanged.
    """
    csv_paths = _resolve_classification_dataset_paths(csv_path)
    df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    df = _normalize_water_soluble_column(df)

    required_base = {"Polymer", "SMILES", CLASS_LABEL_INTERNAL}
    missing_base = required_base - set(df.columns)
    if missing_base:
        raise ValueError(
            f"Step4_2 classification dataset is missing required columns: {sorted(missing_base)}"
        )

    out = df.copy()
    out = fill_missing_polymer_names_from_smiles(out, source_name="Step4_2 classification dataset")
    if "temperature" not in out.columns:
        out["temperature"] = float(default_temperature)
    if "phi" not in out.columns:
        out["phi"] = float(default_phi)
    if "chi" not in out.columns:
        out["chi"] = float(default_chi)

    out["temperature"] = out["temperature"].fillna(float(default_temperature)).astype(float)
    out["phi"] = out["phi"].fillna(float(default_phi)).astype(float)
    out["chi"] = out["chi"].fillna(float(default_chi)).astype(float)
    out[CLASS_LABEL_INTERNAL] = pd.to_numeric(out[CLASS_LABEL_INTERNAL], errors="coerce").fillna(0).astype(int)
    out[CLASS_LABEL_PUBLIC] = out[CLASS_LABEL_INTERNAL]

    polymer_order = sorted(out["Polymer"].astype(str).unique())
    polymer_to_id = {p: i for i, p in enumerate(polymer_order)}
    out["polymer_id"] = out["Polymer"].map(polymer_to_id).astype(int)

    out = out.reset_index(drop=True)
    out["row_id"] = out.index.astype(int)
    return out


def _default_chi_config(config: Dict) -> Dict:
    chi_cfg = config.get("chi_training", {})
    shared = chi_cfg.get("shared", {}) if isinstance(chi_cfg.get("shared", {}), dict) else {}
    split_cfg = shared.get("split", {}) if isinstance(shared.get("split", {}), dict) else {}
    embedding_cfg = shared.get("embedding", {}) if isinstance(shared.get("embedding", {}), dict) else {}
    step41_cfg = (
        chi_cfg.get("step4_1_regression", {})
        if isinstance(chi_cfg.get("step4_1_regression", {}), dict)
        else {}
    )
    step42_cfg = (
        chi_cfg.get("step4_2_classification", {})
        if isinstance(chi_cfg.get("step4_2_classification", {}), dict)
        else {}
    )

    defaults = {
        "dataset_path": "Data/chi/polymers_T_phi.csv",
        "step4_2_dataset_path": [
            "Data/water_solvent/water_miscible_polymer.csv",
            "Data/water_solvent/water_immiscible_polymer.csv",
        ],
        "split_mode": "polymer",
        "classification_split_mode": "random",
        "holdout_test_ratio": None,
        "batch_size": 128,
        "num_epochs": 500,
        "patience": 60,
        "learning_rate": 1.0e-3,
        "weight_decay": 1.0e-5,
        "gradient_clip_norm": 1.0,
        "use_scheduler": True,
        "scheduler_min_lr": 1.0e-6,
        "hidden_sizes": [256, 128],
        "dropout": 0.1,
        "tune": True,
        "n_trials": 50,
        "tuning_epochs": 120,
        "tuning_patience": 20,
        "tuning_objective": "val_r2",
        "tuning_cv_folds": 6,
        "budget_search_epochs": 120,
        "budget_search_patience": 20,
        "epoch_selection_metric": "val_rmse",
        "loss_weighting": "uniform",
        "loss_weight_clip_ratio": 10.0,
        "scheduler_t_max": None,
        "nrmse_std_floor": 0.02,
        "nrmse_clip": 10.0,
        "embedding_batch_size": 128,
        "embedding_timestep": int(config.get("training_property", {}).get("default_timestep", 1)),
        "finetune_last_layers": int(config.get("training_property", {}).get("finetune_last_layers", 0)),
    }
    out = defaults.copy()

    shared_dataset_path = shared.get("dataset_path", chi_cfg.get("dataset_path", defaults["dataset_path"]))
    out["dataset_path"] = str(shared_dataset_path)
    out["step4_1_dataset_path"] = str(step41_cfg.get("dataset_path", shared_dataset_path))
    step42_dataset_fallback = chi_cfg.get(
        "step4_2_dataset_path",
        chi_cfg.get("classification_dataset_path", defaults["step4_2_dataset_path"]),
    )
    out["step4_2_dataset_path"] = step42_cfg.get("dataset_path", step42_dataset_fallback)
    out["split_mode"] = str(shared.get("split_mode", chi_cfg.get("split_mode", defaults["split_mode"])))
    out["classification_split_mode"] = str(
        shared.get(
            "classification_split_mode",
            chi_cfg.get("classification_split_mode", defaults["classification_split_mode"]),
        )
    )
    legacy_test_ratio = split_cfg.get("test_ratio", chi_cfg.get("test_ratio", None))
    out["holdout_test_ratio"] = split_cfg.get(
        "holdout_test_ratio",
        chi_cfg.get(
            "holdout_test_ratio",
            legacy_test_ratio if legacy_test_ratio is not None else defaults["holdout_test_ratio"],
        ),
    )
    out["embedding_batch_size"] = int(
        embedding_cfg.get("batch_size", chi_cfg.get("embedding_batch_size", defaults["embedding_batch_size"]))
    )
    out["embedding_timestep"] = int(
        embedding_cfg.get("timestep", chi_cfg.get("embedding_timestep", defaults["embedding_timestep"]))
    )

    # Step 4_1 (regression) config with legacy flat-key fallback.
    for key in [
        "batch_size",
        "num_epochs",
        "patience",
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "use_scheduler",
        "scheduler_min_lr",
        "hidden_sizes",
        "dropout",
        "tune",
        "n_trials",
        "tuning_epochs",
        "tuning_patience",
        "tuning_objective",
        "tuning_cv_folds",
        "budget_search_epochs",
        "budget_search_patience",
        "epoch_selection_metric",
        "loss_weighting",
        "loss_weight_clip_ratio",
        "scheduler_t_max",
        "nrmse_std_floor",
        "nrmse_clip",
        "finetune_last_layers",
    ]:
        out[key] = step41_cfg.get(key, chi_cfg.get(key, defaults[key]))

    out["optuna_search_space"] = _resolve_optuna_search_space(config, chi_cfg)

    step42_defaults = {
        "batch_size": int(out["batch_size"]),
        "num_epochs": int(out["num_epochs"]),
        "patience": int(out["patience"]),
        "learning_rate": float(out["learning_rate"]),
        "weight_decay": float(out["weight_decay"]),
        "gradient_clip_norm": float(out["gradient_clip_norm"]),
        "use_scheduler": bool(out["use_scheduler"]),
        "scheduler_min_lr": float(out["scheduler_min_lr"]),
        "hidden_sizes": [int(v) for v in out["hidden_sizes"]],
        "dropout": float(out["dropout"]),
        "tune": bool(out["tune"]),
        "n_trials": int(out["n_trials"]),
        "tuning_epochs": int(out["tuning_epochs"]),
        "tuning_patience": int(out["tuning_patience"]),
        "tuning_cv_folds": int(out["tuning_cv_folds"]),
        "finetune_last_layers": 0,
        "optuna_search_space": dict(out["optuna_search_space"]),
    }
    step42_out = step42_defaults.copy()
    step42_out.update(step42_cfg)
    if not isinstance(step42_out.get("optuna_search_space", {}), dict):
        step42_out["optuna_search_space"] = dict(out["optuna_search_space"])
    out["step4_2"] = step42_out
    return out



def build_train_config(args, config: Dict) -> TrainConfig:
    chi_cfg = _default_chi_config(config)
    split_mode = str(args.split_mode if args.split_mode is not None else chi_cfg["split_mode"]).strip().lower()
    if split_mode not in {"polymer", "random"}:
        raise ValueError("chi_training.split_mode must be one of {'polymer','random'}")
    tune_cfg = bool(chi_cfg.get("tune", False))
    tune_flag = bool(args.tune or (tune_cfg and not args.no_tune))
    tuning_objective = _normalize_tuning_objective(
        args.tuning_objective if args.tuning_objective is not None else chi_cfg.get("tuning_objective", "val_r2")
    )
    tuning_cv_folds = int(args.tuning_cv_folds if args.tuning_cv_folds is not None else chi_cfg.get("tuning_cv_folds", 6))
    if tuning_cv_folds < 2:
        raise ValueError("chi_training.tuning_cv_folds must be >= 2")
    epoch_selection_metric = _normalize_epoch_selection_metric(chi_cfg.get("epoch_selection_metric", "val_rmse"))
    # CV-driven defaults. Users can still override holdout via:
    # chi_training.shared.split.holdout_test_ratio.
    holdout_test_ratio_raw = chi_cfg.get("holdout_test_ratio", None)
    holdout_test_ratio = (
        float(holdout_test_ratio_raw)
        if holdout_test_ratio_raw is not None
        else (1.0 / float(tuning_cv_folds))
    )
    if not (0.0 < holdout_test_ratio < 1.0):
        raise ValueError("chi_training.shared.split.holdout_test_ratio must be in (0, 1)")
    dev_ratio = 1.0 - holdout_test_ratio
    if dev_ratio <= 0.0:
        raise ValueError("holdout_test_ratio leaves no development data")
    gradient_clip_norm = float(chi_cfg.get("gradient_clip_norm", 1.0))
    if gradient_clip_norm < 0:
        raise ValueError("chi_training.gradient_clip_norm must be >= 0")
    scheduler_min_lr = float(chi_cfg.get("scheduler_min_lr", 1.0e-6))
    if scheduler_min_lr < 0:
        raise ValueError("chi_training.scheduler_min_lr must be >= 0")
    loss_weighting = str(chi_cfg.get("loss_weighting", "uniform")).strip().lower()
    if loss_weighting not in {"uniform", "polymer_balanced"}:
        raise ValueError("chi_training.step4_1_regression.loss_weighting must be one of {'uniform','polymer_balanced'}")
    loss_weight_clip_ratio = float(chi_cfg.get("loss_weight_clip_ratio", 10.0))
    if loss_weight_clip_ratio <= 0:
        raise ValueError("chi_training.step4_1_regression.loss_weight_clip_ratio must be > 0")
    budget_search_epochs = int(chi_cfg.get("budget_search_epochs", 120))
    budget_search_patience = int(chi_cfg.get("budget_search_patience", 20))
    if budget_search_epochs < 1:
        raise ValueError("chi_training.step4_1_regression.budget_search_epochs must be >= 1")
    if budget_search_patience < 0:
        raise ValueError("chi_training.step4_1_regression.budget_search_patience must be >= 0")
    if budget_search_patience > budget_search_epochs:
        raise ValueError("chi_training.step4_1_regression.budget_search_patience must be <= budget_search_epochs")
    nrmse_std_floor = float(chi_cfg.get("nrmse_std_floor", 0.02))
    if nrmse_std_floor <= 0:
        raise ValueError("chi_training.step4_1_regression.nrmse_std_floor must be > 0")
    nrmse_clip = float(chi_cfg.get("nrmse_clip", 10.0))
    if nrmse_clip <= 0:
        raise ValueError("chi_training.step4_1_regression.nrmse_clip must be > 0")
    scheduler_t_max_raw = chi_cfg.get("scheduler_t_max", None)
    scheduler_t_max = int(scheduler_t_max_raw) if scheduler_t_max_raw is not None else None
    if scheduler_t_max is not None and scheduler_t_max < 1:
        raise ValueError("chi_training.step4_1_regression.scheduler_t_max must be >= 1 when provided")

    return TrainConfig(
        split_mode=split_mode,
        holdout_test_ratio=holdout_test_ratio,
        seed=int(config["data"]["random_seed"]),
        batch_size=int(chi_cfg["batch_size"]),
        num_epochs=int(chi_cfg["num_epochs"]),
        patience=int(chi_cfg["patience"]),
        learning_rate=float(chi_cfg["learning_rate"]),
        weight_decay=float(chi_cfg["weight_decay"]),
        gradient_clip_norm=gradient_clip_norm,
        use_scheduler=bool(chi_cfg.get("use_scheduler", True)),
        scheduler_min_lr=scheduler_min_lr,
        hidden_sizes=[int(x) for x in chi_cfg["hidden_sizes"]],
        dropout=float(chi_cfg["dropout"]),
        tune=tune_flag,
        n_trials=int(args.n_trials or chi_cfg.get("n_trials", 50)),
        tuning_epochs=int(chi_cfg.get("tuning_epochs", 80)),
        tuning_patience=int(chi_cfg.get("tuning_patience", 15)),
        tuning_objective=tuning_objective,
        tuning_cv_folds=tuning_cv_folds,
        budget_search_epochs=budget_search_epochs,
        budget_search_patience=budget_search_patience,
        epoch_selection_metric=epoch_selection_metric,
        loss_weighting=loss_weighting,
        loss_weight_clip_ratio=loss_weight_clip_ratio,
        scheduler_t_max=scheduler_t_max,
        nrmse_std_floor=nrmse_std_floor,
        nrmse_clip=nrmse_clip,
        timestep_for_embedding=int(chi_cfg.get("embedding_timestep", 1)),
        finetune_last_layers=int(chi_cfg.get("finetune_last_layers", 0)),
        optuna_search_space=dict(chi_cfg.get("optuna_search_space", {})),
    )



def _resolve_split_ratios(train_cfg: TrainConfig) -> Dict[str, float]:
    """Resolve concrete train/val/test ratios from holdout test plus internal val split."""
    test_ratio = float(train_cfg.holdout_test_ratio)
    dev_ratio = 1.0 - test_ratio
    # Validation slice is internal only (for early stopping / logging); final fit uses train+val.
    val_ratio = dev_ratio / float(max(2, int(train_cfg.tuning_cv_folds)))
    train_ratio = dev_ratio - val_ratio
    resolved = {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Resolved split ratios must sum to 1.0, got {resolved}")
    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError(f"Resolved split ratios must all be > 0, got {resolved}")
    return resolved


def _build_final_fit_split_df(split_df: pd.DataFrame) -> pd.DataFrame:
    """Use all non-test rows for final model fitting after CV tuning."""
    out = split_df.copy()
    out.loc[out["split"] == "val", "split"] = "train"
    return out


def _compute_polymer_weights(split_df: pd.DataFrame, eps: float = 1.0e-4, clip_ratio: float = 10.0) -> np.ndarray:
    weights = np.ones(len(split_df), dtype=np.float32)
    train_mask = split_df["split"].to_numpy() == "train"
    if not np.any(train_mask):
        return weights

    train_df = split_df.loc[train_mask, ["polymer_id", "chi"]].copy()
    stats = train_df.groupby("polymer_id")["chi"].agg(["count", "var"]).reset_index()
    stats["var"] = pd.to_numeric(stats["var"], errors="coerce").fillna(0.0)
    stats["raw_weight"] = 1.0 / (stats["count"].astype(float) * np.sqrt(stats["var"].astype(float) + float(eps)))
    median_weight = float(np.nanmedian(stats["raw_weight"].to_numpy(dtype=float))) if not stats.empty else np.nan
    clip_ceiling = float(clip_ratio) * median_weight if np.isfinite(median_weight) else np.nan
    if np.isfinite(clip_ceiling):
        stats["clipped_weight"] = np.minimum(stats["raw_weight"].astype(float), clip_ceiling)
    else:
        stats["clipped_weight"] = stats["raw_weight"].astype(float)
    mean_weight = float(np.nanmean(stats["clipped_weight"].to_numpy(dtype=float))) if not stats.empty else np.nan
    if np.isfinite(mean_weight) and mean_weight > 0:
        stats["final_weight"] = stats["clipped_weight"].astype(float) / mean_weight
    else:
        stats["final_weight"] = 1.0

    train_weights = split_df.loc[train_mask, ["polymer_id"]].merge(
        stats[["polymer_id", "final_weight"]],
        on="polymer_id",
        how="left",
    )
    weights[train_mask] = train_weights["final_weight"].fillna(1.0).to_numpy(dtype=np.float32)
    return weights


def make_dataloaders(
    split_df: pd.DataFrame,
    embedding_table: np.ndarray,
    batch_size: int,
    weights: Optional[np.ndarray] = None,
    shuffle_train: bool = True,
) -> Dict[str, DataLoader]:
    loaders = {}
    for split in ["train", "val", "test"]:
        split_mask = split_df["split"] == split
        split_weights = None if weights is None else np.asarray(weights)[split_mask.to_numpy()]
        ds = ChiDataset(split_df[split_mask], embedding_table, weights=split_weights)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train" and shuffle_train),
            num_workers=0,
            pin_memory=False,
        )
    return loaders


def make_token_dataloaders(
    split_df: pd.DataFrame,
    tokenizer,
    batch_size: int,
    weights: Optional[np.ndarray] = None,
    shuffle_train: bool = True,
) -> Dict[str, DataLoader]:
    loaders = {}
    for split in ["train", "val", "test"]:
        split_mask = split_df["split"] == split
        split_weights = None if weights is None else np.asarray(weights)[split_mask.to_numpy()]
        ds = ChiTokenDataset(split_df[split_mask], tokenizer, weights=split_weights)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train" and shuffle_train),
            num_workers=0,
            pin_memory=False,
        )
    return loaders



def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    gradient_clip_norm: float = 0.0,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(mode=train_mode)

    losses: List[float] = []
    losses_mse: List[float] = []
    losses_bce: List[float] = []

    for batch in loader:
        temperature = batch["temperature"].to(device)
        phi = batch["phi"].to(device)
        chi_true = batch["chi"].to(device)
        weight = batch["weight"].to(device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        if "embedding" in batch:
            embedding = batch["embedding"].to(device)
            out = model(
                embedding=embedding,
                temperature=temperature,
                phi=phi,
            )
        else:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                temperature=temperature,
                phi=phi,
            )

        sq_error = (out["chi_pred"] - chi_true) ** 2
        if train_mode:
            loss_mse = torch.mean(weight * sq_error)
        else:
            loss_mse = torch.mean(sq_error)
        loss = loss_mse
        if train_mode:
            loss.backward()
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(gradient_clip_norm))
            optimizer.step()

        losses.append(float(loss.item()))
        losses_mse.append(float(loss_mse.item()))
        losses_bce.append(0.0)

    return {
        "loss": float(np.mean(losses)) if losses else np.nan,
        "loss_mse": float(np.mean(losses_mse)) if losses_mse else np.nan,
        "loss_bce": float(np.mean(losses_bce)) if losses_bce else np.nan,
    }


@torch.no_grad()
def predict_split(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, np.ndarray]:
    model.eval()
    pred_chi: List[np.ndarray] = []
    true_chi: List[np.ndarray] = []

    for batch in loader:
        temperature = batch["temperature"].to(device)
        phi = batch["phi"].to(device)
        chi_true = batch["chi"].to(device)

        if "embedding" in batch:
            embedding = batch["embedding"].to(device)
            out = model(embedding=embedding, temperature=temperature, phi=phi)
        else:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                temperature=temperature,
                phi=phi,
            )

        pred_chi.append(out["chi_pred"].cpu().numpy())
        true_chi.append(chi_true.cpu().numpy())

    return {
        "chi_true": np.concatenate(true_chi, axis=0) if true_chi else np.array([]),
        "chi_pred": np.concatenate(pred_chi, axis=0) if pred_chi else np.array([]),
    }



def train_one_model(
    split_df: pd.DataFrame,
    embedding_table: Optional[np.ndarray],
    train_cfg: TrainConfig,
    device: str,
    hidden_sizes: List[int],
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    num_epochs: int,
    patience: int,
    config: Optional[Dict] = None,
    model_size: Optional[str] = None,
    backbone_checkpoint: Optional[str] = None,
    tokenizer=None,
    finetune_last_layers: int = 0,
    timestep_for_embedding: int = 1,
) -> Tuple[nn.Module, Dict[str, List[float]], Dict[str, Dict[str, np.ndarray]]]:
    loss_weights = None
    if train_cfg.loss_weighting == "polymer_balanced":
        loss_weights = _compute_polymer_weights(
            split_df=split_df,
            eps=1.0e-4,
            clip_ratio=float(train_cfg.loss_weight_clip_ratio),
        )
    if finetune_last_layers > 0:
        if config is None:
            raise ValueError("config is required when finetune_last_layers > 0")
        if tokenizer is None:
            raise ValueError("tokenizer is required when finetune_last_layers > 0")
        dataloaders = make_token_dataloaders(
            split_df,
            tokenizer,
            batch_size=batch_size,
            weights=loss_weights,
            shuffle_train=True,
        )
        _, backbone, _ = load_backbone_from_step1(
            config=config,
            model_size=model_size,
            split_mode=train_cfg.split_mode,
            checkpoint_path=backbone_checkpoint,
            device=device,
        )
        _set_finetune_last_layers(backbone, finetune_last_layers=finetune_last_layers)
        chi_head = DirectChiRegressor(
            embedding_dim=int(backbone.hidden_size),
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        )
        model = BackboneDirectChiModel(
            backbone=backbone,
            chi_head=chi_head,
            timestep=int(timestep_for_embedding),
            pooling="mean",
        ).to(device)
    else:
        if embedding_table is None:
            raise ValueError("embedding_table is required when finetune_last_layers == 0")
        dataloaders = make_dataloaders(
            split_df,
            embedding_table,
            batch_size=batch_size,
            weights=loss_weights,
            shuffle_train=True,
        )
        model = DirectChiRegressor(
            embedding_dim=int(embedding_table.shape[1]),
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = None
    if train_cfg.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(train_cfg.scheduler_t_max or num_epochs)),
            eta_min=float(train_cfg.scheduler_min_lr),
        )

    history = {
        "epoch": [],
        "learning_rate": [],
        "train_loss": [],
        "train_loss_mse": [],
        "train_loss_bce": [],
        "val_loss": [],
        "val_loss_mse": [],
        "val_loss_bce": [],
        "val_rmse": [],
        "val_poly_nrmse": [],
        "steps_per_epoch_train": [],
    }

    best_state = None
    selection_metric_name = str(train_cfg.epoch_selection_metric)
    lower_is_better = _metric_is_lower_better(selection_metric_name)
    best_metric = np.inf if lower_is_better else -np.inf
    wait = 0
    has_val = len(dataloaders["val"].dataset) > 0
    val_polymer_ids = (
        split_df.loc[split_df["split"] == "val", "polymer_id"].to_numpy()
        if has_val
        else np.asarray([], dtype=np.int64)
    )
    steps_per_epoch_train = int(len(dataloaders["train"]))

    for epoch in range(1, num_epochs + 1):
        train_stats = run_epoch(
            model=model,
            loader=dataloaders["train"],
            optimizer=optimizer,
            device=device,
            gradient_clip_norm=float(train_cfg.gradient_clip_norm),
        )
        if has_val:
            val_stats = run_epoch(
                model=model,
                loader=dataloaders["val"],
                optimizer=None,
                device=device,
                gradient_clip_norm=0.0,
            )
            val_pred = predict_split(model, dataloaders["val"], device)
            val_reg = regression_metrics(val_pred["chi_true"], val_pred["chi_pred"])
            val_rmse = float(val_reg["rmse"])
            val_poly_nrmse = polymer_balanced_nrmse(
                val_pred["chi_true"],
                val_pred["chi_pred"],
                val_polymer_ids,
                std_floor=float(train_cfg.nrmse_std_floor),
                nrmse_clip=float(train_cfg.nrmse_clip),
            )
        else:
            val_stats = {"loss": np.nan, "loss_mse": np.nan, "loss_bce": np.nan}
            val_rmse = np.nan
            val_poly_nrmse = np.nan

        history["epoch"].append(epoch)
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
        history["train_loss"].append(train_stats["loss"])
        history["train_loss_mse"].append(train_stats["loss_mse"])
        history["train_loss_bce"].append(train_stats["loss_bce"])
        history["val_loss"].append(val_stats["loss"])
        history["val_loss_mse"].append(val_stats["loss_mse"])
        history["val_loss_bce"].append(val_stats["loss_bce"])
        history["val_rmse"].append(val_rmse)
        history["val_poly_nrmse"].append(float(val_poly_nrmse) if np.isfinite(val_poly_nrmse) else np.nan)
        history["steps_per_epoch_train"].append(steps_per_epoch_train)

        if has_val:
            selection_metric_value = val_rmse if selection_metric_name == "val_rmse" else val_poly_nrmse
            improved = (
                np.isfinite(selection_metric_value)
                and (
                    (lower_is_better and selection_metric_value < best_metric)
                    or ((not lower_is_better) and selection_metric_value > best_metric)
                )
            )
            if improved or best_state is None:
                if np.isfinite(selection_metric_value):
                    best_metric = float(selection_metric_value)
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break
        else:
            # No validation split in final all-train fitting mode: keep latest state.
            best_state = copy.deepcopy(model.state_dict())
        if scheduler is not None:
            scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)

    # Use deterministic (non-shuffled) loaders for split-level prediction CSVs and metrics.
    if finetune_last_layers > 0:
        pred_loaders = make_token_dataloaders(
            split_df,
            tokenizer,
            batch_size=batch_size,
            shuffle_train=False,
        )
    else:
        pred_loaders = make_dataloaders(
            split_df,
            embedding_table,
            batch_size=batch_size,
            shuffle_train=False,
        )

    predictions = {
        split: predict_split(model, loader, device)
        for split, loader in pred_loaders.items()
    }
    return model, history, predictions



def _build_tuning_cv_folds(split_df: pd.DataFrame, train_cfg: TrainConfig) -> Tuple[List[pd.DataFrame], Dict[str, object]]:
    """Create stratified CV folds from all non-test rows for robust hyperparameter tuning."""
    dev_df = split_df[split_df["split"].isin(["train", "val"])].copy().reset_index(drop=True)
    if dev_df.empty:
        raise ValueError("No train/val rows available for Optuna tuning.")

    requested_folds = int(max(2, train_cfg.tuning_cv_folds))

    if train_cfg.split_mode == "polymer":
        unit_df = (
            dev_df[["polymer_id", "water_miscible"]]
            .drop_duplicates(subset=["polymer_id"])
            .sort_values("polymer_id")
            .reset_index(drop=True)
        )
        unit_key = "polymer_id"
        strategy = "polymer_group_stratified"
    else:
        unit_df = (
            dev_df[["row_id", "water_miscible"]]
            .drop_duplicates(subset=["row_id"])
            .sort_values("row_id")
            .reset_index(drop=True)
        )
        unit_key = "row_id"
        strategy = "row_stratified"

    class_counts = unit_df["water_miscible"].value_counts()
    max_folds = int(min(len(unit_df), class_counts.min())) if not class_counts.empty else 0

    if max_folds < 2:
        fallback = dev_df.copy()
        if fallback["split"].nunique() < 2:
            # Last-resort deterministic fallback to avoid hard failure on tiny/degenerate data.
            idx = np.arange(len(fallback))
            fallback["split"] = np.where((idx % 5) == 0, "val", "train")
        return [fallback.reset_index(drop=True)], {
            "strategy": f"{strategy}_fallback_original_split",
            "requested_folds": requested_folds,
            "resolved_folds": 1,
            "dev_rows": int(len(dev_df)),
            "dev_units": int(len(unit_df)),
        }

    resolved_folds = int(min(requested_folds, max_folds))
    skf = StratifiedKFold(n_splits=resolved_folds, shuffle=True, random_state=train_cfg.seed)
    unit_ids = unit_df[unit_key].to_numpy()
    labels = unit_df["water_miscible"].to_numpy(dtype=int)

    folds: List[pd.DataFrame] = []
    for _, val_idx in skf.split(unit_ids, labels):
        val_ids = set(unit_ids[val_idx].tolist())
        fold_df = dev_df.copy()
        fold_df["split"] = np.where(fold_df[unit_key].isin(val_ids), "val", "train")
        folds.append(fold_df.reset_index(drop=True))

    return folds, {
        "strategy": strategy,
        "requested_folds": requested_folds,
        "resolved_folds": resolved_folds,
        "dev_rows": int(len(dev_df)),
        "dev_units": int(len(unit_df)),
    }


def _summarize_tuning_cv_folds(cv_folds: List[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for i, fold_df in enumerate(cv_folds, start=1):
        for split in ["train", "val"]:
            sub = fold_df[fold_df["split"] == split]
            n_rows = int(len(sub))
            n_pos = int(sub["water_miscible"].sum()) if n_rows > 0 else 0
            rows.append(
                {
                    "fold": i,
                    "split": split,
                    "n_rows": n_rows,
                    "n_polymers": int(sub["polymer_id"].nunique()) if n_rows > 0 else 0,
                    "n_positive": n_pos,
                    "n_negative": int(n_rows - n_pos),
                }
            )
    return pd.DataFrame(rows)


def _evaluate_trial_with_cv(
    cv_folds: List[pd.DataFrame],
    embedding_table: Optional[np.ndarray],
    train_cfg: TrainConfig,
    config: Dict,
    model_size: Optional[str],
    backbone_checkpoint: Optional[str],
    tokenizer,
    device: str,
    hidden_sizes: List[int],
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    finetune_last_layers: int,
    collect_val_predictions: bool = False,
) -> Dict[str, object]:
    fold_rows = []
    cv_val_frames: List[pd.DataFrame] = []
    for fold_id, fold_df in enumerate(cv_folds, start=1):
        _, _, preds = train_one_model(
            split_df=fold_df,
            embedding_table=embedding_table,
            train_cfg=train_cfg,
            device=device,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            num_epochs=train_cfg.tuning_epochs,
            patience=train_cfg.tuning_patience,
            config=config,
            model_size=model_size,
            backbone_checkpoint=backbone_checkpoint,
            tokenizer=tokenizer,
            finetune_last_layers=finetune_last_layers,
            timestep_for_embedding=train_cfg.timestep_for_embedding,
        )
        val_pred = preds["val"]
        val_reg = regression_metrics(val_pred["chi_true"], val_pred["chi_pred"])
        val_polymer_ids = fold_df.loc[fold_df["split"] == "val", "polymer_id"].to_numpy()
        val_poly_nrmse = polymer_balanced_nrmse(
            val_pred["chi_true"],
            val_pred["chi_pred"],
            val_polymer_ids,
            std_floor=float(train_cfg.nrmse_std_floor),
            nrmse_clip=float(train_cfg.nrmse_clip),
        )
        fold_rows.append(
            {
                "fold": fold_id,
                "val_n": int(len(val_pred["chi_true"])),
                "val_r2": float(val_reg["r2"]),
                "val_rmse": float(val_reg["rmse"]),
                "val_poly_nrmse": float(val_poly_nrmse) if np.isfinite(val_poly_nrmse) else np.nan,
            }
        )
        if collect_val_predictions:
            val_sub = fold_df[fold_df["split"] == "val"].copy().reset_index(drop=True)
            if len(val_sub) != len(val_pred["chi_true"]):
                raise ValueError(
                    f"CV fold={fold_id} val length mismatch: split_rows={len(val_sub)}, pred_rows={len(val_pred['chi_true'])}"
                )
            val_sub["chi_true"] = np.asarray(val_pred["chi_true"], dtype=float)
            val_sub["chi_pred"] = np.asarray(val_pred["chi_pred"], dtype=float)
            val_sub["fold"] = int(fold_id)
            cv_val_frames.append(
                val_sub[["fold", "polymer_id", "Polymer", "SMILES", "water_miscible", "chi_true", "chi_pred"]].copy()
            )

    fold_metrics_df = pd.DataFrame(fold_rows)
    mean_r2 = float(np.nanmean(fold_metrics_df["val_r2"])) if not fold_metrics_df.empty else np.nan
    mean_rmse = float(np.nanmean(fold_metrics_df["val_rmse"])) if not fold_metrics_df.empty else np.nan
    mean_poly_nrmse = float(np.nanmean(fold_metrics_df["val_poly_nrmse"])) if not fold_metrics_df.empty else np.nan
    cv_val_df = pd.concat(cv_val_frames, ignore_index=True) if cv_val_frames else pd.DataFrame()
    return {
        "cv_val_r2": mean_r2,
        "cv_val_rmse": mean_rmse,
        "cv_val_poly_nrmse": mean_poly_nrmse,
        "fold_metrics": fold_metrics_df,
        "cv_val_predictions": cv_val_df,
    }


def _save_cv_parity_by_fold_figure(
    cv_val_df: pd.DataFrame,
    out_png: Path,
    dpi: int,
    font_size: int,
    split_label: str = "val",
) -> None:
    if cv_val_df.empty:
        return

    apply_publication_figure_style(font_size=font_size, dpi=dpi, remove_titles=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_df = cv_val_df.copy()
    plot_df["fold"] = plot_df["fold"].astype(str)
    n_folds = int(plot_df["fold"].nunique())
    palette = sns.color_palette("tab10", n_colors=max(n_folds, 3))
    sns.scatterplot(
        data=plot_df,
        x="chi_true",
        y="chi_pred",
        hue="fold",
        palette=palette,
        alpha=0.75,
        s=18,
        ax=ax,
    )
    lo = float(min(plot_df["chi_true"].min(), plot_df["chi_pred"].min()))
    hi = float(max(plot_df["chi_true"].max(), plot_df["chi_pred"].max()))
    span = max(hi - lo, 1e-8)
    pad = 0.04 * span
    lo_plot = lo - pad
    hi_plot = hi + pad
    ax.plot([lo_plot, hi_plot], [lo_plot, hi_plot], "k--", linewidth=1.1)
    ax.set_xlim(lo_plot, hi_plot)
    ax.set_ylim(lo_plot, hi_plot)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("True χ")
    ax.set_ylabel("Predicted χ")

    reg = regression_metrics(plot_df["chi_true"], plot_df["chi_pred"])
    metrics_text = (
        f"MAE={reg['mae']:.3f}\n"
        f"RMSE={reg['rmse']:.3f}\n"
        f"R2={reg['r2']:.3f}"
    )
    ax.text(
        0.03,
        0.97,
        metrics_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#666666", "alpha": 0.92},
    )
    ax.set_title(f"CV parity by fold ({split_label} folds, n={len(plot_df)})")
    ax.legend(
        title="CV fold",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def _save_best_metric_vs_trial_figure(
    trial_df: pd.DataFrame,
    out_png: Path,
    y_col: str,
    y_label: str,
    title: str,
    dpi: int,
    font_size: int,
) -> None:
    if trial_df.empty or y_col not in trial_df.columns:
        return
    x = pd.to_numeric(trial_df["trial"], errors="coerce").to_numpy()
    y = pd.to_numeric(trial_df[y_col], errors="coerce").to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return

    apply_publication_figure_style(font_size=font_size, dpi=dpi, remove_titles=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        x[mask],
        y[mask],
        "-o",
        color="#e76f51",
        linewidth=2,
        markersize=4,
        label="Best trial metric so far",
    )
    ax.set_xlabel("Optuna trial")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def _summarize_cv_fold_metrics(
    fold_metrics_df: pd.DataFrame,
    metric_cols: List[str],
) -> pd.DataFrame:
    if fold_metrics_df.empty:
        cols = ["cv_split", "n_folds"] + [f"{m}_{s}" for m in metric_cols for s in ["mean", "std"]]
        return pd.DataFrame(columns=cols)

    rows: List[Dict[str, object]] = []
    for cv_split, sub in fold_metrics_df.groupby("cv_split", sort=True):
        row: Dict[str, object] = {
            "cv_split": str(cv_split),
            "n_folds": int(sub["fold"].nunique()) if "fold" in sub.columns else int(len(sub)),
        }
        for metric in metric_cols:
            vals = pd.to_numeric(sub.get(metric, pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
            finite = vals[np.isfinite(vals)]
            row[f"{metric}_mean"] = float(np.mean(finite)) if finite.size > 0 else np.nan
            row[f"{metric}_std"] = float(np.std(finite)) if finite.size > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _cv_summary_table_to_dict(summary_df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    if summary_df.empty:
        return out
    for _, row in summary_df.iterrows():
        split = str(row.get("cv_split", ""))
        if len(split) == 0:
            continue
        payload: Dict[str, object] = {}
        for col in summary_df.columns:
            if col == "cv_split":
                continue
            val = row[col]
            if pd.isna(val):
                payload[col] = None
            elif isinstance(val, (np.floating, float)):
                payload[col] = float(val)
            elif isinstance(val, (np.integer, int)):
                payload[col] = int(val)
            else:
                payload[col] = val
        out[split] = payload
    return out


def _save_classifier_cv_parity_by_fold_figure(
    cv_pred_df: pd.DataFrame,
    out_png: Path,
    dpi: int,
    font_size: int,
    split_label: str,
) -> None:
    if cv_pred_df.empty:
        return

    apply_publication_figure_style(font_size=font_size, dpi=dpi, remove_titles=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_df = cv_pred_df.copy()
    plot_df["fold"] = plot_df["fold"].astype(str)
    rng = np.random.default_rng(0)
    jitter = rng.normal(loc=0.0, scale=0.03, size=len(plot_df))
    plot_df["water_miscible_jitter"] = np.clip(
        pd.to_numeric(plot_df["water_miscible"], errors="coerce").fillna(0.0).to_numpy(dtype=float) + jitter,
        -0.08,
        1.08,
    )
    n_folds = int(plot_df["fold"].nunique())
    palette = sns.color_palette("tab10", n_colors=max(n_folds, 3))
    sns.scatterplot(
        data=plot_df,
        x="water_miscible_jitter",
        y="class_prob",
        hue="fold",
        palette=palette,
        alpha=0.75,
        s=18,
        ax=ax,
    )
    ax.plot([0.0, 1.0], [0.0, 1.0], "k--", linewidth=1.1)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(CLASS_DISPLAY_TICKLABELS)
    ax.set_xlabel("True class")
    ax.set_ylabel("Predicted water-miscible probability")

    cls = classification_metrics(plot_df["water_miscible"], plot_df["class_prob"])
    metrics_text = (
        f"BalAcc={cls['balanced_accuracy']:.3f}\n"
        f"AUROC={cls['auroc']:.3f}\n"
        f"AUPRC={cls['auprc']:.3f}\n"
        f"Brier={cls['brier']:.3f}"
    )
    ax.text(
        0.03,
        0.97,
        metrics_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#666666", "alpha": 0.92},
    )
    ax.set_title(f"CV parity by fold ({split_label} folds, n={len(plot_df)})")
    ax.legend(
        title="CV fold",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def run_regression_cv_with_best_hyperparameters(
    split_df: pd.DataFrame,
    embedding_table: Optional[np.ndarray],
    train_cfg: TrainConfig,
    config: Dict,
    model_size: Optional[str],
    backbone_checkpoint: Optional[str],
    tokenizer,
    device: str,
    hidden_sizes: List[int],
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    finetune_last_layers: int,
    metrics_dir: Path,
    figures_dir: Path,
    dpi: int,
    font_size: int,
) -> Dict[str, object]:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    cv_folds, cv_info = _build_tuning_cv_folds(split_df=split_df, train_cfg=train_cfg)
    fold_layout_df = _summarize_tuning_cv_folds(cv_folds)
    fold_layout_df.to_csv(metrics_dir / "cv_fold_layout.csv", index=False)

    fold_metric_rows: List[Dict[str, object]] = []
    fold_pred_frames: List[pd.DataFrame] = []
    fold_history_frames: List[pd.DataFrame] = []

    for fold_id, fold_df in enumerate(cv_folds, start=1):
        _, history, preds = train_one_model(
            split_df=fold_df,
            embedding_table=embedding_table,
            train_cfg=train_cfg,
            device=device,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            num_epochs=train_cfg.budget_search_epochs,
            patience=train_cfg.budget_search_patience,
            config=config,
            model_size=model_size,
            backbone_checkpoint=backbone_checkpoint,
            tokenizer=tokenizer,
            finetune_last_layers=finetune_last_layers,
            timestep_for_embedding=train_cfg.timestep_for_embedding,
        )

        fold_hist = pd.DataFrame(history)
        if not fold_hist.empty:
            fold_hist.insert(0, "fold", int(fold_id))
            fold_history_frames.append(fold_hist)

        for raw_split, cv_split in [("train", "train"), ("val", "test")]:
            sub = fold_df[fold_df["split"] == raw_split].copy().reset_index(drop=True)
            pred = preds[raw_split]
            if len(sub) != len(pred["chi_true"]):
                raise ValueError(
                    f"Step4_1 CV fold={fold_id} split={raw_split} length mismatch: "
                    f"split_rows={len(sub)}, pred_rows={len(pred['chi_true'])}"
                )

            chi_true = np.asarray(pred["chi_true"], dtype=float)
            chi_pred = np.asarray(pred["chi_pred"], dtype=float)
            sub["chi_true"] = chi_true
            sub["chi_pred"] = chi_pred
            sub["chi_error"] = chi_pred - chi_true
            sub["fold"] = int(fold_id)
            sub["cv_split"] = cv_split
            sub["raw_split"] = raw_split

            use_cols = [
                "fold",
                "cv_split",
                "raw_split",
                "row_id",
                "polymer_id",
                "Polymer",
                "SMILES",
                "water_miscible",
                "chi_true",
                "chi_pred",
                "chi_error",
            ]
            fold_pred_frames.append(sub[[c for c in use_cols if c in sub.columns]].copy())

            row: Dict[str, object] = {
                "fold": int(fold_id),
                "cv_split": cv_split,
                "raw_split": raw_split,
                "n_rows": int(len(sub)),
            }
            row.update(regression_metrics(chi_true, chi_pred))
            poly_nrmse = polymer_balanced_nrmse(
                chi_true,
                chi_pred,
                sub["polymer_id"].to_numpy(),
                std_floor=float(train_cfg.nrmse_std_floor),
                nrmse_clip=float(train_cfg.nrmse_clip),
            )
            row["poly_nrmse"] = float(poly_nrmse) if np.isfinite(poly_nrmse) else np.nan
            row["val_poly_nrmse"] = row["poly_nrmse"] if raw_split == "val" else np.nan
            row.update(hit_metrics(chi_pred - chi_true, epsilons=[0.02, 0.05, 0.1, 0.2]))
            fold_metric_rows.append(row)

    fold_metrics_df = pd.DataFrame(fold_metric_rows).sort_values(["cv_split", "fold"]).reset_index(drop=True)
    fold_metrics_df.to_csv(metrics_dir / "cv_fold_metrics.csv", index=False)
    cv_summary_df = _summarize_cv_fold_metrics(
        fold_metrics_df=fold_metrics_df,
        metric_cols=["mae", "rmse", "nrmse", "r2", "poly_nrmse", "val_poly_nrmse"],
    )
    cv_summary_df.to_csv(metrics_dir / "cv_metrics_summary.csv", index=False)

    if fold_history_frames:
        pd.concat(fold_history_frames, ignore_index=True).to_csv(metrics_dir / "cv_training_history.csv", index=False)
    else:
        pd.DataFrame().to_csv(metrics_dir / "cv_training_history.csv", index=False)

    if fold_pred_frames:
        cv_pred_df = pd.concat(fold_pred_frames, ignore_index=True)
    else:
        cv_pred_df = pd.DataFrame()
    cv_pred_df.to_csv(metrics_dir / "cv_predictions_all.csv", index=False)
    for split in ["train", "test"]:
        if cv_pred_df.empty or "cv_split" not in cv_pred_df.columns:
            split_pred_df = pd.DataFrame()
        else:
            split_pred_df = cv_pred_df[cv_pred_df["cv_split"] == split].copy()
        split_pred_df.to_csv(metrics_dir / f"cv_predictions_{split}.csv", index=False)
        if not split_pred_df.empty:
            if split == "train":
                _save_cv_parity_by_fold_figure(
                    cv_val_df=split_pred_df,
                    out_png=figures_dir / "cv_parity_train_by_fold.png",
                    dpi=dpi,
                    font_size=font_size,
                    split_label="train",
                )
            else:
                _save_cv_parity_by_fold_figure(
                    cv_val_df=split_pred_df,
                    out_png=figures_dir / "cv_parity_test_by_fold.png",
                    dpi=dpi,
                    font_size=font_size,
                    split_label="test",
                )

    best_epoch_rows: List[Dict[str, object]] = []
    selection_metric = str(train_cfg.epoch_selection_metric)
    lower_is_better = _metric_is_lower_better(selection_metric)
    best_steps: List[int] = []
    for fold_id, fold_hist in enumerate(fold_history_frames, start=1):
        metric_series = pd.to_numeric(fold_hist.get(selection_metric, pd.Series(dtype=float)), errors="coerce")
        finite_mask = np.isfinite(metric_series.to_numpy(dtype=float))
        if finite_mask.any():
            metric_subset = metric_series[finite_mask]
            best_idx = metric_subset.idxmin() if lower_is_better else metric_subset.idxmax()
        else:
            best_idx = fold_hist.index[-1]
        best_row = fold_hist.loc[best_idx]
        best_epoch = int(best_row["epoch"])
        steps_per_epoch_train = int(best_row.get("steps_per_epoch_train", 0))
        if steps_per_epoch_train <= 0:
            steps_per_epoch_train = 1
        best_step = max(1, int(best_epoch * steps_per_epoch_train))
        best_steps.append(best_step)
        best_epoch_rows.append(
            {
                "fold": int(fold_id),
                "best_epoch": best_epoch,
                "steps_per_epoch_train": steps_per_epoch_train,
                "best_step": best_step,
                "n_train_rows": int((cv_folds[fold_id - 1]["split"] == "train").sum()),
                "batch_size": int(batch_size),
                "metric_used": selection_metric,
                "metric_value": float(best_row.get(selection_metric, np.nan)),
            }
        )
    best_epoch_df = pd.DataFrame(best_epoch_rows)
    best_epoch_df.to_csv(metrics_dir / "cv_best_epochs_by_fold.csv", index=False)

    final_fit_df = _build_final_fit_split_df(split_df)
    if finetune_last_layers > 0:
        final_train_loaders = make_token_dataloaders(
            final_fit_df,
            tokenizer,
            batch_size=batch_size,
            shuffle_train=False,
        )
    else:
        final_train_loaders = make_dataloaders(
            final_fit_df,
            embedding_table,
            batch_size=batch_size,
            shuffle_train=False,
        )
    steps_per_epoch_final = max(1, int(len(final_train_loaders["train"])))
    target_steps = max(1, int(np.median(best_steps))) if best_steps else 1
    derived_final_epochs = int(np.ceil(target_steps / float(steps_per_epoch_final)))
    derived_final_epochs = int(np.clip(derived_final_epochs, 1, int(train_cfg.num_epochs)))

    summary_by_split = _cv_summary_table_to_dict(cv_summary_df)
    run_summary = {
        "requested_folds": int(train_cfg.tuning_cv_folds),
        "resolved_folds": int(cv_info.get("resolved_folds", len(cv_folds))),
        "strategy": str(cv_info.get("strategy", "unknown")),
        "dev_rows": int(cv_info.get("dev_rows", 0)),
        "dev_units": int(cv_info.get("dev_units", 0)),
        "budget_search_epochs": int(train_cfg.budget_search_epochs),
        "budget_search_patience": int(train_cfg.budget_search_patience),
        "epoch_selection_metric": selection_metric,
        "final_training_derived_steps": int(target_steps),
        "final_training_derived_epochs": int(derived_final_epochs),
        "steps_per_epoch_final": int(steps_per_epoch_final),
        "summary_by_split": summary_by_split,
    }
    with open(metrics_dir / "cv_run_summary.json", "w") as f:
        json.dump(run_summary, f, indent=2)
    return run_summary


def run_classifier_cv_with_best_hyperparameters(
    split_df: pd.DataFrame,
    embedding_table: Optional[np.ndarray],
    train_cfg: TrainConfig,
    config: Dict,
    model_size: Optional[str],
    backbone_checkpoint: Optional[str],
    tokenizer,
    device: str,
    hidden_sizes: List[int],
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    finetune_last_layers: int,
    metrics_dir: Path,
    figures_dir: Path,
    dpi: int,
    font_size: int,
    backbone_split_mode: Optional[str] = None,
) -> Dict[str, object]:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    cv_folds, cv_info = _build_tuning_cv_folds(split_df=split_df, train_cfg=train_cfg)
    fold_layout_df = _summarize_tuning_cv_folds(cv_folds)
    fold_layout_df.to_csv(metrics_dir / "cv_fold_layout.csv", index=False)

    fold_metric_rows: List[Dict[str, object]] = []
    fold_pred_frames: List[pd.DataFrame] = []
    fold_history_frames: List[pd.DataFrame] = []

    for fold_id, fold_df in enumerate(cv_folds, start=1):
        _, history, preds = train_one_classifier_model(
            split_df=fold_df,
            embedding_table=embedding_table,
            train_cfg=train_cfg,
            device=device,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            num_epochs=train_cfg.num_epochs,
            patience=train_cfg.patience,
            config=config,
            model_size=model_size,
            backbone_checkpoint=backbone_checkpoint,
            tokenizer=tokenizer,
            finetune_last_layers=finetune_last_layers,
            timestep_for_embedding=train_cfg.timestep_for_embedding,
            backbone_split_mode=backbone_split_mode,
        )

        fold_hist = pd.DataFrame(history)
        if not fold_hist.empty:
            fold_hist.insert(0, "fold", int(fold_id))
            fold_history_frames.append(fold_hist)

        for raw_split, cv_split in [("train", "train"), ("val", "test")]:
            sub = fold_df[fold_df["split"] == raw_split].copy().reset_index(drop=True)
            pred = preds[raw_split]
            if len(sub) != len(pred["prob"]):
                raise ValueError(
                    f"Step4_2 CV fold={fold_id} split={raw_split} length mismatch: "
                    f"split_rows={len(sub)}, pred_rows={len(pred['prob'])}"
                )

            class_logit = np.asarray(pred["logit"], dtype=float)
            class_prob = np.asarray(pred["prob"], dtype=float)
            class_true = pd.to_numeric(sub["water_miscible"], errors="coerce").fillna(0).to_numpy(dtype=int)
            sub["class_logit"] = class_logit
            sub["class_prob"] = class_prob
            sub["class_pred"] = (class_prob >= 0.5).astype(int)
            sub["fold"] = int(fold_id)
            sub["cv_split"] = cv_split
            sub["raw_split"] = raw_split

            use_cols = [
                "fold",
                "cv_split",
                "raw_split",
                "row_id",
                "polymer_id",
                "Polymer",
                "SMILES",
                "water_miscible",
                "class_logit",
                "class_prob",
                "class_pred",
            ]
            fold_pred_frames.append(sub[[c for c in use_cols if c in sub.columns]].copy())

            row: Dict[str, object] = {
                "fold": int(fold_id),
                "cv_split": cv_split,
                "raw_split": raw_split,
                "n_rows": int(len(sub)),
            }
            row.update(classification_metrics(class_true, class_prob))
            fold_metric_rows.append(row)

    fold_metrics_df = pd.DataFrame(fold_metric_rows).sort_values(["cv_split", "fold"]).reset_index(drop=True)
    fold_metrics_df.to_csv(metrics_dir / "cv_fold_metrics.csv", index=False)
    cv_summary_df = _summarize_cv_fold_metrics(
        fold_metrics_df=fold_metrics_df,
        metric_cols=["balanced_accuracy", "auroc", "auprc", "brier"],
    )
    cv_summary_df.to_csv(metrics_dir / "cv_metrics_summary.csv", index=False)

    if fold_history_frames:
        pd.concat(fold_history_frames, ignore_index=True).to_csv(metrics_dir / "cv_training_history.csv", index=False)
    else:
        pd.DataFrame().to_csv(metrics_dir / "cv_training_history.csv", index=False)

    if fold_pred_frames:
        cv_pred_df = pd.concat(fold_pred_frames, ignore_index=True)
    else:
        cv_pred_df = pd.DataFrame()
    cv_pred_df.to_csv(metrics_dir / "cv_predictions_all.csv", index=False)
    for split in ["train", "test"]:
        if cv_pred_df.empty or "cv_split" not in cv_pred_df.columns:
            split_df = pd.DataFrame()
        else:
            split_df = cv_pred_df[cv_pred_df["cv_split"] == split].copy()
        split_df.to_csv(metrics_dir / f"cv_predictions_{split}.csv", index=False)
        if not split_df.empty:
            _save_classifier_cv_parity_by_fold_figure(
                cv_pred_df=split_df,
                out_png=figures_dir / f"cv_parity_{split}_by_fold.png",
                dpi=dpi,
                font_size=font_size,
                split_label=split,
            )

    summary_by_split = _cv_summary_table_to_dict(cv_summary_df)
    run_summary = {
        "requested_folds": int(train_cfg.tuning_cv_folds),
        "resolved_folds": int(cv_info.get("resolved_folds", len(cv_folds))),
        "strategy": str(cv_info.get("strategy", "unknown")),
        "dev_rows": int(cv_info.get("dev_rows", 0)),
        "dev_units": int(cv_info.get("dev_units", 0)),
        "num_epochs": int(train_cfg.num_epochs),
        "patience": int(train_cfg.patience),
        "summary_by_split": summary_by_split,
    }
    with open(metrics_dir / "cv_run_summary.json", "w") as f:
        json.dump(run_summary, f, indent=2)
    return run_summary


def tune_hyperparameters(
    split_df: pd.DataFrame,
    embedding_table: Optional[np.ndarray],
    train_cfg: TrainConfig,
    config: Dict,
    model_size: Optional[str],
    backbone_num_layers: int,
    backbone_checkpoint: Optional[str],
    tokenizer,
    device: str,
    tuning_dir: Path,
    dpi: int = 300,
    font_size: int = 16,
) -> Dict:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna is required for --tune. Install it with `pip install optuna` or disable tuning."
        ) from exc

    tuning_dir.mkdir(parents=True, exist_ok=True)
    search_space = train_cfg.optuna_search_space

    def _as_list(key: str, default: List[float]) -> List[float]:
        values = search_space.get(key, default)
        if isinstance(values, list) and len(values) > 0:
            return values
        return default

    def _as_int_list(key: str, default: List[int]) -> List[int]:
        raw = _as_list(key, [int(v) for v in default])
        parsed = []
        for item in raw:
            try:
                parsed.append(int(float(item)))
            except Exception:
                continue
        return parsed if parsed else [int(v) for v in default]

    def _as_float_list(key: str, default: List[float]) -> List[float]:
        raw = _as_list(key, [float(v) for v in default])
        parsed = []
        for item in raw:
            try:
                parsed.append(float(item))
            except Exception:
                continue
        return parsed if parsed else [float(v) for v in default]

    num_layers_space = _as_int_list("num_layers", [1, 2, 3])
    hidden_units_space = _as_int_list("hidden_units", [64, 128, 256, 512])
    dropout_space = _as_float_list("dropout", [0.0, 0.1, 0.2, 0.3])
    lr_space = _as_float_list("learning_rate", [1e-4, 5e-3])
    wd_space = _as_float_list("weight_decay", [1e-7, 1e-3])
    batch_size_space = _as_int_list("batch_size", [16, 32, 64, 128, 256])
    lr_log = bool(search_space.get("learning_rate_log", True))
    wd_log = bool(search_space.get("weight_decay_log", True))
    finetune_raw = search_space.get("finetune_last_layers", [0, int(backbone_num_layers)])
    if isinstance(finetune_raw, list) and len(finetune_raw) > 0:
        finetune_space = []
        for item in finetune_raw:
            try:
                finetune_space.append(int(float(item)))
            except Exception:
                continue
    else:
        finetune_space = [0, int(backbone_num_layers)]
    if len(finetune_space) == 0:
        finetune_space = [0, int(backbone_num_layers)]
    if len(finetune_space) == 2:
        finetune_lo = max(0, min(finetune_space))
        finetune_hi = min(int(backbone_num_layers), max(finetune_space))
        if finetune_hi < finetune_lo:
            clamped = max(0, min(int(backbone_num_layers), finetune_lo))
            finetune_lo = clamped
            finetune_hi = clamped
        finetune_mode = "range"
        finetune_values = []
    else:
        finetune_mode = "categorical"
        finetune_values = sorted(set(int(v) for v in finetune_space if 0 <= int(v) <= int(backbone_num_layers)))
        if len(finetune_values) == 0:
            finetune_values = [0, int(backbone_num_layers)]
        finetune_lo = min(finetune_values)
        finetune_hi = max(finetune_values)
    if finetune_hi > 0 and tokenizer is None:
        raise ValueError(
            "Step4_1 Optuna search includes finetune_last_layers > 0, but tokenizer is unavailable. "
            "Provide Step1 backbone assets/checkpoint so tokenizer can be loaded."
        )

    cv_folds, cv_info = _build_tuning_cv_folds(split_df=split_df, train_cfg=train_cfg)
    _summarize_tuning_cv_folds(cv_folds).to_csv(tuning_dir / "optuna_tuning_cv_folds.csv", index=False)

    objective_name = train_cfg.tuning_objective
    objective_direction = _tuning_objective_direction(objective_name)

    def objective(trial: optuna.Trial) -> float:
        if len(num_layers_space) == 2:
            lo = int(min(num_layers_space))
            hi = int(max(num_layers_space))
            num_layers = trial.suggest_int("num_layers", lo, hi)
        else:
            num_layers = int(trial.suggest_categorical("num_layers", [int(v) for v in num_layers_space]))

        hidden_sizes = [int(trial.suggest_categorical(f"hidden_{i}", hidden_units_space)) for i in range(num_layers)]
        dropout = float(trial.suggest_categorical("dropout", dropout_space))

        if len(lr_space) == 2:
            lr = float(trial.suggest_float("learning_rate", min(lr_space), max(lr_space), log=lr_log))
        else:
            lr = float(trial.suggest_categorical("learning_rate", lr_space))

        if len(wd_space) == 2:
            wd = float(trial.suggest_float("weight_decay", min(wd_space), max(wd_space), log=wd_log))
        else:
            wd = float(trial.suggest_categorical("weight_decay", wd_space))

        batch_size = int(trial.suggest_categorical("batch_size", batch_size_space))
        if finetune_mode == "range":
            finetune_last_layers = int(trial.suggest_int("finetune_last_layers", finetune_lo, finetune_hi))
        else:
            finetune_last_layers = int(trial.suggest_categorical("finetune_last_layers", finetune_values))

        cv_eval = _evaluate_trial_with_cv(
            cv_folds=cv_folds,
            embedding_table=embedding_table,
            train_cfg=train_cfg,
            config=config,
            model_size=model_size,
            backbone_checkpoint=backbone_checkpoint,
            tokenizer=tokenizer,
            device=device,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            learning_rate=lr,
            weight_decay=wd,
            batch_size=batch_size,
            finetune_last_layers=finetune_last_layers,
        )
        val_r2 = float(cv_eval["cv_val_r2"])
        val_rmse = float(cv_eval["cv_val_rmse"])
        val_poly_nrmse = float(cv_eval["cv_val_poly_nrmse"])
        if objective_name == "val_poly_nrmse":
            objective_value = val_poly_nrmse
        else:
            objective_value = val_r2
        invalid_metrics = int(not np.isfinite(objective_value))
        trial.set_user_attr("val_r2", val_r2)
        trial.set_user_attr("val_rmse", val_rmse)
        trial.set_user_attr("val_poly_nrmse", val_poly_nrmse)
        trial.set_user_attr("cv_val_r2", val_r2)
        trial.set_user_attr("cv_val_rmse", val_rmse)
        trial.set_user_attr("cv_val_poly_nrmse", val_poly_nrmse)
        trial.set_user_attr("cv_n_folds", int(len(cv_folds)))
        trial.set_user_attr("tuning_objective", objective_name)
        trial.set_user_attr("invalid_metrics", invalid_metrics)
        if invalid_metrics:
            return 1.0e12 if objective_direction == "minimize" else -1.0e12
        return float(objective_value)

    study = optuna.create_study(direction=objective_direction)
    study.optimize(objective, n_trials=train_cfg.n_trials, show_progress_bar=True)

    trials = []
    for t in study.trials:
        val_r2 = t.user_attrs.get("cv_val_r2", t.user_attrs.get("val_r2", np.nan))
        val_rmse = t.user_attrs.get("cv_val_rmse", t.user_attrs.get("val_rmse", np.nan))
        val_poly_nrmse = t.user_attrs.get("cv_val_poly_nrmse", t.user_attrs.get("val_poly_nrmse", np.nan))
        row = {
            "trial": t.number,
            "state": str(t.state),
            "objective_name": objective_name,
            "objective_direction": objective_direction,
            "objective_value": t.value,
            "value_val_r2": val_r2,
            "val_r2": val_r2,
            "val_rmse": val_rmse,
            "val_poly_nrmse": val_poly_nrmse,
            "invalid_metrics": int(t.user_attrs.get("invalid_metrics", 0)),
            "cv_n_folds": int(t.user_attrs.get("cv_n_folds", len(cv_folds))),
        }
        row.update(t.params)
        trials.append(row)

    pd.DataFrame(trials).to_csv(tuning_dir / "optuna_trials.csv", index=False)
    trial_df = pd.DataFrame(trials).sort_values("trial").reset_index(drop=True)
    if "val_r2" in trial_df.columns:
        trial_df["chi_val_r2"] = trial_df["val_r2"]
    else:
        trial_df["chi_val_r2"] = np.nan
    if "val_rmse" in trial_df.columns:
        trial_df["chi_val_rmse"] = trial_df["val_rmse"]
    else:
        trial_df["chi_val_rmse"] = np.nan
    if "val_poly_nrmse" in trial_df.columns:
        trial_df["chi_val_poly_nrmse"] = trial_df["val_poly_nrmse"]
    else:
        trial_df["chi_val_poly_nrmse"] = np.nan

    # Running best R2 over completed trials (higher is better).
    chi_r2_numeric = pd.to_numeric(trial_df["chi_val_r2"], errors="coerce")
    trial_df["best_chi_val_r2_so_far"] = chi_r2_numeric.cummax()
    objective_numeric = pd.to_numeric(trial_df["objective_value"], errors="coerce")
    if objective_direction == "maximize":
        trial_df["best_objective_so_far"] = objective_numeric.cummax()
    else:
        trial_df["best_objective_so_far"] = objective_numeric.cummin()
    trial_df.to_csv(tuning_dir / "optuna_optimization_chi_r2.csv", index=False)
    trial_df.to_csv(tuning_dir / "optuna_optimization_objective.csv", index=False)
    _save_best_metric_vs_trial_figure(
        trial_df=trial_df,
        out_png=tuning_dir / "optuna_best_metric_by_trial.png",
        y_col="best_objective_so_far",
        y_label=f"Best {objective_name}",
        title=f"Best trial metric vs trial: {objective_name}",
        dpi=dpi,
        font_size=font_size,
    )

    best_params = study.best_params
    best_num_layers = int(best_params["num_layers"])
    best_hidden_sizes = [int(best_params[f"hidden_{i}"]) for i in range(best_num_layers)]
    best_dropout = float(best_params["dropout"])
    best_learning_rate = float(best_params["learning_rate"])
    best_weight_decay = float(best_params["weight_decay"])
    best_batch_size = int(best_params["batch_size"])
    best_finetune_last_layers = int(best_params.get("finetune_last_layers", train_cfg.finetune_last_layers))
    best_cv_eval = _evaluate_trial_with_cv(
        cv_folds=cv_folds,
        embedding_table=embedding_table,
        train_cfg=train_cfg,
        config=config,
        model_size=model_size,
        backbone_checkpoint=backbone_checkpoint,
        tokenizer=tokenizer,
        device=device,
        hidden_sizes=best_hidden_sizes,
        dropout=best_dropout,
        learning_rate=best_learning_rate,
        weight_decay=best_weight_decay,
        batch_size=best_batch_size,
        finetune_last_layers=best_finetune_last_layers,
        collect_val_predictions=True,
    )
    best_cv_eval["fold_metrics"].to_csv(tuning_dir / "best_trial_cv_fold_metrics.csv", index=False)
    best_cv_val_df = best_cv_eval.get("cv_val_predictions", pd.DataFrame())
    if isinstance(best_cv_val_df, pd.DataFrame) and not best_cv_val_df.empty:
        best_cv_val_df.to_csv(tuning_dir / "best_trial_cv_val_predictions.csv", index=False)
        _save_cv_parity_by_fold_figure(
            cv_val_df=best_cv_val_df,
            out_png=tuning_dir / "cv_parity_by_fold.png",
            dpi=dpi,
            font_size=font_size,
        )

    # Figure: trial-by-trial chi R2 and running best chi R2.
    apply_publication_figure_style(font_size=font_size, dpi=dpi, remove_titles=True)
    # Convert Series to NumPy arrays for matplotlib/pandas compatibility.
    trial_numbers = trial_df["trial"].to_numpy()
    chi_val_r2_values = pd.to_numeric(trial_df["chi_val_r2"], errors="coerce").to_numpy()
    best_chi_val_r2_values = pd.to_numeric(trial_df["best_chi_val_r2_so_far"], errors="coerce").to_numpy()
    objective_values = pd.to_numeric(trial_df["objective_value"], errors="coerce").to_numpy()
    best_objective_values = pd.to_numeric(trial_df["best_objective_so_far"], errors="coerce").to_numpy()
    if objective_name == "val_r2":
        chi_r2_plot_label = "Trial chi R2"
        best_chi_r2_plot_label = "Best chi R2 so far"
        chi_r2_ylabel = "Validation chi R2"
        chi_r2_title = "Optuna optimization process based on chi R2"
    else:
        chi_r2_plot_label = "Trial chi R2 (auxiliary)"
        best_chi_r2_plot_label = "Best chi R2 so far (auxiliary)"
        chi_r2_ylabel = "Validation chi R2 (auxiliary)"
        chi_r2_title = "Auxiliary validation chi R2 during Optuna"

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(trial_numbers, chi_val_r2_values, "o", color="#1f77b4", label=chi_r2_plot_label, alpha=0.85)
    ax.plot(
        trial_numbers,
        best_chi_val_r2_values,
        "-",
        color="#d62728",
        linewidth=2,
        label=best_chi_r2_plot_label,
    )
    ax.set_xlabel("Optuna trial")
    ax.set_ylabel(chi_r2_ylabel)
    ax.set_title(chi_r2_title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(tuning_dir / "optuna_optimization_chi_r2.png", dpi=dpi)
    plt.close(fig)

    # Figure: trial-by-trial objective and running best objective.
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(trial_numbers, objective_values, "o", color="#2a9d8f", label="Trial objective", alpha=0.85)
    ax.plot(
        trial_numbers,
        best_objective_values,
        "-",
        color="#e76f51",
        linewidth=2,
        label=f"Best objective so far ({'higher' if objective_direction == 'maximize' else 'lower'} better)",
    )
    ax.set_xlabel("Optuna trial")
    ax.set_ylabel("Objective value")
    ax.set_title(f"Optuna optimization objective: {objective_name}")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(tuning_dir / "optuna_optimization_objective.png", dpi=dpi)
    plt.close(fig)

    with open(tuning_dir / "optuna_best.json", "w") as f:
        invalid_trial_count = int(trial_df["invalid_metrics"].sum()) if "invalid_metrics" in trial_df.columns else 0
        json.dump(
            {
                "best_trial": int(study.best_trial.number),
                "objective": _describe_tuning_objective(train_cfg),
                "objective_name": objective_name,
                "objective_direction": objective_direction,
                "objective_value_at_best_trial": float(study.best_value),
                "best_value_r2": float(study.best_trial.user_attrs.get("cv_val_r2", np.nan)),
                "best_value_poly_nrmse_at_best_trial": float(study.best_trial.user_attrs.get("cv_val_poly_nrmse", np.nan)),
                "best_value_rmse_at_best_trial": float(study.best_trial.user_attrs.get("cv_val_rmse", np.nan)),
                "tuning_cv_folds_requested": int(train_cfg.tuning_cv_folds),
                "tuning_cv_folds_resolved": int(cv_info.get("resolved_folds", len(cv_folds))),
                "tuning_cv_strategy": str(cv_info.get("strategy", "unknown")),
                "invalid_trial_count": invalid_trial_count,
                "backbone_num_layers": int(backbone_num_layers),
                "finetune_last_layers_search_mode": finetune_mode,
                "finetune_last_layers_min": int(finetune_lo),
                "finetune_last_layers_max": int(finetune_hi),
                "best_params": study.best_params,
            },
            f,
            indent=2,
        )
    return study.best_params



def _save_history(history: Dict[str, List[float]], out_csv: Path) -> None:
    pd.DataFrame(history).to_csv(out_csv, index=False)


def _coerce_coeff_matrix(coefficients: np.ndarray, expected_rows: int, split: str) -> np.ndarray:
    """Normalize coefficient outputs to a 2D [rows, coeff_dims] matrix."""
    coeff = np.asarray(coefficients)

    if coeff.size == 0:
        return np.empty((expected_rows, 0), dtype=np.float32)

    if coeff.ndim == 0:
        coeff = coeff.reshape(1, 1)
    elif coeff.ndim == 1:
        if expected_rows <= 0:
            raise ValueError(
                f"Coefficient shape mismatch on split={split}: expected {expected_rows} rows, got 1D array of size {coeff.size}"
            )
        if coeff.size == len(COEFF_NAMES) and expected_rows == 1:
            coeff = coeff.reshape(1, len(COEFF_NAMES))
        elif coeff.size % expected_rows == 0:
            coeff = coeff.reshape(expected_rows, coeff.size // expected_rows)
        else:
            raise ValueError(
                f"Coefficient shape mismatch on split={split}: expected {expected_rows} rows, got 1D array of size {coeff.size}"
            )
    elif coeff.ndim != 2:
        raise ValueError(
            f"Coefficient shape mismatch on split={split}: expected 1D/2D array, got shape {coeff.shape}"
        )

    if coeff.shape[0] != expected_rows:
        if coeff.shape[1] == expected_rows:
            coeff = coeff.T
        else:
            raise ValueError(
                f"Coefficient row mismatch on split={split}: expected {expected_rows} rows, got shape {coeff.shape}"
            )

    return coeff


def _concat_non_empty_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate frames while avoiding pandas all-NA/empty concat deprecation warnings."""
    non_empty = [frame for frame in frames if not frame.empty]
    if non_empty:
        return pd.concat(non_empty, axis=0, ignore_index=True)
    if frames:
        return frames[0].iloc[0:0].copy()
    return pd.DataFrame()


def _clean_numeric_array(values: pd.Series | np.ndarray) -> np.ndarray:
    """Convert series/array to finite 1D float array."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _has_finite_history_values(history: Dict[str, List[float]], key: str) -> bool:
    values = history.get(key, [])
    if len(values) == 0:
        return False
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return bool(np.isfinite(arr).any())


def _get_legend_handles(legend) -> List[object]:
    """Matplotlib compatibility: prefer new legend_handles, fallback to legacy legendHandles."""
    handles = getattr(legend, "legend_handles", None)
    if handles is None:
        handles = getattr(legend, "legendHandles", [])
    return list(handles)


def _plot_kde_safe_1d(
    ax,
    values: pd.Series | np.ndarray,
    *,
    label: str,
    color: str,
    linewidth: float = 2.0,
    fill: bool = False,
    alpha: float = 0.3,
) -> bool:
    arr = _clean_numeric_array(values)
    if arr.size < 2 or np.isclose(np.std(arr), 0.0):
        return False
    try:
        kde_kwargs = {
            "x": np.asarray(arr, dtype=float),
            "ax": ax,
            "label": label,
            "color": color,
            "linewidth": linewidth,
            "fill": fill,
        }
        if fill:
            kde_kwargs["alpha"] = alpha
        sns.kdeplot(**kde_kwargs)
    except Exception as exc:
        if "Multi-dimensional indexing" not in str(exc):
            raise
        bins = int(np.clip(np.sqrt(arr.size), 10, 60))
        sns.histplot(
            x=np.asarray(arr, dtype=float),
            bins=bins,
            stat="density",
            element="step",
            fill=False,
            color=color,
            linewidth=linewidth,
            label=label,
            ax=ax,
        )
    return True


def _plot_class_prob_density_safe(ax, split_df: pd.DataFrame) -> None:
    try:
        sns.kdeplot(
            data=split_df,
            x="class_prob",
            hue="water_miscible",
            hue_order=[0, 1],
            palette=WATER_SOLUBLE_PALETTE,
            common_norm=False,
            fill=True,
            alpha=0.3,
            ax=ax,
        )
        legend = ax.get_legend()
        if legend is not None:
            legend.set_title(CLASS_DISPLAY_LEGEND_TITLE)
            for text in legend.get_texts():
                text.set_text(_class_display_label(text.get_text()))
        return
    except Exception as exc:
        if "Multi-dimensional indexing" not in str(exc):
            raise

    for class_value, color in [(1, "#1f77b4"), (0, "#d62728")]:
        sub = split_df.loc[split_df["water_miscible"] == class_value, "class_prob"]
        _plot_kde_safe_1d(
            ax=ax,
            values=sub,
            label=_class_display_label(class_value),
            color=color,
            linewidth=2.0,
            fill=False,
            alpha=0.3,
        )


def _save_binary_confusion_matrix_figure(
    sub: pd.DataFrame,
    out_png: Path,
    title: str,
    dpi: int,
) -> None:
    if sub.empty or "water_miscible" not in sub.columns:
        return
    y_true = pd.to_numeric(sub["water_miscible"], errors="coerce").fillna(0).to_numpy(dtype=int)
    if "class_pred" in sub.columns:
        y_pred = pd.to_numeric(sub["class_pred"], errors="coerce").fillna(0).to_numpy(dtype=int)
    elif "class_prob" in sub.columns:
        y_pred = (pd.to_numeric(sub["class_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=float) >= 0.5).astype(int)
    else:
        return
    if len(y_true) == 0:
        return

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cbar=False,
        cmap="Blues",
        square=True,
        xticklabels=CLASS_DISPLAY_TICKLABELS,
        yticklabels=CLASS_DISPLAY_TICKLABELS,
        ax=ax,
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"{title} (n={len(y_true)})")
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)



def _save_prediction_csvs(
    split_df: pd.DataFrame,
    predictions: Dict[str, Dict[str, np.ndarray]],
    out_dir: Path,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for split in ["train", "val", "test"]:
        sub = split_df[split_df["split"] == split].copy().reset_index(drop=True)
        pred = predictions[split]
        sub["chi_pred"] = pred["chi_pred"]
        sub["chi_error"] = sub["chi_pred"] - sub["chi"]
        if "logit" in pred:
            sub["class_logit"] = pred["logit"]
        if "prob" in pred:
            sub["class_prob"] = pred["prob"]
            sub["class_pred"] = (sub["class_prob"] >= 0.5).astype(int)

        sub.to_csv(out_dir / f"chi_predictions_{split}.csv", index=False)
        frames.append(sub)

    all_df = _concat_non_empty_frames(frames)
    all_df.to_csv(out_dir / "chi_predictions_all.csv", index=False)
    return all_df



def _collect_metrics(pred_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    class_rows = []
    polymer_rows = []
    polymer_r2_rows = []
    low_chi_rows = []

    for split, sub in pred_df.groupby("split"):
        reg = regression_metrics(sub["chi"], sub["chi_pred"])
        cls = classification_metrics(sub["water_miscible"], sub["class_prob"])
        hit = hit_metrics(sub["chi_error"], epsilons=[0.02, 0.05, 0.1, 0.2])

        row = {"split": split}
        row.update(reg)
        row.update(cls)
        row.update(hit)
        rows.append(row)

        group_metrics = metrics_by_group(
            sub,
            y_true_col="chi",
            y_pred_col="chi_pred",
            group_col="water_miscible",
        )
        group_metrics.insert(0, "split", split)
        class_rows.append(group_metrics)

        poly = (
            sub.groupby(["polymer_id", "Polymer"], as_index=False)[["chi", "chi_pred"]]
            .mean()
            .rename(columns={"chi": "chi_true_mean", "chi_pred": "chi_pred_mean"})
        )
        poly_metric = regression_metrics(poly["chi_true_mean"], poly["chi_pred_mean"], prefix="poly_")
        poly_row = {"split": split}
        poly_row.update(poly_metric)
        polymer_rows.append(poly_row)

        for class_value, class_sub in sub.groupby("water_miscible", sort=True):
            dist_row = {
                "split": split,
                "water_miscible": int(class_value),
                "class_label": _class_display_label(class_value),
            }
            dist_row.update(
                polymer_r2_distribution(
                    y_true=class_sub["chi"],
                    y_pred=class_sub["chi_pred"],
                    polymer_ids=class_sub["polymer_id"],
                )
            )
            polymer_r2_rows.append(dist_row)

    test_df = pred_df[pred_df["split"] == "test"].copy()
    if not test_df.empty:
        for quantile in [0.10, 0.25, 0.50, 0.75]:
            chi_cutoff = float(test_df["chi"].quantile(quantile))
            sub = test_df[test_df["chi"] <= chi_cutoff].copy()
            if sub.empty:
                continue
            reg = regression_metrics(sub["chi"], sub["chi_pred"])
            low_row = {
                "split": "test",
                "subset": f"bottom_{int(round(quantile * 100)):02d}pct",
                "quantile": float(quantile),
                "chi_threshold": chi_cutoff,
                "rank_correlation": reg.get("spearman_r", np.nan),
            }
            low_row.update(reg)
            low_chi_rows.append(low_row)

    pd.DataFrame(rows).to_csv(out_dir / "chi_metrics_overall.csv", index=False)
    pd.concat(class_rows, ignore_index=True).to_csv(out_dir / "chi_metrics_by_class.csv", index=False)
    pd.DataFrame(polymer_rows).to_csv(out_dir / "chi_metrics_polymer_level.csv", index=False)
    pd.DataFrame(polymer_r2_rows).to_csv(out_dir / "chi_metrics_polymer_r2_distribution.csv", index=False)
    pd.DataFrame(low_chi_rows).to_csv(out_dir / "chi_metrics_low_chi_quantiles.csv", index=False)



def _plot_parity_panel(ax, sub: pd.DataFrame, split: str, show_legend: bool) -> None:
    if sub.empty:
        ax.set_axis_off()
        ax.set_title(f"{split.upper()} (empty)")
        return

    sns.scatterplot(
        data=sub,
        x="chi",
        y="chi_pred",
        hue="water_miscible",
        palette=WATER_SOLUBLE_PALETTE,
        alpha=0.75,
        s=18,
        ax=ax,
        legend=show_legend,
    )
    lo = float(min(sub["chi"].min(), sub["chi_pred"].min()))
    hi = float(max(sub["chi"].max(), sub["chi_pred"].max()))
    span = max(hi - lo, 1e-8)
    pad = 0.04 * span
    lo_plot = lo - pad
    hi_plot = hi + pad
    ax.plot([lo_plot, hi_plot], [lo_plot, hi_plot], "k--", linewidth=1.1)
    ax.set_xlim(lo_plot, hi_plot)
    ax.set_ylim(lo_plot, hi_plot)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("True χ")
    ax.set_ylabel("Predicted χ")

    reg = regression_metrics(sub["chi"], sub["chi_pred"])
    metrics_text = (
        f"MAE={reg['mae']:.3f}\n"
        f"RMSE={reg['rmse']:.3f}\n"
        f"R2={reg['r2']:.3f}"
    )
    ax.text(
        0.03,
        0.97,
        metrics_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#666666", "alpha": 0.92},
    )
    ax.set_title(f"{split.upper()} parity (n={len(sub)})")
    legend = ax.get_legend()
    if legend is not None:
        if show_legend:
            legend.remove()
            handles = [
                Line2D([], [], marker="o", linestyle="None", color=WATER_SOLUBLE_PALETTE[0], markersize=6),
                Line2D([], [], marker="o", linestyle="None", color=WATER_SOLUBLE_PALETTE[1], markersize=6),
            ]
            ax.legend(
                handles=handles,
                labels=_class_display_labels([0, 1]),
                title=CLASS_DISPLAY_LEGEND_TITLE,
                loc="upper right",
                frameon=True,
                fancybox=True,
                framealpha=0.92,
                facecolor="white",
                edgecolor="#666666",
            )
        else:
            legend.remove()


def _plot_classifier_parity_panel(ax, sub: pd.DataFrame, split: str, show_legend: bool) -> None:
    if sub.empty:
        ax.set_axis_off()
        ax.set_title(f"{split.upper()} (empty)")
        return

    plot_df = sub.copy()
    rng = np.random.default_rng(0)
    jitter = rng.normal(loc=0.0, scale=0.03, size=len(plot_df))
    plot_df["water_miscible_jitter"] = np.clip(plot_df["water_miscible"].to_numpy(dtype=float) + jitter, -0.08, 1.08)

    sns.scatterplot(
        data=plot_df,
        x="water_miscible_jitter",
        y="class_prob",
        hue="water_miscible",
        palette={1: "#1f77b4", 0: "#d62728"},
        alpha=0.75,
        s=18,
        ax=ax,
        legend=show_legend,
    )
    ax.plot([0.0, 1.0], [0.0, 1.0], "k--", linewidth=1.1)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(CLASS_DISPLAY_TICKLABELS)
    ax.set_xlabel("True class")
    ax.set_ylabel("Predicted water-miscible probability")

    cls = classification_metrics(sub["water_miscible"], sub["class_prob"])
    metrics_text = (
        f"BalAcc={cls['balanced_accuracy']:.3f}\n"
        f"AUROC={cls['auroc']:.3f}\n"
        f"AUPRC={cls['auprc']:.3f}\n"
        f"Brier={cls['brier']:.3f}"
    )
    ax.text(
        0.03,
        0.97,
        metrics_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#666666", "alpha": 0.92},
    )
    ax.set_title(f"{split.upper()} classifier parity (n={len(sub)})")

    legend = ax.get_legend()
    if legend is not None:
        if show_legend:
            handles = _get_legend_handles(legend)
            labels = [t.get_text() for t in legend.get_texts()]
            legend.remove()
            ax.legend(
                handles=handles,
                labels=_class_display_labels(labels),
                title=CLASS_DISPLAY_LEGEND_TITLE,
                loc="upper right",
                frameon=True,
                fancybox=True,
                framealpha=0.92,
                facecolor="white",
                edgecolor="#666666",
            )
        else:
            legend.remove()


def _make_figures(history: Dict[str, List[float]], pred_df: pd.DataFrame, fig_dir: Path, dpi: int, font_size: int) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    apply_publication_figure_style(font_size=font_size, dpi=dpi, remove_titles=True)

    # Loss curve
    has_val_curve = _has_finite_history_values(history, "val_loss")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(history["epoch"], history["train_loss"], label="Train loss")
    if has_val_curve:
        ax.plot(history["epoch"], history["val_loss"], label="Val loss")
        ax.set_title("Step4 chi training loss")
    else:
        ax.set_title("Step4 chi training loss (final fit: train+val)")
        ax.text(
            0.03,
            0.97,
            "No validation split in final fit",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#666666", "alpha": 0.92},
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(fig_dir / "chi_loss_curve.png", dpi=dpi)
    plt.close(fig)

    train = pred_df[pred_df["split"] == "train"].copy()
    fig, ax = plt.subplots(figsize=(6, 5))
    _plot_parity_panel(ax, sub=train, split="train", show_legend=True)
    fig.tight_layout()
    fig.savefig(fig_dir / "chi_parity_train.png", dpi=dpi)
    plt.close(fig)
    _save_binary_confusion_matrix_figure(
        sub=train,
        out_png=fig_dir / "chi_classifier_confusion_matrix_train.png",
        title="Classifier confusion matrix (train)",
        dpi=dpi,
    )

    test = pred_df[pred_df["split"] == "test"].copy()
    fig, ax = plt.subplots(figsize=(6, 5))
    _plot_parity_panel(ax, sub=test, split="test", show_legend=True)
    fig.tight_layout()
    fig.savefig(fig_dir / "chi_parity_test.png", dpi=dpi)
    plt.close(fig)
    _save_binary_confusion_matrix_figure(
        sub=test,
        out_png=fig_dir / "chi_classifier_confusion_matrix_test.png",
        title="Classifier confusion matrix (test)",
        dpi=dpi,
    )

    # Residual histogram by split
    fig, ax = plt.subplots(figsize=(6, 5))
    plotted_any = False
    for split, color in [("train", "#4c78a8"), ("val", "#f58518"), ("test", "#54a24b")]:
        sub = pred_df[pred_df["split"] == split]
        plotted_any = _plot_kde_safe_1d(
            ax=ax,
            values=sub["chi_error"],
            label=split,
            color=color,
            linewidth=2.0,
            fill=False,
            alpha=0.3,
        ) or plotted_any
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("χ prediction error")
    ax.set_title("Residual distribution by split")
    if plotted_any:
        ax.legend(
            loc="upper right",
            frameon=True,
            fancybox=True,
            framealpha=0.92,
            facecolor="white",
            edgecolor="#666666",
        )
    else:
        ax.text(0.5, 0.5, "Insufficient residual variance for KDE", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(fig_dir / "chi_residual_distribution.png", dpi=dpi)
    plt.close(fig)

    for split_name, split_df in [("train", train), ("test", test)]:
        if split_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(6, 5))
        _plot_class_prob_density_safe(ax=ax, split_df=split_df)
        ax.set_xlabel("Predicted water-miscible probability")
        ax.set_title(f"Class probability distribution ({split_name})")
        legend = ax.get_legend()
        if legend is not None:
            handles = _get_legend_handles(legend)
            labels = [t.get_text() for t in legend.get_texts()]
            legend.remove()
            ax.legend(
                handles=handles,
                labels=_class_display_labels(labels),
                title=CLASS_DISPLAY_LEGEND_TITLE,
                loc="upper center",
                ncol=min(2, len(labels)),
                frameon=True,
                fancybox=True,
                framealpha=0.92,
                facecolor="white",
                edgecolor="#666666",
            )
        fig.tight_layout()
        fig.savefig(fig_dir / f"chi_class_prob_distribution_{split_name}.png", dpi=dpi)
        plt.close(fig)

        y_true = split_df["water_miscible"].to_numpy(dtype=int)
        y_prob = split_df["class_prob"].to_numpy(dtype=float)
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            precision, recall, _ = precision_recall_curve(y_true, y_prob)

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, color="#1f77b4", linewidth=2)
            ax.plot([0, 1], [0, 1], "k--", linewidth=1)
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.set_title(f"Classifier ROC ({split_name})")
            fig.tight_layout()
            fig.savefig(fig_dir / f"chi_classifier_roc_{split_name}.png", dpi=dpi)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(recall, precision, color="#d62728", linewidth=2)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(f"Classifier PR ({split_name})")
            fig.tight_layout()
            fig.savefig(fig_dir / f"chi_classifier_pr_{split_name}.png", dpi=dpi)
            plt.close(fig)



def _save_coefficient_summary(
    model: nn.Module,
    embedding_cache_df: pd.DataFrame,
    out_csv: Path,
    device: str,
    tokenizer=None,
    timestep: int = 1,
) -> None:
    raise RuntimeError(
        "Coefficient export is not available after switching Step4_1 regression to direct chi prediction."
    )


def run_classifier_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    gradient_clip_norm: float = 0.0,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    losses: List[float] = []
    for batch in loader:
        label = batch["label"].to(device)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        if "embedding" in batch:
            embedding = batch["embedding"].to(device)
            out = model.compute_loss(embedding=embedding, class_label=label)
        else:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model.compute_loss(
                input_ids=input_ids,
                attention_mask=attention_mask,
                class_label=label,
            )
        loss = out["loss"]
        if train_mode:
            loss.backward()
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(gradient_clip_norm))
            optimizer.step()
        losses.append(float(loss.item()))
    return {"loss": float(np.mean(losses)) if losses else np.nan}


@torch.no_grad()
def predict_classifier_split(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, np.ndarray]:
    model.eval()
    logits: List[np.ndarray] = []
    probs: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for batch in loader:
        label = batch["label"].to(device)
        if "embedding" in batch:
            embedding = batch["embedding"].to(device)
            out = model(embedding=embedding)
        else:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        logit = out["class_logit"]
        logits.append(logit.detach().cpu().numpy())
        probs.append(torch.sigmoid(logit).detach().cpu().numpy())
        labels.append(label.detach().cpu().numpy())
    return {
        "label": np.concatenate(labels, axis=0) if labels else np.array([]),
        "logit": np.concatenate(logits, axis=0) if logits else np.array([]),
        "prob": np.concatenate(probs, axis=0) if probs else np.array([]),
    }


def train_one_classifier_model(
    split_df: pd.DataFrame,
    embedding_table: Optional[np.ndarray],
    train_cfg: TrainConfig,
    device: str,
    hidden_sizes: List[int],
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    num_epochs: int,
    patience: int,
    config: Optional[Dict] = None,
    model_size: Optional[str] = None,
    backbone_checkpoint: Optional[str] = None,
    tokenizer=None,
    finetune_last_layers: int = 0,
    timestep_for_embedding: int = 1,
    backbone_split_mode: Optional[str] = None,
) -> Tuple[nn.Module, Dict[str, List[float]], Dict[str, Dict[str, np.ndarray]]]:
    if finetune_last_layers > 0:
        if config is None:
            raise ValueError("config is required when finetune_last_layers > 0")
        if tokenizer is None:
            raise ValueError("tokenizer is required when finetune_last_layers > 0")
        dataloaders = make_token_dataloaders(
            split_df=split_df,
            tokenizer=tokenizer,
            batch_size=batch_size,
            shuffle_train=True,
        )
        _, backbone, _ = load_backbone_from_step1(
            config=config,
            model_size=model_size,
            split_mode=backbone_split_mode if backbone_split_mode is not None else train_cfg.split_mode,
            checkpoint_path=backbone_checkpoint,
            device=device,
        )
        _set_finetune_last_layers(backbone, finetune_last_layers=finetune_last_layers)
        classifier_head = SolubilityClassifier(
            embedding_dim=int(backbone.hidden_size),
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        )
        model = BackboneSolubilityClassifierModel(
            backbone=backbone,
            classifier_head=classifier_head,
            timestep=int(timestep_for_embedding),
            pooling="mean",
        ).to(device)
    else:
        if embedding_table is None:
            raise ValueError("embedding_table is required when finetune_last_layers == 0")
        dataloaders = make_dataloaders(
            split_df=split_df,
            embedding_table=embedding_table,
            batch_size=batch_size,
            shuffle_train=True,
        )
        model = SolubilityClassifier(
            embedding_dim=int(embedding_table.shape[1]),
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        ).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = None
    if train_cfg.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(num_epochs)),
            eta_min=float(train_cfg.scheduler_min_lr),
        )

    history = {
        "epoch": [],
        "learning_rate": [],
        "train_loss": [],
        "val_loss": [],
        "val_balanced_accuracy": [],
    }

    best_state = None
    best_val_balanced_accuracy = -np.inf
    best_val_loss = np.inf
    wait = 0
    has_val = len(dataloaders["val"].dataset) > 0

    for epoch in range(1, num_epochs + 1):
        train_stats = run_classifier_epoch(
            model=model,
            loader=dataloaders["train"],
            optimizer=optimizer,
            device=device,
            gradient_clip_norm=float(train_cfg.gradient_clip_norm),
        )
        if has_val:
            val_stats = run_classifier_epoch(
                model=model,
                loader=dataloaders["val"],
                optimizer=None,
                device=device,
                gradient_clip_norm=0.0,
            )
            val_pred = predict_classifier_split(model, dataloaders["val"], device)
            val_cls = classification_metrics(val_pred["label"], val_pred["prob"])
            val_bal_acc = float(val_cls["balanced_accuracy"])
        else:
            val_stats = {"loss": np.nan}
            val_bal_acc = np.nan

        history["epoch"].append(epoch)
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
        history["train_loss"].append(train_stats["loss"])
        history["val_loss"].append(val_stats["loss"])
        history["val_balanced_accuracy"].append(val_bal_acc)

        if has_val:
            val_loss = float(val_stats["loss"])
            improved = (
                val_bal_acc > best_val_balanced_accuracy + 1e-12
                or (
                    np.isclose(val_bal_acc, best_val_balanced_accuracy, atol=1e-12, rtol=1e-12)
                    and val_loss < best_val_loss
                )
            )
            if improved:
                best_val_balanced_accuracy = val_bal_acc
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break
        else:
            # No validation split in final all-train fitting mode: keep latest state.
            best_state = copy.deepcopy(model.state_dict())
        if scheduler is not None:
            scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)

    if finetune_last_layers > 0:
        pred_loaders = make_token_dataloaders(
            split_df=split_df,
            tokenizer=tokenizer,
            batch_size=batch_size,
            shuffle_train=False,
        )
    else:
        pred_loaders = make_dataloaders(
            split_df=split_df,
            embedding_table=embedding_table,
            batch_size=batch_size,
            shuffle_train=False,
        )
    predictions = {
        split: predict_classifier_split(model, loader, device)
        for split, loader in pred_loaders.items()
    }
    return model, history, predictions


def _evaluate_classifier_trial_with_cv(
    cv_folds: List[pd.DataFrame],
    embedding_table: Optional[np.ndarray],
    train_cfg: TrainConfig,
    config: Dict,
    model_size: Optional[str],
    backbone_checkpoint: Optional[str],
    tokenizer,
    device: str,
    hidden_sizes: List[int],
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    finetune_last_layers: int,
    backbone_split_mode: Optional[str] = None,
) -> Dict[str, object]:
    fold_rows = []
    for fold_id, fold_df in enumerate(cv_folds, start=1):
        _, _, preds = train_one_classifier_model(
            split_df=fold_df,
            embedding_table=embedding_table,
            train_cfg=train_cfg,
            device=device,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            num_epochs=train_cfg.tuning_epochs,
            patience=train_cfg.tuning_patience,
            config=config,
            model_size=model_size,
            backbone_checkpoint=backbone_checkpoint,
            tokenizer=tokenizer,
            finetune_last_layers=finetune_last_layers,
            timestep_for_embedding=train_cfg.timestep_for_embedding,
            backbone_split_mode=backbone_split_mode,
        )
        val_pred = preds["val"]
        val_cls = classification_metrics(val_pred["label"], val_pred["prob"])
        fold_rows.append(
            {
                "fold": fold_id,
                "val_n": int(len(val_pred["label"])),
                "val_balanced_accuracy": float(val_cls["balanced_accuracy"]),
                "val_auroc": float(val_cls["auroc"]),
                "val_auprc": float(val_cls["auprc"]),
                "val_brier": float(val_cls["brier"]),
            }
        )
    fold_metrics_df = pd.DataFrame(fold_rows)
    return {
        "cv_val_balanced_accuracy": float(np.nanmean(fold_metrics_df["val_balanced_accuracy"])) if not fold_metrics_df.empty else np.nan,
        "cv_val_auroc": float(np.nanmean(fold_metrics_df["val_auroc"])) if not fold_metrics_df.empty else np.nan,
        "cv_val_auprc": float(np.nanmean(fold_metrics_df["val_auprc"])) if not fold_metrics_df.empty else np.nan,
        "cv_val_brier": float(np.nanmean(fold_metrics_df["val_brier"])) if not fold_metrics_df.empty else np.nan,
        "fold_metrics": fold_metrics_df,
    }


def tune_classifier_hyperparameters(
    split_df: pd.DataFrame,
    embedding_table: Optional[np.ndarray],
    train_cfg: TrainConfig,
    config: Dict,
    model_size: Optional[str],
    backbone_num_layers: int,
    backbone_checkpoint: Optional[str],
    tokenizer,
    device: str,
    tuning_dir: Path,
    dpi: int = 300,
    font_size: int = 16,
    backbone_split_mode: Optional[str] = None,
) -> Dict:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna is required for --tune. Install it with `pip install optuna` or disable tuning."
        ) from exc

    tuning_dir.mkdir(parents=True, exist_ok=True)
    search_space = train_cfg.optuna_search_space

    def _as_list(key: str, default: List[float]) -> List[float]:
        values = search_space.get(key, default)
        if isinstance(values, list) and len(values) > 0:
            return values
        return default

    def _as_int_list(key: str, default: List[int]) -> List[int]:
        raw = _as_list(key, [int(v) for v in default])
        parsed = []
        for item in raw:
            try:
                parsed.append(int(float(item)))
            except Exception:
                continue
        return parsed if parsed else [int(v) for v in default]

    def _as_float_list(key: str, default: List[float]) -> List[float]:
        raw = _as_list(key, [float(v) for v in default])
        parsed = []
        for item in raw:
            try:
                parsed.append(float(item))
            except Exception:
                continue
        return parsed if parsed else [float(v) for v in default]

    num_layers_space = _as_int_list("num_layers", [1, 2, 3])
    hidden_units_space = _as_int_list("hidden_units", [64, 128, 256, 512])
    dropout_space = _as_float_list("dropout", [0.0, 0.1, 0.2, 0.3])
    lr_space = _as_float_list("learning_rate", [1e-4, 5e-3])
    wd_space = _as_float_list("weight_decay", [1e-7, 1e-3])
    batch_size_space = _as_int_list("batch_size", [16, 32, 64, 128, 256])
    lr_log = bool(search_space.get("learning_rate_log", True))
    wd_log = bool(search_space.get("weight_decay_log", True))
    finetune_raw = search_space.get("finetune_last_layers", [0, int(backbone_num_layers)])
    if isinstance(finetune_raw, list) and len(finetune_raw) > 0:
        finetune_space = []
        for item in finetune_raw:
            try:
                finetune_space.append(int(float(item)))
            except Exception:
                continue
    else:
        finetune_space = [0, int(backbone_num_layers)]
    if len(finetune_space) == 0:
        finetune_space = [0, int(backbone_num_layers)]
    if len(finetune_space) == 2:
        finetune_lo = max(0, min(finetune_space))
        finetune_hi = min(int(backbone_num_layers), max(finetune_space))
        if finetune_hi < finetune_lo:
            clamped = max(0, min(int(backbone_num_layers), finetune_lo))
            finetune_lo = clamped
            finetune_hi = clamped
        finetune_mode = "range"
        finetune_values = []
    else:
        finetune_mode = "categorical"
        finetune_values = sorted(set(int(v) for v in finetune_space if 0 <= int(v) <= int(backbone_num_layers)))
        if len(finetune_values) == 0:
            finetune_values = [0, int(backbone_num_layers)]
        finetune_lo = min(finetune_values)
        finetune_hi = max(finetune_values)
    if finetune_hi > 0 and tokenizer is None:
        raise ValueError(
            "Step4_2 Optuna search includes finetune_last_layers > 0, but tokenizer is unavailable. "
            "Provide Step1 backbone assets/checkpoint so tokenizer can be loaded."
        )

    cv_folds, cv_info = _build_tuning_cv_folds(split_df=split_df, train_cfg=train_cfg)
    _summarize_tuning_cv_folds(cv_folds).to_csv(tuning_dir / "optuna_tuning_cv_folds.csv", index=False)

    objective_name = "val_balanced_accuracy"
    objective_direction = "maximize"

    def objective(trial: optuna.Trial) -> float:
        if len(num_layers_space) == 2:
            lo = int(min(num_layers_space))
            hi = int(max(num_layers_space))
            num_layers = trial.suggest_int("num_layers", lo, hi)
        else:
            num_layers = int(trial.suggest_categorical("num_layers", [int(v) for v in num_layers_space]))
        hidden_sizes = [int(trial.suggest_categorical(f"hidden_{i}", hidden_units_space)) for i in range(num_layers)]
        dropout = float(trial.suggest_categorical("dropout", dropout_space))
        if len(lr_space) == 2:
            lr = float(trial.suggest_float("learning_rate", min(lr_space), max(lr_space), log=lr_log))
        else:
            lr = float(trial.suggest_categorical("learning_rate", lr_space))
        if len(wd_space) == 2:
            wd = float(trial.suggest_float("weight_decay", min(wd_space), max(wd_space), log=wd_log))
        else:
            wd = float(trial.suggest_categorical("weight_decay", wd_space))
        batch_size = int(trial.suggest_categorical("batch_size", batch_size_space))
        if finetune_mode == "range":
            finetune_last_layers = int(trial.suggest_int("finetune_last_layers", finetune_lo, finetune_hi))
        else:
            finetune_last_layers = int(trial.suggest_categorical("finetune_last_layers", finetune_values))

        cv_eval = _evaluate_classifier_trial_with_cv(
            cv_folds=cv_folds,
            embedding_table=embedding_table,
            train_cfg=train_cfg,
            config=config,
            model_size=model_size,
            backbone_checkpoint=backbone_checkpoint,
            tokenizer=tokenizer,
            device=device,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            learning_rate=lr,
            weight_decay=wd,
            batch_size=batch_size,
            finetune_last_layers=finetune_last_layers,
            backbone_split_mode=backbone_split_mode,
        )
        val_bal_acc = float(cv_eval["cv_val_balanced_accuracy"])
        invalid_metrics = int(not np.isfinite(val_bal_acc))
        trial.set_user_attr("cv_val_balanced_accuracy", val_bal_acc)
        trial.set_user_attr("cv_val_auroc", float(cv_eval["cv_val_auroc"]))
        trial.set_user_attr("cv_val_auprc", float(cv_eval["cv_val_auprc"]))
        trial.set_user_attr("cv_val_brier", float(cv_eval["cv_val_brier"]))
        trial.set_user_attr("cv_n_folds", int(len(cv_folds)))
        trial.set_user_attr("tuning_objective", objective_name)
        trial.set_user_attr("invalid_metrics", invalid_metrics)
        if invalid_metrics:
            return -1.0e12
        return val_bal_acc

    study = optuna.create_study(direction=objective_direction)
    study.optimize(objective, n_trials=train_cfg.n_trials, show_progress_bar=True)

    trials = []
    for t in study.trials:
        row = {
            "trial": t.number,
            "state": str(t.state),
            "objective_name": objective_name,
            "objective_direction": objective_direction,
            "objective_value": t.value,
            "val_balanced_accuracy": t.user_attrs.get("cv_val_balanced_accuracy", np.nan),
            "val_auroc": t.user_attrs.get("cv_val_auroc", np.nan),
            "val_auprc": t.user_attrs.get("cv_val_auprc", np.nan),
            "val_brier": t.user_attrs.get("cv_val_brier", np.nan),
            "invalid_metrics": int(t.user_attrs.get("invalid_metrics", 0)),
            "cv_n_folds": int(t.user_attrs.get("cv_n_folds", len(cv_folds))),
        }
        row.update(t.params)
        trials.append(row)
    trial_df = pd.DataFrame(trials).sort_values("trial").reset_index(drop=True)
    trial_df.to_csv(tuning_dir / "optuna_trials.csv", index=False)
    objective_numeric = pd.to_numeric(trial_df["objective_value"], errors="coerce")
    trial_df["best_objective_so_far"] = objective_numeric.cummax()
    trial_df.to_csv(tuning_dir / "optuna_optimization_objective.csv", index=False)
    _save_best_metric_vs_trial_figure(
        trial_df=trial_df,
        out_png=tuning_dir / "optuna_best_metric_by_trial.png",
        y_col="best_objective_so_far",
        y_label=f"Best {objective_name}",
        title=f"Best trial metric vs trial: {objective_name}",
        dpi=dpi,
        font_size=font_size,
    )

    apply_publication_figure_style(font_size=font_size, dpi=dpi, remove_titles=True)
    # Convert Series to NumPy arrays for matplotlib/pandas compatibility.
    trial_numbers = trial_df["trial"].to_numpy()
    objective_values = pd.to_numeric(trial_df["objective_value"], errors="coerce").to_numpy()
    best_objective_values = pd.to_numeric(trial_df["best_objective_so_far"], errors="coerce").to_numpy()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(trial_numbers, objective_values, "o", color="#2a9d8f", label="Trial balanced accuracy", alpha=0.85)
    ax.plot(
        trial_numbers,
        best_objective_values,
        "-",
        color="#e76f51",
        linewidth=2,
        label="Best balanced accuracy so far",
    )
    ax.set_xlabel("Optuna trial")
    ax.set_ylabel("Validation balanced accuracy")
    ax.set_title("Optuna optimization objective: val_balanced_accuracy")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(tuning_dir / "optuna_optimization_objective.png", dpi=dpi)
    plt.close(fig)

    with open(tuning_dir / "optuna_best.json", "w") as f:
        invalid_trial_count = int(trial_df["invalid_metrics"].sum()) if "invalid_metrics" in trial_df.columns else 0
        json.dump(
            {
                "best_trial": int(study.best_trial.number),
                "objective": "maximize_val_balanced_accuracy",
                "objective_name": objective_name,
                "objective_direction": objective_direction,
                "objective_value_at_best_trial": float(study.best_value),
                "best_value_balanced_accuracy": float(study.best_trial.user_attrs.get("cv_val_balanced_accuracy", np.nan)),
                "best_value_auroc": float(study.best_trial.user_attrs.get("cv_val_auroc", np.nan)),
                "best_value_auprc": float(study.best_trial.user_attrs.get("cv_val_auprc", np.nan)),
                "best_value_brier": float(study.best_trial.user_attrs.get("cv_val_brier", np.nan)),
                "tuning_cv_folds_requested": int(train_cfg.tuning_cv_folds),
                "tuning_cv_folds_resolved": int(cv_info.get("resolved_folds", len(cv_folds))),
                "tuning_cv_strategy": str(cv_info.get("strategy", "unknown")),
                "finetune_last_layers_search_mode": finetune_mode,
                "finetune_last_layers_min": int(finetune_lo),
                "finetune_last_layers_max": int(finetune_hi),
                "invalid_trial_count": invalid_trial_count,
                "best_params": study.best_params,
            },
            f,
            indent=2,
        )
    return study.best_params


def _save_classifier_prediction_csvs(
    split_df: pd.DataFrame,
    predictions: Dict[str, Dict[str, np.ndarray]],
    out_dir: Path,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for split in ["train", "val", "test"]:
        sub = split_df[split_df["split"] == split].copy().reset_index(drop=True)
        pred = predictions[split]
        sub["class_logit"] = pred["logit"]
        sub["class_prob"] = pred["prob"]
        sub["class_pred"] = (sub["class_prob"] >= 0.5).astype(int)
        sub.to_csv(out_dir / f"class_predictions_{split}.csv", index=False)
        frames.append(sub)
    all_df = _concat_non_empty_frames(frames)
    all_df.to_csv(out_dir / "class_predictions_all.csv", index=False)
    return all_df


def _collect_classifier_metrics(pred_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, sub in pred_df.groupby("split"):
        cls = classification_metrics(sub["water_miscible"], sub["class_prob"])
        row = {"split": split}
        row.update(cls)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "class_metrics_overall.csv", index=False)


def _save_regression_prediction_csvs(
    split_df: pd.DataFrame,
    predictions: Dict[str, Dict[str, np.ndarray]],
    out_dir: Path,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for split in ["train", "val", "test"]:
        sub = split_df[split_df["split"] == split].copy().reset_index(drop=True)
        pred = predictions[split]
        sub["chi_pred"] = pred["chi_pred"]
        sub["chi_error"] = sub["chi_pred"] - sub["chi"]
        sub.to_csv(out_dir / f"chi_predictions_{split}.csv", index=False)
        frames.append(sub)
    all_df = _concat_non_empty_frames(frames)
    all_df.to_csv(out_dir / "chi_predictions_all.csv", index=False)
    return all_df


def _collect_regression_metrics(pred_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    class_rows = []
    polymer_rows = []
    polymer_r2_rows = []
    low_chi_rows = []
    for split, sub in pred_df.groupby("split"):
        reg = regression_metrics(sub["chi"], sub["chi_pred"])
        hit = hit_metrics(sub["chi_error"], epsilons=[0.02, 0.05, 0.1, 0.2])
        row = {"split": split}
        row.update(reg)
        row.update(hit)
        rows.append(row)

        group_metrics = metrics_by_group(
            sub,
            y_true_col="chi",
            y_pred_col="chi_pred",
            group_col="water_miscible",
        )
        group_metrics.insert(0, "split", split)
        class_rows.append(group_metrics)

        poly = (
            sub.groupby(["polymer_id", "Polymer"], as_index=False)[["chi", "chi_pred"]]
            .mean()
            .rename(columns={"chi": "chi_true_mean", "chi_pred": "chi_pred_mean"})
        )
        poly_metric = regression_metrics(poly["chi_true_mean"], poly["chi_pred_mean"], prefix="poly_")
        poly_row = {"split": split}
        poly_row.update(poly_metric)
        polymer_rows.append(poly_row)

        for class_value, class_sub in sub.groupby("water_miscible", sort=True):
            dist_row = {
                "split": split,
                "water_miscible": int(class_value),
                "class_label": _class_display_label(class_value),
            }
            dist_row.update(
                polymer_r2_distribution(
                    y_true=class_sub["chi"],
                    y_pred=class_sub["chi_pred"],
                    polymer_ids=class_sub["polymer_id"],
                )
            )
            polymer_r2_rows.append(dist_row)

    test_df = pred_df[pred_df["split"] == "test"].copy()
    if not test_df.empty:
        for quantile in [0.10, 0.25, 0.50, 0.75]:
            chi_cutoff = float(test_df["chi"].quantile(quantile))
            sub = test_df[test_df["chi"] <= chi_cutoff].copy()
            if sub.empty:
                continue
            reg = regression_metrics(sub["chi"], sub["chi_pred"])
            low_row = {
                "split": "test",
                "subset": f"bottom_{int(round(quantile * 100)):02d}pct",
                "quantile": float(quantile),
                "chi_threshold": chi_cutoff,
                "rank_correlation": reg.get("spearman_r", np.nan),
            }
            low_row.update(reg)
            low_chi_rows.append(low_row)

    pd.DataFrame(rows).to_csv(out_dir / "chi_metrics_overall.csv", index=False)
    pd.concat(class_rows, ignore_index=True).to_csv(out_dir / "chi_metrics_by_class.csv", index=False)
    pd.DataFrame(polymer_rows).to_csv(out_dir / "chi_metrics_polymer_level.csv", index=False)
    pd.DataFrame(polymer_r2_rows).to_csv(out_dir / "chi_metrics_polymer_r2_distribution.csv", index=False)
    pd.DataFrame(low_chi_rows).to_csv(out_dir / "chi_metrics_low_chi_quantiles.csv", index=False)


def _make_regression_figures(
    history: Dict[str, List[float]],
    pred_df: pd.DataFrame,
    fig_dir: Path,
    dpi: int,
    font_size: int,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    apply_publication_figure_style(font_size=font_size, dpi=dpi, remove_titles=True)

    has_val_curve = _has_finite_history_values(history, "val_loss")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(history["epoch"], history["train_loss"], label="Train loss")
    if has_val_curve:
        ax.plot(history["epoch"], history["val_loss"], label="Val loss")
        ax.set_title("Step4_1 regression loss")
    else:
        ax.set_title("Step4_1 regression loss (final fit: train+val)")
        ax.text(
            0.03,
            0.97,
            "No validation split in final fit",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#666666", "alpha": 0.92},
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(fig_dir / "chi_loss_curve.png", dpi=dpi)
    plt.close(fig)

    train = pred_df[pred_df["split"] == "train"].copy()
    fig, ax = plt.subplots(figsize=(6, 5))
    _plot_parity_panel(ax, sub=train, split="train", show_legend=True)
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    fig.savefig(fig_dir / "chi_parity_train.png", dpi=dpi)
    plt.close(fig)

    test = pred_df[pred_df["split"] == "test"].copy()
    fig, ax = plt.subplots(figsize=(6, 5))
    _plot_parity_panel(ax, sub=test, split="test", show_legend=True)
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    fig.savefig(fig_dir / "chi_parity_test.png", dpi=dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    plotted_any = False
    for split, color in [("train", "#4c78a8"), ("val", "#f58518"), ("test", "#54a24b")]:
        sub = pred_df[pred_df["split"] == split]
        plotted_any = _plot_kde_safe_1d(
            ax=ax,
            values=sub["chi_error"],
            label=split,
            color=color,
            linewidth=2.0,
            fill=False,
            alpha=0.3,
        ) or plotted_any
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("chi prediction error")
    ax.set_title("Residual distribution by split")
    if plotted_any:
        ax.legend(
            loc="upper right",
            frameon=True,
            fancybox=True,
            framealpha=0.92,
            facecolor="white",
            edgecolor="#666666",
        )
    else:
        ax.text(0.5, 0.5, "Insufficient residual variance for KDE", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(fig_dir / "chi_residual_distribution.png", dpi=dpi)
    plt.close(fig)


def _make_classifier_figures(
    history: Dict[str, List[float]],
    pred_df: pd.DataFrame,
    fig_dir: Path,
    dpi: int,
    font_size: int,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    apply_publication_figure_style(font_size=font_size, dpi=dpi, remove_titles=True)

    has_val_curve = _has_finite_history_values(history, "val_loss")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(history["epoch"], history["train_loss"], label="Train BCE")
    if has_val_curve:
        ax.plot(history["epoch"], history["val_loss"], label="Val BCE")
        ax.set_title("Step4_2 classification loss")
    else:
        ax.set_title("Step4_2 classification loss (final fit: train+val)")
        ax.text(
            0.03,
            0.97,
            "No validation split in final fit",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#666666", "alpha": 0.92},
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE loss")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(fig_dir / "class_loss_curve.png", dpi=dpi)
    plt.close(fig)

    train = pred_df[pred_df["split"] == "train"].copy()
    if not train.empty:
        fig, ax = plt.subplots(figsize=(6, 5))
        _plot_classifier_parity_panel(ax=ax, sub=train, split="train", show_legend=True)
        fig.tight_layout()
        fig.savefig(fig_dir / "class_parity_train.png", dpi=dpi)
        plt.close(fig)
    _save_binary_confusion_matrix_figure(
        sub=train,
        out_png=fig_dir / "class_confusion_matrix_train.png",
        title="Step4_2 confusion matrix (train)",
        dpi=dpi,
    )

    test = pred_df[pred_df["split"] == "test"].copy()
    if not test.empty:
        fig, ax = plt.subplots(figsize=(6, 5))
        _plot_classifier_parity_panel(ax=ax, sub=test, split="test", show_legend=True)
        fig.tight_layout()
        fig.savefig(fig_dir / "class_parity_test.png", dpi=dpi)
        plt.close(fig)
    _save_binary_confusion_matrix_figure(
        sub=test,
        out_png=fig_dir / "class_confusion_matrix_test.png",
        title="Step4_2 confusion matrix (test)",
        dpi=dpi,
    )

    for split_name, split_df in [("train", train), ("test", test)]:
        if split_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(6, 5))
        _plot_class_prob_density_safe(ax=ax, split_df=split_df)
        ax.set_xlabel("Predicted water-miscible probability")
        ax.set_title(f"Class probability distribution ({split_name})")
        legend = ax.get_legend()
        if legend is not None:
            handles = _get_legend_handles(legend)
            labels = [t.get_text() for t in legend.get_texts()]
            legend.remove()
            ax.legend(
                handles=handles,
                labels=_class_display_labels(labels),
                title=CLASS_DISPLAY_LEGEND_TITLE,
                loc="upper center",
                ncol=min(2, len(labels)),
                frameon=True,
                fancybox=True,
                framealpha=0.92,
                facecolor="white",
                edgecolor="#666666",
            )
        fig.tight_layout()
        fig.savefig(fig_dir / f"class_prob_distribution_{split_name}.png", dpi=dpi)
        plt.close(fig)

        y_true = split_df["water_miscible"].to_numpy(dtype=int)
        y_prob = split_df["class_prob"].to_numpy(dtype=float)
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            precision, recall, _ = precision_recall_curve(y_true, y_prob)

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, color="#1f77b4", linewidth=2)
            ax.plot([0, 1], [0, 1], "k--", linewidth=1)
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.set_title(f"Classifier ROC ({split_name})")
            fig.tight_layout()
            fig.savefig(fig_dir / f"classifier_roc_{split_name}.png", dpi=dpi)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(recall, precision, color="#d62728", linewidth=2)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(f"Classifier PR ({split_name})")
            fig.tight_layout()
            fig.savefig(fig_dir / f"classifier_pr_{split_name}.png", dpi=dpi)
            plt.close(fig)


def _merge_predictions(
    reg_predictions: Dict[str, Dict[str, np.ndarray]],
    cls_predictions: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, Dict[str, np.ndarray]]:
    merged: Dict[str, Dict[str, np.ndarray]] = {}
    for split in ["train", "val", "test"]:
        reg = reg_predictions[split]
        cls = cls_predictions[split]
        if len(reg["chi_pred"]) != len(cls["prob"]):
            raise ValueError(
                f"Prediction length mismatch on split={split}: regression={len(reg['chi_pred'])}, classification={len(cls['prob'])}"
            )
        merged[split] = {
            "chi_true": reg["chi_true"],
            "chi_pred": reg["chi_pred"],
            "label": cls["label"],
            "logit": cls["logit"],
            "prob": cls["prob"],
        }
    return merged


def _save_combined_polymer_coefficients(
    reg_model: nn.Module,
    cls_model: nn.Module,
    embedding_cache_df: pd.DataFrame,
    out_csv: Path,
    device: str,
    tokenizer=None,
    timestep: int = 1,
) -> None:
    raise RuntimeError(
        "Combined coefficient export is not available after switching Step4_1 regression to direct chi prediction."
    )


def main(args):
    config = load_step4_config(args.config)
    train_cfg = build_train_config(args, config)
    chi_cfg = _default_chi_config(config)
    reg_split_mode = str(train_cfg.split_mode).strip().lower()
    cls_split_mode = str(chi_cfg.get("classification_split_mode", "random")).strip().lower()
    if cls_split_mode not in {"polymer", "random"}:
        raise ValueError("chi_training.shared.classification_split_mode must be one of {'polymer','random'}")
    backbone_split_mode = reg_split_mode
    if args.finetune_last_layers is not None:
        train_cfg.finetune_last_layers = int(args.finetune_last_layers)
    run_step41 = args.stage in {"both", "step4_1"}
    run_step42 = args.stage in {"both", "step4_2"}

    backbone_cfg = get_model_config(args.model_size, config, model_type="sequence")
    backbone_num_layers = int(backbone_cfg["num_layers"])
    if train_cfg.finetune_last_layers < 0 or train_cfg.finetune_last_layers > backbone_num_layers:
        raise ValueError(
            f"chi_training.step4_1_regression.finetune_last_layers must be in [0, {backbone_num_layers}] "
            f"for model_size={args.model_size}, got {train_cfg.finetune_last_layers}"
        )

    # Stage-separated Step 4 layout:
    #   results_<model_size>/step4_1_regression/<split_mode>/
    #   results_<model_size>/step4_2_classification/
    # This avoids write contention when Step4_1 and Step4_2 run as separate jobs.
    results_dir = Path(get_results_dir(args.model_size, config["paths"]["results_dir"], split_mode=None))
    reg_dir = results_dir / "step4_1_regression" / reg_split_mode
    cls_dir = results_dir / "step4_2_classification"
    reg_shared_dir = reg_dir / "shared"
    cls_shared_dir = cls_dir / "shared"
    if args.stage == "step4_1":
        stage_dir = reg_dir
        stage_step_name = "step4_1_regression"
        pipeline_metrics_run_dir = reg_dir / "pipeline_metrics"
    elif args.stage == "step4_2":
        stage_dir = cls_dir
        stage_step_name = "step4_2_classification"
        pipeline_metrics_run_dir = cls_dir / "pipeline_metrics"
    else:
        stage_dir = results_dir / f"step4_split_pipeline_{reg_split_mode}"
        stage_step_name = "step4_split_pipeline"
        pipeline_metrics_run_dir = stage_dir / "pipeline_metrics"
    reg_metrics_dir = reg_dir / "metrics"
    reg_figures_dir = reg_dir / "figures"
    reg_tuning_dir = reg_dir / "tuning"
    reg_checkpoint_dir = reg_dir / "checkpoints"
    cls_metrics_dir = cls_dir / "metrics"
    cls_figures_dir = cls_dir / "figures"
    cls_tuning_dir = cls_dir / "tuning"
    cls_checkpoint_dir = cls_dir / "checkpoints"
    legacy_checkpoint_dir = results_dir / "checkpoints"

    dir_candidates = [stage_dir, pipeline_metrics_run_dir, legacy_checkpoint_dir]
    if run_step41:
        dir_candidates.extend(
            [
                reg_dir,
                reg_shared_dir,
                reg_metrics_dir,
                reg_figures_dir,
                reg_tuning_dir,
                reg_checkpoint_dir,
            ]
        )
    if run_step42:
        dir_candidates.extend(
            [
                cls_dir,
                cls_shared_dir,
                cls_metrics_dir,
                cls_figures_dir,
                cls_tuning_dir,
                cls_checkpoint_dir,
            ]
        )
    for d in dir_candidates:
        d.mkdir(parents=True, exist_ok=True)

    split_ratios = _resolve_split_ratios(train_cfg)
    # Compute classification-specific split ratios using its own tuning_cv_folds.
    _cls_cv_folds_for_split = int(chi_cfg.get("step4_2", {}).get("tuning_cv_folds", train_cfg.tuning_cv_folds))
    _cls_test = float(train_cfg.holdout_test_ratio)
    _cls_dev = 1.0 - _cls_test
    _cls_val = _cls_dev / float(max(2, _cls_cv_folds_for_split))
    cls_split_ratios = {
        "train_ratio": _cls_dev - _cls_val,
        "val_ratio": _cls_val,
        "test_ratio": _cls_test,
    }
    reg_csv = args.regression_dataset_path or args.dataset_path or chi_cfg["step4_1_dataset_path"]
    cls_csv = args.classification_dataset_path or chi_cfg["step4_2_dataset_path"]

    seed_info = seed_everything(train_cfg.seed)
    save_config(config, stage_dir / "config_used.yaml")
    save_run_metadata(stage_dir, args.config, seed_info)
    write_initial_log(
        step_dir=stage_dir,
        step_name=stage_step_name,
        context={
            "config_path": args.config,
            "model_size": args.model_size,
            "stage": args.stage,
            "results_dir": str(results_dir),
            "stage_output_dir": str(stage_dir),
            "split_mode": reg_split_mode,
            "regression_split_mode": reg_split_mode,
            "classification_split_mode": cls_split_mode,
            "backbone_artifact_split_mode": backbone_split_mode,
            "holdout_test_ratio": float(train_cfg.holdout_test_ratio),
            "resolved_train_ratio": float(split_ratios["train_ratio"]),
            "resolved_val_ratio": float(split_ratios["val_ratio"]),
            "resolved_test_ratio": float(split_ratios["test_ratio"]),
            "final_fit_uses_train_plus_val": True,
            "dataset_path": str(reg_csv),
            "step4_1_dataset_path": str(reg_csv),
            "step4_2_dataset_path": _serialize_path_spec(cls_csv),
            "tune": train_cfg.tune,
            "n_trials": train_cfg.n_trials,
            "tuning_objective": train_cfg.tuning_objective,
            "tuning_cv_folds": train_cfg.tuning_cv_folds,
            "budget_search_epochs": train_cfg.budget_search_epochs,
            "budget_search_patience": train_cfg.budget_search_patience,
            "epoch_selection_metric": train_cfg.epoch_selection_metric,
            "loss_weighting": train_cfg.loss_weighting,
            "loss_weight_clip_ratio": train_cfg.loss_weight_clip_ratio,
            "nrmse_std_floor": train_cfg.nrmse_std_floor,
            "nrmse_clip": train_cfg.nrmse_clip,
            "gradient_clip_norm": train_cfg.gradient_clip_norm,
            "use_scheduler": train_cfg.use_scheduler,
            "scheduler_min_lr": train_cfg.scheduler_min_lr,
            "backbone_num_layers": backbone_num_layers,
            "finetune_last_layers": train_cfg.finetune_last_layers,
            "step4_1_dir": str(reg_dir),
            "step4_2_dir": str(cls_dir),
            "random_seed": train_cfg.seed,
        },
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("Step 4 split pipeline:")
    print("  Step4_1: chi regression")
    print("  Step4_2: water-miscible classification")
    print(f"Stage: {args.stage}")
    print(f"Regression split mode: {reg_split_mode}")
    print(f"Classification split mode: {cls_split_mode}")
    print(f"Backbone artifact split mode: {backbone_split_mode}")
    print(
        "Resolved split ratios: "
        f"train={split_ratios['train_ratio']:.4f}, "
        f"val={split_ratios['val_ratio']:.4f}, "
        f"test={split_ratios['test_ratio']:.4f}"
    )
    print(f"Step4_1 dataset: {reg_csv}")
    print(f"Step4_2 dataset: {cls_csv}")
    print(f"finetune_last_layers (regression): {train_cfg.finetune_last_layers}/{backbone_num_layers}")
    print(f"Device: {device}")
    print("=" * 70)

    reg_df = load_chi_dataset(reg_csv)
    reg_split_assign = make_split_assignments(
        reg_df,
        SplitConfig(
            split_mode=reg_split_mode,
            train_ratio=split_ratios["train_ratio"],
            val_ratio=split_ratios["val_ratio"],
            test_ratio=split_ratios["test_ratio"],
            seed=train_cfg.seed,
        ),
    )
    reg_split_df = add_split_column(reg_df, reg_split_assign)

    cls_df = _load_step42_classification_dataset(cls_csv)
    cls_split_assign = make_split_assignments(
        cls_df,
        SplitConfig(
            split_mode=cls_split_mode,
            train_ratio=cls_split_ratios["train_ratio"],
            val_ratio=cls_split_ratios["val_ratio"],
            test_ratio=cls_split_ratios["test_ratio"],
            seed=train_cfg.seed,
        ),
    )
    cls_split_df = add_split_column(cls_df, cls_split_assign)

    if run_step41:
        reg_split_assign.to_csv(reg_shared_dir / "split_assignments.csv", index=False)
        reg_split_df.to_csv(reg_shared_dir / "chi_dataset_with_split.csv", index=False)
        reg_split_assign.to_csv(reg_shared_dir / f"split_assignments_step4_1_{reg_split_mode}.csv", index=False)
        reg_split_df.to_csv(reg_shared_dir / f"chi_dataset_with_split_step4_1_{reg_split_mode}.csv", index=False)
        reg_split_assign.to_csv(reg_metrics_dir / "split_assignments.csv", index=False)
        reg_split_df.to_csv(reg_metrics_dir / "chi_dataset_with_split.csv", index=False)
    if run_step42:
        cls_split_assign.to_csv(cls_shared_dir / "split_assignments.csv", index=False)
        cls_split_df.to_csv(cls_shared_dir / "chi_dataset_with_split.csv", index=False)
        cls_split_assign.to_csv(cls_shared_dir / f"split_assignments_step4_2_{cls_split_mode}.csv", index=False)
        cls_split_df.to_csv(cls_shared_dir / f"chi_dataset_with_split_step4_2_{cls_split_mode}.csv", index=False)
        cls_split_assign.to_csv(cls_metrics_dir / "split_assignments.csv", index=False)
        cls_split_df.to_csv(cls_metrics_dir / "chi_dataset_with_split.csv", index=False)

    reg_embedding_table: Optional[np.ndarray] = None
    if run_step41:
        reg_polymer_df = reg_split_df[["polymer_id", "Polymer", "SMILES", "water_miscible"]].drop_duplicates("polymer_id")
        reg_emb_cache = build_or_load_embedding_cache(
            polymer_df=reg_polymer_df,
            config=config,
            cache_npz=reg_shared_dir / f"polymer_embeddings_step4_1_{reg_split_mode}.npz",
            model_size=args.model_size,
            split_mode=reg_split_mode,
            checkpoint_path=args.backbone_checkpoint,
            device=device,
            timestep=train_cfg.timestep_for_embedding,
            pooling="mean",
            batch_size=int(chi_cfg["embedding_batch_size"]),
        )
        reg_embedding_table = embedding_table_from_cache(reg_emb_cache)
    else:
        print("Skipping Step4_1 regression stage by request.")

    cls_embedding_table: Optional[np.ndarray] = None
    if run_step42:
        cls_polymer_df = cls_split_df[["polymer_id", "Polymer", "SMILES", "water_miscible"]].drop_duplicates("polymer_id")
        cls_emb_cache = build_or_load_embedding_cache(
            polymer_df=cls_polymer_df,
            config=config,
            cache_npz=cls_shared_dir / f"polymer_embeddings_step4_2_classification_{cls_split_mode}.npz",
            model_size=args.model_size,
            split_mode=backbone_split_mode,
            checkpoint_path=args.backbone_checkpoint,
            device=device,
            timestep=train_cfg.timestep_for_embedding,
            pooling="mean",
            batch_size=int(chi_cfg["embedding_batch_size"]),
        )
        cls_embedding_table = embedding_table_from_cache(cls_emb_cache)
    else:
        print("Skipping Step4_2 classification stage by request.")

    tokenizer_for_training = None

    dpi = int(config.get("plotting", {}).get("dpi", 600))
    font_size = int(config.get("plotting", {}).get("font_size", 16))

    reg_checkpoint_path = reg_checkpoint_dir / "chi_regression_best.pt"
    cls_checkpoint_path = cls_checkpoint_dir / "chi_classifier_best.pt"

    reg_best_params = None
    reg_chosen = None
    reg_finetune_last_layers = None
    reg_use_backbone_finetune = False
    reg_final_train_rows: Optional[int] = None
    reg_final_test_rows: Optional[int] = None
    reg_history: Dict[str, List[float]] = {"epoch": []}
    reg_post_optuna_cv_summary: Optional[Dict[str, object]] = None

    cls_best_params = None
    cls_chosen = None
    cls_finetune_last_layers = None
    cls_use_backbone_finetune = False
    cls_final_train_rows: Optional[int] = None
    cls_final_test_rows: Optional[int] = None
    cls_history: Dict[str, List[float]] = {"epoch": []}
    cls_post_optuna_cv_summary: Optional[Dict[str, object]] = None

    # -------------------------
    # Step 4_1: Regression
    # -------------------------
    if run_step41:
        if reg_embedding_table is None:
            raise RuntimeError("Regression embedding table was not prepared.")
        reg_cfg = copy.deepcopy(train_cfg)
        reg_cfg.tune = bool(args.tune or (bool(reg_cfg.tune) and not args.no_tune))
        if args.n_trials is not None:
            reg_cfg.n_trials = int(args.n_trials)
        if args.tuning_cv_folds is not None:
            reg_cfg.tuning_cv_folds = int(args.tuning_cv_folds)
        if args.tuning_objective is not None:
            reg_cfg.tuning_objective = _normalize_tuning_objective(args.tuning_objective)

        reg_finetune_last_layers = int(reg_cfg.finetune_last_layers)
        reg_use_backbone_finetune = bool(reg_finetune_last_layers > 0)
        if tokenizer_for_training is None and (reg_use_backbone_finetune or reg_cfg.tune):
            tokenizer_for_training, _, _ = load_backbone_from_step1(
                config=config,
                model_size=args.model_size,
                split_mode=backbone_split_mode,
                checkpoint_path=args.backbone_checkpoint,
                device="cpu",
            )
        write_initial_log(
            step_dir=reg_dir,
            step_name="step4_1_regression",
            context={
                "config_path": args.config,
                "model_size": args.model_size,
                "stage": args.stage,
                "results_dir": str(results_dir),
                "stage_output_dir": str(reg_dir),
                "split_mode": reg_split_mode,
                "regression_split_mode": reg_split_mode,
                "backbone_artifact_split_mode": backbone_split_mode,
                "holdout_test_ratio": float(reg_cfg.holdout_test_ratio),
                "resolved_train_ratio": float(split_ratios["train_ratio"]),
                "resolved_val_ratio": float(split_ratios["val_ratio"]),
                "resolved_test_ratio": float(split_ratios["test_ratio"]),
                "final_fit_uses_train_plus_val": True,
                "dataset_path": str(reg_csv),
                "step4_1_dataset_path": str(reg_csv),
                "tune": bool(reg_cfg.tune),
                "n_trials": int(reg_cfg.n_trials),
                "tuning_objective": reg_cfg.tuning_objective,
                "tuning_cv_folds": int(reg_cfg.tuning_cv_folds),
                "budget_search_epochs": int(reg_cfg.budget_search_epochs),
                "budget_search_patience": int(reg_cfg.budget_search_patience),
                "epoch_selection_metric": reg_cfg.epoch_selection_metric,
                "loss_weighting": reg_cfg.loss_weighting,
                "loss_weight_clip_ratio": float(reg_cfg.loss_weight_clip_ratio),
                "nrmse_std_floor": float(reg_cfg.nrmse_std_floor),
                "nrmse_clip": float(reg_cfg.nrmse_clip),
                "gradient_clip_norm": float(reg_cfg.gradient_clip_norm),
                "use_scheduler": bool(reg_cfg.use_scheduler),
                "scheduler_min_lr": float(reg_cfg.scheduler_min_lr),
                "scheduler_t_max": reg_cfg.scheduler_t_max,
                "backbone_num_layers": int(backbone_num_layers),
                "finetune_last_layers": int(reg_cfg.finetune_last_layers),
                "step4_1_dir": str(reg_dir),
                "random_seed": int(reg_cfg.seed),
            },
        )

        if reg_cfg.tune:
            print("Running Optuna for Step4_1 (regression)...")
            reg_best_params = tune_hyperparameters(
                split_df=reg_split_df,
                embedding_table=reg_embedding_table,
                train_cfg=reg_cfg,
                config=config,
                model_size=args.model_size,
                backbone_num_layers=backbone_num_layers,
                backbone_checkpoint=args.backbone_checkpoint,
                tokenizer=tokenizer_for_training,
                device=device,
                tuning_dir=reg_tuning_dir,
                dpi=dpi,
                font_size=font_size,
            )
            print("Step4_1 best params:")
            print(reg_best_params)

        if reg_best_params is None:
            reg_chosen = {
                "hidden_sizes": reg_cfg.hidden_sizes,
                "dropout": reg_cfg.dropout,
                "learning_rate": reg_cfg.learning_rate,
                "weight_decay": reg_cfg.weight_decay,
                "batch_size": reg_cfg.batch_size,
                "finetune_last_layers": int(reg_cfg.finetune_last_layers),
            }
        else:
            reg_num_layers = int(reg_best_params["num_layers"])
            reg_chosen = {
                "hidden_sizes": [int(reg_best_params[f"hidden_{i}"]) for i in range(reg_num_layers)],
                "dropout": float(reg_best_params["dropout"]),
                "learning_rate": float(reg_best_params["learning_rate"]),
                "weight_decay": float(reg_best_params["weight_decay"]),
                "batch_size": int(reg_best_params["batch_size"]),
                "finetune_last_layers": int(reg_best_params.get("finetune_last_layers", reg_cfg.finetune_last_layers)),
            }

        reg_finetune_last_layers = int(reg_chosen["finetune_last_layers"])
        if reg_finetune_last_layers < 0 or reg_finetune_last_layers > backbone_num_layers:
            raise ValueError(
                f"chosen regression finetune_last_layers must be in [0, {backbone_num_layers}], got {reg_finetune_last_layers}"
            )
        reg_use_backbone_finetune = bool(reg_finetune_last_layers > 0)
        print("Running Step4_1 CV budget search with selected hyperparameters...")
        reg_post_optuna_cv_summary = run_regression_cv_with_best_hyperparameters(
            split_df=reg_split_df,
            embedding_table=reg_embedding_table,
            train_cfg=reg_cfg,
            config=config,
            model_size=args.model_size,
            backbone_checkpoint=args.backbone_checkpoint,
            tokenizer=tokenizer_for_training,
            device=device,
            hidden_sizes=reg_chosen["hidden_sizes"],
            dropout=reg_chosen["dropout"],
            learning_rate=reg_chosen["learning_rate"],
            weight_decay=reg_chosen["weight_decay"],
            batch_size=reg_chosen["batch_size"],
            finetune_last_layers=reg_finetune_last_layers,
            metrics_dir=reg_metrics_dir / "cv_best_params",
            figures_dir=reg_figures_dir / "cv_best_params",
            dpi=dpi,
            font_size=font_size,
        )
        reg_final_fit_df = _build_final_fit_split_df(reg_split_df)
        reg_final_train_rows = int((reg_final_fit_df["split"] == "train").sum())
        reg_final_test_rows = int((reg_final_fit_df["split"] == "test").sum())
        reg_final_num_epochs = int(reg_cfg.num_epochs)
        if isinstance(reg_post_optuna_cv_summary, dict):
            reg_final_num_epochs = int(reg_post_optuna_cv_summary.get("final_training_derived_epochs", reg_final_num_epochs))
        reg_final_num_epochs = int(np.clip(reg_final_num_epochs, 1, int(reg_cfg.num_epochs)))
        reg_final_cfg = copy.deepcopy(reg_cfg)
        reg_final_cfg.scheduler_t_max = int(reg_cfg.num_epochs)

        with open(reg_metrics_dir / "chosen_hyperparameters.json", "w") as f:
            json.dump(reg_chosen, f, indent=2)
        with open(reg_metrics_dir / "hyperparameter_selection_summary.json", "w") as f:
            json.dump(
                {
                    "used_optuna": bool(reg_cfg.tune),
                    "optuna_objective": _describe_tuning_objective(reg_cfg),
                    "tuning_objective": reg_cfg.tuning_objective,
                    "tuning_cv_folds": int(reg_cfg.tuning_cv_folds),
                    "epoch_selection_metric": reg_cfg.epoch_selection_metric,
                    "optuna_best_params": reg_best_params,
                    "backbone_num_layers": int(backbone_num_layers),
                    "finetune_last_layers": int(reg_finetune_last_layers),
                    "backbone_finetune_enabled": bool(reg_use_backbone_finetune),
                    "final_training_hyperparameters": reg_chosen,
                    "loss_weighting": reg_cfg.loss_weighting,
                    "loss_weight_clip_ratio": float(reg_cfg.loss_weight_clip_ratio),
                    "nrmse_std_floor": float(reg_cfg.nrmse_std_floor),
                    "nrmse_clip": float(reg_cfg.nrmse_clip),
                    "budget_search_epochs": int(reg_cfg.budget_search_epochs),
                    "budget_search_patience": int(reg_cfg.budget_search_patience),
                    "final_training_num_epochs": int(reg_final_num_epochs),
                    "final_training_patience": int(reg_cfg.patience),
                    "final_training_scheduler_t_max": int(reg_final_cfg.scheduler_t_max),
                    "final_training_derived_steps": (
                        int(reg_post_optuna_cv_summary.get("final_training_derived_steps"))
                        if isinstance(reg_post_optuna_cv_summary, dict)
                        and reg_post_optuna_cv_summary.get("final_training_derived_steps") is not None
                        else None
                    ),
                    "final_training_derived_epochs": (
                        int(reg_post_optuna_cv_summary.get("final_training_derived_epochs"))
                        if isinstance(reg_post_optuna_cv_summary, dict)
                        and reg_post_optuna_cv_summary.get("final_training_derived_epochs") is not None
                        else None
                    ),
                    "final_fit_uses_train_plus_val": True,
                    "final_fit_train_rows": reg_final_train_rows,
                    "final_fit_test_rows": reg_final_test_rows,
                    "post_optuna_cv_retrain": reg_post_optuna_cv_summary,
                },
                f,
                indent=2,
            )

        reg_model, reg_history, reg_predictions = train_one_model(
            split_df=reg_final_fit_df,
            embedding_table=reg_embedding_table,
            train_cfg=reg_final_cfg,
            device=device,
            hidden_sizes=reg_chosen["hidden_sizes"],
            dropout=reg_chosen["dropout"],
            learning_rate=reg_chosen["learning_rate"],
            weight_decay=reg_chosen["weight_decay"],
            batch_size=reg_chosen["batch_size"],
            num_epochs=reg_final_num_epochs,
            patience=reg_cfg.patience,
            config=config,
            model_size=args.model_size,
            backbone_checkpoint=args.backbone_checkpoint,
            tokenizer=tokenizer_for_training,
            finetune_last_layers=reg_finetune_last_layers,
            timestep_for_embedding=reg_final_cfg.timestep_for_embedding,
        )

        reg_checkpoint = {
            "model_state_dict": reg_model.state_dict(),
            "embedding_dim": int(reg_embedding_table.shape[1]),
            "hidden_sizes": reg_chosen["hidden_sizes"],
            "dropout": reg_chosen["dropout"],
            "learning_rate": reg_chosen["learning_rate"],
            "weight_decay": reg_chosen["weight_decay"],
            "batch_size": reg_chosen["batch_size"],
            "used_optuna": bool(reg_cfg.tune),
            "optuna_best_params": reg_best_params,
            "split_mode": reg_cfg.split_mode,
            "timestep_for_embedding": reg_cfg.timestep_for_embedding,
            "loss_weighting": reg_cfg.loss_weighting,
            "loss_weight_clip_ratio": float(reg_cfg.loss_weight_clip_ratio),
            "tuning_objective": reg_cfg.tuning_objective,
            "epoch_selection_metric": reg_cfg.epoch_selection_metric,
            "scheduler_t_max": reg_final_cfg.scheduler_t_max,
            "nrmse_std_floor": float(reg_cfg.nrmse_std_floor),
            "nrmse_clip": float(reg_cfg.nrmse_clip),
            "num_epochs_trained": int(reg_final_num_epochs),
            "backbone_num_layers": int(backbone_num_layers),
            "finetune_last_layers": int(reg_finetune_last_layers),
            "backbone_finetune_enabled": bool(reg_use_backbone_finetune),
            "regression_mode": "direct_chi",
            "dataset_path": str(reg_csv),
            "config": config,
        }
        reg_legacy_path = legacy_checkpoint_dir / "chi_regression_best.pt"
        reg_legacy_joint_path = legacy_checkpoint_dir / "chi_physics_best.pt"
        torch.save(reg_checkpoint, reg_checkpoint_path)
        torch.save(reg_checkpoint, reg_checkpoint_dir / "chi_physics_best.pt")
        # Legacy compatibility paths (deprecated)
        torch.save(reg_checkpoint, reg_legacy_path)
        torch.save(reg_checkpoint, reg_legacy_joint_path)

        _save_history(reg_history, reg_metrics_dir / "chi_training_history.csv")
        reg_pred_df = _save_regression_prediction_csvs(split_df=reg_final_fit_df, predictions=reg_predictions, out_dir=reg_metrics_dir)
        _collect_regression_metrics(reg_pred_df, out_dir=reg_metrics_dir)
        _make_regression_figures(
            history=reg_history,
            pred_df=reg_pred_df,
            fig_dir=reg_figures_dir,
            dpi=dpi,
            font_size=font_size,
        )

    # -------------------------
    # Step 4_2: Classification
    # -------------------------
    if run_step42:
        if cls_embedding_table is None:
            raise RuntimeError("Classification embedding table was not prepared.")
        cls_cfg = copy.deepcopy(train_cfg)
        cls_cfg.split_mode = cls_split_mode
        cls_section = chi_cfg.get("step4_2", {}) if isinstance(chi_cfg.get("step4_2", {}), dict) else {}
        cls_cfg.finetune_last_layers = int(cls_section.get("finetune_last_layers", cls_cfg.finetune_last_layers))
        cls_cfg.batch_size = int(cls_section.get("batch_size", cls_cfg.batch_size))
        cls_cfg.num_epochs = int(cls_section.get("num_epochs", cls_cfg.num_epochs))
        cls_cfg.patience = int(cls_section.get("patience", cls_cfg.patience))
        cls_cfg.learning_rate = float(cls_section.get("learning_rate", cls_cfg.learning_rate))
        cls_cfg.weight_decay = float(cls_section.get("weight_decay", cls_cfg.weight_decay))
        cls_cfg.gradient_clip_norm = float(cls_section.get("gradient_clip_norm", cls_cfg.gradient_clip_norm))
        if cls_cfg.gradient_clip_norm < 0:
            raise ValueError("chi_training.step4_2_classification.gradient_clip_norm must be >= 0")
        cls_cfg.use_scheduler = bool(cls_section.get("use_scheduler", cls_cfg.use_scheduler))
        cls_cfg.scheduler_min_lr = float(cls_section.get("scheduler_min_lr", cls_cfg.scheduler_min_lr))
        if cls_cfg.scheduler_min_lr < 0:
            raise ValueError("chi_training.step4_2_classification.scheduler_min_lr must be >= 0")
        cls_cfg.hidden_sizes = [int(v) for v in cls_section.get("hidden_sizes", cls_cfg.hidden_sizes)]
        cls_cfg.dropout = float(cls_section.get("dropout", cls_cfg.dropout))
        cls_cfg.tune = bool(args.tune or (bool(cls_section.get("tune", cls_cfg.tune)) and not args.no_tune))
        if args.n_trials is None:
            cls_cfg.n_trials = int(cls_section.get("n_trials", cls_cfg.n_trials))
        else:
            cls_cfg.n_trials = int(args.n_trials)
        cls_cfg.tuning_epochs = int(cls_section.get("tuning_epochs", cls_cfg.tuning_epochs))
        cls_cfg.tuning_patience = int(cls_section.get("tuning_patience", cls_cfg.tuning_patience))
        if args.tuning_cv_folds is None:
            cls_cfg.tuning_cv_folds = int(cls_section.get("tuning_cv_folds", cls_cfg.tuning_cv_folds))
        else:
            cls_cfg.tuning_cv_folds = int(args.tuning_cv_folds)
        cls_cfg.optuna_search_space = dict(cls_section.get("optuna_search_space", cls_cfg.optuna_search_space))
        if cls_cfg.finetune_last_layers < 0 or cls_cfg.finetune_last_layers > backbone_num_layers:
            raise ValueError(
                f"chi_training.step4_2_classification.finetune_last_layers must be in [0, {backbone_num_layers}] "
                f"for model_size={args.model_size}, got {cls_cfg.finetune_last_layers}"
            )

        if tokenizer_for_training is None and (cls_cfg.tune or cls_cfg.finetune_last_layers > 0):
            tokenizer_for_training, _, _ = load_backbone_from_step1(
                config=config,
                model_size=args.model_size,
                split_mode=backbone_split_mode,
                checkpoint_path=args.backbone_checkpoint,
                device="cpu",
            )

        if cls_cfg.tune:
            print("Running Optuna for Step4_2 (classification)...")
            cls_best_params = tune_classifier_hyperparameters(
                split_df=cls_split_df,
                embedding_table=cls_embedding_table,
                train_cfg=cls_cfg,
                config=config,
                model_size=args.model_size,
                backbone_num_layers=backbone_num_layers,
                backbone_checkpoint=args.backbone_checkpoint,
                tokenizer=tokenizer_for_training,
                device=device,
                tuning_dir=cls_tuning_dir,
                dpi=dpi,
                font_size=font_size,
                backbone_split_mode=backbone_split_mode,
            )
            print("Step4_2 best params:")
            print(cls_best_params)

        if cls_best_params is None:
            cls_chosen = {
                "hidden_sizes": cls_cfg.hidden_sizes,
                "dropout": cls_cfg.dropout,
                "learning_rate": cls_cfg.learning_rate,
                "weight_decay": cls_cfg.weight_decay,
                "batch_size": cls_cfg.batch_size,
                "finetune_last_layers": int(cls_cfg.finetune_last_layers),
            }
        else:
            cls_num_layers = int(cls_best_params["num_layers"])
            cls_chosen = {
                "hidden_sizes": [int(cls_best_params[f"hidden_{i}"]) for i in range(cls_num_layers)],
                "dropout": float(cls_best_params["dropout"]),
                "learning_rate": float(cls_best_params["learning_rate"]),
                "weight_decay": float(cls_best_params["weight_decay"]),
                "batch_size": int(cls_best_params["batch_size"]),
                "finetune_last_layers": int(cls_best_params.get("finetune_last_layers", cls_cfg.finetune_last_layers)),
            }

        cls_finetune_last_layers = int(cls_chosen["finetune_last_layers"])
        if cls_finetune_last_layers < 0 or cls_finetune_last_layers > backbone_num_layers:
            raise ValueError(
                f"chosen classification finetune_last_layers must be in [0, {backbone_num_layers}], got {cls_finetune_last_layers}"
            )
        cls_use_backbone_finetune = bool(cls_finetune_last_layers > 0)
        if cls_cfg.tune:
            print("Running Step4_2 CV retraining with selected hyperparameters...")
            cls_post_optuna_cv_summary = run_classifier_cv_with_best_hyperparameters(
                split_df=cls_split_df,
                embedding_table=cls_embedding_table,
                train_cfg=cls_cfg,
                config=config,
                model_size=args.model_size,
                backbone_checkpoint=args.backbone_checkpoint,
                tokenizer=tokenizer_for_training,
                device=device,
                hidden_sizes=cls_chosen["hidden_sizes"],
                dropout=cls_chosen["dropout"],
                learning_rate=cls_chosen["learning_rate"],
                weight_decay=cls_chosen["weight_decay"],
                batch_size=cls_chosen["batch_size"],
                finetune_last_layers=cls_finetune_last_layers,
                metrics_dir=cls_metrics_dir / "cv_best_params",
                figures_dir=cls_figures_dir / "cv_best_params",
                dpi=dpi,
                font_size=font_size,
                backbone_split_mode=backbone_split_mode,
            )
        cls_final_fit_df = _build_final_fit_split_df(cls_split_df)
        cls_final_train_rows = int((cls_final_fit_df["split"] == "train").sum())
        cls_final_test_rows = int((cls_final_fit_df["split"] == "test").sum())

        with open(cls_metrics_dir / "chosen_hyperparameters.json", "w") as f:
            json.dump(cls_chosen, f, indent=2)
        with open(cls_metrics_dir / "hyperparameter_selection_summary.json", "w") as f:
            json.dump(
                {
                    "used_optuna": bool(cls_cfg.tune),
                    "optuna_objective": "maximize_val_balanced_accuracy",
                    "tuning_cv_folds": int(cls_cfg.tuning_cv_folds),
                    "optuna_best_params": cls_best_params,
                    "backbone_num_layers": int(backbone_num_layers),
                    "finetune_last_layers": int(cls_finetune_last_layers),
                    "backbone_finetune_enabled": bool(cls_use_backbone_finetune),
                    "final_training_hyperparameters": cls_chosen,
                    "final_training_num_epochs": int(cls_cfg.num_epochs),
                    "final_training_patience": int(cls_cfg.patience),
                    "final_fit_uses_train_plus_val": True,
                    "final_fit_train_rows": cls_final_train_rows,
                    "final_fit_test_rows": cls_final_test_rows,
                    "post_optuna_cv_retrain": cls_post_optuna_cv_summary,
                },
                f,
                indent=2,
            )

        cls_model, cls_history, cls_predictions = train_one_classifier_model(
            split_df=cls_final_fit_df,
            embedding_table=cls_embedding_table,
            train_cfg=cls_cfg,
            device=device,
            hidden_sizes=cls_chosen["hidden_sizes"],
            dropout=cls_chosen["dropout"],
            learning_rate=cls_chosen["learning_rate"],
            weight_decay=cls_chosen["weight_decay"],
            batch_size=cls_chosen["batch_size"],
            num_epochs=cls_cfg.num_epochs,
            patience=cls_cfg.patience,
            config=config,
            model_size=args.model_size,
            backbone_checkpoint=args.backbone_checkpoint,
            tokenizer=tokenizer_for_training,
            finetune_last_layers=cls_finetune_last_layers,
            timestep_for_embedding=cls_cfg.timestep_for_embedding,
            backbone_split_mode=backbone_split_mode,
        )

        cls_checkpoint = {
            "model_state_dict": cls_model.state_dict(),
            "embedding_dim": int(cls_embedding_table.shape[1]),
            "hidden_sizes": cls_chosen["hidden_sizes"],
            "dropout": cls_chosen["dropout"],
            "learning_rate": cls_chosen["learning_rate"],
            "weight_decay": cls_chosen["weight_decay"],
            "batch_size": cls_chosen["batch_size"],
            "used_optuna": bool(cls_cfg.tune),
            "optuna_best_params": cls_best_params,
            "split_mode": cls_cfg.split_mode,
            "timestep_for_embedding": cls_cfg.timestep_for_embedding,
            "backbone_num_layers": int(backbone_num_layers),
            "finetune_last_layers": int(cls_finetune_last_layers),
            "backbone_finetune_enabled": bool(cls_use_backbone_finetune),
            "dataset_path": _serialize_path_spec(cls_csv),
            "config": config,
        }
        cls_legacy_path = legacy_checkpoint_dir / "chi_classifier_best.pt"
        torch.save(cls_checkpoint, cls_checkpoint_path)
        # Legacy compatibility path (deprecated)
        torch.save(cls_checkpoint, cls_legacy_path)

        _save_history(cls_history, cls_metrics_dir / "class_training_history.csv")
        cls_pred_df = _save_classifier_prediction_csvs(split_df=cls_final_fit_df, predictions=cls_predictions, out_dir=cls_metrics_dir)
        _collect_classifier_metrics(cls_pred_df, out_dir=cls_metrics_dir)
        _make_classifier_figures(
            history=cls_history,
            pred_df=cls_pred_df,
            fig_dir=cls_figures_dir,
            dpi=dpi,
            font_size=font_size,
        )

    # Pipeline-level metadata
    chosen_payload: Dict[str, object] = {}
    if reg_chosen is not None:
        chosen_payload["step4_1_regression"] = reg_chosen
    if cls_chosen is not None:
        chosen_payload["step4_2_classification"] = cls_chosen
    with open(pipeline_metrics_run_dir / "chosen_hyperparameters.json", "w") as f:
        json.dump(chosen_payload, f, indent=2)

    hp_summary: Dict[str, object] = {
        "stage": args.stage,
        "used_optuna": bool(reg_cfg.tune) if reg_chosen is not None else bool(train_cfg.tune),
        "tuning_cv_folds": int(reg_cfg.tuning_cv_folds) if reg_chosen is not None else int(train_cfg.tuning_cv_folds),
        "holdout_test_ratio": float(train_cfg.holdout_test_ratio),
        "resolved_train_ratio": float(split_ratios["train_ratio"]),
        "resolved_val_ratio": float(split_ratios["val_ratio"]),
        "resolved_test_ratio": float(split_ratios["test_ratio"]),
        "final_fit_uses_train_plus_val": True,
    }
    if reg_chosen is not None:
        hp_summary["step4_1_regression"] = {
            "objective": _describe_tuning_objective(reg_cfg),
            "optuna_best_params": reg_best_params,
            "final_training_hyperparameters": reg_chosen,
            "loss_weighting": reg_cfg.loss_weighting,
            "loss_weight_clip_ratio": float(reg_cfg.loss_weight_clip_ratio),
            "budget_search_epochs": int(reg_cfg.budget_search_epochs),
            "budget_search_patience": int(reg_cfg.budget_search_patience),
            "epoch_selection_metric": reg_cfg.epoch_selection_metric,
            "nrmse_std_floor": float(reg_cfg.nrmse_std_floor),
            "nrmse_clip": float(reg_cfg.nrmse_clip),
            "post_optuna_cv_retrain": reg_post_optuna_cv_summary,
        }
    if cls_chosen is not None:
        hp_summary["step4_2_classification"] = {
            "objective": "maximize_val_balanced_accuracy",
            "optuna_best_params": cls_best_params,
            "final_training_hyperparameters": cls_chosen,
            "post_optuna_cv_retrain": cls_post_optuna_cv_summary,
        }
    with open(pipeline_metrics_run_dir / "hyperparameter_selection_summary.json", "w") as f:
        json.dump(hp_summary, f, indent=2)

    summary = {
        "step": stage_step_name,
        "stage": args.stage,
        "model_size": args.model_size or "small",
        "split_mode": reg_split_mode,
        "regression_split_mode": reg_split_mode,
        "classification_split_mode": cls_split_mode,
        "holdout_test_ratio": float(train_cfg.holdout_test_ratio),
        "resolved_train_ratio": float(split_ratios["train_ratio"]),
        "resolved_val_ratio": float(split_ratios["val_ratio"]),
        "resolved_test_ratio": float(split_ratios["test_ratio"]),
        "final_fit_uses_train_plus_val": True,
        "step4_1_dataset_path": str(reg_csv),
        "step4_2_dataset_path": _serialize_path_spec(cls_csv),
        "n_data_rows_step4_1": int(len(reg_split_df)),
        "n_polymers_step4_1": int(reg_split_df["polymer_id"].nunique()),
        "n_data_rows_step4_2": int(len(cls_split_df)),
        "n_polymers_step4_2": int(cls_split_df["polymer_id"].nunique()),
        "step4_1_metrics_dir": str(reg_metrics_dir),
        "step4_2_metrics_dir": str(cls_metrics_dir),
        "step4_1_checkpoint": str(reg_checkpoint_path),
        "step4_2_checkpoint": str(cls_checkpoint_path),
    }
    if run_step41:
        reg_overall_df = pd.read_csv(reg_metrics_dir / "chi_metrics_overall.csv")
        test_rows = reg_overall_df[reg_overall_df["split"] == "test"]
        test_row = test_rows.iloc[0] if len(test_rows) > 0 else reg_overall_df.iloc[-1]
        reg_cv_train = {}
        reg_cv_test = {}
        if isinstance(reg_post_optuna_cv_summary, dict):
            reg_cv_by_split = reg_post_optuna_cv_summary.get("summary_by_split", {})
            if isinstance(reg_cv_by_split, dict):
                reg_cv_train = reg_cv_by_split.get("train", {}) if isinstance(reg_cv_by_split.get("train", {}), dict) else {}
                reg_cv_test = reg_cv_by_split.get("test", {}) if isinstance(reg_cv_by_split.get("test", {}), dict) else {}
        summary.update(
            {
                "step4_1_backbone_num_layers": int(backbone_num_layers),
                "step4_1_finetune_last_layers": int(reg_finetune_last_layers),
                "step4_1_backbone_finetune_enabled": bool(reg_use_backbone_finetune),
                "step4_1_final_fit_train_rows": int(reg_final_train_rows) if reg_final_train_rows is not None else np.nan,
                "step4_1_final_fit_test_rows": int(reg_final_test_rows) if reg_final_test_rows is not None else np.nan,
                "step4_1_n_epochs": int(len(reg_history["epoch"])),
                "step4_1_test_mae": float(test_row["mae"]) if "mae" in test_row else np.nan,
                "step4_1_test_rmse": float(test_row["rmse"]) if "rmse" in test_row else np.nan,
                "step4_1_test_r2": float(test_row["r2"]) if "r2" in test_row else np.nan,
                "step4_1_post_optuna_cv_folds": (
                    int(reg_post_optuna_cv_summary["resolved_folds"])
                    if isinstance(reg_post_optuna_cv_summary, dict) and "resolved_folds" in reg_post_optuna_cv_summary
                    else np.nan
                ),
                "step4_1_post_optuna_cv_train_r2_mean": float(reg_cv_train["r2_mean"]) if "r2_mean" in reg_cv_train else np.nan,
                "step4_1_post_optuna_cv_test_r2_mean": float(reg_cv_test["r2_mean"]) if "r2_mean" in reg_cv_test else np.nan,
            }
        )
    if run_step42:
        cls_overall_df = pd.read_csv(cls_metrics_dir / "class_metrics_overall.csv")
        cls_test_rows = cls_overall_df[cls_overall_df["split"] == "test"]
        cls_test_row = cls_test_rows.iloc[0] if len(cls_test_rows) > 0 else cls_overall_df.iloc[-1]
        cls_cv_train = {}
        cls_cv_test = {}
        if isinstance(cls_post_optuna_cv_summary, dict):
            cls_cv_by_split = cls_post_optuna_cv_summary.get("summary_by_split", {})
            if isinstance(cls_cv_by_split, dict):
                cls_cv_train = cls_cv_by_split.get("train", {}) if isinstance(cls_cv_by_split.get("train", {}), dict) else {}
                cls_cv_test = cls_cv_by_split.get("test", {}) if isinstance(cls_cv_by_split.get("test", {}), dict) else {}
        summary.update(
            {
                "step4_2_backbone_num_layers": int(backbone_num_layers),
                "step4_2_finetune_last_layers": int(cls_finetune_last_layers),
                "step4_2_backbone_finetune_enabled": bool(cls_use_backbone_finetune),
                "step4_2_final_fit_train_rows": int(cls_final_train_rows) if cls_final_train_rows is not None else np.nan,
                "step4_2_final_fit_test_rows": int(cls_final_test_rows) if cls_final_test_rows is not None else np.nan,
                "step4_2_n_epochs": int(len(cls_history["epoch"])),
                "step4_2_test_balanced_accuracy": float(cls_test_row["balanced_accuracy"]) if "balanced_accuracy" in cls_test_row else np.nan,
                "step4_2_test_auroc": float(cls_test_row["auroc"]) if "auroc" in cls_test_row else np.nan,
                "step4_2_post_optuna_cv_folds": (
                    int(cls_post_optuna_cv_summary["resolved_folds"])
                    if isinstance(cls_post_optuna_cv_summary, dict) and "resolved_folds" in cls_post_optuna_cv_summary
                    else np.nan
                ),
                "step4_2_post_optuna_cv_train_balanced_accuracy_mean": (
                    float(cls_cv_train["balanced_accuracy_mean"]) if "balanced_accuracy_mean" in cls_cv_train else np.nan
                ),
                "step4_2_post_optuna_cv_test_balanced_accuracy_mean": (
                    float(cls_cv_test["balanced_accuracy_mean"]) if "balanced_accuracy_mean" in cls_cv_test else np.nan
                ),
            }
        )
    save_step_summary(summary, pipeline_metrics_run_dir)
    if run_step41:
        save_artifact_manifest(step_dir=reg_dir, metrics_dir=reg_metrics_dir, figures_dir=reg_figures_dir)
    if run_step42:
        save_artifact_manifest(step_dir=cls_dir, metrics_dir=cls_metrics_dir, figures_dir=cls_figures_dir)

    print("Training complete.")
    print(f"Step4 outputs dir for this run: {stage_dir}")
    print(f"Pipeline metrics for this run: {pipeline_metrics_run_dir}")
    if run_step41:
        print(f"Step4_1 outputs ({reg_split_mode}): {reg_dir}")
        print(f"Step4_1 checkpoint: {reg_checkpoint_path}")
    if run_step42:
        print(f"Step4_2 outputs: {cls_dir}")
        print(f"Step4_2 checkpoint: {cls_checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4: train Step4_1 regression + Step4_2 classification models")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config4.yaml",
        help="Step 4 config path. Step 4-only overlays are merged onto configs/config.yaml.",
    )
    parser.add_argument("--model_size", type=str, default="small", choices=["small", "medium", "large", "xl"], help="Step1 model size tag")
    parser.add_argument(
        "--split_mode",
        type=str,
        default=None,
        choices=["polymer", "random"],
        help=(
            "Optional regression split-mode override "
            "(otherwise uses config chi_training.shared.split_mode). "
            "Step4_2 classification uses chi_training.shared.classification_split_mode."
        ),
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="both",
        choices=["both", "step4_1", "step4_2"],
        help="Run both Step4_1/Step4_2 or a single stage",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Legacy alias for --regression_dataset_path (Step4_1 dataset).",
    )
    parser.add_argument(
        "--regression_dataset_path",
        type=str,
        default=None,
        help="Path to Step4_1 regression dataset CSV (expects chi(T,phi) columns).",
    )
    parser.add_argument(
        "--classification_dataset_path",
        type=str,
        default=None,
        help=(
            "Path/list to Step4_2 classification CSV data "
            "(supports Polymer/SMILES/water_miscible; accepts single CSV, CSV directory, or comma-separated CSVs)."
        ),
    )
    parser.add_argument("--backbone_checkpoint", type=str, default=None, help="Optional backbone checkpoint override")
    parser.add_argument("--tune", action="store_true", help="Enable Optuna tuning")
    parser.add_argument("--no_tune", action="store_true", help="Disable Optuna tuning even if config enables it")
    parser.add_argument("--n_trials", type=int, default=None, help="Optuna trials")
    parser.add_argument(
        "--tuning_objective",
        type=str,
        default=None,
        choices=["val_r2", "val_poly_nrmse"],
        help="Regression Optuna objective for Step4_1 (Step4_2 always tunes balanced_accuracy)",
    )
    parser.add_argument(
        "--tuning_cv_folds",
        type=int,
        default=None,
        help="Number of CV folds for Optuna tuning on all non-test rows",
    )
    parser.add_argument(
        "--finetune_last_layers",
        type=int,
        default=None,
        help="Override chi_training.step4_1_regression.finetune_last_layers (valid range: 0..num_layers)",
    )
    args = parser.parse_args()
    main(args)
