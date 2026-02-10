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
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chi.data import COEFF_NAMES, SplitConfig, add_split_column, load_chi_dataset, make_split_assignments
from src.chi.embeddings import (
    build_or_load_embedding_cache,
    embedding_table_from_cache,
    load_backbone_from_step1,
)
from src.chi.metrics import classification_metrics, hit_metrics, metrics_by_group, regression_metrics
from src.chi.model import PhysicsGuidedChiModel, SolubilityClassifier
from src.utils.config import load_config, save_config
from src.utils.model_scales import get_model_config, get_results_dir
from src.utils.numerics import stable_sigmoid
from src.utils.reproducibility import save_run_metadata, seed_everything
from src.utils.reporting import save_step_summary, save_artifact_manifest, write_initial_log


@dataclass
class TrainConfig:
    split_mode: str
    train_ratio: float
    val_ratio: float
    test_ratio: float
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
    timestep_for_embedding: int
    finetune_last_layers: int
    optuna_search_space: Dict[str, object]


class ChiDataset(Dataset):
    """Row-level chi dataset with cached polymer embeddings."""

    def __init__(self, df: pd.DataFrame, embedding_table: np.ndarray):
        self.df = df.reset_index(drop=True).copy()
        self.embedding_table = embedding_table.astype(np.float32)

        polymer_ids = self.df["polymer_id"].to_numpy(dtype=np.int64)
        self.embedding = torch.tensor(self.embedding_table[polymer_ids], dtype=torch.float32)
        self.temperature = torch.tensor(self.df["temperature"].to_numpy(dtype=np.float32), dtype=torch.float32)
        self.phi = torch.tensor(self.df["phi"].to_numpy(dtype=np.float32), dtype=torch.float32)
        self.chi = torch.tensor(self.df["chi"].to_numpy(dtype=np.float32), dtype=torch.float32)
        self.label = torch.tensor(self.df["water_soluble"].to_numpy(dtype=np.float32), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "embedding": self.embedding[idx],
            "temperature": self.temperature[idx],
            "phi": self.phi[idx],
            "chi": self.chi[idx],
            "label": self.label[idx],
        }


class ChiTokenDataset(Dataset):
    """Row-level chi dataset with tokenized polymer SMILES for backbone finetuning."""

    def __init__(self, df: pd.DataFrame, tokenizer):
        self.df = df.reset_index(drop=True).copy()
        encoded = tokenizer.batch_encode(self.df["SMILES"].astype(str).tolist())
        self.input_ids = torch.tensor(np.asarray(encoded["input_ids"], dtype=np.int64), dtype=torch.long)
        self.attention_mask = torch.tensor(np.asarray(encoded["attention_mask"], dtype=np.int64), dtype=torch.long)
        self.temperature = torch.tensor(self.df["temperature"].to_numpy(dtype=np.float32), dtype=torch.float32)
        self.phi = torch.tensor(self.df["phi"].to_numpy(dtype=np.float32), dtype=torch.float32)
        self.chi = torch.tensor(self.df["chi"].to_numpy(dtype=np.float32), dtype=torch.float32)
        self.label = torch.tensor(self.df["water_soluble"].to_numpy(dtype=np.float32), dtype=torch.float32)

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
        }


class BackbonePhysicsGuidedChiModel(nn.Module):
    """End-to-end Step 4 model: backbone encoder + physics-guided chi head."""

    def __init__(
        self,
        backbone: nn.Module,
        chi_head: PhysicsGuidedChiModel,
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
    raise ValueError("chi_training.step4_1_regression.tuning_objective must be 'val_r2'")


def _describe_tuning_objective(train_cfg: TrainConfig) -> str:
    _ = train_cfg
    return "maximize_val_r2"


def _normalize_water_soluble_column(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in {"water_soluble", "water_solubel", "water_solubility"}:
            rename_map[col] = "water_soluble"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _load_step42_classification_dataset(
    csv_path: str | Path,
    default_temperature: float = 293.15,
    default_phi: float = 0.2,
    default_chi: float = 0.0,
) -> pd.DataFrame:
    """Load Step4_2 classification dataset and normalize required columns.

    Step4_2 now supports classification-only CSV files that may only include
    Polymer/SMILES/water_soluble. Missing physics columns are filled with
    deterministic defaults so existing dataloader interfaces remain unchanged.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Step4_2 classification dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = _normalize_water_soluble_column(df)

    required_base = {"Polymer", "SMILES", "water_soluble"}
    missing_base = required_base - set(df.columns)
    if missing_base:
        raise ValueError(
            f"Step4_2 classification dataset is missing required columns: {sorted(missing_base)}"
        )

    out = df.copy()
    if "temperature" not in out.columns:
        out["temperature"] = float(default_temperature)
    if "phi" not in out.columns:
        out["phi"] = float(default_phi)
    if "chi" not in out.columns:
        out["chi"] = float(default_chi)

    out["temperature"] = out["temperature"].fillna(float(default_temperature)).astype(float)
    out["phi"] = out["phi"].fillna(float(default_phi)).astype(float)
    out["chi"] = out["chi"].fillna(float(default_chi)).astype(float)
    out["water_soluble"] = out["water_soluble"].astype(int)

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
        "dataset_path": "Data/chi/_50_polymers_T_phi.csv",
        "step4_2_dataset_path": "Data/water_solvent/water_solvent_polymers.csv",
        "split_mode": "polymer",
        "train_ratio": 0.70,
        "val_ratio": 0.14,
        "test_ratio": 0.16,
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
    out["step4_2_dataset_path"] = str(step42_cfg.get("dataset_path", step42_dataset_fallback))
    out["split_mode"] = str(shared.get("split_mode", chi_cfg.get("split_mode", defaults["split_mode"])))
    out["train_ratio"] = float(split_cfg.get("train_ratio", chi_cfg.get("train_ratio", defaults["train_ratio"])))
    out["val_ratio"] = float(split_cfg.get("val_ratio", chi_cfg.get("val_ratio", defaults["val_ratio"])))
    out["test_ratio"] = float(split_cfg.get("test_ratio", chi_cfg.get("test_ratio", defaults["test_ratio"])))
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
    split_mode = str(chi_cfg["split_mode"]).strip().lower()
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
    gradient_clip_norm = float(chi_cfg.get("gradient_clip_norm", 1.0))
    if gradient_clip_norm < 0:
        raise ValueError("chi_training.gradient_clip_norm must be >= 0")
    scheduler_min_lr = float(chi_cfg.get("scheduler_min_lr", 1.0e-6))
    if scheduler_min_lr < 0:
        raise ValueError("chi_training.scheduler_min_lr must be >= 0")

    return TrainConfig(
        split_mode=split_mode,
        train_ratio=float(chi_cfg["train_ratio"]),
        val_ratio=float(chi_cfg["val_ratio"]),
        test_ratio=float(chi_cfg["test_ratio"]),
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
        timestep_for_embedding=int(chi_cfg.get("embedding_timestep", 1)),
        finetune_last_layers=int(chi_cfg.get("finetune_last_layers", 0)),
        optuna_search_space=dict(chi_cfg.get("optuna_search_space", {})),
    )



def make_dataloaders(
    split_df: pd.DataFrame,
    embedding_table: np.ndarray,
    batch_size: int,
    shuffle_train: bool = True,
) -> Dict[str, DataLoader]:
    loaders = {}
    for split in ["train", "val", "test"]:
        ds = ChiDataset(split_df[split_df["split"] == split], embedding_table)
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
    shuffle_train: bool = True,
) -> Dict[str, DataLoader]:
    loaders = {}
    for split in ["train", "val", "test"]:
        ds = ChiTokenDataset(split_df[split_df["split"] == split], tokenizer)
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

        loss_mse = torch.nn.functional.mse_loss(out["chi_pred"], chi_true)
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
    logits: List[np.ndarray] = []
    probs: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    coeffs: List[np.ndarray] = []

    for batch in loader:
        temperature = batch["temperature"].to(device)
        phi = batch["phi"].to(device)
        chi_true = batch["chi"].to(device)
        label = batch["label"].to(device)

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
        logits.append(out["class_logit"].cpu().numpy())
        probs.append(torch.sigmoid(out["class_logit"]).cpu().numpy())
        labels.append(label.cpu().numpy())
        coeffs.append(out["coefficients"].cpu().numpy())

    return {
        "chi_true": np.concatenate(true_chi, axis=0) if true_chi else np.array([]),
        "chi_pred": np.concatenate(pred_chi, axis=0) if pred_chi else np.array([]),
        "label": np.concatenate(labels, axis=0) if labels else np.array([]),
        "logit": np.concatenate(logits, axis=0) if logits else np.array([]),
        "prob": np.concatenate(probs, axis=0) if probs else np.array([]),
        "coefficients": np.concatenate(coeffs, axis=0) if coeffs else np.array([]),
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
    if finetune_last_layers > 0:
        if config is None:
            raise ValueError("config is required when finetune_last_layers > 0")
        if tokenizer is None:
            raise ValueError("tokenizer is required when finetune_last_layers > 0")
        dataloaders = make_token_dataloaders(
            split_df,
            tokenizer,
            batch_size=batch_size,
            shuffle_train=True,
        )
        _, backbone, _ = load_backbone_from_step1(
            config=config,
            model_size=model_size,
            checkpoint_path=backbone_checkpoint,
            device=device,
        )
        _set_finetune_last_layers(backbone, finetune_last_layers=finetune_last_layers)
        chi_head = PhysicsGuidedChiModel(
            embedding_dim=int(backbone.hidden_size),
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        )
        model = BackbonePhysicsGuidedChiModel(
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
            shuffle_train=True,
        )
        model = PhysicsGuidedChiModel(
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
        "train_loss_mse": [],
        "train_loss_bce": [],
        "val_loss": [],
        "val_loss_mse": [],
        "val_loss_bce": [],
        "val_rmse": [],
    }

    best_state = None
    best_rmse = np.inf
    wait = 0

    for epoch in range(1, num_epochs + 1):
        train_stats = run_epoch(
            model=model,
            loader=dataloaders["train"],
            optimizer=optimizer,
            device=device,
            gradient_clip_norm=float(train_cfg.gradient_clip_norm),
        )
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

        history["epoch"].append(epoch)
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
        history["train_loss"].append(train_stats["loss"])
        history["train_loss_mse"].append(train_stats["loss_mse"])
        history["train_loss_bce"].append(train_stats["loss_bce"])
        history["val_loss"].append(val_stats["loss"])
        history["val_loss_mse"].append(val_stats["loss_mse"])
        history["val_loss_bce"].append(val_stats["loss_bce"])
        history["val_rmse"].append(val_rmse)

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
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
    """Create stratified CV folds from the train+val subset for robust hyperparameter tuning."""
    dev_df = split_df[split_df["split"].isin(["train", "val"])].copy().reset_index(drop=True)
    if dev_df.empty:
        raise ValueError("No train/val rows available for Optuna tuning.")

    requested_folds = int(max(2, train_cfg.tuning_cv_folds))

    if train_cfg.split_mode == "polymer":
        unit_df = (
            dev_df[["polymer_id", "water_soluble"]]
            .drop_duplicates(subset=["polymer_id"])
            .sort_values("polymer_id")
            .reset_index(drop=True)
        )
        unit_key = "polymer_id"
        strategy = "polymer_group_stratified"
    else:
        unit_df = (
            dev_df[["row_id", "water_soluble"]]
            .drop_duplicates(subset=["row_id"])
            .sort_values("row_id")
            .reset_index(drop=True)
        )
        unit_key = "row_id"
        strategy = "row_stratified"

    class_counts = unit_df["water_soluble"].value_counts()
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
    labels = unit_df["water_soluble"].to_numpy(dtype=int)

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
            n_pos = int(sub["water_soluble"].sum()) if n_rows > 0 else 0
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
        fold_rows.append(
            {
                "fold": fold_id,
                "val_n": int(len(val_pred["chi_true"])),
                "val_r2": float(val_reg["r2"]),
                "val_rmse": float(val_reg["rmse"]),
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
                val_sub[["fold", "polymer_id", "Polymer", "SMILES", "water_soluble", "chi_true", "chi_pred"]].copy()
            )

    fold_metrics_df = pd.DataFrame(fold_rows)
    mean_r2 = float(np.nanmean(fold_metrics_df["val_r2"])) if not fold_metrics_df.empty else np.nan
    mean_rmse = float(np.nanmean(fold_metrics_df["val_rmse"])) if not fold_metrics_df.empty else np.nan
    cv_val_df = pd.concat(cv_val_frames, ignore_index=True) if cv_val_frames else pd.DataFrame()
    return {
        "cv_val_r2": mean_r2,
        "cv_val_rmse": mean_rmse,
        "fold_metrics": fold_metrics_df,
        "cv_val_predictions": cv_val_df,
    }


def _save_cv_parity_by_fold_figure(
    cv_val_df: pd.DataFrame,
    out_png: Path,
    dpi: int,
    font_size: int,
) -> None:
    if cv_val_df.empty:
        return

    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "legend.fontsize": font_size,
        }
    )
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
    ax.set_title(f"CV parity by fold (val folds, n={len(plot_df)})")
    ax.legend(
        title="CV fold",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


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
    font_size: int = 12,
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

    num_layers_space = _as_list("num_layers", [1, 2, 3])
    hidden_units_space = [int(v) for v in _as_list("hidden_units", [64, 128, 256, 512])]
    dropout_space = [float(v) for v in _as_list("dropout", [0.0, 0.1, 0.2, 0.3])]
    lr_space = [float(v) for v in _as_list("learning_rate", [1e-4, 5e-3])]
    wd_space = [float(v) for v in _as_list("weight_decay", [1e-7, 1e-3])]
    batch_size_space = [int(v) for v in _as_list("batch_size", [16, 32, 64, 128, 256])]
    lr_log = bool(search_space.get("learning_rate_log", True))
    wd_log = bool(search_space.get("weight_decay_log", True))
    finetune_raw = search_space.get("finetune_last_layers", [0, int(backbone_num_layers)])
    if isinstance(finetune_raw, list) and len(finetune_raw) > 0:
        finetune_space = [int(v) for v in finetune_raw]
    else:
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
        invalid_metrics = int((not np.isfinite(val_r2)) or (not np.isfinite(val_rmse)))
        trial.set_user_attr("val_r2", val_r2)
        trial.set_user_attr("val_rmse", val_rmse)
        trial.set_user_attr("cv_val_r2", val_r2)
        trial.set_user_attr("cv_val_rmse", val_rmse)
        trial.set_user_attr("cv_n_folds", int(len(cv_folds)))
        trial.set_user_attr("tuning_objective", objective_name)
        trial.set_user_attr("invalid_metrics", invalid_metrics)
        if invalid_metrics:
            return -1.0e12
        return val_r2

    study = optuna.create_study(direction=objective_direction)
    study.optimize(objective, n_trials=train_cfg.n_trials, show_progress_bar=True)

    trials = []
    for t in study.trials:
        val_r2 = t.user_attrs.get("cv_val_r2", t.user_attrs.get("val_r2", np.nan))
        val_rmse = t.user_attrs.get("cv_val_rmse", t.user_attrs.get("val_rmse", np.nan))
        row = {
            "trial": t.number,
            "state": str(t.state),
            "objective_name": objective_name,
            "objective_direction": objective_direction,
            "objective_value": t.value,
            "value_val_r2": val_r2,
            "val_r2": val_r2,
            "val_rmse": val_rmse,
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
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "legend.fontsize": font_size,
        }
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(trial_df["trial"], trial_df["chi_val_r2"], "o", color="#1f77b4", label="Trial chi R2", alpha=0.85)
    ax.plot(
        trial_df["trial"],
        trial_df["best_chi_val_r2_so_far"],
        "-",
        color="#d62728",
        linewidth=2,
        label="Best chi R2 so far",
    )
    ax.set_xlabel("Optuna trial")
    ax.set_ylabel("Validation chi R2")
    ax.set_title("Optuna optimization process based on chi R2")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(tuning_dir / "optuna_optimization_chi_r2.png", dpi=dpi)
    plt.close(fig)

    # Figure: trial-by-trial objective and running best objective.
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(trial_df["trial"], trial_df["objective_value"], "o", color="#2a9d8f", label="Trial objective", alpha=0.85)
    ax.plot(
        trial_df["trial"],
        trial_df["best_objective_so_far"],
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
        sub["class_logit"] = pred["logit"]
        sub["class_prob"] = pred["prob"]
        sub["class_pred"] = (sub["class_prob"] >= 0.5).astype(int)

        coeff = pred["coefficients"]
        for i, name in enumerate(COEFF_NAMES):
            sub[name] = coeff[:, i]

        sub.to_csv(out_dir / f"chi_predictions_{split}.csv", index=False)
        frames.append(sub)

    all_df = pd.concat(frames, axis=0, ignore_index=True)
    all_df.to_csv(out_dir / "chi_predictions_all.csv", index=False)
    return all_df



def _collect_metrics(pred_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    class_rows = []
    polymer_rows = []

    for split, sub in pred_df.groupby("split"):
        reg = regression_metrics(sub["chi"], sub["chi_pred"])
        cls = classification_metrics(sub["water_soluble"], sub["class_prob"])
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
            group_col="water_soluble",
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

    pd.DataFrame(rows).to_csv(out_dir / "chi_metrics_overall.csv", index=False)
    pd.concat(class_rows, ignore_index=True).to_csv(out_dir / "chi_metrics_by_class.csv", index=False)
    pd.DataFrame(polymer_rows).to_csv(out_dir / "chi_metrics_polymer_level.csv", index=False)



def _plot_parity_panel(ax, sub: pd.DataFrame, split: str, show_legend: bool) -> None:
    if sub.empty:
        ax.set_axis_off()
        ax.set_title(f"{split.upper()} (empty)")
        return

    sns.scatterplot(
        data=sub,
        x="chi",
        y="chi_pred",
        hue="water_soluble",
        palette={1: "#1f77b4", 0: "#d62728"},
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
            handles = legend.legendHandles
            labels = [t.get_text() for t in legend.get_texts()]
            legend.remove()
            ax.legend(
                handles=handles,
                labels=labels,
                title="water_soluble",
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
            )
        else:
            legend.remove()


def _make_figures(history: Dict[str, List[float]], pred_df: pd.DataFrame, fig_dir: Path, dpi: int, font_size: int) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "legend.fontsize": font_size,
    })

    # Loss curve
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(history["epoch"], history["train_loss"], label="Train loss")
    ax.plot(history["epoch"], history["val_loss"], label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Step4 chi training loss")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(fig_dir / "chi_loss_curve.png", dpi=dpi)
    plt.close(fig)

    # Test-only parity (water_soluble legend is only shown on test plot).
    test = pred_df[pred_df["split"] == "test"].copy()
    fig, ax = plt.subplots(figsize=(6, 5))
    _plot_parity_panel(ax, sub=test, split="test", show_legend=True)
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    fig.savefig(fig_dir / "chi_parity_test.png", dpi=dpi)
    plt.close(fig)

    # Residual histogram by split
    fig, ax = plt.subplots(figsize=(6, 5))
    for split, color in [("train", "#4c78a8"), ("val", "#f58518"), ("test", "#54a24b")]:
        sub = pred_df[pred_df["split"] == split]
        sns.kdeplot(sub["chi_error"], ax=ax, label=split, color=color, linewidth=2)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("χ prediction error")
    ax.set_title("Residual distribution by split")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(fig_dir / "chi_residual_distribution.png", dpi=dpi)
    plt.close(fig)

    # ROC and PR (test)
    y_true = test["water_soluble"].to_numpy(dtype=int)
    y_prob = test["class_prob"].to_numpy(dtype=float)
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color="#1f77b4", linewidth=2)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("Classifier ROC (test)")
        fig.tight_layout()
        fig.savefig(fig_dir / "chi_classifier_roc_test.png", dpi=dpi)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(recall, precision, color="#d62728", linewidth=2)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Classifier PR (test)")
        fig.tight_layout()
        fig.savefig(fig_dir / "chi_classifier_pr_test.png", dpi=dpi)
        plt.close(fig)



def _save_coefficient_summary(
    model: nn.Module,
    embedding_cache_df: pd.DataFrame,
    out_csv: Path,
    device: str,
    tokenizer=None,
    timestep: int = 1,
) -> None:
    model.eval()
    if isinstance(model, BackbonePhysicsGuidedChiModel):
        if tokenizer is None:
            raise ValueError("tokenizer is required to save coefficients for backbone-finetuned model")
        rec = embedding_cache_df[["polymer_id", "Polymer", "SMILES", "water_soluble"]].copy().reset_index(drop=True)
        encoded = tokenizer.batch_encode(rec["SMILES"].astype(str).tolist())
        input_ids = torch.tensor(np.asarray(encoded["input_ids"], dtype=np.int64), dtype=torch.long, device=device)
        attention_mask = torch.tensor(np.asarray(encoded["attention_mask"], dtype=np.int64), dtype=torch.long, device=device)
        timesteps = torch.full((int(input_ids.shape[0]),), int(timestep), device=device, dtype=torch.long)
        with torch.no_grad():
            emb_t = model.backbone.get_pooled_output(
                input_ids=input_ids,
                timesteps=timesteps,
                attention_mask=attention_mask,
                pooling="mean",
            )
            features = model.chi_head.encoder(emb_t)
            coeff = model.chi_head.coeff_head(features).cpu().numpy()
            logit = model.chi_head.class_head(features).squeeze(-1).cpu().numpy()
            prob = stable_sigmoid(logit)
        df = rec
    else:
        emb = np.stack(embedding_cache_df["embedding"].to_list(), axis=0)
        emb_t = torch.tensor(emb, dtype=torch.float32, device=device)
        with torch.no_grad():
            features = model.encoder(emb_t)
            coeff = model.coeff_head(features).cpu().numpy()
            logit = model.class_head(features).squeeze(-1).cpu().numpy()
            prob = stable_sigmoid(logit)
        df = embedding_cache_df[["polymer_id", "Polymer", "SMILES", "water_soluble"]].copy()

    for i, name in enumerate(COEFF_NAMES):
        df[name] = coeff[:, i]
    df["class_logit"] = logit
    df["class_prob"] = prob
    df.to_csv(out_csv, index=False)


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
    best_val_loss = np.inf
    wait = 0

    for epoch in range(1, num_epochs + 1):
        train_stats = run_classifier_epoch(
            model=model,
            loader=dataloaders["train"],
            optimizer=optimizer,
            device=device,
            gradient_clip_norm=float(train_cfg.gradient_clip_norm),
        )
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

        history["epoch"].append(epoch)
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
        history["train_loss"].append(train_stats["loss"])
        history["val_loss"].append(val_stats["loss"])
        history["val_balanced_accuracy"].append(val_bal_acc)

        val_loss = float(val_stats["loss"])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
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
    font_size: int = 12,
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

    num_layers_space = _as_list("num_layers", [1, 2, 3])
    hidden_units_space = [int(v) for v in _as_list("hidden_units", [64, 128, 256, 512])]
    dropout_space = [float(v) for v in _as_list("dropout", [0.0, 0.1, 0.2, 0.3])]
    lr_space = [float(v) for v in _as_list("learning_rate", [1e-4, 5e-3])]
    wd_space = [float(v) for v in _as_list("weight_decay", [1e-7, 1e-3])]
    batch_size_space = [int(v) for v in _as_list("batch_size", [16, 32, 64, 128, 256])]
    lr_log = bool(search_space.get("learning_rate_log", True))
    wd_log = bool(search_space.get("weight_decay_log", True))
    finetune_raw = search_space.get("finetune_last_layers", [0, int(backbone_num_layers)])
    if isinstance(finetune_raw, list) and len(finetune_raw) > 0:
        finetune_space = [int(v) for v in finetune_raw]
    else:
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

    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "legend.fontsize": font_size,
        }
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(trial_df["trial"], trial_df["objective_value"], "o", color="#2a9d8f", label="Trial balanced accuracy", alpha=0.85)
    ax.plot(
        trial_df["trial"],
        trial_df["best_objective_so_far"],
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
    all_df = pd.concat(frames, axis=0, ignore_index=True)
    all_df.to_csv(out_dir / "class_predictions_all.csv", index=False)
    return all_df


def _collect_classifier_metrics(pred_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, sub in pred_df.groupby("split"):
        cls = classification_metrics(sub["water_soluble"], sub["class_prob"])
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
        coeff = pred["coefficients"]
        for i, name in enumerate(COEFF_NAMES):
            sub[name] = coeff[:, i]
        sub.to_csv(out_dir / f"chi_predictions_{split}.csv", index=False)
        frames.append(sub)
    all_df = pd.concat(frames, axis=0, ignore_index=True)
    all_df.to_csv(out_dir / "chi_predictions_all.csv", index=False)
    return all_df


def _collect_regression_metrics(pred_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    class_rows = []
    polymer_rows = []
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
            group_col="water_soluble",
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

    pd.DataFrame(rows).to_csv(out_dir / "chi_metrics_overall.csv", index=False)
    pd.concat(class_rows, ignore_index=True).to_csv(out_dir / "chi_metrics_by_class.csv", index=False)
    pd.DataFrame(polymer_rows).to_csv(out_dir / "chi_metrics_polymer_level.csv", index=False)


def _make_regression_figures(
    history: Dict[str, List[float]],
    pred_df: pd.DataFrame,
    fig_dir: Path,
    dpi: int,
    font_size: int,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "legend.fontsize": font_size,
    })

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(history["epoch"], history["train_loss"], label="Train loss")
    ax.plot(history["epoch"], history["val_loss"], label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Step4_1 regression loss")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(fig_dir / "chi_loss_curve.png", dpi=dpi)
    plt.close(fig)

    # Test-only parity (water_soluble legend is only shown on test plot).
    test = pred_df[pred_df["split"] == "test"].copy()
    fig, ax = plt.subplots(figsize=(6, 5))
    _plot_parity_panel(ax, sub=test, split="test", show_legend=True)
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    fig.savefig(fig_dir / "chi_parity_test.png", dpi=dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for split, color in [("train", "#4c78a8"), ("val", "#f58518"), ("test", "#54a24b")]:
        sub = pred_df[pred_df["split"] == split]
        sns.kdeplot(sub["chi_error"], ax=ax, label=split, color=color, linewidth=2)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("chi prediction error")
    ax.set_title("Residual distribution by split")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
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
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "legend.fontsize": font_size,
    })

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(history["epoch"], history["train_loss"], label="Train BCE")
    ax.plot(history["epoch"], history["val_loss"], label="Val BCE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title("Step4_2 classification loss")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(fig_dir / "class_loss_curve.png", dpi=dpi)
    plt.close(fig)

    test = pred_df[pred_df["split"] == "test"].copy()
    if not test.empty:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.kdeplot(
            data=test,
            x="class_prob",
            hue="water_soluble",
            common_norm=False,
            fill=True,
            alpha=0.3,
            ax=ax,
        )
        ax.set_xlabel("Predicted soluble probability")
        ax.set_title("Class probability distribution (test)")
        legend = ax.get_legend()
        if legend is not None:
            handles = legend.legendHandles
            labels = [t.get_text() for t in legend.get_texts()]
            legend.remove()
            ax.legend(
                handles=handles,
                labels=labels,
                title="water_soluble",
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
            )
        fig.tight_layout(rect=(0, 0, 0.82, 1))
        fig.savefig(fig_dir / "class_prob_distribution_test.png", dpi=dpi)
        plt.close(fig)

        y_true = test["water_soluble"].to_numpy(dtype=int)
        y_prob = test["class_prob"].to_numpy(dtype=float)
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            precision, recall, _ = precision_recall_curve(y_true, y_prob)

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, color="#1f77b4", linewidth=2)
            ax.plot([0, 1], [0, 1], "k--", linewidth=1)
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.set_title("Classifier ROC (test)")
            fig.tight_layout()
            fig.savefig(fig_dir / "classifier_roc_test.png", dpi=dpi)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(recall, precision, color="#d62728", linewidth=2)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Classifier PR (test)")
            fig.tight_layout()
            fig.savefig(fig_dir / "classifier_pr_test.png", dpi=dpi)
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
            "coefficients": reg["coefficients"],
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
    reg_model.eval()
    cls_model.eval()

    if isinstance(reg_model, BackbonePhysicsGuidedChiModel):
        if tokenizer is None:
            raise ValueError("tokenizer is required to save coefficients for backbone-finetuned model")
        rec = embedding_cache_df[["polymer_id", "Polymer", "SMILES", "water_soluble"]].copy().reset_index(drop=True)
        encoded = tokenizer.batch_encode(rec["SMILES"].astype(str).tolist())
        input_ids = torch.tensor(np.asarray(encoded["input_ids"], dtype=np.int64), dtype=torch.long, device=device)
        attention_mask = torch.tensor(np.asarray(encoded["attention_mask"], dtype=np.int64), dtype=torch.long, device=device)
        timesteps = torch.full((int(input_ids.shape[0]),), int(timestep), device=device, dtype=torch.long)
        with torch.no_grad():
            emb_t = reg_model.backbone.get_pooled_output(
                input_ids=input_ids,
                timesteps=timesteps,
                attention_mask=attention_mask,
                pooling="mean",
            )
            features = reg_model.chi_head.encoder(emb_t)
            coeff = reg_model.chi_head.coeff_head(features).cpu().numpy()
        df = rec
    else:
        emb = np.stack(embedding_cache_df["embedding"].to_list(), axis=0)
        emb_t = torch.tensor(emb, dtype=torch.float32, device=device)
        with torch.no_grad():
            features = reg_model.encoder(emb_t)
            coeff = reg_model.coeff_head(features).cpu().numpy()
        df = embedding_cache_df[["polymer_id", "Polymer", "SMILES", "water_soluble"]].copy()

    emb_for_cls = np.stack(embedding_cache_df["embedding"].to_list(), axis=0)
    emb_cls_t = torch.tensor(emb_for_cls, dtype=torch.float32, device=device)
    with torch.no_grad():
        cls_out = cls_model(embedding=emb_cls_t)
        class_logit = cls_out["class_logit"].detach().cpu().numpy()
        class_prob = torch.sigmoid(cls_out["class_logit"]).detach().cpu().numpy()

    for i, name in enumerate(COEFF_NAMES):
        df[name] = coeff[:, i]
    df["class_logit"] = class_logit
    df["class_prob"] = class_prob
    df.to_csv(out_csv, index=False)


def main(args):
    config = load_config(args.config)
    train_cfg = build_train_config(args, config)
    if args.finetune_last_layers is not None:
        train_cfg.finetune_last_layers = int(args.finetune_last_layers)

    backbone_cfg = get_model_config(args.model_size, config, model_type="sequence")
    backbone_num_layers = int(backbone_cfg["num_layers"])
    if train_cfg.finetune_last_layers < 0 or train_cfg.finetune_last_layers > backbone_num_layers:
        raise ValueError(
            f"chi_training.step4_1_regression.finetune_last_layers must be in [0, {backbone_num_layers}] "
            f"for model_size={args.model_size}, got {train_cfg.finetune_last_layers}"
        )
    use_backbone_finetune = bool(train_cfg.finetune_last_layers > 0)

    results_dir = Path(get_results_dir(args.model_size, config["paths"]["results_dir"]))
    step_dir = results_dir / "step4_chi_training" / train_cfg.split_mode
    shared_dir = step_dir / "shared"
    pipeline_metrics_dir = step_dir / "pipeline_metrics"
    reg_dir = step_dir / "step4_1_regression"
    cls_dir = step_dir / "step4_2_classification"
    reg_metrics_dir = reg_dir / "metrics"
    reg_figures_dir = reg_dir / "figures"
    reg_tuning_dir = reg_dir / "tuning"
    reg_checkpoint_dir = reg_dir / "checkpoints"
    cls_metrics_dir = cls_dir / "metrics"
    cls_figures_dir = cls_dir / "figures"
    cls_tuning_dir = cls_dir / "tuning"
    cls_checkpoint_dir = cls_dir / "checkpoints"
    legacy_checkpoint_dir = results_dir / "checkpoints"

    for d in [
        shared_dir,
        pipeline_metrics_dir,
        reg_metrics_dir,
        reg_figures_dir,
        reg_tuning_dir,
        reg_checkpoint_dir,
        cls_metrics_dir,
        cls_figures_dir,
        cls_tuning_dir,
        cls_checkpoint_dir,
        legacy_checkpoint_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    chi_cfg = _default_chi_config(config)
    reg_csv = args.regression_dataset_path or args.dataset_path or chi_cfg["step4_1_dataset_path"]
    cls_csv = args.classification_dataset_path or chi_cfg["step4_2_dataset_path"]

    seed_info = seed_everything(train_cfg.seed)
    save_config(config, step_dir / "config_used.yaml")
    save_run_metadata(step_dir, args.config, seed_info)
    write_initial_log(
        step_dir=step_dir,
        step_name="step4_chi_training",
        context={
            "config_path": args.config,
            "model_size": args.model_size,
            "results_dir": str(results_dir),
            "split_mode": train_cfg.split_mode,
            "dataset_path": str(reg_csv),
            "step4_1_dataset_path": str(reg_csv),
            "step4_2_dataset_path": str(cls_csv),
            "tune": train_cfg.tune,
            "n_trials": train_cfg.n_trials,
            "tuning_objective": train_cfg.tuning_objective,
            "tuning_cv_folds": train_cfg.tuning_cv_folds,
            "gradient_clip_norm": train_cfg.gradient_clip_norm,
            "use_scheduler": train_cfg.use_scheduler,
            "scheduler_min_lr": train_cfg.scheduler_min_lr,
            "backbone_num_layers": backbone_num_layers,
            "finetune_last_layers": train_cfg.finetune_last_layers,
            "stage4_1_dir": str(reg_dir),
            "stage4_2_dir": str(cls_dir),
            "random_seed": train_cfg.seed,
        },
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("Step 4 split pipeline:")
    print("  Step4_1: chi regression")
    print("  Step4_2: water-soluble classification")
    print(f"Split mode: {train_cfg.split_mode}")
    print(f"Step4_1 dataset: {reg_csv}")
    print(f"Step4_2 dataset: {cls_csv}")
    print(f"finetune_last_layers (regression): {train_cfg.finetune_last_layers}/{backbone_num_layers}")
    print(f"Device: {device}")
    print("=" * 70)

    reg_df = load_chi_dataset(reg_csv)
    reg_split_assign = make_split_assignments(
        reg_df,
        SplitConfig(
            split_mode=train_cfg.split_mode,
            train_ratio=train_cfg.train_ratio,
            val_ratio=train_cfg.val_ratio,
            test_ratio=train_cfg.test_ratio,
            seed=train_cfg.seed,
        ),
    )
    reg_split_df = add_split_column(reg_df, reg_split_assign)

    cls_df = _load_step42_classification_dataset(cls_csv)
    cls_split_assign = make_split_assignments(
        cls_df,
        SplitConfig(
            split_mode=train_cfg.split_mode,
            train_ratio=train_cfg.train_ratio,
            val_ratio=train_cfg.val_ratio,
            test_ratio=train_cfg.test_ratio,
            seed=train_cfg.seed,
        ),
    )
    cls_split_df = add_split_column(cls_df, cls_split_assign)

    # Shared outputs: keep legacy regression filenames plus explicit per-stage split files.
    reg_split_assign.to_csv(shared_dir / "split_assignments.csv", index=False)
    reg_split_df.to_csv(shared_dir / "chi_dataset_with_split.csv", index=False)
    reg_split_assign.to_csv(shared_dir / "split_assignments_step4_1.csv", index=False)
    reg_split_df.to_csv(shared_dir / "chi_dataset_with_split_step4_1.csv", index=False)
    cls_split_assign.to_csv(shared_dir / "split_assignments_step4_2.csv", index=False)
    cls_split_df.to_csv(shared_dir / "chi_dataset_with_split_step4_2.csv", index=False)

    reg_split_assign.to_csv(reg_metrics_dir / "split_assignments.csv", index=False)
    reg_split_df.to_csv(reg_metrics_dir / "chi_dataset_with_split.csv", index=False)
    cls_split_assign.to_csv(cls_metrics_dir / "split_assignments.csv", index=False)
    cls_split_df.to_csv(cls_metrics_dir / "chi_dataset_with_split.csv", index=False)

    reg_polymer_df = reg_split_df[["polymer_id", "Polymer", "SMILES", "water_soluble"]].drop_duplicates("polymer_id")
    reg_emb_cache = build_or_load_embedding_cache(
        polymer_df=reg_polymer_df,
        config=config,
        cache_npz=shared_dir / "polymer_embeddings.npz",
        model_size=args.model_size,
        checkpoint_path=args.backbone_checkpoint,
        device=device,
        timestep=train_cfg.timestep_for_embedding,
        pooling="mean",
        batch_size=int(chi_cfg["embedding_batch_size"]),
    )
    reg_embedding_table = embedding_table_from_cache(reg_emb_cache)

    cls_polymer_df = cls_split_df[["polymer_id", "Polymer", "SMILES", "water_soluble"]].drop_duplicates("polymer_id")
    cls_emb_cache = build_or_load_embedding_cache(
        polymer_df=cls_polymer_df,
        config=config,
        cache_npz=shared_dir / "polymer_embeddings_step4_2_classification.npz",
        model_size=args.model_size,
        checkpoint_path=args.backbone_checkpoint,
        device=device,
        timestep=train_cfg.timestep_for_embedding,
        pooling="mean",
        batch_size=int(chi_cfg["embedding_batch_size"]),
    )
    cls_embedding_table = embedding_table_from_cache(cls_emb_cache)

    tokenizer_for_training = None
    if use_backbone_finetune or train_cfg.tune:
        tokenizer_for_training, _, _ = load_backbone_from_step1(
            config=config,
            model_size=args.model_size,
            checkpoint_path=args.backbone_checkpoint,
            device="cpu",
        )

    dpi = int(config.get("plotting", {}).get("dpi", 300))
    font_size = int(config.get("plotting", {}).get("font_size", 12))

    # -------------------------
    # Step 4_1: Regression
    # -------------------------
    reg_cfg = copy.deepcopy(train_cfg)
    reg_cfg.tuning_objective = "val_r2"

    reg_best_params = None
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

    with open(reg_metrics_dir / "chosen_hyperparameters.json", "w") as f:
        json.dump(reg_chosen, f, indent=2)
    with open(reg_metrics_dir / "hyperparameter_selection_summary.json", "w") as f:
        json.dump(
            {
                "used_optuna": bool(reg_cfg.tune),
                "optuna_objective": _describe_tuning_objective(reg_cfg),
                "tuning_objective": "val_r2",
                "tuning_cv_folds": int(reg_cfg.tuning_cv_folds),
                "optuna_best_params": reg_best_params,
                "backbone_num_layers": int(backbone_num_layers),
                "finetune_last_layers": int(reg_finetune_last_layers),
                "backbone_finetune_enabled": bool(reg_use_backbone_finetune),
                "final_training_hyperparameters": reg_chosen,
                "final_training_num_epochs": int(reg_cfg.num_epochs),
                "final_training_patience": int(reg_cfg.patience),
            },
            f,
            indent=2,
        )

    reg_model, reg_history, reg_predictions = train_one_model(
        split_df=reg_split_df,
        embedding_table=reg_embedding_table,
        train_cfg=reg_cfg,
        device=device,
        hidden_sizes=reg_chosen["hidden_sizes"],
        dropout=reg_chosen["dropout"],
        learning_rate=reg_chosen["learning_rate"],
        weight_decay=reg_chosen["weight_decay"],
        batch_size=reg_chosen["batch_size"],
        num_epochs=reg_cfg.num_epochs,
        patience=reg_cfg.patience,
        config=config,
        model_size=args.model_size,
        backbone_checkpoint=args.backbone_checkpoint,
        tokenizer=tokenizer_for_training,
        finetune_last_layers=reg_finetune_last_layers,
        timestep_for_embedding=reg_cfg.timestep_for_embedding,
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
        "backbone_num_layers": int(backbone_num_layers),
        "finetune_last_layers": int(reg_finetune_last_layers),
        "backbone_finetune_enabled": bool(reg_use_backbone_finetune),
        "dataset_path": str(reg_csv),
        "config": config,
    }
    reg_checkpoint_path = reg_checkpoint_dir / "chi_regression_best.pt"
    reg_legacy_path = legacy_checkpoint_dir / "chi_regression_best.pt"
    reg_legacy_joint_path = legacy_checkpoint_dir / "chi_physics_best.pt"
    torch.save(reg_checkpoint, reg_checkpoint_path)
    torch.save(reg_checkpoint, reg_checkpoint_dir / "chi_physics_best.pt")
    # Legacy compatibility paths (deprecated)
    torch.save(reg_checkpoint, reg_legacy_path)
    torch.save(reg_checkpoint, reg_legacy_joint_path)

    _save_history(reg_history, reg_metrics_dir / "chi_training_history.csv")
    reg_pred_df = _save_regression_prediction_csvs(split_df=reg_split_df, predictions=reg_predictions, out_dir=reg_metrics_dir)
    _collect_regression_metrics(reg_pred_df, out_dir=reg_metrics_dir)
    _make_regression_figures(
        history=reg_history,
        pred_df=reg_pred_df,
        fig_dir=reg_figures_dir,
        dpi=dpi,
        font_size=font_size,
    )
    reg_poly = reg_pred_df[["polymer_id", "Polymer", "SMILES", "water_soluble"] + COEFF_NAMES].drop_duplicates("polymer_id")
    reg_poly.to_csv(reg_metrics_dir / "polymer_coefficients_regression_only.csv", index=False)

    # -------------------------
    # Step 4_2: Classification
    # -------------------------
    cls_cfg = copy.deepcopy(train_cfg)
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
    cls_cfg.tuning_epochs = int(cls_section.get("tuning_epochs", cls_cfg.tuning_epochs))
    cls_cfg.tuning_patience = int(cls_section.get("tuning_patience", cls_cfg.tuning_patience))
    if args.tuning_cv_folds is None:
        cls_cfg.tuning_cv_folds = int(cls_section.get("tuning_cv_folds", cls_cfg.tuning_cv_folds))
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
            checkpoint_path=args.backbone_checkpoint,
            device="cpu",
        )

    cls_best_params = None
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
            },
            f,
            indent=2,
        )

    cls_model, cls_history, cls_predictions = train_one_classifier_model(
        split_df=cls_split_df,
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
        "dataset_path": str(cls_csv),
        "config": config,
    }
    cls_checkpoint_path = cls_checkpoint_dir / "chi_classifier_best.pt"
    cls_legacy_path = legacy_checkpoint_dir / "chi_classifier_best.pt"
    torch.save(cls_checkpoint, cls_checkpoint_path)
    # Legacy compatibility path (deprecated)
    torch.save(cls_checkpoint, cls_legacy_path)

    _save_history(cls_history, cls_metrics_dir / "class_training_history.csv")
    cls_pred_df = _save_classifier_prediction_csvs(split_df=cls_split_df, predictions=cls_predictions, out_dir=cls_metrics_dir)
    _collect_classifier_metrics(cls_pred_df, out_dir=cls_metrics_dir)
    _make_classifier_figures(
        history=cls_history,
        pred_df=cls_pred_df,
        fig_dir=cls_figures_dir,
        dpi=dpi,
        font_size=font_size,
    )

    # Pipeline-level metadata (no combined model outputs).
    with open(pipeline_metrics_dir / "chosen_hyperparameters.json", "w") as f:
        json.dump(
            {
                "step4_1_regression": reg_chosen,
                "step4_2_classification": cls_chosen,
            },
            f,
            indent=2,
        )
    with open(pipeline_metrics_dir / "hyperparameter_selection_summary.json", "w") as f:
        json.dump(
            {
                "used_optuna": bool(train_cfg.tune),
                "step4_1_regression": {
                    "objective": "maximize_val_r2",
                    "optuna_best_params": reg_best_params,
                    "final_training_hyperparameters": reg_chosen,
                },
                "step4_2_classification": {
                    "objective": "maximize_val_balanced_accuracy",
                    "optuna_best_params": cls_best_params,
                    "final_training_hyperparameters": cls_chosen,
                },
                "tuning_cv_folds": int(train_cfg.tuning_cv_folds),
            },
            f,
            indent=2,
        )

    reg_overall_df = pd.read_csv(reg_metrics_dir / "chi_metrics_overall.csv")
    cls_overall_df = pd.read_csv(cls_metrics_dir / "class_metrics_overall.csv")
    test_rows = reg_overall_df[reg_overall_df["split"] == "test"]
    test_row = test_rows.iloc[0] if len(test_rows) > 0 else reg_overall_df.iloc[-1]
    cls_test_rows = cls_overall_df[cls_overall_df["split"] == "test"]
    cls_test_row = cls_test_rows.iloc[0] if len(cls_test_rows) > 0 else cls_overall_df.iloc[-1]
    summary = {
        "step": "step4_chi_training",
        "model_size": args.model_size or "small",
        "split_mode": train_cfg.split_mode,
        "step4_1_dataset_path": str(reg_csv),
        "step4_2_dataset_path": str(cls_csv),
        "n_data_rows": int(len(reg_split_df)),
        "n_polymers": int(reg_split_df["polymer_id"].nunique()),
        "n_data_rows_step4_1": int(len(reg_split_df)),
        "n_polymers_step4_1": int(reg_split_df["polymer_id"].nunique()),
        "n_data_rows_step4_2": int(len(cls_split_df)),
        "n_polymers_step4_2": int(cls_split_df["polymer_id"].nunique()),
        "step4_1_backbone_num_layers": int(backbone_num_layers),
        "step4_1_finetune_last_layers": int(reg_finetune_last_layers),
        "step4_1_backbone_finetune_enabled": bool(reg_use_backbone_finetune),
        "step4_2_backbone_num_layers": int(backbone_num_layers),
        "step4_2_finetune_last_layers": int(cls_finetune_last_layers),
        "step4_2_backbone_finetune_enabled": bool(cls_use_backbone_finetune),
        "step4_1_n_epochs": int(len(reg_history["epoch"])),
        "step4_2_n_epochs": int(len(cls_history["epoch"])),
        "step4_1_metrics_dir": str(reg_metrics_dir),
        "step4_2_metrics_dir": str(cls_metrics_dir),
        "step4_1_checkpoint": str(reg_checkpoint_path),
        "step4_2_checkpoint": str(cls_checkpoint_path),
        "test_mae": float(test_row["mae"]) if "mae" in test_row else np.nan,
        "test_rmse": float(test_row["rmse"]) if "rmse" in test_row else np.nan,
        "test_r2": float(test_row["r2"]) if "r2" in test_row else np.nan,
        "test_balanced_accuracy": float(cls_test_row["balanced_accuracy"]) if "balanced_accuracy" in cls_test_row else np.nan,
        "test_auroc": float(cls_test_row["auroc"]) if "auroc" in cls_test_row else np.nan,
    }
    save_step_summary(summary, pipeline_metrics_dir)
    save_artifact_manifest(step_dir=reg_dir, metrics_dir=reg_metrics_dir, figures_dir=reg_figures_dir)
    save_artifact_manifest(step_dir=cls_dir, metrics_dir=cls_metrics_dir, figures_dir=cls_figures_dir)

    print("Training complete.")
    print(f"Pipeline outputs: {step_dir}")
    print(f"Step4_1 outputs: {reg_dir}")
    print(f"Step4_2 outputs: {cls_dir}")
    print(f"Step4_1 checkpoint: {reg_checkpoint_path}")
    print(f"Step4_2 checkpoint: {cls_checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4: train Step4_1 regression + Step4_2 classification models")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config path")
    parser.add_argument("--model_size", type=str, default="small", choices=["small", "medium", "large", "xl"], help="Step1 model size tag")
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
        help="Path to Step4_2 classification dataset CSV (supports Polymer/SMILES/water_soluble).",
    )
    parser.add_argument("--backbone_checkpoint", type=str, default=None, help="Optional backbone checkpoint override")
    parser.add_argument("--tune", action="store_true", help="Enable Optuna tuning")
    parser.add_argument("--no_tune", action="store_true", help="Disable Optuna tuning even if config enables it")
    parser.add_argument("--n_trials", type=int, default=None, help="Optuna trials")
    parser.add_argument(
        "--tuning_objective",
        type=str,
        default=None,
        choices=["val_r2"],
        help="Regression Optuna objective for Step4_1 (Step4_2 always tunes balanced_accuracy)",
    )
    parser.add_argument(
        "--tuning_cv_folds",
        type=int,
        default=None,
        help="Number of CV folds for Optuna tuning on the train+val subset",
    )
    parser.add_argument(
        "--finetune_last_layers",
        type=int,
        default=None,
        help="Override chi_training.step4_1_regression.finetune_last_layers (valid range: 0..num_layers)",
    )
    args = parser.parse_args()
    main(args)
