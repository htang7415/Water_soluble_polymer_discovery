#!/usr/bin/env python
"""Step 4: Train physics-guided chi(T, phi) model with Optuna."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve, roc_curve
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chi.data import COEFF_NAMES, SplitConfig, add_split_column, load_chi_dataset, make_split_assignments
from src.chi.embeddings import (
    build_or_load_embedding_cache,
    embedding_table_from_cache,
)
from src.chi.metrics import classification_metrics, hit_metrics, metrics_by_group, regression_metrics
from src.chi.model import PhysicsGuidedChiModel
from src.utils.config import load_config, save_config
from src.utils.model_scales import get_results_dir
from src.utils.reproducibility import save_run_metadata, seed_everything


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
    lambda_bce: float
    hidden_sizes: List[int]
    dropout: float
    tune: bool
    n_trials: int
    tuning_epochs: int
    tuning_patience: int
    timestep_for_embedding: int


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



def _default_chi_config(config: Dict) -> Dict:
    chi_cfg = config.get("chi_training", {})
    defaults = {
        "dataset_path": "Data/chi/_50_polymers_T_phi.csv",
        "split_mode": "polymer",
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "batch_size": 128,
        "num_epochs": 500,
        "patience": 60,
        "learning_rate": 1.0e-3,
        "weight_decay": 1.0e-5,
        "lambda_bce": 0.1,
        "hidden_sizes": [256, 128],
        "dropout": 0.1,
        "tune": True,
        "n_trials": 50,
        "tuning_epochs": 120,
        "tuning_patience": 20,
        "embedding_batch_size": 128,
        "embedding_timestep": int(config.get("training_property", {}).get("default_timestep", 1)),
    }
    out = defaults.copy()
    out.update(chi_cfg)
    return out



def build_train_config(args, config: Dict) -> TrainConfig:
    chi_cfg = _default_chi_config(config)
    split_mode = args.split_mode or chi_cfg["split_mode"]
    tune_cfg = bool(chi_cfg.get("tune", False))
    tune_flag = bool(args.tune or (tune_cfg and not args.no_tune))

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
        lambda_bce=float(chi_cfg["lambda_bce"]),
        hidden_sizes=[int(x) for x in chi_cfg["hidden_sizes"]],
        dropout=float(chi_cfg["dropout"]),
        tune=tune_flag,
        n_trials=int(args.n_trials or chi_cfg.get("n_trials", 50)),
        tuning_epochs=int(chi_cfg.get("tuning_epochs", 80)),
        tuning_patience=int(chi_cfg.get("tuning_patience", 15)),
        timestep_for_embedding=int(chi_cfg.get("embedding_timestep", 1)),
    )



def make_dataloaders(split_df: pd.DataFrame, embedding_table: np.ndarray, batch_size: int) -> Dict[str, DataLoader]:
    loaders = {}
    for split in ["train", "val", "test"]:
        ds = ChiDataset(split_df[split_df["split"] == split], embedding_table)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=False,
        )
    return loaders



def run_epoch(
    model: PhysicsGuidedChiModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    lambda_bce: float,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(mode=train_mode)

    losses: List[float] = []
    losses_mse: List[float] = []
    losses_bce: List[float] = []

    for batch in loader:
        embedding = batch["embedding"].to(device)
        temperature = batch["temperature"].to(device)
        phi = batch["phi"].to(device)
        chi_true = batch["chi"].to(device)
        label = batch["label"].to(device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        out = model.compute_loss(
            embedding=embedding,
            temperature=temperature,
            phi=phi,
            chi_true=chi_true,
            class_label=label,
            lambda_bce=lambda_bce,
        )

        loss = out["loss"]
        if train_mode:
            loss.backward()
            optimizer.step()

        losses.append(float(loss.item()))
        losses_mse.append(float(out["loss_mse"].item()))
        losses_bce.append(float(out["loss_bce"].item()))

    return {
        "loss": float(np.mean(losses)) if losses else np.nan,
        "loss_mse": float(np.mean(losses_mse)) if losses_mse else np.nan,
        "loss_bce": float(np.mean(losses_bce)) if losses_bce else np.nan,
    }


@torch.no_grad()
def predict_split(model: PhysicsGuidedChiModel, loader: DataLoader, device: str) -> Dict[str, np.ndarray]:
    model.eval()
    pred_chi: List[np.ndarray] = []
    true_chi: List[np.ndarray] = []
    logits: List[np.ndarray] = []
    probs: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    coeffs: List[np.ndarray] = []

    for batch in loader:
        embedding = batch["embedding"].to(device)
        temperature = batch["temperature"].to(device)
        phi = batch["phi"].to(device)
        chi_true = batch["chi"].to(device)
        label = batch["label"].to(device)

        out = model(embedding=embedding, temperature=temperature, phi=phi)

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
    embedding_table: np.ndarray,
    train_cfg: TrainConfig,
    device: str,
    hidden_sizes: List[int],
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    lambda_bce: float,
    batch_size: int,
    num_epochs: int,
    patience: int,
) -> Tuple[PhysicsGuidedChiModel, Dict[str, List[float]], Dict[str, Dict[str, np.ndarray]]]:
    dataloaders = make_dataloaders(split_df, embedding_table, batch_size=batch_size)

    model = PhysicsGuidedChiModel(
        embedding_dim=int(embedding_table.shape[1]),
        hidden_sizes=hidden_sizes,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history = {
        "epoch": [],
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
            lambda_bce=lambda_bce,
        )
        val_stats = run_epoch(
            model=model,
            loader=dataloaders["val"],
            optimizer=None,
            device=device,
            lambda_bce=lambda_bce,
        )
        val_pred = predict_split(model, dataloaders["val"], device)
        val_reg = regression_metrics(val_pred["chi_true"], val_pred["chi_pred"])
        val_rmse = float(val_reg["rmse"])

        history["epoch"].append(epoch)
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

    if best_state is not None:
        model.load_state_dict(best_state)

    predictions = {
        split: predict_split(model, loader, device)
        for split, loader in dataloaders.items()
    }
    return model, history, predictions



def tune_hyperparameters(
    split_df: pd.DataFrame,
    embedding_table: np.ndarray,
    train_cfg: TrainConfig,
    device: str,
    tuning_dir: Path,
) -> Dict:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna is required for --tune. Install it with `pip install optuna` or disable tuning."
        ) from exc

    tuning_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        num_layers = trial.suggest_int("num_layers", 1, 3)
        hidden_sizes = [trial.suggest_categorical(f"hidden_{i}", [64, 128, 256, 512]) for i in range(num_layers)]
        dropout = trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3])
        lr = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
        lambda_bce = trial.suggest_categorical("lambda_bce", [0.01, 0.05, 0.1, 0.2, 0.5])
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])

        _, _, preds = train_one_model(
            split_df=split_df,
            embedding_table=embedding_table,
            train_cfg=train_cfg,
            device=device,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            learning_rate=lr,
            weight_decay=wd,
            lambda_bce=lambda_bce,
            batch_size=batch_size,
            num_epochs=train_cfg.tuning_epochs,
            patience=train_cfg.tuning_patience,
        )
        val_reg = regression_metrics(preds["val"]["chi_true"], preds["val"]["chi_pred"])
        trial.set_user_attr("val_r2", float(val_reg["r2"]))
        return float(val_reg["rmse"])

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=train_cfg.n_trials, show_progress_bar=True)

    trials = []
    for t in study.trials:
        row = {
            "trial": t.number,
            "state": str(t.state),
            "value_rmse": t.value,
            "val_r2": t.user_attrs.get("val_r2", np.nan),
        }
        row.update(t.params)
        trials.append(row)

    pd.DataFrame(trials).to_csv(tuning_dir / "optuna_trials.csv", index=False)
    with open(tuning_dir / "optuna_best.json", "w") as f:
        json.dump(
            {
                "best_trial": int(study.best_trial.number),
                "best_value_rmse": float(study.best_value),
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
    ax.set_title("Step3 chi training loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "chi_loss_curve.png", dpi=dpi)
    plt.close(fig)

    # Parity plot (test)
    test = pred_df[pred_df["split"] == "test"].copy()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(
        data=test,
        x="chi",
        y="chi_pred",
        hue="water_soluble",
        palette={1: "#1f77b4", 0: "#d62728"},
        alpha=0.8,
        ax=ax,
    )
    lo = float(min(test["chi"].min(), test["chi_pred"].min()))
    hi = float(max(test["chi"].max(), test["chi_pred"].max()))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2)
    ax.set_xlabel("True χ")
    ax.set_ylabel("Predicted χ")
    ax.set_title("Parity plot (test)")
    fig.tight_layout()
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
    ax.legend()
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



def _save_coefficient_summary(model: PhysicsGuidedChiModel, embedding_cache_df: pd.DataFrame, out_csv: Path, device: str) -> None:
    model.eval()
    emb = np.stack(embedding_cache_df["embedding"].to_list(), axis=0)
    emb_t = torch.tensor(emb, dtype=torch.float32, device=device)

    with torch.no_grad():
        features = model.encoder(emb_t)
        coeff = model.coeff_head(features).cpu().numpy()
        logit = model.class_head(features).squeeze(-1).cpu().numpy()
        prob = 1.0 / (1.0 + np.exp(-logit))

    df = embedding_cache_df[["polymer_id", "Polymer", "SMILES", "water_soluble"]].copy()
    for i, name in enumerate(COEFF_NAMES):
        df[name] = coeff[:, i]
    df["class_logit"] = logit
    df["class_prob"] = prob
    df.to_csv(out_csv, index=False)



def main(args):
    config = load_config(args.config)
    train_cfg = build_train_config(args, config)

    results_dir = Path(get_results_dir(args.model_size, config["paths"]["results_dir"]))
    step_dir = results_dir / "step4_chi_training" / train_cfg.split_mode
    metrics_dir = step_dir / "metrics"
    figures_dir = step_dir / "figures"
    tuning_dir = step_dir / "tuning"
    checkpoint_dir = results_dir / "checkpoints"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    seed_info = seed_everything(train_cfg.seed)
    save_config(config, step_dir / "config_used.yaml")
    save_run_metadata(step_dir, args.config, seed_info)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("Step 4: Physics-guided chi(T, phi) training")
    print(f"Split mode: {train_cfg.split_mode}")
    print(f"Device: {device}")
    print("=" * 70)

    # Load data and create splits.
    chi_cfg = _default_chi_config(config)
    chi_csv = args.dataset_path or chi_cfg["dataset_path"]
    df = load_chi_dataset(chi_csv)

    split_assign = make_split_assignments(
        df,
        SplitConfig(
            split_mode=train_cfg.split_mode,
            train_ratio=train_cfg.train_ratio,
            val_ratio=train_cfg.val_ratio,
            test_ratio=train_cfg.test_ratio,
            seed=train_cfg.seed,
        ),
    )
    split_assign.to_csv(metrics_dir / "split_assignments.csv", index=False)
    split_df = add_split_column(df, split_assign)
    split_df.to_csv(metrics_dir / "chi_dataset_with_split.csv", index=False)

    # Embedding cache.
    polymer_df = split_df[["polymer_id", "Polymer", "SMILES", "water_soluble"]].drop_duplicates("polymer_id")
    emb_cache = build_or_load_embedding_cache(
        polymer_df=polymer_df,
        config=config,
        cache_npz=metrics_dir / "polymer_embeddings.npz",
        model_size=args.model_size,
        checkpoint_path=args.backbone_checkpoint,
        device=device,
        timestep=train_cfg.timestep_for_embedding,
        pooling="mean",
        batch_size=int(chi_cfg["embedding_batch_size"]),
    )
    embedding_table = embedding_table_from_cache(emb_cache)

    # Hyperparameter tuning.
    best_params = None
    if train_cfg.tune:
        print("Running Optuna tuning...")
        best_params = tune_hyperparameters(
            split_df=split_df,
            embedding_table=embedding_table,
            train_cfg=train_cfg,
            device=device,
            tuning_dir=tuning_dir,
        )
        print("Best params:")
        print(best_params)

    if best_params is None:
        chosen = {
            "hidden_sizes": train_cfg.hidden_sizes,
            "dropout": train_cfg.dropout,
            "learning_rate": train_cfg.learning_rate,
            "weight_decay": train_cfg.weight_decay,
            "lambda_bce": train_cfg.lambda_bce,
            "batch_size": train_cfg.batch_size,
        }
    else:
        num_layers = int(best_params["num_layers"])
        chosen = {
            "hidden_sizes": [int(best_params[f"hidden_{i}"]) for i in range(num_layers)],
            "dropout": float(best_params["dropout"]),
            "learning_rate": float(best_params["learning_rate"]),
            "weight_decay": float(best_params["weight_decay"]),
            "lambda_bce": float(best_params["lambda_bce"]),
            "batch_size": int(best_params["batch_size"]),
        }

    with open(metrics_dir / "chosen_hyperparameters.json", "w") as f:
        json.dump(chosen, f, indent=2)

    # Final training.
    model, history, predictions = train_one_model(
        split_df=split_df,
        embedding_table=embedding_table,
        train_cfg=train_cfg,
        device=device,
        hidden_sizes=chosen["hidden_sizes"],
        dropout=chosen["dropout"],
        learning_rate=chosen["learning_rate"],
        weight_decay=chosen["weight_decay"],
        lambda_bce=chosen["lambda_bce"],
        batch_size=chosen["batch_size"],
        num_epochs=train_cfg.num_epochs,
        patience=train_cfg.patience,
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "embedding_dim": int(embedding_table.shape[1]),
        "hidden_sizes": chosen["hidden_sizes"],
        "dropout": chosen["dropout"],
        "lambda_bce": chosen["lambda_bce"],
        "split_mode": train_cfg.split_mode,
        "timestep_for_embedding": train_cfg.timestep_for_embedding,
        "dataset_path": str(chi_csv),
        "config": config,
    }
    torch.save(checkpoint, checkpoint_dir / "chi_physics_best.pt")

    _save_history(history, metrics_dir / "chi_training_history.csv")
    pred_df = _save_prediction_csvs(split_df=split_df, predictions=predictions, out_dir=metrics_dir)
    _collect_metrics(pred_df, out_dir=metrics_dir)
    _save_coefficient_summary(model, emb_cache, metrics_dir / "polymer_coefficients.csv", device=device)
    _make_figures(
        history=history,
        pred_df=pred_df,
        fig_dir=figures_dir,
        dpi=int(config.get("plotting", {}).get("dpi", 300)),
        font_size=int(config.get("plotting", {}).get("font_size", 12)),
    )

    print("Training complete.")
    print(f"Outputs: {step_dir}")
    print(f"Checkpoint: {checkpoint_dir / 'chi_physics_best.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4: train physics-guided chi model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config path")
    parser.add_argument("--model_size", type=str, default=None, choices=["small", "medium", "large", "xl"], help="Step1 model size tag")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to chi dataset CSV")
    parser.add_argument("--split_mode", type=str, default=None, choices=["polymer", "random"], help="Split mode")
    parser.add_argument("--backbone_checkpoint", type=str, default=None, help="Optional backbone checkpoint override")
    parser.add_argument("--tune", action="store_true", help="Enable Optuna tuning")
    parser.add_argument("--no_tune", action="store_true", help="Disable Optuna tuning even if config enables it")
    parser.add_argument("--n_trials", type=int, default=None, help="Optuna trials")
    args = parser.parse_args()
    main(args)
