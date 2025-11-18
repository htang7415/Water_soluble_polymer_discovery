"""
K-fold cross-validation for experimental chi prediction.

Performs SMILES-level k-fold CV to evaluate model performance on experimental chi data.
Each fold trains a multi-task model and evaluates on held-out polymers.

Usage:
    python -m src.training.cv_exp_chi --config configs/config.yaml
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.datasets import ExpChiDataset, collate_exp_chi
from src.data.featurization import PolymerFeaturizer
from src.data.splits import create_exp_chi_cv_splits
from src.models.multitask_model import MultiTaskChiSolubilityModel
from src.training.losses import chi_exp_loss, compute_metrics_regression
from src.utils.config import Config, load_config, save_config
from src.utils.logging_utils import create_run_directory, setup_logging, MetricsLogger
from src.utils.seed_utils import set_seed, worker_init_fn


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="K-fold CV for experimental chi",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). If not specified, uses config value.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results. If not specified, uses timestamped directory.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Maximum epochs per fold. If not specified, uses config value.",
    )
    return parser.parse_args()


def train_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config,
    device: torch.device,
    logger,
    max_epochs: int,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Train model for one fold.

    Args:
        model: Model to train
        train_loader: Training dataloader
        val_loader: Validation dataloader
        config: Configuration
        device: Device to use
        logger: Logger instance
        max_epochs: Maximum number of epochs

    Returns:
        best_val_metrics: Dictionary of best validation metrics
        best_val_preds: Best validation predictions
        best_val_targets: Best validation targets
    """
    # Setup optimizer
    optimizer_name = config.training.optimizer.lower()
    lr = config.training.lr_finetune
    weight_decay = config.training.weight_decay

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Setup scheduler
    scheduler = None
    if config.training.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.training.scheduler_factor,
            patience=config.training.scheduler_patience,
            verbose=False,
        )

    # Training loop
    best_val_loss = float('inf')
    best_val_metrics = None
    best_val_preds = None
    best_val_targets = None
    patience_counter = 0
    early_stopping_patience = config.training.early_stopping_patience

    for epoch in range(1, max_epochs + 1):
        # Train
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            x = batch["x"].to(device)
            chi_true = batch["chi_exp"].to(device)
            temp = batch["temperature"].to(device)

            # Forward pass
            outputs = model(x, temperature=temp)
            A, B = outputs["A"], outputs["B"]

            # Compute loss
            loss = chi_exp_loss(A, B, chi_true, temp)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if config.training.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.training.grad_clip_norm
                )

            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validate
        model.eval()
        val_preds, val_targets = [], []
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                chi_true = batch["chi_exp"].to(device)
                temp = batch["temperature"].to(device)

                outputs = model(x, temperature=temp)
                A, B = outputs["A"], outputs["B"]
                chi_pred = outputs["chi"]

                loss = chi_exp_loss(A, B, chi_true, temp)
                val_loss += loss.item()

                val_preds.append(chi_pred.cpu())
                val_targets.append(chi_true.cpu())

        avg_val_loss = val_loss / len(val_loader)
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)

        val_metrics = compute_metrics_regression(val_preds, val_targets)
        val_metrics["loss"] = avg_val_loss

        # Update scheduler
        if scheduler is not None:
            scheduler.step(avg_val_loss)

        # Check for best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_metrics = val_metrics
            best_val_preds = val_preds.numpy()
            best_val_targets = val_targets.numpy()
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if config.training.early_stopping and patience_counter >= early_stopping_patience:
            logger.debug(f"  Early stopping at epoch {epoch}")
            break

    return best_val_metrics, best_val_preds, best_val_targets


def main():
    """Main cross-validation function."""
    # Parse arguments
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Override device if specified
    if args.device is not None:
        config.training.device = args.device

    # Set device
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")

    # Set random seed
    set_seed(config.seed, deterministic=False)

    # Create run directory
    if args.output_dir is not None:
        run_dir = Path(args.output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = create_run_directory(
            Path(config.paths.results_dir),
            "cv_exp_chi",
        )

    # Setup logging
    logger = setup_logging(
        log_dir=run_dir,
        log_file="cv.log",
        console_level=config.logging.console_level,
        file_level=config.logging.file_level,
    )

    logger.info("=" * 80)
    logger.info("K-fold Cross-Validation for Experimental Chi")
    logger.info("=" * 80)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Random seed: {config.seed}")

    # Save config
    save_config(config, run_dir / "config.yaml")

    # Load experimental chi data
    logger.info("=" * 80)
    logger.info("Loading experimental chi data")
    logger.info("=" * 80)

    exp_csv_path = Path(config.paths.exp_chi_csv)
    logger.info(f"Loading from: {exp_csv_path}")

    df_exp = pd.read_csv(exp_csv_path)
    logger.info(f"Loaded {len(df_exp)} measurements from {df_exp['SMILES'].nunique()} unique polymers")

    # Featurize
    logger.info("Featurizing polymers...")
    unique_smiles = df_exp["SMILES"].unique().tolist()
    featurizer = PolymerFeaturizer(config)
    features, smiles_to_idx = featurizer.featurize(unique_smiles)
    feature_dim = featurizer.get_feature_dim()
    logger.info(f"Feature dimension: {feature_dim}")

    # Create full dataset
    full_dataset = ExpChiDataset(df_exp, features, smiles_to_idx)

    # Create CV splits
    logger.info("=" * 80)
    logger.info("Creating CV splits")
    logger.info("=" * 80)

    cv_splits = create_exp_chi_cv_splits(df_exp, config)
    k_folds = len(cv_splits)

    # Set max epochs
    max_epochs = args.max_epochs if args.max_epochs is not None else config.training.num_epochs_finetune
    logger.info(f"Maximum epochs per fold: {max_epochs}")

    # Run CV
    logger.info("=" * 80)
    logger.info(f"Running {k_folds}-fold cross-validation")
    logger.info("=" * 80)

    fold_results = []
    all_fold_preds = []
    all_fold_targets = []

    for fold_idx, (train_indices, val_indices) in enumerate(cv_splits):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Fold {fold_idx + 1}/{k_folds}")
        logger.info(f"{'=' * 80}")

        # Create fold datasets
        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)

        # Create dataloaders
        train_loader = DataLoader(
            train_subset,
            batch_size=config.training.batch_size_exp,
            shuffle=True,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
            collate_fn=collate_exp_chi,
            worker_init_fn=lambda wid: worker_init_fn(wid, config.seed + fold_idx),
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=config.training.batch_size_exp,
            shuffle=False,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
            collate_fn=collate_exp_chi,
        )

        logger.info(f"Train: {len(train_subset)} samples, Val: {len(val_subset)} samples")

        # Build model
        model = MultiTaskChiSolubilityModel(feature_dim, config)
        model = model.to(device)

        # Train fold
        logger.info("Training fold...")
        best_val_metrics, best_val_preds, best_val_targets = train_fold(
            model, train_loader, val_loader, config, device, logger, max_epochs
        )

        # Log fold results
        logger.info(
            f"Fold {fold_idx + 1} - Best Val MAE: {best_val_metrics['mae']:.4f}, "
            f"RMSE: {best_val_metrics['rmse']:.4f}, "
            f"R²: {best_val_metrics['r2']:.4f}"
        )

        # Save fold results
        fold_result = {
            "fold": fold_idx + 1,
            "n_train": len(train_subset),
            "n_val": len(val_subset),
            "val_mae": best_val_metrics["mae"],
            "val_rmse": best_val_metrics["rmse"],
            "val_r2": best_val_metrics["r2"],
            "val_loss": best_val_metrics["loss"],
        }
        fold_results.append(fold_result)

        # Save fold predictions
        fold_dir = run_dir / f"fold_{fold_idx + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame({
            "chi_true": best_val_targets,
            "chi_pred": best_val_preds,
            "error": best_val_preds - best_val_targets,
            "abs_error": np.abs(best_val_preds - best_val_targets),
        }).to_csv(fold_dir / "predictions.csv", index=False)

        # Plot parity for this fold
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(best_val_targets, best_val_preds, alpha=0.5, s=20, edgecolors='none')
        min_val = min(best_val_targets.min(), best_val_preds.min())
        max_val = max(best_val_targets.max(), best_val_preds.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect prediction')
        ax.set_xlabel("True Chi (Experimental)", fontsize=12)
        ax.set_ylabel("Predicted Chi", fontsize=12)
        ax.set_title(f"Fold {fold_idx + 1} - Experimental Chi (MAE={best_val_metrics['mae']:.4f})", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.savefig(fold_dir / "parity_plot.png", dpi=config.plotting.dpi, bbox_inches='tight')
        plt.close()

        # Accumulate predictions for overall plot
        all_fold_preds.append(best_val_preds)
        all_fold_targets.append(best_val_targets)

    # Compute aggregated metrics
    logger.info("\n" + "=" * 80)
    logger.info("Cross-Validation Results Summary")
    logger.info("=" * 80)

    # Save per-fold results
    fold_results_df = pd.DataFrame(fold_results)
    fold_results_df.to_csv(run_dir / "fold_results.csv", index=False)

    # Compute mean and std across folds
    mean_mae = fold_results_df["val_mae"].mean()
    std_mae = fold_results_df["val_mae"].std()
    mean_rmse = fold_results_df["val_rmse"].mean()
    std_rmse = fold_results_df["val_rmse"].std()
    mean_r2 = fold_results_df["val_r2"].mean()
    std_r2 = fold_results_df["val_r2"].std()

    logger.info(f"MAE:  {mean_mae:.4f} ± {std_mae:.4f}")
    logger.info(f"RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
    logger.info(f"R²:   {mean_r2:.4f} ± {std_r2:.4f}")

    # Save aggregated results
    aggregated_results = {
        "k_folds": k_folds,
        "n_total_samples": len(df_exp),
        "n_unique_polymers": df_exp["SMILES"].nunique(),
        "mean_mae": float(mean_mae),
        "std_mae": float(std_mae),
        "mean_rmse": float(mean_rmse),
        "std_rmse": float(std_rmse),
        "mean_r2": float(mean_r2),
        "std_r2": float(std_r2),
        "fold_results": fold_results,
    }

    with open(run_dir / "cv_summary.json", "w") as f:
        json.dump(aggregated_results, f, indent=2)

    logger.info(f"\nCV summary saved to {run_dir / 'cv_summary.json'}")

    # Plot aggregated parity (all folds combined)
    all_fold_preds = np.concatenate(all_fold_preds)
    all_fold_targets = np.concatenate(all_fold_targets)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(all_fold_targets, all_fold_preds, alpha=0.5, s=20, edgecolors='none')
    min_val = min(all_fold_targets.min(), all_fold_preds.min())
    max_val = max(all_fold_targets.max(), all_fold_preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect prediction')
    ax.set_xlabel("True Chi (Experimental)", fontsize=12)
    ax.set_ylabel("Predicted Chi", fontsize=12)
    ax.set_title(
        f"{k_folds}-Fold CV - Experimental Chi\n(MAE={mean_mae:.4f}±{std_mae:.4f})",
        fontsize=14
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(run_dir / "parity_plot_all_folds.png", dpi=config.plotting.dpi, bbox_inches='tight')
    plt.close()

    # Plot CV metrics across folds
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # MAE
    axes[0].bar(fold_results_df["fold"], fold_results_df["val_mae"], alpha=0.7)
    axes[0].axhline(mean_mae, color='r', linestyle='--', label=f'Mean: {mean_mae:.4f}')
    axes[0].set_xlabel("Fold", fontsize=12)
    axes[0].set_ylabel("MAE", fontsize=12)
    axes[0].set_title("MAE across Folds", fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # RMSE
    axes[1].bar(fold_results_df["fold"], fold_results_df["val_rmse"], alpha=0.7)
    axes[1].axhline(mean_rmse, color='r', linestyle='--', label=f'Mean: {mean_rmse:.4f}')
    axes[1].set_xlabel("Fold", fontsize=12)
    axes[1].set_ylabel("RMSE", fontsize=12)
    axes[1].set_title("RMSE across Folds", fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # R²
    axes[2].bar(fold_results_df["fold"], fold_results_df["val_r2"], alpha=0.7)
    axes[2].axhline(mean_r2, color='r', linestyle='--', label=f'Mean: {mean_r2:.4f}')
    axes[2].set_xlabel("Fold", fontsize=12)
    axes[2].set_ylabel("R²", fontsize=12)
    axes[2].set_title("R² across Folds", fontsize=14)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(run_dir / "metrics_across_folds.png", dpi=config.plotting.dpi, bbox_inches='tight')
    plt.close()

    logger.info(f"All results saved to {run_dir}")


if __name__ == "__main__":
    main()
