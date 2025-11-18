"""
Stage 1: DFT chi pretraining script.

Trains encoder + chi head on large DFT dataset to learn polymer representations
and chi(T) = A/T + B prediction.

Usage:
    python -m src.training.train_dft --config configs/config.yaml
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.datasets import DFTChiDataset, collate_dft_chi
from src.data.featurization import PolymerFeaturizer
from src.data.splits import create_dft_splits
from src.models.multitask_model import MultiTaskChiSolubilityModel
from src.training.losses import chi_dft_loss, compute_metrics_regression
from src.utils.config import Config, load_config, save_config
from src.utils.logging_utils import create_run_directory, get_logger, setup_logging, MetricsLogger
from src.utils.seed_utils import set_seed, worker_init_fn


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage 1: DFT chi pretraining",
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
    return parser.parse_args()


def load_and_prepare_data(
    config: Config,
    logger,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Load DFT data, featurize, and create train/val/test dataloaders.

    Args:
        config: Configuration object
        logger: Logger instance

    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        test_loader: Test dataloader
        feature_dim: Feature dimensionality
    """
    logger.info("=" * 80)
    logger.info("Loading and preparing DFT chi data")
    logger.info("=" * 80)

    # Load DFT data
    dft_csv_path = Path(config.paths.dft_chi_csv)
    logger.info(f"Loading DFT data from: {dft_csv_path}")

    df_dft = pd.read_csv(dft_csv_path)
    logger.info(f"Loaded {len(df_dft)} DFT measurements")

    # Get unique SMILES
    unique_smiles = df_dft["SMILES"].unique().tolist()
    logger.info(f"Found {len(unique_smiles)} unique polymers")

    # Featurize
    logger.info("Featurizing polymers...")
    featurizer = PolymerFeaturizer(config)
    features, smiles_to_idx = featurizer.featurize(unique_smiles)
    feature_dim = featurizer.get_feature_dim()
    logger.info(f"Feature dimension: {feature_dim}")

    # Create splits
    logger.info("Creating train/val/test splits...")
    train_df, val_df, test_df = create_dft_splits(df_dft, config)

    # Create datasets
    train_dataset = DFTChiDataset(train_df, features, smiles_to_idx)
    val_dataset = DFTChiDataset(val_df, features, smiles_to_idx)
    test_dataset = DFTChiDataset(test_df, features, smiles_to_idx)

    # Create dataloaders
    batch_size = config.training.batch_size_dft
    num_workers = config.training.num_workers
    pin_memory = config.training.pin_memory

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_dft_chi,
        worker_init_fn=lambda wid: worker_init_fn(wid, config.seed),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_dft_chi,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_dft_chi,
    )

    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, "
                f"Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader, feature_dim


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: Config,
    device: torch.device,
    logger,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: Model to train
        train_loader: Training dataloader
        optimizer: Optimizer
        config: Configuration
        device: Device to use
        logger: Logger instance

    Returns:
        Dictionary of epoch metrics
    """
    model.train()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(train_loader, desc="Training", leave=False)

    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        x = batch["x"].to(device)
        chi_true = batch["chi_dft"].to(device)
        # Use actual temperatures from dataset instead of hard-coded T_ref
        temperature = batch["temperature"].to(device)

        # Forward pass with actual temperatures
        outputs = model(x, temperature=temperature)
        A, B = outputs["A"], outputs["B"]

        # Compute loss with actual temperatures
        loss = chi_dft_loss(A, B, chi_true, temperature=temperature)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if config.training.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.grad_clip_norm
            )

        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        chi_pred = outputs["chi"]
        all_preds.append(chi_pred.detach().cpu())
        all_targets.append(chi_true.detach().cpu())

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # Compute epoch metrics
    avg_loss = total_loss / len(train_loader)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    metrics = compute_metrics_regression(all_preds, all_targets)
    metrics["loss"] = avg_loss

    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    config: Config,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Validate model.

    Args:
        model: Model to validate
        val_loader: Validation dataloader
        config: Configuration
        device: Device to use

    Returns:
        metrics: Dictionary of validation metrics
        predictions: Array of predictions
        targets: Array of true values
    """
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in tqdm(val_loader, desc="Validating", leave=False):
        # Move batch to device
        x = batch["x"].to(device)
        chi_true = batch["chi_dft"].to(device)
        # Use actual temperatures from dataset
        temperature = batch["temperature"].to(device)

        # Forward pass with actual temperatures
        outputs = model(x, temperature=temperature)
        A, B = outputs["A"], outputs["B"]

        # Compute loss with actual temperatures
        loss = chi_dft_loss(A, B, chi_true, temperature=temperature)

        # Track metrics
        total_loss += loss.item()
        chi_pred = outputs["chi"]
        all_preds.append(chi_pred.cpu())
        all_targets.append(chi_true.cpu())

    # Compute metrics
    avg_loss = total_loss / len(val_loader)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    metrics = compute_metrics_regression(all_preds, all_targets)
    metrics["loss"] = avg_loss

    return metrics, all_preds.numpy(), all_targets.numpy()


def plot_parity(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: Path,
    title: str = "DFT Chi Parity Plot",
    dpi: int = 300,
):
    """
    Create parity plot (predicted vs true).

    Args:
        predictions: Predicted values
        targets: True values
        save_path: Path to save figure
        title: Plot title
        dpi: Figure DPI
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot
    ax.scatter(targets, predictions, alpha=0.5, s=20, edgecolors='none')

    # Diagonal line (perfect prediction)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect prediction')

    # Labels and title
    ax.set_xlabel("True Chi", fontsize=12)
    ax.set_ylabel("Predicted Chi", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def save_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: Path,
):
    """
    Save predictions to CSV.

    Args:
        predictions: Predicted values
        targets: True values
        save_path: Path to save CSV
    """
    df = pd.DataFrame({
        "chi_true": targets,
        "chi_pred": predictions,
        "error": predictions - targets,
        "abs_error": np.abs(predictions - targets),
    })
    df.to_csv(save_path, index=False)


def main():
    """Main training function."""
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
            "dft_pretrain",
        )

    # Setup logging
    logger = setup_logging(
        log_dir=run_dir,
        log_file="train.log",
        console_level=config.logging.console_level,
        file_level=config.logging.file_level,
    )

    logger.info("=" * 80)
    logger.info("Stage 1: DFT Chi Pretraining")
    logger.info("=" * 80)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Random seed: {config.seed}")

    # Save config
    save_config(config, run_dir / "config.yaml")
    logger.info(f"Saved configuration to {run_dir / 'config.yaml'}")

    # Load and prepare data
    train_loader, val_loader, test_loader, feature_dim = load_and_prepare_data(config, logger)

    # Build model
    logger.info("=" * 80)
    logger.info("Building model")
    logger.info("=" * 80)

    model = MultiTaskChiSolubilityModel(feature_dim, config)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {n_params:,} trainable parameters")

    # Setup optimizer
    optimizer_name = config.training.optimizer.lower()
    lr = config.training.lr_pretrain
    weight_decay = config.training.weight_decay

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    logger.info(f"Optimizer: {optimizer_name}, lr={lr}, weight_decay={weight_decay}")

    # Setup scheduler
    scheduler = None
    if config.training.use_scheduler:
        scheduler_type = config.training.scheduler_type

        if scheduler_type == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=config.training.scheduler_factor,
                patience=config.training.scheduler_patience,
                verbose=True,
            )
        elif scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.training.scheduler_step_size,
                gamma=config.training.scheduler_factor,
            )
        elif scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.training.num_epochs_pretrain,
            )
        else:
            logger.warning(f"Unknown scheduler type: {scheduler_type}, disabling scheduler")

        if scheduler is not None:
            logger.info(f"Scheduler: {scheduler_type}")

    # Setup metrics logger
    metrics_logger = MetricsLogger(run_dir / "metrics.csv")

    # Training loop
    logger.info("=" * 80)
    logger.info("Starting training")
    logger.info("=" * 80)

    num_epochs = config.training.num_epochs_pretrain
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    early_stopping_patience = config.training.early_stopping_patience

    for epoch in range(1, num_epochs + 1):
        logger.info(f"\nEpoch {epoch}/{num_epochs}")
        logger.info("-" * 80)

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, config, device, logger)

        logger.info(
            f"Train - Loss: {train_metrics['loss']:.4f}, "
            f"MAE: {train_metrics['mae']:.4f}, "
            f"RMSE: {train_metrics['rmse']:.4f}, "
            f"R²: {train_metrics['r2']:.4f}"
        )

        # Validate
        val_metrics, val_preds, val_targets = validate(model, val_loader, config, device)

        logger.info(
            f"Val   - Loss: {val_metrics['loss']:.4f}, "
            f"MAE: {val_metrics['mae']:.4f}, "
            f"RMSE: {val_metrics['rmse']:.4f}, "
            f"R²: {val_metrics['r2']:.4f}"
        )

        # Log metrics
        metrics_logger.log({
            "train_loss": train_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "train_r2": train_metrics["r2"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "lr": optimizer.param_groups[0]["lr"],
        }, step=epoch)

        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

        # Check for best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            checkpoint_path = run_dir / "checkpoints" / "best_model.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["loss"],
                "val_mae": val_metrics["mae"],
                "config": config.to_dict(),
            }, checkpoint_path)

            logger.info(f"Saved best model to {checkpoint_path}")

            # Save best predictions
            save_predictions(
                val_preds,
                val_targets,
                run_dir / "val_predictions_best.csv",
            )

            # Plot parity
            plot_parity(
                val_preds,
                val_targets,
                run_dir / "figures" / "parity_plot_best.png",
                title=f"DFT Chi Parity Plot (Epoch {epoch})",
                dpi=config.plotting.dpi,
            )

        else:
            patience_counter += 1

        # Early stopping
        if config.training.early_stopping and patience_counter >= early_stopping_patience:
            logger.info(
                f"\nEarly stopping triggered after {epoch} epochs "
                f"(best epoch: {best_epoch}, best val loss: {best_val_loss:.4f})"
            )
            break

    # Training complete
    logger.info("=" * 80)
    logger.info("Training complete!")
    logger.info("=" * 80)
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")

    # Load best model and evaluate on test set
    logger.info("\nEvaluating on test set...")
    checkpoint = torch.load(run_dir / "checkpoints" / "best_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics, test_preds, test_targets = validate(model, test_loader, config, device)

    logger.info(
        f"Test  - Loss: {test_metrics['loss']:.4f}, "
        f"MAE: {test_metrics['mae']:.4f}, "
        f"RMSE: {test_metrics['rmse']:.4f}, "
        f"R²: {test_metrics['r2']:.4f}"
    )

    # Save test predictions and plot
    save_predictions(test_preds, test_targets, run_dir / "test_predictions.csv")

    plot_parity(
        test_preds,
        test_targets,
        run_dir / "figures" / "parity_plot_test.png",
        title="DFT Chi Parity Plot (Test Set)",
        dpi=config.plotting.dpi,
    )

    # Save final summary
    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_metrics": test_metrics,
        "run_dir": str(run_dir),
        "model_checkpoint": str(run_dir / "checkpoints" / "best_model.pt"),
    }

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nSummary saved to {run_dir / 'summary.json'}")
    logger.info(f"All results saved to {run_dir}")


if __name__ == "__main__":
    main()
