"""
Hyperparameter optimization using Optuna.

Searches for optimal hyperparameters by training and evaluating models
on a subset of the data with different hyperparameter configurations.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import optuna
from optuna.pruners import MedianPruner, HyperbandPruner
import torch

from src.utils.config import load_config, save_config, update_config
from src.utils.logging_utils import setup_logging, create_run_directory
from src.utils.seed_utils import set_seed


def objective(trial: optuna.Trial, base_config: Any, device: str, logger: logging.Logger) -> float:
    """
    Objective function for Optuna optimization.

    Args:
        trial: Optuna trial object
        base_config: Base configuration
        device: Device to use (cuda/cpu)
        logger: Logger instance

    Returns:
        Objective value (higher is better): roc_auc - alpha * mae_exp
    """
    # Import here to avoid circular dependencies
    from src.data.featurization import load_or_compute_features
    from src.data.datasets import DFTChiDataset, ExpChiDataset, SolubilityDataset
    from src.data.splits import create_dft_splits, create_solubility_splits, create_exp_chi_cv_splits
    from src.models.multitask_model import MultiTaskChiSolubilityModel
    from src.training.losses import multitask_loss
    from src.evaluation.metrics import compute_regression_metrics, compute_classification_metrics
    import pandas as pd
    from torch.utils.data import DataLoader, Subset

    # Suggest hyperparameters
    hparams = {
        "model.encoder_latent_dim": trial.suggest_categorical(
            "encoder_latent_dim", [64, 128, 256]
        ),
        "model.encoder_dropout": trial.suggest_float("encoder_dropout", 0.1, 0.4),
        "model.chi_head_dropout": trial.suggest_float("chi_head_dropout", 0.05, 0.3),
        "model.sol_head_dropout": trial.suggest_float("sol_head_dropout", 0.05, 0.3),
        "training.lr_finetune": trial.suggest_float("lr_finetune", 1e-4, 1e-2, log=True),
        "training.weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "loss_weights.lambda_exp": trial.suggest_float("lambda_exp", 1.0, 5.0),
        "loss_weights.lambda_sol": trial.suggest_float("lambda_sol", 0.5, 2.0),
        "solubility.class_weight_pos": trial.suggest_float("class_weight_pos", 1.0, 5.0),
    }

    logger.info(f"Trial {trial.number}: Testing hyperparameters: {hparams}")

    # Update config
    config = update_config(base_config, hparams)

    # Set seed
    set_seed(config.seed)

    try:
        # Load and featurize data
        logger.info("Loading data...")
        dft_df = pd.read_csv(config.paths.dft_chi_csv)
        exp_df = pd.read_csv(config.paths.exp_chi_csv)
        sol_df = pd.read_csv(config.paths.solubility_csv)

        # Featurize (use cached if available)
        dft_features, dft_smiles_to_idx = load_or_compute_features(
            dft_df, config, cache_prefix="dft", logger=logger
        )
        exp_features, exp_smiles_to_idx = load_or_compute_features(
            exp_df, config, cache_prefix="exp", logger=logger
        )
        sol_features, sol_smiles_to_idx = load_or_compute_features(
            sol_df, config, cache_prefix="sol", logger=logger
        )

        # Create datasets
        dft_dataset = DFTChiDataset(dft_df, dft_features, dft_smiles_to_idx)
        exp_dataset = ExpChiDataset(exp_df, exp_features, exp_smiles_to_idx)
        sol_dataset = SolubilityDataset(sol_df, sol_features, sol_smiles_to_idx)

        # Create splits - NOTE: These functions return DataFrames, not index dicts
        train_dft_df, val_dft_df, test_dft_df = create_dft_splits(dft_df, config)
        train_sol_df, val_sol_df, test_sol_df = create_solubility_splits(sol_df, config)

        # Use first CV fold for exp chi - returns list of (train_indices, val_indices)
        exp_cv_splits = create_exp_chi_cv_splits(exp_df, config)
        exp_train_idx, exp_val_idx = exp_cv_splits[0]

        # Get indices from DataFrames for subsetting
        train_dft_indices = train_dft_df.index.tolist()
        val_sol_indices = val_sol_df.index.tolist()

        # Create DataLoaders (smaller batches for HPO)
        train_dft_loader = DataLoader(
            Subset(dft_dataset, train_dft_indices),
            batch_size=min(128, config.training.batch_size_dft),
            shuffle=True,
        )
        val_exp_loader = DataLoader(
            Subset(exp_dataset, exp_val_idx),
            batch_size=config.training.batch_size_exp,
            shuffle=False,
        )
        val_sol_loader = DataLoader(
            Subset(sol_dataset, val_sol_indices),
            batch_size=config.training.batch_size_sol,
            shuffle=False,
        )

        # Build model
        input_dim = dft_features.shape[1]
        model = MultiTaskChiSolubilityModel(config, input_dim).to(device)

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.lr_finetune,
            weight_decay=config.training.weight_decay,
        )

        # Quick training (reduced epochs for HPO)
        n_epochs = 20  # Much shorter than full training
        best_val_mae = float("inf")

        for epoch in range(n_epochs):
            # Training
            model.train()
            for batch in train_dft_loader:
                x, chi_dft, temp, _ = batch
                x, chi_dft, temp = x.to(device), chi_dft.to(device), temp.to(device)

                optimizer.zero_grad()

                # Forward pass
                outputs = model(x, temp, predict_solubility=False)

                # Compute loss - pass the full outputs dict, not A/B tensors
                loss = multitask_loss(
                    outputs=outputs,
                    temperature=temp,
                    chi_dft_true=chi_dft,
                    chi_exp_true=None,
                    temp_exp=None,
                    solubility_true=None,
                    config=config,
                )

                loss.backward()
                optimizer.step()

            # Validation on exp chi (primary metric)
            if (epoch + 1) % 5 == 0 or epoch == n_epochs - 1:
                model.eval()
                exp_preds, exp_true = [], []

                with torch.no_grad():
                    for batch in val_exp_loader:
                        x, chi_exp, temp, _ = batch
                        x, temp = x.to(device), temp.to(device)

                        outputs = model(x, temp, predict_solubility=False)
                        # Manually compute chi_pred from A and B
                        A, B = outputs["A"], outputs["B"]
                        chi_pred = (A / temp + B).squeeze()

                        exp_preds.append(chi_pred.cpu().numpy())
                        exp_true.append(chi_exp.numpy())

                exp_preds = np.concatenate(exp_preds)
                exp_true = np.concatenate(exp_true)

                exp_metrics = compute_regression_metrics(exp_true, exp_preds)
                val_mae = exp_metrics["mae"]  # Lowercase key

                # Report to Optuna for pruning
                trial.report(val_mae, epoch)

                if trial.should_prune():
                    raise optuna.TrialPruned()

                if val_mae < best_val_mae:
                    best_val_mae = val_mae

        # Evaluate solubility for objective
        model.eval()
        sol_preds, sol_true = [], []

        with torch.no_grad():
            for batch in val_sol_loader:
                x, soluble, _ = batch
                x = x.to(device)

                # Use T_ref for solubility prediction (batch-sized tensor)
                T_ref = torch.full((x.size(0),), config.model.T_ref_K, device=device)
                outputs = model(x, T_ref, predict_solubility=True)
                p_soluble = outputs["p_soluble"].squeeze()

                sol_preds.append(p_soluble.cpu().numpy())
                sol_true.append(soluble.numpy())

        sol_preds = np.concatenate(sol_preds)
        sol_true = np.concatenate(sol_true)

        sol_metrics = compute_classification_metrics(
            sol_true, sol_preds, threshold=config.solubility.decision_threshold
        )
        roc_auc = sol_metrics["roc_auc"]  # Lowercase key

        # Objective: maximize ROC-AUC, minimize MAE
        alpha = config.hparam_search.get("alpha_mae_weight", 0.1)
        objective_value = roc_auc - alpha * best_val_mae

        logger.info(
            f"Trial {trial.number} completed: "
            f"exp_chi_MAE={best_val_mae:.4f}, "
            f"sol_ROC_AUC={roc_auc:.4f}, "
            f"objective={objective_value:.4f}"
        )

        return objective_value

    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        raise


def main():
    """Main hyperparameter optimization routine."""
    parser = argparse.ArgumentParser(description="Hyperparameter optimization with Optuna")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of trials")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Create output directory
    run_dir = create_run_directory(
        Path(config.paths.results_dir),
        "hparam_search",
    )

    # Setup logging
    logger = setup_logging(
        log_dir=run_dir,
        console_level=config.logging.get("console_level", "INFO"),
        file_level=config.logging.get("file_level", "DEBUG"),
    )

    logger.info("=" * 50)
    logger.info("Hyperparameter Optimization with Optuna")
    logger.info("=" * 50)
    logger.info(f"Config: {args.config}")
    logger.info(f"Number of trials: {args.n_trials}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Output directory: {run_dir}")
    logger.info("")

    # Save base config
    save_config(config, run_dir / "base_config.yaml")

    # Create Optuna study
    pruner = None
    if config.hparam_search.get("use_pruning", True):
        pruner_type = config.hparam_search.get("pruner", "median")
        if pruner_type == "median":
            pruner = MedianPruner()
        elif pruner_type == "hyperband":
            pruner = HyperbandPruner()

    study = optuna.create_study(
        direction="maximize",  # Maximize objective
        pruner=pruner,
        study_name="polymer_chi_solubility_hpo",
    )

    # Optimize
    logger.info("Starting optimization...")
    study.optimize(
        lambda trial: objective(trial, config, args.device, logger),
        n_trials=args.n_trials,
        timeout=args.timeout,
    )

    # Results
    logger.info("")
    logger.info("=" * 50)
    logger.info("Optimization Complete!")
    logger.info("=" * 50)
    logger.info(f"Number of finished trials: {len(study.trials)}")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best objective value: {study.best_value:.4f}")
    logger.info("Best hyperparameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")

    # Save results
    trials_df = study.trials_dataframe()
    trials_df.to_csv(run_dir / "trials.csv", index=False)
    logger.info(f"Saved trials to: {run_dir / 'trials.csv'}")

    # Save best config - properly map parameter names back to config structure
    best_hparams = {}
    for k, v in study.best_params.items():
        if "encoder" in k or "head" in k:
            best_hparams[f"model.{k}"] = v
        elif "lr" in k or "weight" in k:
            best_hparams[f"training.{k}"] = v
        elif "lambda" in k:
            best_hparams[f"loss_weights.{k}"] = v
        elif "class_weight" in k:
            best_hparams[f"solubility.{k}"] = v
        else:
            best_hparams[k] = v
    best_config = update_config(config, best_hparams)
    save_config(best_config, run_dir / "best_config.yaml")
    logger.info(f"Saved best config to: {run_dir / 'best_config.yaml'}")

    # Save summary
    summary = {
        "n_trials": len(study.trials),
        "best_trial": study.best_trial.number,
        "best_objective": float(study.best_value),
        "best_params": study.best_params,
    }
    with open(run_dir / "hpo_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Hyperparameter optimization complete!")


if __name__ == "__main__":
    main()
