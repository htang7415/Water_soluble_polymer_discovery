import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.data_utils import ensure_dir, get_d5_path, split_train_val_test_by_group
from src.eval_utils import regression_metrics
from src.models.dit import DiT, DiTConfig
from src.models.heads import RegressionHead
from src.optuna_utils import create_study, save_study, trial_params_to_json
from src.plot_utils import parity_plot, save_loss_plot
from src.tokenizer import SmilesTokenizer
from src.train_utils import EarlyStopping, build_optimizer, maybe_compile
from src.log_utils import start_log, end_log


def build_dataloaders(df, tokenizer, batch_size, seed, shuffle=True):
    smiles = df["SMILES"].tolist()
    y = df["chi"].astype(float).values
    ids = [tokenizer.encode(s) for s in smiles]
    ids = torch.tensor(ids, dtype=torch.long)
    attn = (ids != tokenizer.pad_id).long()
    y = torch.tensor(y, dtype=torch.float)
    dataset = TensorDataset(ids, attn, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def eval_loader(model, head, loader, device):
    model.eval()
    head.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for ids, attn, y in loader:
            ids = ids.to(device)
            attn = attn.to(device)
            y = y.to(device)
            timesteps = torch.zeros(ids.size(0), device=device).long()
            hidden = model.forward_hidden(ids, timesteps, attn)
            pooled = hidden[:, 0, :]
            pred = head(pooled)
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
    return y_true, y_pred


def train_one(model, head, train_loader, val_loader, loss_fn, optimizer, device, max_epochs, patience, trial=None):
    best_metric = -1e9
    best_state = None
    train_losses = []
    val_losses = []
    stopper = EarlyStopping(patience=patience, mode="max")

    for epoch in range(max_epochs):
        model.train()
        head.train()
        total_loss = 0.0
        count = 0
        for ids, attn, y in train_loader:
            ids = ids.to(device)
            attn = attn.to(device)
            y = y.to(device)
            timesteps = torch.zeros(ids.size(0), device=device).long()
            hidden = model.forward_hidden(ids, timesteps, attn)
            pooled = hidden[:, 0, :]
            pred = head(pooled)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += 1
        train_losses.append(total_loss / max(1, count))

        y_true, y_pred = eval_loader(model, head, val_loader, device)
        metrics = regression_metrics(y_true, y_pred)
        val_metric = metrics["r2"]
        val_losses.append(metrics["mae"])

        if val_metric > best_metric:
            best_metric = val_metric
            best_state = {
                "model": {k: v.cpu() for k, v in model.state_dict().items()},
                "head": {k: v.cpu() for k, v in head.state_dict().items()},
            }
        if trial is not None:
            trial.report(-val_metric, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if stopper.step(val_metric):
            break
    return best_metric, best_state, train_losses, val_losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    results_dir = Path(cfg["paths"]["results_dir"]) / "step2_cosmosac_proxy_D5"
    metrics_dir = results_dir / "metrics"
    figures_dir = results_dir / "figures"
    ensure_dir(results_dir)
    ensure_dir(metrics_dir)
    ensure_dir(figures_dir)
    log_path = results_dir / "log.txt"
    start_time = start_log(log_path, "step2_cosmosac_proxy_D5", args.config, device=args.device)

    tokenizer = SmilesTokenizer.from_json(str(Path(cfg["paths"]["results_dir"]) / "step0_tokenizer_split" / "tokenizer.json"))
    d5_path = get_d5_path(args.config, cfg["paths"]["data_dir"])
    df = pd.read_csv(d5_path)

    # split by polymer (SMILES)
    groups = df["SMILES"].astype(str).tolist()
    train_g, val_g, test_g = split_train_val_test_by_group(groups, 0.8, 0.1, 0.1, cfg.get("seed", 42))
    df_train = df[df["SMILES"].isin(train_g)].reset_index(drop=True)
    df_val = df[df["SMILES"].isin(val_g)].reset_index(drop=True)
    df_test = df[df["SMILES"].isin(test_g)].reset_index(drop=True)

    device = args.device

    best_global = {"score": -1e9, "state": None, "train_losses": None, "val_losses": None}

    def objective(trial: optuna.Trial):
        lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        head_dim = trial.suggest_int("head_dim", 64, 512, step=64)
        head_layers = trial.suggest_int("head_layers", 1, 4)
        loss_name = trial.suggest_categorical("loss", ["mse", "huber"])
        freeze_mode = trial.suggest_int("freeze_mode", 0, cfg["backbone"]["num_layers"])

        config = DiTConfig(
            vocab_size=len(tokenizer.vocab),
            hidden_size=cfg["backbone"]["hidden_size"],
            num_layers=cfg["backbone"]["num_layers"],
            num_heads=cfg["backbone"]["num_heads"],
            ffn_hidden_size=cfg["backbone"]["ffn_hidden_size"],
            dropout=dropout,
            max_position_embeddings=cfg["backbone"]["max_position_embeddings"],
            num_steps=cfg["diffusion"]["num_steps"],
        )
        model = DiT(config).to(device)
        # load pretrained backbone
        ckpt = torch.load(Path(cfg["paths"]["results_dir"]) / "step1_dit_pretrain_D8" / "checkpoints" / "model_best.pt", map_location="cpu")
        model.load_state_dict(ckpt["model_state"], strict=False)
        model.set_freeze_mode(freeze_mode)
        model = maybe_compile(model, cfg.get("optimization", {}))

        head = RegressionHead(config.hidden_size, [head_dim] * head_layers, dropout).to(device)
        head = maybe_compile(head, cfg.get("optimization", {}))

        train_loader = build_dataloaders(df_train, tokenizer, cfg["training_head"]["batch_size"], cfg.get("seed", 42), shuffle=True)
        val_loader = build_dataloaders(df_val, tokenizer, cfg["training_head"]["batch_size"], cfg.get("seed", 42), shuffle=False)

        loss_fn = torch.nn.MSELoss() if loss_name == "mse" else torch.nn.SmoothL1Loss()
        optimizer = build_optimizer(list(model.parameters()) + list(head.parameters()), lr, weight_decay)

        best_r2, best_state, train_losses, val_losses = train_one(
            model,
            head,
            train_loader,
            val_loader,
            loss_fn,
            optimizer,
            device,
            cfg["optuna"]["max_epochs"],
            cfg["optuna"]["patience"],
            trial=trial,
        )
        if best_r2 > best_global["score"]:
            best_global["score"] = best_r2
            best_global["state"] = best_state
            best_global["train_losses"] = train_losses
            best_global["val_losses"] = val_losses
        return -best_r2

    study = create_study("step2_cosmosac", direction="min")
    study.optimize(objective, n_trials=cfg["optuna"]["n_trials"])

    best_trial = study.best_trial
    best_params = trial_params_to_json(best_trial)
    with open(results_dir / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    save_study(study, str(metrics_dir / "optuna_study.csv"))

    # rebuild best model
    best_dropout = best_params["dropout"]
    config = DiTConfig(
        vocab_size=len(tokenizer.vocab),
        hidden_size=cfg["backbone"]["hidden_size"],
        num_layers=cfg["backbone"]["num_layers"],
        num_heads=cfg["backbone"]["num_heads"],
        ffn_hidden_size=cfg["backbone"]["ffn_hidden_size"],
        dropout=best_dropout,
        max_position_embeddings=cfg["backbone"]["max_position_embeddings"],
        num_steps=cfg["diffusion"]["num_steps"],
    )
    model = DiT(config).to(device)
    ckpt = torch.load(Path(cfg["paths"]["results_dir"]) / "step1_dit_pretrain_D8" / "checkpoints" / "model_best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.set_freeze_mode(best_params["freeze_mode"])
    model = maybe_compile(model, cfg.get("optimization", {}))
    head = RegressionHead(config.hidden_size, [best_params["head_dim"]] * best_params["head_layers"], best_dropout).to(device)
    head = maybe_compile(head, cfg.get("optimization", {}))

    # load best state from trial attrs
    if best_global["state"]:
        model.load_state_dict(best_global["state"]["model"], strict=False)
        head.load_state_dict(best_global["state"]["head"], strict=False)

    train_loader = build_dataloaders(df_train, tokenizer, cfg["training_head"]["batch_size"], cfg.get("seed", 42), shuffle=True)
    val_loader = build_dataloaders(df_val, tokenizer, cfg["training_head"]["batch_size"], cfg.get("seed", 42), shuffle=False)
    test_loader = build_dataloaders(df_test, tokenizer, cfg["training_head"]["batch_size"], cfg.get("seed", 42), shuffle=False)

    y_true_tr, y_pred_tr = eval_loader(model, head, train_loader, device)
    y_true_va, y_pred_va = eval_loader(model, head, val_loader, device)
    y_true_te, y_pred_te = eval_loader(model, head, test_loader, device)

    metrics = {
        "train": regression_metrics(y_true_tr, y_pred_tr),
        "val": regression_metrics(y_true_va, y_pred_va),
        "test": regression_metrics(y_true_te, y_pred_te),
    }

    # save metrics
    with open(metrics_dir / "metrics.csv", "w") as f:
        f.write("split,r2,mae,rmse,spearman\n")
        for split, m in metrics.items():
            f.write(f"{split},{m['r2']},{m['mae']},{m['rmse']},{m['spearman']}\n")

    # predictions
    pred_rows = []
    for split, y_t, y_p in [
        ("train", y_true_tr, y_pred_tr),
        ("val", y_true_va, y_pred_va),
        ("test", y_true_te, y_pred_te),
    ]:
        for yt, yp in zip(y_t, y_p):
            pred_rows.append({"split": split, "y_true": yt, "y_pred": yp})
    pd.DataFrame(pred_rows).to_csv(metrics_dir / "predictions.csv", index=False)

    # plots
    parity_plot(y_true_tr, y_pred_tr, str(figures_dir / "fig_parity_train.png"), metrics=metrics["train"])
    parity_plot(y_true_va, y_pred_va, str(figures_dir / "fig_parity_val.png"), metrics=metrics["val"])
    parity_plot(y_true_te, y_pred_te, str(figures_dir / "fig_parity_test.png"), metrics=metrics["test"])

    train_losses = best_global.get("train_losses", [])
    val_losses = best_global.get("val_losses", [])
    if train_losses:
        save_loss_plot(train_losses, val_losses, str(figures_dir / "fig_loss.png"))

    # save model
    torch.save({"model_state": model.state_dict(), "head_state": head.state_dict()}, results_dir / "model_best.pt")

    # run info
    run_info = {"seed": cfg.get("seed", 42), "timestamp": datetime.utcnow().isoformat(), "device": device}
    with open(results_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)
    end_log(log_path, start_time, status="completed")


if __name__ == "__main__":
    main()
