import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.data_utils import ensure_dir, split_train_val_test_by_group
from src.eval_utils import classification_metrics, regression_metrics
from src.models.dit import DiT, DiTConfig
from src.models.heads import ChiHead, HansenHead, SolubilityHead
from src.optuna_utils import create_study, save_study
from src.plot_utils import parity_plot, pr_curve_plot, roc_curve_plot, save_loss_plot
from src.tokenizer import SmilesTokenizer
from src.train_utils import EarlyStopping, build_optimizer, maybe_compile
from src.log_utils import start_log, end_log


class MultiTaskDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.smiles = df["SMILES"].astype(str).tolist()
        self.sol = df["water_soluble"].astype(int).tolist()
        self.delta_d = df["delta_d"].astype(float).tolist()
        self.delta_p = df["delta_p"].astype(float).tolist()
        self.delta_h = df["delta_h"].astype(float).tolist()
        self.has_hansen = df["has_hansen"].astype(int).tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        ids = self.tokenizer.encode(self.smiles[idx])
        attn = [1 if t != self.tokenizer.pad_id else 0 for t in ids]
        hansen = [self.delta_d[idx], self.delta_p[idx], self.delta_h[idx]]
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(attn, dtype=torch.long),
            torch.tensor(self.sol[idx], dtype=torch.float),
            torch.tensor(hansen, dtype=torch.float),
            torch.tensor(self.has_hansen[idx], dtype=torch.float),
        )


class ChiEvalDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.smiles = df["SMILES"].astype(str).tolist()
        self.temp = df["temperature"].astype(float).tolist()
        self.chi = df["chi"].astype(float).tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        ids = self.tokenizer.encode(self.smiles[idx])
        attn = [1 if t != self.tokenizer.pad_id else 0 for t in ids]
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(attn, dtype=torch.long),
            torch.tensor(self.temp[idx], dtype=torch.float),
            torch.tensor(self.chi[idx], dtype=torch.float),
        )


def forward_backbone(model, ids, attn):
    timesteps = torch.zeros(ids.size(0), device=ids.device).long()
    hidden = model.forward_hidden(ids, timesteps, attn)
    return hidden[:, 0, :]


def eval_chi_mae(model, chi_head, loader, device):
    model.eval()
    chi_head.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for ids, attn, temp, chi in loader:
            ids = ids.to(device)
            attn = attn.to(device)
            temp = temp.to(device)
            chi = chi.to(device)
            emb = forward_backbone(model, ids, attn)
            pred = chi_head(emb, temp)
            y_true.extend(chi.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
    return regression_metrics(y_true, y_pred)["mae"]


def train_one(model, chi_head, sol_head, hansen_head, train_loader, val_loader, device, params, baseline_mae, d6_loader, trial=None):
    optimizer = build_optimizer(list(model.parameters()) + list(sol_head.parameters()) + list(hansen_head.parameters()), params["lr"], params["weight_decay"])
    stopper = EarlyStopping(params["patience"], mode="max")
    best_state = None
    best_auprc = -1e9
    train_losses = []
    val_losses = []

    for epoch in range(params["max_epochs"]):
        model.train()
        sol_head.train()
        hansen_head.train()
        total_loss = 0.0
        count = 0
        for ids, attn, sol, hansen, has_hansen in train_loader:
            ids = ids.to(device)
            attn = attn.to(device)
            sol = sol.to(device)
            hansen = hansen.to(device)
            has_hansen = has_hansen.to(device)
            emb = forward_backbone(model, ids, attn)
            sol_logits = sol_head(emb)
            hansen_pred = hansen_head(emb)

            sol_loss = torch.nn.functional.binary_cross_entropy_with_logits(sol_logits, sol)
            if has_hansen.sum() > 0:
                mask = has_hansen.view(-1, 1)
                hansen_loss = torch.nn.functional.mse_loss(hansen_pred * mask, hansen * mask)
            else:
                hansen_loss = torch.tensor(0.0, device=device)
            loss = params["lambda_sol"] * sol_loss + params["lambda_hansen"] * hansen_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += 1
        train_losses.append(total_loss / max(1, count))

        # validation metrics
        model.eval()
        sol_head.eval()
        hansen_head.eval()
        y_true, y_prob = [], []
        hansen_true, hansen_pred = [], []
        with torch.no_grad():
            for ids, attn, sol, hansen, has_hansen in val_loader:
                ids = ids.to(device)
                attn = attn.to(device)
                sol = sol.to(device)
                hansen = hansen.to(device)
                has_hansen = has_hansen.to(device)
                emb = forward_backbone(model, ids, attn)
                sol_logits = sol_head(emb)
                prob = torch.sigmoid(sol_logits)
                y_true.extend(sol.cpu().numpy().tolist())
                y_prob.extend(prob.cpu().numpy().tolist())
                if has_hansen.sum() > 0:
                    mask = has_hansen.view(-1, 1)
                    hansen_true.extend((hansen * mask).cpu().numpy().tolist())
                    hansen_pred.extend((hansen_head(emb) * mask).cpu().numpy().tolist())

        metrics = classification_metrics(y_true, y_prob)
        val_losses.append(1 - metrics["auprc"])

        if metrics["auprc"] > best_auprc:
            best_auprc = metrics["auprc"]
            best_state = {
                "model": {k: v.cpu() for k, v in model.state_dict().items()},
                "sol": {k: v.cpu() for k, v in sol_head.state_dict().items()},
                "hansen": {k: v.cpu() for k, v in hansen_head.state_dict().items()},
            }

        if trial is not None:
            trial.report(-metrics["auprc"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if stopper.step(metrics["auprc"]):
            break

    # penalty for chi degradation
    model.load_state_dict(best_state["model"], strict=False)
    chi_mae = eval_chi_mae(model, chi_head, d6_loader, device)
    penalty = 0.0
    if chi_mae > 1.05 * baseline_mae:
        penalty = 100.0
    return best_auprc, best_state, train_losses, val_losses, chi_mae, penalty


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    results_dir = Path(cfg["paths"]["results_dir"]) / "step4_solubility_hansen_D1toD4"
    metrics_dir = results_dir / "metrics"
    figures_dir = results_dir / "figures"
    ensure_dir(results_dir)
    ensure_dir(metrics_dir)
    ensure_dir(figures_dir)
    log_path = results_dir / "log.txt"
    start_time = start_log(log_path, "step4_solubility_hansen_D1toD4", args.config, device=args.device)

    tokenizer = SmilesTokenizer.from_json(str(Path(cfg["paths"]["results_dir"]) / "step0_tokenizer_split" / "tokenizer.json"))

    # load D1-D4
    d1 = pd.read_csv(cfg["paths"]["d1_file"])
    d2 = pd.read_csv(cfg["paths"]["d2_file"])
    d3 = pd.read_csv(cfg["paths"]["d3_file"])
    d4 = pd.read_csv(cfg["paths"]["d4_file"])

    for df in [d1, d3]:
        df["has_hansen"] = 1
    for df in [d2, d4]:
        df["delta_d"] = 0.0
        df["delta_p"] = 0.0
        df["delta_h"] = 0.0
        df["has_hansen"] = 0

    df_all = pd.concat([d1, d2, d3, d4], ignore_index=True)
    df_all = df_all.dropna(subset=["SMILES", "water_soluble"])

    # split by polymer (SMILES)
    groups = df_all["SMILES"].astype(str).tolist()
    train_g, val_g, test_g = split_train_val_test_by_group(groups, 0.8, 0.1, 0.1, cfg.get("seed", 42))
    df_train = df_all[df_all["SMILES"].isin(train_g)].reset_index(drop=True)
    df_val = df_all[df_all["SMILES"].isin(val_g)].reset_index(drop=True)
    df_test = df_all[df_all["SMILES"].isin(test_g)].reset_index(drop=True)

    train_loader = DataLoader(MultiTaskDataset(df_train, tokenizer), batch_size=cfg["training_head"]["batch_size"], shuffle=True)
    val_loader = DataLoader(MultiTaskDataset(df_val, tokenizer), batch_size=cfg["training_head"]["batch_size"], shuffle=False)
    test_loader = DataLoader(MultiTaskDataset(df_test, tokenizer), batch_size=cfg["training_head"]["batch_size"], shuffle=False)

    # D6 loader for chi retention
    d6 = pd.read_csv(cfg["paths"]["d6_file"])
    d6_loader = DataLoader(ChiEvalDataset(d6, tokenizer), batch_size=cfg["training_head"]["batch_size"], shuffle=False)

    # baseline chi MAE
    baseline_mae = None
    metrics_path = Path(cfg["paths"]["results_dir"]) / "step3_exp_chi_T_D6" / "metrics" / "metrics.csv"
    if metrics_path.exists():
        mdf = pd.read_csv(metrics_path)
        row = mdf[mdf["metric"] == "mae"]
        if not row.empty:
            baseline_mae = float(row["mean"].values[0])
    if baseline_mae is None:
        baseline_mae = 1.0

    device = args.device

    def best_params_from_step3(cfg):
        path = Path(cfg["paths"]["results_dir"]) / "step3_exp_chi_T_D6" / "best_params.json"
        if path.exists():
            return json.load(open(path))
        return {"head_dim": 128, "head_layers": 2, "dropout": 0.1, "model": "M1"}

    chi_params = best_params_from_step3(cfg)

    best_global = {"objective": float("inf"), "state": None, "train_losses": None, "val_losses": None}

    def objective(trial: optuna.Trial):
        lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        sol_dim = trial.suggest_int("sol_head_dim", 64, 512, step=64)
        sol_layers = trial.suggest_int("sol_head_layers", 1, 3)
        hansen_dim = trial.suggest_int("hansen_head_dim", 64, 512, step=64)
        hansen_layers = trial.suggest_int("hansen_head_layers", 1, 3)
        lambda_sol = trial.suggest_float("lambda_sol", 0.5, 2.0)
        lambda_hansen = trial.suggest_float("lambda_hansen", 0.5, 2.0)
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

        # load backbone + chi head from step3
        chi_ckpt_path = Path(cfg["paths"]["results_dir"]) / "step3_exp_chi_T_D6" / "checkpoints" / "model_best_fold0.pt"
        if not chi_ckpt_path.exists():
            chi_ckpt_path = Path(cfg["paths"]["results_dir"]) / "step3_exp_chi_T_D6" / "checkpoints" / "model_best_full.pt"
        ckpt = torch.load(chi_ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"], strict=False)
        model.set_freeze_mode(freeze_mode)
        model = maybe_compile(model, cfg.get("optimization", {}))

        chi_head = ChiHead(
            config.hidden_size,
            [chi_params["head_dim"]] * chi_params["head_layers"],
            chi_params["dropout"],
            mode=chi_params["model"],
        ).to(device)
        chi_head = maybe_compile(chi_head, cfg.get("optimization", {}))
        chi_head.load_state_dict(ckpt["head_state"], strict=False)
        for p in chi_head.parameters():
            p.requires_grad = False

        sol_head = SolubilityHead(config.hidden_size, [sol_dim] * sol_layers, dropout).to(device)
        hansen_head = HansenHead(config.hidden_size, [hansen_dim] * hansen_layers, dropout).to(device)
        sol_head = maybe_compile(sol_head, cfg.get("optimization", {}))
        hansen_head = maybe_compile(hansen_head, cfg.get("optimization", {}))

        params = {
            "lr": lr,
            "weight_decay": weight_decay,
            "lambda_sol": lambda_sol,
            "lambda_hansen": lambda_hansen,
            "max_epochs": cfg["optuna"]["max_epochs"],
            "patience": cfg["optuna"]["patience"],
        }
        best_auprc, best_state, train_losses, val_losses, chi_mae, penalty = train_one(
            model,
            chi_head,
            sol_head,
            hansen_head,
            train_loader,
            val_loader,
            device,
            params,
            baseline_mae,
            d6_loader,
            trial=trial,
        )
        objective_value = -best_auprc + penalty
        if objective_value < best_global["objective"]:
            best_global["objective"] = objective_value
            best_global["state"] = best_state
            best_global["train_losses"] = train_losses
            best_global["val_losses"] = val_losses
        return objective_value

    study = create_study("step4_solubility", direction="min")
    study.optimize(objective, n_trials=cfg["optuna"]["n_trials"])

    best_trial = study.best_trial
    best_params = best_trial.params
    with open(results_dir / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    save_study(study, str(metrics_dir / "optuna_study.csv"))

    # rebuild best model and evaluate
    config = DiTConfig(
        vocab_size=len(tokenizer.vocab),
        hidden_size=cfg["backbone"]["hidden_size"],
        num_layers=cfg["backbone"]["num_layers"],
        num_heads=cfg["backbone"]["num_heads"],
        ffn_hidden_size=cfg["backbone"]["ffn_hidden_size"],
        dropout=best_params["dropout"],
        max_position_embeddings=cfg["backbone"]["max_position_embeddings"],
        num_steps=cfg["diffusion"]["num_steps"],
    )
    model = DiT(config).to(device)
    chi_ckpt_path = Path(cfg["paths"]["results_dir"]) / "step3_exp_chi_T_D6" / "checkpoints" / "model_best_fold0.pt"
    if not chi_ckpt_path.exists():
        chi_ckpt_path = Path(cfg["paths"]["results_dir"]) / "step3_exp_chi_T_D6" / "checkpoints" / "model_best_full.pt"
    ckpt = torch.load(chi_ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.set_freeze_mode(best_params["freeze_mode"])
    model = maybe_compile(model, cfg.get("optimization", {}))

    chi_head = ChiHead(
        config.hidden_size,
        [chi_params["head_dim"]] * chi_params["head_layers"],
        chi_params["dropout"],
        mode=chi_params["model"],
    ).to(device)
    chi_head = maybe_compile(chi_head, cfg.get("optimization", {}))
    chi_head.load_state_dict(ckpt["head_state"], strict=False)
    for p in chi_head.parameters():
        p.requires_grad = False

    sol_head = SolubilityHead(config.hidden_size, [best_params["sol_head_dim"]] * best_params["sol_head_layers"], best_params["dropout"]).to(device)
    hansen_head = HansenHead(config.hidden_size, [best_params["hansen_head_dim"]] * best_params["hansen_head_layers"], best_params["dropout"]).to(device)
    sol_head = maybe_compile(sol_head, cfg.get("optimization", {}))
    hansen_head = maybe_compile(hansen_head, cfg.get("optimization", {}))

    if best_global["state"]:
        model.load_state_dict(best_global["state"]["model"], strict=False)
        sol_head.load_state_dict(best_global["state"]["sol"], strict=False)
        hansen_head.load_state_dict(best_global["state"]["hansen"], strict=False)

    # evaluation on train/val/test
    def eval_sol_hansen(loader):
        model.eval()
        sol_head.eval()
        hansen_head.eval()
        y_true, y_prob = [], []
        hansen_true, hansen_pred = [], []
        with torch.no_grad():
            for ids, attn, sol, hansen, has_hansen in loader:
                ids = ids.to(device)
                attn = attn.to(device)
                sol = sol.to(device)
                hansen = hansen.to(device)
                has_hansen = has_hansen.to(device)
                emb = forward_backbone(model, ids, attn)
                prob = torch.sigmoid(sol_head(emb))
                y_true.extend(sol.cpu().numpy().tolist())
                y_prob.extend(prob.cpu().numpy().tolist())
                if has_hansen.sum() > 0:
                    mask_idx = has_hansen.bool()
                    hansen_true.extend(hansen[mask_idx].cpu().numpy().tolist())
                    hansen_pred.extend(hansen_head(emb)[mask_idx].cpu().numpy().tolist())
        return y_true, y_prob, hansen_true, hansen_pred

    y_true_tr, y_prob_tr, h_true_tr, h_pred_tr = eval_sol_hansen(train_loader)
    y_true_va, y_prob_va, h_true_va, h_pred_va = eval_sol_hansen(val_loader)
    y_true_te, y_prob_te, h_true_te, h_pred_te = eval_sol_hansen(test_loader)

    metrics = {
        "train": classification_metrics(y_true_tr, y_prob_tr),
        "val": classification_metrics(y_true_va, y_prob_va),
        "test": classification_metrics(y_true_te, y_prob_te),
    }
    h_true_flat = [v for row in h_true_te for v in row] if h_true_te else []
    h_pred_flat = [v for row in h_pred_te for v in row] if h_pred_te else []
    hansen_mae = regression_metrics(h_true_flat, h_pred_flat)["mae"] if h_true_flat else float("nan")
    chi_mae_after = eval_chi_mae(model, chi_head, d6_loader, device)

    # save metrics
    with open(metrics_dir / "metrics.csv", "w") as f:
        f.write("metric,value\n")
        f.write(f"auprc_val,{metrics['val']['auprc']}\n")
        f.write(f"auroc_val,{metrics['val']['auroc']}\n")
        f.write(f"balanced_acc_val,{metrics['val']['balanced_acc']}\n")
        f.write(f"hansen_mae_test,{hansen_mae}\n")
        f.write(f"chi_mae_before,{baseline_mae}\n")
        f.write(f"chi_mae_after,{chi_mae_after}\n")

    # predictions
    pred_rows = []
    for split, y_t, y_p in [
        ("train", y_true_tr, y_prob_tr),
        ("val", y_true_va, y_prob_va),
        ("test", y_true_te, y_prob_te),
    ]:
        for yt, yp in zip(y_t, y_p):
            pred_rows.append({"split": split, "y_true": yt, "y_pred": yp})
    pd.DataFrame(pred_rows).to_csv(metrics_dir / "predictions.csv", index=False)

    # plots
    try:
        from sklearn.metrics import precision_recall_curve, roc_curve

        precision, recall, _ = precision_recall_curve(y_true_val := y_true_va, y_prob_val := y_prob_va)
        pr_curve_plot(precision, recall, str(figures_dir / "fig_solubility_pr.png"))
        fpr, tpr, _ = roc_curve(y_true_val, y_prob_val)
        roc_curve_plot(fpr, tpr, str(figures_dir / "fig_solubility_roc.png"))
    except Exception:
        pass

    if h_true_flat:
        parity_plot(h_true_flat, h_pred_flat, str(figures_dir / "fig_hansen_parity.png"))

    train_losses = best_global.get("train_losses", [])
    val_losses = best_global.get("val_losses", [])
    if train_losses:
        save_loss_plot(train_losses, val_losses, str(figures_dir / "fig_loss.png"))

    torch.save({"model_state": model.state_dict(), "sol_state": sol_head.state_dict(), "hansen_state": hansen_head.state_dict()}, results_dir / "model_best.pt")

    run_info = {"seed": cfg.get("seed", 42), "timestamp": datetime.utcnow().isoformat(), "device": device}
    with open(results_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)
    end_log(log_path, start_time, status="completed")


if __name__ == "__main__":
    main()
