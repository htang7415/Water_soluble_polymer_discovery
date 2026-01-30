import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import random
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.data_utils import ensure_dir
from src.eval_utils import regression_metrics, trend_error_by_polymer, uncertainty_coverage
from src.models.dit import DiT, DiTConfig
from src.models.heads import ChiHead
from src.optuna_utils import create_study, save_study, trial_logger
from src.plot_utils import parity_plot, save_loss_plot, bar_plot, coverage_plot
from src.tokenizer import SmilesTokenizer
from src.train_utils import EarlyStopping, build_optimizer, maybe_compile, strip_compile_prefix
from src.log_utils import start_log, end_log


class ChiDataset(Dataset):
    def __init__(self, df, tokenizer, cache_tokenization: bool = False):
        self.smiles = df["SMILES"].astype(str).tolist()
        self.temps = df["temperature"].astype(float).tolist()
        self.chi = df["chi"].astype(str).tolist()
        self.polymer = df["Polymer"].astype(str).tolist()
        self.tokenizer = tokenizer
        self.cache_tokenization = cache_tokenization
        self._cache = {}
        if cache_tokenization:
            self._pretokenize()

    def _pretokenize(self):
        for idx, smi in enumerate(self.smiles):
            ids = self.tokenizer.encode(smi)
            attn = [1 if t != self.tokenizer.pad_id else 0 for t in ids]
            self._cache[idx] = (ids, attn)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        if self.cache_tokenization and idx in self._cache:
            ids, attn = self._cache[idx]
        else:
            ids = self.tokenizer.encode(self.smiles[idx])
            attn = [1 if t != self.tokenizer.pad_id else 0 for t in ids]
        chi_val = self.chi[idx]
        temp = self.temps[idx]
        low, high = parse_interval(chi_val)
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(attn, dtype=torch.long),
            torch.tensor(temp, dtype=torch.float),
            torch.tensor(low, dtype=torch.float),
            torch.tensor(high, dtype=torch.float),
            self.polymer[idx],
        )


def parse_interval(value: str):
    try:
        v = float(value)
        return v, v
    except Exception:
        v = value.replace("–", "-")
        if "to" in v:
            parts = [p.strip() for p in v.split("to")]
        elif "-" in v:
            parts = [p.strip() for p in v.split("-")]
        else:
            parts = [v]
        nums = [float(p) for p in parts if p]
        if len(nums) == 1:
            return nums[0], nums[0]
        return min(nums), max(nums)


def _amp_helpers(device: str, use_amp: bool):
    if not use_amp or not device.startswith("cuda"):
        return None, nullcontext
    try:
        from torch.amp import GradScaler, autocast as amp_autocast

        scaler = GradScaler("cuda", enabled=True)

        def autocast_ctx():
            return amp_autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)

        return scaler, autocast_ctx
    except Exception:
        from torch.cuda.amp import GradScaler, autocast as amp_autocast

        scaler = GradScaler(enabled=True)

        def autocast_ctx():
            return amp_autocast(enabled=True, dtype=torch.bfloat16)

        return scaler, autocast_ctx


def _pool_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "cls":
        return hidden[:, 0, :]
    if pooling == "max":
        mask = attention_mask.unsqueeze(-1).bool()
        masked = hidden.masked_fill(~mask, -1e9)
        return masked.max(dim=1).values
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


def interval_loss(pred, low, high, loss_type="mse"):
    # zero loss inside interval; outside use mse or huber
    below = pred < low
    above = pred > high
    target = torch.where(below, low, torch.where(above, high, pred))
    if loss_type == "huber":
        return torch.nn.functional.smooth_l1_loss(pred, target)
    return torch.nn.functional.mse_loss(pred, target)


def forward_batch(model, head, ids, attn, temp, pooling: str, default_timestep: int):
    timesteps = torch.full((ids.size(0),), default_timestep, device=ids.device, dtype=torch.long)
    hidden = model.forward_hidden(ids, timesteps, attn)
    pooled = _pool_hidden(hidden, attn, pooling)
    pred = head(pooled, temp)
    return pred


def eval_dataset(model, head, loader, device, pooling: str, default_timestep: int, autocast_ctx):
    model.eval()
    head.eval()
    y_true, y_pred, temps, polymers = [], [], [], []
    with torch.no_grad():
        for ids, attn, temp, low, high, polymer in loader:
            ids = ids.to(device)
            attn = attn.to(device)
            temp = temp.to(device)
            low = low.to(device)
            high = high.to(device)
            with autocast_ctx():
                pred = forward_batch(model, head, ids, attn, temp, pooling, default_timestep)
            # use midpoint for metrics
            true = (low + high) / 2.0
            y_true.extend(true.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
            temps.extend(temp.cpu().numpy().tolist())
            polymers.extend(list(polymer))
    return y_true, y_pred, temps, polymers


def mc_dropout_predict(model, head, ids, attn, temp, pooling: str, default_timestep: int, k=50):
    preds = []
    model.train()
    head.train()
    with torch.no_grad():
        for _ in range(k):
            pred = forward_batch(model, head, ids, attn, temp, pooling, default_timestep)
            preds.append(pred.cpu().numpy())
    preds = np.stack(preds, axis=0)
    return preds.mean(axis=0), preds.std(axis=0)


def train_fold(
    model,
    head,
    train_loader,
    val_loader,
    device,
    max_epochs,
    patience,
    loss_name,
    lr,
    weight_decay,
    pooling,
    default_timestep,
    autocast_ctx,
    scaler,
    grad_accum_steps,
    trial=None,
):
    optimizer = build_optimizer(list(model.parameters()) + list(head.parameters()), lr=lr, weight_decay=weight_decay)
    stopper = EarlyStopping(patience, mode="min")
    best_val = float("inf")
    best_state = None
    train_losses, val_losses = [], []

    for epoch in range(max_epochs):
        model.train()
        head.train()
        total_loss = 0.0
        count = 0
        accum = 0
        optimizer.zero_grad()
        for ids, attn, temp, low, high, _ in train_loader:
            ids = ids.to(device)
            attn = attn.to(device)
            temp = temp.to(device)
            low = low.to(device)
            high = high.to(device)
            with autocast_ctx():
                pred = forward_batch(model, head, ids, attn, temp, pooling, default_timestep)
                if loss_name == "interval":
                    loss = interval_loss(pred, low, high, loss_type="mse")
                elif loss_name == "huber":
                    loss = torch.nn.functional.smooth_l1_loss(pred, (low + high) / 2.0)
                else:
                    loss = torch.nn.functional.mse_loss(pred, (low + high) / 2.0)
                loss = loss / grad_accum_steps
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum += 1
            if accum % grad_accum_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item() * grad_accum_steps
            count += 1
        if accum % grad_accum_steps != 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        train_losses.append(total_loss / max(1, count))

        # val
        model.eval()
        head.eval()
        total_loss = 0.0
        count = 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for ids, attn, temp, low, high, _ in val_loader:
                ids = ids.to(device)
                attn = attn.to(device)
                temp = temp.to(device)
                low = low.to(device)
                high = high.to(device)
                with autocast_ctx():
                    pred = forward_batch(model, head, ids, attn, temp, pooling, default_timestep)
                    if loss_name == "interval":
                        loss = interval_loss(pred, low, high, loss_type="mse")
                    elif loss_name == "huber":
                        loss = torch.nn.functional.smooth_l1_loss(pred, (low + high) / 2.0)
                    else:
                        loss = torch.nn.functional.mse_loss(pred, (low + high) / 2.0)
                total_loss += loss.item()
                count += 1
                y_true.extend(((low + high) / 2.0).cpu().numpy().tolist())
                y_pred.extend(pred.cpu().numpy().tolist())
        val_loss = total_loss / max(1, count)
        val_losses.append(val_loss)
        val_mae = regression_metrics(y_true, y_pred)["mae"]
        if val_mae < best_val:
            best_val = val_mae
            best_state = {
                "model": {k: v.cpu() for k, v in model.state_dict().items()},
                "head": {k: v.cpu() for k, v in head.state_dict().items()},
            }
        if trial is not None:
            trial.report(best_val, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if stopper.step(val_mae):
            break

    return best_val, best_state, train_losses, val_losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    results_dir = Path(cfg["paths"]["results_dir"]) / "step3_exp_chi_T_D6"
    metrics_dir = results_dir / "metrics"
    figures_dir = results_dir / "figures"
    ensure_dir(results_dir)
    ensure_dir(metrics_dir)
    ensure_dir(figures_dir)
    ensure_dir(results_dir / "checkpoints")
    log_path = results_dir / "log.txt"
    start_time = start_log(log_path, "step3_exp_chi_T_D6", args.config, device=args.device)

    tokenizer = SmilesTokenizer.from_json(str(Path(cfg["paths"]["results_dir"]) / "step0_tokenizer_split" / "tokenizer.json"))

    df = pd.read_csv(cfg["paths"]["d6_file"])
    df = df.dropna(subset=["SMILES", "chi", "temperature"])

    device = args.device
    train_cfg = cfg.get("training_property") or cfg.get("training_head", {})
    head_cfg = cfg.get("property_head", {})
    opt_cfg = cfg.get("optimization", {})
    tuning_cfg = cfg.get("hyperparameter_tuning", {})
    search_space = tuning_cfg.get("search_space", {})
    chi_cfg = cfg.get("chi_head", {})
    chi_model = chi_cfg.get("model", "M1")
    use_amp = bool(opt_cfg.get("use_amp", False)) and device.startswith("cuda")
    grad_accum_steps = max(1, int(opt_cfg.get("gradient_accumulation_steps", 1)))
    pooling = head_cfg.get("pooling", "cls")
    default_timestep = int(train_cfg.get("default_timestep", 0))
    cache_tokenization = bool(train_cfg.get("cache_tokenization", opt_cfg.get("cache_tokenization", False)))
    compile_in_tuning = bool(opt_cfg.get("compile_in_tuning", False))
    num_workers = int(opt_cfg.get("num_workers", 0))
    pin_memory = bool(opt_cfg.get("pin_memory", True))
    prefetch_factor = int(opt_cfg.get("prefetch_factor", 2))
    groups = df["SMILES"].astype(str).tolist()
    unique_groups = sorted(set(groups))

    config = DiTConfig(
        vocab_size=len(tokenizer.vocab),
        hidden_size=cfg["backbone"]["hidden_size"],
        num_layers=cfg["backbone"]["num_layers"],
        num_heads=cfg["backbone"]["num_heads"],
        ffn_hidden_size=cfg["backbone"]["ffn_hidden_size"],
        dropout=cfg["backbone"]["dropout"],
        max_position_embeddings=cfg["backbone"]["max_position_embeddings"],
        num_steps=cfg["diffusion"]["num_steps"],
    )

    def make_loader(dataset, batch_size, shuffle):
        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": num_workers > 0,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor
        return DataLoader(dataset, **loader_kwargs)

    def objective(trial: optuna.Trial):
        # chi head model is chosen via config: M1 (A + B/T) or M2 (A + B/T + C*T)
        model_type = chi_model
        lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        head_dim = trial.suggest_int("head_dim", 64, 512, step=64)
        head_layers = trial.suggest_int("head_layers", 1, 4)
        loss_name = trial.suggest_categorical("loss", ["mse", "huber", "interval"])
        freeze_mode = trial.suggest_int("freeze_mode", 0, cfg["backbone"]["num_layers"])

        fold_mae = []
        fold_rmse = []
        fold_trend = []

        for fold_idx, test_group in enumerate(unique_groups):
            # split groups
            remaining = [g for g in unique_groups if g != test_group]
            rng = random.Random(cfg.get("seed", 42) + fold_idx)
            rng.shuffle(remaining)
            n_val = max(1, int(0.1 * len(remaining)))
            val_groups = set(remaining[:n_val])
            train_groups = set(remaining[n_val:])

            df_train = df[df["SMILES"].isin(train_groups)].reset_index(drop=True)
            df_val = df[df["SMILES"].isin(val_groups)].reset_index(drop=True)

            train_loader = make_loader(
                ChiDataset(df_train, tokenizer, cache_tokenization=cache_tokenization),
                batch_size=train_cfg.get("batch_size", cfg["training_head"]["batch_size"]),
                shuffle=True,
            )
            val_loader = make_loader(
                ChiDataset(df_val, tokenizer, cache_tokenization=cache_tokenization),
                batch_size=train_cfg.get("batch_size", cfg["training_head"]["batch_size"]),
                shuffle=False,
            )

            model = DiT(config).to(device)
            ckpt = torch.load(Path(cfg["paths"]["results_dir"]) / "step1_dit_pretrain_D8" / "checkpoints" / "model_best.pt", map_location="cpu")
            model.load_state_dict(strip_compile_prefix(ckpt["model_state"]), strict=False)
            model.set_freeze_mode(freeze_mode)
            if compile_in_tuning:
                model = maybe_compile(model, cfg.get("optimization", {}))
            head = ChiHead(config.hidden_size, [head_dim] * head_layers, dropout, mode=model_type).to(device)
            if compile_in_tuning:
                head = maybe_compile(head, cfg.get("optimization", {}))

            trial_scaler, trial_autocast = _amp_helpers(device, use_amp)

            best_val, best_state, _, _ = train_fold(
                model,
                head,
                train_loader,
                val_loader,
                device,
                cfg["optuna"]["max_epochs"],
                cfg["optuna"]["patience"],
                loss_name,
                lr,
                weight_decay,
                pooling,
                default_timestep,
                trial_autocast,
                trial_scaler,
                grad_accum_steps,
                trial=trial,
            )

            # val metrics
            if best_state:
                model.load_state_dict(best_state["model"], strict=False)
                head.load_state_dict(best_state["head"], strict=False)
            y_true, y_pred, temps, polymers = eval_dataset(model, head, val_loader, device, pooling, default_timestep, trial_autocast)
            metrics = regression_metrics(y_true, y_pred)
            fold_mae.append(metrics["mae"])
            fold_rmse.append(metrics["rmse"])
            fold_trend.append(trend_error_by_polymer(polymers, temps, y_true, y_pred))

        mean_mae = float(np.mean(fold_mae))
        trial.set_user_attr("chi_model", model_type)
        trial.set_user_attr("rmse", float(np.mean(fold_rmse)))
        trial.set_user_attr("trend", float(np.mean(fold_trend)))
        return mean_mae

    study = create_study("step3_chiT", direction="min")
    study.optimize(objective, n_trials=cfg["optuna"]["n_trials"], callbacks=[trial_logger(str(metrics_dir / "trial.txt"))])

    best_params = study.best_trial.params
    best_params["model"] = chi_model
    with open(results_dir / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    save_study(study, str(metrics_dir / "optuna_study.csv"))

    # Train with best params across folds and collect predictions
    all_train_true, all_train_pred = [], []
    all_val_true, all_val_pred = [], []
    all_test_true, all_test_pred = [], []
    all_test_temp, all_test_poly, all_test_std = [], [], []

    train_log_rows = []
    fold_metrics = []

    for fold_idx, test_group in enumerate(unique_groups):
        remaining = [g for g in unique_groups if g != test_group]
        rng = random.Random(cfg.get("seed", 42) + fold_idx)
        rng.shuffle(remaining)
        n_val = max(1, int(0.1 * len(remaining)))
        val_groups = set(remaining[:n_val])
        train_groups = set(remaining[n_val:])

        df_train = df[df["SMILES"].isin(train_groups)].reset_index(drop=True)
        df_val = df[df["SMILES"].isin(val_groups)].reset_index(drop=True)
        df_test = df[df["SMILES"].isin([test_group])].reset_index(drop=True)

        train_loader = make_loader(
            ChiDataset(df_train, tokenizer, cache_tokenization=cache_tokenization),
            batch_size=train_cfg.get("batch_size", cfg["training_head"]["batch_size"]),
            shuffle=True,
        )
        val_loader = make_loader(
            ChiDataset(df_val, tokenizer, cache_tokenization=cache_tokenization),
            batch_size=train_cfg.get("batch_size", cfg["training_head"]["batch_size"]),
            shuffle=False,
        )
        test_loader = make_loader(
            ChiDataset(df_test, tokenizer, cache_tokenization=cache_tokenization),
            batch_size=train_cfg.get("batch_size", cfg["training_head"]["batch_size"]),
            shuffle=False,
        )

        model = DiT(config).to(device)
        ckpt = torch.load(Path(cfg["paths"]["results_dir"]) / "step1_dit_pretrain_D8" / "checkpoints" / "model_best.pt", map_location="cpu")
        model.load_state_dict(strip_compile_prefix(ckpt["model_state"]), strict=False)
        model.set_freeze_mode(best_params["freeze_mode"])
        model = maybe_compile(model, cfg.get("optimization", {}))
        head = ChiHead(config.hidden_size, [best_params["head_dim"]] * best_params["head_layers"], best_params["dropout"], mode=best_params["model"]).to(device)
        head = maybe_compile(head, cfg.get("optimization", {}))

        optimizer = build_optimizer(list(model.parameters()) + list(head.parameters()), best_params["lr"], best_params["weight_decay"])
        stopper = EarlyStopping(cfg["optuna"]["patience"], mode="min")
        fold_scaler, fold_autocast = _amp_helpers(device, use_amp)

        for epoch in range(cfg["optuna"]["max_epochs"]):
            model.train()
            head.train()
            total_loss = 0.0
            count = 0
            accum = 0
            optimizer.zero_grad()
            for ids, attn, temp, low, high, _ in train_loader:
                ids = ids.to(device)
                attn = attn.to(device)
                temp = temp.to(device)
                low = low.to(device)
                high = high.to(device)
                with fold_autocast():
                    pred = forward_batch(model, head, ids, attn, temp, pooling, default_timestep)
                    if best_params["loss"] == "interval":
                        loss = interval_loss(pred, low, high, loss_type="mse")
                    elif best_params["loss"] == "huber":
                        loss = torch.nn.functional.smooth_l1_loss(pred, (low + high) / 2.0)
                    else:
                        loss = torch.nn.functional.mse_loss(pred, (low + high) / 2.0)
                    loss = loss / grad_accum_steps
                if fold_scaler is not None:
                    fold_scaler.scale(loss).backward()
                else:
                    loss.backward()
                accum += 1
                if accum % grad_accum_steps == 0:
                    if fold_scaler is not None:
                        fold_scaler.step(optimizer)
                        fold_scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                total_loss += loss.item() * grad_accum_steps
                count += 1
            train_loss = total_loss / max(1, count)
            if accum % grad_accum_steps != 0:
                if fold_scaler is not None:
                    fold_scaler.step(optimizer)
                    fold_scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            # val MAE
            y_true, y_pred, _, _ = eval_dataset(model, head, val_loader, device, pooling, default_timestep, fold_autocast)
            val_mae = regression_metrics(y_true, y_pred)["mae"]
            train_log_rows.append({"fold": fold_idx, "epoch": epoch + 1, "train_loss": train_loss, "val_mae": val_mae})
            if stopper.step(val_mae):
                break

        # save checkpoint per fold
        torch.save(
            {
                "model_state": strip_compile_prefix(model.state_dict()),
                "head_state": strip_compile_prefix(head.state_dict()),
            },
            results_dir / "checkpoints" / f"model_best_fold{fold_idx}.pt",
        )

        # predictions
        y_true_tr, y_pred_tr, _, _ = eval_dataset(model, head, train_loader, device, pooling, default_timestep, fold_autocast)
        y_true_va, y_pred_va, _, _ = eval_dataset(model, head, val_loader, device, pooling, default_timestep, fold_autocast)
        y_true_te, y_pred_te, temps, polymers = eval_dataset(model, head, test_loader, device, pooling, default_timestep, fold_autocast)

        all_train_true.extend(y_true_tr)
        all_train_pred.extend(y_pred_tr)
        all_val_true.extend(y_true_va)
        all_val_pred.extend(y_pred_va)
        all_test_true.extend(y_true_te)
        all_test_pred.extend(y_pred_te)
        all_test_temp.extend(temps)
        all_test_poly.extend(polymers)

        # uncertainty for test
        for ids, attn, temp, low, high, polymer in test_loader:
            ids = ids.to(device)
            attn = attn.to(device)
            temp = temp.to(device)
            mean, std = mc_dropout_predict(model, head, ids, attn, temp, pooling, default_timestep, k=50)
            all_test_std.extend(std.tolist())

        m = regression_metrics(y_true_te, y_pred_te)
        fold_metrics.append(m)

    # save logs
    pd.DataFrame(train_log_rows).to_csv(metrics_dir / "train_log.csv", index=False)

    # metrics
    mae_vals = [m["mae"] for m in fold_metrics]
    rmse_vals = [m["rmse"] for m in fold_metrics]
    trend = trend_error_by_polymer(all_test_poly, all_test_temp, all_test_true, all_test_pred)
    coverage = uncertainty_coverage(all_test_true, all_test_pred, all_test_std)

    with open(metrics_dir / "metrics.csv", "w") as f:
        f.write("metric,mean,std\n")
        f.write(f"mae,{np.mean(mae_vals)},{np.std(mae_vals)}\n")
        f.write(f"rmse,{np.mean(rmse_vals)},{np.std(rmse_vals)}\n")
        f.write(f"trend_error,{trend},0\n")
        f.write(f"coverage_68,{coverage['coverage_68']},0\n")
        f.write(f"coverage_95,{coverage['coverage_95']},0\n")

    # predictions
    pred_rows = []
    for split, y_t, y_p, y_s in [
        ("train", all_train_true, all_train_pred, [float("nan")] * len(all_train_true)),
        ("val", all_val_true, all_val_pred, [float("nan")] * len(all_val_true)),
        ("test", all_test_true, all_test_pred, all_test_std),
    ]:
        for yt, yp, ys in zip(y_t, y_p, y_s):
            pred_rows.append({"split": split, "y_true": yt, "y_pred": yp, "y_std": ys})
    pd.DataFrame(pred_rows).to_csv(metrics_dir / "predictions.csv", index=False)

    # plots
    parity_plot(all_train_true, all_train_pred, str(figures_dir / "fig_parity_train.png"))
    parity_plot(all_val_true, all_val_pred, str(figures_dir / "fig_parity_val.png"))
    parity_plot(all_test_true, all_test_pred, str(figures_dir / "fig_parity_test.png"))

    save_loss_plot(
        [r["train_loss"] for r in train_log_rows],
        [r["val_mae"] for r in train_log_rows],
        str(figures_dir / "fig_loss.png"),
    )

    # M1 vs M2 bar plot
    trials = study.trials
    best_m1 = min((t for t in trials if t.params.get("model") == "M1" and t.value is not None), key=lambda t: t.value, default=None)
    best_m2 = min((t for t in trials if t.params.get("model") == "M2" and t.value is not None), key=lambda t: t.value, default=None)
    if best_m1 and best_m2:
        bar_plot(
            ["M1", "M2"],
            [
                [best_m1.value, best_m2.value],
                [best_m1.user_attrs.get("rmse", np.nan), best_m2.user_attrs.get("rmse", np.nan)],
                [best_m1.user_attrs.get("trend", np.nan), best_m2.user_attrs.get("trend", np.nan)],
            ],
            ["MAE", "RMSE", "Trend"],
            str(figures_dir / "fig_M1_vs_M2.png"),
            ylabel="error",
        )

    coverage_plot([0.68, 0.95], [coverage["coverage_68"], coverage["coverage_95"]], str(figures_dir / "fig_uncertainty_coverage.png"))

    # chi(T) examples
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(5, 5))
        plt.rcParams.update({"font.size": 12})
        examples = list(dict.fromkeys(all_test_poly))[:4]
        for ex in examples:
            idxs = [i for i, p in enumerate(all_test_poly) if p == ex]
            temps = np.array([all_test_temp[i] for i in idxs])
            true = np.array([all_test_true[i] for i in idxs])
            pred = np.array([all_test_pred[i] for i in idxs])
            std = np.array([all_test_std[i] for i in idxs])
            order = np.argsort(temps)
            temps = temps[order]
            true = true[order]
            pred = pred[order]
            std = std[order]
            plt.plot(temps, true, "o-", label=f"{ex} true")
            plt.plot(temps, pred, "--", label=f"{ex} pred")
            plt.fill_between(temps, pred - std, pred + std, alpha=0.2)
        plt.xlabel("T")
        plt.ylabel("chi")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(str(figures_dir / "fig_chiT_examples.png"), dpi=300)
        plt.close()
    except Exception:
        pass

    # run info
    run_info = {"seed": cfg.get("seed", 42), "timestamp": datetime.utcnow().isoformat(), "device": device}
    with open(results_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)
    end_log(log_path, start_time, status="completed")


if __name__ == "__main__":
    main()
