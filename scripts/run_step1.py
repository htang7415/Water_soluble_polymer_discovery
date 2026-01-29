import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import warnings
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch
import yaml
from torch.utils.data import DataLoader

from src.tokenizer import SmilesTokenizer
from src.datasets import SmilesDataset, collate_fn
from src.models.dit import DiT, DiTConfig
from src.train_utils import build_optimizer, build_scheduler, save_checkpoint, maybe_compile
from src.plot_utils import save_loss_plot
from src.data_utils import ensure_dir
from src.log_utils import start_log, end_log


def detect_smiles_key(row):
    for key in ["SMILES", "smiles", "p_smiles", "pSMILES"]:
        if key in row:
            return key
    raise KeyError("No SMILES column found")


def detect_smiles_key_from_fields(fields):
    for key in ["smiles", "SMILES", "p_smiles", "pSMILES"]:
        if key in fields:
            return key
    raise KeyError("No SMILES column found")


def load_split_ids(split_path: Path):
    val_ids = set()
    n_train = 0
    n_val = 0
    with open(split_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == "val":
                val_ids.add(int(row["id"]))
                n_val += 1
            else:
                n_train += 1
    return val_ids, n_train, n_val


def mask_tokens(input_ids, tokenizer, t, beta_min, beta_max, num_steps):
    # input_ids: [B, L]
    device = input_ids.device
    beta_t = beta_min + (beta_max - beta_min) * (t.float() / float(num_steps))
    # broadcast beta to batch
    mask_prob = beta_t.unsqueeze(1).clamp(0, 1)
    rand = torch.rand(input_ids.shape, device=device)
    mask = rand < mask_prob
    # do not mask special tokens
    special = (input_ids == tokenizer.pad_id) | (input_ids == tokenizer.bos_id) | (input_ids == tokenizer.eos_id)
    mask = mask & (~special)
    noisy = input_ids.clone()
    noisy[mask] = tokenizer.mask_id
    return noisy, mask


def batch_iterable(d8_path, val_ids, tokenizer, split, batch_size):
    import gzip

    with gzip.open(d8_path, "rt") as f:
        reader = csv.DictReader(f)
        batch = []
        for idx, row in enumerate(reader):
            is_val = idx in val_ids
            if (split == "val" and not is_val) or (split == "train" and is_val):
                continue
            smiles = row[detect_smiles_key(row)]
            ids = tokenizer.encode(smiles, add_special=True, pad=True)
            batch.append(ids)
            if len(batch) >= batch_size:
                yield torch.tensor(batch, dtype=torch.long)
                batch = []
        if batch:
            yield torch.tensor(batch, dtype=torch.long)


def load_smiles_csv(path: Path):
    smiles = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return smiles
        key = detect_smiles_key_from_fields(reader.fieldnames)
        for row in reader:
            smiles.append(row[key])
    return smiles


def build_dataloaders(
    train_csv: Path,
    val_csv: Path,
    tokenizer: SmilesTokenizer,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    cache_tokenization: bool,
) -> Tuple[DataLoader, DataLoader]:
    train_smiles = load_smiles_csv(train_csv)
    val_smiles = load_smiles_csv(val_csv)
    train_dataset = SmilesDataset(train_smiles, tokenizer, cache_tokenization=cache_tokenization)
    val_dataset = SmilesDataset(val_smiles, tokenizer, cache_tokenization=cache_tokenization)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_fn,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def main():
    warnings.filterwarnings(
        "ignore",
        message="enable_nested_tensor is True.*encoder_layer.norm_first.*",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    results_dir = Path(cfg["paths"]["results_dir"]) / "step1_dit_pretrain_D8"
    metrics_dir = results_dir / "metrics"
    figures_dir = results_dir / "figures"
    ensure_dir(results_dir)
    ensure_dir(metrics_dir)
    ensure_dir(figures_dir)
    ensure_dir(results_dir / "checkpoints")
    log_path = results_dir / "log.txt"
    start_time = start_log(log_path, "step1_dit_pretrain_D8", args.config, device=args.device)

    tokenizer = SmilesTokenizer.from_json(str(Path(cfg["paths"]["results_dir"]) / "step0_tokenizer_split" / "tokenizer.json"))

    split_path = Path(cfg["paths"]["results_dir"]) / "step0_tokenizer_split" / "metrics" / "splits_D8.csv"
    val_ids, n_train, n_val = load_split_ids(split_path)

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

    model = DiT(config).to(args.device)
    model = maybe_compile(model, cfg.get("optimization", {}))

    train_cfg = cfg["training_backbone"]
    batch_size = args.batch_size or train_cfg["batch_size"]
    opt_cfg = cfg.get("optimization", {})
    accum_steps = args.grad_accum_steps or opt_cfg.get("gradient_accumulation_steps", train_cfg.get("grad_accum_steps", 1))
    use_amp = opt_cfg.get("use_amp", train_cfg.get("use_amp", False)) if args.amp is None else args.amp
    max_steps = train_cfg["max_steps"]
    eval_every = train_cfg.get("eval_every", 1000)
    save_every = train_cfg.get("save_every", 10000)

    optimizer = build_optimizer(model, train_cfg["learning_rate"], train_cfg["weight_decay"])
    scheduler = build_scheduler(optimizer, train_cfg["warmup_steps"], max_steps)
    # AMP helpers (avoid deprecated cuda.amp APIs)
    try:
        from torch.amp import GradScaler, autocast as amp_autocast
        amp_device = "cuda" if args.device.startswith("cuda") else "cpu"
        amp_dtype = torch.bfloat16 if amp_device == "cuda" else torch.float32
        scaler = GradScaler(amp_device, enabled=use_amp)
        def autocast_ctx():
            return amp_autocast(device_type=amp_device, dtype=amp_dtype, enabled=use_amp)
    except Exception:
        from torch.cuda.amp import GradScaler, autocast as amp_autocast
        amp_device = "cuda"
        amp_dtype = torch.bfloat16
        scaler = GradScaler(enabled=use_amp)
        def autocast_ctx():
            return amp_autocast(enabled=use_amp, dtype=amp_dtype)

    if args.device.startswith("cuda") and opt_cfg.get("cudnn_benchmark", True):
        torch.backends.cudnn.benchmark = True

    best_val = float("inf")
    train_log = []
    global_step = 0
    accum_counter = 0
    running_loss = 0.0
    running_count = 0
    optimizer.zero_grad()

    data_cfg = cfg.get("data", {})
    train_csv = Path(cfg["paths"]["results_dir"]) / "step0_tokenizer_split" / "train_unlabeled.csv"
    val_csv = Path(cfg["paths"]["results_dir"]) / "step0_tokenizer_split" / "val_unlabeled.csv"
    use_preprocessed = data_cfg.get("use_preprocessed_csv", True) and train_csv.exists() and val_csv.exists()
    num_workers = opt_cfg.get("num_workers", 0)
    pin_memory = opt_cfg.get("pin_memory", True) and args.device.startswith("cuda")
    prefetch_factor = opt_cfg.get("prefetch_factor", 2)
    cache_tokenization = opt_cfg.get("cache_tokenization", False)
    non_blocking = pin_memory and args.device.startswith("cuda")

    train_loader = None
    val_loader = None
    if use_preprocessed:
        train_loader, val_loader = build_dataloaders(
            train_csv,
            val_csv,
            tokenizer,
            batch_size,
            num_workers,
            pin_memory,
            prefetch_factor,
            cache_tokenization,
        )

    def iter_batches(split: str):
        if split == "train" and train_loader is not None:
            for batch in train_loader:
                yield batch["input_ids"], batch["attention_mask"]
            return
        if split == "val" and val_loader is not None:
            for batch in val_loader:
                yield batch["input_ids"], batch["attention_mask"]
            return
        for batch in batch_iterable(cfg["paths"]["d8_file"], val_ids, tokenizer, split, batch_size):
            attn = (batch != tokenizer.pad_id).long()
            yield batch, attn

    def eval_val():
        model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for ids, attn in iter_batches("val"):
                ids = ids.to(args.device, non_blocking=non_blocking)
                attn = attn.to(args.device, non_blocking=non_blocking)
                timesteps = torch.randint(1, cfg["diffusion"]["num_steps"] + 1, (ids.size(0),), device=args.device)
                noisy, mask = mask_tokens(ids, tokenizer, timesteps, cfg["diffusion"]["beta_min"], cfg["diffusion"]["beta_max"], cfg["diffusion"]["num_steps"])
                with autocast_ctx():
                    logits = model(noisy, timesteps, attn)
                    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
                    loss = loss_fn(logits.view(-1, logits.size(-1)), ids.view(-1))
                    loss = loss.view(ids.size(0), ids.size(1))
                    masked_loss = loss[mask]
                    if masked_loss.numel() == 0:
                        continue
                    loss = masked_loss.mean()
                total_loss += loss.item()
                count += 1
        model.train()
        return total_loss / max(1, count)

    while global_step < max_steps:
        for ids, attn in iter_batches("train"):
            if global_step >= max_steps:
                break
            ids = ids.to(args.device, non_blocking=non_blocking)
            attn = attn.to(args.device, non_blocking=non_blocking)
            timesteps = torch.randint(1, cfg["diffusion"]["num_steps"] + 1, (ids.size(0),), device=args.device)
            noisy, mask = mask_tokens(ids, tokenizer, timesteps, cfg["diffusion"]["beta_min"], cfg["diffusion"]["beta_max"], cfg["diffusion"]["num_steps"])
            with autocast_ctx():
                logits = model(noisy, timesteps, attn)
                loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
                loss = loss_fn(logits.view(-1, logits.size(-1)), ids.view(-1))
                loss = loss.view(ids.size(0), ids.size(1))
                masked_loss = loss[mask]
                if masked_loss.numel() == 0:
                    continue
                raw_loss = masked_loss.mean()
                loss = raw_loss / max(1, accum_steps)
            scaler.scale(loss).backward()
            accum_counter += 1
            running_loss += raw_loss.item()
            running_count += 1

            if accum_counter >= accum_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["gradient_clip_norm"])
                grads_finite = True
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        grads_finite = False
                        break
                if grads_finite:
                    scaler.step(optimizer)
                    scheduler.step()
                    did_step = True
                else:
                    did_step = False
                scaler.update()
                optimizer.zero_grad()
                accum_counter = 0
                if did_step:
                    global_step += 1

                    if global_step % eval_every == 0 or global_step == max_steps:
                        train_loss = running_loss / max(1, running_count)
                        running_loss = 0.0
                        running_count = 0
                        val_loss = eval_val()
                        train_log.append((global_step, train_loss, val_loss))
                        if val_loss < best_val:
                            best_val = val_loss
                            save_checkpoint(str(results_dir / "checkpoints" / "model_best.pt"), model, optimizer)

                    if save_every and global_step % save_every == 0:
                        save_checkpoint(str(results_dir / "checkpoints" / f"model_step_{global_step}.pt"), model, optimizer)

    save_checkpoint(str(results_dir / "checkpoints" / "model_last.pt"), model, optimizer)

    # logs
    with open(metrics_dir / "train_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "val_loss"])
        for step, tr, va in train_log:
            writer.writerow([step, tr, va])

    if train_log:
        save_loss_plot([x[1] for x in train_log], [x[2] for x in train_log], str(figures_dir / "fig_loss.png"))

    run_info = {
        "seed": cfg.get("seed", 42),
        "timestamp": datetime.utcnow().isoformat(),
        "device": args.device,
    }
    with open(results_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)
    end_log(log_path, start_time, status="completed")


if __name__ == "__main__":
    main()
