import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv
import json
import random
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data_utils import ensure_dir
from src.models.dit import DiT, DiTConfig
from src.tokenizer import SmilesTokenizer
from src.chem_utils import count_stars, compute_sa_score
from src.generative_metrics import GenerativeEvaluator
from src.log_utils import start_log, end_log

BOND_CHARS = set(['-', '=', '#', '/', '\\'])


def _smiles_constraint_violations(smiles: str) -> dict:
    if not smiles:
        return {
            "star_count": True,
            "bond_placement": True,
            "paren_balance": True,
            "empty_parens": True,
            "ring_closure": True,
        }

    star_violation = count_stars(smiles) != 2
    empty_parens = "()" in smiles

    depth = 0
    paren_violation = False
    for ch in smiles:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth < 0:
                paren_violation = True
                break
    if depth != 0:
        paren_violation = True

    bond_violation = False
    prev = None
    for ch in smiles:
        if ch in BOND_CHARS:
            if prev is None or prev in BOND_CHARS or prev in '()':
                bond_violation = True
                break
        if ch.strip() == "":
            continue
        prev = ch

    ring_tokens = re.findall(r'%\d{2}', smiles)
    no_percent = re.sub(r'%\d{2}', '', smiles)
    ring_tokens += re.findall(r'\d', no_percent)
    ring_violation = False
    if ring_tokens:
        counts = Counter(ring_tokens)
        ring_violation = any(c != 2 for c in counts.values())

    return {
        "star_count": star_violation,
        "bond_placement": bond_violation,
        "paren_balance": paren_violation,
        "empty_parens": empty_parens,
        "ring_closure": ring_violation,
    }


def compute_smiles_constraint_metrics(smiles_list, method, representation, model_size):
    total = len(smiles_list)
    violations = {
        "star_count": 0,
        "bond_placement": 0,
        "paren_balance": 0,
        "empty_parens": 0,
        "ring_closure": 0,
    }

    for smiles in smiles_list:
        flags = _smiles_constraint_violations(smiles)
        for key, violated in flags.items():
            if violated:
                violations[key] += 1

    rows = []
    for constraint, count in violations.items():
        rate = count / total if total > 0 else 0.0
        rows.append({
            "method": method,
            "representation": representation,
            "model_size": model_size,
            "constraint": constraint,
            "total": total,
            "violations": count,
            "violation_rate": round(rate, 4),
        })
    return rows


def load_training_smiles(d8_path, split_path, limit):
    val_ids = set()
    with open(split_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == "val":
                val_ids.add(int(row["id"]))
    smiles = []
    with open(d8_path, "rb") as fh:
        import gzip

        with gzip.open(fh, "rt") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                if idx in val_ids:
                    continue
                smiles.append(row.get("SMILES") or row.get("smiles") or row.get("p_smiles") or row.get("pSMILES"))
                if limit and len(smiles) >= limit:
                    break
    return smiles


def sample_batch(model, tokenizer, num_steps, num_samples, batch_size, seq_length, lengths, temperature, use_constraints, device):
    all_smiles = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, num_samples, batch_size):
            bs = min(batch_size, num_samples - start)
            batch_lengths = lengths[start:start + bs]
            ids = torch.full((bs, seq_length), tokenizer.mask_id, dtype=torch.long, device=device)
            attention = torch.zeros_like(ids)
            for i, L in enumerate(batch_lengths):
                L = int(min(L, seq_length))
                ids[i, 0] = tokenizer.bos_id
                if L >= 2:
                    ids[i, L - 1] = tokenizer.eos_id
                if L < seq_length:
                    ids[i, L:] = tokenizer.pad_id
                attention[i, :L] = 1
            for t in range(num_steps, 0, -1):
                timesteps = torch.full((bs,), t, dtype=torch.long, device=device)
                logits = model(ids, timesteps, attention)
                logits = logits / max(temperature, 1e-6)

                # forbid special tokens in masked positions
                for tok in [tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id, tokenizer.mask_id]:
                    logits[:, :, tok] = -1e9

                mask_positions = ids == tokenizer.mask_id

                if use_constraints:
                    star_id = tokenizer.vocab.get("*", -1)
                    for i in range(bs):
                        if count_stars(tokenizer.decode(ids[i].tolist())) >= 2 and star_id >= 0:
                            logits[i, :, star_id] = -1e9

                probs = torch.softmax(logits, dim=-1)

                for i in range(bs):
                    positions = torch.where(mask_positions[i])[0].tolist()
                    if not positions:
                        continue
                    k = max(1, int(len(positions) / t))
                    select = random.sample(positions, min(k, len(positions)))
                    for pos in select:
                        p = probs[i, pos]
                        idx = torch.multinomial(p, 1).item()
                        ids[i, pos] = idx
            for i in range(bs):
                all_smiles.append(tokenizer.decode(ids[i].tolist()))
    return all_smiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--variable_length", action="store_true")
    parser.add_argument("--min_length", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--samples_per_length", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    results_dir = Path(cfg["paths"]["results_dir"]) / "step1_2_sampling"
    metrics_dir = results_dir / "metrics"
    figures_dir = results_dir / "figures"
    ensure_dir(results_dir)
    ensure_dir(metrics_dir)
    ensure_dir(figures_dir)
    log_path = results_dir / "log.txt"
    start_time = start_log(log_path, "step1_2_sampling", args.config, device=args.device)

    tokenizer = SmilesTokenizer.from_json(str(Path(cfg["paths"]["results_dir"]) / "step0_tokenizer_split" / "tokenizer.json"))

    train_split_path = Path(cfg["paths"]["results_dir"]) / "step0_tokenizer_split" / "metrics" / "splits_D8.csv"
    training_smiles = load_training_smiles(cfg["paths"]["d8_file"], train_split_path, cfg.get("sampling", {}).get("novelty_max", 200000))
    training_smiles = set(training_smiles)

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
    ckpt_path = args.checkpoint or (Path(cfg["paths"]["results_dir"]) / "step1_dit_pretrain_D8" / "checkpoints" / "model_best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=False)

    num_samples = args.num_samples or cfg.get("sampling", {}).get("num_samples", 10000)
    batch_size = args.batch_size or cfg.get("sampling", {}).get("batch_size", 256)
    temperature = args.temperature or cfg.get("sampling", {}).get("temperature", 1.0)
    use_constraints = cfg.get("sampling", {}).get("use_constraints", True)

    lengths = []
    if args.variable_length:
        for L in range(args.min_length, args.max_length + 1):
            lengths.extend([L] * args.samples_per_length)
        if len(lengths) < num_samples:
            lengths = (lengths * (num_samples // len(lengths) + 1))[:num_samples]
        else:
            lengths = lengths[:num_samples]
    else:
        # sample lengths from training distribution
        lengths = []
        if training_smiles:
            sampled = random.choices(list(training_smiles), k=num_samples)
            for s in sampled:
                lengths.append(min(len(tokenizer.tokenize(s)) + 2, tokenizer.max_length))
        else:
            lengths = [tokenizer.max_length] * num_samples

    sampling_start = time.time()
    generated_smiles = sample_batch(
        model,
        tokenizer,
        cfg["diffusion"]["num_steps"],
        num_samples,
        batch_size,
        tokenizer.max_length,
        lengths,
        temperature,
        use_constraints,
        args.device,
    )
    sampling_time = time.time() - sampling_start

    with open(metrics_dir / "generated_samples.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles"])
        for s in generated_smiles:
            writer.writerow([s])

    evaluator = GenerativeEvaluator(training_smiles)
    metrics = evaluator.evaluate(
        generated_smiles,
        sample_id=f"uncond_{num_samples}_best_checkpoint",
        sampling_time_sec=sampling_time,
        method="DiT",
        representation="SMILES",
        model_size="base",
    )
    evaluator.format_metrics_csv(metrics).to_csv(metrics_dir / "sampling_generative_metrics.csv", index=False)

    constraint_rows = compute_smiles_constraint_metrics(generated_smiles, "DiT", "SMILES", "base")
    import pandas as pd

    pd.DataFrame(constraint_rows).to_csv(metrics_dir / "constraint_metrics.csv", index=False)

    # plots
    import matplotlib.pyplot as plt

    valid_smiles = evaluator.get_valid_samples(generated_smiles, require_two_stars=True)
    train_sa = [compute_sa_score(s) for s in list(training_smiles)[:5000]]
    train_sa = [s for s in train_sa if s is not None]
    gen_sa = [compute_sa_score(s) for s in valid_smiles[:5000]]
    gen_sa = [s for s in gen_sa if s is not None]

    plt.figure(figsize=(5, 5))
    plt.rcParams.update({"font.size": 12})
    plt.hist(train_sa, bins=50, alpha=0.6, label="Train", histtype="step")
    plt.hist(gen_sa, bins=50, alpha=0.6, label="Generated", histtype="step")
    plt.xlabel("SA Score")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "sa_hist_train_vs_uncond.png", dpi=300)
    plt.close()

    train_lengths = [len(s) for s in list(training_smiles)[:5000]]
    gen_lengths = [len(s) for s in valid_smiles[:5000]]

    plt.figure(figsize=(5, 5))
    plt.rcParams.update({"font.size": 12})
    plt.hist(train_lengths, bins=50, alpha=0.6, label="Train", histtype="step")
    plt.hist(gen_lengths, bins=50, alpha=0.6, label="Generated", histtype="step")
    plt.xlabel("SMILES Length")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "length_hist_train_vs_uncond.png", dpi=300)
    plt.close()

    star_counts = [count_stars(s) for s in valid_smiles]
    plt.figure(figsize=(5, 5))
    plt.rcParams.update({"font.size": 12})
    counts = Counter(star_counts)
    xs = sorted(counts.keys())
    ys = [counts[x] for x in xs]
    plt.bar(xs, ys)
    plt.xlabel("Star Count")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(figures_dir / "star_count_hist_uncond.png", dpi=300)
    plt.close()

    run_info = {"seed": cfg.get("seed", 42), "timestamp": datetime.utcnow().isoformat(), "device": args.device}
    with open(results_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    end_log(log_path, start_time, status="completed")


if __name__ == "__main__":
    main()
