import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv
import gzip
import json
import os
import random
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from src.tokenizer import SmilesTokenizer, build_vocab, count_oov
from src.data_utils import ensure_dir, set_seed, read_csv_rows, write_csv
from src.plot_utils import hist_plot
from src.log_utils import start_log, end_log


def detect_smiles_key(row):
    for key in ["SMILES", "smiles", "p_smiles", "pSMILES"]:
        if key in row:
            return key
    raise KeyError("No SMILES column found")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    data_dir = cfg["paths"]["data_dir"]
    d8_path = cfg["paths"]["d8_file"]
    results_dir = Path(cfg["paths"]["results_dir"]) / "step0_tokenizer_split"
    metrics_dir = results_dir / "metrics"
    figures_dir = results_dir / "figures"
    ensure_dir(results_dir)
    ensure_dir(metrics_dir)
    ensure_dir(figures_dir)
    log_path = results_dir / "log.txt"
    start_time = start_log(log_path, "step0_tokenizer_split", args.config)

    special_tokens = cfg["special_tokens"]
    max_length = cfg["model"]["max_length"]

    # Build vocab from D8 train split while writing splits
    split_path = metrics_dir / "splits_D8.csv"
    lengths_train = []
    lengths_val = []
    max_samples = 200000
    sample_prob = None
    save_unlabeled = cfg.get("data", {}).get("save_unlabeled_csv", False)
    train_unlabeled_path = results_dir / "train_unlabeled.csv"
    val_unlabeled_path = results_dir / "val_unlabeled.csv"

    rng = random.Random(seed)
    vocab_set = None

    train_ctx = open(train_unlabeled_path, "w", newline="") if save_unlabeled else nullcontext()
    val_ctx = open(val_unlabeled_path, "w", newline="") if save_unlabeled else nullcontext()

    with gzip.open(d8_path, "rt") as f, open(split_path, "w", newline="") as out_f, train_ctx as train_f, val_ctx as val_f:
        reader = csv.DictReader(f)
        writer = csv.DictWriter(out_f, fieldnames=["id", "split"])
        writer.writeheader()
        train_writer = None
        val_writer = None
        if save_unlabeled:
            train_writer = csv.DictWriter(train_f, fieldnames=["smiles"])
            val_writer = csv.DictWriter(val_f, fieldnames=["smiles"])
            train_writer.writeheader()
            val_writer.writeheader()
        vocab_set = None
        idx = 0
        for row in reader:
            smiles_key = detect_smiles_key(row)
            smiles = row[smiles_key]
            split = "val" if rng.random() < 0.05 else "train"
            writer.writerow({"id": idx, "split": split})
            if save_unlabeled:
                if split == "train":
                    train_writer.writerow({"smiles": smiles})
                else:
                    val_writer.writerow({"smiles": smiles})
            tokens = SmilesTokenizer.tokenize(smiles)
            if split == "train":
                if vocab_set is None:
                    vocab_set = {}
                    for tok in special_tokens.values():
                        vocab_set[tok] = len(vocab_set)
                for tok in tokens:
                    if tok not in vocab_set:
                        vocab_set[tok] = len(vocab_set)
            # reservoir-ish sampling for length hist
            if len(lengths_train) + len(lengths_val) < max_samples:
                if split == "train":
                    lengths_train.append(len(tokens))
                else:
                    lengths_val.append(len(tokens))
            else:
                # after sample filled, randomly replace
                if sample_prob is None:
                    sample_prob = max_samples / float(idx + 1)
                if rng.random() < sample_prob:
                    if split == "train":
                        lengths_train[rng.randrange(len(lengths_train))] = len(tokens)
                    else:
                        lengths_val[rng.randrange(len(lengths_val))] = len(tokens)
            idx += 1

    vocab = vocab_set or build_vocab([], special_tokens)
    tokenizer = SmilesTokenizer(vocab, special_tokens, max_length=max_length)
    tokenizer.to_json(str(results_dir / "tokenizer.json"))

    # stats and OOV for CSV datasets (D1-D6)
    stats_rows = []
    csv_files = [
        "Watersoluble_Polymers_Hansen.csv",
        "Watersoluble_Polymers_no_Hansen.csv",
        "Waterinsoluble_Polymers_Hansen.csv",
        "Waterinsoluble_Polymers_no_Hansen.csv",
        "OMG_DFT_COSMOSAC_chi.csv",
        "OMG_DFT_COSMOC_chi.csv",
        "Experiment_chi.csv",
    ]

    seen = set()
    for name in csv_files:
        path = Path(data_dir) / name
        if not path.exists() or name in seen:
            continue
        seen.add(name)
        rows = read_csv_rows(str(path))
        if not rows:
            continue
        key = detect_smiles_key(rows[0])
        smiles_list = [r[key] for r in rows]
        oov, total = count_oov(smiles_list, tokenizer.vocab)
        lengths = [len(SmilesTokenizer.tokenize(s)) for s in smiles_list]
        q10, q50, q90 = np.quantile(lengths, [0.1, 0.5, 0.9])
        stats_rows.append(
            {
                "dataset": name,
                "n": len(smiles_list),
                "oov_rate": oov / total if total else 0.0,
                "len_q10": float(q10),
                "len_q50": float(q50),
                "len_q90": float(q90),
            }
        )

    # add D8 sample stats
    if lengths_train:
        q10, q50, q90 = np.quantile(lengths_train, [0.1, 0.5, 0.9])
        stats_rows.append(
            {
                "dataset": "D8_train_sample",
                "n": len(lengths_train),
                "oov_rate": 0.0,
                "len_q10": float(q10),
                "len_q50": float(q50),
                "len_q90": float(q90),
            }
        )
    if lengths_val:
        q10, q50, q90 = np.quantile(lengths_val, [0.1, 0.5, 0.9])
        stats_rows.append(
            {
                "dataset": "D8_val_sample",
                "n": len(lengths_val),
                "oov_rate": 0.0,
                "len_q10": float(q10),
                "len_q50": float(q50),
                "len_q90": float(q90),
            }
        )

    write_csv(str(metrics_dir / "stats.csv"), ["dataset", "n", "oov_rate", "len_q10", "len_q50", "len_q90"], stats_rows)

    # Plots
    if lengths_train or lengths_val:
        hist_plot(
            [lengths_train, lengths_val],
            ["train", "val"],
            str(figures_dir / "fig_len_hist.png"),
            xlabel="token length",
        )

    # OOV bar plot
    if stats_rows:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(5, 5))
        plt.rcParams.update({"font.size": 12})
        labels = [r["dataset"] for r in stats_rows]
        rates = [r["oov_rate"] for r in stats_rows]
        plt.bar(labels, rates)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("OOV rate")
        plt.tight_layout()
        plt.savefig(str(figures_dir / "fig_oov_by_dataset.png"), dpi=300)
        plt.close()

    # run info
    run_info = {
        "seed": seed,
        "timestamp": datetime.utcnow().isoformat(),
        "config": args.config,
    }
    with open(results_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)
    end_log(log_path, start_time, status="completed")


if __name__ == "__main__":
    main()
