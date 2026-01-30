import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv
import gzip
import heapq
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data_utils import ensure_dir
from src.models.dit import DiT, DiTConfig
from src.models.heads import ChiHead, SolubilityHead
from src.plot_utils import hist_plot
from src.tokenizer import SmilesTokenizer
from src.log_utils import start_log, end_log
from src.train_utils import strip_compile_prefix


def detect_smiles_key(row):
    for key in ["SMILES", "smiles", "p_smiles", "pSMILES"]:
        if key in row:
            return key
    raise KeyError("No SMILES column found")


def canonicalize_smiles(smiles):
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return smiles


def load_soluble_set(d1_path, d2_path):
    import pandas as pd

    s = set()
    for path in [d1_path, d2_path]:
        if not Path(path).exists():
            continue
        df = pd.read_csv(path)
        if "SMILES" in df.columns:
            for smi in df["SMILES"].astype(str).tolist():
                s.add(canonicalize_smiles(smi))
    return s


def mc_dropout_predict(head, emb, temp, k=50):
    preds = []
    head.train()
    with torch.no_grad():
        for _ in range(k):
            preds.append(head(emb, temp).cpu().numpy())
    preds = np.stack(preds, axis=0)
    return preds.mean(axis=0), preds.std(axis=0)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--t_target", type=float, default=298.15)
    parser.add_argument("--top_k", type=int, default=500)
    parser.add_argument("--uncertain_k", type=int, default=200)
    parser.add_argument("--mc_passes", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--w_sol", type=float, default=1.0)
    parser.add_argument("--w_chi", type=float, default=1.0)
    parser.add_argument("--w_unc", type=float, default=0.1)
    parser.add_argument("--w_novel", type=float, default=0.1)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train_cfg = cfg.get("training_property") or cfg.get("training_head", {})
    head_cfg = cfg.get("property_head", {})
    pooling = head_cfg.get("pooling", "cls")
    default_timestep = int(train_cfg.get("default_timestep", 0))

    results_dir = Path(cfg["paths"]["results_dir"]) / "step5_screen_D8"
    metrics_dir = results_dir / "metrics"
    figures_dir = results_dir / "figures"
    ensure_dir(results_dir)
    ensure_dir(metrics_dir)
    ensure_dir(figures_dir)
    log_path = results_dir / "log.txt"
    start_time = start_log(log_path, "step5_screen_D8", args.config, device=args.device)

    tokenizer = SmilesTokenizer.from_json(str(Path(cfg["paths"]["results_dir"]) / "step0_tokenizer_split" / "tokenizer.json"))

    chi_params_path = Path(cfg["paths"]["results_dir"]) / "step3_exp_chi_T_D6" / "best_params.json"
    if chi_params_path.exists():
        chi_params = json.load(open(chi_params_path))
    else:
        chi_params = {"head_dim": 128, "head_layers": 2, "dropout": 0.1, "model": "M1"}

    config = DiTConfig(
        vocab_size=len(tokenizer.vocab),
        hidden_size=cfg["backbone"]["hidden_size"],
        num_layers=cfg["backbone"]["num_layers"],
        num_heads=cfg["backbone"]["num_heads"],
        ffn_hidden_size=cfg["backbone"]["ffn_hidden_size"],
        dropout=chi_params.get("dropout", cfg["backbone"]["dropout"]),
        max_position_embeddings=cfg["backbone"]["max_position_embeddings"],
        num_steps=cfg["diffusion"]["num_steps"],
    )
    model = DiT(config).to(args.device)

    # load step4 backbone and sol head
    step4_dir = Path(cfg["paths"]["results_dir"]) / "step4_solubility_hansen_D1toD4"
    ckpt = torch.load(step4_dir / "model_best.pt", map_location="cpu")
    model.load_state_dict(strip_compile_prefix(ckpt["model_state"]), strict=False)
    step4_params_path = step4_dir / "best_params.json"
    if step4_params_path.exists():
        step4_params = json.load(open(step4_params_path))
        sol_dim = step4_params.get("sol_head_dim", 128)
        sol_layers = step4_params.get("sol_head_layers", 2)
    else:
        sol_dim, sol_layers = 128, 2
    sol_head = SolubilityHead(config.hidden_size, [sol_dim] * sol_layers, chi_params.get("dropout", 0.1)).to(args.device)
    sol_head.load_state_dict(strip_compile_prefix(ckpt["sol_state"]), strict=False)

    chi_head = ChiHead(config.hidden_size, [chi_params["head_dim"]] * chi_params["head_layers"], chi_params.get("dropout", 0.1), mode=chi_params["model"]).to(args.device)
    # load chi head from step3
    chi_ckpt_path = Path(cfg["paths"]["results_dir"]) / "step3_exp_chi_T_D6" / "checkpoints" / "model_best_fold0.pt"
    if not chi_ckpt_path.exists():
        chi_ckpt_path = Path(cfg["paths"]["results_dir"]) / "step3_exp_chi_T_D6" / "checkpoints" / "model_best_full.pt"
    chi_ckpt = torch.load(chi_ckpt_path, map_location="cpu")
    chi_head.load_state_dict(strip_compile_prefix(chi_ckpt["head_state"]), strict=False)

    soluble_set = load_soluble_set(cfg["paths"]["d1_file"], cfg["paths"]["d2_file"])

    top_heap = []
    unc_heap = []
    confidence_samples = []
    novelty_samples = []

    model.eval()
    sol_head.eval()
    chi_head.eval()

    with gzip.open(cfg["paths"]["d8_file"], "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles = row[detect_smiles_key(row)]
            ids = torch.tensor([tokenizer.encode(smiles)], dtype=torch.long).to(args.device)
            attn = (ids != tokenizer.pad_id).long()
            temp = torch.tensor([args.t_target], dtype=torch.float).to(args.device)
            with torch.no_grad():
                timesteps = torch.full((1,), default_timestep, device=args.device, dtype=torch.long)
                hidden = model.forward_hidden(ids, timesteps, attn)
                emb = _pool_hidden(hidden, attn, pooling)
                p_sol = torch.sigmoid(sol_head(emb)).item()
            if args.mc_passes > 0:
                mean, std = mc_dropout_predict(chi_head, emb, temp, k=args.mc_passes)
                chi_pred = float(mean[0])
                chi_std = float(std[0])
            else:
                with torch.no_grad():
                    chi_pred = float(chi_head(emb, temp).item())
                chi_std = 0.0

            canon = canonicalize_smiles(smiles)
            novelty = 0.0 if canon in soluble_set else 1.0
            score = args.w_sol * p_sol - args.w_chi * chi_pred - args.w_unc * chi_std + args.w_novel * novelty

            # top candidates heap
            if len(top_heap) < args.top_k:
                heapq.heappush(top_heap, (score, smiles, p_sol, chi_pred, chi_std, novelty))
            else:
                heapq.heappushpop(top_heap, (score, smiles, p_sol, chi_pred, chi_std, novelty))

            # frontier uncertain
            if p_sol >= 0.5:
                if len(unc_heap) < args.uncertain_k:
                    heapq.heappush(unc_heap, (chi_std, smiles, p_sol, chi_pred, novelty))
                else:
                    heapq.heappushpop(unc_heap, (chi_std, smiles, p_sol, chi_pred, novelty))

            # samples for plots
            if len(confidence_samples) < 5000:
                confidence_samples.append(p_sol)
                novelty_samples.append(novelty)

    top_candidates = sorted(top_heap, key=lambda x: -x[0])
    frontier_uncertain = sorted(unc_heap, key=lambda x: -x[0])

    # write outputs
    with open(metrics_dir / "top_candidates.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["score", "SMILES", "P_sol", "chi_pred", "chi_std", "novelty"])
        for row in top_candidates:
            writer.writerow(row)

    with open(metrics_dir / "frontier_uncertain.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chi_std", "SMILES", "P_sol", "chi_pred", "novelty"])
        for row in frontier_uncertain:
            writer.writerow(row)

    summary = {
        "n_top": len(top_candidates),
        "mean_p_sol": float(np.mean(confidence_samples)) if confidence_samples else 0.0,
        "novelty_rate_sample": float(np.mean(novelty_samples)) if novelty_samples else 0.0,
    }
    with open(metrics_dir / "screening_summary.csv", "w") as f:
        f.write("metric,value\n")
        for k, v in summary.items():
            f.write(f"{k},{v}\n")

    # plots
    if confidence_samples:
        hist_plot([confidence_samples], ["P_sol"], str(figures_dir / "fig_confidence_hist.png"), xlabel="P_sol")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(5, 5))
        plt.rcParams.update({"font.size": 12})
        plt.scatter(confidence_samples, novelty_samples, s=8, alpha=0.6)
        plt.xlabel("P_sol")
        plt.ylabel("Novelty")
        plt.tight_layout()
        plt.savefig(str(figures_dir / "fig_novelty_vs_confidence.png"), dpi=300)
        plt.close()

    run_info = {"seed": cfg.get("seed", 42), "timestamp": datetime.utcnow().isoformat(), "device": args.device}
    with open(results_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)
    end_log(log_path, start_time, status="completed")


if __name__ == "__main__":
    main()
