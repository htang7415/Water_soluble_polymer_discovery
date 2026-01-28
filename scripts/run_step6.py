import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from src.data_utils import ensure_dir
from src.plot_utils import bar_plot
from src.log_utils import start_log, end_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    results_dir = Path(cfg["paths"]["results_dir"]) / "step6_paper_summary"
    metrics_dir = results_dir / "metrics"
    figures_dir = results_dir / "figures"
    ensure_dir(results_dir)
    ensure_dir(metrics_dir)
    ensure_dir(figures_dir)
    log_path = results_dir / "log.txt"
    start_time = start_log(log_path, "step6_paper_summary", args.config)

    final_rows = []

    # step2 metrics
    step2_metrics = Path(cfg["paths"]["results_dir"]) / "step2_cosmosac_proxy_D5" / "metrics" / "metrics.csv"
    if step2_metrics.exists():
        df = pd.read_csv(step2_metrics)
        for _, row in df.iterrows():
            final_rows.append({"section": "step2", "metric": f"{row['split']}_r2", "value": row["r2"]})
            final_rows.append({"section": "step2", "metric": f"{row['split']}_mae", "value": row["mae"]})

    # step3 metrics
    step3_metrics = Path(cfg["paths"]["results_dir"]) / "step3_exp_chi_T_D6" / "metrics" / "metrics.csv"
    if step3_metrics.exists():
        df = pd.read_csv(step3_metrics)
        for _, row in df.iterrows():
            final_rows.append({"section": "step3", "metric": row["metric"], "value": row["mean"]})

    # step4 metrics
    step4_metrics = Path(cfg["paths"]["results_dir"]) / "step4_solubility_hansen_D1toD4" / "metrics" / "metrics.csv"
    if step4_metrics.exists():
        df = pd.read_csv(step4_metrics)
        for _, row in df.iterrows():
            final_rows.append({"section": "step4", "metric": row["metric"], "value": row["value"]})

    pd.DataFrame(final_rows).to_csv(metrics_dir / "final_tables.csv", index=False)

    # ablation summary from step3 study
    ablation_rows = []
    study_path = Path(cfg["paths"]["results_dir"]) / "step3_exp_chi_T_D6" / "metrics" / "optuna_study.csv"
    if study_path.exists():
        df = pd.read_csv(study_path)
        if "params_model" in df.columns:
            for model in ["M1", "M2"]:
                df_m = df[df["params_model"] == model]
                if not df_m.empty:
                    best = df_m.sort_values("value").iloc[0]
                    ablation_rows.append({"model": model, "mae": best["value"]})
    pd.DataFrame(ablation_rows).to_csv(metrics_dir / "ablation_summary.csv", index=False)

    # plots
    if ablation_rows:
        bar_plot(
            [r["model"] for r in ablation_rows],
            [[r["mae"] for r in ablation_rows]],
            ["MAE"],
            str(figures_dir / "fig_ablation_chi.png"),
            ylabel="MAE",
        )

    if step4_metrics.exists():
        df = pd.read_csv(step4_metrics)
        if "metric" in df.columns and "value" in df.columns:
            auprc = df[df["metric"] == "auprc_val"]
            if not auprc.empty:
                bar_plot(
                    ["solubility"],
                    [[float(auprc["value"].values[0])]],
                    ["AUPRC"],
                    str(figures_dir / "fig_ablation_solubility.png"),
                    ylabel="AUPRC",
                )

    end_log(log_path, start_time, status="completed")


if __name__ == "__main__":
    main()
