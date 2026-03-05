#!/usr/bin/env python
"""Step 7: chemistry + physics analysis with quantitative figures and tables."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Avoid OpenMP shared-memory issues on restricted environments.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from scipy.optimize import curve_fit
from scipy.stats import mannwhitneyu, pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chi.model import predict_chi_from_coefficients  # noqa: E402
from src.chi.embeddings import build_or_load_embedding_cache  # noqa: E402
from src.utils.chemistry import (  # noqa: E402
    batch_compute_fingerprints,
    canonicalize_smiles,
    compute_fingerprint,
    compute_tanimoto_similarity,
)
from src.utils.config import load_config, save_config  # noqa: E402
from src.utils.figure_style import apply_publication_figure_style  # noqa: E402
from src.utils.model_scales import get_results_dir  # noqa: E402
from src.utils.reproducibility import save_run_metadata, seed_everything  # noqa: E402
from src.utils.reporting import save_artifact_manifest, save_step_summary, write_initial_log  # noqa: E402


CLASS_COL = "water_miscible"
COEFF_COLUMNS = ["a0", "a1", "a2", "a3", "b1", "b2"]
DESCRIPTOR_COLUMNS = [
    "mol_wt",
    "logp",
    "tpsa",
    "hbd",
    "hba",
    "rot_bonds",
    "ring_count",
    "aromatic_rings",
    "hetero_fraction",
    "frac_csp3",
]
FG_COLUMNS = [
    "fg_hydroxyl",
    "fg_carboxyl",
    "fg_amine",
    "fg_ether",
    "fg_ester",
    "fg_carbonyl",
    "fg_nitrile",
    "fg_halogen",
]

FIGURE_ORDER: List[Tuple[str, str, str]] = [
    ("pipeline_selection_success_rates.png", "A0", "Pipeline selection success rates"),
    ("chi_class_delta_heatmap.png", "A1", "Thermodynamic class delta heatmap"),
    ("chi_class_significance_heatmap.png", "A2", "Thermodynamic significance heatmap"),
    ("chi_vs_temperature_by_phi_and_class.png", "A3", "Chi vs temperature by phi and class"),
    ("step3_target_vs_class_means.png", "A4", "Step3 target vs class means"),
    ("step4_test_mae_heatmap.png", "B1", "Step4 test MAE heatmap"),
    ("step4_gradient_consistency_dchi_dT.png", "B2", "Step4 gradient consistency dchi/dT"),
    ("step4_gradient_consistency_dchi_dphi.png", "B3", "Step4 gradient consistency dchi/dphi"),
    ("selection_tradeoff_chi_vs_solubility_confidence.png", "C1", "Selection tradeoff chi vs confidence"),
    ("descriptor_shift_vs_step2_target_pool.png", "C2", "Descriptor shift vs Step2 target pool"),
    ("descriptor_shift_vs_training.png", "C3", "Descriptor shift vs training"),
    ("novelty_similarity_histogram.png", "C4", "Novelty similarity histogram"),
    ("step6_target_polymer_class_coverage.png", "C5", "Step6 target polymer class coverage"),
    ("coefficient_violin_by_class.png", "D1", "Coefficient violin by class"),
    ("coefficient_a1_vs_a3_by_class.png", "D2", "Coefficient a1 vs a3 by class"),
    ("dchi_dT_distribution_by_class.png", "D3", "dchi/dT distribution by class"),
    ("chi_surface_mean_by_class.png", "E1", "Mean chi surface by class"),
    ("spinodal_phase_diagram.png", "E2", "Spinodal phase diagram"),
    ("miscible_fraction_below_spinodal_by_class.png", "E3", "Miscible fraction below spinodal by class"),
    ("free_energy_mixing_by_class.png", "E4", "Free energy mixing by class"),
    ("descriptor_boxplot_by_class.png", "F1", "Descriptor boxplot by class"),
    ("functional_group_frequency_by_class.png", "F2", "Functional group frequency by class"),
    ("logp_vs_mean_chi_by_class.png", "F3", "LogP vs mean chi by class"),
    ("tpsa_vs_mean_chi_by_class.png", "F4", "TPSA vs mean chi by class"),
    ("descriptor_chi_correlation_heatmap.png", "F5", "Descriptor-chi correlation heatmap"),
    ("classification_overlap_counts.png", "H1", "Classification dataset overlap counts"),
    ("classification_vs_chi_descriptor_shift.png", "H2", "Classification vs chi descriptor shift"),
    ("classification_vs_chi_fg_frequency.png", "H3", "Classification vs chi functional-group frequency"),
    ("step1_embedding_pca_by_source.png", "I1", "Step1 embedding PCA by source"),
    ("embedding_pinn_correlation_heatmap.png", "I2", "Embedding-PINN coefficient correlation heatmap"),
    ("pinn_coefficient_sensitivity_by_class.png", "I3", "PINN coefficient sensitivity by class"),
    ("chemical_space_pca_known_vs_discovered.png", "G1", "Chemical space PCA known vs discovered"),
    ("chi_vs_class_prob_scoring_landscape.png", "G2", "Chi vs class probability scoring landscape"),
    ("discovered_descriptor_boxplot.png", "G3", "Discovered descriptor boxplot"),
    ("artifact_counts_by_category.png", "Z1", "Artifact counts by category"),
]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lstrip("\ufeff") for c in out.columns]
    return out


def _ensure_class_col(df: pd.DataFrame) -> pd.DataFrame:
    out = _normalize_columns(df)
    if CLASS_COL in out.columns:
        out[CLASS_COL] = pd.to_numeric(out[CLASS_COL], errors="coerce").fillna(0).astype(int)
        return out

    aliases = {"water_soluble", "water miscible", "water_miscible", "water_missible", "watermiscible"}
    matched = [c for c in out.columns if str(c).strip().lower() in aliases]
    if not matched:
        raise ValueError(f"Missing `{CLASS_COL}` column. Found columns: {list(out.columns)}")
    out = out.rename(columns={matched[0]: CLASS_COL})
    out[CLASS_COL] = pd.to_numeric(out[CLASS_COL], errors="coerce").fillna(0).astype(int)
    return out


def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p is not None and p.exists():
            return p
    return None


def _safe_float(value, default=np.nan) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return np.nan
    vx = float(np.var(x, ddof=1))
    vy = float(np.var(y, ddof=1))
    denom = np.sqrt(((len(x) - 1) * vx + (len(y) - 1) * vy) / max(len(x) + len(y) - 2, 1))
    if np.isclose(denom, 0.0):
        return np.nan
    return float((np.mean(x) - np.mean(y)) / denom)


def _safe_mannwhitney(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return np.nan
    try:
        return float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def _compute_condition_class_contrast(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for (temperature, phi), sub in df.groupby(["temperature", "phi"], sort=True):
        misc = sub.loc[sub[CLASS_COL] == 1, "chi"].to_numpy(dtype=float)
        imm = sub.loc[sub[CLASS_COL] == 0, "chi"].to_numpy(dtype=float)
        row = {
            "temperature": float(temperature),
            "phi": float(phi),
            "n_total": int(len(sub)),
            "n_miscible": int(len(misc)),
            "n_immiscible": int(len(imm)),
            "chi_miscible_mean": _safe_float(np.mean(misc)) if len(misc) else np.nan,
            "chi_immiscible_mean": _safe_float(np.mean(imm)) if len(imm) else np.nan,
            "delta_mean_chi_miscible_minus_immiscible": _safe_float(np.mean(misc) - np.mean(imm))
            if len(misc) and len(imm)
            else np.nan,
            "cohens_d": _cohens_d(misc, imm),
            "mannwhitney_pvalue": _safe_mannwhitney(misc, imm),
        }
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["temperature", "phi"]).reset_index(drop=True)
    out["significant_p_lt_0p05"] = (pd.to_numeric(out["mannwhitney_pvalue"], errors="coerce") < 0.05).astype(int)
    return out


def _plot_condition_heatmap(
    cond_df: pd.DataFrame,
    value_col: str,
    cbar_label: str,
    out_png: Path,
    cmap: str,
    dpi: int,
) -> None:
    if cond_df.empty:
        return
    pivot = cond_df.pivot(index="temperature", columns="phi", values=value_col)
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        pivot,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": cbar_label},
        ax=ax,
    )
    ax.set_xlabel("φ")
    ax.set_ylabel("Temperature (K)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def _plot_chi_vs_temperature_by_phi(df: pd.DataFrame, out_png: Path, dpi: int) -> None:
    if df.empty:
        return
    summary = (
        df.groupby([CLASS_COL, "temperature", "phi"], as_index=False)["chi"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["sem"] = summary["std"] / np.sqrt(np.maximum(summary["count"], 1))
    summary["ci95"] = 1.96 * summary["sem"].fillna(0.0)
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    palette = {1: "#1f77b4", 0: "#d62728"}
    line_style = {1: "-", 0: "--"}
    phi_values = sorted(summary["phi"].unique().tolist())
    for phi in phi_values:
        for cls in [1, 0]:
            sub = summary[(summary["phi"] == phi) & (summary[CLASS_COL] == cls)].sort_values("temperature")
            if sub.empty:
                continue
            lbl = f"φ={phi:g}, class={cls}"
            ax.plot(
                sub["temperature"],
                sub["mean"],
                marker="o",
                linestyle=line_style[cls],
                color=palette[cls],
                alpha=0.85,
                linewidth=1.8,
                label=lbl,
            )
            ax.fill_between(
                sub["temperature"],
                sub["mean"] - sub["ci95"],
                sub["mean"] + sub["ci95"],
                color=palette[cls],
                alpha=0.10,
            )
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Mean χ")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, ncol=1)
    fig.tight_layout(rect=(0, 0, 0.80, 1))
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def _regression_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "r2": np.nan}
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    if len(y_true) > 1 and not np.isclose(np.var(y_true), 0.0):
        sse = float(np.sum((y_true - y_pred) ** 2))
        sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = 1.0 - (sse / sst) if sst > 0 else np.nan
    else:
        r2 = np.nan
    return {"n": int(len(y_true)), "mae": mae, "rmse": rmse, "bias": bias, "r2": r2}


def _compute_step4_condition_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for (temperature, phi), sub in df.groupby(["temperature", "phi"], sort=True):
        y_true = sub["chi"].to_numpy(dtype=float)
        y_pred = sub["chi_pred"].to_numpy(dtype=float)
        reg = _regression_stats(y_true, y_pred)
        rows.append(
            {
                "temperature": float(temperature),
                "phi": float(phi),
                **reg,
            }
        )
    return pd.DataFrame(rows).sort_values(["temperature", "phi"]).reset_index(drop=True)


def _compute_polymer_gradients(df: pd.DataFrame, axis_col: str) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    if axis_col == "temperature":
        group_cols = ["polymer_id", "phi"]
    elif axis_col == "phi":
        group_cols = ["polymer_id", "temperature"]
    else:
        raise ValueError("axis_col must be `temperature` or `phi`.")

    for keys, sub in df.groupby(group_cols, sort=True):
        if len(sub) < 2:
            continue
        x = pd.to_numeric(sub[axis_col], errors="coerce").to_numpy(dtype=float)
        y_true = pd.to_numeric(sub["chi"], errors="coerce").to_numpy(dtype=float)
        y_pred = pd.to_numeric(sub["chi_pred"], errors="coerce").to_numpy(dtype=float)
        if np.unique(x).size < 2:
            continue
        slope_true, _ = np.polyfit(x, y_true, 1)
        slope_pred, _ = np.polyfit(x, y_pred, 1)
        row: Dict[str, float] = {
            "polymer_id": int(keys[0]) if isinstance(keys, tuple) else int(keys),
            "slope_true": float(slope_true),
            "slope_pred": float(slope_pred),
            "slope_error": float(slope_pred - slope_true),
            "slope_sign_agree": int(np.sign(slope_true) == np.sign(slope_pred)),
            "n_points": int(len(sub)),
        }
        if isinstance(keys, tuple) and len(keys) == 2:
            if axis_col == "temperature":
                row["phi"] = float(keys[1])
            else:
                row["temperature"] = float(keys[1])
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_step4_mae_heatmap(cond_df: pd.DataFrame, out_png: Path, dpi: int) -> None:
    if cond_df.empty:
        return
    pivot = cond_df.pivot(index="temperature", columns="phi", values="mae")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        pivot,
        cmap="YlOrRd",
        annot=True,
        fmt=".3f",
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "MAE(χ)"},
        ax=ax,
    )
    ax.set_xlabel("φ")
    ax.set_ylabel("Temperature (K)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def _plot_gradient_scatter(df: pd.DataFrame, axis_label: str, out_png: Path, dpi: int) -> Dict[str, float]:
    if df.empty:
        return {"n": 0, "pearson_r": np.nan, "sign_agreement_rate": np.nan}

    x = pd.to_numeric(df["slope_true"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["slope_pred"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        return {"n": 0, "pearson_r": np.nan, "sign_agreement_rate": np.nan}

    lo = min(np.min(x), np.min(y))
    hi = max(np.max(x), np.max(y))
    if np.isclose(lo, hi):
        lo -= 1.0
        hi += 1.0
    sign_agree = float(np.mean(np.sign(x) == np.sign(y)))
    corr = float(pearsonr(x, y)[0]) if len(x) >= 2 else np.nan

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(x, y, s=26, alpha=0.75, color="#4c78a8")
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.2)
    ax.set_xlabel(f"True dχ/d{axis_label}")
    ax.set_ylabel(f"Predicted dχ/d{axis_label}")
    ax.text(
        0.03,
        0.97,
        f"n={len(x)}\nPearson r={corr:.3f}\nSign agree={sign_agree:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#666666", "alpha": 0.90},
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return {"n": int(len(x)), "pearson_r": corr, "sign_agreement_rate": sign_agree}


def _mol_from_polymer_smiles(smiles: str):
    variants = [smiles, smiles.replace("*", "[H]"), smiles.replace("*", "")]
    for v in variants:
        mol = Chem.MolFromSmiles(v)
        if mol is not None:
            return mol
    return None


def _calc_descriptor_row(smiles: str) -> Dict[str, float]:
    mol = _mol_from_polymer_smiles(smiles)
    if mol is None:
        return {"is_valid_mol": 0, **{k: np.nan for k in DESCRIPTOR_COLUMNS}}

    heavy = float(max(Descriptors.HeavyAtomCount(mol), 1))
    hetero = float(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in {1, 6}))
    out = {
        "is_valid_mol": 1,
        "mol_wt": float(Descriptors.MolWt(mol)),
        "logp": float(Descriptors.MolLogP(mol)),
        "tpsa": float(Descriptors.TPSA(mol)),
        "hbd": float(Lipinski.NumHDonors(mol)),
        "hba": float(Lipinski.NumHAcceptors(mol)),
        "rot_bonds": float(Lipinski.NumRotatableBonds(mol)),
        "ring_count": float(rdMolDescriptors.CalcNumRings(mol)),
        "aromatic_rings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "hetero_fraction": float(hetero / heavy),
        "frac_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
    }
    return out


def _attach_descriptors(df: pd.DataFrame, smiles_col: str = "SMILES") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=list(df.columns) + ["is_valid_mol", *DESCRIPTOR_COLUMNS])
    rows: List[Dict[str, float]] = []
    for smi in df[smiles_col].astype(str).tolist():
        rows.append(_calc_descriptor_row(smi))
    desc = pd.DataFrame(rows)
    return pd.concat([df.reset_index(drop=True), desc.reset_index(drop=True)], axis=1)


def _compare_descriptor_shift(selected_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for col in DESCRIPTOR_COLUMNS:
        s = pd.to_numeric(selected_df[col], errors="coerce").dropna().to_numpy(dtype=float)
        b = pd.to_numeric(baseline_df[col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(s) == 0 or len(b) == 0:
            rows.append(
                {
                    "descriptor": col,
                    "n_selected": int(len(s)),
                    "n_baseline": int(len(b)),
                    "selected_mean": np.nan,
                    "baseline_mean": np.nan,
                    "mean_diff": np.nan,
                    "z_shift_vs_baseline": np.nan,
                    "cohens_d_selected_minus_baseline": np.nan,
                    "mannwhitney_pvalue": np.nan,
                }
            )
            continue
        b_std = float(np.std(b, ddof=0))
        z_shift = (float(np.mean(s)) - float(np.mean(b))) / b_std if not np.isclose(b_std, 0.0) else np.nan
        rows.append(
            {
                "descriptor": col,
                "n_selected": int(len(s)),
                "n_baseline": int(len(b)),
                "selected_mean": float(np.mean(s)),
                "baseline_mean": float(np.mean(b)),
                "mean_diff": float(np.mean(s) - np.mean(b)),
                "z_shift_vs_baseline": float(z_shift) if np.isfinite(z_shift) else np.nan,
                "cohens_d_selected_minus_baseline": _cohens_d(s, b),
                "mannwhitney_pvalue": _safe_mannwhitney(s, b),
            }
        )
    out = pd.DataFrame(rows)
    out["abs_z_shift"] = pd.to_numeric(out["z_shift_vs_baseline"], errors="coerce").abs()
    out = out.sort_values("abs_z_shift", ascending=False).reset_index(drop=True)
    return out


def _plot_descriptor_shift(shift_df: pd.DataFrame, out_png: Path, dpi: int, top_k: int = 8) -> None:
    if shift_df.empty:
        return
    sub = shift_df.head(int(max(1, top_k))).copy()
    sub = sub.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    colors = ["#1f77b4" if v >= 0 else "#d62728" for v in sub["z_shift_vs_baseline"].fillna(0.0)]
    ax.barh(sub["descriptor"], sub["z_shift_vs_baseline"], color=colors, alpha=0.85)
    ax.axvline(0.0, color="black", linewidth=1.2)
    ax.set_xlabel("Shift (selected mean - baseline mean) / baseline std")
    ax.set_ylabel("Descriptor")
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def _load_training_baseline_smiles(path: Path, max_samples: int, seed: int) -> pd.DataFrame:
    raw = _normalize_columns(pd.read_csv(path))
    smiles_col = "p_smiles" if "p_smiles" in raw.columns else ("SMILES" if "SMILES" in raw.columns else "smiles")
    if smiles_col not in raw.columns:
        raise ValueError(f"Training baseline CSV missing smiles column: {path}")
    out = pd.DataFrame({"SMILES": raw[smiles_col].astype(str)})
    out["canonical_smiles"] = out["SMILES"].map(canonicalize_smiles)
    out = out[out["canonical_smiles"].notna()].drop_duplicates("canonical_smiles").reset_index(drop=True)
    if len(out) > int(max_samples):
        out = out.sample(n=int(max_samples), random_state=int(seed)).reset_index(drop=True)
    return out


def _compute_max_similarity_to_baseline(
    selected_smiles: List[str],
    baseline_smiles: List[str],
    novelty_threshold: float,
) -> pd.DataFrame:
    if not selected_smiles:
        return pd.DataFrame(columns=["SMILES", "max_tanimoto_to_training", "is_novel_under_threshold"])
    baseline_fps = []
    baseline_kept = []
    for smi in baseline_smiles:
        fp = compute_fingerprint(smi, fp_type="morgan", radius=2, n_bits=2048)
        if fp is not None:
            baseline_fps.append(fp)
            baseline_kept.append(smi)
    if not baseline_fps:
        return pd.DataFrame(
            {
                "SMILES": selected_smiles,
                "max_tanimoto_to_training": [np.nan] * len(selected_smiles),
                "is_novel_under_threshold": [np.nan] * len(selected_smiles),
            }
        )

    rows = []
    for smi in selected_smiles:
        fp = compute_fingerprint(smi, fp_type="morgan", radius=2, n_bits=2048)
        if fp is None:
            rows.append(
                {
                    "SMILES": smi,
                    "max_tanimoto_to_training": np.nan,
                    "is_novel_under_threshold": np.nan,
                }
            )
            continue
        sims = [compute_tanimoto_similarity(fp, bfp) for bfp in baseline_fps]
        max_sim = float(np.max(sims)) if sims else np.nan
        rows.append(
            {
                "SMILES": smi,
                "max_tanimoto_to_training": max_sim,
                "is_novel_under_threshold": float(max_sim < float(novelty_threshold)) if np.isfinite(max_sim) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _plot_novelty_histogram(df: pd.DataFrame, out_png: Path, dpi: int) -> None:
    if df.empty:
        return
    x = pd.to_numeric(df["max_tanimoto_to_training"], errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.hist(x, bins=20, color="#4c78a8", alpha=0.85, edgecolor="white")
    ax.set_xlabel("Max Tanimoto similarity to training set")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def _plot_selection_tradeoff(df: pd.DataFrame, out_png: Path, dpi: int) -> None:
    required = {"chi_pred_target"}
    if df.empty or not required.issubset(df.columns):
        return

    plot_df = df.copy()
    if "class_prob_lcb" in plot_df.columns:
        y_col = "class_prob_lcb"
    elif "class_prob" in plot_df.columns:
        y_col = "class_prob"
    else:
        return
    if "source_step" not in plot_df.columns:
        plot_df["source_step"] = "unknown"

    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    sns.scatterplot(
        data=plot_df,
        x="chi_pred_target",
        y=y_col,
        hue="source_step",
        style="source_step",
        alpha=0.80,
        s=48,
        ax=ax,
    )
    ax.set_xlabel("Predicted χ at target condition")
    ax.set_ylabel("Conservative soluble confidence" if y_col == "class_prob_lcb" else "Soluble probability")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def _load_target_polymers(path: Optional[Path], source_step: str) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    df = _normalize_columns(pd.read_csv(path))
    if "SMILES" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["source_step"] = source_step
    if "canonical_smiles" not in out.columns:
        out["canonical_smiles"] = out["SMILES"].astype(str).map(canonicalize_smiles)
    return out


def _default_step7_config(config: Dict) -> Dict:
    chi_cfg = config.get("chi_training", {})
    shared = chi_cfg.get("shared", {}) if isinstance(chi_cfg.get("shared", {}), dict) else {}
    step7_cfg = chi_cfg.get("step7_chem_physics", {})
    if not isinstance(step7_cfg, dict):
        step7_cfg = {}
    out = {
        "split_mode": str(shared.get("split_mode", "polymer")),
        "chi_dataset_path": str(shared.get("dataset_path", "Data/chi/_250_polymers_T_phi.csv")),
        "baseline_max_samples": int(step7_cfg.get("baseline_max_samples", 3000)),
        "novelty_similarity_threshold": float(step7_cfg.get("novelty_similarity_threshold", 0.40)),
        "descriptor_top_k": int(step7_cfg.get("descriptor_top_k", 8)),
    }
    return out


def _load_step4_predictions(path: Path) -> pd.DataFrame:
    df = _normalize_columns(pd.read_csv(path))
    required = {"chi", "chi_pred", "temperature", "phi"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Step4 predictions missing columns {sorted(missing)}: {path}")
    if "polymer_id" not in df.columns:
        df["polymer_id"] = np.arange(len(df), dtype=int)
    return df


def _write_science_highlights(summary: Dict[str, object], out_md: Path) -> None:
    lines = [
        "# Step 7 Science Highlights",
        "",
        f"- Model size: `{summary.get('model_size', '')}`",
        f"- Split mode: `{summary.get('split_mode', '')}`",
        f"- Completed analysis blocks: `{summary.get('analysis_blocks_completed', 0)}`",
        f"- Numbered figures generated: `{summary.get('n_numbered_figures', 0)}`",
        "",
        "## Pipeline Inputs (Step1-6)",
        f"- Step1 best validation BPB: `{summary.get('step1_best_val_bpb', np.nan):.4f}`",
        f"- Step2 validity: `{summary.get('step2_validity', np.nan):.4f}`",
        f"- Step2 novelty: `{summary.get('step2_novelty', np.nan):.4f}`",
        f"- Step2 target selection success: `{summary.get('step2_target_selection_success_rate', np.nan):.4f}`",
        f"- Step3 global χ_target: `{summary.get('step3_global_chi_target', np.nan):.4f}`",
        f"- Step4 test R2: `{summary.get('step4_test_r2', np.nan):.4f}`",
        f"- Step4 test balanced accuracy: `{summary.get('step4_test_balanced_accuracy', np.nan):.4f}`",
        f"- Step5 target selection success: `{summary.get('step5_target_selection_success_rate', np.nan):.4f}`",
        f"- Step6 target selection success: `{summary.get('step6_target_selection_success_rate', np.nan):.4f}`",
        "",
        "## Thermodynamics",
        f"- Significant χ class-separation conditions (p < 0.05): `{summary.get('n_significant_conditions', 0)}` / `{summary.get('n_conditions', 0)}`",
        f"- Mean Δχ (miscible - immiscible): `{summary.get('mean_delta_chi_miscible_minus_immiscible', np.nan):.4f}`",
        "",
        "## Step 4 Physics Consistency",
        f"- Step4 test MAE(χ): `{summary.get('step4_test_mae', np.nan):.4f}`",
        f"- dχ/dT sign agreement: `{summary.get('step4_dchi_dT_sign_agreement', np.nan):.4f}`",
        f"- dχ/dϕ sign agreement: `{summary.get('step4_dchi_dphi_sign_agreement', np.nan):.4f}`",
        "",
        "## Coefficient Physics",
        f"- Polymers with coefficients: `{int(summary.get('n_polymers_with_coefficients', 0))}`",
        f"- LCST-like fraction (soluble): `{summary.get('frac_lcst_like_soluble', np.nan):.4f}`",
        f"- LCST-like fraction (insoluble): `{summary.get('frac_lcst_like_insoluble', np.nan):.4f}`",
        "",
        "## Flory-Huggins Phase Analysis",
        f"- Mean fraction below spinodal (soluble): `{summary.get('mean_spinodal_miscible_fraction_soluble', np.nan):.4f}`",
        f"- Mean fraction below spinodal (insoluble): `{summary.get('mean_spinodal_miscible_fraction_insoluble', np.nan):.4f}`",
        "",
        "## Structure-Property",
        f"- Functional-group types analyzed: `{int(summary.get('n_functional_group_types_analyzed', 0))}`",
        "",
        "## Classification Context",
        f"- Classification polymers analyzed: `{int(summary.get('n_classification_polymers', 0))}`",
        f"- Fraction of chi polymers covered by classification set: `{summary.get('chi_coverage_in_classification', np.nan):.4f}`",
        "",
        "## Step1 Embedding Physics",
        f"- Embedding polymers analyzed: `{int(summary.get('n_step1_embedding_polymers', 0))}`",
        f"- Embedding CV-AUC for miscibility: `{summary.get('step1_embedding_cv_auc', np.nan):.4f}`",
        f"- Label disagreements (chi vs classification overlap): `{int(summary.get('embedding_label_disagreement_count', 0))}`",
        "",
        "## PINN Coefficient Importance",
        f"- Dominant sensitivity term (soluble): `{summary.get('pinn_top_driver_soluble', '')}`",
        f"- Dominant sensitivity term (insoluble): `{summary.get('pinn_top_driver_insoluble', '')}`",
        "",
        "## Discovered vs Known",
        f"- Step5 target selection success: `{summary.get('step5_target_selection_success_rate', np.nan):.4f}`",
        f"- Step6 target selection success: `{summary.get('step6_target_selection_success_rate', np.nan):.4f}`",
        "",
        "## Chemistry and Novelty",
        f"- Unique selected candidates analyzed: `{summary.get('n_selected_unique_candidates', 0)}`",
        f"- Novel candidates (max Tanimoto < threshold): `{summary.get('novel_fraction_under_threshold', np.nan):.4f}`",
        f"- Mean max similarity to training: `{summary.get('mean_max_tanimoto_to_training', np.nan):.4f}`",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")


def _write_figure_index(figures_dir: Path, metrics_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for old_alias in sorted(figures_dir.glob("figure??_*.png")):
        old_alias.unlink()
    figure_no = 1
    for file_name, block_code, title in FIGURE_ORDER:
        src = figures_dir / file_name
        if not src.exists():
            continue
        alias_name = f"figure{figure_no:02d}_{file_name}"
        alias_path = figures_dir / alias_name
        if alias_path.exists():
            alias_path.unlink()
        shutil.copy2(src, alias_path)
        rows.append(
            {
                "figure_no": int(figure_no),
                "figure_id": f"Figure {figure_no}",
                "block": block_code,
                "title": title,
                "file_name": file_name,
                "alias_file": alias_name,
            }
        )
        figure_no += 1

    index_df = pd.DataFrame(rows)
    index_df.to_csv(metrics_dir / "figure_index.csv", index=False)
    md_lines = ["# Figure Index", ""]
    for row in rows:
        md_lines.append(f"- Figure {int(row['figure_no'])}: `{row['file_name']}` ({row['title']})")
    (metrics_dir / "figure_index.md").write_text("\n".join(md_lines).rstrip() + "\n", encoding="utf-8")
    return index_df


def _read_summary_row(path: Optional[Path]) -> Dict[str, object]:
    if path is None or not path.exists():
        return {}
    try:
        df = _normalize_columns(pd.read_csv(path))
    except Exception:
        return {}
    if df.empty:
        return {}
    row = df.iloc[0].to_dict()
    return {str(k): v for k, v in row.items()}


def _pick_numeric(row: Dict[str, object], keys: List[str]) -> float:
    for key in keys:
        if key in row:
            val = pd.to_numeric(pd.Series([row[key]]), errors="coerce").iloc[0]
            if pd.notna(val):
                return float(val)
    return float(np.nan)

def _plot_pipeline_success_rates(df: pd.DataFrame, out_png: Path, dpi: int) -> None:
    if df.empty:
        return
    plot_df = df.copy()
    plot_df = plot_df[np.isfinite(pd.to_numeric(plot_df["success_rate"], errors="coerce"))].copy()
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    sns.barplot(data=plot_df, x="step", y="success_rate", color="#4c78a8", ax=ax)
    ax.set_xlabel("Step")
    ax.set_ylabel("Selection success rate")
    ax.set_ylim(0.0, 1.0)
    for i, v in enumerate(plot_df["success_rate"].to_list()):
        ax.text(i, min(v + 0.03, 0.98), f"{v:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def _plot_step3_target_vs_class_means(df: pd.DataFrame, out_png: Path, dpi: int) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.scatter(
        df["chi_miscible_mean"],
        df["target_chi"],
        s=42,
        alpha=0.80,
        color="#1f77b4",
        label="Target vs miscible mean",
    )
    ax.scatter(
        df["chi_immiscible_mean"],
        df["target_chi"],
        s=42,
        alpha=0.80,
        color="#d62728",
        label="Target vs immiscible mean",
    )
    all_vals = pd.concat(
        [
            pd.to_numeric(df["chi_miscible_mean"], errors="coerce"),
            pd.to_numeric(df["chi_immiscible_mean"], errors="coerce"),
            pd.to_numeric(df["target_chi"], errors="coerce"),
        ],
        ignore_index=True,
    ).dropna()
    if len(all_vals) > 0:
        lo = float(all_vals.min())
        hi = float(all_vals.max())
        pad = max((hi - lo) * 0.05, 1e-3)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", color="black", linewidth=1.2)
    ax.set_xlabel("Class mean χ")
    ax.set_ylabel("Step3 target χ")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def _resolve_smiles_column(df: pd.DataFrame) -> Optional[str]:
    for c in ["SMILES", "smiles", "p_smiles"]:
        if c in df.columns:
            return c
    return None


def _chi_formula_for_fit(Xdata: np.ndarray, a0: float, a1: float, a2: float, a3: float, b1: float, b2: float) -> np.ndarray:
    temperature = np.asarray(Xdata[0], dtype=float)
    phi = np.asarray(Xdata[1], dtype=float)
    base = a0 + a1 / temperature + a2 * np.log(temperature) + a3 * temperature
    one_minus_phi = 1.0 - phi
    modifier = 1.0 + b1 * one_minus_phi + b2 * (one_minus_phi ** 2)
    return base * modifier


def _fit_polymer_coefficients_from_data(chi_df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_columns(chi_df).copy()
    smiles_col = _resolve_smiles_column(df)
    if smiles_col is None:
        raise ValueError("Coefficient fitting requires a smiles column in chi dataset.")
    df[smiles_col] = df[smiles_col].astype(str)
    df["canonical_smiles"] = df[smiles_col].map(canonicalize_smiles)
    df["canonical_smiles"] = df["canonical_smiles"].where(df["canonical_smiles"].notna(), df[smiles_col])

    if "polymer_id" in df.columns:
        group_key = "polymer_id"
    elif "Polymer" in df.columns:
        group_key = "Polymer"
    else:
        group_key = "canonical_smiles"

    rows: List[Dict[str, object]] = []
    lower_bounds = np.array([-10.0, -5.0e4, -10.0, -1.0, -10.0, -10.0], dtype=float)
    upper_bounds = np.array([10.0, 5.0e4, 10.0, 1.0, 10.0, 10.0], dtype=float)
    p0_base = np.array([0.5, 100.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

    for key, sub in df.groupby(group_key, sort=False):
        sub = sub.dropna(subset=["temperature", "phi", "chi"]).copy()
        sub["temperature"] = pd.to_numeric(sub["temperature"], errors="coerce")
        sub["phi"] = pd.to_numeric(sub["phi"], errors="coerce")
        sub["chi"] = pd.to_numeric(sub["chi"], errors="coerce")
        sub = sub.dropna(subset=["temperature", "phi", "chi"]).reset_index(drop=True)
        n_points = int(len(sub))
        base_row: Dict[str, object] = {
            "polymer_id": int(sub["polymer_id"].iloc[0]) if "polymer_id" in sub.columns else np.nan,
            "Polymer": str(sub["Polymer"].iloc[0]) if "Polymer" in sub.columns else f"polymer_{key}",
            "SMILES": str(sub[smiles_col].iloc[0]),
            "canonical_smiles": str(sub["canonical_smiles"].iloc[0]),
            CLASS_COL: int(pd.to_numeric(sub.get(CLASS_COL, pd.Series([0])), errors="coerce").fillna(0).iloc[0]),
            "fit_n_points": n_points,
        }
        if n_points < 8:
            rows.append(
                {
                    **base_row,
                    **{c: np.nan for c in COEFF_COLUMNS},
                    "fit_r2": np.nan,
                    "fit_rmse": np.nan,
                    "coeff_source": "fit_from_data_failed",
                }
            )
            continue

        xdata = np.vstack(
            [
                sub["temperature"].to_numpy(dtype=float),
                sub["phi"].to_numpy(dtype=float),
            ]
        )
        ydata = sub["chi"].to_numpy(dtype=float)
        best = None
        for trial in range(3):
            p0 = p0_base.copy()
            if trial > 0:
                rng = np.random.default_rng(2026 + trial)
                p0 += rng.normal(loc=0.0, scale=[0.2, 50.0, 0.2, 0.02, 0.2, 0.2], size=6)
                p0 = np.clip(p0, lower_bounds + 1e-6, upper_bounds - 1e-6)
            try:
                popt, _ = curve_fit(
                    _chi_formula_for_fit,
                    xdata=xdata,
                    ydata=ydata,
                    p0=p0,
                    bounds=(lower_bounds, upper_bounds),
                    maxfev=30000,
                )
                yhat = _chi_formula_for_fit(xdata, *popt)
                sse = float(np.sum((ydata - yhat) ** 2))
                sst = float(np.sum((ydata - np.mean(ydata)) ** 2))
                r2 = float(1.0 - sse / sst) if sst > 0 else np.nan
                rmse = float(np.sqrt(np.mean((ydata - yhat) ** 2)))
                score = (np.nan_to_num(r2, nan=-999.0), -rmse)
                if best is None or score > best["score"]:
                    best = {"popt": popt, "r2": r2, "rmse": rmse, "score": score}
            except Exception:
                continue

        if best is None:
            rows.append(
                {
                    **base_row,
                    **{c: np.nan for c in COEFF_COLUMNS},
                    "fit_r2": np.nan,
                    "fit_rmse": np.nan,
                    "coeff_source": "fit_from_data_failed",
                }
            )
            continue

        rows.append(
            {
                **base_row,
                **{c: float(best["popt"][i]) for i, c in enumerate(COEFF_COLUMNS)},
                "fit_r2": float(best["r2"]),
                "fit_rmse": float(best["rmse"]),
                "coeff_source": "fit_from_data",
            }
        )

    out = pd.DataFrame(rows)
    if "polymer_id" in out.columns:
        out["polymer_id"] = pd.to_numeric(out["polymer_id"], errors="coerce")
    return out


def _load_or_fit_coefficients(coeff_path: Optional[Path], chi_df: pd.DataFrame) -> pd.DataFrame:
    if coeff_path is not None and coeff_path.exists():
        coeff_df = _normalize_columns(pd.read_csv(coeff_path))
        missing_coeff = [c for c in COEFF_COLUMNS if c not in coeff_df.columns]
        if missing_coeff:
            raise ValueError(f"Coefficient file missing required columns {missing_coeff}: {coeff_path}")
        smiles_col = _resolve_smiles_column(coeff_df)
        if smiles_col is None:
            raise ValueError(f"Coefficient file missing smiles column: {coeff_path}")
        coeff_df = coeff_df.copy()
        coeff_df["SMILES"] = coeff_df[smiles_col].astype(str)
        coeff_df["canonical_smiles"] = coeff_df["SMILES"].map(canonicalize_smiles)
        coeff_df["canonical_smiles"] = coeff_df["canonical_smiles"].where(
            coeff_df["canonical_smiles"].notna(), coeff_df["SMILES"]
        )
        coeff_df["coeff_source"] = "step4_coefficients"
        if "fit_r2" not in coeff_df.columns:
            coeff_df["fit_r2"] = np.nan
        if "fit_rmse" not in coeff_df.columns:
            coeff_df["fit_rmse"] = np.nan
        if CLASS_COL not in coeff_df.columns:
            chi_ref = _ensure_class_col(_normalize_columns(chi_df))
            smiles_ref = _resolve_smiles_column(chi_ref)
            if smiles_ref is None:
                coeff_df[CLASS_COL] = np.nan
            else:
                tmp = chi_ref[[smiles_ref, CLASS_COL]].copy()
                tmp["canonical_smiles"] = tmp[smiles_ref].astype(str).map(canonicalize_smiles)
                tmp = (
                    tmp.dropna(subset=["canonical_smiles"])
                    .groupby("canonical_smiles", as_index=False)[CLASS_COL]
                    .agg(lambda s: int(round(float(np.mean(pd.to_numeric(s, errors="coerce").fillna(0))))))
                )
                coeff_df = coeff_df.merge(tmp, on="canonical_smiles", how="left")
        if "Polymer" not in coeff_df.columns:
            coeff_df["Polymer"] = [f"polymer_{i+1}" for i in range(len(coeff_df))]
        if "polymer_id" not in coeff_df.columns:
            coeff_df["polymer_id"] = np.arange(len(coeff_df), dtype=int)
        return coeff_df
    return _fit_polymer_coefficients_from_data(chi_df)


def _compute_dchi_dT(row: pd.Series, T: float, phi: float) -> float:
    a1 = _safe_float(row.get("a1", np.nan))
    a2 = _safe_float(row.get("a2", np.nan))
    a3 = _safe_float(row.get("a3", np.nan))
    b1 = _safe_float(row.get("b1", np.nan))
    b2 = _safe_float(row.get("b2", np.nan))
    if not np.isfinite([a1, a2, a3, b1, b2]).all():
        return np.nan
    one_minus_phi = 1.0 - float(phi)
    modifier = 1.0 + b1 * one_minus_phi + b2 * (one_minus_phi ** 2)
    return float((-a1 / (float(T) ** 2) + a2 / float(T) + a3) * modifier)


def _classify_chi_response(dchi_dT: float, tol: float = 1e-4) -> str:
    if not np.isfinite(dchi_dT):
        return "unknown"
    if dchi_dT > float(tol):
        return "LCST-like"
    if dchi_dT < -float(tol):
        return "UCST-like"
    return "flat"


def _coefficient_class_statistics(coeff_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for c in COEFF_COLUMNS:
        s0 = pd.to_numeric(coeff_df.loc[coeff_df[CLASS_COL] == 0, c], errors="coerce").dropna().to_numpy(dtype=float)
        s1 = pd.to_numeric(coeff_df.loc[coeff_df[CLASS_COL] == 1, c], errors="coerce").dropna().to_numpy(dtype=float)
        rows.append(
            {
                "coefficient": c,
                "n_insoluble": int(len(s0)),
                "n_soluble": int(len(s1)),
                "mean_insoluble": float(np.mean(s0)) if len(s0) else np.nan,
                "std_insoluble": float(np.std(s0, ddof=0)) if len(s0) else np.nan,
                "median_insoluble": float(np.median(s0)) if len(s0) else np.nan,
                "mean_soluble": float(np.mean(s1)) if len(s1) else np.nan,
                "std_soluble": float(np.std(s1, ddof=0)) if len(s1) else np.nan,
                "median_soluble": float(np.median(s1)) if len(s1) else np.nan,
                "delta_mean_soluble_minus_insoluble": float(np.mean(s1) - np.mean(s0))
                if (len(s0) and len(s1))
                else np.nan,
                "mannwhitney_pvalue": _safe_mannwhitney(s1, s0),
            }
        )
    return pd.DataFrame(rows)


def _plot_coefficient_violins(coeff_df: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    if coeff_df.empty:
        return
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.8))
    class_label = coeff_df[CLASS_COL].map({1: "soluble", 0: "insoluble"}).fillna("unknown")
    plot_df = coeff_df.copy()
    plot_df["class_label"] = class_label
    for ax, col in zip(axes.flat, COEFF_COLUMNS):
        sub = plot_df[["class_label", col]].copy()
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
        sub = sub.dropna(subset=[col])
        if sub.empty:
            ax.set_axis_off()
            continue
        sns.violinplot(data=sub, x="class_label", y=col, ax=ax, palette=["#d62728", "#1f77b4"])
        ax.set_xlabel("")
        ax.set_ylabel(col)
    fig.tight_layout()
    fig.savefig(figures_dir / "coefficient_violin_by_class.png", dpi=dpi)
    plt.close(fig)


def _plot_coeff_scatter(
    coeff_df: pd.DataFrame,
    col_x: str,
    col_y: str,
    figures_dir: Path,
    out_name: str,
    dpi: int,
) -> None:
    if coeff_df.empty or col_x not in coeff_df.columns or col_y not in coeff_df.columns:
        return
    plot_df = coeff_df.copy()
    plot_df[col_x] = pd.to_numeric(plot_df[col_x], errors="coerce")
    plot_df[col_y] = pd.to_numeric(plot_df[col_y], errors="coerce")
    plot_df = plot_df.dropna(subset=[col_x, col_y])
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sns.scatterplot(
        data=plot_df,
        x=col_x,
        y=col_y,
        hue=CLASS_COL,
        palette={0: "#d62728", 1: "#1f77b4"},
        alpha=0.80,
        s=42,
        ax=ax,
    )
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)
    fig.tight_layout()
    fig.savefig(figures_dir / out_name, dpi=dpi)
    plt.close(fig)


def _plot_dchi_dT_distribution(coeff_df: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    if coeff_df.empty or "dchi_dT_ref" not in coeff_df.columns:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for cls, color, label in [(1, "#1f77b4", "soluble"), (0, "#d62728", "insoluble")]:
        vals = pd.to_numeric(
            coeff_df.loc[coeff_df[CLASS_COL] == cls, "dchi_dT_ref"],
            errors="coerce",
        ).dropna()
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=20, alpha=0.45, color=color, label=label, edgecolor="white")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.1)
    ax.set_xlabel("dχ/dT at T=293.15K, φ=0.2")
    ax.set_ylabel("Count")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(figures_dir / "dchi_dT_distribution_by_class.png", dpi=dpi)
    plt.close(fig)


def _chi_spinodal(phi: float) -> float:
    phi_val = float(phi)
    if phi_val <= 0.0 or phi_val >= 1.0:
        return float(np.nan)
    return float(1.0 / (2.0 * phi_val * (1.0 - phi_val)))


def _compute_chi_surfaces(coeff_df: pd.DataFrame, T_grid: np.ndarray, phi_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    T_mesh, phi_mesh = np.meshgrid(T_grid, phi_grid, indexing="ij")

    def _surface_for_class(class_value: int) -> np.ndarray:
        sub = coeff_df[coeff_df[CLASS_COL] == class_value].copy()
        if sub.empty:
            return np.full_like(T_mesh, np.nan, dtype=float)
        mats = []
        for _, row in sub.iterrows():
            coeff = np.array([_safe_float(row[c]) for c in COEFF_COLUMNS], dtype=float)
            if not np.isfinite(coeff).all():
                continue
            mats.append(predict_chi_from_coefficients(coeff, T_mesh, phi_mesh))
        if not mats:
            return np.full_like(T_mesh, np.nan, dtype=float)
        return np.nanmean(np.stack(mats, axis=0), axis=0)

    return _surface_for_class(1), _surface_for_class(0)


def _plot_chi_surface_heatmap(
    chi_soluble_surface: np.ndarray,
    chi_insoluble_surface: np.ndarray,
    T_grid: np.ndarray,
    phi_grid: np.ndarray,
    figures_dir: Path,
    dpi: int,
) -> None:
    if chi_soluble_surface.size == 0 or chi_insoluble_surface.size == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), sharey=True)
    vmin = np.nanmin([np.nanmin(chi_soluble_surface), np.nanmin(chi_insoluble_surface)])
    vmax = np.nanmax([np.nanmax(chi_soluble_surface), np.nanmax(chi_insoluble_surface)])
    for ax, mat, title in [
        (axes[0], chi_soluble_surface, "Soluble mean χ(T,φ)"),
        (axes[1], chi_insoluble_surface, "Insoluble mean χ(T,φ)"),
    ]:
        if np.isnan(mat).all():
            ax.set_axis_off()
            continue
        sns.heatmap(
            pd.DataFrame(mat, index=T_grid, columns=phi_grid),
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            cbar=(ax is axes[1]),
            cbar_kws={"label": "χ"},
        )
        ax.set_xlabel("φ")
        ax.set_ylabel("Temperature (K)")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(figures_dir / "chi_surface_mean_by_class.png", dpi=dpi)
    plt.close(fig)


def _compute_spinodal_analysis(
    coeff_df: pd.DataFrame,
    chi_df: pd.DataFrame,
    chi_target_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    T_vals = sorted(pd.to_numeric(chi_df["temperature"], errors="coerce").dropna().unique().tolist())
    phi_vals = sorted(pd.to_numeric(chi_df["phi"], errors="coerce").dropna().unique().tolist())
    phi_vals = [float(p) for p in phi_vals if (p > 0.0 and p < 1.0)]

    if len(T_vals) == 0 or len(phi_vals) == 0:
        return pd.DataFrame(), pd.DataFrame()

    spinodal_rows: List[Dict[str, object]] = []
    freeE_rows: List[Dict[str, object]] = []
    targets = chi_target_df.copy()
    targets["temperature"] = pd.to_numeric(targets["temperature"], errors="coerce")
    targets["phi"] = pd.to_numeric(targets["phi"], errors="coerce")
    targets["target_chi"] = pd.to_numeric(targets["target_chi"], errors="coerce")
    targets = targets.dropna(subset=["temperature", "phi", "target_chi"]).reset_index(drop=True)

    for _, row in coeff_df.iterrows():
        coeff = np.array([_safe_float(row[c]) for c in COEFF_COLUMNS], dtype=float)
        if not np.isfinite(coeff).all():
            continue
        cond_hits = []
        chi_vals = []
        spin_vals = []
        for t in T_vals:
            for p in phi_vals:
                chi_val = float(predict_chi_from_coefficients(coeff, np.array(t), np.array(p)))
                spin = _chi_spinodal(p)
                chi_vals.append(chi_val)
                spin_vals.append(spin)
                cond_hits.append(float(chi_val < spin) if np.isfinite(spin) else np.nan)
        cond_hits_arr = np.asarray(cond_hits, dtype=float)

        target_hit_flags = []
        for _, tr in targets.iterrows():
            t = float(tr["temperature"])
            p = float(tr["phi"])
            chi_pred = float(predict_chi_from_coefficients(coeff, np.array(t), np.array(p)))
            target_hit = int(chi_pred <= float(tr["target_chi"]))
            target_hit_flags.append(target_hit)
            p_clip = float(np.clip(p, 1e-6, 1 - 1e-6))
            delta_g = float((1.0 - p_clip) * np.log(1.0 - p_clip) + chi_pred * p_clip * (1.0 - p_clip))
            freeE_rows.append(
                {
                    "polymer_id": row.get("polymer_id", np.nan),
                    "Polymer": row.get("Polymer", ""),
                    "SMILES": row.get("SMILES", ""),
                    CLASS_COL: int(_safe_float(row.get(CLASS_COL, np.nan), default=np.nan))
                    if np.isfinite(_safe_float(row.get(CLASS_COL, np.nan), default=np.nan))
                    else np.nan,
                    "temperature": t,
                    "phi": p,
                    "target_chi": float(tr["target_chi"]),
                    "chi_pred_target_condition": chi_pred,
                    "deltaG_mix_over_nRT": delta_g,
                    "chi_target_condition_hit": target_hit,
                }
            )

        spinodal_rows.append(
            {
                "polymer_id": row.get("polymer_id", np.nan),
                "Polymer": row.get("Polymer", ""),
                "SMILES": row.get("SMILES", ""),
                CLASS_COL: int(_safe_float(row.get(CLASS_COL, np.nan), default=np.nan))
                if np.isfinite(_safe_float(row.get(CLASS_COL, np.nan), default=np.nan))
                else np.nan,
                "n_conditions": int(np.isfinite(cond_hits_arr).sum()),
                "fraction_below_spinodal": float(np.nanmean(cond_hits_arr)) if np.isfinite(cond_hits_arr).any() else np.nan,
                "mean_pred_chi_over_grid": float(np.nanmean(np.asarray(chi_vals, dtype=float))),
                "mean_spinodal_over_grid": float(np.nanmean(np.asarray(spin_vals, dtype=float))),
                "chi_target_condition_flag": int(np.all(np.asarray(target_hit_flags, dtype=int) == 1))
                if target_hit_flags
                else np.nan,
            }
        )
    return pd.DataFrame(spinodal_rows), pd.DataFrame(freeE_rows)


def _plot_spinodal_diagram(
    coeff_df: pd.DataFrame,
    chi_target_df: pd.DataFrame,
    figures_dir: Path,
    dpi: int,
    T_ref: float = 293.15,
) -> None:
    if coeff_df.empty:
        return
    phi_line = np.linspace(0.02, 0.98, 200)
    spin_line = np.array([_chi_spinodal(p) for p in phi_line], dtype=float)

    def _mean_class_curve(cls: int) -> np.ndarray:
        mats = []
        sub = coeff_df[coeff_df[CLASS_COL] == cls]
        for _, row in sub.iterrows():
            coeff = np.array([_safe_float(row[c]) for c in COEFF_COLUMNS], dtype=float)
            if not np.isfinite(coeff).all():
                continue
            mats.append(predict_chi_from_coefficients(coeff, np.full_like(phi_line, float(T_ref)), phi_line))
        if not mats:
            return np.full_like(phi_line, np.nan, dtype=float)
        return np.nanmean(np.asarray(mats, dtype=float), axis=0)

    mean_sol = _mean_class_curve(1)
    mean_ins = _mean_class_curve(0)

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.plot(phi_line, spin_line, color="black", linestyle="--", linewidth=1.7, label="Spinodal χs(φ)")
    if np.isfinite(mean_sol).any():
        ax.plot(phi_line, mean_sol, color="#1f77b4", linewidth=2.0, label=f"Soluble mean χ(φ), T={T_ref:g}K")
    if np.isfinite(mean_ins).any():
        ax.plot(phi_line, mean_ins, color="#d62728", linewidth=2.0, label=f"Insoluble mean χ(φ), T={T_ref:g}K")

    tgt = chi_target_df.copy()
    if {"phi", "target_chi"}.issubset(tgt.columns):
        tgt["phi"] = pd.to_numeric(tgt["phi"], errors="coerce")
        tgt["target_chi"] = pd.to_numeric(tgt["target_chi"], errors="coerce")
        tgt = tgt.dropna(subset=["phi", "target_chi"])
        if not tgt.empty:
            ax.scatter(tgt["phi"], tgt["target_chi"], color="#2ca02c", s=36, alpha=0.9, label="χ_target")

    ax.set_xlabel("φ")
    ax.set_ylabel("χ")
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(figures_dir / "spinodal_phase_diagram.png", dpi=dpi)
    plt.close(fig)


def _plot_miscible_fraction_boxplot(spinodal_df: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    if spinodal_df.empty or "fraction_below_spinodal" not in spinodal_df.columns:
        return
    plot_df = spinodal_df.copy()
    plot_df["class_label"] = plot_df[CLASS_COL].map({1: "soluble", 0: "insoluble"}).fillna("unknown")
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    sns.boxplot(data=plot_df, x="class_label", y="fraction_below_spinodal", ax=ax)
    ax.set_xlabel("Class")
    ax.set_ylabel("Fraction of (T,φ) below spinodal")
    fig.tight_layout()
    fig.savefig(figures_dir / "miscible_fraction_below_spinodal_by_class.png", dpi=dpi)
    plt.close(fig)


def _plot_free_energy_boxplot(freeE_df: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    if freeE_df.empty or "deltaG_mix_over_nRT" not in freeE_df.columns:
        return
    plot_df = freeE_df.copy()
    plot_df["class_label"] = plot_df[CLASS_COL].map({1: "soluble", 0: "insoluble"}).fillna("unknown")
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    sns.boxplot(data=plot_df, x="class_label", y="deltaG_mix_over_nRT", ax=ax)
    ax.set_xlabel("Class")
    ax.set_ylabel("ΔG_mix/(nRT) at target")
    fig.tight_layout()
    fig.savefig(figures_dir / "free_energy_mixing_by_class.png", dpi=dpi)
    plt.close(fig)


def _detect_functional_groups(mol) -> Dict[str, int]:
    if mol is None:
        return {k: 0 for k in FG_COLUMNS}
    patt_hydroxyl = Chem.MolFromSmarts("[OX2H]")
    patt_carboxyl = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
    patt_amine = Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]")
    patt_ether = Chem.MolFromSmarts("[OD2]([#6])[#6]")
    patt_ester = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
    patt_carbonyl = Chem.MolFromSmarts("[CX3]=[OX1]")
    patt_nitrile = Chem.MolFromSmarts("[CX2]#[NX1]")
    patt_halogen = Chem.MolFromSmarts("[F,Cl]")

    ester_matches = mol.GetSubstructMatches(patt_ester)
    carboxyl_matches = mol.GetSubstructMatches(patt_carboxyl)
    ester_carbons = {m[0] for m in ester_matches if len(m) > 0}
    carboxyl_carbons = {m[0] for m in carboxyl_matches if len(m) > 0}

    ether_flag = 0
    for m in mol.GetSubstructMatches(patt_ether):
        if len(m) < 1:
            continue
        o_idx = int(m[0])
        oxygen_atom = mol.GetAtomWithIdx(o_idx)
        is_ester_like = False
        for nbr in oxygen_atom.GetNeighbors():
            if nbr.GetAtomicNum() != 6:
                continue
            for bond in nbr.GetBonds():
                other = bond.GetOtherAtom(nbr)
                if other.GetAtomicNum() == 8 and bond.GetBondTypeAsDouble() == 2.0:
                    is_ester_like = True
                    break
            if is_ester_like:
                break
        if not is_ester_like:
            ether_flag = 1
            break

    carbonyl_flag = 0
    for m in mol.GetSubstructMatches(patt_carbonyl):
        if len(m) < 1:
            continue
        carbon_idx = int(m[0])
        if carbon_idx in ester_carbons or carbon_idx in carboxyl_carbons:
            continue
        carbonyl_flag = 1
        break

    return {
        "fg_hydroxyl": int(mol.HasSubstructMatch(patt_hydroxyl)),
        "fg_carboxyl": int(mol.HasSubstructMatch(patt_carboxyl)),
        "fg_amine": int(mol.HasSubstructMatch(patt_amine)),
        "fg_ether": int(ether_flag),
        "fg_ester": int(mol.HasSubstructMatch(patt_ester)),
        "fg_carbonyl": int(carbonyl_flag),
        "fg_nitrile": int(mol.HasSubstructMatch(patt_nitrile)),
        "fg_halogen": int(mol.HasSubstructMatch(patt_halogen)),
    }


def _build_chi_dataset_descriptor_df(chi_df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_class_col(_normalize_columns(chi_df)).copy()
    smiles_col = _resolve_smiles_column(df)
    if smiles_col is None:
        raise ValueError("Descriptor block requires a smiles column in chi dataset.")
    df[smiles_col] = df[smiles_col].astype(str)
    df["canonical_smiles"] = df[smiles_col].map(canonicalize_smiles)
    df = df[df["canonical_smiles"].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    agg_rows = []
    for canon, sub in df.groupby("canonical_smiles", sort=False):
        smiles = str(sub[smiles_col].iloc[0])
        polymer_name = str(sub["Polymer"].iloc[0]) if "Polymer" in sub.columns else canon
        mean_chi = float(pd.to_numeric(sub["chi"], errors="coerce").mean())
        class_label = int(round(float(pd.to_numeric(sub[CLASS_COL], errors="coerce").mean())))
        desc = _calc_descriptor_row(smiles)
        mol = _mol_from_polymer_smiles(smiles)
        fgs = _detect_functional_groups(mol)
        agg_rows.append(
            {
                "Polymer": polymer_name,
                "SMILES": smiles,
                "canonical_smiles": canon,
                "chi_mean": mean_chi,
                CLASS_COL: class_label,
                **desc,
                **fgs,
            }
        )
    return pd.DataFrame(agg_rows)


def _plot_functional_group_bars(desc_df: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    if desc_df.empty:
        return
    rows = []
    for fg in FG_COLUMNS:
        for cls in [1, 0]:
            vals = pd.to_numeric(desc_df.loc[desc_df[CLASS_COL] == cls, fg], errors="coerce").dropna()
            rows.append(
                {
                    "functional_group": fg.replace("fg_", ""),
                    "class_label": "soluble" if cls == 1 else "insoluble",
                    "fraction_present": float(vals.mean()) if len(vals) else np.nan,
                }
            )
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    sns.barplot(
        data=plot_df,
        y="functional_group",
        x="fraction_present",
        hue="class_label",
        orient="h",
        ax=ax,
    )
    ax.set_xlabel("Fraction present")
    ax.set_ylabel("Functional group")
    fig.tight_layout()
    fig.savefig(figures_dir / "functional_group_frequency_by_class.png", dpi=dpi)
    plt.close(fig)


def _plot_chi_vs_descriptor_scatter(
    desc_df: pd.DataFrame,
    xcol: str,
    xlabel: str,
    out_name: str,
    figures_dir: Path,
    dpi: int,
) -> None:
    if desc_df.empty or xcol not in desc_df.columns:
        return
    plot_df = desc_df.copy()
    plot_df[xcol] = pd.to_numeric(plot_df[xcol], errors="coerce")
    plot_df["chi_mean"] = pd.to_numeric(plot_df["chi_mean"], errors="coerce")
    plot_df = plot_df.dropna(subset=[xcol, "chi_mean"])
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    sns.scatterplot(
        data=plot_df,
        x=xcol,
        y="chi_mean",
        hue=CLASS_COL,
        palette={0: "#d62728", 1: "#1f77b4"},
        s=42,
        alpha=0.8,
        ax=ax,
    )
    sns.regplot(data=plot_df, x=xcol, y="chi_mean", scatter=False, color="black", line_kws={"linewidth": 1.2}, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean χ")
    fig.tight_layout()
    fig.savefig(figures_dir / out_name, dpi=dpi)
    plt.close(fig)


def _plot_descriptor_corr_heatmap(desc_df: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    if desc_df.empty:
        return
    cols = [c for c in DESCRIPTOR_COLUMNS if c in desc_df.columns] + ["chi_mean", CLASS_COL]
    corr_df = desc_df[cols].copy()
    for c in cols:
        corr_df[c] = pd.to_numeric(corr_df[c], errors="coerce")
    corr_mat = corr_df.corr(method="pearson", min_periods=3)
    if corr_mat.empty:
        return
    cg = sns.clustermap(corr_mat, cmap="coolwarm", center=0.0, figsize=(7.2, 7.2))
    cg.savefig(figures_dir / "descriptor_chi_correlation_heatmap.png", dpi=dpi)
    plt.close(cg.fig)


def _load_inverse_candidates(path: Path, source_step: Optional[str] = None) -> pd.DataFrame:
    df = _normalize_columns(pd.read_csv(path))
    smiles_col = _resolve_smiles_column(df)
    if smiles_col is None:
        return pd.DataFrame(columns=["SMILES", "chi_pred_target", "class_prob", "joint_hit", "source_step"])
    out = df.copy()
    out["SMILES"] = out[smiles_col].astype(str)
    if "chi_pred_target" not in out.columns:
        out["chi_pred_target"] = np.nan
    if "class_prob" not in out.columns:
        if "class_prob_lcb" in out.columns:
            out["class_prob"] = out["class_prob_lcb"]
        else:
            out["class_prob"] = np.nan
    if "joint_hit" not in out.columns:
        if {"soluble_hit", "property_hit"}.issubset(out.columns):
            if "polymer_class_hit" in out.columns:
                out["joint_hit"] = (
                    (pd.to_numeric(out["soluble_hit"], errors="coerce").fillna(0).astype(int) == 1)
                    & (pd.to_numeric(out["property_hit"], errors="coerce").fillna(0).astype(int) == 1)
                    & (pd.to_numeric(out["polymer_class_hit"], errors="coerce").fillna(0).astype(int) == 1)
                ).astype(int)
            else:
                out["joint_hit"] = (
                    (pd.to_numeric(out["soluble_hit"], errors="coerce").fillna(0).astype(int) == 1)
                    & (pd.to_numeric(out["property_hit"], errors="coerce").fillna(0).astype(int) == 1)
                ).astype(int)
        else:
            out["joint_hit"] = np.nan
    if source_step is None:
        source_step = "step5" if "step5" in str(path).lower() else ("step6" if "step6" in str(path).lower() else "inverse")
    out["source_step"] = str(source_step)
    keep_cols = ["SMILES", "chi_pred_target", "class_prob", "joint_hit", "source_step"]
    for c in COEFF_COLUMNS:
        if c in out.columns:
            keep_cols.append(c)
    return out[keep_cols].copy()


def _compute_pca_coordinates(smiles_groups_dict: Dict[str, List[str]], n_components: int = 2) -> pd.DataFrame:
    rows = []
    fps = []
    for group, smiles_list in smiles_groups_dict.items():
        if not smiles_list:
            continue
        canon_list = []
        for s in smiles_list:
            c = canonicalize_smiles(str(s))
            if c is not None:
                canon_list.append(c)
        canon_unique = list(dict.fromkeys(canon_list))
        group_fps, idxs = batch_compute_fingerprints(canon_unique, fp_type="morgan", radius=2, n_bits=2048)
        for fp, idx in zip(group_fps, idxs):
            rows.append({"group": group, "SMILES": canon_unique[int(idx)]})
            fps.append(fp.astype(float))
    if not rows or len(fps) < 2:
        return pd.DataFrame(columns=["SMILES", "group", "PC1", "PC2"])
    X = np.vstack(fps)
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X)
    out = pd.DataFrame(rows)
    out["PC1"] = coords[:, 0]
    out["PC2"] = coords[:, 1] if coords.shape[1] > 1 else 0.0
    return out


def _plot_chemical_space_pca(pca_df: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    if pca_df.empty:
        return
    style = {
        "known_soluble": {"color": "#1f77b4", "marker": "o"},
        "known_insoluble": {"color": "#d62728", "marker": "o"},
        "discovered_step5": {"color": "#2ca02c", "marker": "*"},
        "discovered_step6": {"color": "#17a65b", "marker": "*"},
    }
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    for group, sub in pca_df.groupby("group", sort=False):
        s = style.get(group, {"color": "#7f7f7f", "marker": "o"})
        ax.scatter(
            sub["PC1"],
            sub["PC2"],
            color=s["color"],
            marker=s["marker"],
            alpha=0.75,
            s=65 if s["marker"] == "*" else 34,
            label=group,
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(figures_dir / "chemical_space_pca_known_vs_discovered.png", dpi=dpi)
    plt.close(fig)


def _plot_chi_scoring_landscape(inv_cand_df: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    if inv_cand_df.empty:
        return
    plot_df = inv_cand_df.copy()
    plot_df["chi_pred_target"] = pd.to_numeric(plot_df["chi_pred_target"], errors="coerce")
    plot_df["class_prob"] = pd.to_numeric(plot_df["class_prob"], errors="coerce")
    plot_df["joint_hit"] = pd.to_numeric(plot_df["joint_hit"], errors="coerce")
    plot_df = plot_df.dropna(subset=["chi_pred_target", "class_prob"])
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    sns.scatterplot(
        data=plot_df,
        x="chi_pred_target",
        y="class_prob",
        hue="joint_hit",
        palette={0.0: "#9e9e9e", 1.0: "#2ca02c"},
        style="source_step" if "source_step" in plot_df.columns else None,
        alpha=0.80,
        s=38,
        ax=ax,
    )
    ax.set_xlabel("Predicted χ at target")
    ax.set_ylabel("Class probability")
    fig.tight_layout()
    fig.savefig(figures_dir / "chi_vs_class_prob_scoring_landscape.png", dpi=dpi)
    plt.close(fig)


def _plot_descriptor_boxplot_by_class(desc_df: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    if desc_df.empty:
        return
    key_cols = ["mol_wt", "logp", "tpsa", "hbd", "hba"]
    fig, axes = plt.subplots(1, 5, figsize=(16.8, 4.0), sharex=False)
    for ax, col in zip(axes.flat, key_cols):
        sub = desc_df[[CLASS_COL, col]].copy()
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
        sub = sub.dropna(subset=[col])
        if sub.empty:
            ax.set_axis_off()
            continue
        sub["class_label"] = sub[CLASS_COL].map({1: "soluble", 0: "insoluble"})
        sns.boxplot(data=sub, x="class_label", y=col, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel(col)
        ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(figures_dir / "descriptor_boxplot_by_class.png", dpi=dpi)
    plt.close(fig)


def _compute_discovered_descriptor_stats(known_sol: pd.DataFrame, known_ins: pd.DataFrame, discovered: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [
        ("known_soluble", known_sol),
        ("known_insoluble", known_ins),
        ("discovered", discovered),
    ]
    for desc in DESCRIPTOR_COLUMNS:
        for name, gdf in groups:
            vals = pd.to_numeric(gdf.get(desc, pd.Series([], dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
            rows.append(
                {
                    "descriptor": desc,
                    "group": name,
                    "n": int(len(vals)),
                    "mean": float(np.mean(vals)) if len(vals) else np.nan,
                    "std": float(np.std(vals, ddof=0)) if len(vals) else np.nan,
                    "median": float(np.median(vals)) if len(vals) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _plot_discovered_descriptor_boxplot(
    known_sol: pd.DataFrame,
    known_ins: pd.DataFrame,
    discovered: pd.DataFrame,
    figures_dir: Path,
    dpi: int,
) -> None:
    key_cols = ["mol_wt", "logp", "tpsa", "hbd", "hba"]
    all_rows = []
    for group_name, gdf in [("known_soluble", known_sol), ("known_insoluble", known_ins), ("discovered", discovered)]:
        for col in key_cols:
            if col not in gdf.columns:
                continue
            vals = pd.to_numeric(gdf[col], errors="coerce").dropna().to_numpy(dtype=float)
            for v in vals.tolist():
                all_rows.append({"group": group_name, "descriptor": col, "value": float(v)})
    plot_df = pd.DataFrame(all_rows)
    if plot_df.empty:
        return
    fig, axes = plt.subplots(1, 5, figsize=(16.8, 4.0), sharex=False)
    for ax, col in zip(axes.flat, key_cols):
        sub = plot_df[plot_df["descriptor"] == col].copy()
        if sub.empty:
            ax.set_axis_off()
            continue
        sns.boxplot(data=sub, x="group", y="value", ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel(col)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figures_dir / "discovered_descriptor_boxplot.png", dpi=dpi)
    plt.close(fig)


def _classification_dataset_path_candidates(config: Dict) -> List[Path]:
    chi_cfg = config.get("chi_training", {})
    cls_cfg = chi_cfg.get("step4_2_classification", {}) if isinstance(chi_cfg.get("step4_2_classification", {}), dict) else {}
    raw = cls_cfg.get("dataset_path", [])
    items: List[str] = []
    if isinstance(raw, (list, tuple)):
        items = [str(x).strip() for x in raw if str(x).strip()]
    elif isinstance(raw, str) and raw.strip():
        items = [tok.strip() for tok in raw.split(",") if tok.strip()]

    if not items:
        items = [
            "Data/water_solvent/water_miscible_polymer.csv",
            "Data/water_solvent/water_immiscible_polymer.csv",
        ]

    out: List[Path] = []
    for item in items:
        p = Path(item)
        if p.is_dir():
            out.extend(sorted(p.glob("*.csv")))
        else:
            out.append(p)
    # keep order, remove duplicates
    seen = set()
    dedup: List[Path] = []
    for p in out:
        k = str(p)
        if k in seen:
            continue
        seen.add(k)
        dedup.append(p)
    return dedup


def _load_classification_dataset_from_config(config: Dict) -> pd.DataFrame:
    paths = _classification_dataset_path_candidates(config)
    frames: List[pd.DataFrame] = []
    for p in paths:
        if p.exists() and p.is_file():
            try:
                f = _normalize_columns(pd.read_csv(p))
            except Exception:
                continue
            f["source_file"] = str(p)
            frames.append(f)
    if not frames:
        return pd.DataFrame(columns=["Polymer", "SMILES", "canonical_smiles", CLASS_COL, "source_file"])

    df = pd.concat(frames, ignore_index=True)
    df = _ensure_class_col(df)
    smiles_col = _resolve_smiles_column(df)
    if smiles_col is None:
        return pd.DataFrame(columns=["Polymer", "SMILES", "canonical_smiles", CLASS_COL, "source_file"])
    if smiles_col != "SMILES":
        df = df.rename(columns={smiles_col: "SMILES"})
    if "Polymer" not in df.columns:
        df["Polymer"] = df["SMILES"].astype(str)

    df["canonical_smiles"] = df["SMILES"].astype(str).map(canonicalize_smiles)
    df = df[df["canonical_smiles"].notna()].copy()

    agg_rows: List[Dict[str, object]] = []
    for can, sub in df.groupby("canonical_smiles", sort=False):
        y = pd.to_numeric(sub[CLASS_COL], errors="coerce").dropna().to_numpy(dtype=float)
        agg_rows.append(
            {
                "Polymer": str(sub["Polymer"].iloc[0]),
                "SMILES": str(sub["SMILES"].iloc[0]),
                "canonical_smiles": str(can),
                CLASS_COL: int(float(np.mean(y)) >= 0.5) if len(y) else int(pd.to_numeric(sub[CLASS_COL], errors="coerce").fillna(0).iloc[0]),
                "source_file": ";".join(sorted(set(sub["source_file"].astype(str).tolist()))),
            }
        )
    return pd.DataFrame(agg_rows)


def _build_descriptor_fg_table(poly_df: pd.DataFrame) -> pd.DataFrame:
    if poly_df.empty:
        return pd.DataFrame(columns=["Polymer", "SMILES", "canonical_smiles", CLASS_COL, "is_valid_mol", *DESCRIPTOR_COLUMNS, *FG_COLUMNS])
    rows: List[Dict[str, object]] = []
    for _, row in poly_df.iterrows():
        smi = str(row.get("SMILES", ""))
        mol = _mol_from_polymer_smiles(smi)
        desc = _calc_descriptor_row(smi)
        fgs = _detect_functional_groups(mol)
        rows.append(
            {
                "Polymer": str(row.get("Polymer", smi)),
                "SMILES": smi,
                "canonical_smiles": str(row.get("canonical_smiles", canonicalize_smiles(smi) or smi)),
                CLASS_COL: int(pd.to_numeric(pd.Series([row.get(CLASS_COL, 0)]), errors="coerce").fillna(0).iloc[0]),
                **desc,
                **fgs,
            }
        )
    return pd.DataFrame(rows)


def _compute_classification_context(
    chi_df: pd.DataFrame, class_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    chi_u = _normalize_columns(chi_df)[["Polymer", "SMILES", CLASS_COL]].drop_duplicates("SMILES").copy()
    chi_u["canonical_smiles"] = chi_u["SMILES"].astype(str).map(canonicalize_smiles)
    chi_u = chi_u[chi_u["canonical_smiles"].notna()].drop_duplicates("canonical_smiles").reset_index(drop=True)

    cls_u = class_df[["Polymer", "SMILES", "canonical_smiles", CLASS_COL]].drop_duplicates("canonical_smiles").reset_index(drop=True)

    chi_set = set(chi_u["canonical_smiles"].tolist())
    cls_set = set(cls_u["canonical_smiles"].tolist())
    overlap = chi_set & cls_set
    chi_only = chi_set - cls_set
    cls_only = cls_set - chi_set

    overlap_df = pd.DataFrame(
        [
            {"metric": "n_chi_unique", "value": int(len(chi_set))},
            {"metric": "n_classification_unique", "value": int(len(cls_set))},
            {"metric": "n_overlap", "value": int(len(overlap))},
            {"metric": "n_chi_only", "value": int(len(chi_only))},
            {"metric": "n_classification_only", "value": int(len(cls_only))},
            {
                "metric": "chi_coverage_in_classification",
                "value": float(len(overlap) / max(len(chi_set), 1)),
            },
        ]
    )

    chi_desc = _build_descriptor_fg_table(chi_u)
    cls_desc = _build_descriptor_fg_table(cls_u)

    desc_rows: List[Dict[str, object]] = []
    for cls in [1, 0]:
        for dcol in DESCRIPTOR_COLUMNS:
            x = pd.to_numeric(chi_desc.loc[chi_desc[CLASS_COL] == cls, dcol], errors="coerce").dropna().to_numpy(dtype=float)
            y = pd.to_numeric(cls_desc.loc[cls_desc[CLASS_COL] == cls, dcol], errors="coerce").dropna().to_numpy(dtype=float)
            y_std = float(np.std(y, ddof=0)) if len(y) else np.nan
            mean_x = float(np.mean(x)) if len(x) else np.nan
            mean_y = float(np.mean(y)) if len(y) else np.nan
            z_shift = (mean_x - mean_y) / y_std if (np.isfinite(y_std) and not np.isclose(y_std, 0.0)) else np.nan
            desc_rows.append(
                {
                    "descriptor": dcol,
                    CLASS_COL: int(cls),
                    "n_chi": int(len(x)),
                    "n_classification": int(len(y)),
                    "chi_mean": mean_x,
                    "classification_mean": mean_y,
                    "delta_chi_minus_classification": (mean_x - mean_y) if (np.isfinite(mean_x) and np.isfinite(mean_y)) else np.nan,
                    "z_shift_vs_classification": z_shift,
                    "mannwhitney_pvalue": _safe_mannwhitney(x, y),
                }
            )
    desc_shift_df = pd.DataFrame(desc_rows)

    fg_rows: List[Dict[str, object]] = []
    for cls in [1, 0]:
        for fg in FG_COLUMNS:
            x = pd.to_numeric(chi_desc.loc[chi_desc[CLASS_COL] == cls, fg], errors="coerce").dropna().to_numpy(dtype=float)
            y = pd.to_numeric(cls_desc.loc[cls_desc[CLASS_COL] == cls, fg], errors="coerce").dropna().to_numpy(dtype=float)
            frac_x = float(np.mean(x)) if len(x) else np.nan
            frac_y = float(np.mean(y)) if len(y) else np.nan
            fg_rows.append(
                {
                    "functional_group": fg.replace("fg_", ""),
                    CLASS_COL: int(cls),
                    "n_chi": int(len(x)),
                    "n_classification": int(len(y)),
                    "chi_fraction": frac_x,
                    "classification_fraction": frac_y,
                    "delta_chi_minus_classification": (frac_x - frac_y) if (np.isfinite(frac_x) and np.isfinite(frac_y)) else np.nan,
                    "mannwhitney_pvalue": _safe_mannwhitney(x, y),
                }
            )
    fg_cmp_df = pd.DataFrame(fg_rows)
    return overlap_df, desc_shift_df, fg_cmp_df, chi_desc, cls_desc


def _plot_classification_overlap_counts(overlap_df: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    if overlap_df.empty:
        return
    keys = ["n_chi_unique", "n_classification_unique", "n_overlap", "n_chi_only", "n_classification_only"]
    labels = ["chi", "classification", "overlap", "chi_only", "classification_only"]
    vals = []
    for k in keys:
        sub = overlap_df.loc[overlap_df["metric"] == k, "value"]
        vals.append(float(sub.iloc[0]) if not sub.empty else np.nan)
    plot_df = pd.DataFrame({"group": labels, "count": vals})
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    sns.barplot(data=plot_df, x="group", y="count", color="#4c78a8", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Unique polymers")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figures_dir / "classification_overlap_counts.png", dpi=dpi)
    plt.close(fig)


def _plot_classification_descriptor_shift(desc_shift_df: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    if desc_shift_df.empty:
        return
    plot_df = desc_shift_df.copy()
    plot_df = plot_df[np.isfinite(pd.to_numeric(plot_df["z_shift_vs_classification"], errors="coerce"))].copy()
    if plot_df.empty:
        return
    plot_df["abs_z"] = pd.to_numeric(plot_df["z_shift_vs_classification"], errors="coerce").abs()
    plot_df = plot_df.sort_values("abs_z", ascending=False).head(12).copy()
    if plot_df.empty:
        return
    plot_df["class_name"] = plot_df[CLASS_COL].map({1: "miscible", 0: "immiscible"})
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    sns.barplot(
        data=plot_df,
        x="z_shift_vs_classification",
        y="descriptor",
        hue="class_name",
        orient="h",
        ax=ax,
    )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Z-shift (chi subset minus classification)")
    ax.set_ylabel("Descriptor")
    fig.tight_layout()
    fig.savefig(figures_dir / "classification_vs_chi_descriptor_shift.png", dpi=dpi)
    plt.close(fig)


def _plot_classification_fg_shift(fg_cmp_df: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    if fg_cmp_df.empty:
        return
    plot_df = fg_cmp_df.copy()
    plot_df = plot_df[np.isfinite(pd.to_numeric(plot_df["delta_chi_minus_classification"], errors="coerce"))].copy()
    if plot_df.empty:
        return
    plot_df["abs_delta"] = pd.to_numeric(plot_df["delta_chi_minus_classification"], errors="coerce").abs()
    plot_df = plot_df.sort_values("abs_delta", ascending=False).head(12).copy()
    if plot_df.empty:
        return
    plot_df["class_name"] = plot_df[CLASS_COL].map({1: "miscible", 0: "immiscible"})
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    sns.barplot(
        data=plot_df,
        x="delta_chi_minus_classification",
        y="functional_group",
        hue="class_name",
        orient="h",
        ax=ax,
    )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Delta fraction (chi subset minus classification)")
    ax.set_ylabel("Functional group")
    fig.tight_layout()
    fig.savefig(figures_dir / "classification_vs_chi_fg_frequency.png", dpi=dpi)
    plt.close(fig)


def _build_embedding_union_polymer_df(
    chi_df: pd.DataFrame, class_df: pd.DataFrame
) -> Tuple[pd.DataFrame, int]:
    chi_u = _normalize_columns(chi_df)[["Polymer", "SMILES", CLASS_COL]].drop_duplicates("SMILES").copy()
    chi_u["canonical_smiles"] = chi_u["SMILES"].astype(str).map(canonicalize_smiles)
    chi_u = chi_u[chi_u["canonical_smiles"].notna()].drop_duplicates("canonical_smiles").reset_index(drop=True)
    chi_u = chi_u.rename(columns={CLASS_COL: "chi_class"})

    cls_u = class_df[["Polymer", "SMILES", "canonical_smiles", CLASS_COL]].drop_duplicates("canonical_smiles").copy()
    cls_u = cls_u.rename(columns={CLASS_COL: "classification_class", "Polymer": "Polymer_cls", "SMILES": "SMILES_cls"})

    merged = chi_u.merge(
        cls_u[["canonical_smiles", "Polymer_cls", "SMILES_cls", "classification_class"]],
        on="canonical_smiles",
        how="outer",
    )

    has_chi = merged["chi_class"].notna()
    has_cls = merged["classification_class"].notna()
    mismatch_count = int(
        ((pd.to_numeric(merged["chi_class"], errors="coerce") != pd.to_numeric(merged["classification_class"], errors="coerce")) & has_chi & has_cls).sum()
    )

    merged["source_group"] = np.where(
        has_chi & has_cls,
        "chi_and_classification",
        np.where(has_chi, "chi_only", "classification_only"),
    )
    merged["Polymer"] = merged["Polymer"].where(has_chi, merged["Polymer_cls"])
    merged["SMILES"] = merged["SMILES"].where(has_chi, merged["SMILES_cls"])
    merged[CLASS_COL] = pd.to_numeric(merged["classification_class"], errors="coerce")
    merged[CLASS_COL] = merged[CLASS_COL].where(has_cls, pd.to_numeric(merged["chi_class"], errors="coerce"))
    merged[CLASS_COL] = merged[CLASS_COL].fillna(0).astype(int)

    out = merged[["Polymer", "SMILES", "canonical_smiles", CLASS_COL, "source_group"]].copy()
    out = out[out["canonical_smiles"].notna()].drop_duplicates("canonical_smiles").reset_index(drop=True)
    out["polymer_id"] = np.arange(len(out), dtype=int)
    return out, mismatch_count


def _compute_step1_embedding_coordinates(
    config: Dict,
    union_df: pd.DataFrame,
    cache_npz: Path,
    model_size: str,
    split_mode: str,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    emb_in = union_df[["polymer_id", "Polymer", "SMILES", CLASS_COL]].copy()
    emb_cfg = config.get("chi_training", {}).get("shared", {}).get("embedding", {})
    timestep = int(emb_cfg.get("timestep", config.get("training_property", {}).get("default_timestep", 1)))
    batch_size = int(emb_cfg.get("batch_size", 128))
    emb_df = build_or_load_embedding_cache(
        polymer_df=emb_in,
        config=config,
        cache_npz=cache_npz,
        model_size=model_size,
        split_mode=split_mode,
        checkpoint_path=None,
        device="cpu",
        timestep=timestep,
        pooling="mean",
        batch_size=batch_size,
    )
    out = emb_df.merge(
        union_df[["polymer_id", "canonical_smiles", "source_group"]],
        on="polymer_id",
        how="left",
    )
    X = np.stack(out["embedding"].to_list(), axis=0).astype(float, copy=False)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X)
    out["PC1"] = pcs[:, 0]
    out["PC2"] = pcs[:, 1]
    out["embedding_norm"] = np.linalg.norm(X, axis=1)
    info = {
        "pc1_explained_variance_ratio": float(pca.explained_variance_ratio_[0]) if len(pca.explained_variance_ratio_) >= 1 else np.nan,
        "pc2_explained_variance_ratio": float(pca.explained_variance_ratio_[1]) if len(pca.explained_variance_ratio_) >= 2 else np.nan,
    }
    return out, info


def _embedding_cv_auc(X: np.ndarray, y: np.ndarray, seed: int = 42) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2 or len(y) < 20:
        return {"n": int(len(y)), "n_folds": 0, "auc_mean": np.nan, "auc_std": np.nan}
    class_counts = np.bincount(y, minlength=2)
    min_class = int(np.min(class_counts[class_counts > 0])) if np.any(class_counts > 0) else 0
    n_folds = int(min(5, min_class))
    if n_folds < 2:
        return {"n": int(len(y)), "n_folds": 0, "auc_mean": np.nan, "auc_std": np.nan}
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    aucs: List[float] = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2500, class_weight="balanced")
        clf.fit(X[tr], y[tr])
        prob = clf.predict_proba(X[te])[:, 1]
        aucs.append(float(roc_auc_score(y[te], prob)))
    return {"n": int(len(y)), "n_folds": int(n_folds), "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs, ddof=0))}


def _compute_embedding_classification_signal(emb_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    if emb_df.empty:
        return pd.DataFrame(columns=["subset", "n", "n_folds", "auc_mean", "auc_std"])
    X_all = np.stack(emb_df["embedding"].to_list(), axis=0).astype(float, copy=False)
    y_all = pd.to_numeric(emb_df[CLASS_COL], errors="coerce").fillna(0).astype(int).to_numpy(dtype=int)
    rows: List[Dict[str, object]] = []
    for subset_name, mask in [
        ("all_union", np.ones(len(emb_df), dtype=bool)),
        ("chi_related", emb_df["source_group"].isin(["chi_and_classification", "chi_only"]).to_numpy(dtype=bool)),
        ("classification_only", (emb_df["source_group"] == "classification_only").to_numpy(dtype=bool)),
    ]:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            rows.append({"subset": subset_name, "n": 0, "n_folds": 0, "auc_mean": np.nan, "auc_std": np.nan})
            continue
        stats = _embedding_cv_auc(X_all[idx], y_all[idx], seed=seed)
        rows.append({"subset": subset_name, **stats})
    return pd.DataFrame(rows)


def _plot_step1_embedding_pca(emb_df: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    if emb_df.empty:
        return
    plot_df = emb_df.copy()
    plot_df["class_name"] = plot_df[CLASS_COL].map({1: "miscible", 0: "immiscible"})
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    sns.scatterplot(
        data=plot_df,
        x="PC1",
        y="PC2",
        hue="class_name",
        style="source_group",
        alpha=0.80,
        s=44,
        ax=ax,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    fig.savefig(figures_dir / "step1_embedding_pca_by_source.png", dpi=dpi)
    plt.close(fig)


def _compute_embedding_pinn_correlations(
    emb_df: pd.DataFrame, coeff_df: pd.DataFrame, chi_df: pd.DataFrame
) -> pd.DataFrame:
    if emb_df.empty or coeff_df.empty:
        return pd.DataFrame(columns=["variable", "pc_axis", "pearson_r", "pvalue", "n"])
    coeff = coeff_df.copy()
    if "canonical_smiles" not in coeff.columns:
        coeff["canonical_smiles"] = coeff["SMILES"].astype(str).map(canonicalize_smiles)
    chi_mean = (
        _normalize_columns(chi_df)
        .assign(canonical_smiles=lambda d: d["SMILES"].astype(str).map(canonicalize_smiles))
        .groupby("canonical_smiles", as_index=False)["chi"]
        .mean()
        .rename(columns={"chi": "chi_mean"})
    )
    merged = emb_df.merge(
        coeff[["canonical_smiles", *COEFF_COLUMNS, "dchi_dT_ref"]],
        on="canonical_smiles",
        how="inner",
    ).merge(chi_mean, on="canonical_smiles", how="left")
    rows: List[Dict[str, object]] = []
    for var in [*COEFF_COLUMNS, "dchi_dT_ref", "chi_mean"]:
        for pc in ["PC1", "PC2"]:
            x = pd.to_numeric(merged[pc], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(merged[var], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if int(mask.sum()) >= 3:
                r, p = pearsonr(x[mask], y[mask])
                rows.append({"variable": var, "pc_axis": pc, "pearson_r": float(r), "pvalue": float(p), "n": int(mask.sum())})
            else:
                rows.append({"variable": var, "pc_axis": pc, "pearson_r": np.nan, "pvalue": np.nan, "n": int(mask.sum())})
    return pd.DataFrame(rows)


def _plot_embedding_pinn_corr_heatmap(corr_df: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    if corr_df.empty:
        return
    pivot = corr_df.pivot(index="variable", columns="pc_axis", values="pearson_r")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(4.6, 5.6))
    sns.heatmap(
        pivot,
        cmap="coolwarm",
        center=0.0,
        annot=True,
        fmt=".2f",
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Pearson r"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(figures_dir / "embedding_pinn_correlation_heatmap.png", dpi=dpi)
    plt.close(fig)


def _compute_pinn_coefficient_sensitivity(
    coeff_df: pd.DataFrame,
    T_grid: np.ndarray,
    phi_grid: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    valid = coeff_df.dropna(subset=COEFF_COLUMNS).copy()
    if valid.empty:
        return pd.DataFrame(), pd.DataFrame()
    TT, PP = np.meshgrid(np.asarray(T_grid, dtype=float), np.asarray(phi_grid, dtype=float), indexing="ij")
    T = TT.ravel()
    P = PP.ravel()
    om = 1.0 - P
    logT = np.log(np.maximum(T, 1e-12))

    rows: List[Dict[str, object]] = []
    for _, row in valid.iterrows():
        a0, a1, a2, a3, b1, b2 = [float(row[c]) for c in COEFF_COLUMNS]
        base = a0 + a1 / T + a2 * logT + a3 * T
        mod = 1.0 + b1 * om + b2 * (om ** 2)
        raw = {
            "a0": float(np.mean(np.abs(mod * a0))),
            "a1": float(np.mean(np.abs((mod / T) * a1))),
            "a2": float(np.mean(np.abs((mod * logT) * a2))),
            "a3": float(np.mean(np.abs((mod * T) * a3))),
            "b1": float(np.mean(np.abs((base * om) * b1))),
            "b2": float(np.mean(np.abs((base * (om ** 2)) * b2))),
        }
        denom = float(sum(raw.values()))
        if denom > 0:
            norm = {k: float(v / denom) for k, v in raw.items()}
        else:
            norm = {k: np.nan for k in raw}
        rows.append(
            {
                "Polymer": row.get("Polymer", ""),
                "SMILES": row.get("SMILES", ""),
                "canonical_smiles": canonicalize_smiles(str(row.get("SMILES", ""))),
                CLASS_COL: int(pd.to_numeric(pd.Series([row.get(CLASS_COL, 0)]), errors="coerce").fillna(0).iloc[0]),
                **norm,
            }
        )
    per_poly = pd.DataFrame(rows)
    if per_poly.empty:
        return per_poly, pd.DataFrame()
    long_df = per_poly.melt(
        id_vars=["Polymer", "SMILES", "canonical_smiles", CLASS_COL],
        value_vars=COEFF_COLUMNS,
        var_name="coefficient",
        value_name="normalized_sensitivity",
    )
    class_stats = (
        long_df.groupby([CLASS_COL, "coefficient"], as_index=False)
        .agg(
            mean=("normalized_sensitivity", "mean"),
            std=("normalized_sensitivity", "std"),
            median=("normalized_sensitivity", "median"),
            n=("normalized_sensitivity", "count"),
        )
        .reset_index(drop=True)
    )
    return per_poly, class_stats


def _plot_pinn_coefficient_sensitivity(per_poly_df: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    if per_poly_df.empty:
        return
    long_df = per_poly_df.melt(
        id_vars=[CLASS_COL],
        value_vars=COEFF_COLUMNS,
        var_name="coefficient",
        value_name="normalized_sensitivity",
    )
    long_df = long_df[np.isfinite(pd.to_numeric(long_df["normalized_sensitivity"], errors="coerce"))].copy()
    if long_df.empty:
        return
    long_df["class_name"] = long_df[CLASS_COL].map({1: "miscible", 0: "immiscible"})
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    sns.boxplot(data=long_df, x="coefficient", y="normalized_sensitivity", hue="class_name", ax=ax)
    ax.set_xlabel("PINN coefficient")
    ax.set_ylabel("Normalized sensitivity contribution")
    fig.tight_layout()
    fig.savefig(figures_dir / "pinn_coefficient_sensitivity_by_class.png", dpi=dpi)
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    step7_cfg = _default_step7_config(config)
    split_mode = str(args.split_mode or step7_cfg["split_mode"]).strip().lower()
    model_size = args.model_size or "small"

    results_base = str(config["paths"]["results_dir"])
    results_dir = Path(get_results_dir(model_size, results_base, split_mode=None))
    base_results_dir = Path(results_base)

    step_dir = results_dir / "step7_chem_physics_analysis" / split_mode
    metrics_dir = step_dir / "metrics"
    figures_dir = step_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    seed_value = int(args.random_seed if args.random_seed is not None else config["data"]["random_seed"])
    seed_info = seed_everything(seed_value)
    save_config(config, step_dir / "config_used.yaml")
    save_run_metadata(step_dir, args.config, seed_info)

    dpi = int(args.dpi if args.dpi is not None else config.get("plotting", {}).get("dpi", 600))
    font_size = int(args.font_size if args.font_size is not None else config.get("plotting", {}).get("font_size", 16))
    apply_publication_figure_style(font_size=font_size, dpi=dpi, remove_titles=True)

    chi_path = Path(args.chi_dataset_path or step7_cfg["chi_dataset_path"])
    step1_summary_path = _first_existing(
        [
            results_dir / "step1_backbone" / "metrics" / "step_summary.csv",
            base_results_dir / "step1_backbone" / "metrics" / "step_summary.csv",
        ]
    )
    step2_summary_path = _first_existing(
        [
            results_dir / "step2_sampling" / "metrics" / "step_summary.csv",
            base_results_dir / "step2_sampling" / "metrics" / "step_summary.csv",
        ]
    )
    step2_targets_path = _first_existing(
        [
            results_dir / "step2_sampling" / "metrics" / "target_polymers.csv",
            base_results_dir / "step2_sampling" / "metrics" / "target_polymers.csv",
        ]
    )
    step3_summary_path = _first_existing(
        [
            results_dir / "step3_chi_target_learning" / split_mode / "metrics" / "step_summary.csv",
            base_results_dir / "step3_chi_target_learning" / split_mode / "metrics" / "step_summary.csv",
        ]
    )
    step3_targets_path = _first_existing(
        [
            results_dir / "step3_chi_target_learning" / split_mode / "metrics" / "chi_target_for_inverse_design.csv",
            base_results_dir / "step3_chi_target_learning" / split_mode / "metrics" / "chi_target_for_inverse_design.csv",
        ]
    )
    step3_dataset_with_split_path = _first_existing(
        [
            results_dir / "step3_chi_target_learning" / split_mode / "metrics" / "chi_dataset_with_split.csv",
            base_results_dir / "step3_chi_target_learning" / split_mode / "metrics" / "chi_dataset_with_split.csv",
        ]
    )
    step4_summary_path = _first_existing(
        [
            results_dir / f"step4_split_pipeline_{split_mode}" / "pipeline_metrics" / "step_summary.csv",
            base_results_dir / f"step4_split_pipeline_{split_mode}" / "pipeline_metrics" / "step_summary.csv",
            results_dir / "step4_chi_training" / split_mode / "metrics" / "step_summary.csv",
            base_results_dir / "step4_chi_training" / split_mode / "metrics" / "step_summary.csv",
        ]
    )
    step4_path = (
        Path(args.step4_predictions_csv)
        if args.step4_predictions_csv
        else _first_existing(
            [
                results_dir / "step4_1_regression" / split_mode / "metrics" / "chi_predictions_all.csv",
                results_dir / "step4_chi_training" / split_mode / "step4_1_regression" / "metrics" / "chi_predictions_all.csv",
                base_results_dir / "step4_1_regression" / split_mode / "metrics" / "chi_predictions_all.csv",
                base_results_dir / "step4_chi_training" / split_mode / "step4_1_regression" / "metrics" / "chi_predictions_all.csv",
            ]
        )
    )
    step4_coeff_path = _first_existing(
        [
            results_dir / "step4_1_regression" / split_mode / "metrics" / "polymer_coefficients_regression_only.csv",
            base_results_dir / "step4_1_regression" / split_mode / "metrics" / "polymer_coefficients_regression_only.csv",
            results_dir / "step4_chi_training" / split_mode / "step4_1_regression" / "metrics" / "polymer_coefficients_regression_only.csv",
            base_results_dir / "step4_chi_training" / split_mode / "step4_1_regression" / "metrics" / "polymer_coefficients_regression_only.csv",
        ]
    )
    step4_reg_overall_path = _first_existing(
        [
            results_dir / "step4_1_regression" / split_mode / "metrics" / "chi_metrics_overall.csv",
            base_results_dir / "step4_1_regression" / split_mode / "metrics" / "chi_metrics_overall.csv",
            results_dir / "step4_chi_training" / split_mode / "step4_1_regression" / "metrics" / "chi_metrics_overall.csv",
            base_results_dir / "step4_chi_training" / split_mode / "step4_1_regression" / "metrics" / "chi_metrics_overall.csv",
        ]
    )
    step4_cls_overall_path = _first_existing(
        [
            results_dir / "step4_2_classification" / "metrics" / "class_metrics_overall.csv",
            base_results_dir / "step4_2_classification" / "metrics" / "class_metrics_overall.csv",
            results_dir / "step4_chi_training" / split_mode / "step4_2_classification" / "metrics" / "class_metrics_overall.csv",
            base_results_dir / "step4_chi_training" / split_mode / "step4_2_classification" / "metrics" / "class_metrics_overall.csv",
        ]
    )
    step5_targets_path = (
        Path(args.step5_targets_csv)
        if args.step5_targets_csv
        else _first_existing(
            [
                results_dir / "step5_water_soluble_inverse_design" / split_mode / "metrics" / "target_polymers.csv",
                base_results_dir / "step5_water_soluble_inverse_design" / split_mode / "metrics" / "target_polymers.csv",
            ]
        )
    )
    step5_summary_path = _first_existing(
        [
            results_dir / "step5_water_soluble_inverse_design" / split_mode / "metrics" / "step_summary.csv",
            base_results_dir / "step5_water_soluble_inverse_design" / split_mode / "metrics" / "step_summary.csv",
        ]
    )
    step6_targets_path = (
        Path(args.step6_targets_csv)
        if args.step6_targets_csv
        else _first_existing(
            [
                results_dir / "step6_polymer_class_water_soluble_inverse_design" / split_mode / "metrics" / "target_polymers.csv",
                base_results_dir / "step6_polymer_class_water_soluble_inverse_design" / split_mode / "metrics" / "target_polymers.csv",
            ]
        )
    )
    step6_summary_path = _first_existing(
        [
            results_dir / "step6_polymer_class_water_soluble_inverse_design" / split_mode / "metrics" / "step_summary.csv",
            base_results_dir / "step6_polymer_class_water_soluble_inverse_design" / split_mode / "metrics" / "step_summary.csv",
        ]
    )
    step5_inverse_candidates_path = _first_existing(
        [
            results_dir / "step5_water_soluble_inverse_design" / split_mode / "metrics" / "inverse_candidates_all.csv",
            base_results_dir / "step5_water_soluble_inverse_design" / split_mode / "metrics" / "inverse_candidates_all.csv",
        ]
    )
    step6_inverse_candidates_path = _first_existing(
        [
            results_dir / "step6_polymer_class_water_soluble_inverse_design" / split_mode / "metrics" / "inverse_candidates_all.csv",
            base_results_dir / "step6_polymer_class_water_soluble_inverse_design" / split_mode / "metrics" / "inverse_candidates_all.csv",
        ]
    )
    baseline_train_path = (
        Path(args.baseline_train_csv)
        if args.baseline_train_csv
        else _first_existing([results_dir / "train_unlabeled.csv", base_results_dir / "train_unlabeled.csv"])
    )
    classification_dataset_paths = _classification_dataset_path_candidates(config)
    classification_existing_paths = [p for p in classification_dataset_paths if p.exists()]
    embedding_cache_path = step_dir / "cache" / f"step1_embedding_union_{split_mode}.npz"

    write_initial_log(
        step_dir=step_dir,
        step_name="step7_chem_physics_analysis",
        context={
            "config_path": args.config,
            "model_size": model_size,
            "split_mode": split_mode,
            "results_dir": str(results_dir),
            "chi_dataset_path": str(chi_path),
            "step1_summary_path": str(step1_summary_path) if step1_summary_path is not None else "",
            "step2_summary_path": str(step2_summary_path) if step2_summary_path is not None else "",
            "step2_targets_path": str(step2_targets_path) if step2_targets_path is not None else "",
            "step3_summary_path": str(step3_summary_path) if step3_summary_path is not None else "",
            "step3_targets_path": str(step3_targets_path) if step3_targets_path is not None else "",
            "step3_dataset_with_split_path": str(step3_dataset_with_split_path) if step3_dataset_with_split_path is not None else "",
            "step4_summary_path": str(step4_summary_path) if step4_summary_path is not None else "",
            "step4_predictions_path": str(step4_path) if step4_path is not None else "",
            "step4_coeff_path": str(step4_coeff_path) if step4_coeff_path is not None else "",
            "step4_reg_overall_path": str(step4_reg_overall_path) if step4_reg_overall_path is not None else "",
            "step4_cls_overall_path": str(step4_cls_overall_path) if step4_cls_overall_path is not None else "",
            "step5_summary_path": str(step5_summary_path) if step5_summary_path is not None else "",
            "step5_targets_path": str(step5_targets_path) if step5_targets_path is not None else "",
            "step6_summary_path": str(step6_summary_path) if step6_summary_path is not None else "",
            "step6_targets_path": str(step6_targets_path) if step6_targets_path is not None else "",
            "step5_inverse_candidates_path": str(step5_inverse_candidates_path) if step5_inverse_candidates_path is not None else "",
            "step6_inverse_candidates_path": str(step6_inverse_candidates_path) if step6_inverse_candidates_path is not None else "",
            "baseline_train_path": str(baseline_train_path) if baseline_train_path is not None else "",
            "classification_dataset_paths": ";".join(str(p) for p in classification_dataset_paths),
            "step1_embedding_cache_path": str(embedding_cache_path),
            "baseline_max_samples": int(args.baseline_max_samples or step7_cfg["baseline_max_samples"]),
            "novelty_similarity_threshold": float(
                args.novelty_similarity_threshold or step7_cfg["novelty_similarity_threshold"]
            ),
            "random_seed": seed_value,
        },
    )

    artifact_status = pd.DataFrame(
        [
            {"artifact": "chi_dataset", "path": str(chi_path), "required": 1, "exists": int(chi_path.exists())},
            {
                "artifact": "step1_summary",
                "path": str(step1_summary_path) if step1_summary_path is not None else "",
                "required": 0,
                "exists": int(step1_summary_path is not None and step1_summary_path.exists()),
            },
            {
                "artifact": "step2_summary",
                "path": str(step2_summary_path) if step2_summary_path is not None else "",
                "required": 0,
                "exists": int(step2_summary_path is not None and step2_summary_path.exists()),
            },
            {
                "artifact": "step2_target_polymers",
                "path": str(step2_targets_path) if step2_targets_path is not None else "",
                "required": 0,
                "exists": int(step2_targets_path is not None and step2_targets_path.exists()),
            },
            {
                "artifact": "step3_summary",
                "path": str(step3_summary_path) if step3_summary_path is not None else "",
                "required": 0,
                "exists": int(step3_summary_path is not None and step3_summary_path.exists()),
            },
            {
                "artifact": "step3_targets",
                "path": str(step3_targets_path) if step3_targets_path is not None else "",
                "required": 0,
                "exists": int(step3_targets_path is not None and step3_targets_path.exists()),
            },
            {
                "artifact": "step3_dataset_with_split",
                "path": str(step3_dataset_with_split_path) if step3_dataset_with_split_path is not None else "",
                "required": 0,
                "exists": int(step3_dataset_with_split_path is not None and step3_dataset_with_split_path.exists()),
            },
            {
                "artifact": "step4_summary",
                "path": str(step4_summary_path) if step4_summary_path is not None else "",
                "required": 0,
                "exists": int(step4_summary_path is not None and step4_summary_path.exists()),
            },
            {
                "artifact": "step4_predictions",
                "path": str(step4_path) if step4_path is not None else "",
                "required": 0,
                "exists": int(step4_path is not None and step4_path.exists()),
            },
            {
                "artifact": "step4_coefficients",
                "path": str(step4_coeff_path) if step4_coeff_path is not None else "",
                "required": 0,
                "exists": int(step4_coeff_path is not None and step4_coeff_path.exists()),
            },
            {
                "artifact": "step4_reg_overall_metrics",
                "path": str(step4_reg_overall_path) if step4_reg_overall_path is not None else "",
                "required": 0,
                "exists": int(step4_reg_overall_path is not None and step4_reg_overall_path.exists()),
            },
            {
                "artifact": "step4_cls_overall_metrics",
                "path": str(step4_cls_overall_path) if step4_cls_overall_path is not None else "",
                "required": 0,
                "exists": int(step4_cls_overall_path is not None and step4_cls_overall_path.exists()),
            },
            {
                "artifact": "step5_summary",
                "path": str(step5_summary_path) if step5_summary_path is not None else "",
                "required": 0,
                "exists": int(step5_summary_path is not None and step5_summary_path.exists()),
            },
            {
                "artifact": "step5_target_polymers",
                "path": str(step5_targets_path) if step5_targets_path is not None else "",
                "required": 0,
                "exists": int(step5_targets_path is not None and step5_targets_path.exists()),
            },
            {
                "artifact": "step6_summary",
                "path": str(step6_summary_path) if step6_summary_path is not None else "",
                "required": 0,
                "exists": int(step6_summary_path is not None and step6_summary_path.exists()),
            },
            {
                "artifact": "step6_target_polymers",
                "path": str(step6_targets_path) if step6_targets_path is not None else "",
                "required": 0,
                "exists": int(step6_targets_path is not None and step6_targets_path.exists()),
            },
            {
                "artifact": "step5_inverse_candidates",
                "path": str(step5_inverse_candidates_path) if step5_inverse_candidates_path is not None else "",
                "required": 0,
                "exists": int(step5_inverse_candidates_path is not None and step5_inverse_candidates_path.exists()),
            },
            {
                "artifact": "step6_inverse_candidates",
                "path": str(step6_inverse_candidates_path) if step6_inverse_candidates_path is not None else "",
                "required": 0,
                "exists": int(step6_inverse_candidates_path is not None and step6_inverse_candidates_path.exists()),
            },
            {
                "artifact": "baseline_training_smiles",
                "path": str(baseline_train_path) if baseline_train_path is not None else "",
                "required": 0,
                "exists": int(baseline_train_path is not None and baseline_train_path.exists()),
            },
            {
                "artifact": "classification_dataset_configured",
                "path": ";".join(str(p) for p in classification_dataset_paths),
                "required": 0,
                "exists": int(len(classification_existing_paths) > 0),
            },
            {
                "artifact": "step1_embedding_cache",
                "path": str(embedding_cache_path),
                "required": 0,
                "exists": int(embedding_cache_path.exists()),
            },
        ]
    )
    artifact_status.to_csv(metrics_dir / "input_artifact_status.csv", index=False)

    if not chi_path.exists():
        raise FileNotFoundError(f"Required chi dataset not found: {chi_path}")

    summary: Dict[str, object] = {
        "step": "step7_chem_physics_analysis",
        "model_size": model_size,
        "split_mode": split_mode,
        "chi_dataset_path": str(chi_path),
        "step1_summary_path": str(step1_summary_path) if step1_summary_path is not None else "",
        "step2_summary_path": str(step2_summary_path) if step2_summary_path is not None else "",
        "step2_targets_path": str(step2_targets_path) if step2_targets_path is not None else "",
        "step3_summary_path": str(step3_summary_path) if step3_summary_path is not None else "",
        "step3_targets_path": str(step3_targets_path) if step3_targets_path is not None else "",
        "step3_dataset_with_split_path": str(step3_dataset_with_split_path) if step3_dataset_with_split_path is not None else "",
        "step4_summary_path": str(step4_summary_path) if step4_summary_path is not None else "",
        "step4_predictions_path": str(step4_path) if step4_path is not None else "",
        "step4_coeff_path": str(step4_coeff_path) if step4_coeff_path is not None else "",
        "step4_reg_overall_path": str(step4_reg_overall_path) if step4_reg_overall_path is not None else "",
        "step4_cls_overall_path": str(step4_cls_overall_path) if step4_cls_overall_path is not None else "",
        "step5_summary_path": str(step5_summary_path) if step5_summary_path is not None else "",
        "step5_targets_path": str(step5_targets_path) if step5_targets_path is not None else "",
        "step6_summary_path": str(step6_summary_path) if step6_summary_path is not None else "",
        "step6_targets_path": str(step6_targets_path) if step6_targets_path is not None else "",
        "step5_inverse_candidates_path": str(step5_inverse_candidates_path) if step5_inverse_candidates_path is not None else "",
        "step6_inverse_candidates_path": str(step6_inverse_candidates_path) if step6_inverse_candidates_path is not None else "",
        "baseline_train_path": str(baseline_train_path) if baseline_train_path is not None else "",
        "classification_dataset_paths": ";".join(str(p) for p in classification_dataset_paths),
        "step1_embedding_cache_path": str(embedding_cache_path),
        "step1_best_val_bpb": np.nan,
        "step1_best_val_loss": np.nan,
        "step2_validity": np.nan,
        "step2_novelty": np.nan,
        "step2_target_selection_success_rate": np.nan,
        "step3_global_chi_target": np.nan,
        "step3_global_balanced_accuracy": np.nan,
        "step3_target_closer_to_miscible_fraction": np.nan,
        "step4_test_r2": np.nan,
        "step4_test_balanced_accuracy": np.nan,
        "step5_target_selection_success_rate": np.nan,
        "step6_target_selection_success_rate": np.nan,
        "n_conditions": 0,
        "n_significant_conditions": 0,
        "mean_delta_chi_miscible_minus_immiscible": np.nan,
        "step4_test_mae": np.nan,
        "step4_dchi_dT_sign_agreement": np.nan,
        "step4_dchi_dphi_sign_agreement": np.nan,
        "n_polymers_with_coefficients": 0,
        "frac_lcst_like_soluble": np.nan,
        "frac_lcst_like_insoluble": np.nan,
        "mean_spinodal_miscible_fraction_soluble": np.nan,
        "mean_spinodal_miscible_fraction_insoluble": np.nan,
        "n_functional_group_types_analyzed": int(len(FG_COLUMNS)),
        "n_numbered_figures": 0,
        "n_classification_polymers": 0,
        "chi_coverage_in_classification": np.nan,
        "embedding_label_disagreement_count": 0,
        "n_step1_embedding_polymers": 0,
        "step1_embedding_cv_auc": np.nan,
        "step1_embedding_pc1_explained_variance": np.nan,
        "step1_embedding_pc2_explained_variance": np.nan,
        "pinn_top_driver_soluble": "",
        "pinn_top_driver_insoluble": "",
        "n_selected_unique_candidates": 0,
        "mean_max_tanimoto_to_training": np.nan,
        "novel_fraction_under_threshold": np.nan,
    }
    analysis_blocks: List[str] = []

    print("=" * 72)
    print("Step 7: chemical + physics analysis")
    print(f"model_size={model_size}, split_mode={split_mode}")
    print(f"chi_dataset={chi_path}")
    print("=" * 72)

    # 0) Ingest pipeline summaries from Step1-6 for cross-step science context.
    step_rollup_rows: List[Dict[str, object]] = []
    step1_row = _read_summary_row(step1_summary_path)
    step2_row = _read_summary_row(step2_summary_path)
    step3_row = _read_summary_row(step3_summary_path)
    step4_row = _read_summary_row(step4_summary_path)
    step5_row = _read_summary_row(step5_summary_path)
    step6_row = _read_summary_row(step6_summary_path)

    if step1_row:
        summary["step1_best_val_bpb"] = _pick_numeric(step1_row, ["best_val_bpb"])
        summary["step1_best_val_loss"] = _pick_numeric(step1_row, ["best_val_loss"])
    if step2_row:
        summary["step2_validity"] = _pick_numeric(step2_row, ["validity"])
        summary["step2_novelty"] = _pick_numeric(step2_row, ["novelty"])
        summary["step2_target_selection_success_rate"] = _pick_numeric(
            step2_row, ["target_polymer_selection_success_rate"]
        )
    if step3_row:
        summary["step3_global_chi_target"] = _pick_numeric(step3_row, ["global_chi_target"])
        summary["step3_global_balanced_accuracy"] = _pick_numeric(
            step3_row, ["global_balanced_accuracy", "condition_test_balanced_accuracy"]
        )
    if step5_row:
        summary["step5_target_selection_success_rate"] = _pick_numeric(
            step5_row, ["target_polymer_selection_success_rate"]
        )
    if step6_row:
        summary["step6_target_selection_success_rate"] = _pick_numeric(
            step6_row, ["target_polymer_selection_success_rate"]
        )

    # Step4 metrics: prefer step summary; fallback to overall metrics files.
    summary["step4_test_r2"] = _pick_numeric(step4_row, ["step4_1_test_r2", "step4_test_r2"])
    summary["step4_test_balanced_accuracy"] = _pick_numeric(
        step4_row,
        ["step4_2_test_balanced_accuracy", "step4_test_balanced_accuracy"],
    )
    if not np.isfinite(float(summary["step4_test_r2"])) and step4_reg_overall_path is not None and step4_reg_overall_path.exists():
        reg_df = _normalize_columns(pd.read_csv(step4_reg_overall_path))
        if "split" in reg_df.columns and (reg_df["split"] == "test").any():
            reg_df = reg_df[reg_df["split"] == "test"].copy()
        if not reg_df.empty:
            summary["step4_test_r2"] = _pick_numeric(reg_df.iloc[0].to_dict(), ["r2"])
    if (
        not np.isfinite(float(summary["step4_test_balanced_accuracy"]))
        and step4_cls_overall_path is not None
        and step4_cls_overall_path.exists()
    ):
        cls_df = _normalize_columns(pd.read_csv(step4_cls_overall_path))
        if "split" in cls_df.columns and (cls_df["split"] == "test").any():
            cls_df = cls_df[cls_df["split"] == "test"].copy()
        if not cls_df.empty:
            summary["step4_test_balanced_accuracy"] = _pick_numeric(cls_df.iloc[0].to_dict(), ["balanced_accuracy"])

    step_rollup_rows.append(
        {
            "step": "step1",
            "summary_path": str(step1_summary_path) if step1_summary_path is not None else "",
            "available": int(bool(step1_row)),
            "key_metric_name": "best_val_bpb",
            "key_metric_value": summary["step1_best_val_bpb"],
        }
    )
    step_rollup_rows.append(
        {
            "step": "step2",
            "summary_path": str(step2_summary_path) if step2_summary_path is not None else "",
            "available": int(bool(step2_row)),
            "key_metric_name": "target_polymer_selection_success_rate",
            "key_metric_value": summary["step2_target_selection_success_rate"],
        }
    )
    step_rollup_rows.append(
        {
            "step": "step3",
            "summary_path": str(step3_summary_path) if step3_summary_path is not None else "",
            "available": int(bool(step3_row)),
            "key_metric_name": "global_chi_target",
            "key_metric_value": summary["step3_global_chi_target"],
        }
    )
    step_rollup_rows.append(
        {
            "step": "step4",
            "summary_path": str(step4_summary_path) if step4_summary_path is not None else "",
            "available": int(bool(step4_row) or (step4_reg_overall_path is not None) or (step4_cls_overall_path is not None)),
            "key_metric_name": "test_r2",
            "key_metric_value": summary["step4_test_r2"],
        }
    )
    step_rollup_rows.append(
        {
            "step": "step5",
            "summary_path": str(step5_summary_path) if step5_summary_path is not None else "",
            "available": int(bool(step5_row)),
            "key_metric_name": "target_polymer_selection_success_rate",
            "key_metric_value": summary["step5_target_selection_success_rate"],
        }
    )
    step_rollup_rows.append(
        {
            "step": "step6",
            "summary_path": str(step6_summary_path) if step6_summary_path is not None else "",
            "available": int(bool(step6_row)),
            "key_metric_name": "target_polymer_selection_success_rate",
            "key_metric_value": summary["step6_target_selection_success_rate"],
        }
    )

    rollup_df = pd.DataFrame(step_rollup_rows)
    rollup_df.to_csv(metrics_dir / "step1_to_step6_summary_rollup.csv", index=False)

    success_rows: List[Dict[str, object]] = []
    for step_name, key in [
        ("step2", "step2_target_selection_success_rate"),
        ("step5", "step5_target_selection_success_rate"),
        ("step6", "step6_target_selection_success_rate"),
    ]:
        rate = _safe_float(summary.get(key, np.nan))
        success_rows.append({"step": step_name, "success_rate": rate})
    success_df = pd.DataFrame(success_rows)
    success_df.to_csv(metrics_dir / "step2_step5_step6_success_rates.csv", index=False)
    _plot_pipeline_success_rates(
        success_df,
        out_png=figures_dir / "pipeline_selection_success_rates.png",
        dpi=dpi,
    )
    summary["n_steps_with_available_summary"] = int(rollup_df["available"].sum())
    analysis_blocks.append("step1_to_step6_summary_rollup")

    # A) Dataset-level thermodynamic class-separation analysis.
    chi_df = _ensure_class_col(pd.read_csv(chi_path))
    required_cols = {"temperature", "phi", "chi", CLASS_COL}
    missing_cols = required_cols - set(chi_df.columns)
    if missing_cols:
        raise ValueError(f"Chi dataset missing required columns: {sorted(missing_cols)}")
    chi_df["temperature"] = pd.to_numeric(chi_df["temperature"], errors="coerce")
    chi_df["phi"] = pd.to_numeric(chi_df["phi"], errors="coerce")
    chi_df["chi"] = pd.to_numeric(chi_df["chi"], errors="coerce")
    chi_df = chi_df.dropna(subset=["temperature", "phi", "chi"]).reset_index(drop=True)

    cond_contrast_df = _compute_condition_class_contrast(chi_df)
    cond_contrast_df.to_csv(metrics_dir / "chi_class_contrast_by_condition.csv", index=False)
    _plot_condition_heatmap(
        cond_df=cond_contrast_df,
        value_col="delta_mean_chi_miscible_minus_immiscible",
        cbar_label="Δχ (miscible - immiscible)",
        out_png=figures_dir / "chi_class_delta_heatmap.png",
        cmap="RdBu_r",
        dpi=dpi,
    )
    significance_df = cond_contrast_df.copy()
    significance_df["minus_log10_pvalue"] = -np.log10(
        np.clip(pd.to_numeric(significance_df["mannwhitney_pvalue"], errors="coerce"), a_min=1e-12, a_max=1.0)
    )
    _plot_condition_heatmap(
        cond_df=significance_df,
        value_col="minus_log10_pvalue",
        cbar_label="-log10(p-value)",
        out_png=figures_dir / "chi_class_significance_heatmap.png",
        cmap="viridis",
        dpi=dpi,
    )
    _plot_chi_vs_temperature_by_phi(
        df=chi_df,
        out_png=figures_dir / "chi_vs_temperature_by_phi_and_class.png",
        dpi=dpi,
    )
    summary["n_conditions"] = int(len(cond_contrast_df))
    summary["n_significant_conditions"] = int(cond_contrast_df["significant_p_lt_0p05"].sum())
    summary["mean_delta_chi_miscible_minus_immiscible"] = _safe_float(
        pd.to_numeric(cond_contrast_df["delta_mean_chi_miscible_minus_immiscible"], errors="coerce").mean()
    )
    analysis_blocks.append("dataset_thermodynamic_contrast")

    # A2) Step3 target context under class-separated thermodynamics.
    if step3_targets_path is not None and step3_targets_path.exists():
        step3_targets_df = _normalize_columns(pd.read_csv(step3_targets_path))
        req = {"temperature", "phi", "target_chi"}
        if req.issubset(step3_targets_df.columns):
            sub = step3_targets_df[["temperature", "phi", "target_chi"]].copy()
            sub["temperature"] = pd.to_numeric(sub["temperature"], errors="coerce")
            sub["phi"] = pd.to_numeric(sub["phi"], errors="coerce")
            sub["target_chi"] = pd.to_numeric(sub["target_chi"], errors="coerce")
            sub = sub.dropna(subset=["temperature", "phi", "target_chi"]).copy()

            joined = sub.merge(
                cond_contrast_df[
                    [
                        "temperature",
                        "phi",
                        "chi_miscible_mean",
                        "chi_immiscible_mean",
                        "delta_mean_chi_miscible_minus_immiscible",
                    ]
                ],
                on=["temperature", "phi"],
                how="left",
            )
            joined["target_minus_miscible_mean"] = joined["target_chi"] - joined["chi_miscible_mean"]
            joined["target_minus_immiscible_mean"] = joined["target_chi"] - joined["chi_immiscible_mean"]
            joined["abs_target_to_miscible"] = joined["target_minus_miscible_mean"].abs()
            joined["abs_target_to_immiscible"] = joined["target_minus_immiscible_mean"].abs()
            joined["target_closer_to_miscible"] = (
                joined["abs_target_to_miscible"] < joined["abs_target_to_immiscible"]
            ).astype(int)
            joined.to_csv(metrics_dir / "step3_target_context_vs_class_contrast.csv", index=False)

            summary["step3_target_closer_to_miscible_fraction"] = _safe_float(
                pd.to_numeric(joined["target_closer_to_miscible"], errors="coerce").mean()
            )
            _plot_step3_target_vs_class_means(
                joined,
                out_png=figures_dir / "step3_target_vs_class_means.png",
                dpi=dpi,
            )
            analysis_blocks.append("step3_target_context_vs_class_contrast")
        else:
            pd.DataFrame(
                [{"note": "Step3 targets file exists but required columns are missing."}]
            ).to_csv(metrics_dir / "step3_target_context_skipped_reason.csv", index=False)
    else:
        pd.DataFrame(
            [{"note": "Step3 targets not found; skipped target-context analysis."}]
        ).to_csv(metrics_dir / "step3_target_context_skipped_reason.csv", index=False)

    # B) Step4 condition-level error + thermodynamic-gradient consistency.
    if step4_path is not None and step4_path.exists():
        step4_df = _load_step4_predictions(step4_path)
        eval_df = step4_df.copy()
        if "split" in eval_df.columns and (eval_df["split"] == "test").any():
            eval_df = eval_df[eval_df["split"] == "test"].copy()
        eval_df = eval_df.dropna(subset=["temperature", "phi", "chi", "chi_pred"]).reset_index(drop=True)
        cond_err_df = _compute_step4_condition_metrics(eval_df)
        cond_err_df.to_csv(metrics_dir / "step4_condition_error_metrics.csv", index=False)
        _plot_step4_mae_heatmap(cond_err_df, figures_dir / "step4_test_mae_heatmap.png", dpi=dpi)

        grad_t_df = _compute_polymer_gradients(eval_df, axis_col="temperature")
        grad_phi_df = _compute_polymer_gradients(eval_df, axis_col="phi")
        grad_t_df.to_csv(metrics_dir / "step4_gradient_consistency_dchi_dT.csv", index=False)
        grad_phi_df.to_csv(metrics_dir / "step4_gradient_consistency_dchi_dphi.csv", index=False)

        stats_t = _plot_gradient_scatter(
            grad_t_df,
            axis_label="T",
            out_png=figures_dir / "step4_gradient_consistency_dchi_dT.png",
            dpi=dpi,
        )
        stats_phi = _plot_gradient_scatter(
            grad_phi_df,
            axis_label="φ",
            out_png=figures_dir / "step4_gradient_consistency_dchi_dphi.png",
            dpi=dpi,
        )

        summary["step4_test_mae"] = _safe_float(pd.to_numeric(cond_err_df["mae"], errors="coerce").mean())
        summary["step4_dchi_dT_sign_agreement"] = _safe_float(stats_t["sign_agreement_rate"])
        summary["step4_dchi_dphi_sign_agreement"] = _safe_float(stats_phi["sign_agreement_rate"])
        analysis_blocks.append("step4_condition_error_and_gradient_consistency")
    else:
        pd.DataFrame(
            [{"note": "Step4 predictions not found; skipped Step4 consistency analysis."}]
        ).to_csv(metrics_dir / "step4_analysis_skipped_reason.csv", index=False)

    # C) Step5/6 chemistry descriptor + novelty analysis.
    step5_df = _load_target_polymers(step5_targets_path, source_step="step5") if step5_targets_path else pd.DataFrame()
    step6_df = _load_target_polymers(step6_targets_path, source_step="step6") if step6_targets_path else pd.DataFrame()
    selected_df = pd.concat([step5_df, step6_df], ignore_index=True) if (not step5_df.empty or not step6_df.empty) else pd.DataFrame()

    if not selected_df.empty:
        selected_df["canonical_smiles"] = selected_df["canonical_smiles"].where(
            selected_df["canonical_smiles"].notna(),
            selected_df["SMILES"].astype(str).map(canonicalize_smiles),
        )
        selected_df = selected_df[selected_df["canonical_smiles"].notna()].copy()
        selected_unique = selected_df.drop_duplicates(subset=["canonical_smiles"]).reset_index(drop=True)
        selected_unique.to_csv(metrics_dir / "selected_candidates_unique.csv", index=False)
        _plot_selection_tradeoff(
            df=selected_df,
            out_png=figures_dir / "selection_tradeoff_chi_vs_solubility_confidence.png",
            dpi=dpi,
        )

        selected_desc = _attach_descriptors(selected_unique, smiles_col="SMILES")
        selected_desc.to_csv(metrics_dir / "selected_candidates_descriptors.csv", index=False)

        summary["n_selected_unique_candidates"] = int(len(selected_unique))
        summary["n_selected_step5"] = int(step5_df["canonical_smiles"].nunique()) if not step5_df.empty else 0
        summary["n_selected_step6"] = int(step6_df["canonical_smiles"].nunique()) if not step6_df.empty else 0

        if step2_targets_path is not None and step2_targets_path.exists():
            step2_targets_df = _normalize_columns(pd.read_csv(step2_targets_path))
            if "SMILES" in step2_targets_df.columns:
                step2_targets_df = step2_targets_df[["SMILES"]].copy()
                step2_targets_df["canonical_smiles"] = step2_targets_df["SMILES"].astype(str).map(canonicalize_smiles)
                step2_targets_df = step2_targets_df[step2_targets_df["canonical_smiles"].notna()].drop_duplicates(
                    "canonical_smiles"
                )
                step2_targets_desc = _attach_descriptors(step2_targets_df, smiles_col="SMILES")
                step2_targets_desc.to_csv(metrics_dir / "step2_target_pool_descriptors.csv", index=False)
                shift_step2_df = _compare_descriptor_shift(selected_desc, step2_targets_desc)
                shift_step2_df.to_csv(metrics_dir / "descriptor_shift_vs_step2_target_pool.csv", index=False)
                _plot_descriptor_shift(
                    shift_df=shift_step2_df,
                    out_png=figures_dir / "descriptor_shift_vs_step2_target_pool.png",
                    dpi=dpi,
                    top_k=int(args.descriptor_top_k or step7_cfg["descriptor_top_k"]),
                )
                summary["n_step2_target_pool_unique"] = int(len(step2_targets_df))
                analysis_blocks.append("selected_vs_step2_target_pool_descriptor_shift")

        if baseline_train_path is not None and baseline_train_path.exists():
            baseline_df = _load_training_baseline_smiles(
                path=baseline_train_path,
                max_samples=int(args.baseline_max_samples or step7_cfg["baseline_max_samples"]),
                seed=seed_value,
            )
            baseline_df.to_csv(metrics_dir / "baseline_training_smiles_sample.csv", index=False)
            baseline_desc = _attach_descriptors(baseline_df, smiles_col="SMILES")
            baseline_desc.to_csv(metrics_dir / "baseline_training_descriptors.csv", index=False)

            shift_df = _compare_descriptor_shift(selected_desc, baseline_desc)
            shift_df.to_csv(metrics_dir / "descriptor_shift_vs_training.csv", index=False)
            _plot_descriptor_shift(
                shift_df=shift_df,
                out_png=figures_dir / "descriptor_shift_vs_training.png",
                dpi=dpi,
                top_k=int(args.descriptor_top_k or step7_cfg["descriptor_top_k"]),
            )

            novelty_df = _compute_max_similarity_to_baseline(
                selected_smiles=selected_unique["SMILES"].astype(str).tolist(),
                baseline_smiles=baseline_df["SMILES"].astype(str).tolist(),
                novelty_threshold=float(args.novelty_similarity_threshold or step7_cfg["novelty_similarity_threshold"]),
            )
            novelty_df.to_csv(metrics_dir / "novelty_vs_training.csv", index=False)
            _plot_novelty_histogram(
                novelty_df,
                out_png=figures_dir / "novelty_similarity_histogram.png",
                dpi=dpi,
            )
            summary["mean_max_tanimoto_to_training"] = _safe_float(
                pd.to_numeric(novelty_df["max_tanimoto_to_training"], errors="coerce").mean()
            )
            summary["novel_fraction_under_threshold"] = _safe_float(
                pd.to_numeric(novelty_df["is_novel_under_threshold"], errors="coerce").mean()
            )

        if "target_polymer_class" in step6_df.columns and not step6_df.empty:
            class_cov = (
                step6_df[["target_polymer_class", "canonical_smiles"]]
                .drop_duplicates()
                .groupby("target_polymer_class", as_index=False)
                .size()
                .rename(columns={"size": "n_unique_candidates"})
                .sort_values("n_unique_candidates", ascending=False)
            )
            class_cov.to_csv(metrics_dir / "step6_target_polymer_class_coverage.csv", index=False)
            fig, ax = plt.subplots(figsize=(6.8, 4.8))
            sns.barplot(data=class_cov, x="target_polymer_class", y="n_unique_candidates", color="#4c78a8", ax=ax)
            ax.set_xlabel("Target polymer class")
            ax.set_ylabel("Unique selected candidates")
            ax.tick_params(axis="x", rotation=25)
            fig.tight_layout()
            fig.savefig(figures_dir / "step6_target_polymer_class_coverage.png", dpi=dpi)
            plt.close(fig)

        analysis_blocks.append("step5_step6_descriptor_and_novelty_analysis")
    else:
        pd.DataFrame(
            [{"note": "Step5/Step6 target_polymers.csv not found; skipped chemistry candidate analysis."}]
        ).to_csv(metrics_dir / "candidate_analysis_skipped_reason.csv", index=False)

    # D) Physical interpretation of Step4 χ(T,φ) coefficients.
    coeff_df = _load_or_fit_coefficients(step4_coeff_path, chi_df)
    if not coeff_df.empty:
        for c in COEFF_COLUMNS:
            coeff_df[c] = pd.to_numeric(coeff_df[c], errors="coerce")
        coeff_df[CLASS_COL] = pd.to_numeric(coeff_df[CLASS_COL], errors="coerce")
        coeff_df["dchi_dT_ref"] = coeff_df.apply(
            lambda r: _compute_dchi_dT(r, T=293.15, phi=0.2), axis=1
        )
        coeff_df["chi_response_type"] = coeff_df["dchi_dT_ref"].map(_classify_chi_response)
        coeff_df.to_csv(metrics_dir / "per_polymer_coefficients.csv", index=False)

        coeff_stat_df = _coefficient_class_statistics(coeff_df)
        coeff_stat_df.to_csv(metrics_dir / "coefficient_class_statistics.csv", index=False)

        _plot_coefficient_violins(coeff_df, figures_dir=figures_dir, dpi=dpi)
        _plot_coeff_scatter(
            coeff_df,
            col_x="a1",
            col_y="a3",
            figures_dir=figures_dir,
            out_name="coefficient_a1_vs_a3_by_class.png",
            dpi=dpi,
        )
        _plot_dchi_dT_distribution(coeff_df, figures_dir=figures_dir, dpi=dpi)

        summary["n_polymers_with_coefficients"] = int(
            coeff_df[COEFF_COLUMNS].notna().all(axis=1).sum()
        )
        sub_sol = coeff_df[coeff_df[CLASS_COL] == 1]
        sub_ins = coeff_df[coeff_df[CLASS_COL] == 0]
        summary["frac_lcst_like_soluble"] = _safe_float(
            pd.to_numeric((sub_sol["chi_response_type"] == "LCST-like").astype(float), errors="coerce").mean()
        )
        summary["frac_lcst_like_insoluble"] = _safe_float(
            pd.to_numeric((sub_ins["chi_response_type"] == "LCST-like").astype(float), errors="coerce").mean()
        )
        analysis_blocks.append("coefficient_physics_interpretation")
    else:
        pd.DataFrame(
            [{"note": "No coefficients available; skipped Block D coefficient interpretation."}]
        ).to_csv(metrics_dir / "block_d_skipped_reason.csv", index=False)

    # E) Flory-Huggins χ(T,φ) surfaces + spinodal/free-energy analysis.
    if not coeff_df.empty:
        if step3_targets_path is not None and step3_targets_path.exists():
            chi_target_df = _normalize_columns(pd.read_csv(step3_targets_path))
        else:
            chi_target_df = pd.DataFrame(columns=["temperature", "phi", "target_chi"])
        if "target_chi" not in chi_target_df.columns and "chi_target" in chi_target_df.columns:
            chi_target_df["target_chi"] = chi_target_df["chi_target"]
        for c in ["temperature", "phi", "target_chi"]:
            if c in chi_target_df.columns:
                chi_target_df[c] = pd.to_numeric(chi_target_df[c], errors="coerce")
        chi_target_df = chi_target_df.dropna(subset=[c for c in ["temperature", "phi", "target_chi"] if c in chi_target_df.columns])
        if chi_target_df.empty and np.isfinite(_safe_float(summary.get("step3_global_chi_target", np.nan))):
            chi_target_df = pd.DataFrame(
                [{"temperature": 293.15, "phi": 0.2, "target_chi": _safe_float(summary["step3_global_chi_target"])}]
            )

        T_grid = np.array(sorted(pd.to_numeric(chi_df["temperature"], errors="coerce").dropna().unique().tolist()), dtype=float)
        phi_grid = np.array(sorted(pd.to_numeric(chi_df["phi"], errors="coerce").dropna().unique().tolist()), dtype=float)
        if len(T_grid) >= 1 and len(phi_grid) >= 1:
            chi_surface_sol, chi_surface_ins = _compute_chi_surfaces(coeff_df, T_grid=T_grid, phi_grid=phi_grid)
            _plot_chi_surface_heatmap(
                chi_soluble_surface=chi_surface_sol,
                chi_insoluble_surface=chi_surface_ins,
                T_grid=T_grid,
                phi_grid=phi_grid,
                figures_dir=figures_dir,
                dpi=dpi,
            )

        spinodal_df, freeE_df = _compute_spinodal_analysis(coeff_df, chi_df=chi_df, chi_target_df=chi_target_df)
        spinodal_df.to_csv(metrics_dir / "spinodal_miscibility_analysis.csv", index=False)
        freeE_df.to_csv(metrics_dir / "free_energy_analysis_at_target.csv", index=False)

        _plot_spinodal_diagram(
            coeff_df=coeff_df,
            chi_target_df=chi_target_df,
            figures_dir=figures_dir,
            dpi=dpi,
            T_ref=293.15,
        )
        _plot_miscible_fraction_boxplot(spinodal_df, figures_dir=figures_dir, dpi=dpi)
        _plot_free_energy_boxplot(freeE_df, figures_dir=figures_dir, dpi=dpi)

        summary["mean_spinodal_miscible_fraction_soluble"] = _safe_float(
            pd.to_numeric(spinodal_df.loc[spinodal_df[CLASS_COL] == 1, "fraction_below_spinodal"], errors="coerce").mean()
        )
        summary["mean_spinodal_miscible_fraction_insoluble"] = _safe_float(
            pd.to_numeric(spinodal_df.loc[spinodal_df[CLASS_COL] == 0, "fraction_below_spinodal"], errors="coerce").mean()
        )
        analysis_blocks.append("flory_huggins_spinodal_free_energy")
    else:
        pd.DataFrame(
            [{"note": "No coefficients available; skipped Block E spinodal/free-energy analysis."}]
        ).to_csv(metrics_dir / "block_e_skipped_reason.csv", index=False)

    # F) Chemical structure -> χ correlation on labeled dataset.
    chi_desc_input = (
        _normalize_columns(pd.read_csv(step3_dataset_with_split_path))
        if (step3_dataset_with_split_path is not None and step3_dataset_with_split_path.exists())
        else chi_df.copy()
    )
    desc_df = _build_chi_dataset_descriptor_df(chi_desc_input)
    if not desc_df.empty:
        desc_df.to_csv(metrics_dir / "chi_dataset_descriptor_stats.csv", index=False)

        # Functional-group frequency by class.
        fg_rows = []
        for fg in FG_COLUMNS:
            vals_sol = pd.to_numeric(desc_df.loc[desc_df[CLASS_COL] == 1, fg], errors="coerce").dropna().to_numpy(dtype=float)
            vals_ins = pd.to_numeric(desc_df.loc[desc_df[CLASS_COL] == 0, fg], errors="coerce").dropna().to_numpy(dtype=float)
            fg_rows.append(
                {
                    "functional_group": fg.replace("fg_", ""),
                    "fraction_soluble": float(np.mean(vals_sol)) if len(vals_sol) else np.nan,
                    "fraction_insoluble": float(np.mean(vals_ins)) if len(vals_ins) else np.nan,
                    "delta_fraction_soluble_minus_insoluble": float(np.mean(vals_sol) - np.mean(vals_ins))
                    if (len(vals_sol) and len(vals_ins))
                    else np.nan,
                    "mannwhitney_pvalue": _safe_mannwhitney(vals_sol, vals_ins),
                }
            )
        fg_df = pd.DataFrame(fg_rows)
        fg_df.to_csv(metrics_dir / "functional_group_frequency.csv", index=False)

        # Descriptor correlations with chi_mean and water_miscible.
        corr_rows = []
        for dcol in DESCRIPTOR_COLUMNS:
            x = pd.to_numeric(desc_df[dcol], errors="coerce")
            y_chi = pd.to_numeric(desc_df["chi_mean"], errors="coerce")
            y_cls = pd.to_numeric(desc_df[CLASS_COL], errors="coerce")
            mask_chi = np.isfinite(x) & np.isfinite(y_chi)
            mask_cls = np.isfinite(x) & np.isfinite(y_cls)
            if int(mask_chi.sum()) >= 3:
                r_chi = float(pearsonr(x[mask_chi], y_chi[mask_chi])[0])
            else:
                r_chi = np.nan
            if int(mask_cls.sum()) >= 3:
                r_cls = float(pearsonr(x[mask_cls], y_cls[mask_cls])[0])
            else:
                r_cls = np.nan
            corr_rows.append(
                {
                    "descriptor": dcol,
                    "pearson_r_with_chi_mean": r_chi,
                    "pearson_r_with_water_miscible": r_cls,
                    "n_for_chi_corr": int(mask_chi.sum()),
                    "n_for_class_corr": int(mask_cls.sum()),
                }
            )
        corr_out = pd.DataFrame(corr_rows).sort_values(
            "pearson_r_with_chi_mean", key=lambda s: s.abs(), ascending=False
        )
        corr_out.to_csv(metrics_dir / "descriptor_correlation_with_chi.csv", index=False)

        _plot_descriptor_boxplot_by_class(desc_df, figures_dir=figures_dir, dpi=dpi)
        _plot_functional_group_bars(desc_df, figures_dir=figures_dir, dpi=dpi)
        _plot_chi_vs_descriptor_scatter(
            desc_df=desc_df,
            xcol="logp",
            xlabel="LogP",
            out_name="logp_vs_mean_chi_by_class.png",
            figures_dir=figures_dir,
            dpi=dpi,
        )
        _plot_chi_vs_descriptor_scatter(
            desc_df=desc_df,
            xcol="tpsa",
            xlabel="TPSA",
            out_name="tpsa_vs_mean_chi_by_class.png",
            figures_dir=figures_dir,
            dpi=dpi,
        )
        _plot_descriptor_corr_heatmap(desc_df, figures_dir=figures_dir, dpi=dpi)
        summary["n_functional_group_types_analyzed"] = int(len(FG_COLUMNS))
        analysis_blocks.append("structure_property_correlation")
    else:
        pd.DataFrame(
            [{"note": "No valid labeled descriptor rows available; skipped Block F."}]
        ).to_csv(metrics_dir / "block_f_skipped_reason.csv", index=False)

    # H) Classification-dataset context and representativeness vs chi subset.
    class_df = _load_classification_dataset_from_config(config)
    if not class_df.empty:
        overlap_df, desc_shift_cls_df, fg_cmp_df, chi_desc_h, cls_desc_h = _compute_classification_context(chi_df, class_df)
        overlap_df.to_csv(metrics_dir / "classification_dataset_overlap_summary.csv", index=False)
        desc_shift_cls_df.to_csv(metrics_dir / "classification_vs_chi_descriptor_shift.csv", index=False)
        fg_cmp_df.to_csv(metrics_dir / "classification_vs_chi_functional_group_frequency.csv", index=False)
        chi_desc_h.to_csv(metrics_dir / "chi_subset_descriptor_stats_for_classification_context.csv", index=False)
        cls_desc_h.to_csv(metrics_dir / "classification_dataset_descriptor_stats.csv", index=False)

        _plot_classification_overlap_counts(overlap_df, figures_dir=figures_dir, dpi=dpi)
        _plot_classification_descriptor_shift(desc_shift_cls_df, figures_dir=figures_dir, dpi=dpi)
        _plot_classification_fg_shift(fg_cmp_df, figures_dir=figures_dir, dpi=dpi)

        summary["n_classification_polymers"] = int(len(class_df))
        cov = overlap_df.loc[overlap_df["metric"] == "chi_coverage_in_classification", "value"]
        summary["chi_coverage_in_classification"] = float(cov.iloc[0]) if not cov.empty else np.nan
        analysis_blocks.append("classification_dataset_context_analysis")
    else:
        pd.DataFrame(
            [{"note": "Classification dataset not found or unreadable; skipped Block H."}]
        ).to_csv(metrics_dir / "block_h_skipped_reason.csv", index=False)

    # I) Step1 embedding + PINN coefficient coupling analysis.
    class_for_embed = class_df
    if class_for_embed.empty:
        chi_only = _normalize_columns(chi_df)[["Polymer", "SMILES", CLASS_COL]].drop_duplicates("SMILES").copy()
        chi_only["canonical_smiles"] = chi_only["SMILES"].astype(str).map(canonicalize_smiles)
        class_for_embed = chi_only[chi_only["canonical_smiles"].notna()].copy()

    union_embed_df, mismatch_count = _build_embedding_union_polymer_df(chi_df, class_for_embed)
    if union_embed_df.empty:
        pd.DataFrame(
            [{"note": "No polymers available to compute Step1 embedding analysis."}]
        ).to_csv(metrics_dir / "block_i_skipped_reason.csv", index=False)
    else:
        try:
            embedding_cache_path.parent.mkdir(parents=True, exist_ok=True)
            emb_coords, emb_info = _compute_step1_embedding_coordinates(
                config=config,
                union_df=union_embed_df,
                cache_npz=embedding_cache_path,
                model_size=model_size,
                split_mode=split_mode,
            )
            emb_out = emb_coords[
                [
                    "polymer_id",
                    "Polymer",
                    "SMILES",
                    "canonical_smiles",
                    CLASS_COL,
                    "source_group",
                    "PC1",
                    "PC2",
                    "embedding_norm",
                ]
            ].copy()
            emb_out.to_csv(metrics_dir / "step1_embedding_coordinates.csv", index=False)
            _plot_step1_embedding_pca(emb_coords, figures_dir=figures_dir, dpi=dpi)

            emb_auc_df = _compute_embedding_classification_signal(emb_coords, seed=seed_value)
            emb_auc_df.to_csv(metrics_dir / "step1_embedding_classification_signal.csv", index=False)

            summary["n_step1_embedding_polymers"] = int(len(emb_coords))
            summary["embedding_label_disagreement_count"] = int(mismatch_count)
            summary["step1_embedding_pc1_explained_variance"] = float(
                emb_info.get("pc1_explained_variance_ratio", np.nan)
            )
            summary["step1_embedding_pc2_explained_variance"] = float(
                emb_info.get("pc2_explained_variance_ratio", np.nan)
            )
            auc_all = emb_auc_df.loc[emb_auc_df["subset"] == "all_union", "auc_mean"]
            summary["step1_embedding_cv_auc"] = float(auc_all.iloc[0]) if not auc_all.empty else np.nan

            emb_pinn_corr_df = _compute_embedding_pinn_correlations(emb_coords, coeff_df, chi_df)
            emb_pinn_corr_df.to_csv(metrics_dir / "embedding_pinn_correlation.csv", index=False)
            _plot_embedding_pinn_corr_heatmap(emb_pinn_corr_df, figures_dir=figures_dir, dpi=dpi)

            T_grid = np.array(
                sorted(pd.to_numeric(chi_df["temperature"], errors="coerce").dropna().unique().tolist()),
                dtype=float,
            )
            phi_grid = np.array(
                sorted(pd.to_numeric(chi_df["phi"], errors="coerce").dropna().unique().tolist()),
                dtype=float,
            )
            sens_poly_df, sens_class_df = _compute_pinn_coefficient_sensitivity(
                coeff_df=coeff_df,
                T_grid=T_grid if len(T_grid) else np.array([293.15], dtype=float),
                phi_grid=phi_grid if len(phi_grid) else np.array([0.2], dtype=float),
            )
            sens_poly_df.to_csv(metrics_dir / "pinn_coefficient_sensitivity.csv", index=False)
            sens_class_df.to_csv(metrics_dir / "pinn_coefficient_sensitivity_by_class.csv", index=False)
            _plot_pinn_coefficient_sensitivity(sens_poly_df, figures_dir=figures_dir, dpi=dpi)

            if not sens_class_df.empty:
                sol = sens_class_df[sens_class_df[CLASS_COL] == 1].sort_values("mean", ascending=False)
                ins = sens_class_df[sens_class_df[CLASS_COL] == 0].sort_values("mean", ascending=False)
                summary["pinn_top_driver_soluble"] = str(sol["coefficient"].iloc[0]) if not sol.empty else ""
                summary["pinn_top_driver_insoluble"] = str(ins["coefficient"].iloc[0]) if not ins.empty else ""

            analysis_blocks.append("step1_embedding_and_pinn_coupling_analysis")
        except Exception as exc:
            pd.DataFrame(
                [{"note": f"Step1 embedding analysis failed and was skipped: {exc}"}]
            ).to_csv(metrics_dir / "block_i_skipped_reason.csv", index=False)

    # G) Discovered candidates vs known chemical space comparison.
    inv_frames = []
    if step5_inverse_candidates_path is not None and step5_inverse_candidates_path.exists():
        inv_frames.append(_load_inverse_candidates(step5_inverse_candidates_path, source_step="step5"))
    if step6_inverse_candidates_path is not None and step6_inverse_candidates_path.exists():
        inv_frames.append(_load_inverse_candidates(step6_inverse_candidates_path, source_step="step6"))
    inv_all_df = pd.concat(inv_frames, ignore_index=True) if inv_frames else pd.DataFrame()

    if not inv_all_df.empty:
        inv_all_df.to_csv(metrics_dir / "inverse_candidates_all_combined.csv", index=False)

        known_source = desc_df if not desc_df.empty else _build_chi_dataset_descriptor_df(chi_df)
        known_sol_df = known_source[known_source[CLASS_COL] == 1].copy()
        known_ins_df = known_source[known_source[CLASS_COL] == 0].copy()
        discovered_smiles = (
            inv_all_df["SMILES"].astype(str).map(canonicalize_smiles).dropna().drop_duplicates().tolist()
        )
        discovered_df = pd.DataFrame({"SMILES": discovered_smiles})
        discovered_desc = _attach_descriptors(discovered_df, smiles_col="SMILES")

        pca_df = _compute_pca_coordinates(
            {
                "known_soluble": known_sol_df["SMILES"].astype(str).dropna().tolist() if not known_sol_df.empty else [],
                "known_insoluble": known_ins_df["SMILES"].astype(str).dropna().tolist() if not known_ins_df.empty else [],
                "discovered_step5": inv_all_df.loc[inv_all_df["source_step"] == "step5", "SMILES"].astype(str).tolist(),
                "discovered_step6": inv_all_df.loc[inv_all_df["source_step"] == "step6", "SMILES"].astype(str).tolist(),
            },
            n_components=2,
        )
        pca_df.to_csv(metrics_dir / "chemical_space_pca_coordinates.csv", index=False)
        _plot_chemical_space_pca(pca_df, figures_dir=figures_dir, dpi=dpi)

        disc_stats = _compute_discovered_descriptor_stats(known_sol_df, known_ins_df, discovered_desc)
        disc_stats.to_csv(metrics_dir / "discovered_descriptor_stats.csv", index=False)
        _plot_discovered_descriptor_boxplot(
            known_sol=known_sol_df,
            known_ins=known_ins_df,
            discovered=discovered_desc,
            figures_dir=figures_dir,
            dpi=dpi,
        )
        _plot_chi_scoring_landscape(inv_all_df, figures_dir=figures_dir, dpi=dpi)
        analysis_blocks.append("discovered_vs_known_chemical_space")
    else:
        pd.DataFrame(
            [{"note": "No inverse_candidates_all.csv found from Step5/6; skipped Block G."}]
        ).to_csv(metrics_dir / "block_g_skipped_reason.csv", index=False)

    summary["analysis_blocks_completed"] = int(len(analysis_blocks))
    summary["analysis_block_names"] = ",".join(analysis_blocks)

    figure_index_df = _write_figure_index(figures_dir=figures_dir, metrics_dir=metrics_dir)
    summary["n_numbered_figures"] = int(len(figure_index_df))
    _write_science_highlights(summary, metrics_dir / "science_highlights.md")
    with open(metrics_dir / "step7_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    save_step_summary(summary, metrics_dir)
    save_artifact_manifest(step_dir=step_dir, metrics_dir=metrics_dir, figures_dir=figures_dir)

    print("Step 7 complete.")
    print(f"Output directory: {step_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 7: chemistry + physics analysis")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--model_size", type=str, default="small", help="small|medium|large|xl")
    parser.add_argument("--split_mode", type=str, default=None, help="polymer|random")
    parser.add_argument("--chi_dataset_path", type=str, default=None, help="Override chi dataset CSV")
    parser.add_argument("--step4_predictions_csv", type=str, default=None, help="Override Step4 chi_predictions_all.csv")
    parser.add_argument("--step5_targets_csv", type=str, default=None, help="Override Step5 target_polymers.csv")
    parser.add_argument("--step6_targets_csv", type=str, default=None, help="Override Step6 target_polymers.csv")
    parser.add_argument("--baseline_train_csv", type=str, default=None, help="Override baseline training smiles CSV")
    parser.add_argument("--baseline_max_samples", type=int, default=None, help="Max baseline smiles sampled for descriptor/novelty analysis")
    parser.add_argument("--novelty_similarity_threshold", type=float, default=None, help="Novelty threshold on max Tanimoto similarity")
    parser.add_argument("--descriptor_top_k", type=int, default=None, help="Top-K descriptors shown in descriptor shift figure")
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--dpi", type=int, default=None, help="Figure DPI")
    parser.add_argument("--font_size", type=int, default=None, help="Figure font size")
    main(parser.parse_args())
