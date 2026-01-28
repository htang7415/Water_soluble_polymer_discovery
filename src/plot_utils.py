from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_FIGSIZE = (5, 5)
DEFAULT_FONT_SIZE = 12


def _apply_style(figsize=DEFAULT_FIGSIZE, font_size=DEFAULT_FONT_SIZE):
    plt.rcParams.update({"font.size": font_size})
    plt.figure(figsize=figsize)


def save_loss_plot(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    out_path: str,
    figsize=DEFAULT_FIGSIZE,
    font_size=DEFAULT_FONT_SIZE,
):
    _apply_style(figsize, font_size)
    plt.plot(train_losses, label="train")
    if val_losses:
        plt.plot(val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def parity_plot(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    out_path: str,
    title: Optional[str] = None,
    metrics: Optional[Dict[str, float]] = None,
    figsize=DEFAULT_FIGSIZE,
    font_size=DEFAULT_FONT_SIZE,
):
    _apply_style(figsize, font_size)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    min_v = min(y_true.min(), y_pred.min())
    max_v = max(y_true.max(), y_pred.max())
    plt.scatter(y_true, y_pred, s=12, alpha=0.7)
    plt.plot([min_v, max_v], [min_v, max_v], "k--", lw=1)
    plt.xlabel("True")
    plt.ylabel("Pred")
    if title:
        plt.title(title)
    if metrics:
        text = "\n".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
        plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, va="top")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def hist_plot(
    data_list: List[Sequence[float]],
    labels: List[str],
    out_path: str,
    xlabel: str,
    ylabel: str = "count",
    figsize=DEFAULT_FIGSIZE,
    font_size=DEFAULT_FONT_SIZE,
):
    _apply_style(figsize, font_size)
    for data, label in zip(data_list, labels):
        plt.hist(data, bins=30, alpha=0.6, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if labels:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def bar_plot(
    labels: List[str],
    values: List[Sequence[float]],
    value_labels: List[str],
    out_path: str,
    ylabel: str,
    figsize=DEFAULT_FIGSIZE,
    font_size=DEFAULT_FONT_SIZE,
):
    _apply_style(figsize, font_size)
    x = np.arange(len(labels))
    width = 0.35
    for i, vals in enumerate(values):
        plt.bar(x + i * width, vals, width, label=value_labels[i])
    plt.xticks(x + width * (len(values) - 1) / 2, labels)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def pr_curve_plot(precision: Sequence[float], recall: Sequence[float], out_path: str):
    _apply_style(DEFAULT_FIGSIZE, DEFAULT_FONT_SIZE)
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def roc_curve_plot(fpr: Sequence[float], tpr: Sequence[float], out_path: str):
    _apply_style(DEFAULT_FIGSIZE, DEFAULT_FONT_SIZE)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def coverage_plot(expected: Sequence[float], observed: Sequence[float], out_path: str):
    _apply_style(DEFAULT_FIGSIZE, DEFAULT_FONT_SIZE)
    plt.plot(expected, observed, "o-")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("Expected coverage")
    plt.ylabel("Observed coverage")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
