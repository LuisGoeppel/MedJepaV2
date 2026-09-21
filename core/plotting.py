"""Stateless plotting functions used by training and experiment reports."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data import CLASS_NAMES


def plot_history_grid(history, keys, epochs, shape, figsize, path):
    """Plot one history series per panel, with consistent labels and styling."""
    figure, axes = plt.subplots(*shape, figsize=figsize, squeeze=False)
    for axis, key in zip(axes.flat, keys):
        if key in history:
            axis.plot(epochs, history[key])
        axis.set_title(key)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def plot_training_history(history: dict[str, list[float]], dirs: dict[str, Path]) -> None:
    epochs = np.arange(1, len(history["lejepa"]) + 1)
    plot_history_grid(
        history,
        ["lejepa", "invariance", "sigreg", "lr"],
        epochs,
        (1, 4),
        (16, 4),
        dirs["plots"] / "training_history.png",
    )

    plt.figure(figsize=(15, 8))
    diag_groups = [
        ("standard deviation", ["raw_proj_std", "loss_proj_std", "emb_std"]),
        ("mean vector norm", ["raw_proj_norm", "loss_proj_norm", "emb_norm"]),
        ("effective rank", ["raw_proj_effective_rank", "loss_proj_effective_rank", "emb_effective_rank"]),
    ]
    for row, (title, keys) in enumerate(diag_groups):
        ax = plt.subplot(3, 1, row + 1)
        for key in keys:
            if key in history and len(history[key]) == len(epochs):
                ax.plot(epochs, history[key], label=key)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(dirs["plots"] / "training_diagnostics.png", dpi=200)
    plt.savefig(dirs["plots"] / "projection_embedding_diagnostics.png", dpi=200)
    plt.close()

    plot_history_grid(
        history,
        ["lejepa", "invariance", "sigreg", "raw_proj_std", "loss_proj_std", "emb_std"],
        epochs,
        (2, 3),
        (14, 8),
        dirs["plots"] / "collapse_diagnostics.png",
    )

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history.get("epoch_time_sec", []))
    plt.title("epoch_time_sec")
    plt.xlabel("Epoch")
    plt.ylabel("Seconds")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(dirs["plots"] / "epoch_times.png", dpi=200)
    plt.close()

    timing_keys = [
        "data_wait_and_augmentation_time_sec",
        "h2d_transfer_time_sec",
        "forward_loss_time_sec",
        "backward_optimizer_time_sec",
        "metrics_bookkeeping_time_sec",
    ]
    plt.figure(figsize=(10, 5))
    for key in timing_keys:
        if key in history:
            plt.plot(epochs, history[key], label=key.replace("_time_sec", ""))
    plt.title("Epoch timing components")
    plt.xlabel("Epoch")
    plt.ylabel("Seconds")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(dirs["plots"] / "epoch_timing_components.png", dpi=200)
    plt.close()

    fraction_keys = [
        "data_wait_and_augmentation_fraction",
        "h2d_transfer_fraction",
        "forward_loss_fraction",
        "backward_optimizer_fraction",
        "metrics_bookkeeping_fraction",
    ]
    plt.figure(figsize=(10, 5))
    for key in fraction_keys:
        if key in history:
            plt.plot(epochs, history[key], label=key.replace("_fraction", ""))
    plt.title("Epoch timing fractions")
    plt.xlabel("Epoch")
    plt.ylabel("Fraction of epoch wall time")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(dirs["plots"] / "epoch_timing_fractions.png", dpi=200)
    plt.close()


def plot_history(history: dict[str, list[Any]], out_dir: Path) -> None:
    keys = ["train_loss", "train_accuracy", "val_loss", "val_balanced_accuracy", "val_macro_f1", "lr_head"]
    plot_history_grid(
        history, keys, np.arange(1, len(history["train_loss"]) + 1), (2, 3), (14, 8), out_dir / "training_curves.png"
    )


def plot_confusion(cm: list[list[int]], title: str, out_path: Path) -> None:
    arr = np.asarray(cm)
    plt.figure(figsize=(6, 5))
    plt.imshow(arr)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(3), CLASS_NAMES, rotation=30, ha="right")
    plt.yticks(range(3), CLASS_NAMES)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            plt.text(j, i, str(int(arr[i, j])), ha="center", va="center")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_class_distribution(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_path: Path,
) -> None:
    splits = {"train_used": train_df, "val": val_df, "test": test_df}
    counts = np.array([[int((df["target_collapsed"] == c).sum()) for c in range(3)] for df in splits.values()])
    x = np.arange(len(splits))
    bottom = np.zeros(len(splits))
    plt.figure(figsize=(7, 5))
    for c, name in enumerate(CLASS_NAMES):
        plt.bar(x, counts[:, c], bottom=bottom, label=name)
        bottom += counts[:, c]
    plt.xticks(x, list(splits.keys()))
    plt.ylabel("Rows")
    plt.title("Class distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_metric_curve(mean_df: pd.DataFrame, metric: str, ylabel: str, title: str, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for mode, sub in mean_df.groupby("mode"):
        sub = sub.sort_values("budget_numeric")
        plt.plot(sub["budget_numeric"], sub[metric], marker="o", label=mode)
    plt.xscale("log")
    plt.xlabel("Labeled train samples")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_summary_outputs(df: pd.DataFrame, mean_df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return

    _plot_metric_curve(
        mean_df,
        "test_balanced_accuracy",
        "Test balanced accuracy",
        "Label-efficiency comparison",
        out_dir / "summary_test_balanced_accuracy.png",
    )
    _plot_metric_curve(
        mean_df,
        "test_macro_f1",
        "Test macro F1",
        "Label-efficiency macro-F1 comparison",
        out_dir / "summary_test_macro_f1.png",
    )
    _plot_metric_curve(
        mean_df,
        "test_min_class_recall",
        "Minimum test class recall",
        "Worst-class recall across label budgets",
        out_dir / "summary_test_min_class_recall.png",
    )

    # Per-class recall summary.
    plt.figure(figsize=(12, 4))
    for i, class_name in enumerate(CLASS_NAMES):
        ax = plt.subplot(1, 3, i + 1)
        metric = f"{class_name}_recall"
        for mode, sub in mean_df.groupby("mode"):
            sub = sub.sort_values("budget_numeric")
            ax.plot(sub["budget_numeric"], sub[metric], marker="o", label=mode)
        ax.set_xscale("log")
        ax.set_title(f"{class_name} recall")
        ax.set_xlabel("Labeled samples")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_ylabel("Test recall")
        if i == 2:
            ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "summary_test_per_class_recall.png", dpi=200)
    plt.close()

    # Best epoch by mode.
    plt.figure(figsize=(8, 5))
    for mode, sub in mean_df.groupby("mode"):
        sub = sub.sort_values("budget_numeric")
        plt.plot(sub["budget_numeric"], sub["best_epoch"], marker="o", label=mode)
    plt.xscale("log")
    plt.xlabel("Labeled train samples")
    plt.ylabel("Best epoch")
    plt.title("Best checkpoint epoch")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "summary_best_epoch.png", dpi=200)
    plt.close()

    # Subset schedule is identical across modes for a seed; average across duplicates/modes.
    schedule = (
        df.groupby(["budget", "budget_numeric"], as_index=False)[
            [
                "effective_balance_degree",
                "train_routine",
                "train_follow_up",
                "train_biopsy",
            ]
        ]
        .mean()
        .sort_values("budget_numeric")
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        schedule["budget_numeric"],
        schedule["effective_balance_degree"],
        marker="o",
    )
    plt.xscale("log")
    plt.xlabel("Labeled train samples")
    plt.ylabel("Effective balance degree")
    plt.ylim(-0.03, 1.03)
    plt.title("Progressive balance schedule")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "summary_balance_degree.png", dpi=200)
    plt.close()

    # Normalized class composition across budgets.
    total = (schedule["train_routine"] + schedule["train_follow_up"] + schedule["train_biopsy"]).to_numpy()
    x = np.arange(len(schedule))
    routine = schedule["train_routine"].to_numpy() / total
    follow = schedule["train_follow_up"].to_numpy() / total
    biopsy = schedule["train_biopsy"].to_numpy() / total
    plt.figure(figsize=(9, 5))
    plt.bar(x, routine, label="routine")
    plt.bar(x, follow, bottom=routine, label="follow_up")
    plt.bar(x, biopsy, bottom=routine + follow, label="biopsy")
    plt.xticks(x, schedule["budget"].astype(str).tolist())
    plt.xlabel("Label budget")
    plt.ylabel("Fraction of training subset")
    plt.title("Training-subset class composition")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "summary_subset_class_composition.png", dpi=200)
    plt.close()

    # Predicted class fractions at the selected checkpoints. This exposes class collapse.
    plt.figure(figsize=(12, 4))
    for i, mode in enumerate(sorted(mean_df["mode"].unique())):
        ax = plt.subplot(1, len(mean_df["mode"].unique()), i + 1)
        sub = mean_df[mean_df["mode"] == mode].sort_values("budget_numeric")
        for class_name in CLASS_NAMES:
            ax.plot(
                sub["budget_numeric"],
                sub[f"pred_{class_name}_fraction"],
                marker="o",
                label=class_name,
            )
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_title(mode)
        ax.set_xlabel("Labeled samples")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_ylabel("Predicted test fraction")
        if i == len(mean_df["mode"].unique()) - 1:
            ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "summary_predicted_class_fractions.png", dpi=200)
    plt.close()
