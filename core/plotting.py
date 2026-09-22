"""Stateless plotting functions used by training and experiment reports."""

from __future__ import annotations
from typing import Any
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
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


def plot_data_ablation_summary(output_dir, summary_df, all_results):
    pdf_path = output_dir / "summary_report.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(14, max(5, 0.45 * len(summary_df) + 2)))
        ax.axis("off")
        ax.set_title("MG supervised BI-RADS baselines: summary", fontsize=14, pad=16)
        display_cols = [
            "status",
            "experiment",
            "train_size_requested",
            "test_accuracy",
            "test_balanced_accuracy",
            "test_macro_f1",
            "best_epoch",
            "reason",
        ]
        display_df = summary_df[[c for c in display_cols if c in summary_df.columns]].copy()
        for col in ["test_accuracy", "test_balanced_accuracy", "test_macro_f1"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
        table = ax.table(
            cellText=display_df.fillna("").values, colLabels=display_df.columns, loc="center", cellLoc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.25)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        completed = summary_df[summary_df["status"] == "completed"].copy()
        if not completed.empty:
            fig, ax = plt.subplots(figsize=(13, 5))
            x = np.arange(len(completed))
            ax.bar(x, completed["test_balanced_accuracy"].astype(float).values)
            ax.set_xticks(x)
            ax.set_xticklabels(completed["experiment"].astype(str).tolist(), rotation=45, ha="right")
            ax.set_ylabel("Test balanced accuracy")
            ax.set_ylim(0.0, 1.0)
            ax.set_title("Test balanced accuracy by experiment")
            ax.grid(axis="y", alpha=0.25)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        for result in all_results:
            if result.get("status") != "completed":
                continue
            exp_name = result.get("experiment", {}).get("name", "unknown")
            cm = result.get("test_metrics_best_model", {}).get("confusion_matrix")
            if cm is None:
                continue
            arr = np.array(cm, dtype=int)
            fig, ax = plt.subplots(figsize=(5.6, 5.0))
            im = ax.imshow(arr)
            ax.set_title(f"{exp_name}\nTest confusion matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_xticks(range(len(CLASS_NAMES)))
            ax.set_yticks(range(len(CLASS_NAMES)))
            ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
            ax.set_yticklabels(CLASS_NAMES)
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    ax.text(j, i, str(arr[i, j]), ha="center", va="center")
            fig.colorbar(im, ax=ax)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"Wrote aggregate summary: {output_dir / 'summary.csv'}")
    print(f"Wrote aggregate JSON:    {output_dir / 'summary.json'}")
    print(f"Wrote summary PDF:       {pdf_path}")


def plot_resolution_summary(output_dir, summary_df, results):
    pdf_path = output_dir / "summary_report.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(14, max(5, 0.45 * max(1, len(summary_df)) + 2)))
        ax.axis("off")
        ax.set_title("MG supervised resolution/backbone baseline", fontsize=14, pad=16)
        display_cols = [
            "status",
            "backbone",
            "image_size",
            "epochs_run",
            "test_balanced_accuracy",
            "test_macro_f1",
            "test_accuracy",
            "best_epoch_val_bal_acc",
        ]
        display_df = summary_df[[c for c in display_cols if c in summary_df.columns]].copy()
        for col in ["test_accuracy", "test_balanced_accuracy", "test_macro_f1"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
        table = ax.table(
            cellText=display_df.fillna("").values, colLabels=display_df.columns, loc="center", cellLoc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.25)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        completed = summary_df[summary_df["status"] == "completed"].copy()
        if not completed.empty:
            labels = [
                f"{b}\n{int(s)}" for b, s in zip(completed["backbone"].astype(str), completed["image_size"].astype(int))
            ]
            for metric, title in [
                ("test_balanced_accuracy", "Test balanced accuracy"),
                ("test_macro_f1", "Test macro F1"),
            ]:
                fig, ax = plt.subplots(figsize=(11, 5))
                x = np.arange(len(completed))
                ax.bar(x, completed[metric].astype(float).values)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=30, ha="right")
                ax.set_ylabel(title)
                ax.set_title(f"{title} by backbone and image size")
                ax.grid(axis="y", alpha=0.25)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        for result in results:
            if result.get("status") != "completed":
                continue
            exp_name = result.get("experiment", {}).get("name", "unknown")
            cm = (result.get("test_metrics_primary_best_bal_acc", {}) or {}).get("confusion_matrix")
            if cm is None:
                continue
            arr = np.array(cm, dtype=int)
            fig, ax = plt.subplots(figsize=(5.6, 5.0))
            im = ax.imshow(arr)
            ax.set_title(f"{exp_name}\nTest confusion matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_xticks(range(len(CLASS_NAMES)))
            ax.set_yticks(range(len(CLASS_NAMES)))
            ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
            ax.set_yticklabels(CLASS_NAMES)
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    ax.text(j, i, str(arr[i, j]), ha="center", va="center")
            fig.colorbar(im, ax=ax)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"Wrote summary CSV:  {output_dir / 'summary.csv'}")
    print(f"Wrote summary JSON: {output_dir / 'summary.json'}")
    print(f"Wrote summary PDF:  {pdf_path}")


def plot_baseline_history(history, out_path):
    if not history:
        return
    if isinstance(history, list):
        history = {key: [row[key] for row in history] for key in history[0]}
    keys = ["train_loss", "val_loss", "val_balanced_accuracy", "val_macro_f1", "lr", "train_accuracy"]
    plot_history_grid(history, keys, np.arange(1, len(history["train_loss"]) + 1), (2, 3), (14, 8), out_path)
