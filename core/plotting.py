"""Stateless plotting functions used by training and experiment reports."""

from __future__ import annotations
import math
import os
import base64
import io
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


def as_float_array(values: Any) -> np.ndarray:
    if not isinstance(values, list):
        return np.asarray([], dtype=float)
    out: list[float] = []
    for value in values:
        try:
            x = float(value)
        except Exception:
            x = math.nan
        out.append(x)
    return np.asarray(out, dtype=float)


def history_len(history: dict[str, Any]) -> int:
    if "lejepa" in history and isinstance(history["lejepa"], list):
        return len(history["lejepa"])
    lengths = [len(v) for v in history.values() if isinstance(v, list)]
    return max(lengths) if lengths else 0


def plot_series(ax: plt.Axes, epochs: np.ndarray, history: dict[str, Any], key: str, label: str | None = None) -> bool:
    y = as_float_array(history.get(key, []))
    if len(y) == 0:
        return False
    n = min(len(epochs), len(y))
    if n == 0:
        return False
    ax.plot(epochs[:n], y[:n], label=label or key)
    return True


def savefig_atomic(path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # Keep the final image extension as the actual suffix so that matplotlib
    # can infer/write the correct file format. For example, use
    # training_history.tmp.png instead of training_history.png.tmp.
    suffix = path.suffix.lstrip(".")
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)

    plt.savefig(tmp, dpi=dpi, format=suffix, bbox_inches="tight")
    os.replace(tmp, path)


def plot_training_history(history: dict[str, Any], output_dir: Path | dict[str, Path], dpi: int = 200) -> None:
    # Training passes its output-directory mapping; the CLI passes a path.
    output_dir = Path(output_dir["plots"] if isinstance(output_dir, dict) else output_dir)
    n_epochs = history_len(history)
    if n_epochs == 0:
        raise ValueError("No epoch history found. Is training_history.json still empty?")
    epochs = np.arange(1, n_epochs + 1)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Main objective history.
    plt.figure(figsize=(16, 4))
    for i, key in enumerate(["lejepa", "invariance", "sigreg", "lr"]):
        ax = plt.subplot(1, 4, i + 1)
        plot_series(ax, epochs, history, key)
        ax.set_title(key)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig_atomic(output_dir / "training_history.png", dpi=dpi)
    plt.close()

    # 2) Collapse diagnostics in one compact 2x3 grid.
    plt.figure(figsize=(15, 8))
    for i, key in enumerate(["lejepa", "invariance", "sigreg", "raw_proj_std", "loss_proj_std", "emb_std"]):
        ax = plt.subplot(2, 3, i + 1)
        plot_series(ax, epochs, history, key)
        ax.set_title(key)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig_atomic(output_dir / "collapse_diagnostics.png", dpi=dpi)
    plt.close()

    # 3) Projection/embedding diagnostics.
    plt.figure(figsize=(15, 8))
    diag_groups = [
        ("standard deviation", ["raw_proj_std", "loss_proj_std", "emb_std"]),
        ("mean vector norm", ["raw_proj_norm", "loss_proj_norm", "emb_norm"]),
        ("effective rank", ["raw_proj_effective_rank", "loss_proj_effective_rank", "emb_effective_rank"]),
    ]
    for row, (title, keys) in enumerate(diag_groups):
        ax = plt.subplot(3, 1, row + 1)
        any_line = False
        for key in keys:
            any_line = plot_series(ax, epochs, history, key) or any_line
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        if any_line:
            ax.legend(fontsize=8)
    plt.tight_layout()
    savefig_atomic(output_dir / "training_diagnostics.png", dpi=dpi)
    savefig_atomic(output_dir / "projection_embedding_diagnostics.png", dpi=dpi)
    plt.close()

    # 4) Epoch time.
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    plot_series(ax, epochs, history, "epoch_time_sec")
    ax.set_title("epoch_time_sec")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Seconds")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig_atomic(output_dir / "epoch_times.png", dpi=dpi)
    plt.close()

    # 5) Timing components.
    timing_keys = [
        "data_wait_and_augmentation_time_sec",
        "h2d_transfer_time_sec",
        "forward_loss_time_sec",
        "backward_optimizer_time_sec",
        "metrics_bookkeeping_time_sec",
    ]
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    any_line = False
    for key in timing_keys:
        any_line = plot_series(ax, epochs, history, key, label=key.replace("_time_sec", "")) or any_line
    ax.set_title("Epoch timing components")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Seconds")
    ax.grid(alpha=0.3)
    if any_line:
        ax.legend(fontsize=8)
    plt.tight_layout()
    savefig_atomic(output_dir / "epoch_timing_components.png", dpi=dpi)
    plt.close()

    # 6) Timing fractions.
    fraction_keys = [
        "data_wait_and_augmentation_fraction",
        "h2d_transfer_fraction",
        "forward_loss_fraction",
        "backward_optimizer_fraction",
        "metrics_bookkeeping_fraction",
    ]
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    any_line = False
    for key in fraction_keys:
        any_line = plot_series(ax, epochs, history, key, label=key.replace("_fraction", "")) or any_line
    ax.set_title("Epoch timing fractions")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Fraction of epoch wall time")
    ax.grid(alpha=0.3)
    if any_line:
        ax.legend(fontsize=8)
    plt.tight_layout()
    savefig_atomic(output_dir / "epoch_timing_fractions.png", dpi=dpi)
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
    plt.xlabel("Training entries (including repeats)")
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
        ax.set_xlabel("Training entries (including repeats)")
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
    plt.xlabel("Training entries (including repeats)")
    plt.ylabel("Best epoch")
    plt.title("Best checkpoint epoch")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "summary_best_epoch.png", dpi=200)
    plt.close()

    if (df["subset_strategy"] == "oversampling").all():
        for name in ("summary_balance_degree.png", "summary_subset_class_composition.png"):
            (out_dir / name).unlink(missing_ok=True)
        # Selection is shared across modes. Count each seed/budget only once.
        schedule = df.drop_duplicates(["seed", "budget"]).groupby(
            ["budget", "budget_numeric"], as_index=False
        )[[f"train_unique_{name}" for name in CLASS_NAMES] + ["train_unique_rows", "train_repeated_rows"]].mean()
        schedule = schedule.sort_values("budget_numeric")
        x = np.arange(len(schedule))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for i, name in enumerate(CLASS_NAMES):
            axes[0].bar(x + (i - 1) * 0.25, schedule[f"train_unique_{name}"], width=0.25, label=name)
        axes[0].set(title="Unique images per class", ylabel="Unique training images (mean across seeds)")
        axes[1].bar(x, schedule["train_unique_rows"], label="Unique images")
        axes[1].bar(x, schedule["train_repeated_rows"], bottom=schedule["train_unique_rows"], label="Repeated entries")
        axes[1].set(title="Training entries: unique and repeated", ylabel="Number of entries")
        for ax in axes:
            ax.set_xticks(x, schedule["budget"].astype(str))
            ax.set_xlabel("Training-entry budget (including repeats)")
            ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "summary_sampling_unique_images.png", dpi=200)
        plt.close(fig)
    else:
        # Subset schedule is identical across modes for a seed; average across duplicates/modes.
        (out_dir / "summary_sampling_unique_images.png").unlink(missing_ok=True)
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
        plt.xlabel("Training entries (including repeats)")
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
        plt.xlabel("Training-entry budget")
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
        ax.set_xlabel("Training entries (including repeats)")
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


def figure_to_base64(fig, dpi=135):
    """Serialize and close a figure for a self-contained HTML report."""
    with io.BytesIO() as buffer:
        fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def plot_finite_histogram(ax, values, bins=60):
    """Handle nearly constant floating-point data without duplicate bin edges."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    limits = None
    if len(values) and np.isclose(values.min(), values.max(), rtol=1e-12, atol=1e-12):
        center = float(values.mean())
        margin = max(1.0, abs(center)) * 0.01
        limits = (center - margin, center + margin)
    return ax.hist(values, bins=bins, range=limits)


def plot_baseline_history(history, out_path):
    if not history:
        return
    if isinstance(history, list):
        history = {key: [row[key] for row in history] for key in history[0]}
    keys = ["train_loss", "val_loss", "val_balanced_accuracy", "val_macro_f1", "lr", "train_accuracy"]
    plot_history_grid(history, keys, np.arange(1, len(history["train_loss"]) + 1), (2, 3), (14, 8), out_path)
